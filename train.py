"""Stage 2 training — Flow Matching on RAE latents.

JAX/Flax NNX + TPU data parallelism via jax.sharding.Mesh.

Key TPU optimizations:
- Data sharded across all TPU cores via NamedSharding(mesh, P("data"))
- Model params replicated (NamedSharding(mesh, P()))
- Gradients auto-reduced via sharding (no explicit pmean needed)
- Gradient checkpointing on DiT blocks for memory efficiency
- bfloat16 compute precision
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from omegaconf import OmegaConf
import optax

# ── project imports ──────────────────────────────────────────────
from stage1 import RAE
from stage2 import DiTwDDTHead, create_transport, Sampler
from stage2.models.lightningDiT import LightningDiT
from utils.device_utils import create_mesh, get_data_sharding, get_replicated_sharding, shard_batch, print_device_info
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs, update_ema, center_crop_arr
from utils.optim_utils import build_optimizer_with_schedule
from utils.resume_utils import save_checkpoint, restore_checkpoint, configure_experiment_dirs, build_checkpoint_manager
from utils import wandb_utils
from data import build_dataloader


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 2 transport model on RAE latents (JAX)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--results-dir", type=str, default="ckpts")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--dataset-type", type=str, default=None,
                    choices=["imagefolder", "tfds"],
                    help="Data source type. Overrides config 'data.source'.")
    p.add_argument("--tfds-builder-dir", type=str, default=None,
                    help="Path to custom TFDS builder dir (e.g. tfds_builders/celebahq256).")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--global-seed", type=int, default=None)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Devices & Mesh ──
    print_device_info()
    mesh = create_mesh()  # Mesh(devices, ("data",))
    data_sharding = get_data_sharding(mesh)
    repl_sharding = get_replicated_sharding(mesh)
    num_devices = jax.device_count()
    process_index = jax.process_index()
    is_main = process_index == 0

    # ── Config ──
    full_cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, sampler_config,
     guidance_config, misc_config, training_config, eval_config) = parse_configs(full_cfg)

    misc = OmegaConf.to_container(misc_config or {}, resolve=True)
    transport_cfg = OmegaConf.to_container(transport_config or {}, resolve=True)
    sampler_cfg = OmegaConf.to_container(sampler_config or {}, resolve=True)
    guidance_cfg = OmegaConf.to_container(guidance_config or {}, resolve=True)
    training_cfg = OmegaConf.to_container(training_config or {}, resolve=True)

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    # latent_size in config is [C, H, W] (PyTorch convention)
    # Convert to [H, W, C] for JAX NHWC
    cfg_latent = list(misc.get("latent_size", [768, 16, 16]))
    latent_size_chw = tuple(int(d) for d in cfg_latent)  # (C, H, W)
    latent_size = (latent_size_chw[1], latent_size_chw[2], latent_size_chw[0])  # (H, W, C) for NHWC
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    # Training hypers
    per_device_batch = int(training_cfg.get("batch_size", 16))
    global_batch_size = per_device_batch * num_devices
    # If global_batch_size is explicitly set, use it
    if "global_batch_size" in training_cfg:
        global_batch_size = int(training_cfg["global_batch_size"])
        per_device_batch = global_batch_size // num_devices
        assert global_batch_size % num_devices == 0, \
            f"global_batch_size {global_batch_size} must be divisible by {num_devices} devices"

    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 10))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    global_seed = args.global_seed if args.global_seed is not None else int(training_cfg.get("global_seed", 0))

    # Guidance
    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    use_guidance = guidance_scale > 1.0

    # Precision
    compute_dtype = jnp.bfloat16 if args.precision == "bf16" else jnp.float32

    # ── Experiment dir ──
    experiment_dir, checkpoint_dir = configure_experiment_dirs(
        args.results_dir, f"stage2-{Path(args.config).stem}",
    )
    ckpt_mngr = build_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING)
    logger = logging.getLogger("train_stage2")

    # ── WandB ──
    if args.wandb and is_main:
        wandb_utils.initialize(
            config=OmegaConf.to_container(full_cfg, resolve=True),
            entity=os.environ.get("ENTITY", ""),
            exp_name=f"stage2-{Path(args.config).stem}",
            project_name="rae-jax-stage2",
        )

    # ── Seed ──
    rng = jax.random.PRNGKey(global_seed)

    # ── RAE (frozen encoder+decoder) ──
    rngs = nnx.Rngs(params=0, dropout=1)
    logger.info("Loading frozen RAE...")
    rae_params = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    rae = RAE(**rae_params, rngs=rngs)
    logger.info("RAE loaded (encoder frozen, used for latent encoding).")

    # ── DiT model ──
    logger.info("Instantiating DiT model...")
    model_params = dict(OmegaConf.to_container(model_config.get("params", {}), resolve=True))
    model_target = model_config.get("target", "stage2.DiTwDDTHead")
    if "DDT" in model_target:
        model = DiTwDDTHead(**model_params, rngs=rngs, dtype=compute_dtype)
    else:
        model = LightningDiT(**model_params, rngs=rngs, dtype=compute_dtype)

    model_param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    logger.info(f"DiT parameters: {model_param_count / 1e6:.2f}M")

    # ── Replicate model params across all devices ──
    # Use split/merge to properly separate graph structure from state
    graphdef, model_state = nnx.split(model)
    model_state = jax.device_put(model_state, repl_sharding)

    # ── EMA (replicated) ──
    ema_state = jax.tree.map(jnp.copy, model_state)

    # ── Optimizer ──
    opt_cfg = training_cfg.get("optimizer", {})
    sched_cfg = training_cfg.get("scheduler", {})

    # Compute steps for LR schedule
    data_cfg = OmegaConf.to_container(full_cfg.get("data", {}), resolve=True) if "data" in full_cfg else {}
    dataset_size = int(data_cfg.get("num_train_samples", 1281167))  # default ImageNet
    steps_per_epoch_est = dataset_size // global_batch_size
    total_training_steps = num_epochs * steps_per_epoch_est
    warmup_steps = int(sched_cfg.get("warmup_epochs", 40)) * steps_per_epoch_est

    optimizer = build_optimizer_with_schedule(
        lr=float(opt_cfg.get("lr", 2e-4)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        clip_grad=clip_grad,
        schedule_type=str(sched_cfg.get("type", "linear")),
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        final_lr=float(sched_cfg.get("final_lr", 2e-5)),
        warmup_from_zero=bool(sched_cfg.get("warmup_from_zero", False)),
    )
    opt_state = optimizer.init(nnx.state(model))
    opt_state = jax.device_put(opt_state, repl_sharding)
    logger.info(f"Optimizer: AdamW lr={opt_cfg.get('lr')}, schedule={sched_cfg.get('type')}, "
                f"warmup={warmup_steps}, total={total_training_steps}")

    # ── Transport ──
    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sample_fn = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sample_fn = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Sampler mode {sampler_mode}")

    # ── Data ──
    # Read data config from YAML, with CLI args as override
    data_cfg_full = OmegaConf.to_container(full_cfg.get("data", {}), resolve=True) if "data" in full_cfg else {}
    dataset_type = args.dataset_type or data_cfg_full.get("source", "imagefolder")
    tfds_builder_dir = args.tfds_builder_dir or data_cfg_full.get("tfds_builder_dir", None)
    tfds_name = data_cfg_full.get("dataset_name", None)

    ds, steps_per_epoch = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=global_batch_size,
        dataset_type=dataset_type,
        split="train",
        tfds_name=tfds_name,
        tfds_builder_dir=tfds_builder_dir,
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Global batch: {global_batch_size}, Per-device: {per_device_batch}")

    # ── JIT-compiled train step ──
    # With mesh-based sharding:
    # - data is sharded along "data" axis → each device gets per_device_batch
    # - model params are replicated → same on all devices
    # - gradients computed on sharded data → auto-reduced when applied to replicated params
    @jax.jit
    def train_step(model_state, opt_state, ema_state, batch_x, batch_y, step_rng):
        """Single training step with flow matching loss.

        Data parallelism is handled via mesh sharding:
        - batch_x is sharded across devices (each gets B/N_devices)
        - model_state is replicated
        - grad(loss) on sharded data → mean grad auto-computed by XLA
        """

        def loss_fn(params):
            # Reconstruct model from graphdef + params for forward pass
            m = nnx.merge(graphdef, params)
            rng_local = step_rng

            def model_forward(xt, t, y):
                return m(xt, t, y, training=True, rng=rng_local)

            terms = transport.training_losses(
                model_forward, batch_x, step_rng, y=batch_y,
            )
            return jnp.mean(terms["loss"])

        loss, grads = jax.value_and_grad(loss_fn)(model_state)

        # Gradient clipping (global norm)
        if clip_grad > 0:
            grad_leaves = jax.tree.leaves(grads)
            global_norm = jnp.sqrt(
                jnp.sum(jnp.array([jnp.sum(g ** 2) for g in grad_leaves]))
            )
            scale = jnp.minimum(1.0, clip_grad / (global_norm + 1e-6))
            grads = jax.tree.map(lambda g: g * scale, grads)

        # Optimizer step
        updates, new_opt_state = optimizer.update(grads, opt_state, model_state)
        new_model_state = optax.apply_updates(model_state, updates)

        # EMA update
        new_ema = jax.tree.map(
            lambda e, m: e * ema_decay + m * (1.0 - ema_decay),
            ema_state, new_model_state,
        )

        return loss, new_model_state, new_opt_state, new_ema

    # ── Eval ──
    eval_cfg = OmegaConf.to_container(eval_config or {}, resolve=True) if eval_config else {}
    do_eval = bool(eval_cfg)
    eval_interval = int(eval_cfg.get("eval_interval", 0))
    reference_npz_path = eval_cfg.get("reference_npz_path", None)

    # ── Training loop ──
    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Compute dtype: {compute_dtype}")

    model_state = nnx.state(model)

    with mesh:
        for epoch in range(num_epochs):
            # Checkpoint at epoch start
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and is_main and epoch > 0:
                logger.info(f"Saving checkpoint at epoch {epoch}...")
                save_checkpoint(
                    ckpt_mngr, global_step,
                    jax.device_get(model_state),
                    jax.device_get(ema_state),
                    jax.device_get(opt_state),
                )

            for step_data in ds:
                # ── Shard data across TPU cores ──
                batch = shard_batch(step_data, mesh)
                images = batch["image"]
                labels = batch.get("label", jnp.zeros(images.shape[0], dtype=jnp.int32))

                # Encode images to latents with frozen RAE
                # RAE.encode: (B, H, W, 3) → (B, 16, 16, 768) NHWC
                rng, rng_enc = jax.random.split(rng)
                latents = jax.lax.stop_gradient(
                    rae.encode(images, rng=rng_enc, training=False)
                )

                rng, step_rng = jax.random.split(rng)

                loss, model_state, opt_state, ema_state = train_step(
                    model_state, opt_state, ema_state,
                    latents, labels, step_rng,
                )

                running_loss += float(loss)
                global_step += 1

                # Logging
                if log_interval > 0 and global_step % log_interval == 0 and is_main:
                    avg_loss = running_loss / log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = log_interval / elapsed

                    stats = {
                        "train/loss": avg_loss,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                    }
                    logger.info(
                        f"[Epoch {epoch} | Step {global_step}] "
                        + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                    )
                    if args.wandb:
                        wandb_utils.log(stats, step=global_step)

                    running_loss = 0.0
                    start_time = time.time()

                # Visual sampling
                if sample_every > 0 and global_step % sample_every == 0 and is_main:
                    logger.info("Generating EMA samples...")
                    _generate_samples(
                        graphdef, ema_state, eval_sample_fn,
                        latent_size, labels[:8], rng,
                        use_guidance, guidance_scale, null_label,
                        global_step, args.wandb,
                    )

                # Distributed eval
                if do_eval and eval_interval > 0 and global_step % eval_interval == 0:
                    logger.info("Starting distributed evaluation...")
                    eval_model = nnx.merge(graphdef, ema_state)
                    from eval import evaluate_generation_distributed
                    eval_stats = evaluate_generation_distributed(
                        model=eval_model,
                        ema_state=ema_state,
                        sample_fn=eval_sample_fn,
                        latent_size=latent_size,
                        num_classes=num_classes,
                        null_label=null_label,
                        use_guidance=use_guidance,
                        guidance_scale=guidance_scale,
                        # rae_decode_fn=rae.decode,
                        num_samples=1000,
                        batch_size=per_device_batch,
                        experiment_dir=experiment_dir,
                        global_step=global_step,
                        reference_npz_path=reference_npz_path,
                        mesh=mesh,
                    )
                    if eval_stats and args.wandb and is_main:
                        eval_stats = {f"eval/{k}": v for k, v in eval_stats.items()}
                        wandb_utils.log(eval_stats, step=global_step)

    # Final checkpoint
    if is_main:
        logger.info("Saving final checkpoint...")
        save_checkpoint(
            ckpt_mngr, global_step,
            jax.device_get(model_state),
            jax.device_get(ema_state),
            jax.device_get(opt_state),
        )

    logger.info("Training complete!")


def _generate_samples(graphdef, ema_state, sample_fn, latent_size,
                      labels, rng, use_guidance, cfg_scale, null_label,
                      global_step, use_wandb):
    """Generate visual samples using EMA model (on main process only)."""
    n = min(8, labels.shape[0])
    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, (n, *latent_size))

    # Reconstruct model with EMA weights
    ema_model = nnx.merge(graphdef, ema_state)

    if use_guidance:
        z = jnp.concatenate([z, z], axis=0)
        y = labels[:n]
        y_null = jnp.full((n,), null_label, dtype=jnp.int32)
        y = jnp.concatenate([y, y_null], axis=0)

        samples = sample_fn(z, partial(ema_model.forward_with_cfg, cfg_scale=cfg_scale))
    else:
        y = labels[:n]
        model_fn = lambda x, t, y_: ema_model(x, t, y_, training=False)
        samples = sample_fn(z, model_fn, y=y)

    if use_guidance:
        samples = samples[:n]

    # Note: RAE decode requires the decoder to be loaded
    print(f"Generated {n} latent samples at step {global_step}")


if __name__ == "__main__":
    main()
