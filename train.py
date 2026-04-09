"""Stage 2 training — Flow Matching on RAE latents.

JAX/Flax NNX + TPU data parallelism via jax.sharding.Mesh.

Optimizations vs original:
- rae.encode() is JIT-compiled and fused into the data pipeline
- No per-step float() sync (running_loss accumulates on-device via jnp)
- Gradient norm computed fully on-device (no Python list comprehension)
- donate_argnums on train_step to avoid buffer copies
- shard_batch uses jax.device_put directly without intermediate jnp.array()
- encode_and_shard fused JIT to minimize host-device roundtrips
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
from tqdm import tqdm

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
# ── Hardcoded constants (change here, not via CLI) ───────────────────────
# RAE encoder / decoder
_RAE_ENCODER_CLS         = "Dinov2withNorm"
_RAE_ENCODER_CONFIG_PATH = "facebook/dinov2-base"
_RAE_ENCODER_INPUT_SIZE  = 224
_RAE_DINOV2_PATH         = "facebook/dinov2-base"
_RAE_DECODER_CONFIG_PATH = "configs/decoder/ViTXL"
_RAE_NOISE_TAU           = 0.0   # 0 at Stage 2 inference

# DiT model (XL by default)
_MODEL_TARGET       = "stage2.models.DDT.DiTwDDTHead"
_INPUT_SIZE         = 16           # latent spatial size
_PATCH_SIZE         = 1
_IN_CHANNELS        = 768
_HIDDEN_SIZE        = [1152, 2048]
_DEPTH              = [28, 2]
_NUM_HEADS          = [16, 16]
_MLP_RATIO          = 4.0
_CLASS_DROPOUT_PROB = 0.1
_NUM_CLASSES        = 1000
_LATENT_SIZE_CHW    = [768, 16, 16]  # [C, H, W]

# Transport / flow matching
_PATH_TYPE       = "Linear"
_PREDICTION      = "velocity"
_TIME_DIST_TYPE  = "logit-normal_0_1"
_SHIFT_BASE      = 4096

# Sampler (ODE / eval only)
_SAMPLER_MODE    = "ODE"
_SAMPLING_METHOD = "euler"
_NUM_STEPS       = 50

# Optimizer / schedule
_BETAS          = (0.9, 0.95)
_WEIGHT_DECAY   = 0.0
_EMA_DECAY      = 0.9995
_SCHEDULE_TYPE  = "linear"
_WARMUP_EPOCHS  = 40
_FINAL_LR       = 2e-5
_CLIP_GRAD      = 1.0

# Logging / checkpointing
_LOG_INTERVAL        = 100
_SAMPLE_EVERY        = 10000
_CKPT_INTERVAL       = 10   # in global steps


# CLI
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train Stage 2 DiT on RAE latents (JAX)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ────────────────────────────────────────────────────────
    p.add_argument("--data-path", type=str, required=True,
                   help="Dataset root (ImageFolder) or TFDS data_dir")
    p.add_argument("--results-dir", type=str, default="ckpts/stage2",
                   help="Root directory for checkpoints")
    p.add_argument("--experiment-name", type=str, default="stage2_run",
                   help="Sub-folder name inside results-dir")
    p.add_argument("--rae-checkpoint", type=str, default=None,
                   help="Stage 1 checkpoint .pkl to load decoder weights from")
    p.add_argument("--normalization-stat-path", type=str, default=None,
                   help="Latent normalization stats .pt/.npz")
    p.add_argument("--pretrained-decoder-path", type=str, default=None,
                   help="Pretrained decoder .pt weights")
    p.add_argument("--reference-npz-path", type=str, default=None,
                   help="Pre-computed FID reference .npz")

    # ── Data ────────────────────────────────────────────────────────
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--dataset-type", type=str, default="imagefolder",
                   choices=["imagefolder", "tfds"])
    p.add_argument("--tfds-name", type=str, default=None,
                   help="TFDS dataset name (e.g. celebahq256, imagenet2012)")
    p.add_argument("--tfds-builder-dir", type=str, default=None,
                   help="Path to custom TFDS builder directory")
    p.add_argument("--num-train-samples", type=int, default=1281167,
                   help="Dataset size — used to compute steps/epoch")

    # ── Training knobs ───────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--global-batch-size", type=int, default=1024,
                   help="Total batch across all TPU devices")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--global-seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["fp32", "bf16"])

    # ── Eval ────────────────────────────────────────────────────────
    p.add_argument("--eval-fid-every", type=int, default=0,
                   help="Evaluate FID every N steps (0 = disabled)")
    p.add_argument("--num-fid-samples", type=int, default=50000)

    # ── WandB ───────────────────────────────────────────────────────
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="rae-jax-stage2")
    p.add_argument("--wandb-entity", type=str, default="")

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

    # ── Derive constants from hardcoded defaults + CLI args ──────────
    num_classes     = _NUM_CLASSES
    null_label      = _NUM_CLASSES          # null class = num_classes
    latent_size_chw = tuple(_LATENT_SIZE_CHW)  # (C, H, W)
    latent_size     = (latent_size_chw[1], latent_size_chw[2], latent_size_chw[0])  # (H, W, C) NHWC
    shift_dim       = math.prod(latent_size)
    time_dist_shift = math.sqrt(shift_dim / _SHIFT_BASE)

    global_batch_size = args.global_batch_size
    per_device_batch  = global_batch_size // num_devices
    assert global_batch_size % num_devices == 0, \
        f"global_batch_size {global_batch_size} must be divisible by {num_devices} devices"

    ema_decay          = _EMA_DECAY
    num_epochs         = args.epochs
    log_interval       = _LOG_INTERVAL
    sample_every       = _SAMPLE_EVERY
    checkpoint_interval = _CKPT_INTERVAL
    clip_grad          = _CLIP_GRAD
    global_seed        = args.global_seed

    guidance_scale  = 1.0       # CFG disabled by default during training
    use_guidance    = False

    compute_dtype = jnp.bfloat16 if args.precision == "bf16" else jnp.float32

    # ── Experiment dir ──
    experiment_dir, checkpoint_dir = configure_experiment_dirs(
        args.results_dir, args.experiment_name,
    )
    ckpt_mngr = build_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING)
    logger = logging.getLogger("train_stage2")

    # ── WandB ──
    if args.wandb and is_main:
        wandb_utils.initialize(
            config=vars(args),
            entity=args.wandb_entity or os.environ.get("ENTITY", ""),
            exp_name=args.experiment_name,
            project_name=args.wandb_project,
        )

    # ── Seed ──
    rng = jax.random.PRNGKey(global_seed)

    # ── RAE (frozen encoder + decoder for eval) ──
    rngs = nnx.Rngs(params=0, dropout=1)
    logger.info("Loading RAE...")
    rae_params = dict(
        encoder_cls=_RAE_ENCODER_CLS,
        encoder_config_path=_RAE_ENCODER_CONFIG_PATH,
        encoder_input_size=_RAE_ENCODER_INPUT_SIZE,
        encoder_params={"dinov2_path": _RAE_DINOV2_PATH, "normalize": True},
        decoder_config_path=_RAE_DECODER_CONFIG_PATH,
        noise_tau=_RAE_NOISE_TAU,
        reshape_to_2d=True,
    )
    if args.pretrained_decoder_path:
        rae_params["pretrained_decoder_path"] = args.pretrained_decoder_path
    if args.normalization_stat_path:
        rae_params["normalization_stat_path"] = args.normalization_stat_path
    rae = RAE(**rae_params, rngs=rngs)

    # Load pretrained Stage 1 weights
    if args.rae_checkpoint:
        import pickle
        logger.info(f"Loading Stage 1 RAE weights from: {args.rae_checkpoint}")
        try:
            with open(args.rae_checkpoint, "rb") as f:
                ckpt = pickle.load(f)
            # Stage 1 checkpoint structure: {"model": ..., "ema": ..., ...}
            raw_state = ckpt.get("ema", ckpt.get("model"))
            if raw_state is not None:
                raw_state = jax.tree.map(
                    lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
                    raw_state,
                )
                nnx.update(rae.decoder, raw_state)
                logger.info("RAE decoder weights loaded successfully.")
            else:
                logger.warning("Checkpoint has no 'ema' or 'model' key — using random RAE weights!")
        except Exception as e:
            logger.warning(f"Failed to load RAE checkpoint: {e}")
    else:
        logger.warning("No --rae-checkpoint provided! RAE evaluation decode will use RANDOM weights.")

    # Split RAE so we can JIT encode separately
    rae_graphdef, rae_state = nnx.split(rae)
    rae_state = jax.device_put(rae_state, repl_sharding)
    logger.info("RAE loaded (encoder frozen, used for latent encoding).")

    # ── DiT model ──
    logger.info("Instantiating DiT model...")
    model_params = dict(
        input_size=_INPUT_SIZE,
        patch_size=_PATCH_SIZE,
        in_channels=_IN_CHANNELS,
        hidden_size=_HIDDEN_SIZE,
        depth=_DEPTH,
        num_heads=_NUM_HEADS,
        mlp_ratio=_MLP_RATIO,
        class_dropout_prob=_CLASS_DROPOUT_PROB,
        num_classes=_NUM_CLASSES,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    )
    model_target = _MODEL_TARGET
    if "DDT" in model_target:
        model = DiTwDDTHead(**model_params, rngs=rngs, dtype=compute_dtype)
    else:
        model = LightningDiT(**model_params, rngs=rngs, dtype=compute_dtype)

    model_param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    logger.info(f"DiT parameters: {model_param_count / 1e6:.2f}M")

    # ── Extract model params (Param-only leaves) and replicate ──
    model_state = nnx.state(model)
    model_state = jax.device_put(model_state, repl_sharding)

    # ── EMA (replicated) ──
    ema_state = jax.tree.map(jnp.copy, model_state)

    # ── Optimizer ──
    dataset_size        = args.num_train_samples
    steps_per_epoch_est = dataset_size // global_batch_size
    total_training_steps = num_epochs * steps_per_epoch_est
    warmup_steps        = _WARMUP_EPOCHS * steps_per_epoch_est

    optimizer = build_optimizer_with_schedule(
        lr=args.lr,
        betas=_BETAS,
        weight_decay=_WEIGHT_DECAY,
        clip_grad=_CLIP_GRAD,
        schedule_type=_SCHEDULE_TYPE,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        final_lr=_FINAL_LR,
        warmup_from_zero=False,
    )
    opt_state = optimizer.init(model_state)
    opt_state = jax.device_put(opt_state, repl_sharding)
    logger.info(f"Optimizer: AdamW lr={args.lr}, schedule={_SCHEDULE_TYPE}, "
                f"warmup={warmup_steps}, total={total_training_steps}")

    # ── Resume from Stage 2 checkpoint if available ──
    restored_ckpt, restored_step = restore_checkpoint(ckpt_mngr)
    global_step_resume = 0
    if restored_ckpt is not None:
        logger.info(f"Resuming DiT from Stage 2 checkpoint at step {restored_step}...")
        model_state = jax.device_put(
            jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, restored_ckpt["model"]),
            repl_sharding,
        )
        ema_state = jax.device_put(
            jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, restored_ckpt["ema"]),
            repl_sharding,
        )
        if "opt_state" in restored_ckpt:
            try:
                opt_state = jax.device_put(
                    jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, restored_ckpt["opt_state"]),
                    repl_sharding,
                )
                logger.info("Optimizer state restored.")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state (will use fresh): {e}")
        global_step_resume = restored_step
        logger.info(f"Resumed successfully from step {restored_step}.")
    else:
        logger.info("No Stage 2 checkpoint found — starting from scratch with random DiT weights.")

    # ── Transport ──
    transport = create_transport(
        path_type=_PATH_TYPE,
        prediction=_PREDICTION,
        loss_weight=None,
        time_dist_type=_TIME_DIST_TYPE,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)
    sampler_params = dict(sampling_method=_SAMPLING_METHOD, num_steps=_NUM_STEPS,
                          atol=1e-6, rtol=1e-3, reverse=False)
    if _SAMPLER_MODE == "ODE":
        eval_sample_fn = transport_sampler.sample_ode(**sampler_params)
    else:
        eval_sample_fn = transport_sampler.sample_sde(**sampler_params)

    # ── Data ──
    dataset_type     = args.dataset_type
    tfds_builder_dir = args.tfds_builder_dir
    tfds_name        = args.tfds_name

    ds, steps_per_epoch = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=global_batch_size,
        dataset_type=dataset_type,
        split="train",
        tfds_name=tfds_name,
        tfds_builder_dir=tfds_builder_dir,
    )
    ds_valid, _ = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=global_batch_size,
        dataset_type=dataset_type,
        split="validation" if dataset_type == "tfds" else "val",
        tfds_name=tfds_name,
        tfds_builder_dir=tfds_builder_dir,
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Global batch: {global_batch_size}, Per-device: {per_device_batch}")

    # ─────────────────────────────────────────────────────────────
    # OPT 1: JIT-compiled RAE encode step
    # ─────────────────────────────────────────────────────────────
    @jax.jit
    def encode_batch(rae_st, images, rng):
        """Encode images → latents fully on-device (JIT compiled)."""
        rae_model = nnx.merge(rae_graphdef, rae_st)
        latents = jax.lax.stop_gradient(
            rae_model.encode(images, rng=rng, training=False)
        )
        return latents

    # ─────────────────────────────────────────────────────────────
    # OPT 2: JIT train step — uses nnx.update (same pattern as Stage 1)
    # ─────────────────────────────────────────────────────────────
    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def train_step(model_state, opt_state, ema_state, batch_x, batch_y, step_rng):
        """Single training step — fully on-device, no Python sync."""

        def loss_fn(params):
            nnx.update(model, params)
            def model_forward(xt, t, y):
                return model(xt, t, y, training=True, rng=step_rng, return_activations=True)

            terms, acts = transport.training_losses(
                model_forward, batch_x, step_rng, has_aux=True, y=batch_y,
            )
            v_mag = jnp.sqrt(jnp.mean(jnp.square(terms["pred"])))
            return jnp.mean(terms["loss"]), (v_mag, acts)

        (loss, (v_mag, acts)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)

        if clip_grad > 0:
            global_norm = jnp.sqrt(
                jax.tree.reduce(
                    lambda acc, g: acc + jnp.sum(g.astype(jnp.float32) ** 2),
                    grads,
                    initializer=jnp.zeros(()),
                )
            )
            scale = jnp.minimum(1.0, clip_grad / (global_norm + 1e-6))
            grads = jax.tree.map(lambda g: g * scale, grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, model_state)
        new_model_state = optax.apply_updates(model_state, updates)

        new_ema = jax.tree.map(
            lambda e, m: e * ema_decay + m * (1.0 - ema_decay),
            ema_state, new_model_state,
        )

        return loss, new_model_state, new_opt_state, new_ema, v_mag, acts

    # ── Eval ──
    do_eval            = args.eval_fid_every > 0
    eval_interval      = 0   # inline FID triggered by eval_fid_every
    reference_npz_path = args.reference_npz_path
    eval_fid_every     = args.eval_fid_every

    # ── Training loop ──
    global_step = global_step_resume
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    # OPT 4: Accumulate loss as JAX array to avoid per-step host sync
    running_stats = {}
    
    @jax.jit
    def valid_step(model_state, batch_x, batch_y, step_rng):
        def loss_fn(params):
            nnx.update(model, params)
            def model_forward(xt, t, y):
                return model(xt, t, y, training=False, rng=step_rng)
            terms = transport.training_losses(model_forward, batch_x, step_rng, y=batch_y)
            return jnp.mean(terms["loss"])
        return loss_fn(model_state)


    start_time = time.time()

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Compute dtype: {compute_dtype}")



    with mesh:
        for epoch in range(start_epoch, num_epochs):
            # Checkpoint is now saved at step level

            if is_main:
                steps_iter = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{num_epochs}")
            else:
                steps_iter = range(steps_per_epoch)

            for _ in steps_iter:
                step_data = next(ds)

                # OPT 5: Shard image batch directly without intermediate jnp.array()
                images = jax.device_put(
                    jnp.asarray(step_data["image"]), data_sharding
                )
                label_np = step_data.get("label", None)
                if label_np is not None:
                    labels = jax.device_put(jnp.asarray(label_np), data_sharding)
                else:
                    labels = jax.device_put(
                        jnp.zeros(images.shape[0], dtype=jnp.int32), data_sharding
                    )

                # OPT 1 applied: encode with JIT-compiled function
                rng, rng_enc = jax.random.split(rng)
                latents = encode_batch(rae_state, images, rng_enc)

                rng, step_rng = jax.random.split(rng)

                loss, model_state, opt_state, ema_state, v_mag, acts = train_step(
                    model_state, opt_state, ema_state,
                    latents, labels, step_rng,
                )

                # OPT 4: accumulate on-device, only sync at log boundary
                if "loss" not in running_stats:
                    running_stats["loss"] = loss
                    running_stats["v_magnitude_prime"] = v_mag
                    for k, v in acts.items():
                        running_stats[f"activations/{k}"] = v
                else:
                    running_stats["loss"] += loss
                    running_stats["v_magnitude_prime"] += v_mag
                    for k, v in acts.items():
                        running_stats[f"activations/{k}"] += v
                
                global_step += 1

                # Checkpoint at specific global steps
                if checkpoint_interval > 0 and global_step % checkpoint_interval == 0 and is_main:
                    logger.info(f"Saving checkpoint at step {global_step}...")
                    save_checkpoint(
                        ckpt_mngr, global_step,
                        jax.device_get(model_state),
                        jax.device_get(ema_state),
                        jax.device_get(opt_state),
                    )

                # Logging — only sync to host every log_interval steps
                if log_interval > 0 and global_step % log_interval == 0 and is_main:
                    avg_stats = {k: float(v) / log_interval for k, v in running_stats.items()}
                    elapsed = time.time() - start_time
                    steps_per_sec = log_interval / elapsed
                    
                    # Compute validation loss on current validation batch
                    try:
                        valid_data = next(ds_valid)
                    except TypeError:
                        # Sometimes iterators need iter()
                        if not hasattr(ds_valid, "__next__"):
                            ds_valid = iter(ds_valid)
                        try:
                            valid_data = next(ds_valid)
                        except StopIteration:
                            ds_valid = iter(ds_valid)
                            valid_data = next(ds_valid)
                    
                    valid_images = jax.device_put(jnp.asarray(valid_data["image"]), data_sharding)
                    valid_label_np = valid_data.get("label", None)
                    if valid_label_np is not None:
                        valid_labels = jax.device_put(jnp.asarray(valid_label_np), data_sharding)
                    else:
                        valid_labels = jax.device_put(jnp.zeros(valid_images.shape[0], dtype=jnp.int32), data_sharding)
                        
                    rng, val_rng = jax.random.split(rng)
                    valid_latents = encode_batch(rae_state, valid_images, val_rng)
                    loss_valid = float(valid_step(model_state, valid_latents, valid_labels, val_rng))

                    stats = {
                        "training/loss": avg_stats["loss"],
                        "training/loss_valid": loss_valid,
                        "training/v_magnitude_prime": avg_stats["v_magnitude_prime"],
                        "training/steps_per_sec": steps_per_sec,
                        "training/epoch": epoch,
                        "training/lr": float(optimizer.learning_rate(opt_state[1].count)
                                          if hasattr(optimizer, 'learning_rate') else args.lr),
                    }
                    for k, v in avg_stats.items():
                        if k.startswith("activations/"):
                            stats[f"training/{k}"] = v
                            
                    if is_main and hasattr(steps_iter, "set_postfix"):
                        steps_iter.set_postfix({
                            "loss": f"{stats['training/loss']:.4f}",
                            "val_loss": f"{stats['training/loss_valid']:.4f}",
                            "lr": f"{stats['training/lr']:.2e}",
                            "iter/s": f"{steps_per_sec:.2f}"
                        })
                        
                    if args.wandb:
                        wandb_utils.log(stats, step=global_step)

                    running_stats = {}
                    start_time = time.time()

                # Visual sampling
                if sample_every > 0 and global_step % sample_every == 0 and is_main:
                    logger.info("Generating EMA samples...")
                    rae_model_gen = nnx.merge(rae_graphdef, rae_state)
                    # Helper decode function for denoise visualization
                    @jax.jit
                    def decode_fn(z):
                        return jax.lax.stop_gradient(rae_model_gen.decode(z))

                    _generate_samples(
                        model, ema_state, eval_sample_fn,
                        latent_size, labels[:8], rng,
                        use_guidance, guidance_scale, null_label,
                        global_step, args.wandb,
                        rae_decode_fn=decode_fn
                    )

                # Distributed eval
                if (do_eval and eval_interval > 0 and global_step % eval_interval == 0) or \
                   (eval_fid_every > 0 and global_step % eval_fid_every == 0):
                    num_samples = args.num_fid_samples if (eval_fid_every > 0 and global_step % eval_fid_every == 0) else 1000
                    logger.info(f"Starting distributed evaluation ({num_samples} samples)...")
                    
                    if reference_npz_path is None:
                        reference_npz_path = os.path.join(experiment_dir, f"fid_ref_stat_{num_samples}.npz")
                        
                    if is_main and not os.path.exists(reference_npz_path):
                        logger.info(f"Reference NPZ {reference_npz_path} not found. Generating it now via subprocess (to avoid consuming TPU memory)...")
                        import subprocess
                        import sys
                        cmd = [
                            sys.executable, "create_fid_ref.py",
                            "--data-path", args.data_path,
                            "--out-path", reference_npz_path,
                            "--num-samples", str(num_samples),
                            "--batch-size", "128",
                            "--dataset-type", dataset_type
                        ]
                        if tfds_name:
                            cmd.extend(["--tfds-name", tfds_name])
                        if tfds_builder_dir:
                            cmd.extend(["--tfds-builder-dir", tfds_builder_dir])
                            
                        try:
                            subprocess.run(cmd, check=True)
                            logger.info("Reference NPZ generation successful.")
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Failed to generate Reference NPZ: {e}")
                            
                    try:
                        jax.experimental.multihost_utils.sync_global_devices("eval_ref_npz_done")
                    except Exception:
                        pass
                        
                    nnx.update(model, ema_state)
                    eval_model = model
                    rae_model_eval = nnx.merge(rae_graphdef, rae_state)
                    
                    @jax.jit
                    def eval_decode_fn(z):
                        return jax.lax.stop_gradient(rae_model_eval.decode(z))
                        
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
                        num_samples=num_samples,
                        batch_size=min(per_device_batch * 4, 512),  # 4x bigger for eval throughput
                        experiment_dir=experiment_dir,
                        global_step=global_step,
                        reference_npz_path=reference_npz_path,
                        rae_decode_fn=eval_decode_fn,
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


def _generate_samples(model, ema_state, sample_fn, latent_size,
                      labels, rng, use_guidance, cfg_scale, null_label,
                      global_step, use_wandb, rae_decode_fn=None):
    """Generate visual samples using EMA model (on main process only)."""
    n = min(8, labels.shape[0])
    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, (n, *latent_size))

    nnx.update(model, ema_state)
    ema_model = model

    log_every = 10
    if use_guidance:
        z_in = jnp.concatenate([z, z], axis=0)
        y = labels[:n]
        y_null = jnp.full((n,), null_label, dtype=jnp.int32)
        y_in = jnp.concatenate([y, y_null], axis=0)
        samples, intermediates = sample_fn(
            z_in, partial(ema_model.forward_with_cfg, cfg_scale=cfg_scale),
            return_intermediates=True, log_every=log_every,
        )
        samples = samples[:n]
        intermediates = [step[:n] for step in intermediates]
    else:
        y = labels[:n]
        # Use y=None, **kw to match transport.py drift convention (passes y as keyword)
        model_fn = lambda x, t, y=None, **kw: ema_model(x, t, y, training=False)
        samples, intermediates = sample_fn(
            z, model_fn, y=y,
            return_intermediates=True, log_every=log_every,
        )

    if rae_decode_fn is not None and use_wandb:
        num_inter = len(intermediates)
        print(f"Decoding {num_inter} intermediate denoising steps for WandB...")

        for idx, inter_z in enumerate(intermediates):
            # intermediates[0] = after step log_every, [1] = 2*log_every, ...
            ode_step = (idx + 1) * log_every
            images = np.array(rae_decode_fn(inter_z))
            images = np.clip(images, 0.0, 1.0)
            if images.ndim == 4 and images.shape[1] in (1, 3, 4):  # NCHW -> NHWC
                images = np.transpose(images, (0, 2, 3, 1))
            wandb_utils.log_image(
                images,
                key=f"sample_denoise/step_{ode_step:03d}",
                step=global_step,
            )

        # Log final clean sample
        final_images = np.array(rae_decode_fn(samples))
        final_images = np.clip(final_images, 0.0, 1.0)
        if final_images.ndim == 4 and final_images.shape[1] in (1, 3, 4):
            final_images = np.transpose(final_images, (0, 2, 3, 1))
        wandb_utils.log_image(final_images, key="sample/final", step=global_step)

    print(f"Generated {n} latent samples at step {global_step}")


if __name__ == "__main__":
    main()
