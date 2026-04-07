"""RAE Stage 1 Training — JAX/Flax NNX on TPUv4e-8.

Port of PyTorch train_stage1.py with:
- Data parallelism via jax.sharding.Mesh (8 TPU cores)
- JIT-compiled train step (generator + discriminator)
- LPIPS + GAN loss with adaptive weight
- EMA tracking
- Orbax checkpointing
- WandB logging

Usage:
    python train_stage1.py --config configs/DINOv2-B_decXL.yaml
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import time
from collections import defaultdict
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from omegaconf import OmegaConf

from stage1 import RAE
from disc import build_discriminator, LPIPS, hinge_d_loss, vanilla_g_loss, DiffAug
from disc.gan_loss import vanilla_d_loss
from data import build_dataset
from utils.device_utils import create_mesh, get_data_sharding, get_replicated_sharding, shard_batch, print_device_info
from utils.optim_utils import build_optimizer_with_schedule
from utils.train_utils import update_ema, parse_configs
from utils.resume_utils import configure_experiment_dirs, build_checkpoint_manager, save_checkpoint, restore_checkpoint
from utils import wandb_utils


# ---------------------------------------------------------------------------
# Adaptive discriminator weight (matches PyTorch calculate_adaptive_weight)
# ---------------------------------------------------------------------------
def calculate_adaptive_weight(
    rec_loss: jnp.ndarray,
    g_loss: jnp.ndarray,
    decoder_last_layer_params,
    disc_weight: float = 0.75,
    max_d_weight: float = 10000.0,
) -> jnp.ndarray:
    """Calculate adaptive weight to balance rec loss and GAN loss.

    Based on gradient norms of the last decoder layer.
    """
    rec_grads = jax.grad(lambda p: rec_loss)(decoder_last_layer_params)
    g_grads = jax.grad(lambda p: g_loss)(decoder_last_layer_params)

    rec_grad_norm = jnp.sqrt(jnp.sum(rec_grads ** 2) + 1e-6)
    g_grad_norm = jnp.sqrt(jnp.sum(g_grads ** 2) + 1e-6)

    d_weight = jnp.clip(rec_grad_norm / g_grad_norm, 0.0, max_d_weight)
    return d_weight * disc_weight


# NOTE: The old train_step_generator / train_step_discriminator functions with
# jax.lax.pmean have been removed. They required pmap's axis_name context which
# we don't use. Instead, we use mesh-based sharding (see _simple variants below)
# which auto-reduces gradients via XLA when data is sharded + params replicated.



# ---------------------------------------------------------------------------
# Discriminator train step (JIT-compiled)
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, donate_argnums=(0, 1))
def train_step_discriminator(
    disc_params,
    disc_opt_state,
    real_images,
    fake_images,
    disc_model,
    diffaug,
    disc_optimizer,
    rng,
    loss_type: str = "hinge",
):
    """Single discriminator training step."""
    rng_real, rng_fake = jax.random.split(rng)

    def disc_loss_fn(d_params):
        # Convert to NCHW
        real_nchw = real_images.transpose(0, 3, 1, 2)

        real_aug = diffaug(real_nchw, rng_real)
        fake_aug = diffaug(fake_images, rng_fake)  # fake is already NCHW

        logits_real = disc_model.classify(real_aug)
        logits_fake = disc_model.classify(fake_aug)

        if loss_type == "hinge":
            d_loss = hinge_d_loss(logits_real, logits_fake)
        else:
            d_loss = vanilla_d_loss(logits_real, logits_fake)

        return d_loss

    d_loss, grads = jax.value_and_grad(disc_loss_fn)(disc_params)

    grads = jax.lax.pmean(grads, axis_name="data")
    d_loss = jax.lax.pmean(d_loss, axis_name="data")

    updates, new_opt_state = disc_optimizer.update(grads, disc_opt_state, disc_params)
    new_params = jax.tree.map(lambda p, u: p + u, disc_params, updates)

    return new_params, new_opt_state, d_loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(config):
    """Main training function."""
    # Parse config sections
    rae_cfg, train_cfg, gan_cfg, eval_cfg = parse_configs(config)

    # Device setup
    print_device_info()
    mesh = create_mesh()
    data_sharding = get_data_sharding(mesh)
    repl_sharding = get_replicated_sharding(mesh)

    num_devices = jax.device_count()
    global_batch_size = train_cfg.global_batch_size
    per_device_batch = global_batch_size // num_devices
    assert global_batch_size % num_devices == 0, \
        f"Batch size {global_batch_size} not divisible by {num_devices} devices"

    print(f"[Train] Global batch: {global_batch_size}, Per-device: {per_device_batch}")

    # Experiment directories
    exp_dir, ckpt_dir = configure_experiment_dirs(
        config.experiment.results_dir,
        config.experiment.experiment_name,
    )

    # WandB
    if config.wandb.enabled and jax.process_index() == 0:
        wandb_utils.initialize(
            OmegaConf.to_container(config, resolve=True),
            entity=config.wandb.get("entity", os.environ.get("ENTITY", "")),
            exp_name=config.wandb.experiment_name,
            project_name=config.wandb.project,
        )

    # --- Build models ---
    rng = jax.random.PRNGKey(train_cfg.seed)
    rng, rng_rae, rng_disc, rng_lpips = jax.random.split(rng, 4)

    # RAE model
    rae_params = OmegaConf.to_container(rae_cfg.params, resolve=True)
    rae_model = RAE(
        **rae_params,
        rngs=nnx.Rngs(int(rng_rae[0])),
    )

    # Discriminator
    disc_model, diffaug = build_discriminator(
        OmegaConf.to_container(gan_cfg.disc, resolve=True),
        rng=rng_disc,
    )

    # LPIPS (frozen)
    lpips_model = LPIPS(rngs=nnx.Rngs(int(rng_lpips[0])))

    # --- Build data loader ---
    data_cfg = config.data
    train_iter = build_dataset(
        source=data_cfg.source,
        dataset_name=data_cfg.get("dataset_name", "celebahq256"),
        data_dir=data_cfg.get("data_dir"),
        tfds_builder_dir=data_cfg.get("tfds_builder_dir"),
        image_size=data_cfg.image_size,
        batch_size=global_batch_size,
        seed=train_cfg.seed,
    )

    # --- Compute steps ---
    # For TFDS we estimate steps_per_epoch from dataset size
    # CelebAHQ: ~30k images
    dataset_size = data_cfg.get("num_train_samples", 30000)
    steps_per_epoch = dataset_size // global_batch_size
    total_steps = train_cfg.epochs * steps_per_epoch
    warmup_steps = train_cfg.scheduler.warmup_epochs * steps_per_epoch

    print(f"[Train] Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    # --- Build optimizers ---
    sched_cfg = train_cfg.scheduler
    opt_cfg = train_cfg.optimizer

    # Generator optimizer (decoder only)
    gen_optimizer = build_optimizer_with_schedule(
        lr=opt_cfg.lr,
        betas=tuple(opt_cfg.betas),
        weight_decay=opt_cfg.weight_decay,
        clip_grad=train_cfg.clip_grad,
        schedule_type=sched_cfg.type,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        final_lr=sched_cfg.final_lr,
        warmup_from_zero=sched_cfg.warmup_from_zero,
    )

    # Discriminator optimizer
    disc_sched = gan_cfg.disc.scheduler
    disc_opt_cfg = gan_cfg.disc.optimizer
    disc_warmup = disc_sched.warmup_epochs * steps_per_epoch

    disc_optimizer = build_optimizer_with_schedule(
        lr=disc_opt_cfg.lr,
        betas=tuple(disc_opt_cfg.betas),
        weight_decay=disc_opt_cfg.weight_decay,
        schedule_type=disc_sched.type,
        warmup_steps=disc_warmup,
        total_steps=total_steps,
        final_lr=disc_sched.final_lr,
        warmup_from_zero=disc_sched.warmup_from_zero,
    )

    # --- Initialize optimizer states ---
    # Extract decoder params (trainable)
    decoder_params = nnx.state(rae_model.decoder)
    decoder_opt_state = gen_optimizer.init(decoder_params)
    ema_params = jax.tree.map(jnp.copy, decoder_params)

    # Extract disc params (trainable heads only)
    disc_params = nnx.state(disc_model)
    disc_opt_state = disc_optimizer.init(disc_params)

    # --- Checkpoint ---
    ckpt_mngr = build_checkpoint_manager(ckpt_dir, max_to_keep=5)

    # Try restore
    ckpt, restored_step = restore_checkpoint(ckpt_mngr)
    start_step = 0
    if ckpt is not None:
        decoder_params = ckpt["model"]
        ema_params = ckpt["ema"]
        decoder_opt_state = ckpt["opt_state"]
        start_step = restored_step
        print(f"[Train] Resumed from step {start_step}")

    # --- Loss config ---
    loss_cfg = OmegaConf.to_container(gan_cfg.loss, resolve=True)

    # --- Training loop ---
    log_interval = train_cfg.log_interval
    sample_every = train_cfg.sample_every
    ckpt_interval = train_cfg.checkpoint_interval  # in epochs

    metrics_history = defaultdict(list)
    step = start_step
    start_time = time.time()

    print(f"[Train] Starting training from step {step}")

    with mesh:
        for epoch in range(train_cfg.epochs):
            epoch_start = epoch * steps_per_epoch
            if step >= epoch_start + steps_per_epoch:
                continue  # skip completed epochs on resume

            for local_step in range(steps_per_epoch):
                if step < start_step:
                    step += 1
                    continue

                # --- Get batch ---
                batch = next(train_iter)
                batch = shard_batch(batch, mesh)
                images = batch["image"]  # (B, H, W, 3) in [0, 1]

                rng, rng_step = jax.random.split(rng)
                rng_gen, rng_disc_step = jax.random.split(rng_step)

                # --- Generator step ---
                decoder_params, decoder_opt_state, ema_params, gen_metrics = \
                    train_step_generator_simple(
                        decoder_params, decoder_opt_state, ema_params,
                        images, rae_model, disc_model, lpips_model, diffaug,
                        gen_optimizer, rng_gen, epoch,
                        train_cfg.ema_decay, loss_cfg, mesh,
                    )

                # --- Discriminator step ---
                disc_loss_val = jnp.zeros(())
                if epoch >= loss_cfg["disc_upd_start"]:
                    # Get reconstruction for disc (no grad)
                    rng_rec, _ = jax.random.split(rng_disc_step)
                    x_rec = jax.lax.stop_gradient(
                        rae_model.forward(images, rng=rng_rec, training=False)
                    )

                    disc_params, disc_opt_state, disc_loss_val = \
                        train_step_disc_simple(
                            disc_params, disc_opt_state,
                            images, x_rec, disc_model, diffaug,
                            disc_optimizer, rng_disc_step,
                            loss_cfg["disc_loss"], mesh,
                        )

                # --- Logging ---
                for k, v in gen_metrics.items():
                    metrics_history[k].append(float(v))
                metrics_history["d_loss"].append(float(disc_loss_val))

                if (step + 1) % log_interval == 0 and jax.process_index() == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = log_interval / elapsed

                    summary = {
                        f"train/{k}": sum(metrics_history[k][-log_interval:]) / log_interval
                        for k in metrics_history
                    }
                    summary["train/steps_per_sec"] = steps_per_sec
                    summary["train/epoch"] = epoch
                    summary["train/step"] = step + 1

                    print(f"[Step {step + 1}/{total_steps}] "
                          f"epoch={epoch} "
                          f"loss={summary.get('train/total_loss', 0):.4f} "
                          f"rec={summary.get('train/rec_loss', 0):.4f} "
                          f"lpips={summary.get('train/lpips_loss', 0):.4f} "
                          f"g={summary.get('train/g_loss', 0):.4f} "
                          f"d={summary.get('train/d_loss', 0):.4f} "
                          f"({steps_per_sec:.1f} steps/s)")

                    if config.wandb.enabled:
                        wandb_utils.log(summary, step=step + 1)

                    start_time = time.time()

                # --- Sample ---
                if (step + 1) % sample_every == 0 and jax.process_index() == 0:
                    sample_images = images[:8]
                    rng_sample, _ = jax.random.split(rng)
                    x_rec_sample = rae_model.forward(sample_images, rng=rng_sample, training=False)
                    x_rec_np = np.array(x_rec_sample)
                    wandb_utils.log_image(x_rec_np, key="reconstructions", step=step + 1)

                step += 1

            # --- Epoch checkpoint ---
            if (epoch + 1) % ckpt_interval == 0:
                save_checkpoint(
                    ckpt_mngr, step,
                    jax.device_get(decoder_params),
                    jax.device_get(ema_params),
                    jax.device_get(decoder_opt_state),
                )

    print(f"[Train] Training complete. Final step: {step}")


# ---------------------------------------------------------------------------
# Train step variants for mesh-based data parallelism
#
# TPU gradient sync strategy:
#   - Data tensors (images) are SHARDED across devices via NamedSharding(mesh, P("data"))
#   - Model params are REPLICATED via NamedSharding(mesh, P())
#   - When computing grad(loss) on sharded data with replicated params,
#     XLA automatically inserts an all-reduce (mean) on gradients.
#   - This means we do NOT need explicit jax.lax.pmean calls.
#   - This ONLY works when running inside `with mesh:` context AND
#     data is properly sharded via shard_batch().
# ---------------------------------------------------------------------------
@jax.jit
def train_step_generator_simple(
    decoder_params, decoder_opt_state, ema_params,
    images, rae_model, disc_model, lpips_model, diffaug,
    gen_optimizer, rng, epoch, ema_decay, loss_cfg, mesh,
):
    """Generator step with mesh-based data parallelism.

    Gradient sync: automatic via XLA when data is sharded + params replicated.
    """
    disc_start = loss_cfg["disc_start"]
    lpips_start = loss_cfg["lpips_start"]
    perceptual_weight = loss_cfg["perceptual_weight"]
    disc_weight_val = loss_cfg["disc_weight"]

    rng_noise, rng_aug = jax.random.split(rng)

    def loss_fn(dec_params):
        z = rae_model.encode(images, rng=rng_noise, training=True)
        x_rec = rae_model.decode(z)
        target = images.transpose(0, 3, 1, 2)

        # jnp.mean over sharded batch → XLA computes global mean
        rec_loss = jnp.mean(jnp.abs(x_rec - target))
        total_loss = rec_loss
        lpips_val = jnp.zeros(())
        g_loss_val = jnp.zeros(())

        # LPIPS
        lpips_val = jax.lax.cond(
            epoch >= lpips_start,
            lambda: lpips_model(x_rec, target),
            lambda: jnp.zeros(()),
        )
        total_loss = total_loss + perceptual_weight * lpips_val

        # GAN loss
        def _gan_loss():
            x_aug = diffaug(x_rec, rng_aug)
            logits_fake = disc_model.classify(x_aug)
            return vanilla_g_loss(logits_fake)

        g_loss_val = jax.lax.cond(
            epoch >= disc_start,
            _gan_loss,
            lambda: jnp.zeros(()),
        )
        total_loss = total_loss + disc_weight_val * g_loss_val

        return total_loss, (rec_loss, lpips_val, g_loss_val)

    (total_loss, (rec_loss, lpips_val, g_loss_val)), grads = \
        jax.value_and_grad(loss_fn, has_aux=True)(decoder_params)

    # NOTE: grads are auto-reduced (mean) by XLA because:
    #   - `images` is sharded (P("data"))
    #   - `decoder_params` is replicated (P())
    #   - grad of replicated w.r.t. sharded → auto all-reduce mean

    updates, new_opt_state = gen_optimizer.update(grads, decoder_opt_state, decoder_params)
    new_params = jax.tree.map(lambda p, u: p + u, decoder_params, updates)
    new_ema = update_ema(ema_params, new_params, ema_decay)

    metrics = {
        "total_loss": total_loss,
        "rec_loss": rec_loss,
        "lpips_loss": lpips_val,
        "g_loss": g_loss_val,
    }

    return new_params, new_opt_state, new_ema, metrics


@jax.jit
def train_step_disc_simple(
    disc_params, disc_opt_state,
    real_images, fake_images, disc_model, diffaug,
    disc_optimizer, rng, loss_type, mesh,
):
    """Discriminator step with mesh-based data parallelism.

    Gradient sync: automatic via XLA (same as generator step).
    """
    rng_real, rng_fake = jax.random.split(rng)

    def disc_loss_fn(d_params):
        real_nchw = real_images.transpose(0, 3, 1, 2)
        real_aug = diffaug(real_nchw, rng_real)
        fake_aug = diffaug(fake_images, rng_fake)

        logits_real = disc_model.classify(real_aug)
        logits_fake = disc_model.classify(fake_aug)

        if loss_type == "hinge":
            return hinge_d_loss(logits_real, logits_fake)
        else:
            return vanilla_d_loss(logits_real, logits_fake)

    d_loss, grads = jax.value_and_grad(disc_loss_fn)(disc_params)

    # NOTE: grads auto-reduced by XLA (sharded data + replicated params)

    updates, new_opt_state = disc_optimizer.update(grads, disc_opt_state, disc_params)
    new_params = jax.tree.map(lambda p, u: p + u, disc_params, updates)

    return new_params, new_opt_state, d_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAE Stage 1 Training (JAX)")
    parser.add_argument("--config", type=str, default="configs/DINOv2-B_decXL.yaml",
                        help="Path to config YAML")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable WandB logging")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results directory")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.wandb:
        config.wandb.enabled = True
    if args.results_dir:
        config.experiment.results_dir = args.results_dir

    train(config)


if __name__ == "__main__":
    main()
