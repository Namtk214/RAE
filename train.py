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
# CLI
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 2 transport model on RAE latents (JAX)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (optional — if omitted, uses hardcoded defaults + CLI flags)")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--results-dir", type=str, default="ckpts")
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Experiment name (default: auto from config or 'stage2_run')")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--dataset-type", type=str, default=None,
                    choices=["imagefolder", "tfds"],
                    help="Data source type. Overrides config 'data.source'.")
    p.add_argument("--tfds-builder-dir", type=str, default=None,
                    help="Path to custom TFDS builder dir (e.g. tfds_builders/celebahq256).")
    p.add_argument("--tfds-name", type=str, default=None,
                   help="TFDS dataset name (e.g. celebahq256)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="rae-jax-stage2")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--global-seed", type=int, default=None)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--eval-fid-every", type=int, default=0, help="Evaluate FID with samples every N steps")
    p.add_argument("--num-fid-samples", type=int, default=50000, help="Number of samples for FID")
    p.add_argument("--rae-checkpoint", type=str, default=None,
                    help="Path to Stage 1 checkpoint pkl file to load RAE weights from.")
    p.add_argument("--reference-npz-path", type=str, default=None,
                   help="Pre-computed FID reference .npz")
    p.add_argument("--normalization-stat-path", type=str, default=None,
                   help="Latent normalization stats .npz")
    p.add_argument("--pretrained-decoder-path", type=str, default=None,
                   help="Pretrained decoder .pt weights")

    # ── Training knobs (used when --config is not provided) ──────
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-train-samples", type=int, default=None,
                   help="Dataset size — used to compute steps/epoch")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Number of classes (1 for unconditional, 1000 for ImageNet)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Hardcoded defaults (used when --config is not provided)
# ─────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    # RAE
    rae_params=dict(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="facebook/dinov2-base",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "facebook/dinov2-base", "normalize": True},
        decoder_config_path="configs/decoder/ViTXL",
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path="models/stats/dinov2/normalization_stats.npz",
    ),
    # DiT model (DiT-S for CelebAHQ256)
    model_target="stage2.models.DDT.DiTwDDTHead",
    model_params=dict(
        input_size=16, patch_size=1, in_channels=768,
        hidden_size=[384, 2048], depth=[12, 2], num_heads=[6, 16],
        mlp_ratio=4.0, class_dropout_prob=0.0, num_classes=1,
        use_qknorm=False, use_swiglu=True, use_rope=True,
        use_rmsnorm=True, wo_shift=False, use_pos_embed=True,
    ),
    # Misc
    num_classes=1, null_label=1,
    latent_size=[768, 16, 16],
    time_dist_shift_dim=196608,
    time_dist_shift_base=4096,
    # Transport
    transport_params=dict(path_type="Linear", prediction="velocity",
                          time_dist_type="logit-normal_0_1"),
    # Sampler
    sampler_mode="ODE",
    sampler_params=dict(sampling_method="euler", num_steps=50,
                        atol=1e-6, rtol=1e-3, reverse=False),
    # Training (CelebAHQ256 defaults)
    batch_size=16, global_batch_size=128,
    ema_decay=0.9995, epochs=200, log_interval=50,
    sample_every=5000, checkpoint_interval=5000, clip_grad=1.0,
    global_seed=42,
    # Optimizer
    lr=2e-4, betas=[0.9, 0.95], weight_decay=0.0,
    schedule_type="linear", warmup_epochs=10, final_lr=2e-5,
    warmup_from_zero=False,
    # Guidance
    guidance_scale=1.0,
    # Data (CelebAHQ256)
    num_train_samples=30000, dataset_source="tfds",
)


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

    # ── Config: load from YAML or use hardcoded defaults ──
    if args.config is not None:
        full_cfg = OmegaConf.load(args.config)
        (rae_config, model_config, transport_config, sampler_config,
         guidance_config, misc_config, training_config, eval_config) = parse_configs(full_cfg)

        misc = OmegaConf.to_container(misc_config or {}, resolve=True)
        transport_cfg = OmegaConf.to_container(transport_config or {}, resolve=True)
        sampler_cfg = OmegaConf.to_container(sampler_config or {}, resolve=True)
        guidance_cfg = OmegaConf.to_container(guidance_config or {}, resolve=True)
        training_cfg = OmegaConf.to_container(training_config or {}, resolve=True)
        data_cfg = OmegaConf.to_container(full_cfg.get("data", {}), resolve=True) if "data" in full_cfg else {}
        eval_cfg_raw = OmegaConf.to_container(eval_config or {}, resolve=True) if eval_config else {}

        rae_params = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
        model_params = dict(OmegaConf.to_container(model_config.get("params", {}), resolve=True))
        model_target = model_config.get("target", "stage2.DiTwDDTHead")

        num_classes = int(misc.get("num_classes", 1000))
        null_label = int(misc.get("null_label", num_classes))
        cfg_latent = list(misc.get("latent_size", [768, 16, 16]))
        shift_dim = misc.get("time_dist_shift_dim", None)
        shift_base = misc.get("time_dist_shift_base", 4096)

        opt_cfg = training_cfg.get("optimizer", {})
        sched_cfg = training_cfg.get("scheduler", {})

        per_device_batch = int(training_cfg.get("batch_size", 16))
        global_batch_size = per_device_batch * num_devices
        if "global_batch_size" in training_cfg:
            global_batch_size = int(training_cfg["global_batch_size"])
            per_device_batch = global_batch_size // num_devices

        ema_decay = float(training_cfg.get("ema_decay", 0.9995))
        num_epochs = int(training_cfg.get("epochs", 1400))
        log_interval = int(training_cfg.get("log_interval", 100))
        sample_every = int(training_cfg.get("sample_every", 2500))
        checkpoint_interval = int(training_cfg.get("checkpoint_interval", 10))
        clip_grad = float(training_cfg.get("clip_grad", 1.0))
        global_seed = args.global_seed if args.global_seed is not None else int(training_cfg.get("global_seed", 0))

        guidance_scale = float(guidance_cfg.get("scale", 1.0))

        dataset_size = int(data_cfg.get("num_train_samples", 1281167))
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 40))

        lr = float(opt_cfg.get("lr", 2e-4))
        betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        schedule_type = str(sched_cfg.get("type", "linear"))
        final_lr = float(sched_cfg.get("final_lr", 2e-5))
        warmup_from_zero = bool(sched_cfg.get("warmup_from_zero", False))

        transport_params_raw = dict(transport_cfg.get("params", {}))
        transport_params_raw.pop("time_dist_shift", None)

        sampler_mode = sampler_cfg.get("mode", "ODE").upper()
        sampler_params_raw = dict(sampler_cfg.get("params", {}))

        dataset_type = args.dataset_type or data_cfg.get("source", "imagefolder")
        tfds_builder_dir = args.tfds_builder_dir or data_cfg.get("tfds_builder_dir", None)
        tfds_name = args.tfds_name or data_cfg.get("dataset_name", None)

        do_eval_cfg = bool(eval_cfg_raw)
        eval_interval = int(eval_cfg_raw.get("eval_interval", 0))
        reference_npz_path = args.reference_npz_path or eval_cfg_raw.get("reference_npz_path", None)

        exp_name = args.experiment_name or f"stage2-{Path(args.config).stem}"
        wandb_config = OmegaConf.to_container(full_cfg, resolve=True)

    else:
        # ── No config file: use hardcoded defaults ──
        D = _DEFAULTS

        rae_params = dict(D["rae_params"])
        if args.pretrained_decoder_path:
            rae_params["pretrained_decoder_path"] = args.pretrained_decoder_path
        if args.normalization_stat_path:
            rae_params["normalization_stat_path"] = args.normalization_stat_path

        model_params = dict(D["model_params"])
        model_target = D["model_target"]

        num_classes = args.num_classes if args.num_classes is not None else D["num_classes"]
        null_label = num_classes
        # Override model_params num_classes if CLI flag provided
        if args.num_classes is not None:
            model_params["num_classes"] = num_classes
        cfg_latent = list(D["latent_size"])
        shift_dim = D.get("time_dist_shift_dim", None)
        shift_base = D["time_dist_shift_base"]

        global_batch_size = args.global_batch_size or D["global_batch_size"]
        per_device_batch = global_batch_size // num_devices

        ema_decay = D["ema_decay"]
        num_epochs = args.epochs or D["epochs"]
        log_interval = D["log_interval"]
        sample_every = D["sample_every"]
        checkpoint_interval = D["checkpoint_interval"]
        clip_grad = D["clip_grad"]
        global_seed = args.global_seed if args.global_seed is not None else D["global_seed"]

        guidance_scale = D["guidance_scale"]

        dataset_size = args.num_train_samples or D["num_train_samples"]
        warmup_epochs = D["warmup_epochs"]

        lr = args.lr or D["lr"]
        betas = tuple(D["betas"])
        weight_decay = D["weight_decay"]
        schedule_type = D["schedule_type"]
        final_lr = D["final_lr"]
        warmup_from_zero = D["warmup_from_zero"]

        transport_params_raw = dict(D["transport_params"])
        sampler_mode = D["sampler_mode"]
        sampler_params_raw = dict(D["sampler_params"])

        dataset_type = args.dataset_type or D["dataset_source"]
        tfds_builder_dir = args.tfds_builder_dir
        tfds_name = args.tfds_name

        do_eval_cfg = False
        eval_interval = 0
        reference_npz_path = args.reference_npz_path

        exp_name = args.experiment_name or "stage2_run"
        wandb_config = vars(args)

    # ── Derived values (common to both modes) ──
    latent_size_chw = tuple(int(d) for d in cfg_latent)  # (C, H, W)
    latent_size = (latent_size_chw[1], latent_size_chw[2], latent_size_chw[0])  # (H, W, C) NHWC
    if shift_dim is None:
        shift_dim = math.prod(latent_size)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    num_processes = jax.process_count()
    per_host_batch = global_batch_size // num_processes
    assert global_batch_size % num_devices == 0, \
        f"global_batch_size {global_batch_size} must be divisible by {num_devices} devices"
    assert global_batch_size % num_processes == 0, \
        f"global_batch_size {global_batch_size} must be divisible by {num_processes} processes"

    use_guidance = guidance_scale > 1.0
    compute_dtype = jnp.bfloat16 if args.precision == "bf16" else jnp.float32

    steps_per_epoch_est = dataset_size // global_batch_size
    total_training_steps = num_epochs * steps_per_epoch_est
    warmup_steps = warmup_epochs * steps_per_epoch_est

    # ── Experiment dir ──
    experiment_dir, checkpoint_dir = configure_experiment_dirs(
        args.results_dir, exp_name,
    )
    ckpt_mngr = build_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING)
    logger = logging.getLogger("train_stage2")

    # ── WandB ──
    if args.wandb and is_main:
        wandb_utils.initialize(
            config=wandb_config,
            entity=args.wandb_entity or os.environ.get("ENTITY", ""),
            exp_name=exp_name,
            project_name=args.wandb_project,
        )

    # ── Seed ──
    rng = jax.random.PRNGKey(global_seed)

    # ── RAE (frozen encoder + decoder for eval) ──
    rngs = nnx.Rngs(params=0, dropout=1)
    logger.info("Loading RAE...")
    rae = RAE(**rae_params, rngs=rngs)

    # Load pretrained Stage 1 weights
    if args.rae_checkpoint:
        import pickle
        logger.info(f"Loading Stage 1 RAE weights from: {args.rae_checkpoint}")
        try:
            with open(args.rae_checkpoint, "rb") as f:
                ckpt = pickle.load(f)
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
    if "DDT" in model_target:
        model = DiTwDDTHead(**model_params, rngs=rngs, dtype=compute_dtype)
    else:
        model = LightningDiT(**model_params, rngs=rngs, dtype=compute_dtype)

    model_param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    logger.info(f"DiT parameters: {model_param_count / 1e6:.2f}M")

    # ── Replicate model params across all devices ──
    graphdef, model_state = nnx.split(model)
    model_state = jax.device_put(model_state, repl_sharding)

    # ── EMA (replicated) ──
    ema_state = jax.tree.map(jnp.copy, model_state)

    # ── Optimizer ──
    optimizer = build_optimizer_with_schedule(
        lr=lr, betas=betas, weight_decay=weight_decay, clip_grad=clip_grad,
        schedule_type=schedule_type, warmup_steps=warmup_steps,
        total_steps=total_training_steps, final_lr=final_lr,
        warmup_from_zero=warmup_from_zero,
    )
    opt_state = optimizer.init(nnx.state(model))
    opt_state = jax.device_put(opt_state, repl_sharding)
    logger.info(f"Optimizer: AdamW lr={lr}, schedule={schedule_type}, "
                f"warmup={warmup_steps}, total={total_training_steps}")

    # ── Resume from Stage 2 checkpoint if available ──
    restored_ckpt, restored_step = restore_checkpoint(ckpt_mngr)
    global_step_resume = 0

    # Debug: print param stats BEFORE restore
    def _param_stats(state, label):
        leaves = jax.tree.leaves(state)
        all_vals = [np.asarray(l).ravel() for l in leaves if hasattr(l, 'shape') and l.size > 0]
        if all_vals:
            cat = np.concatenate(all_vals)
            logger.info(f"  [{label}] #params={cat.size:,}  mean={cat.mean():.6f}  std={cat.std():.6f}  "
                        f"min={cat.min():.6f}  max={cat.max():.6f}")
        else:
            logger.info(f"  [{label}] (no params found)")

    logger.info("=== DEBUG: DiT param stats BEFORE checkpoint restore ===")
    _param_stats(model_state, "model_state")
    _param_stats(ema_state, "ema_state")

    if restored_ckpt is not None:
        logger.info(f"Resuming DiT from Stage 2 checkpoint at step {restored_step}...")
        logger.info(f"  Checkpoint keys: {list(restored_ckpt.keys())}")
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

        # Debug: print param stats AFTER restore
        logger.info("=== DEBUG: DiT param stats AFTER checkpoint restore ===")
        _param_stats(model_state, "model_state")
        _param_stats(ema_state, "ema_state")
        logger.info(f"✅ Resumed successfully from step {restored_step}.")
    else:
        logger.info("⚠️ No Stage 2 checkpoint found — starting from scratch with random DiT weights.")

    # ── Transport ──
    transport = create_transport(**transport_params_raw, time_dist_shift=time_dist_shift)

    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sample_fn = transport_sampler.sample_ode(**sampler_params_raw)
    elif sampler_mode == "SDE":
        eval_sample_fn = transport_sampler.sample_sde(**sampler_params_raw)
    else:
        raise NotImplementedError(f"Sampler mode {sampler_mode}")

    # ── Data — each host reads per_host_batch rows ──
    ds, steps_per_epoch = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=per_host_batch,
        dataset_type=dataset_type,
        split="train",
        tfds_name=tfds_name,
        tfds_builder_dir=tfds_builder_dir,
    )
    ds_valid, _ = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=per_host_batch,
        dataset_type=dataset_type,
        split="validation" if dataset_type == "tfds" else "val",
        tfds_name=tfds_name,
        tfds_builder_dir=tfds_builder_dir,
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Global batch: {global_batch_size}, Per-device: {per_device_batch}")

    # ─────────────────────────────────────────────────────────────
    # OPT 1: JIT-compiled RAE encode step
    # Avoids re-tracing on every Python call and keeps encoding on TPU
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
    # OPT 2: JIT train step with donate_argnums
    # donate_argnums donates the input buffers so JAX can reuse them
    # without extra allocation (zero-copy param updates on TPU)
    # ─────────────────────────────────────────────────────────────
    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def train_step(model_state, opt_state, ema_state, batch_x, batch_y, step_rng):
        """Single training step — fully on-device, no Python sync."""

        def loss_fn(params):
            m = nnx.merge(graphdef, params)
            def model_forward(xt, t, y):
                return m(xt, t, y, training=True, rng=step_rng, return_activations=True)
            
            terms, acts = transport.training_losses(
                model_forward, batch_x, step_rng, has_aux=True, y=batch_y,
            )
            v_mag = jnp.sqrt(jnp.mean(jnp.square(terms["pred"])))
            return jnp.mean(terms["loss"]), (v_mag, acts)

        (loss, (v_mag, acts)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)

        # OPT 3: Gradient norm fully on-device (no Python list comprehension)
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

        # EMA update
        new_ema = jax.tree.map(
            lambda e, m: e * ema_decay + m * (1.0 - ema_decay),
            ema_state, new_model_state,
        )

        return loss, new_model_state, new_opt_state, new_ema, v_mag, acts

    # ── Eval ──
    do_eval = do_eval_cfg or args.eval_fid_every > 0
    eval_fid_every = args.eval_fid_every

    # ── Training loop ──
    global_step = global_step_resume
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    # OPT 4: Accumulate loss as JAX array to avoid per-step host sync
    running_stats = {}
    
    @jax.jit
    def valid_step(model_state, batch_x, batch_y, step_rng):
        def loss_fn(params):
            m = nnx.merge(graphdef, params)
            def model_forward(xt, t, y):
                return m(xt, t, y, training=False, rng=step_rng)
            terms = transport.training_losses(model_forward, batch_x, step_rng, y=batch_y)
            return jnp.mean(terms["loss"])
        return loss_fn(model_state)

    start_time = time.time()

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Compute dtype: {compute_dtype}")

    model_state = nnx.state(model)

    with mesh:
        for epoch in range(start_epoch, num_epochs):
            # Checkpoint is now saved at step level

            if is_main:
                steps_iter = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{num_epochs}")
            else:
                steps_iter = range(steps_per_epoch)

            for _ in steps_iter:
                step_data = next(ds)

                # OPT 5: Shard image batch using multi-host-safe shard_batch
                sharded = shard_batch(step_data, mesh)
                images = sharded["image"]
                label_np = step_data.get("label", None)
                if label_np is not None:
                    labels = sharded.get("label", jax.device_put(
                        jnp.zeros(images.shape[0], dtype=jnp.int32), data_sharding
                    ))
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

                # Checkpoint at specific global steps — save on ALL workers for multi-host resume
                if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                    if is_main:
                        logger.info(f"Saving checkpoint at step {global_step}...")
                    save_checkpoint(
                        ckpt_mngr, global_step,
                        jax.device_get(model_state),
                        jax.device_get(ema_state),
                        jax.device_get(opt_state),
                    )

                # Logging — only sync to host every log_interval steps
                if log_interval > 0 and global_step % log_interval == 0:
                    # All workers must run validation (JIT collective)
                    try:
                        valid_data = next(ds_valid)
                    except TypeError:
                        if not hasattr(ds_valid, "__next__"):
                            ds_valid = iter(ds_valid)
                        try:
                            valid_data = next(ds_valid)
                        except StopIteration:
                            ds_valid = iter(ds_valid)
                            valid_data = next(ds_valid)
                    
                    valid_sharded = shard_batch(valid_data, mesh)
                    valid_images = valid_sharded["image"]
                    valid_label_np = valid_data.get("label", None)
                    if valid_label_np is not None:
                        valid_labels = valid_sharded.get("label", jax.device_put(
                            jnp.zeros(valid_images.shape[0], dtype=jnp.int32), data_sharding
                        ))
                    else:
                        valid_labels = jax.device_put(jnp.zeros(valid_images.shape[0], dtype=jnp.int32), data_sharding)
                        
                    rng, val_rng = jax.random.split(rng)
                    valid_latents = encode_batch(rae_state, valid_images, val_rng)
                    loss_valid = float(valid_step(model_state, valid_latents, valid_labels, val_rng))

                    # Only log/print on main process
                    if is_main:
                        avg_stats = {k: float(v) / log_interval for k, v in running_stats.items()}
                        elapsed = time.time() - start_time
                        steps_per_sec = log_interval / elapsed

                        stats = {
                            "training/loss": avg_stats["loss"],
                            "training/loss_valid": loss_valid,
                            "training/v_magnitude_prime": avg_stats["v_magnitude_prime"],
                            "training/steps_per_sec": steps_per_sec,
                            "training/epoch": epoch,
                            "training/lr": float(optimizer.learning_rate(opt_state[1].count)
                                              if hasattr(optimizer, 'learning_rate') else lr),
                        }
                        for k, v in avg_stats.items():
                            if k.startswith("activations/"):
                                stats[f"training/{k}"] = v
                                
                        if hasattr(steps_iter, "set_postfix"):
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
                if sample_every > 0 and global_step % sample_every == 0:
                    if is_main:
                        logger.info("Generating EMA samples...")
                        logger.info(f"=== DEBUG: EMA state used for SAMPLING at step {global_step} ===")
                        _param_stats(ema_state, "ema_state_for_sampling")
                    rae_model_gen = nnx.merge(rae_graphdef, rae_state)
                    # Helper decode function for denoise visualization
                    @jax.jit
                    def decode_fn(z):
                        return jax.lax.stop_gradient(rae_model_gen.decode(z))

                    _generate_samples(
                        graphdef, ema_state, eval_sample_fn,
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
                        logger.info(f"Reference NPZ {reference_npz_path} not found. Generating it now via subprocess...")
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
                        
                    logger.info(f"=== DEBUG: EMA state used for FID EVALUATION at step {global_step} ===")
                    _param_stats(ema_state, "ema_state_for_fid")
                    eval_model = nnx.merge(graphdef, ema_state)
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
                        batch_size=min(per_device_batch * 4, 512),
                        experiment_dir=experiment_dir,
                        global_step=global_step,
                        reference_npz_path=reference_npz_path,
                        rae_decode_fn=eval_decode_fn,
                        mesh=mesh,
                    )
                    if eval_stats and args.wandb and is_main:
                        eval_stats = {f"eval/{k}": v for k, v in eval_stats.items()}
                        wandb_utils.log(eval_stats, step=global_step)

    # Final checkpoint — save on ALL workers
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
                      global_step, use_wandb, rae_decode_fn=None):
    """Generate visual samples using EMA model.
    
    IMPORTANT for multi-host: ALL workers must execute every JIT call
    (sampling + decode) symmetrically. Only the main worker logs to WandB.
    """
    is_main = jax.process_index() == 0
    n = min(8, labels.shape[0])
    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, (n, *latent_size))

    ema_model = nnx.merge(graphdef, ema_state)

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
        model_fn = lambda x, t, y=None, **kw: ema_model(x, t, y, training=False)
        samples, intermediates = sample_fn(
            z, model_fn, y=y,
            return_intermediates=True, log_every=log_every,
        )

    # ALL workers must call rae_decode_fn (JIT-compiled) to stay in sync.
    # Only main worker logs decoded images to WandB.
    if rae_decode_fn is not None:
        num_inter = len(intermediates)
        if is_main:
            print(f"Decoding {num_inter} intermediate denoising steps...")

        for idx, inter_z in enumerate(intermediates):
            ode_step = (idx + 1) * log_every
            images = np.array(rae_decode_fn(inter_z))  # ALL workers call this
            if use_wandb and is_main:
                images = np.clip(images, 0.0, 1.0)
                if images.ndim == 4 and images.shape[1] in (1, 3, 4):
                    images = np.transpose(images, (0, 2, 3, 1))
                wandb_utils.log_image(
                    images,
                    key=f"sample_denoise/step_{ode_step:03d}",
                    step=global_step,
                )

        # Decode final sample — ALL workers call this
        final_images = np.array(rae_decode_fn(samples))
        if use_wandb and is_main:
            final_images = np.clip(final_images, 0.0, 1.0)
            if final_images.ndim == 4 and final_images.shape[1] in (1, 3, 4):
                final_images = np.transpose(final_images, (0, 2, 3, 1))
            wandb_utils.log_image(final_images, key="sample/final", step=global_step)

    if is_main:
        print(f"Generated {n} latent samples at step {global_step}")


if __name__ == "__main__":
    main()
