"""Distributed sampling — generate images across all TPU cores. JAX port."""

from __future__ import annotations

import argparse
import math
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from PIL import Image

from stage1 import RAE
from stage2 import DiTwDDTHead, create_transport, Sampler
from stage2.models.lightningDiT import LightningDiT
from utils.device_utils import create_mesh, get_replicated_sharding
from utils.train_utils import parse_configs
from utils.resume_utils import restore_checkpoint, build_checkpoint_manager


def main(args):
    mesh = create_mesh()
    repl_sharding = get_replicated_sharding(mesh)
    num_devices = jax.device_count()
    is_main = jax.process_index() == 0

    rng = jax.random.PRNGKey(args.seed + jax.process_index())

    full_cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, sampler_config,
     guidance_config, misc, _, _) = parse_configs(full_cfg)

    misc = OmegaConf.to_container(misc or {}, resolve=True)
    num_classes = misc.get("num_classes", 1000)
    cfg_latent = list(misc.get("latent_size", [768, 16, 16]))
    latent_size = (int(cfg_latent[1]), int(cfg_latent[2]), int(cfg_latent[0]))  # (H, W, C)
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    # ── Transport + sampler ────────────────────────────────────────
    transport_params = dict(OmegaConf.to_container(transport_config.get("params", {}), resolve=True))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sampler_cfg = OmegaConf.to_container(sampler_config or {}, resolve=True)
    sample_fn = sampler.sample_ode(**dict(sampler_cfg.get("params", {})))

    # ── DiT model ─────────────────────────────────────────────────
    rngs = nnx.Rngs(params=0, dropout=1)
    model_params = dict(OmegaConf.to_container(model_config.get("params", {}), resolve=True))
    model_target = model_config.get("target", "stage2.DiTwDDTHead")
    if "DDT" in model_target:
        model = DiTwDDTHead(**model_params, rngs=rngs)
    else:
        model = LightningDiT(**model_params, rngs=rngs)

    # Load DiT checkpoint
    if args.ckpt_dir:
        mngr = build_checkpoint_manager(args.ckpt_dir)
        ckpt, step = restore_checkpoint(mngr)
        if ckpt is None:
            if is_main:
                print(f"[Warning] No checkpoint found in {args.ckpt_dir}")
        else:
            key = "ema" if args.use_ema else "model"
            graphdef, _ = nnx.split(model)
            model_state = ckpt[key]
            model_state = jax.device_put(model_state, repl_sharding)
            model = nnx.merge(graphdef, model_state)
            if is_main:
                print(f"[ckpt] loaded DiT {'EMA' if args.use_ema else 'model'} from step {step}")

    # ── RAE decoder ───────────────────────────────────────────────
    rae_params_dict = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    rae_params_dict["noise_tau"] = 0.0
    rae = RAE(**rae_params_dict, rngs=rngs)

    if args.rae_ckpt_dir:
        mngr_rae = build_checkpoint_manager(args.rae_ckpt_dir)
        ckpt_rae, step_rae = restore_checkpoint(mngr_rae)
        if ckpt_rae is not None:
            key = "ema" if args.use_ema else "model"
            decoder_state = ckpt_rae[key]
            graphdef_dec, _ = nnx.split(rae.decoder)
            loaded_decoder = nnx.merge(graphdef_dec, decoder_state)
            nnx.update(rae.decoder, nnx.state(loaded_decoder))
            if is_main:
                print(f"[ckpt] loaded RAE decoder from step {step_rae}")

    # ── Guidance config ────────────────────────────────────────────
    guidance_cfg = OmegaConf.to_container(guidance_config or {}, resolve=True)
    guidance_scale = args.cfg_scale or float(guidance_cfg.get("scale", 1.0))
    use_guidance = guidance_scale > 1.0

    # ── Distributed sampling ───────────────────────────────────────
    total_samples = args.num_fid_samples
    samples_per_process = total_samples // num_devices
    # Handle remainder
    if jax.process_index() < (total_samples % num_devices):
        samples_per_process += 1

    all_samples = []
    processed = 0

    if is_main:
        print(f"Generating {total_samples} samples ({samples_per_process} per process)...")
    start = time.time()

    with mesh:
        for batch_start in range(0, samples_per_process, args.batch_size):
            n = min(args.batch_size, samples_per_process - batch_start)
            rng, noise_rng, label_rng = jax.random.split(rng, 3)
            z = jax.random.normal(noise_rng, (n, *latent_size))
            y = jax.random.randint(label_rng, (n,), 0, num_classes)

            if use_guidance:
                z_in = jnp.concatenate([z, z], axis=0)
                y_null = jnp.full((n,), num_classes, dtype=jnp.int32)
                y_in = jnp.concatenate([y, y_null], axis=0)
                model_fn = partial(model.forward_with_cfg, cfg_scale=guidance_scale)
                latents = sample_fn(z_in, model_fn, y=y_in)
                latents = latents[:n]
            else:
                model_fn = lambda x, t, y: model(x, t, y, training=False)
                latents = sample_fn(z, model_fn, y=y)

            # Decode latents → images via RAE
            images = rae.decode(latents)
            images = jnp.clip(images, 0.0, 1.0)
            imgs_np = np.array(images)

            # RAE decode returns NCHW → convert to NHWC
            if imgs_np.ndim == 4 and imgs_np.shape[1] in (1, 3, 4):
                imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))

            all_samples.append((imgs_np * 255).astype(np.uint8))
            processed += n

            if is_main and processed % (args.batch_size * 10) == 0:
                elapsed = time.time() - start
                print(f"[progress] {processed}/{samples_per_process} ({elapsed:.1f}s)")

    all_samples_np = np.concatenate(all_samples, axis=0)

    # ── Save shard ─────────────────────────────────────────────────
    os.makedirs(args.sample_dir, exist_ok=True)
    shard_path = os.path.join(args.sample_dir, f"samples_{jax.process_index():02d}.npz")
    np.savez(shard_path, arr_0=all_samples_np)

    if is_main:
        elapsed = time.time() - start
        print(f"[done] {all_samples_np.shape[0]} samples saved to {shard_path} ({elapsed:.1f}s)")
        print(f"Total across {num_devices} devices: ~{total_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-fid-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-dir", type=str, default="samples/gen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Stage-2 DiT checkpoint directory")
    parser.add_argument("--rae-ckpt-dir", type=str, default=None,
                        help="Stage-1 RAE checkpoint directory (for decoder)")
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()
    main(args)
