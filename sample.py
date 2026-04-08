"""Sample new images from a pre-trained DiT (single device). JAX port."""

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
from utils.train_utils import parse_configs
from utils.resume_utils import restore_checkpoint, build_checkpoint_manager


def main(args):
    rng = jax.random.PRNGKey(args.seed)

    full_cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, sampler_config,
     guidance_config, misc, _, _) = parse_configs(full_cfg)

    misc = OmegaConf.to_container(misc or {}, resolve=True)
    num_classes = misc.get("num_classes", 1000)
    # latent_size from config is (C, H, W) — convert to NHWC (H, W, C)
    cfg_latent = list(misc.get("latent_size", [768, 16, 16]))
    latent_size = (int(cfg_latent[1]), int(cfg_latent[2]), int(cfg_latent[0]))  # (H, W, C)
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"latent_size (H,W,C)={latent_size}, time_dist_shift={time_dist_shift:.4f}")

    # ── Transport ──────────────────────────────────────────────────
    transport_params = dict(OmegaConf.to_container(transport_config.get("params", {}), resolve=True))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)

    sampler_cfg = OmegaConf.to_container(sampler_config or {}, resolve=True)
    mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Sampler mode {mode}")

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
            print(f"[Warning] No checkpoint found in {args.ckpt_dir}, using random weights.")
        else:
            key = "ema" if args.use_ema else "model"
            graphdef, _ = nnx.split(model)
            model = nnx.merge(graphdef, ckpt[key])
            print(f"[ckpt] loaded DiT {'EMA' if args.use_ema else 'model'} from step {step}")

    # ── RAE decoder ───────────────────────────────────────────────
    rae_params_dict = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    rae_params_dict["noise_tau"] = 0.0  # inference: no noise
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
            print(f"[ckpt] loaded RAE decoder from step {step_rae}")

    # ── Sample ────────────────────────────────────────────────────
    class_labels = args.class_labels or [0, 0]
    n = len(class_labels)
    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, (n, *latent_size))
    y = jnp.array(class_labels, dtype=jnp.int32)

    guidance_cfg = OmegaConf.to_container(guidance_config or {}, resolve=True)
    guidance_scale = args.cfg_scale or float(guidance_cfg.get("scale", 1.0))

    if guidance_scale > 1.0:
        z_in = jnp.concatenate([z, z], axis=0)
        y_null = jnp.full((n,), num_classes, dtype=jnp.int32)
        y_in = jnp.concatenate([y, y_null], axis=0)
        model_fn = partial(model.forward_with_cfg, cfg_scale=guidance_scale)
        model_kwargs = dict(y=y_in)
    else:
        z_in = z
        model_fn = lambda x, t, y: model(x, t, y, training=False)
        model_kwargs = dict(y=y)

    print(f"Sampling {n} images (cfg_scale={guidance_scale})...")
    start = time.time()
    samples = sample_fn(z_in, model_fn, **model_kwargs)
    if guidance_scale > 1.0:
        samples = samples[:n]
    print(f"Sampling took {time.time() - start:.2f}s")

    # ── Decode latents → images with RAE ──────────────────────────
    images = rae.decode(samples)
    images = jnp.clip(images, 0.0, 1.0)

    # ── Save ──────────────────────────────────────────────────────
    images_np = np.array(images)
    # decode returns NCHW → convert to NHWC for saving
    if images_np.ndim == 4 and images_np.shape[1] in (1, 3, 4):
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    images_uint8 = (images_np * 255).astype(np.uint8)

    os.makedirs(args.output_dir, exist_ok=True)
    for i, img in enumerate(images_uint8):
        out_path = os.path.join(args.output_dir, f"sample_{i:04d}.png")
        Image.fromarray(img).save(out_path)
    print(f"Saved {len(images_uint8)} samples to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="CFG scale (overrides config). Use >1 to enable CFG.")
    parser.add_argument("--class-labels", type=int, nargs="+", default=None,
                        help="Class labels to sample. E.g. --class-labels 0 0 0")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Stage-2 DiT checkpoint directory")
    parser.add_argument("--rae-ckpt-dir", type=str, default=None,
                        help="Stage-1 RAE checkpoint directory (for decoder)")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--output-dir", type=str, default="samples/gen")
    args = parser.parse_args()
    main(args)
