"""Sample new images from a pre-trained DiT. JAX port."""

from __future__ import annotations

import argparse
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from PIL import Image

from stage2 import DiTwDDTHead, create_transport, Sampler
from stage2.models.lightningDiT import LightningDiT
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def main(args):
    rng = jax.random.PRNGKey(args.seed)

    full_cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, sampler_config,
     guidance_config, misc, _, _) = parse_configs(full_cfg)

    misc = OmegaConf.to_container(misc or {}, resolve=True)
    num_classes = misc.get("num_classes", 1000)
    latent_size = tuple(int(d) for d in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"Using time_dist_shift={time_dist_shift:.4f}")

    # Transport
    transport_params = dict(OmegaConf.to_container(transport_config.get("params", {}), resolve=True))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)

    sampler_cfg = OmegaConf.to_container(sampler_config or {}, resolve=True)
    mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    else:
        raise NotImplementedError(f"Sampler mode {mode}")

    # Model
    rngs = nnx.Rngs(params=0)
    model_params = dict(OmegaConf.to_container(model_config.get("params", {}), resolve=True))
    model_target = model_config.get("target", "stage2.DiTwDDTHead")
    if "DDT" in model_target:
        model = DiTwDDTHead(**model_params, rngs=rngs)
    else:
        model = LightningDiT(**model_params, rngs=rngs)

    # TODO: Load checkpoint weights
    # ckpt_path = model_config.get("ckpt", None)
    # if ckpt_path: load_checkpoint(...)

    # RAE
    # rae = instantiate_from_config(rae_config, rngs=rngs)

    # Sample
    class_labels = [207, 360]
    n = len(class_labels)
    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, (n, *latent_size))
    y = jnp.array(class_labels, dtype=jnp.int32)

    guidance_cfg = OmegaConf.to_container(guidance_config or {}, resolve=True)
    guidance_scale = float(guidance_cfg.get("scale", 1.0))

    if guidance_scale > 1.0:
        z = jnp.concatenate([z, z], axis=0)
        y_null = jnp.full((n,), num_classes, dtype=jnp.int32)
        y = jnp.concatenate([y, y_null], axis=0)
        t_min = float(guidance_cfg.get("t_min", 0.0))
        t_max = float(guidance_cfg.get("t_max", 1.0))

        from functools import partial
        model_fn = partial(model.forward_with_cfg,
                           cfg_scale=guidance_scale,
                           cfg_interval=(t_min, t_max))
        model_kwargs = dict(y=y)
    else:
        model_fn = lambda x, t, y: model(x, t, y, training=False)
        model_kwargs = dict(y=y)

    start = time.time()
    samples = sample_fn(z, model_fn, **model_kwargs)
    if guidance_scale > 1.0:
        samples = samples[:n]
    print(f"Sampling took {time.time() - start:.2f}s")

    # Decode with RAE
    # images = rae.decode(samples)
    # images = jnp.clip(images, 0, 1)
    images = jnp.clip(samples, 0, 1)  # placeholder

    # Save
    images_np = np.array(images)
    if images_np.shape[1] in (1, 3, 4):  # NCHW → NHWC
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    images_uint8 = (images_np * 255).astype(np.uint8)

    for i, img in enumerate(images_uint8):
        Image.fromarray(img).save(f"sample_{i}.png")
    print(f"Saved {len(images_uint8)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
