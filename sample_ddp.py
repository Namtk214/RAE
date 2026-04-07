"""Distributed sampling — generate images across all TPU cores. JAX port."""

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
from functools import partial

from stage2 import DiTwDDTHead, create_transport, Sampler
from stage2.models.lightningDiT import LightningDiT
from utils.device_utils import setup_mesh
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def main(args):
    mesh = setup_mesh()
    num_devices = jax.device_count()
    process_index = jax.process_index()
    is_main = process_index == 0

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

    # Transport + sampler
    transport_params = dict(OmegaConf.to_container(transport_config.get("params", {}), resolve=True))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sampler_cfg = OmegaConf.to_container(sampler_config or {}, resolve=True)
    sample_fn = sampler.sample_ode(**dict(sampler_cfg.get("params", {})))

    # Model
    rngs = nnx.Rngs(params=0)
    model_params = dict(OmegaConf.to_container(model_config.get("params", {}), resolve=True))
    model_target = model_config.get("target", "stage2.DiTwDDTHead")
    if "DDT" in model_target:
        model = DiTwDDTHead(**model_params, rngs=rngs)
    else:
        model = LightningDiT(**model_params, rngs=rngs)

    # TODO: Load checkpoint

    guidance_cfg = OmegaConf.to_container(guidance_config or {}, resolve=True)
    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    use_guidance = guidance_scale > 1.0

    # Distribute sampling
    samples_per_device = args.num_samples // num_devices
    all_samples = []

    for batch_start in range(0, samples_per_device, args.batch_size):
        n = min(args.batch_size, samples_per_device - batch_start)
        rng, noise_rng, label_rng = jax.random.split(rng, 3)
        z = jax.random.normal(noise_rng, (n, *latent_size))
        y = jax.random.randint(label_rng, (n,), 0, num_classes)

        if use_guidance:
            z = jnp.concatenate([z, z], axis=0)
            y_null = jnp.full((n,), num_classes, dtype=jnp.int32)
            y = jnp.concatenate([y, y_null], axis=0)
            model_fn = partial(model.forward_with_cfg, cfg_scale=guidance_scale)
        else:
            model_fn = lambda x, t, y: model(x, t, y, training=False)

        latents = sample_fn(z, model_fn, y=y)
        if use_guidance:
            latents = latents[:n]

        # TODO: decode with RAE
        # images = rae.decode(latents)
        imgs = jnp.clip(latents, 0, 1)
        all_samples.append(np.array(imgs))

    all_samples = np.concatenate(all_samples, axis=0)

    # Save shard
    os.makedirs(args.output_dir, exist_ok=True)
    shard_path = os.path.join(args.output_dir, f"samples_{process_index:02d}.npz")
    np.savez(shard_path, arr_0=all_samples)

    if is_main:
        print(f"Saved {all_samples.shape[0]} samples to {shard_path}")
        print(f"Total across {num_devices} devices: {args.num_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="samples/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
