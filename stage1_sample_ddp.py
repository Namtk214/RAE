"""Distributed stage-1 reconstruction — encode + decode across all TPU cores. JAX port."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from PIL import Image

from utils.device_utils import setup_mesh
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from data import build_dataloader


def main(args):
    mesh = setup_mesh()
    num_devices = jax.device_count()
    process_index = jax.process_index()
    is_main = process_index == 0

    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("Config must have a stage_1 section.")

    rngs = nnx.Rngs(params=0)
    # rae = instantiate_from_config(rae_config, rngs=rngs)

    global_batch_size = args.batch_size * num_devices
    ds, steps = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=global_batch_size,
        dataset_type="imagefolder",
        split="train",
    )

    output_dir = args.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving reconstructions to {output_dir}")

    all_recons = []
    processed = 0
    max_samples = args.num_samples or float("inf")

    for step_data in ds:
        if processed >= max_samples:
            break

        images = step_data["image"]  # (B, H, W, C) float32
        # latents = rae.encode(images)
        # recons = rae.decode(latents)
        recons = images  # placeholder

        recons = jnp.clip(recons, 0.0, 1.0)
        recons_np = np.array(recons)

        # Convert and save
        if recons_np.shape[-1] != 3 and recons_np.shape[1] == 3:
            recons_np = np.transpose(recons_np, (0, 2, 3, 1))  # NCHW → NHWC
        recons_uint8 = (recons_np * 255).astype(np.uint8)

        for i, img in enumerate(recons_uint8):
            idx = processed + i
            if idx >= max_samples:
                break
            if is_main:
                Image.fromarray(img).save(os.path.join(output_dir, f"{idx:06d}.png"))

        processed += recons_uint8.shape[0]

        if is_main and processed % (global_batch_size * 10) == 0:
            print(f"[progress] {processed} images reconstructed")
            all_recons.append(recons_uint8)

    # Pack into NPZ
    if is_main and all_recons:
        combined = np.concatenate(all_recons, axis=0)
        npz_path = os.path.join(output_dir, "reconstructions.npz")
        np.savez(npz_path, arr_0=combined[:int(max_samples)])
        print(f"Saved {combined.shape[0]} reconstructions to {npz_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="samples/recon")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    main(args)
