"""Distributed stage-1 reconstruction — encode + decode across all TPU cores. JAX port."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from PIL import Image

from stage1 import RAE
from utils.device_utils import create_mesh, get_data_sharding, shard_batch
from utils.train_utils import parse_configs
from utils.resume_utils import restore_checkpoint, build_checkpoint_manager
from data import build_dataloader


def main(args):
    mesh = create_mesh()
    data_sharding = get_data_sharding(mesh)
    num_devices = jax.device_count()
    is_main = jax.process_index() == 0

    # ── Config & RAE ─────────────────────────────────────────────
    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("Config must have a stage_1 section.")

    rngs = nnx.Rngs(params=0, dropout=1)
    rae_params = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    rae_params["noise_tau"] = 0.0  # inference: no noise
    rae = RAE(**rae_params, rngs=rngs)

    # ── Load checkpoint ───────────────────────────────────────────
    if args.ckpt_dir is not None:
        mngr = build_checkpoint_manager(args.ckpt_dir)
        ckpt, step = restore_checkpoint(mngr)
        if ckpt is None:
            if is_main:
                print(f"[Warning] No checkpoint found in {args.ckpt_dir}, using random weights.")
        else:
            key = "ema" if args.use_ema else "model"
            decoder_state = ckpt[key]
            graphdef, _ = nnx.split(rae.decoder)
            loaded_decoder = nnx.merge(graphdef, decoder_state)
            nnx.update(rae.decoder, nnx.state(loaded_decoder))
            if is_main:
                print(f"[ckpt] loaded {'EMA' if args.use_ema else 'model'} from step {step}")

    # JIT encode+decode
    @jax.jit
    def reconstruct(images):
        latents = jax.lax.stop_gradient(rae.encode(images, rng=None, training=False))
        recons = rae.decode(latents)
        return jnp.clip(recons, 0.0, 1.0)

    # ── Data ──────────────────────────────────────────────────────
    global_batch_size = args.batch_size * num_devices
    ds, steps = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=global_batch_size,
        dataset_type=args.dataset_type,
        split="train",
        tfds_name=args.tfds_name,
        tfds_builder_dir=args.tfds_builder_dir,
    )

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving reconstructions to {args.output_dir}")

    all_recons = []
    processed = 0
    max_samples = args.num_samples or float("inf")

    with mesh:
        for step_data in ds:
            if processed >= max_samples:
                break

            images = jax.device_put(jnp.asarray(step_data["image"]), data_sharding)
            recons = reconstruct(images)

            recons_np = np.array(recons)
            # recons may be NCHW → convert to NHWC
            if recons_np.ndim == 4 and recons_np.shape[1] in (1, 3, 4):
                recons_np = np.transpose(recons_np, (0, 2, 3, 1))
            recons_uint8 = (recons_np * 255).astype(np.uint8)

            for i, img in enumerate(recons_uint8):
                idx = processed + i
                if idx >= max_samples:
                    break
                if is_main:
                    Image.fromarray(img).save(os.path.join(args.output_dir, f"{idx:06d}.png"))

            all_recons.append(recons_uint8)
            processed += recons_uint8.shape[0]

            if is_main and processed % (global_batch_size * 10) == 0:
                print(f"[progress] {processed} images reconstructed")

    # Save all as NPZ
    if is_main and all_recons:
        combined = np.concatenate(all_recons, axis=0)
        if max_samples != float("inf"):
            combined = combined[:int(max_samples)]
        npz_path = os.path.join(args.output_dir, "reconstructions.npz")
        np.savez(npz_path, arr_0=combined)
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
    parser.add_argument("--dataset-type", type=str, default="tfds",
                        choices=["imagefolder", "tfds"])
    parser.add_argument("--tfds-name", type=str, default="celebahq256")
    parser.add_argument("--tfds-builder-dir", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoint directory. Auto-resumes latest.")
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()
    main(args)
