"""Distributed latent mean/var estimation for stage-1 RAE — JAX port.

Computes running mean and variance of latents produced by the frozen RAE encoder
over the training dataset. Output is saved as a .npz file with 'mean' and 'var'
keys, compatible with RAE._load_stats() (which also reads PyTorch .pt files).

Uses Welford's online algorithm for numerical stability.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from stage1 import RAE
from utils.device_utils import create_mesh, get_data_sharding, get_replicated_sharding, shard_batch
from utils.train_utils import parse_configs
from data import build_dataloader


def _local_array(x):
    """Collect addressable (local) shards of a global jax.Array into numpy.

    On single-host this is equivalent to np.array(x).
    On multi-host it avoids the error from accessing non-local shards.
    """
    local_shards = [s.data for s in x.addressable_shards]
    return np.concatenate([np.asarray(s) for s in local_shards], axis=0)


def _welford_update(
    count: int,
    mean: np.ndarray,
    M2: np.ndarray,
    batch: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Online Welford algorithm for numerically-stable mean/variance.

    Args:
        count: current sample count
        mean:  running mean, shape = latent_shape
        M2:    running sum of squared deviations, same shape
        batch: new data (B, ...) — numpy array

    Returns:
        Updated (count, mean, M2)
    """
    for x in batch:
        count += 1
        delta = x - mean
        mean = mean + delta / count
        delta2 = x - mean
        M2 = M2 + delta * delta2
    return count, mean, M2


def main():
    parser = argparse.ArgumentParser(
        description="Compute RAE latent statistics for stage-2 normalization"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Stage-1 config YAML (stage_1 section required)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset (TFDS data_dir or ImageFolder root)")
    parser.add_argument("--output-dir", type=str, default="models/stats/dinov2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-device batch size")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max samples to process (None = full dataset)")
    parser.add_argument("--dataset-type", type=str, default="tfds",
                        choices=["imagefolder", "tfds"])
    parser.add_argument("--tfds-name", type=str, default="celebahq256")
    parser.add_argument("--tfds-builder-dir", type=str, default=None)
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────
    mesh = create_mesh()
    data_sharding = get_data_sharding(mesh)
    num_devices = jax.device_count()
    is_main = jax.process_index() == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[init] {num_devices} devices | output → {args.output_dir}")

    # ── Load config & build RAE ─────────────────────────────────────
    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("Config must have a 'stage_1' section.")

    rngs = nnx.Rngs(params=0, dropout=1)
    rae_params = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    # NOTE: do NOT pass normalization_stat_path here — we're computing the stats
    rae_params.pop("normalization_stat_path", None)
    rae = RAE(**rae_params, rngs=rngs)

    if is_main:
        print("[init] RAE encoder loaded (frozen DINOv2).")

    # JIT-compile encode for speed
    @jax.jit
    def encode(images):
        return jax.lax.stop_gradient(
            rae.encode(images, rng=None, training=False)
        )

    # ── Data ───────────────────────────────────────────────────────
    global_batch_size = args.batch_size * num_devices
    num_processes = jax.process_count()
    per_host_batch = global_batch_size // num_processes
    ds, steps = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=per_host_batch,
        dataset_type=args.dataset_type,
        split="train",
        tfds_name=args.tfds_name,
        tfds_builder_dir=args.tfds_builder_dir,
    )

    if is_main:
        print(f"[data] {steps} steps/epoch, global_batch={global_batch_size}")

    # ── Probe latent shape (use replicated sharding to avoid divisibility issue) ──
    repl_sharding = get_replicated_sharding(mesh)
    first_batch = next(iter(ds))
    # Use shard_batch for multi-host correct sharding
    probe_sharded = shard_batch(first_batch, mesh)
    probe_latent = _local_array(encode(probe_sharded["image"]))
    latent_shape = probe_latent.shape[1:]  # (H, W, C) or (N, C)
    if is_main:
        print(f"[probe] latent shape per sample: {latent_shape}")

    # ── Welford running stats ──────────────────────────────────────
    count = 0
    mean = np.zeros(latent_shape, dtype=np.float64)
    M2 = np.zeros(latent_shape, dtype=np.float64)

    processed = 0
    max_samples = args.num_samples or float("inf")

    with mesh:
        for step_data in ds:
            if processed >= max_samples:
                break

            sharded = shard_batch(step_data, mesh)
            latents = _local_array(encode(sharded["image"]))  # (local_B, H, W, C) float32

            count, mean, M2 = _welford_update(count, mean, M2, latents)
            processed += latents.shape[0]

            if is_main and processed % (global_batch_size * 20) == 0:
                print(f"[progress] {processed} / {int(max_samples) if max_samples != float('inf') else steps * global_batch_size} samples")

    if count < 2:
        raise RuntimeError("Need at least 2 samples to compute variance.")

    var = M2 / count  # population variance

    if is_main:
        out_path = os.path.join(args.output_dir, "normalization_stats.npz")
        np.savez(out_path, mean=mean.astype(np.float32), var=var.astype(np.float32))
        print(f"[done] saved → {out_path}")
        print(f"  mean shape: {mean.shape}, mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"  var  shape: {var.shape},  var  range: [{var.min():.4f},  {var.max():.4f}]")


if __name__ == "__main__":
    main()
