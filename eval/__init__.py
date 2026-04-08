"""Evaluation package — metrics and distributed eval across TPU cores.

Port of PyTorch distributed evaluation. Supports:
- evaluate_reconstruction_distributed: encode+decode across all devices, gather on host 0
- evaluate_generation_distributed: sample across all devices, gather on host 0
- Both use file-based NPZ shard exchange (same pattern as PyTorch DDP)
"""

from .utils import calculate_psnr, calculate_ssim, calculate_lpips
from .fid import calculate_rfid, calculate_gfid

import os
import sys
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# Non-distributed metrics
# ---------------------------------------------------------------------------
def compute_reconstruction_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    batch_size: int = 128,
    metrics_to_compute=("psnr", "ssim", "rfid"),
) -> Dict[str, float]:
    """Compute reconstruction-quality metrics between reference and reconstructed images."""
    results = {}
    if "psnr" in metrics_to_compute:
        results["psnr"] = calculate_psnr(ref_arr, rec_arr, batch_size)
    if "ssim" in metrics_to_compute:
        results["ssim"] = calculate_ssim(ref_arr, rec_arr, batch_size)
    if "rfid" in metrics_to_compute:
        results["rfid"] = calculate_rfid(ref_arr, rec_arr, batch_size)
    return results


def compute_generation_metrics(
    ref_arr,
    gen_arr: np.ndarray,
    batch_size: int = 128,
    device: str = "tpu",
) -> Dict[str, float]:
    """Compute generation metrics (FID)."""
    fid = calculate_gfid(gen_arr, ref_arr, batch_size, device=device)
    return {"fid": fid}


# ---------------------------------------------------------------------------
# Distributed generation evaluation (port of PyTorch evaluate_generation_distributed)
# ---------------------------------------------------------------------------
def evaluate_generation_distributed(
    model,
    ema_state,
    sample_fn,
    latent_size: tuple,
    num_classes: int,
    null_label: int,
    use_guidance: bool,
    guidance_scale: float,
    num_samples: int,
    batch_size: int,
    experiment_dir: str,
    global_step: int,
    reference_npz_path: Optional[str] = None,
    rae_decode_fn=None,
    mesh=None,
    device: str = "tpu",
) -> Optional[Dict[str, float]]:
    """Distributed generation evaluation — each process generates its shard, host 0 computes metrics.

    Strategy (mirrors PyTorch DDP):
    1. Each process generates `chunk = num_samples // num_processes` samples
    2. Each process saves its shard as NPZ to shared filesystem
    3. Host 0 combines all shards and computes FID

    Args:
        model: DiT model
        ema_state: EMA weights to use for sampling
        sample_fn: ODE/SDE sampler function
        latent_size: (C, H, W) latent shape
        num_classes: number of classes
        null_label: label for unconditional (usually num_classes)
        use_guidance: whether to use CFG
        guidance_scale: CFG scale
        num_samples: total samples to generate across all processes
        batch_size: per-device batch size
        experiment_dir: path for temporary NPZ shards
        global_step: current training step
        reference_npz_path: path to reference NPZ for FID
        rae_decode_fn: optional RAE decode function (latents → images)
        mesh: JAX mesh for sharding
    """
    from flax import nnx

    process_index = jax.process_index()
    num_processes = jax.process_count()
    is_main = process_index == 0

    # Setup data sharding so all cores participate during sampling
    if mesh is not None:
        data_sharding = NamedSharding(mesh, P("data"))
    else:
        data_sharding = None

    # Create temp directory on host 0
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if is_main:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"\n[Eval] Starting distributed generation at step {global_step}")
        print(f"🚀 [Chẩn đoán FID] Lát nữa FID sẽ được tính bằng backend: {device.upper()}")

    # Barrier: wait for directory creation
    # On multi-host TPU, use jax.experimental.multihost_utils
    try:
        jax.experimental.multihost_utils.sync_global_devices("eval_dir_create")
    except Exception:
        pass  # Single-host: no sync needed

    # Calculate per-process shard
    chunk = num_samples // num_processes
    if process_index < num_processes - 1:
        start, end = process_index * chunk, (process_index + 1) * chunk
    else:
        start, end = process_index * chunk, num_samples
    local_num = end - start

    # The model passed here is usually already configured with the correct weights (e.g. EMA)
    # in the main training loop, so we do not need to manually swap states here.
    # We simply use `model` as is.

    # Generate samples
    generations = []
    rng = jax.random.PRNGKey(global_step * num_processes + process_index)
    generated = 0

    num_local_devices = jax.local_device_count()

    from tqdm import tqdm
    pbar = tqdm(total=local_num, desc=f"Process {process_index} Generation") if is_main else None

    from flax import nnx

    # Pre-build compiled forward functions
    if use_guidance:
        @jax.jit
        def compiled_sample_fn(z, y):
            z_cfg = jnp.concatenate([z, z], axis=0)
            y_null = jnp.full((z.shape[0],), null_label, dtype=jnp.int32)
            y_cfg = jnp.concatenate([y, y_null], axis=0)
            model_fn_cfg = partial(model.forward_with_cfg, cfg_scale=guidance_scale)
            samples = sample_fn(z_cfg, model_fn_cfg, y=y_cfg)
            if isinstance(samples, (list, tuple)):
                samples = samples[-1]
            return samples[:z.shape[0]]
    else:
        @jax.jit
        def compiled_sample_fn(z, y):
            model_fn = lambda x, t, y=None, **kw: model(x, t, y, training=False)
            samples = sample_fn(z, model_fn, y=y)
            if isinstance(samples, (list, tuple)):
                samples = samples[-1]
            return samples

    while generated < local_num:
        n_raw = min(batch_size, local_num - generated)  # real samples this iter
        rng, noise_rng, label_rng = jax.random.split(rng, 3)

        # Always generate a FULL batch_size to keep compiled_sample_fn shape stable
        # (avoids JIT recompilation on the last smaller batch).
        z = jax.random.normal(noise_rng, (batch_size, *latent_size))
        y = jax.random.randint(label_rng, (batch_size,), 0, num_classes)

        # Shard z and y across all local devices so every core participates
        if data_sharding is not None and batch_size % num_local_devices == 0:
            z = jax.device_put(z, data_sharding)
            y = jax.device_put(y, data_sharding)

        samples = compiled_sample_fn(z, y)
        samples = samples[:n_raw]  # Discard padding, keep only real samples

        # Decode latents → images (if RAE provided)
        if rae_decode_fn is not None:
            images = rae_decode_fn(samples)
            images = jnp.clip(images, 0.0, 1.0)
        else:
            images = jnp.clip(samples, 0.0, 1.0)

        # Convert to uint8 NHWC
        imgs_np = np.array(images)
        if imgs_np.ndim == 4 and imgs_np.shape[1] in (1, 3, 4):  # NCHW → NHWC
            imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
        imgs_uint8 = (imgs_np * 255).astype(np.uint8)
        generations.append(imgs_uint8)
        generated += n_raw

        if pbar is not None:
            pbar.update(n_raw)

    if pbar is not None:
        pbar.close()

    if is_main:
        print(f"[Process {process_index}] Generated {generated} samples")

    generations = np.concatenate(generations, axis=0)

    # Save shard NPZ
    shard_path = os.path.join(temp_dir, f"gen_{global_step:07d}_{process_index:02d}.npz")
    np.savez(shard_path, arr_0=generations)

    # Restore original weights is skipped since we didn't swap anything (model is already cloned for eval).

    # Barrier: wait for all processes
    try:
        jax.experimental.multihost_utils.sync_global_devices("eval_gen_done")
    except Exception:
        pass

    # Host 0 computes metrics
    metrics = None
    if is_main:
        all_gens = []
        for r in range(num_processes):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            shard_data = np.load(shard_file)["arr_0"]
            all_gens.append(shard_data)

        combined = np.concatenate(all_gens, axis=0)[:num_samples]
        print(f"[Eval] Combined generation shape: {combined.shape}")

        if reference_npz_path and os.path.exists(reference_npz_path):
            ref_stats = np.load(reference_npz_path)
            metrics = compute_generation_metrics(ref_stats, combined, 128, device=device)
            print(f"[Eval] Step {global_step} — FID: {metrics.get('fid', -1):.4f}")
        else:
            print(f"[Eval] No reference NPZ found at {reference_npz_path}")
            metrics = {"fid": -1.0}

        # Cleanup
        for r in range(num_processes):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    # Final barrier
    try:
        jax.experimental.multihost_utils.sync_global_devices("eval_metrics_done")
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Distributed reconstruction evaluation (port of PyTorch evaluate_reconstruction_distributed)
# ---------------------------------------------------------------------------
def evaluate_reconstruction_distributed(
    model_fn,
    val_dataset,
    num_samples: int,
    batch_size: int,
    experiment_dir: str,
    global_step: int,
    reference_npz_path: Optional[str] = None,
    metrics_to_compute=("psnr", "ssim", "rfid"),
    mesh=None,
) -> Optional[Dict[str, float]]:
    """Distributed reconstruction evaluation — each process reconstructs its shard.

    Args:
        model_fn: callable (images: NHWC float32) → reconstructed images (NHWC float32)
        val_dataset: iterable yielding {"image": array, ...} batches
        num_samples: total samples to reconstruct
        batch_size: per-device batch size
        experiment_dir: path for temporary NPZ shards
        global_step: current training step
        reference_npz_path: path to reference NPZ for metrics
        metrics_to_compute: tuple of metric names
        mesh: JAX mesh
    """
    process_index = jax.process_index()
    num_processes = jax.process_count()
    is_main = process_index == 0

    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if is_main:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"\n[Eval] Starting distributed reconstruction at step {global_step}")

    try:
        jax.experimental.multihost_utils.sync_global_devices("eval_recon_dir")
    except Exception:
        pass

    # Calculate per-process shard
    chunk = num_samples // num_processes
    if process_index < num_processes - 1:
        start, end = process_index * chunk, (process_index + 1) * chunk
    else:
        start, end = process_index * chunk, num_samples
    local_num = end - start

    reconstructions = []
    processed = 0

    for step_data in val_dataset:
        if processed >= local_num:
            break

        images = step_data["image"]
        recon = model_fn(images)
        recon = jnp.clip(recon, 0.0, 1.0)

        recon_np = np.array(recon)
        if recon_np.ndim == 4 and recon_np.shape[1] in (1, 3, 4):
            recon_np = np.transpose(recon_np, (0, 2, 3, 1))
        recon_uint8 = (recon_np * 255).astype(np.uint8)

        for img in recon_uint8:
            if processed >= local_num:
                break
            reconstructions.append(img)
            processed += 1

    if is_main:
        print(f"[Process {process_index}] Reconstructed {processed} images")

    reconstructions = np.stack(reconstructions) if reconstructions else np.empty((0,))

    shard_path = os.path.join(temp_dir, f"recon_{global_step:07d}_{process_index:02d}.npz")
    np.savez(shard_path, arr_0=reconstructions)

    try:
        jax.experimental.multihost_utils.sync_global_devices("eval_recon_done")
    except Exception:
        pass

    metrics = None
    if is_main:
        all_recons = []
        for r in range(num_processes):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            shard_data = np.load(shard_file)["arr_0"]
            all_recons.append(shard_data)

        combined = np.concatenate(all_recons, axis=0)[:num_samples]
        print(f"[Eval] Combined reconstruction shape: {combined.shape}")

        if reference_npz_path and os.path.exists(reference_npz_path):
            ref_images = np.load(reference_npz_path)["arr_0"]
            metrics = compute_reconstruction_metrics(
                ref_images, combined, 128, metrics_to_compute
            )
            print(f"[Eval] Step {global_step} Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")
        else:
            print(f"[Eval] No reference NPZ at {reference_npz_path}")
            metrics = {}

        # Cleanup
        for r in range(num_processes):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    try:
        jax.experimental.multihost_utils.sync_global_devices("eval_recon_metrics_done")
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Simple (non-distributed) evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_reconstruction(
    model_fn,
    images: jnp.ndarray,
    batch_size: int = 32,
    metrics_to_compute=("psnr", "ssim"),
) -> Dict[str, float]:
    """Evaluate reconstruction quality on a batch of images (single-device)."""
    N = images.shape[0]
    all_recons = []

    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size]
        recon = model_fn(batch)
        recon = jnp.clip(recon, 0.0, 1.0)
        all_recons.append(np.array(recon))

    all_recons = np.concatenate(all_recons, axis=0)
    ref_uint8 = (np.array(images) * 255).astype(np.uint8)
    rec_uint8 = (all_recons * 255).astype(np.uint8)

    return compute_reconstruction_metrics(ref_uint8, rec_uint8, batch_size, metrics_to_compute)


def evaluate_generation(
    sample_fn,
    model_fn,
    rae_decode_fn,
    num_samples: int,
    latent_shape: tuple,
    rng: jax.random.PRNGKey,
    batch_size: int = 32,
    reference_stats=None,
    **sample_kwargs,
) -> Dict[str, float]:
    """Generate samples and compute FID (single-device)."""
    all_samples = []
    generated = 0

    while generated < num_samples:
        n = min(batch_size, num_samples - generated)
        rng, noise_rng = jax.random.split(rng)
        z = jax.random.normal(noise_rng, (n, *latent_shape))

        latents = sample_fn(z, model_fn, **sample_kwargs)
        if isinstance(latents, (list, tuple)):
            latents = latents[-1]

        images = rae_decode_fn(latents)
        images = jnp.clip(images, 0.0, 1.0)

        imgs_np = np.array(images)
        if imgs_np.shape[1] in (1, 3, 4):
            imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
        imgs_uint8 = (imgs_np * 255).astype(np.uint8)
        all_samples.append(imgs_uint8)
        generated += n

    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]

    if reference_stats is None:
        return {"fid": -1.0}

    return compute_generation_metrics(reference_stats, all_samples, batch_size)


# ---------------------------------------------------------------------------
# CLI for standalone metric computation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-img", type=str, default="samples/imagenet-256-val.npz")
    parser.add_argument("--rec-img", type=str, default="samples/sdvae-ft-mse-f8d4.npz")
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    ref_img = np.load(args.ref_img)["arr_0"]
    rec_img = np.load(args.rec_img)["arr_0"]
    print(f"Loaded images: ref: {ref_img.shape}, rec: {rec_img.shape}")

    psnr = calculate_psnr(ref_img, rec_img, args.bs)
    print(f"PSNR: {psnr:.6f}")
    lpips = calculate_lpips(ref_img, rec_img, args.bs)
    print(f"LPIPS: {lpips:.6f}")
    ssim_val = calculate_ssim(ref_img, rec_img, args.bs)
    print(f"SSIM: {ssim_val:.6f}")
    rfid = calculate_rfid(ref_img, rec_img, args.bs)
    print(f"rFID: {rfid:.6f}")
