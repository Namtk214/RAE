"""Evaluation package — metrics and distributed eval across TPU cores.

Follows shortcut-models approach:
- evaluate_generation_distributed: each process generates samples, extracts
  InceptionV3 activations (JAX-native, on-device), gathers 2048-d activation
  vectors via process_allgather, host-0 computes µ/Σ and FID.
- evaluate_reconstruction_distributed: encode+decode across all devices.
- fid_from_stats uses eye*1e-6 regularization to avoid sqrtm instability.
"""

from .utils import calculate_psnr, calculate_ssim, calculate_lpips
from .fid import calculate_rfid, calculate_gfid, fid_from_stats, moments_from_activations, get_fid_network

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
    """Distributed generation evaluation — shortcut-models style.

    Each process generates samples and extracts InceptionV3 activations
    on-device (JAX-native). Then activations (2048-d float32) are gathered
    via process_allgather — much cheaper than gathering full images as NPZ.
    Host-0 computes µ/Σ and FID using fid_from_stats (with regularization).

    Args:
        model: DiT model (EMA weights already loaded)
        ema_state: EMA weights for sampling
        sample_fn: ODE/SDE sampler function
        latent_size: (H, W, C) latent shape — NHWC convention
        num_classes: number of classes
        null_label: label for unconditional (usually num_classes)
        use_guidance: whether to use CFG
        guidance_scale: CFG scale
        num_samples: total samples across all processes
        batch_size: per-device batch size
        experiment_dir: path for temp files (not used for image exchange anymore)
        global_step: current training step (for logging)
        reference_npz_path: path to reference NPZ with keys 'mu' and 'sigma'
        rae_decode_fn: optional RAE decode function (latents → NHWC [0,1] images)
        mesh: JAX mesh for sharding
    """
    from flax import nnx
    from .fid import get_fid_network, preprocess_for_inception, fid_from_stats, moments_from_activations

    process_index = jax.process_index()
    num_processes = jax.process_count()
    is_main = process_index == 0

    # Setup sharding
    if mesh is not None:
        data_sharding = NamedSharding(mesh, P("data"))
    else:
        data_sharding = None

    if is_main:
        print(f"\n[Eval] Starting distributed FID evaluation at step {global_step}")
        print(f"[Eval] Generating {num_samples} samples, gathering InceptionV3 activations...")

    # ── Build JAX InceptionV3 (same as shortcut-models) ──
    # Build once, JIT-compiled — runs fully on TPU/GPU in JAX
    fid_fn = get_fid_network()

    @jax.jit
    def inception_batch(images_nhwc_01):
        """Resize + scale to [-1,1] + extract 2048-d features in JAX."""
        # images_nhwc_01: NHWC float32 in [0, 1]
        x = images_nhwc_01 * 2.0 - 1.0          # → [-1, 1]
        x = jax.image.resize(
            x, (x.shape[0], 299, 299, x.shape[3]),
            method="bilinear", antialias=False,
        )
        x = jnp.clip(x, -1.0, 1.0)
        acts = fid_fn(x)[..., 0, 0, :]           # [B, 2048]
        return acts

    # Build compiled sampling functions
    if use_guidance:
        @jax.jit
        def compiled_sample_fn(z, y):
            z_cfg = jnp.concatenate([z, z], axis=0)
            y_null = jnp.full((z.shape[0],), null_label, dtype=jnp.int32)
            y_cfg = jnp.concatenate([y, y_null], axis=0)
            from functools import partial
            model_fn_cfg = partial(model.forward_with_cfg, cfg_scale=guidance_scale)
            samples = sample_fn(z_cfg, model_fn_cfg, y=y_cfg)
            if isinstance(samples, (list, tuple)):
                samples = samples[-1]
            return samples[:z.shape[0]]
    else:
        @jax.jit
        def compiled_sample_fn(z, y):
            model_fn = lambda x, t, y=None, **kw: model(x, t, y, training=False)  # noqa: E731
            samples = sample_fn(z, model_fn, y=y)
            if isinstance(samples, (list, tuple)):
                samples = samples[-1]
            return samples

    # ── Per-process shard ──
    chunk = num_samples // num_processes
    start = process_index * chunk
    end = (process_index + 1) * chunk if process_index < num_processes - 1 else num_samples
    local_num = end - start

    rng = jax.random.PRNGKey(global_step * num_processes + process_index)
    generated = 0
    num_local_devices = jax.local_device_count()

    # Collect activations (float32 [N, 2048]) — much smaller than images
    local_activations = []

    from tqdm import tqdm as _tqdm
    pbar = _tqdm(total=local_num, desc=f"Process {process_index} Gen+Inception") if is_main else None

    while generated < local_num:
        n_raw = min(batch_size, local_num - generated)
        rng, noise_rng, label_rng = jax.random.split(rng, 3)

        # Always full batch_size to keep JIT shape stable
        z = jax.random.normal(noise_rng, (batch_size, *latent_size))
        y = jax.random.randint(label_rng, (batch_size,), 0, num_classes)

        if data_sharding is not None and batch_size % num_local_devices == 0:
            z = jax.device_put(z, data_sharding)
            y = jax.device_put(y, data_sharding)

        # Sample latents
        samples = compiled_sample_fn(z, y)
        samples = samples[:n_raw]

        # Decode latents → images [0, 1] NHWC
        if rae_decode_fn is not None:
            images = rae_decode_fn(samples)
            images = jnp.clip(images, 0.0, 1.0)
        else:
            images = jnp.clip(samples, 0.0, 1.0)

        # Ensure NHWC (decode might return NCHW)
        if images.ndim == 4 and images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
            images = jnp.transpose(images, (0, 2, 3, 1))

        # Extract InceptionV3 activations ON-DEVICE (JAX JIT)
        acts = inception_batch(images)  # [n_raw, 2048]
        local_activations.append(np.array(acts))

        generated += n_raw
        if pbar is not None:
            pbar.update(n_raw)

    if pbar is not None:
        pbar.close()

    # Stack local activations: [local_num, 2048]
    local_acts = np.concatenate(local_activations, axis=0)  # [local_num, 2048]
    if is_main:
        print(f"[Process {process_index}] Collected {local_acts.shape[0]} activations, shape {local_acts.shape}")

    # ── Gather activations across all processes (shortcut-models style) ──
    # process_allgather gathers along a new leading axis → [num_processes, local_num, 2048]
    # then reshape to [total_samples, 2048]
    try:
        all_acts = jax.experimental.multihost_utils.process_allgather(
            jnp.array(local_acts)
        )
        all_acts = np.array(all_acts)
        # Shape: [num_processes, local_num, 2048] → [num_processes * local_num, 2048]
        if all_acts.ndim == 3:
            all_acts = all_acts.reshape(-1, all_acts.shape[-1])
        all_acts = all_acts[:num_samples]  # trim to exact num_samples
    except Exception as e:
        # Single-host fallback: local_acts is already all we have
        if is_main:
            print(f"[Eval] process_allgather not available ({e}), using local activations only.")
        all_acts = local_acts

    # ── Host 0 computes FID ──
    metrics = None
    if is_main:
        print(f"[Eval] Combined activations shape: {all_acts.shape}")

        if reference_npz_path and os.path.exists(reference_npz_path):
            ref = np.load(reference_npz_path)
            mu_ref, sigma_ref = ref["mu"], ref["sigma"]
            mu_gen, sigma_gen = moments_from_activations(all_acts)
            fid_score = fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
            metrics = {"fid": fid_score}
            print(f"[Eval] Step {global_step} — FID: {fid_score:.4f}")
        else:
            print(f"[Eval] No reference NPZ found at {reference_npz_path}")
            metrics = {"fid": -1.0}

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
