"""Eval utilities — LPIPS, VGG16, helper functions. JAX port.

Note: PSNR and SSIM are computed purely in JAX (no PyTorch dependency needed).
LPIPS and FID rely on PyTorch models at evaluation time, so they import torch
only when called (not at module load time).
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# -------------------------------------------------------------------
# Array conversion helpers
# -------------------------------------------------------------------
def to_jax_tensor(arr: np.ndarray) -> jnp.ndarray:
    """Convert uint8 (B, H, W, C) → float32 (B, C, H, W) in [0, 1]."""
    x = jnp.array(arr, dtype=jnp.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return jnp.transpose(x, (0, 3, 1, 2))


# -------------------------------------------------------------------
# PSNR — pure JAX
# -------------------------------------------------------------------
def calculate_psnr(arr1: np.ndarray, arr2: np.ndarray,
                   batch_size: int = 128, device: str = "cpu",
                   disable_bar: bool = True) -> float:
    """PSNR between two image arrays (B, H, W, C) uint8."""
    B = arr1.shape[0]
    psnr_sum = 0.0

    for i in range(0, B, batch_size):
        b1 = to_jax_tensor(arr1[i:i + batch_size])
        b2 = to_jax_tensor(arr2[i:i + batch_size])
        mse = jnp.mean((b1 - b2) ** 2, axis=(1, 2, 3))
        mse = jnp.clip(mse, a_min=1e-10)
        batch_psnr = 20.0 * jnp.log10(1.0 / jnp.sqrt(mse))
        psnr_sum += float(jnp.sum(batch_psnr))

    return psnr_sum / B


# -------------------------------------------------------------------
# SSIM — pure JAX (simplified, uses mean-based formula)
# -------------------------------------------------------------------
def _ssim_per_channel(img1, img2, C1=0.01 ** 2, C2=0.03 ** 2):
    """Compute SSIM between two images (B, C, H, W)."""
    mu1 = jnp.mean(img1, axis=(2, 3), keepdims=True)
    mu2 = jnp.mean(img2, axis=(2, 3), keepdims=True)
    sigma1_sq = jnp.var(img1, axis=(2, 3), keepdims=True)
    sigma2_sq = jnp.var(img2, axis=(2, 3), keepdims=True)
    sigma12 = jnp.mean((img1 - mu1) * (img2 - mu2), axis=(2, 3), keepdims=True)

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return jnp.mean(ssim_map, axis=(1, 2, 3))


def calculate_ssim(arr1: np.ndarray, arr2: np.ndarray,
                   batch_size: int = 128, device: str = "cpu",
                   disable_bar: bool = True) -> float:
    """SSIM between two image arrays (B, H, W, C) uint8."""
    B = arr1.shape[0]
    ssim_sum = 0.0

    for i in range(0, B, batch_size):
        b1 = to_jax_tensor(arr1[i:i + batch_size])
        b2 = to_jax_tensor(arr2[i:i + batch_size])
        batch_ssim = _ssim_per_channel(b1, b2)
        ssim_sum += float(jnp.sum(batch_ssim))

    return ssim_sum / B


# -------------------------------------------------------------------
# LPIPS — delegates to PyTorch (lazy import)
# -------------------------------------------------------------------
def calculate_lpips(arr1: np.ndarray, arr2: np.ndarray,
                    batch_size: int = 128, device: str = "cpu",
                    disable_bar: bool = True) -> float:
    """LPIPS via pytorch (uses VGG16 pretrained weights)."""
    import torch
    from disc.lpips import LPIPS as LPIPS_Module

    dev = torch.device(device)
    B = arr1.shape[0]
    lpips_model = LPIPS_Module().eval()

    lpips_vals = []
    for i in range(0, B, batch_size):
        b1 = _np_to_torch(arr1[i:i + batch_size]).to(dev)
        b2 = _np_to_torch(arr2[i:i + batch_size]).to(dev)
        # LPIPS expects [-1, 1]
        b1 = (b1 - 0.5) * 2.0
        b2 = (b2 - 0.5) * 2.0
        with torch.no_grad():
            val = lpips_model(b1, b2).squeeze()
        lpips_vals.append(val.cpu().numpy())

    return float(np.concatenate(lpips_vals).mean())


def _np_to_torch(arr: np.ndarray):
    """(B, H, W, C) uint8 → (B, C, H, W) float32 torch tensor."""
    import torch
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    if x.max() > 1.5:
        x = x / 255.0
    return x
