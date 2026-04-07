"""DINOv2-with-registers encoder (frozen) — JAX port using HuggingFace Flax model."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import FlaxAutoModel, AutoConfig


class Dinov2withNorm(nnx.Module):
    """Frozen DINOv2-with-registers encoder.

    Wraps HuggingFace FlaxDinov2Model, strips CLS + register tokens,
    applies optional LayerNorm (without affine), and stops gradient.

    Output: (B, num_patches, hidden_size) where num_patches = (input_size // patch_size)^2
    For DINOv2-B with 224 input: (B, 256, 768)
    """

    def __init__(
        self,
        dinov2_path: str = "facebook/dinov2-with-registers-base",
        normalize: bool = True,
        input_size: int = 224,
    ):
        self.dinov2_path = dinov2_path
        self.normalize = normalize
        self.input_size = input_size

        # Load HuggingFace config to get arch info
        self.hf_config = AutoConfig.from_pretrained(dinov2_path)
        self.hidden_size = self.hf_config.hidden_size
        self.patch_size = self.hf_config.patch_size
        self.num_register_tokens = getattr(self.hf_config, "num_register_tokens", 0)
        self.num_patches = (input_size // self.patch_size) ** 2

        # Load the actual Flax model (frozen)
        self._model = FlaxAutoModel.from_pretrained(dinov2_path)

        # ImageNet normalization for DINOv2
        self.pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        self.pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode images to latent patches.

        Args:
            x: Images in range [0, 1], shape (B, H, W, 3) — NHWC format for JAX.

        Returns:
            Latent tokens (B, num_patches, hidden_size), gradient stopped.
        """
        B, H, W, C = x.shape

        # Resize to encoder_input_size if needed
        if H != self.input_size or W != self.input_size:
            x = jax.image.resize(x, (B, self.input_size, self.input_size, C), method="bicubic")

        # Normalize with ImageNet mean/std
        x = (x - self.pixel_mean) / self.pixel_std

        # HuggingFace Flax DINOv2 expects NCHW
        x_nchw = x.transpose(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        outputs = self._model(pixel_values=x_nchw, train=False)
        last_hidden = outputs.last_hidden_state  # (B, 1 + num_register + num_patches, hidden_size)

        # Strip CLS token (1) and register tokens
        num_skip = 1 + self.num_register_tokens
        z = last_hidden[:, num_skip:, :]  # (B, num_patches, hidden_size)

        # Optional LayerNorm without affine (as in PyTorch Dinov2withNorm)
        if self.normalize:
            z = _layer_norm_no_affine(z)

        # Frozen — no gradients
        z = jax.lax.stop_gradient(z)
        return z


def _layer_norm_no_affine(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """LayerNorm without learnable scale/bias (elementwise_affine=False)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)
