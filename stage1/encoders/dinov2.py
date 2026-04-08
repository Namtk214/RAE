"""DINOv2-base encoder (frozen) — JAX/Flax via HuggingFace FlaxDinov2Model."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoConfig, FlaxDinov2Model


class Dinov2withNorm(nnx.Module):
    """Frozen DINOv2-base encoder.

    Uses FlaxDinov2Model (facebook/dinov2-base), strips CLS token,
    applies optional LayerNorm (no affine), stops gradient.

    Output: (B, num_patches, hidden_size)
    For DINOv2-B with 224 input: (B, 256, 768)
    """

    def __init__(
        self,
        dinov2_path: str = "facebook/dinov2-base",
        normalize: bool = True,
        input_size: int = 224,
    ):
        self.normalize = normalize
        self.input_size = input_size

        cfg = AutoConfig.from_pretrained(dinov2_path)
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)
        self.num_patches = (input_size // self.patch_size) ** 2

        self._model = FlaxDinov2Model.from_pretrained(dinov2_path)

        # ImageNet normalization
        self.pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        self.pixel_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode images → latent patches.

        Args:
            x: (B, H, W, 3) float32 in [0, 1] — NHWC.
        Returns:
            (B, num_patches, hidden_size), gradient stopped.
        """
        B, H, W, C = x.shape

        if H != self.input_size or W != self.input_size:
            x = jax.image.resize(x, (B, self.input_size, self.input_size, C), method="bicubic")

        x = (x - self.pixel_mean) / self.pixel_std
        x_nchw = x.transpose(0, 3, 1, 2)               # NHWC → NCHW

        outputs = self._model(pixel_values=x_nchw, train=False)
        last_hidden = outputs.last_hidden_state          # (B, 1+num_patches, hidden)

        # Strip CLS token (and register tokens if any)
        num_skip = 1 + self.num_register_tokens
        z = last_hidden[:, num_skip:, :]                 # (B, num_patches, hidden)

        if self.normalize:
            z = _layer_norm_no_affine(z)

        return jax.lax.stop_gradient(z)


def _layer_norm_no_affine(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)
