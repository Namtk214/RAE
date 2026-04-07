"""MAE encoder (ViT-MAE, frozen) — JAX port using HuggingFace Flax model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import FlaxViTMAEModel, AutoConfig


class MAEwNorm(nnx.Module):
    """Frozen MAE encoder (ViT-MAE). Strips CLS token, applies LayerNorm without affine."""

    def __init__(self, model_name: str = "facebook/vit-mae-base", input_size: int = 224):
        self.model_name = model_name
        self.hf_config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = self.hf_config.hidden_size
        self.patch_size = self.hf_config.patch_size
        self.num_patches = (input_size // self.patch_size) ** 2
        self._model = FlaxViTMAEModel.from_pretrained(model_name)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, 3) in [0, 1] → (B, num_patches, hidden_size)."""
        outputs = self._model(pixel_values=x, deterministic=True)
        features = outputs.last_hidden_state[:, 1:]  # remove CLS token
        features = _layer_norm_no_affine(features)
        return jax.lax.stop_gradient(features)


def _layer_norm_no_affine(x, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)
