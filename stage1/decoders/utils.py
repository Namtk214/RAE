"""ViTMAEConfig, activation functions, sincos positional embedding — JAX port."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------
def _gelu(x):
    return jax.nn.gelu(x, approximate=False)

def _gelu_new(x):
    return jax.nn.gelu(x, approximate=True)

ACT2FN = {
    "gelu": _gelu,
    "gelu_new": _gelu_new,
    "relu": jax.nn.relu,
    "selu": jax.nn.selu,
    "silu": jax.nn.silu,
    "tanh": jnp.tanh,
}


# ---------------------------------------------------------------------------
# Config dataclass (replaces HuggingFace PretrainedConfig)
# ---------------------------------------------------------------------------
@dataclass
class ViTMAEConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    decoder_num_attention_heads: int = 16
    decoder_hidden_size: int = 512
    decoder_num_hidden_layers: int = 8
    decoder_intermediate_size: int = 2048
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False

    @classmethod
    def from_json(cls, path: str) -> "ViTMAEConfig":
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Sincos positional embeddings
# ---------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, add_cls_token: bool = False) -> np.ndarray:
    """Generate 2D sin-cos positional embeddings.

    Returns array of shape (grid_size*grid_size, embed_dim) or
    (1 + grid_size*grid_size, embed_dim) if add_cls_token.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, grid_size**2)

    emb = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if add_cls_token:
        cls_embed = np.zeros((1, embed_dim), dtype=np.float32)
        emb = np.concatenate([cls_embed, emb], axis=0)
    return emb


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Simple ModelOutput-like container
# ---------------------------------------------------------------------------
@dataclass
class ModelOutput:
    """Base dataclass for model outputs (replaces transformers.ModelOutput)."""
    pass
