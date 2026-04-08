"""Model utilities for DiT — RoPE, RMSNorm, SwiGLU, embeddings. JAX/Flax NNX port."""

from __future__ import annotations

import math
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# ---------------------------------------------------------------------------
# Modulation / Gating helpers
# ---------------------------------------------------------------------------
def ddt_modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Per-segment modulation: x * (1 + scale) + shift."""
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx != L:
        repeat = Lx // L
        shift = jnp.repeat(shift, repeat, axis=1)
        scale = jnp.repeat(scale, repeat, axis=1)
    return x * (1 + scale) + shift


def ddt_gate(x: jnp.ndarray, gate: jnp.ndarray) -> jnp.ndarray:
    """Per-segment gating: x * gate."""
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx != L:
        repeat = Lx // L
        gate = jnp.repeat(gate, repeat, axis=1)
    return x * gate


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim, dtype=dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_float = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_float ** 2, axis=-1, keepdims=True) + self.eps)
        normed = x_float / rms
        return (normed * self.weight.value).astype(x.dtype)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------
class SwiGLUFFN(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = 0,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        out_features = out_features or in_features
        self.w12 = nnx.Linear(in_features, 2 * hidden_features, use_bias=True, dtype=dtype, rngs=rngs)
        self.w3 = nnx.Linear(hidden_features, out_features, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = jax.nn.silu(x1) * x2
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# rotary positional embedding (RoPE) — "VisionRotaryEmbeddingFast"
# ---------------------------------------------------------------------------
def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate pairs: (..., d) -> reshape to (..., d//2, 2) -> (-x2, x1) -> reshape back."""
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return jnp.stack((-x2, x1), axis=-1).reshape(*x1.shape[:-1], d)


def build_rope_freqs(dim: int, pt_seq_len: int = 16, ft_seq_len: int = None, theta: float = 10000.0):
    """Pre-compute 2D RoPE cos/sin frequency tables.

    Returns:
        freqs_cos, freqs_sin: arrays of shape (seq_len^2, dim)
    """
    if ft_seq_len is None:
        ft_seq_len = pt_seq_len

    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[:dim // 2].astype(np.float32) / dim))
    t = np.arange(ft_seq_len, dtype=np.float32) / ft_seq_len * pt_seq_len

    # 1D freqs for each axis
    f = np.outer(t, freqs)  # (seq_len, dim//2)
    f = np.repeat(f, 2, axis=-1)  # (seq_len, dim)

    # 2D: combine h and w via concatenation
    freqs_h = f[:, np.newaxis, :]  # (H, 1, dim)
    freqs_w = f[np.newaxis, :, :]  # (1, W, dim)

    # Broadcast and concatenate
    H, W = ft_seq_len, ft_seq_len
    freqs_2d = np.concatenate([
        np.broadcast_to(freqs_h, (H, W, f.shape[-1])),
        np.broadcast_to(freqs_w, (H, W, f.shape[-1])),
    ], axis=-1)

    freqs_2d = freqs_2d.reshape(H * W, -1)

    return jnp.array(np.cos(freqs_2d), dtype=jnp.float32), jnp.array(np.sin(freqs_2d), dtype=jnp.float32)


def apply_rope(x: jnp.ndarray, freqs_cos: jnp.ndarray, freqs_sin: jnp.ndarray) -> jnp.ndarray:
    """Apply RoPE to tensor x of shape (B, num_heads, L, dim).

    freqs_cos, freqs_sin: (L, dim)
    """
    _, _, Lt, _ = x.shape
    L, _ = freqs_cos.shape
    repeat_factor = Lt // L
    if repeat_factor != 1:
        freqs_cos = jnp.repeat(freqs_cos, repeat_factor, axis=0)
        freqs_sin = jnp.repeat(freqs_sin, repeat_factor, axis=0)
    return x * freqs_cos + _rotate_half(x) * freqs_sin


# ---------------------------------------------------------------------------
# NormAttention
# ---------------------------------------------------------------------------
class NormAttention(nnx.Module):
    """Attention with optional QK-Norm and RoPE support."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, dtype=dtype, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, dtype=dtype, rngs=rngs)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, rngs=rngs, dtype=dtype) if use_rmsnorm else nnx.LayerNorm(self.head_dim, dtype=dtype, rngs=rngs)
            self.k_norm = RMSNorm(self.head_dim, rngs=rngs, dtype=dtype) if use_rmsnorm else nnx.LayerNorm(self.head_dim, dtype=dtype, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: jnp.ndarray, rope=None) -> jnp.ndarray:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None:
            freqs_cos, freqs_sin = rope
            q = apply_rope(q, freqs_cos, freqs_sin)
            k = apply_rope(k, freqs_cos, freqs_sin)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = jax.nn.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# GaussianFourierEmbedding (for timesteps)
# ---------------------------------------------------------------------------
class GaussianFourierEmbedding(nnx.Module):
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.W = nnx.Param(jax.random.normal(rngs.params(), (embedding_size,)) * scale)
        self.mlp_0 = nnx.Linear(embedding_size * 2, hidden_size, use_bias=True, dtype=dtype, rngs=rngs)
        self.mlp_1 = nnx.Linear(hidden_size, hidden_size, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        W = jax.lax.stop_gradient(self.W.value)
        t_proj = t[:, None] * W[None, :] * 2 * jnp.pi
        t_embed = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)
        h = jax.nn.silu(self.mlp_0(t_embed))
        return self.mlp_1(h)


# ---------------------------------------------------------------------------
# LabelEmbedder (class labels with dropout for CFG)
# ---------------------------------------------------------------------------
class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        use_cfg = dropout_prob > 0
        self.embedding_table = nnx.Embed(num_classes + int(use_cfg), hidden_size, dtype=dtype, rngs=rngs)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        drop_ids = jax.random.uniform(rng, (labels.shape[0],)) < self.dropout_prob
        return jnp.where(drop_ids, self.num_classes, labels)

    def __call__(self, labels: jnp.ndarray, training: bool = False,
                 rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        if training and self.dropout_prob > 0 and rng is not None:
            labels = self.token_drop(labels, rng)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# Sincos 2D positional embeddings (reused from model_utils.py)
# ---------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """2D sincos positional embedding. Returns (grid_size*grid_size, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)

    # Each axis gets embed_dim//2 of the total embedding
    # We use embed_dim//4 frequencies per axis (sin+cos doubles → embed_dim//2)
    half = embed_dim // 4
    omega = np.arange(half, dtype=np.float64) / half
    omega = 1.0 / (10000 ** omega)

    out_h = np.outer(grid[1], omega)  # (N, embed_dim//4)
    out_w = np.outer(grid[0], omega)  # (N, embed_dim//4)

    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1).astype(np.float32)  # (N, embed_dim//2)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1).astype(np.float32)  # (N, embed_dim//2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (N, embed_dim)
