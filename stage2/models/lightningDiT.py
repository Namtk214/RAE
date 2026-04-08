"""LightningDiT — single-stream DiT (no DDT head). JAX/Flax NNX port."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .model_utils import (
    RMSNorm, SwiGLUFFN, NormAttention,
    GaussianFourierEmbedding, LabelEmbedder,
    ddt_modulate, build_rope_freqs,
    get_2d_sincos_pos_embed,
)
from .DDT import PatchEmbed, _Mlp


class LightningDiTBlock(nnx.Module):
    """Single-stream DiT block with AdaLN modulation."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 use_qknorm=False, use_swiglu=True, use_rmsnorm=True, wo_shift=False,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.wo_shift = wo_shift

        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size, rngs=rngs, dtype=dtype)
            self.norm2 = RMSNorm(hidden_size, rngs=rngs, dtype=dtype)
        else:
            self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)
            self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)

        self.attn = NormAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, rngs=rngs, dtype=dtype)

        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden), rngs=rngs, dtype=dtype)
        else:
            self.mlp = _Mlp(hidden_size, mlp_hidden, rngs=rngs, dtype=dtype)

        num_mod = 4 if wo_shift else 6
        self.adaLN_mod_linear = nnx.Linear(hidden_size, num_mod * hidden_size, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x, c, feat_rope=None):
        mod = jax.nn.silu(c)
        mod = self.adaLN_mod_linear(mod)

        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(mod, 4, axis=-1)
            shift_msa = jnp.zeros_like(scale_msa)
            shift_mlp = jnp.zeros_like(scale_mlp)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)

        # Original LightningDiT uses unsqueeze(1) * gating, not DDTGate
        x = x + gate_msa[:, None, :] * self.attn(
            ddt_modulate(self.norm1(x), shift_msa[:, None, :], scale_msa[:, None, :]),
            rope=feat_rope
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            ddt_modulate(self.norm2(x), shift_mlp[:, None, :], scale_mlp[:, None, :])
        )
        return x


class LightningFinalLayer(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size, rngs=rngs, dtype=dtype)
        else:
            self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, patch_size * patch_size * out_channels, use_bias=True, dtype=dtype, rngs=rngs)
        self.adaLN_linear = nnx.Linear(hidden_size, 2 * hidden_size, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x, c):
        mod = jax.nn.silu(c)
        mod = self.adaLN_linear(mod)
        shift, scale = jnp.split(mod, 2, axis=-1)
        x = ddt_modulate(self.norm_final(x), shift[:, None, :], scale[:, None, :])
        return self.linear(x)


class LightningDiT(nnx.Module):
    """Single-stream Diffusion Transformer."""

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 768,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.use_rope = use_rope

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, rngs=rngs, dtype=dtype)
        self.t_embedder = GaussianFourierEmbedding(hidden_size, rngs=rngs, dtype=dtype)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, rngs=rngs, dtype=dtype)

        num_patches = self.x_embedder.num_patches
        pos = get_2d_sincos_pos_embed(hidden_size, int(math.sqrt(num_patches)))
        self.pos_embed = nnx.Param(jnp.array(pos[None, ...], dtype=dtype))

        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = build_rope_freqs(half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.blocks = [
            LightningDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm, use_swiglu=use_swiglu,
                              use_rmsnorm=use_rmsnorm, wo_shift=wo_shift,
                              rngs=rngs, dtype=dtype)
            for _ in range(depth)
        ]
        self.final_layer = LightningFinalLayer(hidden_size, patch_size, self.out_channels,
                                               use_rmsnorm=use_rmsnorm, rngs=rngs, dtype=dtype)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))
        return x.reshape(x.shape[0], c, h * p, w * p)

    def __call__(self, x, t, y, training=True, rng=None):
        x = self.x_embedder(x) + self.pos_embed.value
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, training=training, rng=rng)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if self.learn_sigma:
            x = x[:, :self.in_channels]
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0.0, 1.0)):
        half = x[:x.shape[0] // 2]
        combined = jnp.concatenate([half, half], axis=0)
        model_out = self(combined, t, y, training=False)
        eps = model_out[:, :self.in_channels]
        rest = model_out[:, self.in_channels:]
        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
        t_min, t_max = cfg_interval
        t_half = t[:t.shape[0] // 2]
        mask = ((t_half >= t_min) & (t_half <= t_max)).reshape(-1, *([1] * (cond_eps.ndim - 1)))
        half_eps = jnp.where(mask, uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps)
        eps = jnp.concatenate([half_eps, half_eps], axis=0)
        return jnp.concatenate([eps, rest], axis=1)
