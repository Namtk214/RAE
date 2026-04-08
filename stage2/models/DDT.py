"""DiTwDDTHead — Diffusion Transformer with DDT Head. JAX/Flax NNX port."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .model_utils import (
    RMSNorm, SwiGLUFFN, NormAttention,
    GaussianFourierEmbedding, LabelEmbedder,
    ddt_modulate, ddt_gate,
    build_rope_freqs, get_2d_sincos_pos_embed,
)


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------
class PatchEmbed(nnx.Module):
    """Patch embedding via Conv2d."""
    def __init__(self, input_size: int, patch_size: int, in_chans: int, embed_dim: int,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nnx.Conv(
            in_chans, embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='VALID',
            use_bias=True,
            dtype=dtype, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, C) → (B, num_patches, embed_dim)"""
        x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


# ---------------------------------------------------------------------------
# LightningDDTBlock
# ---------------------------------------------------------------------------
class LightningDDTBlock(nnx.Module):
    """DiT block with AdaLN modulation, RoPE, QK-Norm, RMSNorm, SwiGLU."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.wo_shift = wo_shift

        # Norms
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size, rngs=rngs, dtype=dtype)
            self.norm2 = RMSNorm(hidden_size, rngs=rngs, dtype=dtype)
        else:
            self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)
            self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)

        # Attention
        self.attn = NormAttention(
            hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm,
            rngs=rngs, dtype=dtype,
        )

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden), rngs=rngs, dtype=dtype)
        else:
            self.mlp = _Mlp(hidden_size, mlp_hidden, rngs=rngs, dtype=dtype)

        # AdaLN modulation
        num_mod = 4 if wo_shift else 6
        self.adaLN_mod_linear = nnx.Linear(hidden_size, num_mod * hidden_size, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, feat_rope=None) -> jnp.ndarray:
        if c.ndim < x.ndim:
            c = c[:, None, :]  # (B, 1, C)

        mod = jax.nn.silu(c)
        mod = self.adaLN_mod_linear(mod)

        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(mod, 4, axis=-1)
            shift_msa = jnp.zeros_like(scale_msa)
            shift_mlp = jnp.zeros_like(scale_mlp)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)

        # Attention with modulation + gating
        x = x + ddt_gate(
            self.attn(ddt_modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope),
            gate_msa
        )
        # MLP with modulation + gating
        x = x + ddt_gate(
            self.mlp(ddt_modulate(self.norm2(x), shift_mlp, scale_mlp)),
            gate_mlp
        )
        return x


class _Mlp(nnx.Module):
    """Simple MLP (GELU approximate)."""
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.fc1 = nnx.Linear(in_features, hidden_features, dtype=dtype, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, in_features, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=True))


# ---------------------------------------------------------------------------
# DDTFinalLayer
# ---------------------------------------------------------------------------
class DDTFinalLayer(nnx.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 use_rmsnorm: bool = False, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size, rngs=rngs, dtype=dtype)
        else:
            self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, patch_size * patch_size * out_channels, use_bias=True, dtype=dtype, rngs=rngs)
        self.adaLN_linear = nnx.Linear(hidden_size, 2 * hidden_size, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        if c.ndim < x.ndim:
            c = c[:, None, :]
        mod = jax.nn.silu(c)
        mod = self.adaLN_linear(mod)
        shift, scale = jnp.split(mod, 2, axis=-1)
        x = ddt_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# DiTwDDTHead — main model
# ---------------------------------------------------------------------------
class DiTwDDTHead(nnx.Module):
    """Diffusion Transformer with DDT (Dual-head DiT) architecture.

    Architecture:
        - s_embedder: patch embed for encoder path
        - Encoder blocks (depth[0]) with enc_feat_rope
        - s_projector: project from encoder to decoder hidden size
        - x_embedder: patch embed for decoder path
        - Decoder blocks (depth[1]) with dec_feat_rope
        - final_layer: AdaLN → Linear → unpatchify
    """

    def __init__(
        self,
        input_size: int = 1,
        patch_size: Union[list, int] = 1,
        in_channels: int = 768,
        hidden_size: list = [1152, 2048],
        depth: list = [28, 2],
        num_heads: Union[list, int] = [16, 16],
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)
        self.num_encoder_blocks = depth[0]
        self.num_decoder_blocks = depth[1]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope
        self.use_pos_embed = use_pos_embed

        # Patch sizes
        if isinstance(patch_size, (int, float)):
            patch_size = [int(patch_size), int(patch_size)]
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]

        s_channel_per_token = in_channels * self.s_patch_size ** 2
        x_channel_per_token = in_channels * self.x_patch_size ** 2
        self.x_channel_per_token = x_channel_per_token

        # Embedders
        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, s_channel_per_token,
                                     self.encoder_hidden_size, rngs=rngs, dtype=dtype)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, x_channel_per_token,
                                     self.decoder_hidden_size, rngs=rngs, dtype=dtype)

        if self.encoder_hidden_size != self.decoder_hidden_size:
            self.s_projector = nnx.Linear(self.encoder_hidden_size, self.decoder_hidden_size, dtype=dtype, rngs=rngs)
        else:
            self.s_projector = None

        # Timestep + label embeddings
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size, rngs=rngs, dtype=dtype)
        self.y_embedder = LabelEmbedder(num_classes, self.encoder_hidden_size, class_dropout_prob, rngs=rngs, dtype=dtype)

        # Final layer
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token,
            use_rmsnorm=use_rmsnorm, rngs=rngs, dtype=dtype
        )

        # Positional embedding
        if use_pos_embed:
            num_patches = self.s_embedder.num_patches
            pos = get_2d_sincos_pos_embed(self.encoder_hidden_size, int(math.sqrt(num_patches)))
            self.pos_embed = nnx.Param(jnp.array(pos[None, ...], dtype=dtype))
        else:
            self.pos_embed = None

        # RoPE
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        if use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw = int(math.sqrt(self.s_embedder.num_patches))
            self.enc_rope = build_rope_freqs(enc_half_head_dim, pt_seq_len=hw)

            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_dec = int(math.sqrt(self.x_embedder.num_patches))
            self.dec_rope = build_rope_freqs(dec_half_head_dim, pt_seq_len=hw_dec)
        else:
            self.enc_rope = None
            self.dec_rope = None

        # Transformer blocks
        self.blocks = []
        for i in range(self.num_blocks):
            if i < self.num_encoder_blocks:
                h = self.encoder_hidden_size
                nh = enc_num_heads
            else:
                h = self.decoder_hidden_size
                nh = dec_num_heads

            self.blocks.append(
                LightningDDTBlock(
                    h, nh, mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm, use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu, wo_shift=wo_shift,
                    rngs=rngs, dtype=dtype,
                )
            )

    def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """(B, N, patch^2 * C) → (B, C, H*p, W*p)"""
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], h, w, p, p, c)
        # nhwpqc -> nchpwq
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))
        return x.reshape(x.shape[0], c, h * p, w * p)

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        y: jnp.ndarray,
        s: Optional[jnp.ndarray] = None,
        training: bool = True,
        rng: Optional[jax.random.PRNGKey] = None,
        return_activations: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        """Forward pass.

        Args:
            x: Input latent (B, H, W, C) NHWC
            t: Timestep (B,)
            y: Class labels (B,)
            s: Optional pre-computed encoder output
            training: Whether in training mode (for label dropout)
            rng: PRNG key for label dropout

        Returns:
            Predicted velocity/noise (B, C, H', W')
        """
        # Timestep + label conditioning
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, training=training, rng=rng)
        c = jax.nn.silu(t_emb + y_emb)

        activations = {}
        if s is None:
            # Encoder path
            s = self.s_embedder(x)
            if self.pos_embed is not None:
                s = s + self.pos_embed.value

            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=self.enc_rope)
                if return_activations:
                    activations[f"encoder_block_{i}"] = jnp.sqrt(jnp.mean(jnp.square(s)))

            # Broadcast t to spatial dims and combine
            t_broadcast = t_emb[:, None, :].repeat(s.shape[1], axis=1)
            s = jax.nn.silu(t_broadcast + s)

        # Project encoder → decoder dim
        if self.s_projector is not None:
            s = self.s_projector(s)

        # Decoder path
        x = self.x_embedder(x)

        for i in range(self.num_encoder_blocks, self.num_blocks):
            x = self.blocks[i](x, s, feat_rope=self.dec_rope)
            if return_activations:
                activations[f"decoder_block_{i - self.num_encoder_blocks}"] = jnp.sqrt(jnp.mean(jnp.square(x)))

        x = self.final_layer(x, s)
        x = self.unpatchify(x)  # (B, C, H, W) NCHW
        x = x.transpose(0, 2, 3, 1)  # → (B, H, W, C) NHWC
        if return_activations:
            return x, activations
        return x

    def forward_with_cfg(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        y: jnp.ndarray,
        cfg_scale: float,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
    ) -> jnp.ndarray:
        """Forward with classifier-free guidance."""
        half = x[:x.shape[0] // 2]
        combined = jnp.concatenate([half, half], axis=0)
        model_out = self(combined, t, y, training=False)

        eps = model_out[:, :self.in_channels]
        rest = model_out[:, self.in_channels:]

        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)

        guid_min, guid_max = cfg_interval
        t_half = t[:t.shape[0] // 2]
        mask = ((t_half >= guid_min) & (t_half <= guid_max)).reshape(-1, *([1] * (cond_eps.ndim - 1)))

        half_eps = jnp.where(mask, uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps)
        eps = jnp.concatenate([half_eps, half_eps], axis=0)
        return jnp.concatenate([eps, rest], axis=1)
