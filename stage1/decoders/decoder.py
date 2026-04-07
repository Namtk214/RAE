"""GeneralDecoder — ViT-MAE decoder in Flax NNX. Port of PyTorch decoder.py."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .utils import ACT2FN, ViTMAEConfig, get_2d_sincos_pos_embed

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Non-trainable buffer variable type
# ---------------------------------------------------------------------------
class Buffer(nnx.Variable):
    """Non-trainable container for fixed arrays (e.g. pos embeddings)."""
    pass


# ---------------------------------------------------------------------------
# Decoder output
# ---------------------------------------------------------------------------
@dataclass
class ViTMAEDecoderOutput:
    logits: Array
    hidden_states: Optional[Tuple[Array, ...]] = None
    attentions: Optional[Tuple[Array, ...]] = None


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------
class ViTMAESelfAttention(nnx.Module):
    def __init__(self, config: ViTMAEConfig, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        assert hidden_size % num_heads == 0

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = num_heads * self.attention_head_size
        self.scale = 1.0 / math.sqrt(self.attention_head_size)

        self.query = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, dtype=dtype, rngs=rngs)
        self.key = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, dtype=dtype, rngs=rngs)
        self.value = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, dtype=dtype, rngs=rngs)

    def _reshape_for_scores(self, x: Array) -> Array:
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))  # (B, H, L, D)

    def __call__(self, hidden_states: Array) -> Array:
        q = self._reshape_for_scores(self.query(hidden_states))
        k = self._reshape_for_scores(self.key(hidden_states))
        v = self._reshape_for_scores(self.value(hidden_states))

        attn_scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * self.scale
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)

        context = jnp.matmul(attn_probs, v)
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = jnp.reshape(context, context.shape[:-2] + (self.all_head_size,))
        return context


# ---------------------------------------------------------------------------
# Attention output projection
# ---------------------------------------------------------------------------
class ViTMAESelfOutput(nnx.Module):
    def __init__(self, config: ViTMAEConfig, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.dense = nnx.Linear(config.hidden_size, config.hidden_size, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        return self.dense(hidden_states)


# ---------------------------------------------------------------------------
# Full attention block
# ---------------------------------------------------------------------------
class ViTMAEAttention(nnx.Module):
    def __init__(self, config: ViTMAEConfig, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.attention = ViTMAESelfAttention(config, rngs=rngs, dtype=dtype)
        self.output = ViTMAESelfOutput(config, rngs=rngs, dtype=dtype)

    def __call__(self, hidden_states: Array) -> Array:
        attn_out = self.attention(hidden_states)
        return self.output(attn_out)


# ---------------------------------------------------------------------------
# FFN intermediate
# ---------------------------------------------------------------------------
class ViTMAEIntermediate(nnx.Module):
    def __init__(self, config: ViTMAEConfig, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.dense = nnx.Linear(config.hidden_size, config.intermediate_size, dtype=dtype, rngs=rngs)
        act = config.hidden_act
        if isinstance(act, str):
            if act not in ACT2FN:
                raise ValueError(f"Unsupported activation: {act}")
            self.activation = ACT2FN[act]
        else:
            self.activation = act

    def __call__(self, hidden_states: Array) -> Array:
        return self.activation(self.dense(hidden_states))


# ---------------------------------------------------------------------------
# FFN output (with residual)
# ---------------------------------------------------------------------------
class ViTMAEOutput(nnx.Module):
    def __init__(self, config: ViTMAEConfig, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.dense = nnx.Linear(config.intermediate_size, config.hidden_size, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: Array, input_tensor: Array) -> Array:
        return self.dense(hidden_states) + input_tensor


# ---------------------------------------------------------------------------
# Single transformer layer
# ---------------------------------------------------------------------------
class ViTMAELayer(nnx.Module):
    def __init__(self, config: ViTMAEConfig, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.attention = ViTMAEAttention(config, rngs=rngs, dtype=dtype)
        self.intermediate = ViTMAEIntermediate(config, rngs=rngs, dtype=dtype)
        self.output = ViTMAEOutput(config, rngs=rngs, dtype=dtype)
        self.layernorm_before = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs)
        self.layernorm_after = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        # Pre-LN attention
        attn_out = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attn_out + hidden_states
        # Pre-LN FFN
        layer_out = self.layernorm_after(hidden_states)
        layer_out = self.intermediate(layer_out)
        layer_out = self.output(layer_out, hidden_states)
        return layer_out


# ---------------------------------------------------------------------------
# GeneralDecoder
# ---------------------------------------------------------------------------
class GeneralDecoder(nnx.Module):
    """ViT-MAE decoder — port of PyTorch GeneralDecoder."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        num_patches: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: jnp.dtype = jnp.float32,
    ):
        decoder_config = copy.deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size

        self.config = config
        self.decoder_config = decoder_config
        self.num_patches = num_patches
        self.dtype = dtype

        # Embed projection: encoder_hidden → decoder_hidden
        self.decoder_embed = nnx.Linear(config.hidden_size, decoder_config.hidden_size, dtype=dtype, rngs=rngs)

        # Fixed sin-cos positional embeddings (with CLS token)
        pos_embed = get_2d_sincos_pos_embed(decoder_config.hidden_size, int(math.sqrt(num_patches)), add_cls_token=True)
        self.decoder_pos_embed = Buffer(jnp.asarray(pos_embed, dtype=dtype)[None, ...])

        # Transformer layers (with gradient checkpointing for TPU memory)
        self.decoder_layers = [
            ViTMAELayer(decoder_config, rngs=rngs, dtype=dtype)
            for _ in range(decoder_config.num_hidden_layers)
        ]

        self.decoder_norm = nnx.LayerNorm(decoder_config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs)
        self.decoder_pred = nnx.Linear(
            decoder_config.hidden_size,
            config.patch_size * config.patch_size * config.num_channels,
            dtype=dtype,
            rngs=rngs,
        )

        # Trainable CLS token
        self.trainable_cls_token = nnx.Param(
            jnp.zeros((1, 1, decoder_config.hidden_size), dtype=dtype)
        )

    def interpolate_latent(self, x: Array) -> Array:
        batch_size, length, channels = x.shape
        if length == self.num_patches:
            return x
        height = width = int(math.sqrt(length))
        target_hw = int(math.sqrt(self.num_patches))
        x_img = jnp.reshape(x, (batch_size, height, width, channels))
        x_img = jax.image.resize(x_img, (batch_size, target_hw, target_hw, channels), method="linear")
        return jnp.reshape(x_img, (batch_size, self.num_patches, channels))

    def unpatchify(self, patchified: Array) -> Array:
        """(B, N, patch_size^2 * C) → (B, C, H, W)"""
        p = self.config.patch_size
        c = self.config.num_channels
        h = w = int(math.sqrt(patchified.shape[1]))
        x = jnp.reshape(patchified, (patchified.shape[0], h, w, p, p, c))
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))  # (B, C, h, p, w, p)
        return jnp.reshape(x, (patchified.shape[0], c, h * p, w * p))

    def __call__(self, hidden_states: Array, *, drop_cls_token: bool = False) -> ViTMAEDecoderOutput:
        x = self.decoder_embed(hidden_states)

        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)

        cls_token = jnp.broadcast_to(self.trainable_cls_token.value, (x_.shape[0],) + self.trainable_cls_token.value.shape[1:])
        x = jnp.concatenate([cls_token, x_], axis=1)
        hidden_states = x + self.decoder_pos_embed.value

        # Forward through decoder layers with gradient checkpointing
        for layer in self.decoder_layers:
            hidden_states = nnx.remat(layer)(hidden_states)

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]  # Remove CLS token

        return ViTMAEDecoderOutput(logits=logits)

    def load_pretrained_torch(self, path: str):
        """Load PyTorch pretrained decoder weights and convert to JAX."""
        import torch

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "ema" in state:
            state = state["ema"]

        # Convert PyTorch decoder weights to this module's structure
        # The mapping follows the PyTorch GeneralDecoder naming convention
        def _to_jax(t):
            arr = t.detach().cpu().numpy()
            # PyTorch Linear stores weight as (out, in), Flax NNX as (in, out)
            return arr

        # decoder_embed
        self.decoder_embed.kernel.value = jnp.array(_to_jax(state["decoder_embed.weight"]).T)
        self.decoder_embed.bias.value = jnp.array(_to_jax(state["decoder_embed.bias"]))

        # decoder_norm
        self.decoder_norm.scale.value = jnp.array(_to_jax(state["decoder_norm.weight"]))
        self.decoder_norm.bias.value = jnp.array(_to_jax(state["decoder_norm.bias"]))

        # decoder_pred
        self.decoder_pred.kernel.value = jnp.array(_to_jax(state["decoder_pred.weight"]).T)
        self.decoder_pred.bias.value = jnp.array(_to_jax(state["decoder_pred.bias"]))

        # CLS token
        if "cls_token" in state:
            self.trainable_cls_token.value = jnp.array(_to_jax(state["cls_token"]))

        # Layers
        for i, layer in enumerate(self.decoder_layers):
            prefix = f"decoder_layers.{i}."

            # layernorm_before
            layer.layernorm_before.scale.value = jnp.array(_to_jax(state[f"{prefix}layernorm_before.weight"]))
            layer.layernorm_before.bias.value = jnp.array(_to_jax(state[f"{prefix}layernorm_before.bias"]))

            # attention Q/K/V
            layer.attention.attention.query.kernel.value = jnp.array(_to_jax(state[f"{prefix}attention.attention.query.weight"]).T)
            layer.attention.attention.query.bias.value = jnp.array(_to_jax(state[f"{prefix}attention.attention.query.bias"]))
            layer.attention.attention.key.kernel.value = jnp.array(_to_jax(state[f"{prefix}attention.attention.key.weight"]).T)
            layer.attention.attention.key.bias.value = jnp.array(_to_jax(state[f"{prefix}attention.attention.key.bias"]))
            layer.attention.attention.value.kernel.value = jnp.array(_to_jax(state[f"{prefix}attention.attention.value.weight"]).T)
            layer.attention.attention.value.bias.value = jnp.array(_to_jax(state[f"{prefix}attention.attention.value.bias"]))

            # attention output dense
            layer.attention.output.dense.kernel.value = jnp.array(_to_jax(state[f"{prefix}attention.output.dense.weight"]).T)
            layer.attention.output.dense.bias.value = jnp.array(_to_jax(state[f"{prefix}attention.output.dense.bias"]))

            # layernorm_after
            layer.layernorm_after.scale.value = jnp.array(_to_jax(state[f"{prefix}layernorm_after.weight"]))
            layer.layernorm_after.bias.value = jnp.array(_to_jax(state[f"{prefix}layernorm_after.bias"]))

            # intermediate
            layer.intermediate.dense.kernel.value = jnp.array(_to_jax(state[f"{prefix}intermediate.dense.weight"]).T)
            layer.intermediate.dense.bias.value = jnp.array(_to_jax(state[f"{prefix}intermediate.dense.bias"]))

            # output
            layer.output.dense.kernel.value = jnp.array(_to_jax(state[f"{prefix}output.dense.weight"]).T)
            layer.output.dense.bias.value = jnp.array(_to_jax(state[f"{prefix}output.dense.bias"]))

        print(f"[GeneralDecoder] Loaded pretrained weights from {path}")
