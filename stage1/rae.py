"""RAE model — combines frozen encoder + trainable decoder. Port of PyTorch rae.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .encoders import get_encoder
from .decoders import GeneralDecoder
from .decoders.utils import ViTMAEConfig


class RAE(nnx.Module):
    """Representation Autoencoder.

    Combines a frozen representation encoder (DINOv2) and a trainable ViT-MAE decoder.
    Supports latent noising (tau) during training and optional normalization stats.
    """

    def __init__(
        self,
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-with-registers-base",
        encoder_input_size: int = 224,
        encoder_params: Optional[dict] = None,
        decoder_config_path: str = "configs/decoder/ViTXL",
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: jnp.dtype = jnp.float32,
    ):
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        self.dtype = dtype

        # --- Encoder (frozen) ---
        encoder_kwargs = encoder_params or {}
        encoder_kwargs.setdefault("input_size", encoder_input_size)
        if "dinov2_path" not in encoder_kwargs:
            encoder_kwargs["dinov2_path"] = encoder_config_path

        EncoderCls = get_encoder(encoder_cls)
        self.encoder = EncoderCls(**encoder_kwargs)
        self.encoder_input_size = encoder_input_size
        self.hidden_size = self.encoder.hidden_size
        self.patch_size_encoder = self.encoder.patch_size
        self.num_patches = self.encoder.num_patches

        # --- Decoder (trainable) ---
        decoder_config = self._load_decoder_config(decoder_config_path)
        decoder_config.hidden_size = self.hidden_size
        decoder_config.patch_size = 16  # image_patch_size for reconstruction
        decoder_config.image_size = int(16 * math.sqrt(self.num_patches))

        self.decoder = GeneralDecoder(
            config=decoder_config,
            num_patches=self.num_patches,
            rngs=rngs,
            dtype=dtype,
        )

        if pretrained_decoder_path is not None:
            self.decoder.load_pretrained_torch(pretrained_decoder_path)

        # --- Latent normalization stats ---
        self.latent_mean = None
        self.latent_var = None
        self.eps = 1e-5
        if normalization_stat_path is not None:
            self._load_stats(normalization_stat_path)

        # --- ImageNet normalization params (for decode denormalization) ---
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _load_decoder_config(config_path: str) -> ViTMAEConfig:
        """Load decoder config from JSON file or directory containing config.json."""
        p = Path(config_path)
        if p.is_dir():
            p = p / "config.json"
        if p.exists():
            return ViTMAEConfig.from_json(str(p))
        return ViTMAEConfig()

    def _load_stats(self, path: str):
        """Load pre-computed latent mean/var.

        Supports two formats:
        - .npz (output of calculate_stat.py): keys 'mean' and 'var', shape (H, W, C) JAX NHWC
        - .pt  (PyTorch format): keys 'mean' and 'var', shape (1, C, H, W) NCHW
        """
        if path.endswith(".npz"):
            stats = np.load(path)
            if "mean" in stats:
                m = stats["mean"]
                # If shape is (H, W, C) add batch dim → (1, H, W, C)
                self.latent_mean = m[None] if m.ndim == 3 else m
            if "var" in stats:
                v = stats["var"]
                self.latent_var = v[None] if v.ndim == 3 else v
        else:
            # PyTorch .pt / .pth format
            import torch
            stats = torch.load(path, map_location="cpu")
            if "mean" in stats:
                # PyTorch shape: (1, C, H, W) → JAX NHWC: (1, H, W, C)
                self.latent_mean = np.array(stats["mean"].numpy()).transpose(0, 2, 3, 1)
            if "var" in stats:
                self.latent_var = np.array(stats["var"].numpy()).transpose(0, 2, 3, 1)

    def encode(self, x: jnp.ndarray, *, rng: Optional[jax.random.PRNGKey] = None, training: bool = False) -> jnp.ndarray:
        """Encode images to latent space.

        Args:
            x: Images in [0, 1] range, shape (B, H, W, 3) — NHWC.
            rng: PRNG key for noise injection (required if training=True and noise_tau > 0).
            training: Whether to apply noise injection.

        Returns:
            Latent z of shape (B, H_lat, W_lat, C) if reshape_to_2d, else (B, N, C).
        """
        # Encoder forward (frozen, stop_gradient handled inside encoder)
        z = self.encoder(x)  # (B, num_patches, hidden_size)

        # Noise injection during training
        if training and self.noise_tau > 0 and rng is not None:
            rng1, rng2 = jax.random.split(rng)
            rand_factor = jax.random.uniform(rng1, (z.shape[0], 1, 1))  # per-sample scalar
            noise = jax.random.normal(rng2, z.shape, dtype=z.dtype)
            z = z + self.noise_tau * rand_factor * noise

        # Reshape to 2D spatial if requested
        if self.reshape_to_2d:
            B, N, C = z.shape
            h = w = int(math.sqrt(N))
            z = z.reshape(B, h, w, C)  # (B, 16, 16, 768) for DINOv2-B

        # Normalize latents
        if self.latent_mean is not None:
            mean = jnp.array(self.latent_mean, dtype=z.dtype)
            var = jnp.array(self.latent_var, dtype=z.dtype)
            z = (z - mean) / jnp.sqrt(var + self.eps)

        return z

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode latents to reconstructed images.

        Args:
            z: Latent tokens (B, H, W, C) or (B, N, C).

        Returns:
            Reconstructed images (B, 3, H_img, W_img) in [0, 1] range.
        """
        # De-normalize latents
        if self.latent_mean is not None:
            mean = jnp.array(self.latent_mean, dtype=z.dtype)
            var = jnp.array(self.latent_var, dtype=z.dtype)
            z = z * jnp.sqrt(var + self.eps) + mean

        # Flatten to sequence if 4D
        if z.ndim == 4:
            B, H, W, C = z.shape
            z = z.reshape(B, H * W, C)

        # Decoder forward
        dec_out = self.decoder(z, drop_cls_token=False)
        x_rec = self.decoder.unpatchify(dec_out.logits)  # (B, C, H, W)

        # Denormalize with ImageNet stats: x_rec is in normalized space
        img_mean = jnp.array(self.img_mean, dtype=x_rec.dtype).reshape(1, 3, 1, 1)
        img_std = jnp.array(self.img_std, dtype=x_rec.dtype).reshape(1, 3, 1, 1)
        x_rec = x_rec * img_std + img_mean
        x_rec = jnp.clip(x_rec, 0.0, 1.0)

        return x_rec

    def forward(self, x: jnp.ndarray, *, rng: Optional[jax.random.PRNGKey] = None, training: bool = False) -> jnp.ndarray:
        """Full forward pass: encode → decode.

        Args:
            x: Images in [0, 1], shape (B, H, W, 3).

        Returns:
            Reconstructed images (B, 3, H_img, W_img) in [0, 1].
        """
        z = self.encode(x, rng=rng, training=training)
        return self.decode(z)
