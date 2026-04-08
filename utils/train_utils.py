"""Training utilities — EMA, center_crop, data helpers."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def update_ema(ema_params, model_params, decay: float = 0.9999):
    """Exponential moving average update.

    ema = decay * ema + (1 - decay) * model
    """
    return jax.tree.map(
        lambda ema, model: decay * ema + (1 - decay) * model,
        ema_params,
        model_params,
    )


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center cropping (ADM style)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def images_to_batch(images: list, image_size: int = 256) -> np.ndarray:
    """Convert list of PIL images to numpy batch (B, H, W, 3) in [0, 1]."""
    batch = []
    for img in images:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = center_crop_arr(img, image_size)
        batch.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(batch, axis=0)


# ---------------------------------------------------------------------------
# Config parsing (OmegaConf compatible)
# ---------------------------------------------------------------------------
def parse_configs(config):
    """Parse YAML config into component sections.

    For stage-1 training returns 4 values:
        (rae_config, training_config, gan_config, eval_config)
    For stage-2 training returns 8 values:
        (rae_config, model_config, transport_config, sampler_config,
         guidance_config, misc_config, training_config, eval_config)

    Auto-detects stage-2 configs by checking for 'stage_2' key.
    """
    from omegaconf import OmegaConf, DictConfig

    if isinstance(config, str):
        config = OmegaConf.load(config)

    rae_config = config.get("stage_1", None)
    training_config = config.get("training", None)
    eval_config = config.get("eval", None)

    # Stage-2 specific sections
    model_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc_config = config.get("misc", None)
    data_config = config.get("data", None)

    if model_config is not None:
        # Stage-2 mode: return all 8 sections
        return (rae_config, model_config, transport_config, sampler_config,
                guidance_config, misc_config, training_config, eval_config)
    else:
        # Stage-1 mode: return 4 sections (backward compat)
        gan_config = config.get("gan", None)
        return rae_config, training_config, gan_config, eval_config


def requires_grad(params, flag: bool = True):
    """For JAX, this is a no-op — gradient control is handled via stop_gradient."""
    pass
