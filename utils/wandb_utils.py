"""WandB logging utilities — JAX port."""

from __future__ import annotations

import math
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


_wandb_initialized = False


def is_main_process() -> bool:
    return jax.process_index() == 0


def initialize(config: dict, entity: str, exp_name: str, project_name: str):
    """Initialize WandB run (only on main process)."""
    global _wandb_initialized
    if not is_main_process():
        return

    import wandb

    if "WANDB_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_KEY"])

    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config,
        resume="allow",
        reinit=True,
    )
    _wandb_initialized = True


def log(stats: dict, step: Optional[int] = None):
    """Log metrics to WandB."""
    if not is_main_process():
        return
    import wandb
    wandb.log({k: float(v) if hasattr(v, 'item') else v for k, v in stats.items()}, step=step)


def log_image(images: np.ndarray, key: str = "samples", step: Optional[int] = None):
    """Log image grid to WandB.

    Args:
        images: numpy array (B, C, H, W) or (B, H, W, C) in [0, 1].
    """
    if not is_main_process():
        return

    import wandb

    if images.ndim == 4 and images.shape[1] in (1, 3):
        # NCHW → NHWC
        images = np.transpose(images, (0, 2, 3, 1))

    images = np.clip(images * 255, 0, 255).astype(np.uint8)

    nrow = max(1, round(math.sqrt(images.shape[0])))
    grid = _make_grid(images, nrow)

    wandb.log({key: wandb.Image(grid)}, step=step)


def _make_grid(images: np.ndarray, nrow: int) -> np.ndarray:
    """Create image grid from batch (B, H, W, C)."""
    B, H, W, C = images.shape
    ncol = math.ceil(B / nrow)

    grid = np.zeros((nrow * H, ncol * W, C), dtype=np.uint8)
    for idx in range(B):
        r = idx // ncol
        c = idx % ncol
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = images[idx]

    return grid
