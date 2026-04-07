"""Checkpoint utilities using Orbax for TPU-compatible distributed checkpointing."""

from __future__ import annotations

import os
from typing import Optional, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


def build_checkpoint_manager(
    workdir: str,
    save_interval_steps: int = 1,
    max_to_keep: int = 5,
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,
        max_to_keep=max_to_keep,
    )
    return ocp.CheckpointManager(
        workdir,
        options=options,
    )


def save_checkpoint(
    mngr: ocp.CheckpointManager,
    step: int,
    model_params: dict,
    ema_params: dict,
    opt_state: dict,
    extra: Optional[dict] = None,
):
    """Save checkpoint with model, EMA, optimizer state."""
    ckpt = {
        "model": model_params,
        "ema": ema_params,
        "opt_state": opt_state,
    }
    if extra:
        ckpt.update(extra)

    mngr.save(step, args=ocp.args.StandardSave(ckpt))

    if jax.process_index() == 0:
        print(f"[Checkpoint] Saved at step {step}")


def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    step: Optional[int] = None,
    target: Optional[dict] = None,
) -> Tuple[Optional[dict], Optional[int]]:
    """Restore latest or specific checkpoint."""
    if step is None:
        step = mngr.latest_step()

    if step is None:
        return None, None

    if target is not None:
        ckpt = mngr.restore(step, args=ocp.args.StandardRestore(target))
    else:
        ckpt = mngr.restore(step)

    if jax.process_index() == 0:
        print(f"[Checkpoint] Restored from step {step}")

    return ckpt, step


def configure_experiment_dirs(results_dir: str, experiment_name: str) -> Tuple[str, str]:
    """Create experiment and checkpoint directories."""
    experiment_dir = os.path.join(results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    if jax.process_index() == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir
