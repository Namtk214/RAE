"""Checkpoint utilities — pickle-based for TPU single-host training."""

from __future__ import annotations

import os
import glob
import pickle
from typing import Optional, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class SimpleCheckpointManager:
    """Simple checkpoint manager using pickle.
    
    Avoids orbax async executor issues on TPU.
    Uses pickle to handle arbitrary NNX state structures.
    """
    
    def __init__(self, workdir: str, max_to_keep: int = 5):
        self.workdir = workdir
        self.max_to_keep = max_to_keep
        os.makedirs(workdir, exist_ok=True)
    
    def latest_step(self) -> Optional[int]:
        """Find the latest checkpoint step."""
        pattern = os.path.join(self.workdir, "ckpt_*.pkl")
        files = sorted(glob.glob(pattern))
        if not files:
            # Also check for old .npz format
            pattern_npz = os.path.join(self.workdir, "ckpt_*.npz")
            files = sorted(glob.glob(pattern_npz))
            if not files:
                return None
        latest = files[-1]
        step_str = os.path.basename(latest).split("_")[1].split(".")[0]
        return int(step_str)
    
    def save(self, step: int, ckpt: dict):
        """Save checkpoint using pickle."""
        filepath = os.path.join(self.workdir, f"ckpt_{step:07d}.pkl")
        
        # Convert all JAX arrays to numpy for serialization
        ckpt_np = jax.tree.map(
            lambda x: np.asarray(x) if hasattr(x, 'shape') else x,
            ckpt
        )
        
        with open(filepath, 'wb') as f:
            pickle.dump(ckpt_np, f)
        
        # Clean up old checkpoints
        pattern = os.path.join(self.workdir, "ckpt_*.pkl")
        files = sorted(glob.glob(pattern))
        while len(files) > self.max_to_keep:
            os.remove(files.pop(0))
        
        if jax.process_index() == 0:
            print(f"[Checkpoint] Saved at step {step} -> {filepath}", flush=True)
    
    def restore(self, step: Optional[int] = None) -> Tuple[Optional[dict], Optional[int]]:
        """Restore checkpoint."""
        if step is None:
            step = self.latest_step()
        if step is None:
            return None, None
        
        filepath = os.path.join(self.workdir, f"ckpt_{step:07d}.pkl")
        if not os.path.exists(filepath):
            return None, None
        
        with open(filepath, 'rb') as f:
            ckpt = pickle.load(f)
        
        # Convert numpy arrays back to JAX arrays
        ckpt = jax.tree.map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            ckpt
        )
        
        if jax.process_index() == 0:
            print(f"[Checkpoint] Restored from step {step}", flush=True)
        
        return ckpt, step


def build_checkpoint_manager(
    workdir: str,
    save_interval_steps: int = 1,
    max_to_keep: int = 5,
) -> SimpleCheckpointManager:
    """Create a simple checkpoint manager."""
    return SimpleCheckpointManager(workdir, max_to_keep=max_to_keep)


def save_checkpoint(
    mngr: SimpleCheckpointManager,
    step: int,
    model_params,
    ema_params,
    opt_state,
    extra: Optional[dict] = None,
):
    """Save checkpoint with model, EMA, optimizer state."""
    # Convert to numpy via jax.device_get
    model_np = jax.device_get(model_params)
    ema_np = jax.device_get(ema_params)
    opt_np = jax.device_get(opt_state)
    
    ckpt = {
        "model": model_np,
        "ema": ema_np,
        "opt_state": opt_np,
    }
    if extra:
        ckpt.update(extra)
    
    mngr.save(step, ckpt)


def restore_checkpoint(
    mngr: SimpleCheckpointManager,
    step: Optional[int] = None,
    target: Optional[dict] = None,
) -> Tuple[Optional[dict], Optional[int]]:
    """Restore latest or specific checkpoint."""
    return mngr.restore(step)


def configure_experiment_dirs(results_dir: str, experiment_name: str) -> Tuple[str, str]:
    """Create experiment and checkpoint directories on ALL workers."""
    experiment_dir = os.path.join(results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # All workers must create dirs so checkpoint files can be saved/restored symmetrically
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir

