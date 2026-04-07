"""Model utilities — instantiate_from_config, etc. JAX port."""

from __future__ import annotations

import importlib
from typing import Any

from flax import nnx
from omegaconf import OmegaConf


def get_obj_from_str(string: str, reload: bool = False):
    """Import a class from a dotted string path."""
    module_path, cls_name = string.rsplit(".", 1)
    if reload:
        mod = importlib.import_module(module_path)
        importlib.reload(mod)
    return getattr(importlib.import_module(module_path, package=None), cls_name)


def instantiate_from_config(config, **extra_kwargs) -> Any:
    """Instantiate a class from OmegaConf config with 'target' and 'params' keys.

    Example config:
        target: stage1.RAE
        params:
            encoder_cls: Dinov2withNorm
            ...
        ckpt: path/to/checkpoint.pt  # optional
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")

    params = dict(config.get("params", {}))
    params.update(extra_kwargs)

    cls = get_obj_from_str(config["target"])
    obj = cls(**params)

    # Optional checkpoint loading
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        _load_checkpoint_into(obj, ckpt_path)

    return obj


def _load_checkpoint_into(obj: Any, ckpt_path: str):
    """Load PyTorch checkpoint into Flax NNX model (basic support)."""
    import torch

    state = torch.load(ckpt_path, map_location="cpu")

    # Handle training checkpoint format
    if isinstance(state, dict):
        if "ema" in state:
            state = state["ema"]
        elif "model" in state:
            state = state["model"]

    # If the object has a load_pretrained_torch method, use it
    if hasattr(obj, "load_pretrained_torch"):
        obj.load_pretrained_torch(ckpt_path)
    else:
        print(f"[Warning] No load_pretrained_torch method on {type(obj).__name__}, skipping weight load from {ckpt_path}")
