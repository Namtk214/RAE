"""Extract RAE decoder weights and save as standalone .npz file. JAX port.

Usage:
    python extract_decoder.py \
        --config configs/stage1/training/DINOv2-B_decXL.yaml \
        --ckpt ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints/ckpt_XXXXXXX.pkl \
        --use-ema \
        --out models/decoders/dinov2/wReg_base/ViTXL_n08/model.npz
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from stage1 import RAE
from utils.train_utils import parse_configs


def main():
    parser = argparse.ArgumentParser(description="Extract decoder weights from a Stage-1 checkpoint.")
    parser.add_argument("--config", type=str, required=True,
                        help="Stage-1 training YAML config")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint .pkl file or directory. "
                             "If directory, uses latest ckpt_*.pkl inside it.")
    parser.add_argument("--use-ema", action="store_true",
                        help="Extract EMA weights instead of model weights")
    parser.add_argument("--out", type=str, required=True,
                        help="Output path for the decoder .npz file")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("No 'stage_1' section found in config.")

    # ── Build RAE to get graph structure ──────────────────────────
    rngs = nnx.Rngs(params=0, dropout=1)
    rae_params = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    rae = RAE(**rae_params, rngs=rngs)
    print(f"[init] RAE built. Decoder params: "
          f"{sum(p.size for p in jax.tree.leaves(nnx.state(rae.decoder))) / 1e6:.2f}M")

    # ── Resolve checkpoint path ───────────────────────────────────
    ckpt_path = args.ckpt
    if ckpt_path is not None:
        p = Path(ckpt_path)
        if p.is_dir():
            # Find latest ckpt_*.pkl in directory
            import glob
            candidates = sorted(glob.glob(str(p / "ckpt_*.pkl")))
            if not candidates:
                raise FileNotFoundError(f"No ckpt_*.pkl files found in {p}")
            ckpt_path = candidates[-1]
            print(f"[ckpt] auto-selected latest: {ckpt_path}")

    # ── Load checkpoint ───────────────────────────────────────────
    if ckpt_path is None or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    key = "ema" if args.use_ema else "model"
    if key not in ckpt:
        available = list(ckpt.keys())
        raise KeyError(f"Key '{key}' not found in checkpoint. Available: {available}")

    decoder_state = ckpt[key]
    print(f"[ckpt] loaded {'EMA' if args.use_ema else 'model'} weights from {ckpt_path}")

    # ── Extract only decoder sub-state ───────────────────────────
    # The checkpoint stores the full decoder state (since only decoder is trained)
    # Convert all numpy arrays to ensure serializability
    decoder_state_np = jax.tree.map(
        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
        decoder_state,
    )

    # ── Save ──────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten state tree into key→array dict for .npz
    flat_leaves, treedef = jax.tree.flatten(decoder_state_np)
    save_dict = {f"param_{i:06d}": np.asarray(p) for i, p in enumerate(flat_leaves)}

    # Also save metadata for reconstruction
    save_dict["__num_params__"] = np.array(len(flat_leaves))
    np.savez(str(out_path), **save_dict)

    total = sum(p.size for p in flat_leaves)
    print(f"[done] saved {len(flat_leaves)} arrays ({total / 1e6:.2f}M params) → {out_path}")


if __name__ == "__main__":
    main()
