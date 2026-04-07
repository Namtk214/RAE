"""Extract RAE decoder weights and save as standalone file. JAX port."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def main():
    parser = argparse.ArgumentParser(description="Extract decoder weights from a checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("No stage_1 section found in config.")

    rngs = nnx.Rngs(params=0)
    # rae = instantiate_from_config(rae_config, rngs=rngs)

    # If checkpoint provided, load it
    if args.ckpt is not None:
        from utils.resume_utils import load_checkpoint
        # checkpoint loading logic here
        print(f"Would load checkpoint from {args.ckpt} (use_ema={args.use_ema})")

    # Extract decoder state
    # decoder_state = nnx.state(rae.decoder)
    # For now, placeholder:
    decoder_state = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    flat_state = jax.tree.leaves(decoder_state)
    np.savez(str(out_path),
             **{f"param_{i}": np.array(p) for i, p in enumerate(flat_state)})

    print(f"Saved decoder to: {out_path}")
    print(f"Keys: {len(flat_state)}")


if __name__ == "__main__":
    main()
