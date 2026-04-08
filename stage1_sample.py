"""Stage 1 reconstruction demo — encode + decode a single image. JAX port."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from PIL import Image

from stage1 import RAE
from utils.train_utils import parse_configs
from utils.resume_utils import restore_checkpoint, build_checkpoint_manager


def load_image(path: Path, image_size: int = 256) -> jnp.ndarray:
    """Load image → (1, H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return jnp.array(arr[None, ...])


def save_image(arr: jnp.ndarray, path: Path):
    """Save (1, H, W, 3) or (1, C, H, W) float32 image to file."""
    img = np.array(arr[0])
    if img.shape[0] in (1, 3, 4) and img.ndim == 3:  # CHW → HWC
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct an image with Stage-1 RAE")
    parser.add_argument("--config", required=True, help="YAML config with stage_1 section")
    parser.add_argument("--image", type=Path, default=Path("assets/pixabay_cat.png"))
    parser.add_argument("--output", type=Path, default=Path("recon.png"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoint directory (auto-resumes latest). "
                             "If None, uses random weights.")
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA weights from checkpoint")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    # ── Load config & build RAE ──────────────────────────────────
    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("No stage_1 section found in config.")

    rngs = nnx.Rngs(params=0, dropout=1)
    rae_params = dict(OmegaConf.to_container(rae_config.get("params", {}), resolve=True))
    # inference: no noise
    rae_params["noise_tau"] = 0.0
    rae = RAE(**rae_params, rngs=rngs)

    # ── Load checkpoint if provided ───────────────────────────────
    if args.ckpt_dir is not None:
        mngr = build_checkpoint_manager(args.ckpt_dir)
        ckpt, step = restore_checkpoint(mngr)
        if ckpt is None:
            print(f"[Warning] No checkpoint found in {args.ckpt_dir}, using random weights.")
        else:
            key = "ema" if args.use_ema else "model"
            decoder_state = ckpt[key]
            # Load decoder weights into model
            graphdef, _ = nnx.split(rae.decoder)
            loaded_decoder = nnx.merge(graphdef, decoder_state)
            nnx.update(rae.decoder, nnx.state(loaded_decoder))
            print(f"[ckpt] loaded {'EMA' if args.use_ema else 'model'} weights from step {step}")

    # ── Encode & decode ───────────────────────────────────────────
    image = load_image(args.image, args.image_size)
    rng = jax.random.PRNGKey(0)

    latent = rae.encode(image, rng=rng, training=False)
    recon = rae.decode(latent)

    recon = jnp.clip(recon, 0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(recon, args.output)

    print(f"Saved reconstruction to {args.output.resolve()}")
    print(f"Input:  {image.shape}  (NHWC)")
    print(f"Latent: {latent.shape}")
    print(f"Recon:  {recon.shape}")


if __name__ == "__main__":
    main()
