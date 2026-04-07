"""Stage 1 reconstruction demo — encode + decode a single image. JAX port."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from PIL import Image

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def load_image(path: Path, image_size: int = 256) -> jnp.ndarray:
    """Load image → (1, H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return jnp.array(arr[None, ...])


def save_image(arr: jnp.ndarray, path: Path):
    """Save (1, H, W, 3) float32 image to file."""
    img = np.array(arr[0])
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct an image with Stage-1 RAE")
    parser.add_argument("--config", required=True, help="YAML config with stage_1 section")
    parser.add_argument("--image", type=Path, default=Path("assets/pixabay_cat.png"))
    parser.add_argument("--output", type=Path, default=Path("recon.png"))
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("No stage_1 section found in config.")

    rngs = nnx.Rngs(params=0)
    # rae = instantiate_from_config(rae_config, rngs=rngs)

    image = load_image(args.image, args.image_size)

    # latent = rae.encode(image)
    # recon = rae.decode(latent)
    # For now, identity:
    recon = image
    latent = image

    recon = jnp.clip(recon, 0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(recon, args.output)

    print(f"Saved reconstruction to {args.output.resolve()}")
    print(f"Input: {image.shape}, Latent: {latent.shape}, Recon: {recon.shape}")


if __name__ == "__main__":
    main()
