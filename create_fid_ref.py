"""Script to compute FID reference statistics (mu, sigma) for a dataset.

Uses the SAME JAX-native InceptionV3 backend as eval/fid.py to ensure
reference stats and generated-image stats are computed in the same feature space.

NOTE: This script runs as a subprocess from train.py while the parent process
already holds the TPU. We force JAX_PLATFORMS=cpu before any JAX import to
avoid device contention. InceptionV3 inference for reference stats is a
one-time cost and runs fast enough on CPU.
"""
# Must be set BEFORE any jax import so JAX doesn't try to init TPU.
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import numpy as np
from tqdm import tqdm
from data import build_dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID reference statistics (mu, sigma) using JAX InceptionV3."
    )
    parser.add_argument("--data-path",        type=str, required=True)
    parser.add_argument("--out-path",         type=str, required=True)
    parser.add_argument("--image-size",       type=int, default=256)
    parser.add_argument("--batch-size",       type=int, default=64,
                        help="Batch size for InceptionV3 feature extraction.")
    parser.add_argument("--num-samples",      type=int, default=50000)
    parser.add_argument("--dataset-type",     type=str, default="tfds", choices=["imagefolder", "tfds"])
    parser.add_argument("--tfds-name",        type=str, default="celebahq256")
    parser.add_argument("--tfds-builder-dir", type=str, default=None)
    parser.add_argument("--split",            type=str, default="train",
                        help="Dataset split: 'train' or 'validation'. Default: train.")
    args = parser.parse_args()

    print(f"Loading dataset (split='{args.split}') from: {args.data_path}")
    ds, steps = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        split=args.split,
        tfds_name=args.tfds_name,
        tfds_builder_dir=args.tfds_builder_dir,
    )

    # ── Load JAX InceptionV3 (same backend as evaluate_generation_distributed) ──
    print("Loading JAX InceptionV3 (pretrained weights)...")
    from eval.fid import get_fid_network, fid_from_stats, moments_from_activations
    fid_fn = get_fid_network()
    print("InceptionV3 ready.")

    import jax
    import jax.numpy as jnp

    @jax.jit
    def extract_acts(images_nhwc_01):
        """images_nhwc_01: NHWC float32 in [0, 1] → [B, 2048] activations."""
        x = images_nhwc_01 * 2.0 - 1.0
        x = jax.image.resize(
            x, (x.shape[0], 299, 299, x.shape[3]),
            method="bilinear", antialias=False,
        )
        x = jnp.clip(x, -1.0, 1.0)
        return fid_fn(x)[..., 0, 0, :]   # [B, 2048]

    # ── Collect images and extract features ──
    all_activations = []
    processed = 0
    print(f"Extracting InceptionV3 features for up to {args.num_samples} images...")

    for step_data in ds:
        if processed >= args.num_samples:
            break

        imgs = np.array(step_data["image"])

        # Normalise to float32 NHWC in [0, 1]
        if imgs.dtype == np.uint8:
            imgs = imgs.astype(np.float32) / 255.0
        elif imgs.max() > 1.5:
            imgs = imgs.astype(np.float32) / 255.0
        else:
            imgs = imgs.astype(np.float32)

        # NCHW → NHWC if needed
        if imgs.ndim == 4 and imgs.shape[1] in (1, 3, 4) and imgs.shape[-1] not in (1, 3, 4):
            imgs = np.transpose(imgs, (0, 2, 3, 1))

        # Only take what we still need
        remaining = args.num_samples - processed
        imgs = imgs[:remaining]

        acts = np.array(extract_acts(jnp.array(imgs)))  # [B, 2048]
        all_activations.append(acts)
        processed += imgs.shape[0]
        print(f"  Processed {processed}/{args.num_samples}", end="\r")

    print()
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"Total activations collected: {all_activations.shape}")

    # ── Compute µ and Σ ──
    mu, sigma = moments_from_activations(all_activations)
    print(f"mu shape: {mu.shape},  sigma shape: {sigma.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    np.savez(args.out_path, mu=mu, sigma=sigma)
    print(f"Saved reference stats to: {args.out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
