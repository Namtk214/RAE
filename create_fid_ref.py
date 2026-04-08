"""Script to compute FID reference statistics (mu, sigma) for a dataset.

Uses the SAME TF Keras InceptionV3 backend as eval/fid.py to ensure
reference stats and generated-image stats are computed in the same feature space.
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
from data import build_dataloader


def _compute_inception_moments_tf(images_uint8: np.ndarray, batch_size: int) -> tuple:
    """Compute InceptionV3 features using TF Keras — same as eval/fid.py."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("Using TF TPU backend for InceptionV3.")
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print(f"TPU not found ({e}), falling back to TF CPU/GPU backend.")

    # Ensure NHWC uint8
    x = images_uint8
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:   # NCHW -> NHWC
        x = np.transpose(x, (0, 2, 3, 1))

    with strategy.scope():
        base = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            pooling='avg',
        )

    @tf.function
    def extract_features(batch_f32):
        return base(batch_f32, training=False)

    feats_list = []
    n = len(x)
    pbar = tqdm(range(0, n, batch_size), desc="Extracting InceptionV3 features")
    for i in pbar:
        batch = x[i:i + batch_size].astype(np.float32)
        batch = (batch / 127.5) - 1.0                       # scale to [-1, 1]
        batch_tf = tf.image.resize(batch, [299, 299])       # resize to 299x299
        feats = extract_features(batch_tf)
        feats_list.append(feats.numpy())
        pbar.set_postfix({"processed": min(i + batch_size, n)})

    feats = np.concatenate(feats_list, axis=0).astype(np.float64)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID reference statistics (mu, sigma) using TF Keras InceptionV3."
    )
    parser.add_argument("--data-path",        type=str, required=True)
    parser.add_argument("--out-path",         type=str, required=True)
    parser.add_argument("--image-size",       type=int, default=256)
    parser.add_argument("--batch-size",       type=int, default=128)
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

    # Collect raw images into memory
    all_images = []
    processed = 0
    print(f"Collecting up to {args.num_samples} images...")
    for step_data in ds:
        if processed >= args.num_samples:
            break
        imgs = np.array(step_data["image"])
        # Normalise to uint8 NHWC
        if imgs.dtype != np.uint8:
            if imgs.max() <= 2.0:
                imgs = imgs * 255.0
            imgs = np.clip(imgs, 0, 255).astype(np.uint8)
        if imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[-1] != 3:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        all_images.append(imgs)
        processed += imgs.shape[0]
        print(f"  Collected {processed}/{args.num_samples}", end="\r")

    print()
    all_images = np.concatenate(all_images, axis=0)[:args.num_samples]
    print(f"Total collected: {all_images.shape}")

    print("Extracting InceptionV3 features via TF Keras (same backend as FID eval)...")
    mu, sigma = _compute_inception_moments_tf(all_images, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    np.savez(args.out_path, mu=mu, sigma=sigma)
    print(f"Saved reference stats to: {args.out_path}")
    print(f"  mu shape: {mu.shape}, sigma shape: {sigma.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
