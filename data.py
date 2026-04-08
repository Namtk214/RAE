"""Data pipeline — supports TFDS (celebahq256) and ImageFolder for TPU training."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# TFDS-based dataset (CelebAHQ256, ImageNet, etc.)
# ---------------------------------------------------------------------------
def build_tfds_dataset(
    dataset_name: str = "celebahq256",
    split: str = "train",
    image_size: int = 256,
    batch_size: int = 512,
    seed: int = 0,
    data_dir: Optional[str] = None,
    tfds_builder_dir: Optional[str] = None,
    num_prefetch: int = 4,
    shuffle_buffer: int = 10000,
):
    """Build a tf.data pipeline from a TFDS dataset.

    Args:
        dataset_name: e.g. 'celebahq256', 'imagenet2012'
        split: 'train' or 'validation'
        image_size: Target image resolution
        batch_size: Global batch size (will be split across TPU cores)
        seed: Random seed
        data_dir: Where TFDS data is stored
        tfds_builder_dir: Path to custom TFDS builder directory
        num_prefetch: Number of batches to prefetch
        shuffle_buffer: Size of shuffle buffer

    Returns:
        Iterator yielding dicts with 'image' and 'label' numpy arrays.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # Suppress TF warnings on TPU
    tf.config.set_visible_devices([], 'GPU')

    # Load from custom builder dir if provided
    if tfds_builder_dir is not None:
        builder = tfds.builder_from_directory(tfds_builder_dir)
        ds = builder.as_dataset(split=split)
    else:
        ds = tfds.load(
            dataset_name,
            split=split,
            data_dir=data_dir,
            as_supervised=False,
        )

    def _preprocess(example):
        image = example["image"]
        # Ensure float32 in [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Resize if needed
        image = tf.image.resize(image, [image_size, image_size], method="bilinear")

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        label = example.get("label", 0)
        return {"image": image, "label": label}

    ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(num_prefetch)
    ds = ds.repeat()

    return iter(tfds.as_numpy(ds))


# ---------------------------------------------------------------------------
# ImageFolder-based dataset
# ---------------------------------------------------------------------------
def build_imagefolder_dataset(
    data_dir: str,
    image_size: int = 256,
    batch_size: int = 512,
    seed: int = 0,
    num_workers: int = 8,
    resize_size: int = 384,
    random_crop: bool = True,
):
    """Build a data pipeline from an ImageFolder directory structure.

    Expects: data_dir/class_name/image.jpg structure.
    Uses tf.data for TPU-efficient loading.

    Returns:
        Iterator yielding dicts with 'image' and 'label' numpy arrays.
    """
    import tensorflow as tf

    tf.config.set_visible_devices([], 'GPU')

    # Discover class directories
    data_path = Path(data_dir)
    class_dirs = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}

    # Collect all image paths and labels
    all_paths = []
    all_labels = []
    extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = data_path / cls_name
        for img_path in cls_dir.iterdir():
            if img_path.suffix in extensions:
                all_paths.append(str(img_path))
                all_labels.append(cls_idx)

    # Create tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices({
        "path": all_paths,
        "label": all_labels,
    })

    def _load_and_preprocess(example):
        path = example["path"]
        label = example["label"]

        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        if random_crop:
            # Resize to larger size first, then random crop
            image = tf.image.resize(image, [resize_size, resize_size], method="bicubic")
            image = tf.image.random_crop(image, [image_size, image_size, 3])
        else:
            # Center crop (ADM style)
            shape = tf.shape(image)
            h, w = shape[0], shape[1]
            min_dim = tf.minimum(h, w)
            scale = tf.cast(image_size, tf.float32) / tf.cast(min_dim, tf.float32)
            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, [new_h, new_w], method="bicubic")
            image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        return {"image": image, "label": label}

    ds = ds.shuffle(len(all_paths), seed=seed)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()

    return iter(ds.as_numpy_iterator())


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------
def build_dataset(
    source: str = "tfds",  # "tfds" or "imagefolder"
    dataset_name: str = "celebahq256",
    data_dir: Optional[str] = None,
    tfds_builder_dir: Optional[str] = None,
    split: str = "train",
    image_size: int = 256,
    batch_size: int = 512,
    seed: int = 0,
    num_workers: int = 8,
    **kwargs,
):
    """Unified dataset builder supporting both TFDS and ImageFolder.

    Returns:
        Iterator yielding {'image': (B, H, W, 3), 'label': (B,)} numpy arrays.
    """
    if source == "tfds":
        return build_tfds_dataset(
            dataset_name=dataset_name,
            split=split,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            data_dir=data_dir,
            tfds_builder_dir=tfds_builder_dir,
            **kwargs,
        )
    elif source == "imagefolder":
        if data_dir is None:
            raise ValueError("data_dir is required for ImageFolder source")
        return build_imagefolder_dataset(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown data source: {source}. Use 'tfds' or 'imagefolder'.")


# ---------------------------------------------------------------------------
# Dataloader wrapper — returns (iterator, steps_per_epoch)
# ---------------------------------------------------------------------------
def build_dataloader(
    data_path: str,
    image_size: int = 256,
    batch_size: int = 512,
    dataset_type: str = "imagefolder",
    split: str = "train",
    tfds_name: Optional[str] = None,
    tfds_builder_dir: Optional[str] = None,
    seed: int = 0,
    **kwargs,
) -> Tuple:
    """Build data iterator and compute steps_per_epoch.

    Args:
        data_path: Path to image folder OR tfds data_dir.
        image_size: Target image resolution.
        batch_size: Global batch size.
        dataset_type: 'imagefolder' or 'tfds'.
        split: 'train' or 'validation'.
        tfds_name: TFDS dataset name (e.g. 'celebahq256'). Used when dataset_type='tfds'.
        tfds_builder_dir: Path to custom TFDS builder directory.
        seed: Random seed.
        **kwargs: Extra args forwarded to underlying builder.

    Returns:
        (iterator, steps_per_epoch) tuple.
    """
    if dataset_type == "imagefolder":
        # Count images to compute steps_per_epoch
        data_dir = Path(data_path)
        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        num_images = sum(
            1 for cls_dir in data_dir.iterdir() if cls_dir.is_dir()
            for img in cls_dir.iterdir() if img.suffix in extensions
        )
        steps_per_epoch = num_images // batch_size

        it = build_imagefolder_dataset(
            data_dir=data_path,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        return it, steps_per_epoch

    elif dataset_type == "tfds":
        import tensorflow_datasets as tfds

        # Determine dataset size for steps_per_epoch
        if tfds_builder_dir is not None:
            builder = tfds.builder_from_directory(tfds_builder_dir)
        else:
            builder = tfds.builder(tfds_name or "celebahq256", data_dir=data_path)

        info = builder.info
        num_images = info.splits[split].num_examples
        steps_per_epoch = num_images // batch_size

        it = build_tfds_dataset(
            dataset_name=tfds_name or "celebahq256",
            split=split,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            data_dir=data_path,
            tfds_builder_dir=tfds_builder_dir,
            **kwargs,
        )
        return it, steps_per_epoch

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'imagefolder' or 'tfds'.")

