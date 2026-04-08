"""FID computation — uses torch-fidelity InceptionV3 (PyTorch, lazy import)."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def _fid_from_moments(mu1, sigma1, mu2, sigma2) -> float:
    """Compute FID from mean/covariance moments."""
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


def _compute_inception_moments_from_arr(arr: np.ndarray, batch_size: int, device: str):
    """Compute InceptionV3 2048-d moments from image array (PyTorch backend, CPU/GPU)."""
    import torch
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    x = arr
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {x.shape}")

    if x.shape[-1] == 3:  # NHWC → NCHW
        x = np.transpose(x, (0, 3, 1, 2))

    if x.dtype != np.uint8:
        x_f = x.astype(np.float32)
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0
        x = np.clip(x_f, 0, 255).astype(np.uint8)

    xt = torch.from_numpy(x).to(device=device, dtype=torch.uint8)
    fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=['2048']).to(device).eval()

    feats = []
    with torch.no_grad():
        for i in range(0, xt.shape[0], batch_size):
            batch = xt[i:i + batch_size]
            f = fe(batch)[0]
            feats.append(f.detach().cpu())

    feats = torch.cat(feats, dim=0).double().numpy()
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _compute_inception_moments_tf(arr: np.ndarray, batch_size: int):
    """Compute InceptionV3 2048-d moments using TensorFlow — runs on TPU natively.

    PyTorch (``torch_fidelity``) cannot execute on TPU hardware.  This backend
    leverages ``tf.keras.applications.InceptionV3`` which supports TPU via
    ``tf.distribute.TPUStrategy``.  The 2048-d average-pooled features are
    numerically equivalent to those produced by the PyTorch InceptionV3.
    """
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Let TF use TPU, not GPU

    # Detect and initialise TPU
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        # print("====== [DEBUG FID] Dùng: TENSORFLOW TPU backend ======")
    except Exception as e:
        strategy = tf.distribute.get_strategy()  # fallback: CPU / GPU
        # print(f"====== [DEBUG FID] Lỗi khi tạo TPU ({e}) — chạy TF trên default device (CPU/GPU) ======")

    # Prepare image array: ensure NHWC uint8
    x = arr
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {x.shape}")
    if x.shape[1] == 3 and x.shape[-1] != 3:  # NCHW → NHWC
        x = np.transpose(x, (0, 2, 3, 1))
    if x.dtype != np.uint8:
        x_f = x.astype(np.float32)
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0
        x = np.clip(x_f, 0, 255).astype(np.uint8)

    # Build InceptionV3 (average-pooled 2048-d output) under strategy scope
    with strategy.scope():
        base = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            pooling='avg',   # directly outputs (B, 2048)
        )

    @tf.function
    def extract_features(batch_f32):
        return base(batch_f32, training=False)

    feats_list = []
    n = len(x)
    for i in range(0, n, batch_size):
        batch = x[i:i + batch_size].astype(np.float32)
        # InceptionV3 expects inputs scaled to [-1, 1]
        batch = (batch / 127.5) - 1.0
        # Resize to 299×299 (InceptionV3 canonical input size)
        batch_tf = tf.image.resize(batch, [299, 299])
        feats = extract_features(batch_tf)
        feats_list.append(feats.numpy())

    feats = np.concatenate(feats_list, axis=0).astype(np.float64)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def calculate_gfid(
    arr1: np.ndarray,
    ref_arr: dict,
    batch_size: int = 64,
    device: str = "tpu",
) -> float:
    """FID: generated images vs reference statistics (mu, sigma).

    ``device`` controls which backend extracts InceptionV3 features:

    * ``'tpu'``  — TensorFlow/Keras backend with ``TPUStrategy`` (recommended
      on TPU hosts; automatically falls back to CPU if no TPU is detected).
    * ``'cuda'`` — PyTorch backend running on GPU.
    * ``'cpu'``  — PyTorch backend running on CPU (slowest).
    """
    mu_ref, sigma_ref = ref_arr["mu"], ref_arr["sigma"]

    if device == "tpu":
        try:
            mu_gen, sigma_gen = _compute_inception_moments_tf(arr1, batch_size)
        except Exception as e:
            print(f"====== [DEBUG FID] Backend TF bị sập ({e}); Fallback về PYTORCH CPU ======")
            mu_gen, sigma_gen = _compute_inception_moments_from_arr(arr1, batch_size, "cpu")
    else:
        # PyTorch backend (gpu or cpu)
        print(f"====== [DEBUG FID] Dùng: PYTORCH backend, thiết bị: {device.upper()} ======")
        mu_gen, sigma_gen = _compute_inception_moments_from_arr(arr1, batch_size, device)

    return _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)


def calculate_rfid(arr1, arr2=None, bs=64, device="cpu", fid_statistics_file=None):
    """Reconstruction FID between two image arrays."""
    from torch_fidelity import calculate_metrics

    from .utils import ImgArrDataset

    arr1_ds = ImgArrDataset(arr1)

    if fid_statistics_file is not None:
        metrics_kwargs = dict(
            input1=arr1_ds, input2=None,
            fid_statistics_file=fid_statistics_file,
            batch_size=bs, fid=True, cuda=(device != "cpu"),
        )
    else:
        if arr2 is None:
            raise ValueError("Either arr2 or fid_statistics_file must be provided.")
        arr2_ds = ImgArrDataset(arr2)
        metrics_kwargs = dict(
            input1=arr1_ds, input2=arr2_ds,
            batch_size=bs, fid=True, cuda=(device != "cpu"),
        )

    metrics = calculate_metrics(**metrics_kwargs)
    return metrics["frechet_inception_distance"]


class ImgArrDataset:
    """Torch Dataset wrapper for torch-fidelity: (B, H, W, C) uint8."""

    def __init__(self, arr: np.ndarray):
        import torch
        self.arr = arr
        self._torch = torch

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self._torch.from_numpy(self.arr[idx]).permute(2, 0, 1)
