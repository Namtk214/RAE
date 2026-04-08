"""FID computation — JAX-native InceptionV3.

Ported from shortcut-models/utils/fid.py (https://github.com/ericl122333/shortcut-models)
which in turn adapts https://github.com/matthias-wright/jax-fid/

Key design decisions (matching shortcut-models):
  - InceptionV3 runs fully in JAX (NHWC), no PyTorch/TF dependency for FID
  - Input range: [-1, 1] float32 NHWC
  - Output: [B, 1, 1, 2048] → use [..., 0, 0, :] to get [B, 2048]
  - Pretrained weights downloaded from Dropbox (pickle format)
  - fid_from_stats has eye*1e-6 regularization to avoid sqrtm instability
"""

from __future__ import annotations

import functools
import os
import pickle
import tempfile
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import requests
import scipy
from tqdm import tqdm

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any


# ---------------------------------------------------------------------------
# Pretrained weight download
# ---------------------------------------------------------------------------
INCEPTION_WEIGHTS_URL = (
    "https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1"
)


def download(url: str, ckpt_dir: str = "data") -> str:
    """Download file from *url* to *ckpt_dir* (skip if already present)."""
    name = url[url.rfind("/") + 1 : url.rfind("?")]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading InceptionV3 weights from "{url[:url.rfind("?")]}"...')
        os.makedirs(ckpt_dir, exist_ok=True)

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))
        pbar = tqdm(total=total, unit="iB", unit_scale=True)

        tmp_path = ckpt_file + ".temp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                pbar.update(len(chunk))
                f.write(chunk)
        pbar.close()

        if total and pbar.n != total:
            print("Download incomplete — please retry.")
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, ckpt_file)
    return ckpt_file


def _get(d: Optional[dict], key: str) -> Optional[dict]:
    if d is None or key not in d:
        return None
    return d[key]


# ---------------------------------------------------------------------------
# JAX InceptionV3 — NHWC convention
# ---------------------------------------------------------------------------

class BatchNorm(nn.Module):
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones
    mean_init: Callable = lambda s: jnp.zeros(s, jnp.float32)  # noqa: E731
    var_init: Callable = lambda s: jnp.ones(s, jnp.float32)    # noqa: E731
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        from flax.linen.module import merge_param
        from jax import lax

        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = tuple(x.ndim + a if a < 0 else a for a in axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        initializing = self.is_mutable_collection("params")
        ra_mean = self.variable("batch_stats", "mean", self.mean_init, reduced_feature_shape)
        ra_var  = self.variable("batch_stats", "var",  self.var_init,  reduced_feature_shape)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis)
            if self.axis_name is not None and not initializing:
                concatenated = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(concatenated, axis_name=self.axis_name,
                              axis_index_groups=self.axis_index_groups), 2
                )
            var = mean2 - lax.square(mean)
            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value  = self.momentum * ra_var.value  + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = jax.lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param("scale", self.scale_init, reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param("bias", self.bias_init, reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)


def _pool(inputs, init, reduce_fn, window_shape, strides, padding):
    strides = strides or (1,) * len(window_shape)
    strides_full = (1,) + strides + (1,)
    dims = (1,) + window_shape + (1,)
    is_single = inputs.ndim == len(dims) - 1
    if is_single:
        inputs = inputs[None]
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        padding = ((0, 0),) + padding + ((0, 0),)
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides_full, padding)
    if is_single:
        y = jnp.squeeze(y, axis=0)
    return y


def avg_pool(inputs, window_shape, strides=None, padding="VALID"):
    assert inputs.ndim == 4
    y = _pool(inputs, 0.0, jax.lax.add, window_shape, strides, padding)
    ones = jnp.ones((1, inputs.shape[1], inputs.shape[2], 1), dtype=inputs.dtype)
    counts = jax.lax.conv_general_dilated(
        ones,
        jnp.expand_dims(jnp.ones(window_shape, inputs.dtype), axis=(-2, -1)),
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=nn.linear._conv_dimension_numbers(ones.shape),
        feature_group_count=1,
    )
    return y / counts


class BasicConv2d(nn.Module):
    out_channels: int
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    strides: Optional[Iterable[int]] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = "valid"
    use_bias: bool = False
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.lecun_normal() if pd is None else lambda *_: jnp.array(pd["conv"]["kernel"]),
            bias_init=nn.initializers.zeros if pd is None else lambda *_: jnp.array(pd["conv"]["bias"]),
            dtype=self.dtype,
        )(x)
        if pd is None:
            x = BatchNorm(epsilon=0.001, momentum=0.1, use_running_average=not train, dtype=self.dtype)(x)
        else:
            x = BatchNorm(
                epsilon=0.001, momentum=0.1,
                bias_init=lambda *_: jnp.array(pd["bn"]["bias"]),
                scale_init=lambda *_: jnp.array(pd["bn"]["scale"]),
                mean_init=lambda _: jnp.array(pd["bn"]["mean"]),
                var_init=lambda _: jnp.array(pd["bn"]["var"]),
                use_running_average=not train,
                dtype=self.dtype,
            )(x)
        return jax.nn.relu(x)


class InceptionA(nn.Module):
    pool_features: int
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b1 = BasicConv2d(64, (1,1), params_dict=_get(pd,"branch1x1"), dtype=self.dtype)(x, train)
        b5 = BasicConv2d(48, (1,1), params_dict=_get(pd,"branch5x5_1"), dtype=self.dtype)(x, train)
        b5 = BasicConv2d(64, (5,5), padding=((2,2),(2,2)), params_dict=_get(pd,"branch5x5_2"), dtype=self.dtype)(b5, train)
        b3 = BasicConv2d(64, (1,1), params_dict=_get(pd,"branch3x3dbl_1"), dtype=self.dtype)(x, train)
        b3 = BasicConv2d(96, (3,3), padding=((1,1),(1,1)), params_dict=_get(pd,"branch3x3dbl_2"), dtype=self.dtype)(b3, train)
        b3 = BasicConv2d(96, (3,3), padding=((1,1),(1,1)), params_dict=_get(pd,"branch3x3dbl_3"), dtype=self.dtype)(b3, train)
        bp = avg_pool(x, (3,3), strides=(1,1), padding=((1,1),(1,1)))
        bp = BasicConv2d(self.pool_features, (1,1), params_dict=_get(pd,"branch_pool"), dtype=self.dtype)(bp, train)
        return jnp.concatenate([b1, b5, b3, bp], axis=-1)


class InceptionB(nn.Module):
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b3 = BasicConv2d(384, (3,3), strides=(2,2), params_dict=_get(pd,"branch3x3"), dtype=self.dtype)(x, train)
        bd = BasicConv2d(64, (1,1), params_dict=_get(pd,"branch3x3dbl_1"), dtype=self.dtype)(x, train)
        bd = BasicConv2d(96, (3,3), padding=((1,1),(1,1)), params_dict=_get(pd,"branch3x3dbl_2"), dtype=self.dtype)(bd, train)
        bd = BasicConv2d(96, (3,3), strides=(2,2), params_dict=_get(pd,"branch3x3dbl_3"), dtype=self.dtype)(bd, train)
        bp = nn.max_pool(x, (3,3), strides=(2,2))
        return jnp.concatenate([b3, bd, bp], axis=-1)


class InceptionC(nn.Module):
    channels_7x7: int
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        c = self.channels_7x7
        b1 = BasicConv2d(192, (1,1), params_dict=_get(pd,"branch1x1"), dtype=self.dtype)(x, train)
        b7 = BasicConv2d(c,   (1,1), params_dict=_get(pd,"branch7x7_1"), dtype=self.dtype)(x, train)
        b7 = BasicConv2d(c,   (1,7), padding=((0,0),(3,3)), params_dict=_get(pd,"branch7x7_2"), dtype=self.dtype)(b7, train)
        b7 = BasicConv2d(192, (7,1), padding=((3,3),(0,0)), params_dict=_get(pd,"branch7x7_3"), dtype=self.dtype)(b7, train)
        bd = BasicConv2d(c,   (1,1), params_dict=_get(pd,"branch7x7dbl_1"), dtype=self.dtype)(x, train)
        bd = BasicConv2d(c,   (7,1), padding=((3,3),(0,0)), params_dict=_get(pd,"branch7x7dbl_2"), dtype=self.dtype)(bd, train)
        bd = BasicConv2d(c,   (1,7), padding=((0,0),(3,3)), params_dict=_get(pd,"branch7x7dbl_3"), dtype=self.dtype)(bd, train)
        bd = BasicConv2d(c,   (7,1), padding=((3,3),(0,0)), params_dict=_get(pd,"branch7x7dbl_4"), dtype=self.dtype)(bd, train)
        bd = BasicConv2d(192, (1,7), padding=((0,0),(3,3)), params_dict=_get(pd,"branch7x7dbl_5"), dtype=self.dtype)(bd, train)
        bp = avg_pool(x, (3,3), strides=(1,1), padding=((1,1),(1,1)))
        bp = BasicConv2d(192, (1,1), params_dict=_get(pd,"branch_pool"), dtype=self.dtype)(bp, train)
        return jnp.concatenate([b1, b7, bd, bp], axis=-1)


class InceptionD(nn.Module):
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b3 = BasicConv2d(192, (1,1), params_dict=_get(pd,"branch3x3_1"), dtype=self.dtype)(x, train)
        b3 = BasicConv2d(320, (3,3), strides=(2,2), params_dict=_get(pd,"branch3x3_2"), dtype=self.dtype)(b3, train)
        b7 = BasicConv2d(192, (1,1), params_dict=_get(pd,"branch7x7x3_1"), dtype=self.dtype)(x, train)
        b7 = BasicConv2d(192, (1,7), padding=((0,0),(3,3)), params_dict=_get(pd,"branch7x7x3_2"), dtype=self.dtype)(b7, train)
        b7 = BasicConv2d(192, (7,1), padding=((3,3),(0,0)), params_dict=_get(pd,"branch7x7x3_3"), dtype=self.dtype)(b7, train)
        b7 = BasicConv2d(192, (3,3), strides=(2,2), params_dict=_get(pd,"branch7x7x3_4"), dtype=self.dtype)(b7, train)
        bp = nn.max_pool(x, (3,3), strides=(2,2))
        return jnp.concatenate([b3, b7, bp], axis=-1)


class InceptionE(nn.Module):
    pooling: Callable
    params_dict: Optional[dict] = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        pd = self.params_dict
        b1 = BasicConv2d(320, (1,1), params_dict=_get(pd,"branch1x1"), dtype=self.dtype)(x, train)
        b3 = BasicConv2d(384, (1,1), params_dict=_get(pd,"branch3x3_1"), dtype=self.dtype)(x, train)
        b3a = BasicConv2d(384, (1,3), padding=((0,0),(1,1)), params_dict=_get(pd,"branch3x3_2a"), dtype=self.dtype)(b3, train)
        b3b = BasicConv2d(384, (3,1), padding=((1,1),(0,0)), params_dict=_get(pd,"branch3x3_2b"), dtype=self.dtype)(b3, train)
        b3 = jnp.concatenate([b3a, b3b], axis=-1)
        bd = BasicConv2d(448, (1,1), params_dict=_get(pd,"branch3x3dbl_1"), dtype=self.dtype)(x, train)
        bd = BasicConv2d(384, (3,3), padding=((1,1),(1,1)), params_dict=_get(pd,"branch3x3dbl_2"), dtype=self.dtype)(bd, train)
        bda = BasicConv2d(384, (1,3), padding=((0,0),(1,1)), params_dict=_get(pd,"branch3x3dbl_3a"), dtype=self.dtype)(bd, train)
        bdb = BasicConv2d(384, (3,1), padding=((1,1),(0,0)), params_dict=_get(pd,"branch3x3dbl_3b"), dtype=self.dtype)(bd, train)
        bd = jnp.concatenate([bda, bdb], axis=-1)
        bp = self.pooling(x, window_shape=(3,3), strides=(1,1), padding=((1,1),(1,1)))
        bp = BasicConv2d(192, (1,1), params_dict=_get(pd,"branch_pool"), dtype=self.dtype)(bp, train)
        return jnp.concatenate([b1, b3, bd, bp], axis=-1)


class InceptionV3(nn.Module):
    """InceptionV3 for FID computation. Input: NHWC float32 in [-1, 1].
    Output: [B, 1, 1, 2048] (global avg pool, keepdims=True).
    Use [..., 0, 0, :] to get [B, 2048] activations.
    """
    include_head: bool = False
    pretrained: bool = False
    ckpt_path: str = INCEPTION_WEIGHTS_URL
    dtype: str = "float32"
    params_dict: Optional[dict] = None

    def setup(self):
        if self.pretrained:
            ckpt_file = download(self.ckpt_path)
            self._params_dict = pickle.load(open(ckpt_file, "rb"))
        else:
            self._params_dict = self.params_dict

    @nn.compact
    def __call__(self, x, train=False):
        pd = self._params_dict
        x = BasicConv2d(32,  (3,3), strides=(2,2), params_dict=_get(pd,"Conv2d_1a_3x3"), dtype=self.dtype)(x, train)
        x = BasicConv2d(32,  (3,3), params_dict=_get(pd,"Conv2d_2a_3x3"), dtype=self.dtype)(x, train)
        x = BasicConv2d(64,  (3,3), padding=((1,1),(1,1)), params_dict=_get(pd,"Conv2d_2b_3x3"), dtype=self.dtype)(x, train)
        x = nn.max_pool(x, (3,3), strides=(2,2))
        x = BasicConv2d(80,  (1,1), params_dict=_get(pd,"Conv2d_3b_1x1"), dtype=self.dtype)(x, train)
        x = BasicConv2d(192, (3,3), params_dict=_get(pd,"Conv2d_4a_3x3"), dtype=self.dtype)(x, train)
        x = nn.max_pool(x, (3,3), strides=(2,2))
        x = InceptionA(32,  params_dict=_get(pd,"Mixed_5b"), dtype=self.dtype)(x, train)
        x = InceptionA(64,  params_dict=_get(pd,"Mixed_5c"), dtype=self.dtype)(x, train)
        x = InceptionA(64,  params_dict=_get(pd,"Mixed_5d"), dtype=self.dtype)(x, train)
        x = InceptionB(params_dict=_get(pd,"Mixed_6a"), dtype=self.dtype)(x, train)
        x = InceptionC(128, params_dict=_get(pd,"Mixed_6b"), dtype=self.dtype)(x, train)
        x = InceptionC(160, params_dict=_get(pd,"Mixed_6c"), dtype=self.dtype)(x, train)
        x = InceptionC(160, params_dict=_get(pd,"Mixed_6d"), dtype=self.dtype)(x, train)
        x = InceptionC(192, params_dict=_get(pd,"Mixed_6e"), dtype=self.dtype)(x, train)
        x = InceptionD(params_dict=_get(pd,"Mixed_7a"), dtype=self.dtype)(x, train)
        x = InceptionE(avg_pool,     params_dict=_get(pd,"Mixed_7b"), dtype=self.dtype)(x, train)
        x = InceptionE(nn.max_pool,  params_dict=_get(pd,"Mixed_7c"), dtype=self.dtype)(x, train)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)  # [B, 1, 1, 2048]
        return x


# ---------------------------------------------------------------------------
# Public API — matching shortcut-models interface
# ---------------------------------------------------------------------------

def get_fid_network(ckpt_dir: str = "data"):
    """Build and JIT-compile the InceptionV3 feature extractor.

    Returns a callable `apply_fn(images)` where:
      - images: NHWC float32 in [-1, 1], any size (will be resized to 299x299)
      - returns: [B, 1, 1, 2048]

    Usage (mirrors shortcut-models):
        get_fid_activations = get_fid_network()
        acts = get_fid_activations(images)[..., 0, 0, :]  # [B, 2048]
    """
    model = InceptionV3(pretrained=True, ckpt_path=INCEPTION_WEIGHTS_URL)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 299, 299, 3))
    params = model.init(rng, dummy, train=False)
    apply_fn = jax.jit(functools.partial(model.apply, train=False))
    apply_fn = functools.partial(apply_fn, params)
    return apply_fn


def fid_from_stats(mu1, sigma1, mu2, sigma2) -> float:
    """Compute FID from (mu, sigma) moments.

    Matches shortcut-models: adds eye*1e-6 offset before sqrtm to avoid
    numerical instability with near-singular covariance matrices.
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * 1e-6
    covmean, _ = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
    covmean = np.real(covmean)
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(fid, 0.0)


def preprocess_for_inception(images: np.ndarray) -> jnp.ndarray:
    """Prepare image array for InceptionV3.

    Accepts:
      - NHWC uint8 [0, 255]
      - NHWC float32 [0, 1]
      - NCHW float32 [0, 1]

    Returns: NHWC float32 resized to 299x299, scaled to [-1, 1].
    """
    x = np.asarray(images)
    # NCHW → NHWC
    if x.ndim == 4 and x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (0, 2, 3, 1))
    # uint8 or float [0,1] → float [-1, 1]
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 127.5 - 1.0
    else:
        x = x.astype(np.float32)
        if x.max() > 1.5:   # [0, 255] float → [-1, 1]
            x = x / 127.5 - 1.0
        else:               # [0, 1] → [-1, 1]
            x = x * 2.0 - 1.0
    # Resize to 299x299
    x_jax = jnp.array(x)
    x_jax = jax.image.resize(x_jax, (x_jax.shape[0], 299, 299, x_jax.shape[3]),
                              method="bilinear", antialias=False)
    return jnp.clip(x_jax, -1.0, 1.0)


def compute_fid_activations(
    images: np.ndarray,
    fid_fn,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract InceptionV3 2048-d activations from image array.

    Args:
        images: NHWC uint8 or float32
        fid_fn: callable from get_fid_network()
        batch_size: batch size for feature extraction

    Returns:
        activations: np.ndarray of shape [N, 2048]
    """
    n = len(images)
    feats = []
    pbar = tqdm(range(0, n, batch_size), desc="Extracting InceptionV3 features", leave=False)
    for i in pbar:
        batch = images[i : i + batch_size]
        batch_jax = preprocess_for_inception(batch)
        acts = fid_fn(batch_jax)[..., 0, 0, :]  # [B, 2048]
        feats.append(np.array(acts))
    return np.concatenate(feats, axis=0)


def moments_from_activations(activations: np.ndarray):
    """Compute (mu, sigma) from [N, 2048] activations."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


# ---------------------------------------------------------------------------
# Reconstruction FID (kept for backward compat — uses torch_fidelity)
# ---------------------------------------------------------------------------

def calculate_rfid(arr1, arr2=None, bs=64, device="cpu", fid_statistics_file=None):
    """Reconstruction FID between two image arrays (uses torch_fidelity)."""
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


# Keep old calculate_gfid as alias for backward compat
def calculate_gfid(arr1: np.ndarray, ref_arr: dict, batch_size: int = 64, **kwargs) -> float:
    """Legacy wrapper — now uses JAX InceptionV3 internally."""
    fid_fn = get_fid_network()
    acts = compute_fid_activations(arr1, fid_fn, batch_size)
    mu_gen, sigma_gen = moments_from_activations(acts)
    return fid_from_stats(mu_gen, sigma_gen, ref_arr["mu"], ref_arr["sigma"])
