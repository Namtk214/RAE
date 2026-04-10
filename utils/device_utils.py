"""TPU device mesh and sharding utilities for data parallelism."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


def create_mesh(axis_name: str = "data") -> Mesh:
    """Create a 1D data-parallel mesh over all available devices.

    For TPUv4e-8: 8 devices → Mesh(['data'], (8,))
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=(axis_name,))
    return mesh


def get_data_sharding(mesh: Mesh, axis_name: str = "data") -> NamedSharding:
    """Sharding for batch dimension (first axis = data axis)."""
    return NamedSharding(mesh, P(axis_name))


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding for replicated tensors (model params, etc.)."""
    return NamedSharding(mesh, P())


def shard_batch(batch: dict, mesh: Mesh, axis_name: str = "data") -> dict:
    """Shard a batch dict along the data axis — multi-host correct.

    On multi-host TPU (e.g. v4-16), each host reads only its local slice of
    the global batch (global_batch // num_hosts rows). JAX then assembles
    the per-host slices into one global sharded array via
    make_array_from_process_local_data.

    On single-host this degenerates to a simple device_put.
    """
    data_sharding = get_data_sharding(mesh, axis_name)
    num_processes = jax.process_count()

    def _shard(x):
        if not isinstance(x, (jnp.ndarray, np.ndarray)):
            return x
        x = np.asarray(x)
        if num_processes == 1:
            # Single-host fast path
            return jax.device_put(jnp.array(x), data_sharding)
        # Multi-host: x is the local slice (global_batch / num_hosts rows).
        # Use make_array_from_process_local_data so each host contributes
        # its own rows without needing to agree on the full global tensor.
        global_shape = (x.shape[0] * num_processes,) + x.shape[1:]
        return jax.make_array_from_process_local_data(data_sharding, x, global_shape)

    return jax.tree.map(_shard, batch)


def print_device_info():
    """Print device info for debugging."""
    devices = jax.devices()
    print(f"[DeviceUtils] {len(devices)} devices available:")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d}")
    print(f"  Process index: {jax.process_index()}")
    print(f"  Process count: {jax.process_count()}")


# Alias for backward compatibility
setup_mesh = create_mesh

