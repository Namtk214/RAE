"""Transport utility functions — JAX port."""

import jax.numpy as jnp


class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def mean_flat(x: jnp.ndarray) -> jnp.ndarray:
    """Take the mean over all non-batch dimensions."""
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def log_state(state):
    result = []
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    return '\n'.join(result)
