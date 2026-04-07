"""ODE/SDE integrators — JAX port."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class sde:
    """SDE solver (Euler-Maruyama / Heun)."""

    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type, time_dist_shift):
        assert t0 < t1
        self.num_timesteps = num_steps
        t = np.linspace(t0, t1, num_steps)
        t = 1 - t
        t = time_dist_shift * t / (1 + (time_dist_shift - 1) * t)
        self.t = jnp.array(t, dtype=jnp.float32)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def _euler_step(self, x, mean_x, t_curr, t_next, model, rng, **model_kwargs):
        noise = jax.random.normal(rng, x.shape)
        dw = noise * jnp.sqrt(jnp.abs(t_curr - t_next))
        t_vec = jnp.full((x.shape[0],), t_curr)
        drift = self.drift(x, t_vec, model, **model_kwargs)
        diffusion = self.diffusion(x, t_vec)
        mean_x = x - drift * (t_curr - t_next)
        x = mean_x + jnp.sqrt(2 * diffusion) * dw
        return x, mean_x

    def sample(self, init, model, rng, **model_kwargs):
        x = init
        mean_x = init
        samples = []
        for i in range(len(self.t) - 1):
            rng, step_rng = jax.random.split(rng)
            x, mean_x = self._euler_step(x, mean_x, self.t[i], self.t[i + 1], model, step_rng, **model_kwargs)
            samples.append(x)
        return samples


class ode:
    """ODE solver (Euler / Heun)."""

    def __init__(self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol, time_dist_shift):
        assert t0 < t1
        self.drift = drift
        t = np.linspace(t0, t1, num_steps)
        t = 1 - t
        t = time_dist_shift * t / (1 + (time_dist_shift - 1) * t)
        self.t = jnp.array(t, dtype=jnp.float32)
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        """Simple Euler ODE solver."""
        for i in range(len(self.t) - 1):
            t_curr = self.t[i]
            t_next = self.t[i + 1]
            dt = t_next - t_curr
            t_vec = jnp.full((x.shape[0],), t_curr)
            v = self.drift(x, t_vec, model, **model_kwargs)
            x = x + v * dt
        return x
