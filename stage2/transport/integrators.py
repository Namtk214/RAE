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

    def sample(self, init, model, rng, return_intermediates=False, log_every=10, **model_kwargs):
        """Euler-Maruyama SDE solver using jax.lax.scan for TPU efficiency."""
        x = init
        mean_x = init
        t = self.t
        num_steps = len(t) - 1

        if not return_intermediates:
            # Fast path: only return final state, no intermediate storage
            def step_fn(carry, i):
                x, mean_x, rng = carry
                t_curr = t[i]
                t_next = t[i + 1]
                rng, step_rng = jax.random.split(rng)
                noise = jax.random.normal(step_rng, x.shape)
                dw = noise * jnp.sqrt(jnp.abs(t_curr - t_next))
                t_vec = jnp.full((x.shape[0],), t_curr)
                drift = self.drift(x, t_vec, model, **model_kwargs)
                diffusion = self.diffusion(x, t_vec)
                mean_x = x - drift * (t_curr - t_next)
                x = mean_x + jnp.sqrt(2 * diffusion) * dw
                return (x, mean_x, rng), None  # None = don't accumulate output

            (x_final, _, _), _ = jax.lax.scan(
                step_fn, (x, mean_x, rng), jnp.arange(num_steps)
            )
            return x_final
        else:
            # Slow path: collect intermediates (only for visualization)
            def step_fn_full(carry, i):
                x, mean_x, rng = carry
                t_curr = t[i]
                t_next = t[i + 1]
                rng, step_rng = jax.random.split(rng)
                noise = jax.random.normal(step_rng, x.shape)
                dw = noise * jnp.sqrt(jnp.abs(t_curr - t_next))
                t_vec = jnp.full((x.shape[0],), t_curr)
                drift = self.drift(x, t_vec, model, **model_kwargs)
                diffusion = self.diffusion(x, t_vec)
                mean_x = x - drift * (t_curr - t_next)
                x = mean_x + jnp.sqrt(2 * diffusion) * dw
                store = x if (i + 1) % log_every == 0 else jnp.zeros_like(x)
                return (x, mean_x, rng), store

            (x_final, _, _), xs = jax.lax.scan(
                step_fn_full, (x, mean_x, rng), jnp.arange(num_steps)
            )
            indices = [i for i in range(num_steps) if (i + 1) % log_every == 0]
            intermediates = [xs[i] for i in indices]
            if num_steps % log_every != 0:
                intermediates.append(x_final)
            return x_final, [init] + intermediates


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

    def sample(self, x, model, return_intermediates=False, log_every=10, **model_kwargs):
        """Euler ODE solver using jax.lax.scan for TPU efficiency."""
        t = self.t
        num_steps = len(t) - 1

        if not return_intermediates:
            # Fast path: scan with no output accumulation → minimal memory
            def step_fn(carry, i):
                x = carry
                t_curr = t[i]
                t_next = t[i + 1]
                dt = t_next - t_curr
                t_vec = jnp.full((x.shape[0],), t_curr)
                v = self.drift(x, t_vec, model, **model_kwargs)
                x = x + v * dt
                return x, None  # None = don't accumulate

            x_final, _ = jax.lax.scan(step_fn, x, jnp.arange(num_steps))
            return x_final
        else:
            # Slow path: collect intermediates every log_every steps
            def step_fn_full(carry, i):
                x = carry
                t_curr = t[i]
                t_next = t[i + 1]
                dt = t_next - t_curr
                t_vec = jnp.full((x.shape[0],), t_curr)
                v = self.drift(x, t_vec, model, **model_kwargs)
                x = x + v * dt
                store = x if (i + 1) % log_every == 0 else jnp.zeros_like(x)
                return x, store

            x_final, xs = jax.lax.scan(step_fn_full, x, jnp.arange(num_steps))
            indices = [i for i in range(num_steps) if (i + 1) % log_every == 0]
            intermediates = [xs[i] for i in indices]
            if num_steps % log_every != 0:
                intermediates.append(x_final)
            return x_final, [x] + intermediates
