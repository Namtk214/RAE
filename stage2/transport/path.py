"""Transport paths (coupling plans) — JAX port. Linear, VP, GVP."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def expand_t_like_x(t, x):
    """Reshape t: (B,) → (B, 1, ..., 1) to broadcast with x."""
    dims = [1] * (x.ndim - 1)
    return t.reshape(t.shape[0], *dims)


class ICPlan:
    """Linear Coupling Plan (Interpolant / Conditional OT)."""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        return 1 - t, -jnp.ones_like(t)

    def compute_sigma_t(self, t):
        return t, jnp.ones_like(t)

    def compute_d_alpha_alpha_ratio_t(self, t):
        return -1 / (1 - t)

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        if form == "constant":
            return norm
        elif form == "SBDM":
            return norm * self.compute_drift(x, t)[1]
        elif form == "sigma":
            return norm * self.compute_sigma_t(t)[0]
        elif form == "linear":
            return norm * t
        else:
            raise NotImplementedError(f"Diffusion form {form} not implemented")

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - x) / var
        return score

    def compute_mu_t(self, t, x0, x1):
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class VPCPlan(ICPlan):
    """VP (Variance Preserving) Coupling Plan."""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _log_mean_coeff(self, t):
        return -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min

    def _d_log_mean_coeff(self, t):
        return 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t):
        p = 2 * self._log_mean_coeff(t)
        alpha_t = jnp.sqrt(1 - jnp.exp(p))
        d_alpha_t = jnp.exp(p) * (2 * self._d_log_mean_coeff(t)) / (-2 * alpha_t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = jnp.exp(self._log_mean_coeff(t))
        d_sigma_t = sigma_t * self._d_log_mean_coeff(t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


class GVPCPlan(ICPlan):
    """General VP Coupling Plan (cosine schedule)."""

    def compute_alpha_t(self, t):
        alpha_t = jnp.cos(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * jnp.sin(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = jnp.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * jnp.cos(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return -np.pi / 2 * jnp.tan(t * np.pi / 2)
