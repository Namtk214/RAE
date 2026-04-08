"""Transport — flow matching training loss and sampling. JAX port."""

from __future__ import annotations

import enum
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import path
from .utils import mean_flat
from .integrators import ode, sde


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """Transport / Flow Matching framework."""

    def __init__(self, *, model_type, path_type, loss_type, time_dist_type,
                 time_dist_shift, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(self, train_eps, sample_eps, *, sde=False, eval=False,
                       reverse=False, last_step_size=0.0, diffusion_form="SBDM"):
        t0 = 0.0
        t1 = 1.0 - 1.0 / 1000

        eps = train_eps if not eval else sample_eps
        if isinstance(self.path_sampler, path.VPCPlan):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif isinstance(self.path_sampler, (path.ICPlan, path.GVPCPlan)) and \
                (self.model_type != ModelType.VELOCITY or sde):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample(self, x1, rng):
        """Sample x0 & t for training.

        Args:
            x1: data point (B, ...)
            rng: PRNG key

        Returns:
            t, x0, x1
        """
        rng_noise, rng_t = jax.random.split(rng)
        x0 = jax.random.normal(rng_noise, x1.shape, dtype=x1.dtype)

        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        dist_options = self.time_dist_type.split("_")
        if dist_options[0] == "uniform":
            t = jax.random.uniform(rng_t, (x1.shape[0],)) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            z = jax.random.normal(rng_t, (x1.shape[0],)) * sigma + mu
            t = jax.nn.sigmoid(z)
            t = jnp.clip(t, t0, t1)
        else:
            raise NotImplementedError(f"Unknown time distribution: {self.time_dist_type}")

        # Apply time distribution shift
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def training_losses(self, model_fn, x1, rng, has_aux=False, **model_kwargs):
        """Compute flow matching training loss.

        Args:
            model_fn: callable (xt, t, **kwargs) → predicted output or (output, aux)
            x1: data samples
            rng: PRNG key
            has_aux: whether model_fn returns aux data

        Returns:
            dict with 'loss' and 'pred' keys, optionally tuple (dict, aux) if has_aux
        """
        t, x0, x1 = self.sample(x1, rng)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        
        if has_aux:
            model_output, aux = model_fn(xt, t, **model_kwargs)
        else:
            model_output = model_fn(xt, t, **model_kwargs)
            aux = None

        terms = {"pred": model_output}
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat((model_output - ut) ** 2)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))

            if self.loss_type == WeightType.VELOCITY:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type == WeightType.LIKELIHOOD:
                weight = drift_var / (sigma_t ** 2)
            else:
                weight = 1

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * (model_output - x0) ** 2)
            else:
                terms["loss"] = mean_flat(weight * (model_output * sigma_t + x0) ** 2)

        return terms if not has_aux else (terms, aux)

    def get_drift(self):
        """Get drift function for ODE/SDE sampling."""
        if self.model_type == ModelType.VELOCITY:
            return lambda x, t, model, **kw: model(x, t, **kw)

        def noise_ode(x, t, model, **kw):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **kw)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        return noise_ode

    def get_score(self):
        if self.model_type == ModelType.VELOCITY:
            return lambda x, t, model, **kw: self.path_sampler.get_score_from_velocity(model(x, t, **kw), x, t)
        elif self.model_type == ModelType.NOISE:
            return lambda x, t, model, **kw: model(x, t, **kw) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        else:
            return lambda x, t, model, **kw: model(x, t, **kw)


class Sampler:
    """Sampler for the transport model (ODE/SDE)."""

    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = transport.get_drift()
        self.score = transport.get_score()

    def sample_ode(self, *, sampling_method="euler", num_steps=50,
                   atol=1e-6, rtol=1e-3, reverse=False):
        if reverse:
            drift = lambda x, t, model, **kw: self.drift(x, jnp.ones_like(t) * (1 - t), model, **kw)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )

        _ode = ode(
            drift=drift, t0=t0, t1=t1, sampler_type=sampling_method,
            num_steps=num_steps, atol=atol, rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )
        return _ode.sample
