"""Optimizer and scheduler utilities using optax."""

from __future__ import annotations

from typing import Optional

import optax


def build_optimizer(
    lr: float = 2e-4,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.0,
    clip_grad: float = 0.0,
) -> optax.GradientTransformation:
    """Build AdamW optimizer with optional gradient clipping.

    Matches PyTorch: AdamW(lr, betas, weight_decay).
    """
    chain = []

    if clip_grad > 0:
        chain.append(optax.clip_by_global_norm(clip_grad))

    chain.append(optax.adamw(
        learning_rate=lr,
        b1=betas[0],
        b2=betas[1],
        weight_decay=weight_decay,
    ))

    return optax.chain(*chain) if len(chain) > 1 else chain[0]


def build_optimizer_with_schedule(
    lr: float = 2e-4,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.0,
    clip_grad: float = 0.0,
    schedule_type: str = "cosine",
    warmup_steps: int = 0,
    total_steps: int = 100000,
    final_lr: float = 2e-5,
    warmup_from_zero: bool = True,
) -> optax.GradientTransformation:
    """Build optimizer with LR schedule.

    Supports: cosine, linear, constant.
    """
    # Build LR schedule
    if schedule_type == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0 if warmup_from_zero else lr,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=final_lr,
        )
    elif schedule_type == "linear":
        if warmup_steps > 0:
            schedule = optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0 if warmup_from_zero else lr,
                        end_value=lr,
                        transition_steps=warmup_steps,
                    ),
                    optax.linear_schedule(
                        init_value=lr,
                        end_value=final_lr,
                        transition_steps=total_steps - warmup_steps,
                    ),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.linear_schedule(lr, final_lr, total_steps)
    elif schedule_type == "constant":
        if warmup_steps > 0:
            schedule = optax.join_schedules(
                [
                    optax.linear_schedule(0.0 if warmup_from_zero else lr, lr, warmup_steps),
                    optax.constant_schedule(lr),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.constant_schedule(lr)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Build optimizer chain
    chain = []
    if clip_grad > 0:
        chain.append(optax.clip_by_global_norm(clip_grad))

    chain.append(optax.adamw(
        learning_rate=schedule,
        b1=betas[0],
        b2=betas[1],
        weight_decay=weight_decay,
    ))

    return optax.chain(*chain) if len(chain) > 1 else chain[0]


def get_lr_from_opt_state(opt_state, step: int, schedule_fn=None) -> float:
    """Extract current learning rate."""
    if schedule_fn is not None:
        return float(schedule_fn(step))
    return 0.0
