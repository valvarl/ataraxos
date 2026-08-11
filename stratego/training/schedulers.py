"""Power-law annealing schedulers for learning rate and regularization temperature.

Implements the hyperparameter schedules from the Ataraxos paper (arXiv:2511.07312):
- Setup regularization temperature: α = 0.1 / iter^0.3
- Move regularization temperature (magnet KL): α = 0.05 / iter^0.3
- Move network LR: clip(0.5 / iter^1.1, 5e-6, 1e-4)
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def setup_reg_temp(iteration: int, numerator: float = 0.1, exponent: float = 0.3) -> float:
    """Setup network regularization temperature: α = numerator / iter^exponent.

    Args:
        iteration: Current training iteration (1-indexed).
        numerator: Schedule numerator (default 0.1 from paper).
        exponent: Power-law exponent (default 0.3 from paper).

    Returns:
        Regularization temperature for the current iteration.
    """
    if iteration < 1:
        return numerator  # avoid div-by-zero on first iteration
    return float(numerator / (iteration**exponent))


def move_reg_temp(iteration: int, numerator: float = 0.05, exponent: float = 0.3) -> float:
    """Move network reverse-KL-to-magnet temperature: α = numerator / iter^exponent.

    Args:
        iteration: Current training iteration (1-indexed).
        numerator: Schedule numerator (default 0.05 from paper).
        exponent: Power-law exponent (default 0.3 from paper).

    Returns:
        Regularization temperature for the current iteration.
    """
    if iteration < 1:
        return numerator
    return float(numerator / (iteration**exponent))


def move_lr_value(
    iteration: int,
    numerator: float = 0.5,
    exponent: float = 1.1,
    lr_min: float = 5e-6,
    lr_max: float = 1e-4,
) -> float:
    """Move network LR with power-law decay and clipping.

    lr = clip(numerator / iter^exponent, lr_min, lr_max)

    Args:
        iteration: Current training iteration (1-indexed).
        numerator: Schedule numerator (default 0.5 from paper).
        exponent: Power-law exponent (default 1.1 from paper).
        lr_min: Minimum learning rate floor (default 5e-6).
        lr_max: Maximum learning rate ceiling (default 1e-4).

    Returns:
        Clipped learning rate for the current iteration.
    """
    if iteration < 1:
        return lr_max
    raw = float(numerator / (iteration**exponent))
    return max(lr_min, min(lr_max, raw))


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    numerator: float,
    exponent: float,
    lr_min: float,
    lr_max: float,
) -> LambdaLR:
    """Create a PyTorch LambdaLR scheduler with power-law annealing.

    The lambda function computes: clip(numerator / step^exponent, lr_min, lr_max) / lr_max
    so that LambdaLR multiplies the base_lr (= lr_max) by this factor.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        numerator: Power-law numerator.
        exponent: Power-law exponent.
        lr_min: Minimum learning rate.
        lr_max: Maximum learning rate (should match optimizer's base_lr).

    Returns:
        A LambdaLR scheduler instance.
    """

    def lr_lambda(step: int) -> float:
        if step < 1:
            return 1.0
        raw = float(numerator / (step**exponent))
        clipped = max(lr_min, min(lr_max, raw))
        return float(clipped / lr_max)

    return LambdaLR(optimizer, lr_lambda)
