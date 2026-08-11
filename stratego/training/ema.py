"""Exponential Moving Average (Polyak averaging) for model parameters.

Maintains a shadow copy of model parameters that tracks a smoothed average
over training. Used for all three networks (setup, move, belief) with
smoothing factor 0.999 as specified in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters (Polyak averaging).

    Maintains a shadow copy of model parameters. After each optimizer.step(),
    call ema.update(model) to update the shadow with smoothing factor 0.999.

    For evaluation, call ema.apply(model) to swap in EMA weights, then
    ema.restore(model, backup) to swap back.
    """

    def __init__(self, smoothing: float = 0.999) -> None:
        self.smoothing = smoothing
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, model: nn.Module) -> None:
        """Initialize shadow from model parameters.

        Args:
            model: The model to track. Automatically unwraps DDP wrappers.
        """
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow: shadow = smoothing * shadow + (1 - smoothing) * param.

        Args:
            model: The model whose current parameters to blend into the shadow.
        """
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].data.mul_(self.smoothing).add_(
                    param.data, alpha=(1.0 - self.smoothing)
                )

    def apply(self, model: nn.Module) -> None:
        """Swap EMA weights into model parameters.

        Call this before evaluation. Save a backup of original weights first
        if you need to restore them later.

        Args:
            model: The model to swap weights into.
        """
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original weights from a backup dict.

        Args:
            model: The model to restore weights into.
            backup: Dict mapping parameter names to their original tensors.
        """
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow parameters for checkpointing."""
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load shadow parameters from a checkpoint.

        Args:
            state: Dict of parameter name → tensor from a previous state_dict().
        """
        self.shadow = state
