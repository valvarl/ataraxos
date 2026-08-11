"""Trainer for the setup network (arXiv:2511.07312).

Implements PPO with entropy regularization for training the autoregressive setup
network. Paper hyperparameters: 5 epochs, batch 1024/GPU, Adam lr=5e-5,
grad clip 0.5, EMA 0.999, alpha=0.1/iter^0.3.
"""

from __future__ import annotations

import torch

from stratego.constants import (
    SETUP_BATCH_SIZE_PER_GPU,
    SETUP_EMA_SMOOTHING,
    SETUP_ENTROPY_COEFF,
    SETUP_KL_COEFF,
    SETUP_LR,
    SETUP_MAX_GRAD_NORM,
    SETUP_PPO_CLIP,
    SETUP_VALUE_COEFF,
)
from stratego.networks.setup_net import SetupNetwork
from stratego.training.ema import EMA
from stratego.training.losses import setup_loss

__all__ = ["SetupTrainer"]


class SetupTrainer:
    """Trains the setup network with PPO + entropy regularization.

    Paper hyperparams: 5 epochs, batch 1024/GPU, Adam lr=5e-5,
    grad clip 0.5, EMA 0.999, alpha=0.1/iter^0.3.

    Parameters
    ----------
    model:
        The SetupNetwork to train.
    lr:
        Adam learning rate. Default: ``SETUP_LR`` (5e-5).
    grad_norm:
        Max gradient norm for clipping. Default: ``SETUP_MAX_GRAD_NORM`` (0.5).
    ema_smoothing:
        EMA smoothing factor. Default: ``SETUP_EMA_SMOOTHING`` (0.999).
    ppo_clip:
        PPO clip ratio epsilon. Default: ``SETUP_PPO_CLIP`` (0.2).
    kl_coeff:
        KL penalty coefficient. Default: ``SETUP_KL_COEFF`` (0.1).
    value_coeff:
        Value loss coefficient. Default: ``SETUP_VALUE_COEFF`` (0.5).
    entropy_coeff:
        Entropy loss coefficient. Default: ``SETUP_ENTROPY_COEFF`` (1.0).
    """

    def __init__(
        self,
        model: SetupNetwork,
        lr: float = SETUP_LR,
        grad_norm: float = SETUP_MAX_GRAD_NORM,
        ema_smoothing: float = SETUP_EMA_SMOOTHING,
        ppo_clip: float = SETUP_PPO_CLIP,
        kl_coeff: float = SETUP_KL_COEFF,
        value_coeff: float = SETUP_VALUE_COEFF,
        entropy_coeff: float = SETUP_ENTROPY_COEFF,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ema = EMA(ema_smoothing)
        self.ema.register(model)
        self.grad_norm = grad_norm
        self.ppo_clip = ppo_clip
        self.kl_coeff = kl_coeff
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.iteration = 0
        self._backup: dict[str, torch.Tensor] | None = None

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on a batch of setup data.

        Args:
            batch: Dict with keys ``tokens``, ``target_outcome``,
                ``target_next_piece``, ``advantages``, ``conditional_entropy``,
                and ``old_policy_probs``.

        Returns:
            Dict of loss metrics (policy_loss, value_loss, entropy_loss,
            kl, total).
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        value, entropy, policy = self.model(batch["tokens"])

        # Compute loss
        loss, metrics = setup_loss(
            value_logits=value,
            entropy_pred=entropy,
            policy_logits=policy,
            target_outcome=batch["target_outcome"],
            target_next_piece=batch["target_next_piece"],
            advantages=batch["advantages"],
            conditional_entropy=batch["conditional_entropy"],
            old_policy_probs=batch["old_policy_probs"],
            new_policy_probs=torch.softmax(policy, dim=-1),
            ppo_clip=self.ppo_clip,
            kl_coeff=self.kl_coeff,
            value_coeff=self.value_coeff,
            entropy_coeff=self.entropy_coeff,
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.ema.update(self.model)

        self.iteration += 1
        return metrics

    def train_epoch(
        self,
        data: list[dict[str, torch.Tensor]],
        batch_size: int = SETUP_BATCH_SIZE_PER_GPU,
    ) -> list[dict[str, float]]:
        """Train for one epoch over all data.

        Args:
            data: List of batch dicts, each with tensor values.
            batch_size: Number of examples per gradient step.

        Returns:
            List of metric dicts, one per gradient step.
        """
        results: list[dict[str, float]] = []
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            metrics = self.train_step(self._collate(chunk))
            results.append(metrics)
        return results

    def _collate(
        self,
        batches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Stack list of batch dicts into single batch dict."""
        return {
            k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]
        }

    def get_eval_model(self) -> SetupNetwork:
        """Return model with EMA weights for evaluation.

        Saves a backup of current weights before swapping in EMA weights.
        Call ``restore_model`` after evaluation to swap back.
        """
        self._backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        self.ema.apply(self.model)
        return self.model

    def restore_model(self) -> None:
        """Restore original weights saved by ``get_eval_model``."""
        backup = self._backup
        if backup is None:
            raise RuntimeError(
                "restore_model() called without get_eval_model()"
            )
        self.ema.restore(self.model, backup)
        self._backup = None
