"""Belief network trainer: log-likelihood training on self-play games.

Implements the training loop for the belief network (arXiv:2511.07312).
The belief network predicts opponent hidden piece types from the infostate
using an encoder-decoder transformer. Training uses negative log-likelihood
loss, dropout 0.2 for OOD generalization (e.g. human opponents), and Polyak
(EMA) averaging with smoothing 0.999.

Paper: 4 H100s × 4 days of training on self-play data.
"""

from __future__ import annotations

import torch

from stratego.networks.belief_net import BeliefNetwork
from stratego.training.ema import EMA
from stratego.training.losses import belief_loss

__all__ = ["BeliefTrainer"]


class BeliefTrainer:
    """Trains the belief network to predict opponent hidden piece types.

    Paper: log-likelihood loss, dropout 0.2, train on self-play games
    of final setup+move nets. 4 H100s × 4 days.
    """

    def __init__(
        self,
        model: BeliefNetwork,
        lr: float = 1e-4,
        grad_norm: float = 1.0,
        ema_smoothing: float = 0.999,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ema = EMA(ema_smoothing)
        self.ema.register(model)
        self.grad_norm = grad_norm

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on a batch of belief data.

        batch keys: infostate (B,488,10,10), hidden_mask (B,100) bool,
                    target_tokens (B,S) int — ground truth piece types
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(batch["infostate"], batch["hidden_mask"], batch["target_tokens"])

        # Compute loss
        loss = belief_loss(logits, batch["target_tokens"])

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.ema.update(self.model)

        return {"loss": loss.item()}

    def train_epoch(
        self,
        data: list[dict[str, torch.Tensor]],
        batch_size: int = 256,
    ) -> list[dict[str, float]]:
        """Train for one epoch over all data."""
        results: list[dict[str, float]] = []
        for i in range(0, len(data), batch_size):
            batch = self._collate(data[i : i + batch_size])
            results.append(self.train_step(batch))
        return results

    def _collate(self, batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack list of batch dicts into single batch dict."""
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}

    def get_eval_model(self) -> BeliefNetwork:
        """Return model with EMA weights for evaluation."""
        self.ema.apply(self.model)
        return self.model
