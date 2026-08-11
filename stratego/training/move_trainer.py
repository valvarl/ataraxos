"""Trainer for the move network (arXiv:2511.07312).

Implements PPO with magnet KL regularization for training the move network.
Paper hyperparameters: 1 epoch, 202 batches per iteration, Adam optimizer
with lr=clip(0.5/iter^1.1, 5e-6, 1e-4), grad clip 0.267, EMA 0.999,
magnet KL coefficient alpha=0.05/iter^0.3.
"""

from __future__ import annotations

import torch

from stratego.constants import (
    MOVE_EMA_SMOOTHING,
    MOVE_KL_COEFF,
    MOVE_LR_EXPONENT,
    MOVE_LR_MAX,
    MOVE_LR_MIN,
    MOVE_LR_NUMERATOR,
    MOVE_MAGNET_KL_EXPONENT,
    MOVE_MAGNET_KL_NUMERATOR,
    MOVE_MAX_GRAD_NORM,
    MOVE_PPO_CLIP,
)
from stratego.networks.move_net import MoveNetwork
from stratego.training.ema import EMA
from stratego.training.losses import move_loss

__all__ = ["MoveTrainer"]


class MoveTrainer:
    """Trains the move network with PPO + magnet KL regularization.

    Paper hyperparams: 1 epoch, 202 batches, Adam lr=clip(0.5/iter^1.1, 5e-6, 1e-4),
    grad clip 0.267, EMA 0.999, alpha=0.05/iter^0.3.

    Parameters
    ----------
    model:
        The MoveNetwork to train.
    lr_max:
        Maximum learning rate (ceiling of the clip). Default: MOVE_LR_MAX (1e-4).
    grad_norm:
        Maximum gradient norm for clipping. Default: MOVE_MAX_GRAD_NORM (0.267).
    ema_smoothing:
        EMA smoothing factor. Default: MOVE_EMA_SMOOTHING (0.999).
    ppo_clip:
        PPO clipping range. Default: MOVE_PPO_CLIP (0.2).
    kl_coeff:
        KL divergence coefficient to data-collection policy. Default: MOVE_KL_COEFF (0.1).
    """

    def __init__(
        self,
        model: MoveNetwork,
        lr_max: float = MOVE_LR_MAX,
        grad_norm: float = MOVE_MAX_GRAD_NORM,
        ema_smoothing: float = MOVE_EMA_SMOOTHING,
        ppo_clip: float = MOVE_PPO_CLIP,
        kl_coeff: float = MOVE_KL_COEFF,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
        self.ema = EMA(ema_smoothing)
        self.ema.register(model)
        self.grad_norm = grad_norm
        self.ppo_clip = ppo_clip
        self.kl_coeff = kl_coeff
        self.iteration: int = 0

    def get_lr(self, iter_num: int) -> float:
        """LR schedule: clip(0.5 / iter^1.1, 5e-6, 1e-4).

        Args:
            iter_num: Current training iteration (1-indexed).

        Returns:
            Clipped learning rate for the current iteration.
        """
        if iter_num < 1:
            return float(MOVE_LR_MAX)
        raw = MOVE_LR_NUMERATOR / (iter_num ** MOVE_LR_EXPONENT)
        return float(max(MOVE_LR_MIN, min(MOVE_LR_MAX, raw)))

    def get_magnet_kl_coeff(self, iter_num: int) -> float:
        """Magnet KL coefficient: alpha = 0.05 / iter^0.3.

        Args:
            iter_num: Current training iteration (1-indexed).

        Returns:
            Magnet KL coefficient for the current iteration.
        """
        if iter_num < 1:
            return float(MOVE_MAGNET_KL_NUMERATOR)
        return float(MOVE_MAGNET_KL_NUMERATOR / (iter_num ** MOVE_MAGNET_KL_EXPONENT))

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on a batch of move data.

        Args:
            batch: Dict with keys 'infostate', 'target_move_idx', 'advantages',
                'outcome_probs', 'old_policy_probs', 'magnet_probs', and
                optionally 'legal_move_mask'.

        Returns:
            Dict of scalar metric values from move_loss plus 'lr' and
            'magnet_kl_coeff'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        value, policy = self.model(batch["infostate"], batch.get("legal_move_mask"))

        # Compute new policy probs for played move
        policy_probs = torch.softmax(policy, dim=-1)
        new_move_probs = policy_probs.gather(
            1, batch["target_move_idx"].unsqueeze(-1)
        ).squeeze(-1)

        # Compute loss
        magnet_kl_coeff = self.get_magnet_kl_coeff(self.iteration)
        loss, metrics = move_loss(
            value_logits=value,
            policy_logits=policy,
            target_move_idx=batch["target_move_idx"],
            advantages=batch["advantages"],
            outcome_probs=batch["outcome_probs"],
            old_policy_probs=batch["old_policy_probs"],
            new_policy_probs=new_move_probs,
            magnet_probs=batch["magnet_probs"],
            ppo_clip=self.ppo_clip,
            kl_coeff=self.kl_coeff,
            magnet_kl_coeff=magnet_kl_coeff,
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        # Update LR for this iteration
        lr = self.get_lr(self.iteration)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.step()
        self.ema.update(self.model)
        self.iteration += 1

        metrics["lr"] = lr
        metrics["magnet_kl_coeff"] = magnet_kl_coeff
        return metrics

    def get_eval_model(self) -> MoveNetwork:
        """Return model with EMA weights applied.

        Swaps EMA shadow weights into the model in-place. Call before
        evaluation. The caller is responsible for restoring original weights
        if continued training is needed.
        """
        self.ema.apply(self.model)
        return self.model
