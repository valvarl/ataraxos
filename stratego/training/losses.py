"""Loss functions for setup, move, and belief networks (arXiv:2511.07312).

Setup: L_setup = L_π + 0.5·L_v + L_h
Move:   L_move   = L_π + L_v
Belief: L_belief = -log P(target | input)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

__all__ = ["belief_loss", "move_loss", "setup_loss"]

_EPS = 1e-8


def setup_loss(
    value_logits: torch.Tensor,
    entropy_pred: torch.Tensor,
    policy_logits: torch.Tensor,  # noqa: ARG001 — reserved for future use
    target_outcome: torch.Tensor,
    target_next_piece: torch.Tensor,
    advantages: torch.Tensor,
    conditional_entropy: torch.Tensor,
    old_policy_probs: torch.Tensor,
    new_policy_probs: torch.Tensor,
    ppo_clip: float = 0.2,
    kl_coeff: float = 0.1,
    value_coeff: float = 0.5,
    entropy_coeff: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_setup = L_π + 0.5·L_v + L_h."""
    B, S, C = new_policy_probs.shape

    # PPO ratio at played piece positions → (B, S)
    old_p = old_policy_probs.gather(-1, target_next_piece.unsqueeze(-1)).squeeze(-1)
    new_p = new_policy_probs.gather(-1, target_next_piece.unsqueeze(-1)).squeeze(-1)
    r = new_p / (old_p + _EPS)
    clipped_r = r.clamp(1 - ppo_clip, 1 + ppo_clip)
    adv = advantages.unsqueeze(-1).expand_as(r)
    policy_loss = -torch.min(r * adv, clipped_r * adv).mean()

    # KL to data-collection policy (forward KL, per-position then mean)
    min_S = min(new_policy_probs.size(1), old_policy_probs.size(1))
    np_p = new_policy_probs[:, :min_S]
    op_p = old_policy_probs[:, :min_S]
    log_ratio = (np_p + _EPS).log() - (op_p + _EPS).log()
    kl = (np_p * log_ratio).sum(-1).mean()
    policy_loss = policy_loss + kl_coeff * kl

    # Value loss: cross-entropy with scalar outcome → class index
    # win=1→0, loss=-1→1, draw=0→2
    target_idx = torch.zeros_like(target_outcome, dtype=torch.long)
    target_idx[target_outcome == -1] = 1
    target_idx[target_outcome == 0] = 2
    value_loss = F.cross_entropy(value_logits, target_idx)

    # Conditional entropy prediction loss (MSE, normalised by /10)
    entropy_loss = ((conditional_entropy / 10.0 - entropy_pred) ** 2).mean()

    total = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "kl": kl.item(),
        "total": total.item(),
    }


def move_loss(
    value_logits: torch.Tensor,
    policy_logits: torch.Tensor,  # noqa: ARG001 — reserved for future use
    target_move_idx: torch.Tensor,  # noqa: ARG001 — reserved for future use
    advantages: torch.Tensor,
    outcome_probs: torch.Tensor,
    old_policy_probs: torch.Tensor,
    new_policy_probs: torch.Tensor,
    magnet_probs: torch.Tensor,
    ppo_clip: float = 0.2,
    kl_coeff: float = 0.1,
    magnet_kl_coeff: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_move = L_π + L_v."""
    # PPO ratio (scalar per batch element)
    r = new_policy_probs / (old_policy_probs + _EPS)
    clipped_r = r.clamp(1 - ppo_clip, 1 + ppo_clip)
    policy_loss = -torch.min(r * advantages, clipped_r * advantages).mean()

    # KL to data-collection policy
    log_ratio = (new_policy_probs + _EPS).log() - (old_policy_probs + _EPS).log()
    kl_policy = (new_policy_probs * log_ratio).mean()
    policy_loss = policy_loss + kl_coeff * kl_policy

    # KL to magnet policy
    log_ratio_m = (new_policy_probs + _EPS).log() - (magnet_probs + _EPS).log()
    kl_magnet = (new_policy_probs * log_ratio_m).mean()
    policy_loss = policy_loss + magnet_kl_coeff * kl_magnet

    # Value loss: cross-entropy with outcome probs (soft labels)
    value_loss = F.cross_entropy(value_logits, outcome_probs)

    total = policy_loss + value_loss
    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_policy": kl_policy.item(),
        "kl_magnet": kl_magnet.item(),
        "total": total.item(),
    }


def belief_loss(
    pred_logits: torch.Tensor,
    target_pieces: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of ground-truth hidden piece types."""
    B, S, C = pred_logits.shape
    return F.cross_entropy(pred_logits.reshape(B * S, C), target_pieces.reshape(B * S))
