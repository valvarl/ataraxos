"""λ-returns computation, Monte Carlo returns, and advantage filtering.

Implements the return estimation primitives for the Ataraxos training pipeline:

- **TD(λ) returns**: G^λ_t = r_t + (1-λ)v(x_{t+1}) + λG^λ_{t+1}  (γ=1, episodic)
- **Monte Carlo returns**: final game outcome broadcast to all setup steps
- **Advantage estimation**: δ = G^λ - baseline
- **Advantage filtering**: top-quantile AND magnitude threshold (paper §move-learning)
- **Outcome probability estimation**: λ-return over vector-valued predictions
- **Expected value**: E[v] = P(win) - P(loss)

Reference: methods.tex — move net uses λ=0.5 for advantage, λ=0.8 for outcome probs.
"""

from __future__ import annotations

import torch


def lambda_return(
    values: torch.Tensor,
    rewards: torch.Tensor,
    lambda_: float,
    terminal_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute TD(λ) return via backward recursion.

    Formula (γ=1, no discounting — Stratego is episodic):
        G^λ_t = r_t + (1-λ)·v(x_{t+1}) + λ·G^λ_{t+1}

    Args:
        values: (T,) predicted values [v(x_0), v(x_1), ..., v(x_{T-1})].
        rewards: (T,) rewards [r_0, r_1, ..., r_{T-1}]. Typically 0 except terminal.
        lambda_: λ parameter (0.5 for advantage, 0.8 for outcome).
        terminal_value: scalar tensor or None. If provided, G^λ_{T-1} = terminal_value.

    Returns:
        returns: (T,) λ-return G^λ_t for each timestep t.
    """
    t_len = values.shape[0]
    returns = torch.empty_like(values)

    # Base case: last timestep
    if terminal_value is not None:
        returns[-1] = terminal_value
    else:
        returns[-1] = rewards[-1]

    # Backward recursion
    one_minus_lam = 1.0 - lambda_
    for t in range(t_len - 2, -1, -1):
        returns[t] = rewards[t] + one_minus_lam * values[t + 1] + lambda_ * returns[t + 1]

    return returns


def advantage(returns: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Compute advantage δ = G^λ - baseline.

    Args:
        returns: (T,) λ-returns.
        baseline: (T,) or scalar baseline (typically E[v_θ_t(x)]).

    Returns:
        δ: (T,) advantage estimates.
    """
    return returns - baseline


def monte_carlo_return(outcome: float, num_steps: int) -> torch.Tensor:
    """MC return for setup network: final game outcome broadcast to all steps.

    Args:
        outcome: game outcome (win=1, loss=-1, draw=0).
        num_steps: number of setup steps (40 piece placements).

    Returns:
        returns: (num_steps,) all equal to outcome.
    """
    return torch.full((num_steps,), float(outcome))


def outcome_probs_from_lambda_return(
    value_sequence: torch.Tensor,
    terminal_outcome: int | None,
    lambda_: float = 0.8,
) -> torch.Tensor:
    """Estimate outcome probabilities (ξ) via λ-return over vector values.

    For the move network: ξ = λ-return with λ=0.8 over {v_θ_t(x')}
    (and o as one-hot if game finished).

    Args:
        value_sequence: (T, 3) — predicted [win, loss, draw] probabilities at each step.
        terminal_outcome: one-hot encoded outcome if game finished (0=win, 1=loss, 2=draw),
            else None for ongoing games.
        lambda_: λ=0.8 for outcome estimation.

    Returns:
        outcome_probs: (3,) — estimated [P(win), P(loss), P(draw)].
    """
    t_len, num_components = value_sequence.shape
    rewards = torch.zeros_like(value_sequence)

    # Build terminal one-hot if game finished
    terminal_value: torch.Tensor | None = None
    if terminal_outcome is not None:
        terminal_value = torch.zeros(num_components, dtype=value_sequence.dtype, device=value_sequence.device)
        terminal_value[terminal_outcome] = 1.0

    # Compute λ-return for each component independently
    result = torch.empty(num_components, dtype=value_sequence.dtype, device=value_sequence.device)
    for k in range(num_components):
        tv_k: torch.Tensor | None = None
        if terminal_value is not None:
            tv_k = terminal_value[k]
        g = lambda_return(value_sequence[:, k], rewards[:, k], lambda_=lambda_, terminal_value=tv_k)
        result[k] = g[0]

    return result


def filter_advantages(
    advantages: torch.Tensor,
    quantile: float = 0.75,
    magnitude_threshold: float = 0.01,
) -> torch.Tensor:
    """Filter moves by advantage magnitude for training.

    From paper: include a move in training data if BOTH:
    1. |δ| is in the top (1-quantile) fraction by magnitude (top 25% when quantile=0.75)
    2. |δ| >= magnitude_threshold (0.01)

    The effective threshold is max(quantile_threshold, magnitude_threshold).

    Args:
        advantages: (N,) advantage estimates.
        quantile: quantile cutoff (0.75 → keep top 25% by magnitude).
        magnitude_threshold: minimum |δ| to include (0.01).

    Returns:
        mask: (N,) boolean tensor — True for moves to include in training.
    """
    abs_adv = advantages.abs()
    quantile_thresh = torch.quantile(abs_adv, quantile)
    mag_thresh = torch.tensor(magnitude_threshold, dtype=abs_adv.dtype, device=abs_adv.device)
    effective_thresh = torch.maximum(quantile_thresh, mag_thresh)
    return abs_adv >= effective_thresh


def expected_value(value_probs: torch.Tensor) -> torch.Tensor:
    """E[v] = P(win)·1 + P(loss)·(-1) + P(draw)·0 = P(win) - P(loss).

    Args:
        value_probs: (..., 3) — [P(win), P(loss), P(draw)].

    Returns:
        expected: (...) — scalar expected outcome (win=1, loss=-1, draw=0).
    """
    return value_probs[..., 0] - value_probs[..., 1]


__all__ = [
    "advantage",
    "expected_value",
    "filter_advantages",
    "lambda_return",
    "monte_carlo_return",
    "outcome_probs_from_lambda_return",
]
