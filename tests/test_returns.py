"""Tests for stratego.training.returns — λ-returns, MC returns, advantage filtering."""

from __future__ import annotations

import pytest
import torch

from stratego.training.returns import (
    advantage,
    expected_value,
    filter_advantages,
    lambda_return,
    monte_carlo_return,
    outcome_probs_from_lambda_return,
)

# ---------------------------------------------------------------------------
# λ-return tests
# ---------------------------------------------------------------------------


class TestLambdaReturn:
    """Tests for the lambda_return function."""

    def test_single_step_with_terminal(self) -> None:
        """Single-step episode with terminal reward → return = terminal value."""
        values = torch.tensor([0.5])
        rewards = torch.tensor([0.0])
        result = lambda_return(values, rewards, lambda_=0.5, terminal_value=torch.tensor(1.0))
        assert result.shape == (1,)
        torch.testing.assert_close(result, torch.tensor([1.0]))

    def test_single_step_no_terminal(self) -> None:
        """Single-step episode without terminal → return = last reward."""
        values = torch.tensor([0.5])
        rewards = torch.tensor([0.7])
        result = lambda_return(values, rewards, lambda_=0.5)
        torch.testing.assert_close(result, torch.tensor([0.7]))

    def test_two_step_lambda_zero_td0(self) -> None:
        """Two-step, λ=0 → return[0] = r_0 + v(x_1) (TD(0))."""
        values = torch.tensor([0.3, 0.6])
        rewards = torch.tensor([0.1, 0.2])
        result = lambda_return(values, rewards, lambda_=0.0)
        # G_1 = r_1 = 0.2
        # G_0 = r_0 + (1-0)*v_1 + 0*G_1 = 0.1 + 0.6 = 0.7
        expected = torch.tensor([0.7, 0.2])
        torch.testing.assert_close(result, expected)

    def test_two_step_lambda_one_mc(self) -> None:
        """Two-step, λ=1 → return[0] = r_0 + r_1 (Monte Carlo)."""
        values = torch.tensor([0.3, 0.6])
        rewards = torch.tensor([0.1, 0.2])
        result = lambda_return(values, rewards, lambda_=1.0)
        # G_1 = r_1 = 0.2
        # G_0 = r_0 + (1-1)*v_1 + 1*G_1 = 0.1 + 0 + 0.2 = 0.3
        expected = torch.tensor([0.3, 0.2])
        torch.testing.assert_close(result, expected)

    def test_two_step_lambda_half(self) -> None:
        """Two-step, λ=0.5 → return[0] = r_0 + 0.5*v_1 + 0.5*r_1."""
        values = torch.tensor([0.3, 0.6])
        rewards = torch.tensor([0.1, 0.2])
        result = lambda_return(values, rewards, lambda_=0.5)
        # G_1 = r_1 = 0.2
        # G_0 = r_0 + 0.5*v_1 + 0.5*G_1 = 0.1 + 0.3 + 0.1 = 0.5
        expected = torch.tensor([0.5, 0.2])
        torch.testing.assert_close(result, expected)

    def test_zero_rewards_with_terminal(self) -> None:
        """Zero rewards except terminal → returns blend values and terminal."""
        values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        rewards = torch.zeros(5)
        terminal = torch.tensor(1.0)
        result = lambda_return(values, rewards, lambda_=0.5, terminal_value=terminal)
        # G_4 = 1.0 (terminal)
        # G_3 = 0 + 0.5*0.5 + 0.5*1.0 = 0.75
        # G_2 = 0 + 0.5*0.4 + 0.5*0.75 = 0.575
        # G_1 = 0 + 0.5*0.3 + 0.5*0.575 = 0.4375
        # G_0 = 0 + 0.5*0.2 + 0.5*0.4375 = 0.31875
        expected = torch.tensor([0.31875, 0.4375, 0.575, 0.75, 1.0])
        torch.testing.assert_close(result, expected)
        # Last step must equal terminal
        torch.testing.assert_close(result[-1], terminal)

    def test_backward_recursion_5step(self) -> None:
        """Backward recursion correctness on 5-step episode."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rewards = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        lam = 0.5
        result = lambda_return(values, rewards, lambda_=lam)
        # G_4 = r_4 = 0.5
        # G_3 = 0.4 + 0.5*5.0 + 0.5*0.5 = 3.15
        # G_2 = 0.3 + 0.5*4.0 + 0.5*3.15 = 3.875
        # G_1 = 0.2 + 0.5*3.0 + 0.5*3.875 = 3.6375
        # G_0 = 0.1 + 0.5*2.0 + 0.5*3.6375 = 2.91875
        expected = torch.tensor([2.91875, 3.6375, 3.875, 3.15, 0.5])
        torch.testing.assert_close(result, expected)

    def test_terminal_overrides_last_reward(self) -> None:
        """When terminal_value is provided, last step return = terminal_value (not r_{T-1})."""
        values = torch.tensor([0.5, 0.6])
        rewards = torch.tensor([0.0, 99.0])  # r_1 = 99 but should be overridden
        result = lambda_return(values, rewards, lambda_=0.5, terminal_value=torch.tensor(1.0))
        assert result[-1].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Advantage tests
# ---------------------------------------------------------------------------


class TestAdvantage:
    """Tests for the advantage function."""

    def test_advantage_basic(self) -> None:
        """advantage = return - baseline."""
        returns = torch.tensor([1.0, 0.5, -0.3])
        baseline = torch.tensor([0.8, 0.6, -0.1])
        result = advantage(returns, baseline)
        expected = torch.tensor([0.2, -0.1, -0.2])
        torch.testing.assert_close(result, expected)

    def test_advantage_scalar_baseline(self) -> None:
        """Scalar baseline broadcasts correctly."""
        returns = torch.tensor([1.0, 0.5, -0.3])
        baseline = torch.tensor(0.5)
        result = advantage(returns, baseline)
        expected = torch.tensor([0.5, 0.0, -0.8])
        torch.testing.assert_close(result, expected)

    def test_advantage_positive_negative_zero(self) -> None:
        """Positive, negative, and zero advantages."""
        returns = torch.tensor([1.0, 0.0, 0.5])
        baseline = torch.tensor([0.5, 0.5, 0.5])
        result = advantage(returns, baseline)
        assert result[0].item() > 0  # positive
        assert result[1].item() < 0  # negative
        assert result[2].item() == pytest.approx(0.0)  # zero


# ---------------------------------------------------------------------------
# Monte Carlo return tests
# ---------------------------------------------------------------------------


class TestMonteCarloReturn:
    """Tests for the monte_carlo_return function."""

    def test_mc_return_win(self) -> None:
        """MC return for win: all steps = 1."""
        result = monte_carlo_return(outcome=1.0, num_steps=40)
        assert result.shape == (40,)
        assert torch.all(result == 1.0)

    def test_mc_return_loss(self) -> None:
        """MC return for loss: all steps = -1."""
        result = monte_carlo_return(outcome=-1.0, num_steps=40)
        assert result.shape == (40,)
        assert torch.all(result == -1.0)

    def test_mc_return_draw(self) -> None:
        """MC return for draw: all steps = 0."""
        result = monte_carlo_return(outcome=0.0, num_steps=40)
        assert result.shape == (40,)
        assert torch.all(result == 0.0)


# ---------------------------------------------------------------------------
# Outcome probs from λ-return tests
# ---------------------------------------------------------------------------


class TestOutcomeProbsFromLambdaReturn:
    """Tests for outcome_probs_from_lambda_return."""

    def test_single_step_terminal_red_win(self) -> None:
        """Single-step, terminal → probs = one-hot(red_win)."""
        value_seq = torch.tensor([[0.3, 0.4, 0.3]])
        result = outcome_probs_from_lambda_return(value_seq, terminal_outcome=0, lambda_=0.8)
        expected = torch.tensor([1.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_single_step_terminal_blue_win(self) -> None:
        """Single-step, terminal → probs = one-hot(blue_win)."""
        value_seq = torch.tensor([[0.3, 0.4, 0.3]])
        result = outcome_probs_from_lambda_return(value_seq, terminal_outcome=1, lambda_=0.8)
        expected = torch.tensor([0.0, 1.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_single_step_terminal_draw(self) -> None:
        """Single-step, terminal → probs = one-hot(draw)."""
        value_seq = torch.tensor([[0.3, 0.4, 0.3]])
        result = outcome_probs_from_lambda_return(value_seq, terminal_outcome=2, lambda_=0.8)
        expected = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_multi_step_lambda_zero(self) -> None:
        """Multi-step, λ=0 → probs = value prediction at step 1 (TD(0) bootstrap)."""
        value_seq = torch.tensor([
            [0.3, 0.4, 0.3],
            [0.6, 0.2, 0.2],
        ])
        result = outcome_probs_from_lambda_return(value_seq, terminal_outcome=0, lambda_=0.0)
        # λ=0: G_1 = one_hot(0) = [1,0,0]
        # G_0 = (1-0)*v_1 + 0*G_1 = v_1 = [0.6, 0.2, 0.2]
        expected = torch.tensor([0.6, 0.2, 0.2])
        torch.testing.assert_close(result, expected)

    def test_multi_step_lambda_one_terminal(self) -> None:
        """Multi-step, λ=1, terminal → probs = terminal one-hot (MC)."""
        value_seq = torch.tensor([
            [0.3, 0.4, 0.3],
            [0.6, 0.2, 0.2],
            [0.5, 0.3, 0.2],
        ])
        result = outcome_probs_from_lambda_return(value_seq, terminal_outcome=0, lambda_=1.0)
        # λ=1: all returns = terminal one-hot = [1, 0, 0]
        expected = torch.tensor([1.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_multi_step_no_terminal(self) -> None:
        """Multi-step, no terminal → λ-return with last step = reward (0)."""
        value_seq = torch.tensor([
            [0.3, 0.4, 0.3],
            [0.6, 0.2, 0.2],
        ])
        result = outcome_probs_from_lambda_return(value_seq, terminal_outcome=None, lambda_=0.5)
        # No terminal: G_1 = r_1 = 0 (last reward, no bootstrap)
        # G_0 = r_0 + (1-0.5)*v_1 + 0.5*G_1 = 0 + 0.5*v_1 + 0 = 0.5*v_1
        # Component 0: 0.5*0.6 = 0.3, Component 1: 0.5*0.2 = 0.1, Component 2: 0.5*0.2 = 0.1
        expected = torch.tensor([0.3, 0.1, 0.1])
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Advantage filtering tests
# ---------------------------------------------------------------------------


class TestFilterAdvantages:
    """Tests for filter_advantages."""

    def test_top_25_percent_selected(self) -> None:
        """Top 25% by magnitude selected when all > 0.01."""
        # 100 values from 0.01 to 1.00
        advantages = torch.arange(1, 101, dtype=torch.float32) / 100.0
        mask = filter_advantages(advantages, quantile=0.75, magnitude_threshold=0.01)
        # 75th percentile ≈ 0.7525, so values >= 0.76 are selected → 25 values
        assert mask.sum().item() == 25
        # All selected values should be >= 0.76
        assert torch.all(advantages[mask] >= 0.75)

    def test_magnitude_threshold_excludes_small(self) -> None:
        """Moves with |δ| < 0.01 excluded even if in top 25%."""
        # All values very small — top 25% by magnitude still < 0.01
        advantages = torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008])
        mask = filter_advantages(advantages, quantile=0.75, magnitude_threshold=0.01)
        # max(quantile_threshold, 0.01) = 0.01, and all |δ| < 0.01 → none selected
        assert mask.sum().item() == 0

    def test_all_below_threshold_none_selected(self) -> None:
        """When all moves have |δ| < 0.01, none selected."""
        advantages = torch.tensor([0.001, 0.002, 0.003, 0.004])
        mask = filter_advantages(advantages, quantile=0.75, magnitude_threshold=0.01)
        assert mask.sum().item() == 0

    def test_quantile_vs_magnitude_max_applies(self) -> None:
        """The max of quantile threshold and magnitude threshold is used."""
        # Quantile threshold will be ~0.005 (below 0.01), magnitude threshold = 0.01
        advantages = torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008])
        mask = filter_advantages(advantages, quantile=0.75, magnitude_threshold=0.01)
        # max(quantile, 0.01) = 0.01 → none selected since all < 0.01
        assert mask.sum().item() == 0

        # Now with some values above 0.01
        advantages2 = torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.02])
        mask2 = filter_advantages(advantages2, quantile=0.75, magnitude_threshold=0.01)
        # quantile threshold ≈ 0.00625, max(0.00625, 0.01) = 0.01
        # Only 0.02 >= 0.01 → 1 selected
        assert mask2.sum().item() == 1

    def test_negative_advantages_filtered_by_magnitude(self) -> None:
        """Negative advantages are filtered by absolute value."""
        advantages = torch.tensor([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
        mask = filter_advantages(advantages, quantile=0.75, magnitude_threshold=0.01)
        # abs values: [0.5, 0.3, 0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        # 75th percentile of abs ≈ 0.55
        # Selected: |δ| >= 0.55 → [0.7, 0.9] and [-0.5 is 0.5 < 0.55]
        # Actually: sorted abs = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.9]
        # 75th pct: index = 0.75*7 = 5.25 → 0.5 + 0.25*(0.7-0.5) = 0.55
        # |δ| >= 0.55 → 0.7, 0.9 → 2 selected
        assert mask.sum().item() == 2
        assert mask[6].item() is True  # 0.7
        assert mask[7].item() is True  # 0.9


# ---------------------------------------------------------------------------
# Expected value tests
# ---------------------------------------------------------------------------


class TestExpectedValue:
    """Tests for expected_value."""

    def test_expected_value_win_certain(self) -> None:
        """Certain win → E[v] = 1."""
        probs = torch.tensor([1.0, 0.0, 0.0])
        result = expected_value(probs)
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_expected_value_loss_certain(self) -> None:
        """Certain loss → E[v] = -1."""
        probs = torch.tensor([0.0, 1.0, 0.0])
        result = expected_value(probs)
        torch.testing.assert_close(result, torch.tensor(-1.0))

    def test_expected_value_mixed(self) -> None:
        """Mixed probabilities → E[v] = P(win) - P(loss)."""
        probs = torch.tensor([0.6, 0.3, 0.1])
        result = expected_value(probs)
        torch.testing.assert_close(result, torch.tensor(0.3))

    def test_expected_value_batch(self) -> None:
        """Batch input: (B, 3) → (B,)."""
        probs = torch.tensor([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.33, 0.33, 0.34],
        ])
        result = expected_value(probs)
        expected = torch.tensor([0.3, -0.3, 0.0])
        torch.testing.assert_close(result, expected)
