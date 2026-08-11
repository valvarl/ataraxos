"""Tests for stratego.training.schedulers — power-law annealing schedulers."""

from __future__ import annotations

import pytest
import torch

from stratego.training.schedulers import (
    make_lr_scheduler,
    move_lr_value,
    move_reg_temp,
    setup_reg_temp,
)

# ---------------------------------------------------------------------------
# setup_reg_temp: α = 0.1 / iter^0.3
# ---------------------------------------------------------------------------


class TestSetupRegTemp:
    def test_iter_1(self) -> None:
        assert setup_reg_temp(1) == pytest.approx(0.1 / 1**0.3)

    def test_iter_10(self) -> None:
        assert setup_reg_temp(10) == pytest.approx(0.1 / 10**0.3)

    def test_iter_100(self) -> None:
        assert setup_reg_temp(100) == pytest.approx(0.1 / 100**0.3)

    def test_iter_1000(self) -> None:
        assert setup_reg_temp(1000) == pytest.approx(0.1 / 1000**0.3)

    def test_iter_zero_returns_numerator(self) -> None:
        """iter < 1 should return numerator to avoid div-by-zero."""
        assert setup_reg_temp(0) == pytest.approx(0.1)

    def test_decreasing(self) -> None:
        """Temperature must decrease monotonically."""
        vals = [setup_reg_temp(i) for i in range(1, 100)]
        for a, b in zip(vals, vals[1:], strict=False):
            assert a >= b


# ---------------------------------------------------------------------------
# move_reg_temp: α = 0.05 / iter^0.3
# ---------------------------------------------------------------------------


class TestMoveRegTemp:
    def test_iter_1(self) -> None:
        assert move_reg_temp(1) == pytest.approx(0.05 / 1**0.3)

    def test_iter_10(self) -> None:
        assert move_reg_temp(10) == pytest.approx(0.05 / 10**0.3)

    def test_iter_100(self) -> None:
        assert move_reg_temp(100) == pytest.approx(0.05 / 100**0.3)

    def test_iter_1000(self) -> None:
        assert move_reg_temp(1000) == pytest.approx(0.05 / 1000**0.3)

    def test_iter_zero_returns_numerator(self) -> None:
        assert move_reg_temp(0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# move_lr_value: clip(0.5 / iter^1.1, 5e-6, 1e-4)
# ---------------------------------------------------------------------------


class TestMoveLrValue:
    def test_iter_1_clipped_to_max(self) -> None:
        """0.5 / 1^1.1 = 0.5, which exceeds lr_max=1e-4 → clipped."""
        assert move_lr_value(1) == pytest.approx(1e-4)

    def test_iter_10(self) -> None:
        raw = 0.5 / 10**1.1
        expected = max(5e-6, min(1e-4, raw))
        assert move_lr_value(10) == pytest.approx(expected)

    def test_iter_100(self) -> None:
        raw = 0.5 / 100**1.1
        expected = max(5e-6, min(1e-4, raw))
        assert move_lr_value(100) == pytest.approx(expected)

    def test_iter_1000(self) -> None:
        raw = 0.5 / 1000**1.1
        expected = max(5e-6, min(1e-4, raw))
        assert move_lr_value(1000) == pytest.approx(expected)

    def test_never_exceeds_lr_max(self) -> None:
        for i in range(1, 10000):
            assert move_lr_value(i) <= 1e-4 + 1e-12

    def test_never_below_lr_min(self) -> None:
        for i in range(1, 100000):
            assert move_lr_value(i) >= 5e-6 - 1e-12

    def test_iter_zero_returns_max(self) -> None:
        assert move_lr_value(0) == pytest.approx(1e-4)

    def test_very_large_iter_clipped_to_min(self) -> None:
        """At very large iterations, raw value → 0, should be clipped to lr_min."""
        assert move_lr_value(10_000_000) == pytest.approx(5e-6)


# ---------------------------------------------------------------------------
# make_lr_scheduler: LambdaLR with power-law annealing
# ---------------------------------------------------------------------------


class TestMakeLrScheduler:
    def test_initial_lr(self) -> None:
        """At step 0, lambda returns 1.0 → base_lr unchanged."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        make_lr_scheduler(opt, numerator=0.5, exponent=1.1, lr_min=5e-6, lr_max=1e-4)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)

    def test_step_reduces_lr(self) -> None:
        """After stepping, LR should decrease (or stay at min)."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        sched = make_lr_scheduler(opt, numerator=0.5, exponent=1.1, lr_min=5e-6, lr_max=1e-4)
        initial_lr = opt.param_groups[0]["lr"]
        sched.step()
        sched.step()
        sched.step()
        # After 3 steps, LR should be ≤ initial
        assert opt.param_groups[0]["lr"] <= initial_lr + 1e-12

    def test_scheduler_matches_move_lr_value(self) -> None:
        """LambdaLR at step N should produce lr_max * (move_lr_value(N) / lr_max)."""
        model = torch.nn.Linear(4, 4)
        base_lr = 1e-4
        opt = torch.optim.Adam(model.parameters(), lr=base_lr)
        sched = make_lr_scheduler(opt, numerator=0.5, exponent=1.1, lr_min=5e-6, lr_max=1e-4)
        for _ in range(50):
            sched.step()
        step = 50
        expected_lr = move_lr_value(step)
        assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-5)
