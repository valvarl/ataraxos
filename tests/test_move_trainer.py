"""Tests for stratego.training.move_trainer.MoveTrainer.

TDD: these tests are written before the implementation.
Covers: construction, train_step output, LR schedule, magnet KL schedule,
gradient clipping, EMA update, iteration counter, loss finiteness,
get_eval_model.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.utils

from stratego.constants import (
    MOVE_KL_COEFF,
    MOVE_LR_MAX,
    MOVE_LR_MIN,
    MOVE_MAGNET_KL_NUMERATOR,
    MOVE_MAX_GRAD_NORM,
    MOVE_PPO_CLIP,
    NUM_INFOSTATE_CHANNELS,
    NUM_SQUARES,
)
from stratego.networks.move_net import MoveNetwork
from stratego.training.ema import EMA
from stratego.training.move_trainer import MoveTrainer


def _make_small_move_net() -> MoveNetwork:
    """Small MoveNetwork for fast tests (same architecture, tiny dims)."""
    return MoveNetwork(depth=2, dim=64, heads=2, ff=128)


def _make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Create a random move training batch."""
    return {
        "infostate": torch.randn(batch_size, NUM_INFOSTATE_CHANNELS, 10, 10),
        "target_move_idx": torch.randint(0, NUM_SQUARES * NUM_SQUARES, (batch_size,)),
        "advantages": torch.randn(batch_size),
        "outcome_probs": torch.softmax(torch.randn(batch_size, 3), dim=-1),
        "old_policy_probs": torch.rand(batch_size) + 1e-8,
        "magnet_probs": torch.rand(batch_size) + 1e-8,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMoveTrainerConstruction:
    def test_construction_defaults(self) -> None:
        """Trainer initializes with paper-default attributes."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        assert trainer.model is model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.ema, EMA)
        assert trainer.grad_norm == MOVE_MAX_GRAD_NORM
        assert trainer.ppo_clip == MOVE_PPO_CLIP
        assert trainer.kl_coeff == MOVE_KL_COEFF
        assert trainer.iteration == 0

    def test_construction_custom_hyperparams(self) -> None:
        """Custom hyperparams are respected."""
        model = _make_small_move_net()
        trainer = MoveTrainer(
            model,
            lr_max=1e-3,
            grad_norm=1.0,
            ema_smoothing=0.99,
            ppo_clip=0.1,
            kl_coeff=0.05,
        )
        assert trainer.grad_norm == 1.0
        assert trainer.ppo_clip == 0.1
        assert trainer.kl_coeff == 0.05
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-3


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_produces_finite_metrics(self) -> None:
        """train_step returns a dict of finite float metrics."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        batch = _make_batch(2)
        metrics = trainer.train_step(batch)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is not float"
            assert torch.isfinite(torch.tensor(val)).item(), f"{key} is not finite"

    def test_expected_metric_keys(self) -> None:
        """Metrics dict contains move_loss keys plus lr and magnet_kl_coeff."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        batch = _make_batch(2)
        metrics = trainer.train_step(batch)
        expected = {
            "policy_loss",
            "value_loss",
            "kl_policy",
            "kl_magnet",
            "total",
            "lr",
            "magnet_kl_coeff",
        }
        assert expected.issubset(metrics.keys())

    def test_loss_finite_across_multiple_steps(self) -> None:
        """Loss is finite across multiple train_steps with fresh random inputs."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        for _ in range(3):
            batch = _make_batch(2)
            metrics = trainer.train_step(batch)
            assert torch.isfinite(torch.tensor(metrics["total"])).item()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


class TestLRSchedule:
    def test_lr_below_one_returns_max(self) -> None:
        """iter < 1 returns lr_max."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        assert trainer.get_lr(0) == MOVE_LR_MAX

    def test_lr_clipped_to_max(self) -> None:
        """Small iterations are clipped to lr_max (1e-4)."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        for it in [1, 10, 100]:
            assert trainer.get_lr(it) == MOVE_LR_MAX, f"iter={it}"

    def test_lr_clipped_to_min(self) -> None:
        """Very large iterations are clipped to lr_min (5e-6)."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        assert trainer.get_lr(100_000) == MOVE_LR_MIN

    def test_lr_unclipped_intermediate(self) -> None:
        """Intermediate iterations produce unclipped power-law LR."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        lr = trainer.get_lr(5000)
        assert MOVE_LR_MIN < lr < MOVE_LR_MAX
        expected = 0.5 / (5000 ** 1.1)
        assert lr == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Magnet KL schedule
# ---------------------------------------------------------------------------


class TestMagnetKLSchedule:
    def test_magnet_kl_below_one(self) -> None:
        """iter < 1 returns numerator (0.05)."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        assert trainer.get_magnet_kl_coeff(0) == MOVE_MAGNET_KL_NUMERATOR

    def test_magnet_kl_known_values(self) -> None:
        """Known values at iter=1, 10, 100."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        assert trainer.get_magnet_kl_coeff(1) == pytest.approx(0.05, abs=1e-10)
        assert trainer.get_magnet_kl_coeff(10) == pytest.approx(0.05 / (10 ** 0.3), abs=1e-10)
        assert trainer.get_magnet_kl_coeff(100) == pytest.approx(0.05 / (100 ** 0.3), abs=1e-10)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


class TestGradientClipping:
    def test_clip_grad_norm_called_with_configured_max(self) -> None:
        """clip_grad_norm_ is called with the configured grad_norm."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model, grad_norm=0.267)
        batch = _make_batch(2)
        with patch(
            "torch.nn.utils.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as spy:
            trainer.train_step(batch)
        spy.assert_called_once()
        args, _kwargs = spy.call_args
        assert args[1] == 0.267  # max_norm positional arg


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    def test_ema_shadow_updated_after_step(self) -> None:
        """After train_step, EMA shadow has changed from initial values."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        initial_shadow = {k: v.clone() for k, v in trainer.ema.shadow.items()}
        batch = _make_batch(2)
        trainer.train_step(batch)
        any_changed = any(
            not torch.equal(trainer.ema.shadow[name], initial_shadow[name])
            for name in trainer.ema.shadow
        )
        assert any_changed


# ---------------------------------------------------------------------------
# Iteration counter
# ---------------------------------------------------------------------------


class TestIterationCounter:
    def test_increments_per_step(self) -> None:
        """Iteration counter increments by 1 per train_step."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        assert trainer.iteration == 0
        batch = _make_batch(2)
        trainer.train_step(batch)
        assert trainer.iteration == 1
        trainer.train_step(batch)
        assert trainer.iteration == 2


# ---------------------------------------------------------------------------
# get_eval_model
# ---------------------------------------------------------------------------


class TestGetEvalModel:
    def test_applies_ema_weights(self) -> None:
        """get_eval_model returns model with EMA shadow weights applied."""
        model = _make_small_move_net()
        trainer = MoveTrainer(model)
        batch = _make_batch(2)
        trainer.train_step(batch)
        eval_model = trainer.get_eval_model()
        for name, param in eval_model.named_parameters():
            if name in trainer.ema.shadow:
                assert torch.allclose(param.data, trainer.ema.shadow[name])
