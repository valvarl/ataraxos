"""Tests for stratego.training.setup_trainer — SetupTrainer.

TDD: tests written before the implementation.
Covers: construction, train_step (loss finite, gradient clipping, EMA update,
iteration counter), _collate, train_epoch (multiple batches, different sizes),
loss decrease over steps, and eval model apply/restore.
"""

from __future__ import annotations

import math

import pytest
import torch

from stratego.constants import (
    SETUP_ENTROPY_COEFF,
    SETUP_KL_COEFF,
    SETUP_MAX_GRAD_NORM,
    SETUP_PPO_CLIP,
    SETUP_VALUE_COEFF,
)
from stratego.networks.setup_net import SetupNetwork
from stratego.training.ema import EMA
from stratego.training.setup_trainer import SetupTrainer
from stratego.types import NUM_PIECE_TYPES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model() -> SetupNetwork:
    """Small SetupNetwork for fast tests."""
    return SetupNetwork(depth=1, dim=32, heads=2, ff=64)


def _make_batch(batch_size: int = 4, seq_len: int = 10) -> dict[str, torch.Tensor]:
    """Create a valid training batch."""
    return {
        "tokens": torch.randint(0, NUM_PIECE_TYPES, (batch_size, seq_len)),
        "target_outcome": torch.tensor([1.0, -1.0, 0.0] * batch_size)[:batch_size],
        "target_next_piece": torch.randint(0, NUM_PIECE_TYPES, (batch_size, seq_len)),
        "advantages": torch.randn(batch_size),
        "conditional_entropy": torch.rand(batch_size) * 10,
        "old_policy_probs": torch.softmax(
            torch.randn(batch_size, seq_len, NUM_PIECE_TYPES), dim=-1
        ),
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        """Default hyperparams match paper constants."""
        model = _make_tiny_model()
        trainer = SetupTrainer(model)
        assert trainer.model is model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.ema, EMA)
        assert trainer.grad_norm == SETUP_MAX_GRAD_NORM
        assert trainer.ppo_clip == SETUP_PPO_CLIP
        assert trainer.kl_coeff == SETUP_KL_COEFF
        assert trainer.value_coeff == SETUP_VALUE_COEFF
        assert trainer.entropy_coeff == SETUP_ENTROPY_COEFF
        assert trainer.iteration == 0

    def test_custom_hyperparams(self) -> None:
        """Custom hyperparams are stored correctly."""
        model = _make_tiny_model()
        trainer = SetupTrainer(
            model,
            lr=1e-3,
            grad_norm=1.0,
            ema_smoothing=0.99,
            ppo_clip=0.1,
            kl_coeff=0.05,
            value_coeff=1.0,
            entropy_coeff=0.5,
        )
        assert trainer.grad_norm == 1.0
        assert trainer.ppo_clip == 0.1
        assert trainer.kl_coeff == 0.05
        assert trainer.value_coeff == 1.0
        assert trainer.entropy_coeff == 0.5

    def test_ema_registered(self) -> None:
        """EMA shadow is initialized from model parameters."""
        model = _make_tiny_model()
        trainer = SetupTrainer(model)
        assert len(trainer.ema.shadow) > 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in trainer.ema.shadow
                assert torch.allclose(trainer.ema.shadow[name], param.data)

    def test_optimizer_is_adam(self) -> None:
        """Optimizer is Adam with the specified learning rate."""
        model = _make_tiny_model()
        trainer = SetupTrainer(model, lr=1e-3)
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        param_group = trainer.optimizer.param_groups[0]
        assert param_group["lr"] == 1e-3


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_returns_metrics_dict(self) -> None:
        """train_step returns a dict with expected metric keys."""
        trainer = SetupTrainer(_make_tiny_model())
        metrics = trainer.train_step(_make_batch())
        assert isinstance(metrics, dict)
        for key in ("policy_loss", "value_loss", "entropy_loss", "kl", "total"):
            assert key in metrics

    def test_metrics_are_finite_floats(self) -> None:
        """All metric values are finite Python floats."""
        trainer = SetupTrainer(_make_tiny_model())
        metrics = trainer.train_step(_make_batch())
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is not a float"
            assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_gradient_clipped(self) -> None:
        """After train_step, total gradient norm is <= grad_norm."""
        trainer = SetupTrainer(_make_tiny_model(), grad_norm=0.5)
        trainer.train_step(_make_batch())
        grads = [p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None]
        assert len(grads) > 0
        total_norm = torch.norm(torch.cat(grads))
        assert total_norm.item() <= 0.5 + 1e-5

    def test_gradient_clipped_tight(self) -> None:
        """With grad_norm=0, all gradients are zero after clipping."""
        trainer = SetupTrainer(_make_tiny_model(), grad_norm=0.0)
        trainer.train_step(_make_batch())
        for p in trainer.model.parameters():
            if p.grad is not None:
                assert p.grad.abs().max().item() == 0.0

    def test_ema_updated_after_step(self) -> None:
        """EMA shadow changes after a train_step."""
        trainer = SetupTrainer(_make_tiny_model())
        old_shadow = {k: v.clone() for k, v in trainer.ema.shadow.items()}
        trainer.train_step(_make_batch())
        changed = any(
            not torch.allclose(trainer.ema.shadow[name], old_shadow[name])
            for name in trainer.ema.shadow
        )
        assert changed

    def test_iteration_incremented(self) -> None:
        """iteration counter increments after each train_step."""
        trainer = SetupTrainer(_make_tiny_model())
        batch = _make_batch()
        assert trainer.iteration == 0
        trainer.train_step(batch)
        assert trainer.iteration == 1
        trainer.train_step(batch)
        assert trainer.iteration == 2

    def test_model_in_train_mode(self) -> None:
        """Model is set to train mode during train_step."""
        trainer = SetupTrainer(_make_tiny_model())
        trainer.model.eval()
        trainer.train_step(_make_batch())
        assert trainer.model.training


# ---------------------------------------------------------------------------
# _collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_collate_concatenates_dim0(self) -> None:
        """_collate concatenates tensors along dim 0."""
        trainer = SetupTrainer(_make_tiny_model())
        b1 = _make_batch(batch_size=2, seq_len=10)
        b2 = _make_batch(batch_size=3, seq_len=10)
        collated = trainer._collate([b1, b2])
        for key in b1:
            assert collated[key].shape[0] == 5

    def test_collate_preserves_values(self) -> None:
        """_collate preserves the original tensor values in order."""
        trainer = SetupTrainer(_make_tiny_model())
        b1 = _make_batch(batch_size=2, seq_len=5)
        b2 = _make_batch(batch_size=2, seq_len=5)
        collated = trainer._collate([b1, b2])
        assert torch.equal(collated["tokens"][:2], b1["tokens"])
        assert torch.equal(collated["tokens"][2:], b2["tokens"])
        assert torch.equal(collated["advantages"][:2], b1["advantages"])
        assert torch.equal(collated["advantages"][2:], b2["advantages"])


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    def test_multiple_batches(self) -> None:
        """train_epoch processes data in chunks of batch_size."""
        trainer = SetupTrainer(_make_tiny_model())
        data = [_make_batch(batch_size=2, seq_len=10) for _ in range(6)]
        results = trainer.train_epoch(data, batch_size=2)
        assert len(results) == 3
        for m in results:
            assert "total" in m

    def test_different_batch_sizes(self) -> None:
        """train_epoch works with different batch_size values."""
        trainer = SetupTrainer(_make_tiny_model())
        data = [_make_batch(batch_size=1, seq_len=10) for _ in range(8)]
        results = trainer.train_epoch(data, batch_size=4)
        assert len(results) == 2
        results = trainer.train_epoch(data, batch_size=3)
        assert len(results) == 3  # ceil(8/3) = 3

    def test_batch_size_exceeds_data(self) -> None:
        """When batch_size >= len(data), a single batch is processed."""
        trainer = SetupTrainer(_make_tiny_model())
        data = [_make_batch(batch_size=1, seq_len=10) for _ in range(3)]
        results = trainer.train_epoch(data, batch_size=100)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Loss decrease
# ---------------------------------------------------------------------------


class TestLossDecreases:
    def test_loss_decreases_over_steps(self) -> None:
        """With a fixed batch and high lr, total loss should decrease."""
        torch.manual_seed(42)
        model = _make_tiny_model()
        trainer = SetupTrainer(model, lr=1e-2, kl_coeff=0.0, entropy_coeff=0.0)
        batch = _make_batch(batch_size=4, seq_len=10)
        batch["advantages"] = torch.ones(4)
        batch["target_next_piece"] = torch.zeros(4, 10, dtype=torch.long)

        first_metrics = trainer.train_step(batch)
        for _ in range(30):
            metrics = trainer.train_step(batch)
        assert metrics["total"] < first_metrics["total"]


# ---------------------------------------------------------------------------
# Eval model
# ---------------------------------------------------------------------------


class TestEvalModel:
    def test_get_eval_model_applies_ema(self) -> None:
        """get_eval_model swaps EMA weights into the model."""
        trainer = SetupTrainer(_make_tiny_model())
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(torch.randn_like(p))
        eval_model = trainer.get_eval_model()
        assert eval_model is trainer.model
        for name, param in trainer.model.named_parameters():
            if name in trainer.ema.shadow:
                assert torch.allclose(param.data, trainer.ema.shadow[name])

    def test_restore_model_after_eval(self) -> None:
        """restore_model puts back the pre-eval weights."""
        trainer = SetupTrainer(_make_tiny_model())
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(torch.randn_like(p))
        perturbed = {n: p.data.clone() for n, p in trainer.model.named_parameters()}
        trainer.get_eval_model()
        trainer.restore_model()
        for name, param in trainer.model.named_parameters():
            assert torch.allclose(param.data, perturbed[name])

    def test_restore_without_get_raises(self) -> None:
        """restore_model without get_eval_model raises RuntimeError."""
        trainer = SetupTrainer(_make_tiny_model())
        with pytest.raises(RuntimeError, match="restore_model"):
            trainer.restore_model()
