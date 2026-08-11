"""Tests for stratego.training.belief_trainer — BeliefTrainer.

TDD: these tests are written before the implementation.
Covers: construction, train_step finite loss, gradient flow, parameter
update, EMA shadow update, train_epoch batching, _collate stacking, and
get_eval_model EMA weight swap.
"""

from __future__ import annotations

import torch

from stratego.networks.belief_net import BeliefNetwork
from stratego.training.belief_trainer import BeliefTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> BeliefNetwork:
    """Small belief network for fast tests (1+1 blocks, dim=64)."""
    return BeliefNetwork(enc_depth=1, dec_blocks=1, dim=64, heads=2, ff=128)


def _make_batch(batch_size: int = 2, seq_len: int = 5) -> dict[str, torch.Tensor]:
    """Build a small belief training batch."""
    return {
        "infostate": torch.randn(batch_size, 488, 10, 10),
        "hidden_mask": torch.ones(batch_size, 100, dtype=torch.bool),
        "target_tokens": torch.randint(0, 12, (batch_size, seq_len)),
    }


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_construction_sets_attributes(self) -> None:
        model = _make_model()
        trainer = BeliefTrainer(model, lr=1e-4, grad_norm=1.0, ema_smoothing=0.999)
        assert trainer.model is model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.grad_norm == 1.0
        # EMA shadow populated from model parameters
        assert len(trainer.ema.shadow) > 0

    def test_default_arguments(self) -> None:
        trainer = BeliefTrainer(_make_model())
        assert trainer.grad_norm == 1.0
        assert trainer.ema.smoothing == 0.999

    def test_ema_shadow_matches_model_at_init(self) -> None:
        model = _make_model()
        trainer = BeliefTrainer(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(trainer.ema.shadow[name], param.data)


# ---------------------------------------------------------------------------
# 2. train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_returns_finite_loss(self) -> None:
        trainer = BeliefTrainer(_make_model())
        out = trainer.train_step(_make_batch())
        assert set(out.keys()) == {"loss"}
        assert torch.isfinite(torch.tensor(out["loss"])).item()

    def test_gradient_flows_to_params(self) -> None:
        """After train_step, every trainable parameter has a non-None grad."""
        model = _make_model()
        trainer = BeliefTrainer(model)
        trainer.train_step(_make_batch())
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads if g is not None)

    def test_params_change_after_step(self) -> None:
        """optimizer.step() actually updates model parameters."""
        model = _make_model()
        trainer = BeliefTrainer(model)
        before = {n: p.data.clone() for n, p in model.named_parameters()}
        trainer.train_step(_make_batch())
        changed = any(
            not torch.allclose(before[n], p.data)
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        assert changed


# ---------------------------------------------------------------------------
# 3. EMA update
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    def test_ema_shadow_changes_after_step(self) -> None:
        """train_step calls ema.update, blending post-step params into shadow."""
        model = _make_model()
        trainer = BeliefTrainer(model)
        before = {k: v.clone() for k, v in trainer.ema.shadow.items()}
        trainer.train_step(_make_batch())
        changed = any(
            not torch.allclose(before[k], trainer.ema.shadow[k]) for k in before
        )
        assert changed

    def test_ema_shadow_blends_init_and_step(self) -> None:
        """With smoothing s, shadow = s*init + (1-s)*post_step for one step."""
        model = _make_model()
        trainer = BeliefTrainer(model, ema_smoothing=0.9)
        init = {k: v.clone() for k, v in trainer.ema.shadow.items()}
        trainer.train_step(_make_batch())
        # Pull post-step params and verify the smoothing formula on one tensor.
        name = next(n for n, _ in model.named_parameters() if _.requires_grad)
        post_step = dict(model.named_parameters())[name].data
        expected = 0.9 * init[name] + 0.1 * post_step
        assert torch.allclose(trainer.ema.shadow[name], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    def test_runs_multiple_batches(self) -> None:
        trainer = BeliefTrainer(_make_model())
        data = [_make_batch(batch_size=2, seq_len=4) for _ in range(5)]
        results = trainer.train_epoch(data, batch_size=2)
        # range(0, 5, 2) -> [0, 2, 4] -> 3 batches
        assert len(results) == 3
        for r in results:
            assert "loss" in r
            assert torch.isfinite(torch.tensor(r["loss"])).item()

    def test_default_batch_size(self) -> None:
        """Default batch_size=256 collapses 10 single-element batches into 1."""
        trainer = BeliefTrainer(_make_model())
        data = [_make_batch(batch_size=1) for _ in range(10)]
        results = trainer.train_epoch(data)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 5. _collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_stacks_tensors_correctly(self) -> None:
        trainer = BeliefTrainer(_make_model())
        batches = [_make_batch(batch_size=2, seq_len=5) for _ in range(3)]
        collated = trainer._collate(batches)
        assert collated["infostate"].shape == (6, 488, 10, 10)
        assert collated["hidden_mask"].shape == (6, 100)
        assert collated["hidden_mask"].dtype == torch.bool
        assert collated["target_tokens"].shape == (6, 5)
        assert collated["target_tokens"].dtype == torch.long

    def test_collate_preserves_values(self) -> None:
        """Concatenation preserves row order: first chunk then second chunk."""
        trainer = BeliefTrainer(_make_model())
        b1 = _make_batch(batch_size=2, seq_len=3)
        b2 = _make_batch(batch_size=2, seq_len=3)
        collated = trainer._collate([b1, b2])
        assert torch.equal(collated["infostate"][:2], b1["infostate"])
        assert torch.equal(collated["infostate"][2:], b2["infostate"])
        assert torch.equal(collated["target_tokens"][2:], b2["target_tokens"])


# ---------------------------------------------------------------------------
# 6. get_eval_model
# ---------------------------------------------------------------------------


class TestGetEvalModel:
    def test_returns_model_with_ema_weights(self) -> None:
        """get_eval_model swaps EMA shadow weights into the model."""
        model = _make_model()
        trainer = BeliefTrainer(model)
        # Take a step so the EMA shadow diverges from the (post-step) params.
        trainer.train_step(_make_batch())
        eval_model = trainer.get_eval_model()
        assert eval_model is model
        for name, param in model.named_parameters():
            if name in trainer.ema.shadow:
                assert torch.allclose(param.data, trainer.ema.shadow[name])
