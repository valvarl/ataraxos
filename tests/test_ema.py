"""Tests for stratego.training.ema — Polyak averaging (EMA)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from stratego.training.ema import EMA


def _make_model() -> nn.Module:
    """Tiny model for testing."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestEMARegister:
    def test_register_initializes_shadow(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)
        assert len(ema.shadow) > 0

    def test_register_shadow_matches_model(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(ema.shadow[name], param.data)

    def test_register_skips_frozen_params(self) -> None:
        model = _make_model()
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        ema = EMA(smoothing=0.999)
        ema.register(model)
        for name, _param in model[0].named_parameters():
            assert f"0.{name}" not in ema.shadow


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    def test_update_modifies_shadow(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Perturb model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        ema.update(model)
        # Shadow should have changed
        for name in ema.shadow:
            assert not torch.allclose(ema.shadow[name], old_shadow[name])

    def test_update_smoothing_formula(self) -> None:
        """Verify shadow = smoothing * shadow + (1 - smoothing) * param."""
        model = nn.Linear(2, 2, bias=False)
        ema = EMA(smoothing=0.9)
        ema.register(model)

        original_shadow = ema.shadow["weight"].clone()
        with torch.no_grad():
            model.weight.fill_(10.0)

        ema.update(model)
        expected = 0.9 * original_shadow + 0.1 * 10.0
        assert torch.allclose(ema.shadow["weight"], expected)

    def test_convergence_after_many_steps(self) -> None:
        """After many updates with constant params, shadow → param value."""
        model = nn.Linear(2, 2, bias=False)
        ema = EMA(smoothing=0.999)
        ema.register(model)

        with torch.no_grad():
            model.weight.fill_(5.0)

        # 0.999^10000 ≈ e^(-10) ≈ 4.5e-5, so shadow ≈ 5.0 within 1e-3
        for _ in range(10000):
            ema.update(model)

        assert torch.allclose(ema.shadow["weight"], torch.full_like(ema.shadow["weight"], 5.0), atol=1e-3)


# ---------------------------------------------------------------------------
# Apply / Restore
# ---------------------------------------------------------------------------


class TestEMAApplyRestore:
    def test_apply_swaps_ema_weights(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)

        # Perturb model away from shadow
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        ema.apply(model)
        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert torch.allclose(param.data, ema.shadow[name])

    def test_restore_puts_back_originals(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)

        # Save backup before apply
        backup = {name: param.data.clone() for name, param in model.named_parameters()}

        # Perturb shadow and apply
        for k in ema.shadow:
            ema.shadow[k].fill_(0.0)
        ema.apply(model)

        # Restore from backup
        ema.restore(model, backup)
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, backup[name])


# ---------------------------------------------------------------------------
# DDP unwrapping
# ---------------------------------------------------------------------------


class TestEMADDP:
    def test_register_unwraps_ddp(self) -> None:
        """EMA should handle DDP-wrapped models via .module."""
        model = _make_model()
        ddp_model = MagicMock(spec=nn.parallel.DistributedDataParallel)
        ddp_model.module = model

        ema = EMA(smoothing=0.999)
        ema.register(ddp_model)
        assert len(ema.shadow) > 0

    def test_update_unwraps_ddp(self) -> None:
        model = _make_model()
        ddp_model = MagicMock(spec=nn.parallel.DistributedDataParallel)
        ddp_model.module = model

        ema = EMA(smoothing=0.999)
        ema.register(ddp_model)
        ema.update(ddp_model)  # should not raise


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------


class TestEMAStateDict:
    def test_state_dict_returns_shadow(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)
        sd = ema.state_dict()
        assert sd is ema.shadow

    def test_load_state_dict(self) -> None:
        model = _make_model()
        ema = EMA(smoothing=0.999)
        ema.register(model)
        sd = ema.state_dict()

        ema2 = EMA(smoothing=0.999)
        ema2.load_state_dict(sd)
        for k in sd:
            assert torch.allclose(ema2.shadow[k], sd[k])
