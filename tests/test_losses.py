"""Tests for stratego.training.losses — setup, move, and belief loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from stratego.training.losses import belief_loss, move_loss, setup_loss


class TestSetupLoss:
    def test_output_scalar_finite_has_grad(self) -> None:
        B, S, C = 4, 10, 12
        value_logits = torch.randn(B, 3, requires_grad=True)
        entropy_pred = torch.randn(B, requires_grad=True)
        new_p = torch.softmax(torch.randn(B, S, C, requires_grad=True), dim=-1)
        total, m = setup_loss(
            value_logits=value_logits, entropy_pred=entropy_pred,
            policy_logits=torch.randn(B, S, C),
            target_outcome=torch.tensor([1.0, -1.0, 0.0, 1.0]),
            target_next_piece=torch.randint(0, C, (B, S)),
            advantages=torch.randn(B),
            conditional_entropy=torch.rand(B) * 10,
            old_policy_probs=torch.softmax(torch.randn(B, S, C), dim=-1),
            new_policy_probs=new_p,
        )
        assert total.dim() == 0
        assert torch.isfinite(total).item()
        assert total.requires_grad
        total.backward()
        assert value_logits.grad is not None
        assert entropy_pred.grad is not None

    def test_value_loss_correct_vs_incorrect(self) -> None:
        idx = torch.tensor([0, 1, 2])
        correct = torch.tensor([[10, -10, -10], [-10, 10, -10], [-10, -10, 10.0]])
        wrong = torch.tensor([[-10, 10, -10], [10, -10, -10], [10, -10, -10.0]])
        assert F.cross_entropy(correct, idx) < F.cross_entropy(wrong, idx)

    def test_ppo_clip(self) -> None:
        old_p = torch.full((1, 1, 12), 1 / 12)
        new_p = torch.full((1, 1, 12), 0.01)
        new_p[0, 0, 5] = 0.89
        _, m = setup_loss(
            value_logits=torch.randn(1, 3), entropy_pred=torch.randn(1),
            policy_logits=torch.randn(1, 1, 12),
            target_outcome=torch.tensor([1.0]),
            target_next_piece=torch.tensor([[5]]),
            advantages=torch.tensor([1.0]),
            conditional_entropy=torch.rand(1) * 10,
            old_policy_probs=old_p, new_policy_probs=new_p,
            kl_coeff=0.0, value_coeff=0.0, entropy_coeff=0.0,
        )
        r = 0.89 / (1 / 12)
        exp = -min(r * 1.0, min(max(r, 0.8), 1.2) * 1.0)
        assert m["policy_loss"] == pytest.approx(exp, abs=1e-4)

    def test_entropy_mse(self) -> None:
        ep = torch.tensor([0.5, 0.3])
        ce = torch.tensor([5.0, 3.0])
        _, m = setup_loss(
            value_logits=torch.randn(2, 3), entropy_pred=ep,
            policy_logits=torch.randn(2, 4, 12),
            target_outcome=torch.tensor([1.0, -1.0]),
            target_next_piece=torch.randint(0, 12, (2, 4)),
            advantages=torch.randn(2), conditional_entropy=ce,
            old_policy_probs=torch.softmax(torch.randn(2, 4, 12), -1),
            new_policy_probs=torch.softmax(torch.randn(2, 4, 12), -1),
            kl_coeff=0.0, value_coeff=0.0, entropy_coeff=1.0,
        )
        assert m["entropy_loss"] == pytest.approx(((ce / 10 - ep) ** 2).mean(), abs=1e-5)

    def test_kl_term(self) -> None:
        B, S, C = 2, 3, 12
        old_p = torch.softmax(torch.randn(B, S, C), -1)
        new_p = torch.softmax(torch.randn(B, S, C), -1)
        exp_kl = (new_p * (new_p.log() - old_p.log())).sum(-1).mean()
        _, m = setup_loss(
            value_logits=torch.randn(B, 3), entropy_pred=torch.randn(B),
            policy_logits=torch.randn(B, S, C),
            target_outcome=torch.tensor([1.0, -1.0]),
            target_next_piece=torch.randint(0, C, (B, S)),
            advantages=torch.randn(B), conditional_entropy=torch.rand(B) * 10,
            old_policy_probs=old_p, new_policy_probs=new_p,
            kl_coeff=0.1, value_coeff=0.0, entropy_coeff=0.0,
        )
        assert m["kl"] == pytest.approx(exp_kl, abs=1e-5)


class TestMoveLoss:
    def test_output_scalar_finite_has_grad(self) -> None:
        B = 4
        vl = torch.randn(B, 3, requires_grad=True)
        np_ = torch.rand(B, requires_grad=True)
        total, m = move_loss(
            value_logits=vl, policy_logits=torch.randn(B, 10000),
            target_move_idx=torch.randint(0, 10000, (B,)),
            advantages=torch.randn(B),
            outcome_probs=torch.softmax(torch.randn(B, 3), -1),
            old_policy_probs=torch.rand(B) + _EPS,
            new_policy_probs=np_,
            magnet_probs=torch.rand(B) + _EPS,
        )
        assert total.dim() == 0
        assert torch.isfinite(total).item()
        total.backward()
        assert vl.grad is not None
        assert np_.grad is not None

    def test_ppo_clip(self) -> None:
        old_p = torch.tensor([0.01])
        new_p = torch.tensor([0.5], requires_grad=True)
        _, m = move_loss(
            value_logits=torch.randn(1, 3), policy_logits=torch.randn(1, 10000),
            target_move_idx=torch.tensor([0]),
            advantages=torch.tensor([1.0]),
            outcome_probs=torch.softmax(torch.randn(1, 3), -1),
            old_policy_probs=old_p, new_policy_probs=new_p,
            magnet_probs=torch.tensor([0.5]),
            kl_coeff=0.0, magnet_kl_coeff=0.0,
        )
        r = 0.5 / 0.01
        exp = -min(r * 1.0, min(max(r, 0.8), 1.2) * 1.0)
        assert m["policy_loss"] == pytest.approx(exp, abs=1e-4)

    def test_kl_magnet(self) -> None:
        new_p = torch.tensor([0.5, 0.3])
        mag_p = torch.tensor([0.4, 0.2])
        exp = (new_p * ((new_p + _EPS).log() - (mag_p + _EPS).log())).mean()
        _, m = move_loss(
            value_logits=torch.randn(2, 3), policy_logits=torch.randn(2, 10000),
            target_move_idx=torch.tensor([0, 1]),
            advantages=torch.randn(2),
            outcome_probs=torch.softmax(torch.randn(2, 3), -1),
            old_policy_probs=torch.rand(2) + _EPS, new_policy_probs=new_p,
            magnet_probs=mag_p,
            kl_coeff=0.0, magnet_kl_coeff=1.0,
        )
        assert m["kl_magnet"] == pytest.approx(exp, abs=1e-5)

    def test_value_cross_entropy(self) -> None:
        vl = torch.randn(3, 3)
        op = torch.softmax(torch.randn(3, 3), -1)
        exp = F.cross_entropy(vl, op)
        _, m = move_loss(
            value_logits=vl, policy_logits=torch.randn(3, 10000),
            target_move_idx=torch.tensor([0, 1, 2]),
            advantages=torch.randn(3), outcome_probs=op,
            old_policy_probs=torch.rand(3) + _EPS,
            new_policy_probs=torch.rand(3) + _EPS,
            magnet_probs=torch.rand(3) + _EPS,
            kl_coeff=0.0, magnet_kl_coeff=0.0,
        )
        assert m["value_loss"] == pytest.approx(exp.item(), abs=1e-5)


class TestBeliefLoss:
    def test_nll(self) -> None:
        logits = torch.randn(4, 5, 12)
        targets = torch.randint(0, 12, (4, 5))
        exp = F.cross_entropy(logits.reshape(20, 12), targets.reshape(20))
        assert belief_loss(logits, targets) == pytest.approx(exp, abs=1e-5)

    def test_grad(self) -> None:
        logits = torch.randn(2, 3, 12, requires_grad=True)
        targets = torch.randint(0, 12, (2, 3))
        belief_loss(logits, targets).backward()
        assert logits.grad is not None

    def test_finite(self) -> None:
        logits = torch.randn(8, 10, 12)
        targets = torch.randint(0, 12, (8, 10))
        assert torch.isfinite(belief_loss(logits, targets)).item()


_EPS = 1e-8
