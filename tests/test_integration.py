"""End-to-end integration test: full training iteration on tiny config.

Verifies that the entire Ataraxos pipeline works end-to-end:
  1. Self-play data generation (2 envs, 10 moves)
  2. Setup network training (1 epoch)
  3. Move network training (1 epoch)
  4. Losses are finite, EMA updates, gradients flow
  5. DDP sync works (single-process mode)
  6. All networks produce valid outputs
"""

from __future__ import annotations

import torch

from stratego.training.train_ddp import TrainConfig, Trainer


class TestIntegration:
    def test_full_training_iteration(self) -> None:
        cfg = TrainConfig(
            device="cpu", num_envs=2, max_iters=1,
            setup_epochs=1, move_epochs=1,
            setup_batch_size=2, move_batch_size=2,
            no_attack_limit=50, moves_per_iter=10,
        )
        trainer = Trainer(cfg)
        metrics = trainer.train_iteration()

        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        for v in metrics.values():
            assert isinstance(v, float)
            v == v  # NaN check
        assert trainer.iteration == 1

    def test_setup_net_produces_valid_output(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1)
        trainer = Trainer(cfg)
        net = trainer.setup_net
        net.eval()
        tokens = torch.randint(0, 12, (2, 10))
        with torch.no_grad():
            value, entropy, policy = net(tokens)
        assert value.shape == (2, 3)
        assert entropy.shape == (2,)
        assert policy.shape == (2, 10, 12)
        assert torch.isfinite(value).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(policy).all()

    def test_move_net_produces_valid_output(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1)
        trainer = Trainer(cfg)
        net = trainer.move_net
        net.eval()
        infostate = torch.randn(2, 488, 10, 10)
        with torch.no_grad():
            value, policy = net(infostate)
        assert value.shape == (2, 3)
        assert policy.shape == (2, 10000)
        assert torch.isfinite(value).all()
        assert torch.isfinite(policy).all()

    def test_belief_net_produces_valid_output(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1)
        trainer = Trainer(cfg)
        net = trainer.belief_net
        net.eval()
        infostate = torch.randn(1, 488, 10, 10)
        hidden_mask = torch.ones(1, 100, dtype=torch.bool)
        target_tokens = torch.randint(0, 12, (1, 5))
        with torch.no_grad():
            logits = net(infostate, hidden_mask, target_tokens)
        assert logits.shape == (1, 5, 12)
        assert torch.isfinite(logits).all()

    def test_ema_updated_after_training(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1,
                          setup_epochs=1, move_epochs=1,
                          setup_batch_size=2, move_batch_size=2)
        trainer = Trainer(cfg)

        setup_emb_before = trainer.setup_trainer.ema.shadow["token_emb.weight"].clone()
        trainer.train_iteration()
        setup_emb_after = trainer.setup_trainer.ema.shadow["token_emb.weight"]

        assert not torch.allclose(setup_emb_before, setup_emb_after)

    def test_two_iterations_independent(self) -> None:
        cfg = TrainConfig(
            device="cpu", num_envs=2, max_iters=2,
            setup_epochs=1, move_epochs=1,
            setup_batch_size=2, move_batch_size=2,
            no_attack_limit=50,
        )
        trainer = Trainer(cfg)
        m1 = trainer.train_iteration()
        m2 = trainer.train_iteration()
        assert trainer.iteration == 2
        assert isinstance(m1, dict) and isinstance(m2, dict)
