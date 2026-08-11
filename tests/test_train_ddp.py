"""Tests for stratego.training.train_ddp — main training loop integration."""

from __future__ import annotations

import torch

from stratego.training.train_ddp import TrainConfig, Trainer


class TestTrainConfig:
    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.num_envs == 4
        assert cfg.max_iters == 10
        assert cfg.device == "cuda"

    def test_custom(self) -> None:
        cfg = TrainConfig(num_envs=8, max_iters=100, device="cpu")
        assert cfg.num_envs == 8
        assert cfg.max_iters == 100


class TestTrainer:
    def test_construction(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1)
        trainer = Trainer(cfg)
        assert trainer.iteration == 0
        assert trainer.setup_net is not None
        assert trainer.move_net is not None
        assert trainer.belief_net is not None

    def test_extract_setup_data(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1)
        trainer = Trainer(cfg)
        games = trainer.selfplay.generate_games(2)
        data = trainer._extract_setup_data(games)
        assert len(data) > 0
        assert "tokens" in data[0]
        assert "target_outcome" in data[0]

    def test_extract_move_data(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1)
        trainer = Trainer(cfg)
        games = trainer.selfplay.generate_games(2)
        data = trainer._extract_move_data(games)
        assert len(data) > 0
        assert "infostate" in data[0]
        assert "target_move_idx" in data[0]

    def test_collate(self) -> None:
        batches = [{"a": torch.tensor([1]), "b": torch.tensor([2.0])},
                    {"a": torch.tensor([3]), "b": torch.tensor([4.0])}]
        result = Trainer._collate_setup(batches)
        assert result["a"].shape == (2,)
        assert result["b"].shape == (2,)

    def test_train_iteration_runs(self) -> None:
        cfg = TrainConfig(device="cpu", num_envs=2, max_iters=1,
                          setup_epochs=1, move_epochs=1,
                          setup_batch_size=2, move_batch_size=2)
        trainer = Trainer(cfg)
        metrics = trainer.train_iteration()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert trainer.iteration == 1
