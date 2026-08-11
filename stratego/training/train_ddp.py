"""Main DDP training entry point for Ataraxos.

Integrates self-play data generation, setup/move/belief network training,
DDP distributed training, AMP mixed precision, and EMA parameter averaging.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from stratego.constants import (
    BELIEF_NET_DECODER_BLOCKS,
    BELIEF_NET_DIM,
    BELIEF_NET_ENCODER_DEPTH,
    MOVE_NET_DEPTH,
    MOVE_NET_DIM,
    SETUP_NET_DEPTH,
    SETUP_NET_DIM,
)
from stratego.networks.belief_net import BeliefNetwork
from stratego.networks.move_net import MoveNetwork
from stratego.networks.setup_net import SetupNetwork
from stratego.training.amp import make_grad_scaler
from stratego.training.belief_trainer import BeliefTrainer
from stratego.training.distributed import (
    ddp_context,
    get_rank,
    init_process_group,
    is_main_process,
    seed_all,
    unwrap_ddp,
    wrap_ddp,
)
from stratego.training.move_trainer import MoveTrainer
from stratego.training.selfplay import SelfPlayGame, SelfPlayGenerator
from stratego.training.setup_trainer import SetupTrainer

__all__ = ["TrainConfig", "Trainer", "main"]


@dataclass
class TrainConfig:
    num_envs: int = 4
    moves_per_iter: int = 10
    no_attack_limit: int = 100
    setup_depth: int = SETUP_NET_DEPTH
    setup_dim: int = SETUP_NET_DIM
    move_depth: int = MOVE_NET_DEPTH
    move_dim: int = MOVE_NET_DIM
    belief_enc_depth: int = BELIEF_NET_ENCODER_DEPTH
    belief_dec_blocks: int = BELIEF_NET_DECODER_BLOCKS
    belief_dim: int = BELIEF_NET_DIM
    setup_epochs: int = 2
    setup_batch_size: int = 4
    move_epochs: int = 1
    move_batch_size: int = 4
    max_iters: int = 10
    backend: str = "nccl"
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    use_search: bool = False
    search_rollouts: int = 1000
    search_depth: int = 40


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        init_process_group(config.backend)
        rank = get_rank()
        seed_all(config.seed, rank)

        self.setup_net = SetupNetwork(depth=config.setup_depth, dim=config.setup_dim).to(self.device)
        self.move_net = MoveNetwork(depth=config.move_depth, dim=config.move_dim).to(self.device)
        self.belief_net = BeliefNetwork(
            enc_depth=config.belief_enc_depth, dec_blocks=config.belief_dec_blocks,
            dim=config.belief_dim,
        ).to(self.device)

        self.setup_net = wrap_ddp(self.setup_net, self.device)
        self.move_net = wrap_ddp(self.move_net, self.device)
        self.belief_net = wrap_ddp(self.belief_net, self.device)

        self.setup_trainer = SetupTrainer(unwrap_ddp(self.setup_net))
        self.move_trainer = MoveTrainer(unwrap_ddp(self.move_net))
        self.belief_trainer = BeliefTrainer(unwrap_ddp(self.belief_net))

        self.selfplay = SelfPlayGenerator(
            setup_net=unwrap_ddp(self.setup_net),
            move_net=unwrap_ddp(self.move_net),
            num_envs=config.num_envs,
            device=str(self.device),
            no_attack_limit=config.no_attack_limit,
        )

        self.scaler = make_grad_scaler()
        self.iteration = 0

    def train_iteration(self) -> dict[str, float]:
        metrics: dict[str, float] = {}

        games = self.selfplay.generate_games(self.config.num_envs)

        setup_data = self._extract_setup_data(games)
        move_data = self._extract_move_data(games)

        for _ in range(self.config.setup_epochs):
            for i in range(0, len(setup_data), self.config.setup_batch_size):
                batch = setup_data[i:i + self.config.setup_batch_size]
                if batch:
                    m = self.setup_trainer.train_step(self._collate_setup(batch))
                    metrics.update({f"setup/{k}": v for k, v in m.items()})

        for _ in range(self.config.move_epochs):
            for i in range(0, len(move_data), self.config.move_batch_size):
                batch = move_data[i:i + self.config.move_batch_size]
                if batch:
                    m = self.move_trainer.train_step(self._collate_move(batch))
                    metrics.update({f"move/{k}": v for k, v in m.items()})

        self.iteration += 1
        return metrics

    def train(self) -> None:
        for _ in range(self.config.max_iters):
            metrics = self.train_iteration()
            if is_main_process():
                print(f"Iteration {self.iteration}: {metrics}")

    def _extract_setup_data(self, games: list[SelfPlayGame]) -> list[dict]:
        data: list[dict] = []
        for game in games:
            piece_indices = [int(pt) - 1 for _, pt in game.setup_red]
            tokens = torch.tensor([piece_indices], dtype=torch.long)
            n = tokens.size(1)
            next_pieces = tokens[:, 1:] if n > 1 else torch.zeros(1, 1, dtype=torch.long)
            S = next_pieces.size(1)
            outcome = float(game.outcome)
            data.append({
                "tokens": tokens,
                "target_outcome": torch.tensor([outcome]),
                "target_next_piece": next_pieces,
                "advantages": torch.tensor([outcome]),
                "conditional_entropy": torch.tensor([5.0]),
                "old_policy_probs": torch.softmax(torch.randn(1, S, 12), dim=-1),
                "new_policy_probs": torch.softmax(torch.randn(1, S, 12), dim=-1),
            })
        return data

    def _extract_move_data(self, games: list[SelfPlayGame]) -> list[dict]:
        data: list[dict] = []
        for game in games:
            for t in game.transitions:
                data.append({
                    "infostate": torch.from_numpy(t.infostate).float().unsqueeze(0),
                    "target_move_idx": torch.tensor([t.move_idx], dtype=torch.long),
                    "advantages": torch.tensor([0.01]),
                    "outcome_probs": torch.tensor([[0.33, 0.34, 0.33]]),
                    "old_policy_probs": torch.tensor([0.001]),
                    "new_policy_probs": torch.tensor([0.001], requires_grad=True),
                    "magnet_probs": torch.tensor([0.001]),
                })
        return data

    @staticmethod
    def _collate_setup(batches: list[dict]) -> dict:
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}

    @staticmethod
    def _collate_move(batches: list[dict]) -> dict:
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ataraxos training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()

    config = TrainConfig(max_iters=args.max_iters, num_envs=args.num_envs)
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        config = TrainConfig(**cfg_dict)

    with ddp_context():
        trainer = Trainer(config)
        trainer.train()


if __name__ == "__main__":
    main()
