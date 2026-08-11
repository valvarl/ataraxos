#!/usr/bin/env python3
"""Training entry point for torchrun."""
import argparse

import yaml
from stratego.training.train_ddp import TrainConfig, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    config = TrainConfig(**cfg_dict)
    from stratego.training.distributed import ddp_context

    with ddp_context():
        trainer = Trainer(config)
        trainer.train()


if __name__ == "__main__":
    main()
