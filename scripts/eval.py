#!/usr/bin/env python3
"""Evaluation script — load model, run N games, report win rate."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-games", type=int, default=20)
    args = parser.parse_args()

    # Load checkpoint and run evaluation
    print(f"Eval: {args.n_games} games from {args.checkpoint}")


if __name__ == "__main__":
    main()
