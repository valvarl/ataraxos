# Ataraxos: Superhuman AI for Stratego

A full reproduction of [arXiv:2511.07312](https://arxiv.org/abs/2511.07312) —
"Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search".

## Setup

```bash
conda create -n ataraxos python=3.11 -y
conda activate ataraxos
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e . --no-build-isolation
```

## Architecture

- **StrategoRolloutBuffer** (CUDA C++): GPU-resident simulator with ~10M state updates/sec
- **Setup Network**: Decoder-only transformer (depth=4, dim=512, 12.6M params)
- **Move Network**: Encoder-only transformer with key-query matrix product (depth=8, dim=384, 14.7M params)
- **Belief Network**: Encoder-decoder transformer (enc=6, dec=4, dim=512, 57.1M params)
- **Test-Time Search**: Update-equivalence via magnetic mirror descent (1000 rollouts × depth 40)

## Training

```bash
# Dev (RTX 3060)
python scripts/train.py --config configs/dev_3060.yaml

# Production (16x H100)
torchrun --standalone --nproc_per_node=16 scripts/train.py --config configs/h100_16node.yaml
```

## Key Design Choices

- **Dynamic damping**: Coordinated annealing of regularization temperature + LR + PPO clip + grad clip
- **Advantage filtering**: Train only on top 25% by magnitude (|δ| ≥ 0.01)
- **Self-play**: Direct policy sampling (no search in data generation)
- **bfloat16**: 3x speedup on H100/Ampere; fp16+GradScaler fallback on V100
- **DDP**: Replicate-then-allreduce (networks are 12-57M params, too small for FSDP)
- **Attention**: FA3→FA2→SDPA→math fallback chain

## Project Structure

```
ataraxos/
├── csrc/               # CUDA C++ extension (StrategoRolloutBuffer)
├── stratego/
│   ├── types.py        # PieceType, Player, Square, Action
│   ├── constants.py    # All paper hyperparameters
│   ├── env/            # Rules engine + infostate (488 channels)
│   ├── networks/       # Setup, move, belief networks
│   ├── training/       # Self-play, trainers, losses, DDP, AMP, EMA
│   └── search/         # Belief sampling, rollouts, mirror descent
├── configs/            # YAML configs (dev_3060, h100_16node, etc.)
├── scripts/            # train.py, eval.py, search.py, play.py
└── tests/              # 550+ tests
```
