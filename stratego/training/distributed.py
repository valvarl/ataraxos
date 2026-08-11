"""Distributed training utilities: DDP setup, torchrun helpers, per-rank seed offset."""

# ruff: noqa: N817 (DDP is the standard PyTorch acronym)
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DDPConfig:
    world_size: int = 1
    backend: str = "nccl"
    seed: int = 42


def is_distributed() -> bool:
    """True if running under torchrun (RANK env var set)."""
    return "RANK" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def init_process_group(backend: str = "nccl") -> None:
    if not is_distributed():
        return
    dist.init_process_group(backend=backend, init_method="env://")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_all(seed: int, rank: int = 0) -> None:
    """Seed all RNGs with per-rank offset for env diversity."""
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap model in DDP if distributed, else return as-is."""
    if not is_distributed() or not torch.cuda.is_available():
        return model
    return DDP(model, device_ids=[get_local_rank()], find_unused_parameters=False)  # type: ignore[no-any-return]


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP model to get underlying module."""
    if isinstance(model, DDP):
        return model.module  # type: ignore[no-any-return]
    return model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce mean across processes."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def gather_tensor(tensor: torch.Tensor, dst: int = 0) -> list[torch.Tensor] | None:
    """Gather tensor from all processes to dst rank."""
    if not dist.is_initialized():
        return [tensor]
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.gather(tensor, gather_list=gathered, dst=dst)
    return gathered if get_rank() == dst else None


@contextmanager
def ddp_context(backend: str = "nccl"):
    """Context manager for DDP: init on enter, cleanup on exit."""
    init_process_group(backend)
    try:
        yield
    finally:
        cleanup()
