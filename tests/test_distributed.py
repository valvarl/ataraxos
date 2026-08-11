"""Tests for stratego.training.distributed — DDP setup, torchrun helpers, per-rank seed offset."""

from __future__ import annotations

import os
import random
from unittest.mock import patch

import numpy as np
import torch

from stratego.training.distributed import (
    DDPConfig,
    all_reduce_mean,
    ddp_context,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    seed_all,
    unwrap_ddp,
    wrap_ddp,
)

# ---------------------------------------------------------------------------
# is_distributed / env helpers
# ---------------------------------------------------------------------------


class TestIsDistributed:
    def test_returns_false_without_rank_env(self) -> None:
        """is_distributed() must return False when RANK env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_distributed() is False

    def test_returns_true_with_rank_env(self) -> None:
        """is_distributed() must return True when RANK env var is set."""
        with patch.dict(os.environ, {"RANK": "0"}, clear=False):
            assert is_distributed() is True


class TestGetRank:
    def test_default_zero(self) -> None:
        """get_rank() defaults to 0 when RANK is unset."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_rank() == 0

    def test_reads_rank_env(self) -> None:
        """get_rank() reads the RANK env var."""
        with patch.dict(os.environ, {"RANK": "3"}, clear=False):
            assert get_rank() == 3


class TestGetLocalRank:
    def test_default_zero(self) -> None:
        """get_local_rank() defaults to 0 when LOCAL_RANK is unset."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_local_rank() == 0

    def test_reads_local_rank_env(self) -> None:
        """get_local_rank() reads the LOCAL_RANK env var."""
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=False):
            assert get_local_rank() == 2


class TestGetWorldSize:
    def test_default_one(self) -> None:
        """get_world_size() defaults to 1 when WORLD_SIZE is unset."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_world_size() == 1

    def test_reads_world_size_env(self) -> None:
        """get_world_size() reads the WORLD_SIZE env var."""
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False):
            assert get_world_size() == 8


class TestIsMainProcess:
    def test_main_when_rank_zero(self) -> None:
        """is_main_process() returns True when rank is 0."""
        with patch.dict(os.environ, {"RANK": "0"}, clear=False):
            assert is_main_process() is True

    def test_not_main_when_rank_nonzero(self) -> None:
        """is_main_process() returns False when rank is nonzero."""
        with patch.dict(os.environ, {"RANK": "5"}, clear=False):
            assert is_main_process() is False

    def test_main_by_default(self) -> None:
        """is_main_process() returns True when no RANK env (single-process)."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_main_process() is True


# ---------------------------------------------------------------------------
# seed_all
# ---------------------------------------------------------------------------


class TestSeedAll:
    def test_same_seed_same_state(self) -> None:
        """seed_all with same seed produces identical RNG states."""
        seed_all(123, rank=0)
        r1 = random.random()
        n1 = np.random.random()
        t1 = torch.randn(1).item()

        seed_all(123, rank=0)
        r2 = random.random()
        n2 = np.random.random()
        t2 = torch.randn(1).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2

    def test_rank_offset_produces_different_seeds(self) -> None:
        """seed_all with different rank offsets produces different RNG states."""
        seed_all(42, rank=0)
        r0 = random.random()

        seed_all(42, rank=1)
        r1 = random.random()

        assert r0 != r1

    def test_seeds_numpy_and_torch(self) -> None:
        """seed_all seeds numpy and torch RNGs too."""
        seed_all(999, rank=0)
        n_before = np.random.random()
        t_before = torch.randn(1).item()

        seed_all(999, rank=0)
        n_after = np.random.random()
        t_after = torch.randn(1).item()

        assert n_before == n_after
        assert t_before == t_after


# ---------------------------------------------------------------------------
# wrap_ddp / unwrap_ddp
# ---------------------------------------------------------------------------


class TestWrapUnwrapDDP:
    def test_wrap_returns_model_when_not_distributed(self) -> None:
        """wrap_ddp returns the model as-is when not running distributed."""
        model = torch.nn.Linear(4, 2)
        with patch.dict(os.environ, {}, clear=True):
            result = wrap_ddp(model, device=torch.device("cpu"))
        assert result is model

    def test_unwrap_returns_model_when_not_ddp(self) -> None:
        """unwrap_ddp returns the model as-is when it is not a DDP wrapper."""
        model = torch.nn.Linear(4, 2)
        assert unwrap_ddp(model) is model


# ---------------------------------------------------------------------------
# all_reduce_mean
# ---------------------------------------------------------------------------


class TestAllReduceMean:
    def test_returns_tensor_unchanged_when_not_distributed(self) -> None:
        """all_reduce_mean returns tensor unchanged when dist is not initialized."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce_mean(t)
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# ddp_context
# ---------------------------------------------------------------------------


class TestDDPContext:
    def test_noop_when_not_distributed(self) -> None:
        """ddp_context is a no-op when not running under torchrun."""
        with patch.dict(os.environ, {}, clear=True), ddp_context():
            pass  # should not raise


# ---------------------------------------------------------------------------
# DDPConfig
# ---------------------------------------------------------------------------


class TestDDPConfig:
    def test_defaults(self) -> None:
        """DDPConfig has expected defaults."""
        cfg = DDPConfig()
        assert cfg.world_size == 1
        assert cfg.backend == "nccl"
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        """DDPConfig accepts custom values."""
        cfg = DDPConfig(world_size=4, backend="gloo", seed=7)
        assert cfg.world_size == 4
        assert cfg.backend == "gloo"
        assert cfg.seed == 7
