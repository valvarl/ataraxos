"""Pytest configuration and shared fixtures for the Ataraxos test suite."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible tests."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture()
def device() -> torch.device:
    """Return the default torch device — CUDA if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def cuda_available() -> bool:
    """True iff a CUDA GPU is visible to PyTorch."""
    return torch.cuda.is_available()
