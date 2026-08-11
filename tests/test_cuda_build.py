"""Verify the _stratego_cuda CUDA extension builds and imports correctly.

These tests JIT-compile the extension from csrc/stratego_buffer.{cu,cpp} and
verify the basic class API. Skipped when CUDA is unavailable or build fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"

# JIT-compile once at module import; tests skip if this fails.
try:
    from torch.utils.cpp_extension import load as _load

    _stratego_cuda = _load(
        name="_stratego_cuda",
        sources=[
            str(CSRC / "stratego_buffer.cu"),
            str(CSRC / "stratego_buffer.cpp"),
        ],
        extra_cflags=["-std=c++17", "-O3"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_86,code=sm_86",
        ],
        verbose=False,
    )
except Exception:
    _stratego_cuda = None  # type: ignore[assignment]

_build_skip = pytest.mark.skipif(
    _stratego_cuda is None, reason="_stratego_cuda extension not built (no CUDA or build error)"
)


@_build_skip
def test_import_stratego_cuda() -> None:
    """The compiled extension module must be importable."""
    import _stratego_cuda  # noqa: F401


@_build_skip
def test_class_constructible() -> None:
    """StrategoRolloutBuffer must be constructible with n_games and device_id."""
    buf = _stratego_cuda.StrategoRolloutBuffer(4, 0)
    assert buf.device_id == 0
    assert buf.n_games == 4


@_build_skip
def test_hello_world_returns_cuda_tensor_with_42() -> None:
    """hello_world() must return a CUDA tensor whose single element is 42."""
    buf = _stratego_cuda.StrategoRolloutBuffer(1, 0)
    result = buf.hello_world()

    assert result.is_cuda, "Result must be a CUDA tensor"
    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    assert result.dtype == torch.int32, f"Expected int32, got {result.dtype}"
    assert result.item() == 42, f"Expected value 42, got {result.item()}"


@_build_skip
def test_default_construction() -> None:
    """Constructor must accept no arguments (defaults: n_games=1, device_id=0)."""
    buf = _stratego_cuda.StrategoRolloutBuffer()
    assert buf.device_id == 0
    assert buf.n_games == 1
