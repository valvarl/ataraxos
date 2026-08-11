"""Tests for stratego.training.amp — autocast + GradScaler utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from stratego.training.amp import autocast, make_grad_scaler

# ---------------------------------------------------------------------------
# autocast
# ---------------------------------------------------------------------------


class TestAutocast:
    def test_cpu_noop(self) -> None:
        """On CPU, autocast should be a no-op (no dtype change)."""
        with autocast(device_type="cpu"):
            x = torch.randn(4, 4)
            y = x @ x.T
            assert y.dtype == torch.float32

    def test_disabled_is_noop(self) -> None:
        """When enabled=False, autocast should be a no-op."""
        with autocast(device_type="cuda", enabled=False):
            x = torch.randn(4, 4)
            y = x @ x.T
            assert y.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_bf16(self) -> None:
        """On bf16-capable GPU, autocast should use bfloat16."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip("bf16 not supported on this GPU")
        with autocast(device_type="cuda"):
            x = torch.randn(4, 4, device="cuda")
            y = x @ x.T
            # Inside autocast with bf16, matmul output should be bf16
            assert y.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_fp16_fallback(self) -> None:
        """On fp16-only GPU (mocked), autocast should use float16."""
        if torch.cuda.is_bf16_supported():
            pytest.skip("GPU supports bf16, cannot test fp16 fallback natively")
        with autocast(device_type="cuda"):
            x = torch.randn(4, 4, device="cuda")
            y = x @ x.T
            assert y.dtype == torch.float16

    def test_cpu_explicit_device_type(self) -> None:
        """Explicitly passing device_type='cpu' should be a no-op."""
        with autocast(device_type="cpu", enabled=True):
            x = torch.randn(2, 2)
            assert x.dtype == torch.float32


# ---------------------------------------------------------------------------
# make_grad_scaler
# ---------------------------------------------------------------------------


class TestMakeGradScaler:
    def test_no_cuda_returns_none(self) -> None:
        """Without CUDA, make_grad_scaler should return None."""
        with patch("stratego.training.amp.torch.cuda.is_available", return_value=False):
            assert make_grad_scaler() is None

    def test_bf16_returns_none(self) -> None:
        """On bf16-capable GPU, GradScaler is not needed → None."""
        with (
            patch("stratego.training.amp.torch.cuda.is_available", return_value=True),
            patch("stratego.training.amp.torch.cuda.is_bf16_supported", return_value=True),
        ):
            assert make_grad_scaler() is None

    def test_fp16_returns_scaler(self) -> None:
        """On fp16-only GPU (V100), should return a GradScaler."""
        with (
            patch("stratego.training.amp.torch.cuda.is_available", return_value=True),
            patch("stratego.training.amp.torch.cuda.is_bf16_supported", return_value=False),
        ):
            scaler = make_grad_scaler()
            assert scaler is not None
            assert isinstance(scaler, torch.amp.GradScaler)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_gpu(self) -> None:
        """On the actual GPU: bf16 → None, fp16 → GradScaler."""
        if torch.cuda.is_bf16_supported():
            assert make_grad_scaler() is None
        else:
            scaler = make_grad_scaler()
            assert scaler is not None
            assert isinstance(scaler, torch.amp.GradScaler)
