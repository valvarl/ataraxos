"""Automatic Mixed Precision (AMP) utilities with GPU-aware dtype selection.

- H100 (sm_90), Ampere (sm_80+): bfloat16 (no GradScaler needed)
- V100 (sm_70): float16 (GradScaler needed to prevent underflow)
- CPU: float32 (no autocast)
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def autocast(
    device_type: str = "cuda", enabled: bool = True
) -> Generator[None, None, None]:
    """Autocast context that auto-selects bf16 or fp16 based on GPU capability.

    On bf16-capable GPUs (Ampere+, H100), uses bfloat16 which has the same
    dynamic range as fp32 and does not require gradient scaling.

    On fp16-only GPUs (V100), uses float16 which requires a GradScaler to
    prevent gradient underflow.

    On CPU or when disabled, acts as a no-op.

    Args:
        device_type: Target device type ("cuda" or "cpu").
        enabled: Whether autocast is active.

    Yields:
        None
    """
    if not enabled or device_type == "cpu":
        yield
        return

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield
    elif torch.cuda.is_available():
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            yield
    else:
        yield


def make_grad_scaler() -> torch.amp.GradScaler | None:
    """Create a GradScaler if needed (fp16/V100), else None.

    bfloat16 has the same exponent range as float32, so gradient scaling
    is unnecessary. Only float16 (V100) requires a GradScaler.

    Returns:
        A GradScaler instance for fp16 training, or None if not needed.
    """
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return None  # bf16 doesn't need scaler
    return torch.amp.GradScaler("cuda")
