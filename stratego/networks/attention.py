"""Unified MultiHeadAttention with automatic backend selection.

Backend priority (runtime auto-detect):
  1. FlashAttention 3 — Hopper sm_90+ (``flash_attn_interface``)
  2. FlashAttention 2 — Ampere sm_80+ (``flash_attn``)
  3. PyTorch SDPA     — all CUDA GPUs (``F.scaled_dot_product_attention``)
  4. Math fallback    — CPU (naive Q·K^T·scale → softmax → V)

Notes:
  - FA3 does **not** support dropout; the module falls back to FA2/SDPA when
    ``dropout > 0`` and the module is in training mode.
  - V100 (sm_70) has no native bf16 — callers must use fp16 on that hardware.
  - GQA (grouped-query attention) is supported: set ``num_kv_heads < num_heads``.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Module-level cached import probes for optional FlashAttention packages
# ---------------------------------------------------------------------------

_FA3_AVAILABLE: bool | None = None
_FA2_AVAILABLE: bool | None = None


def _check_fa3() -> bool:
    """Probe for FlashAttention 3 (``flash_attn_interface``)."""
    global _FA3_AVAILABLE  # noqa: PLW0603
    if _FA3_AVAILABLE is None:
        try:
            import flash_attn_interface  # noqa: F401

            _FA3_AVAILABLE = True
        except ImportError:
            _FA3_AVAILABLE = False
    return _FA3_AVAILABLE


def _check_fa2() -> bool:
    """Probe for FlashAttention 2 (``flash_attn``)."""
    global _FA2_AVAILABLE  # noqa: PLW0603
    if _FA2_AVAILABLE is None:
        try:
            import flash_attn  # noqa: F401

            _FA2_AVAILABLE = True
        except ImportError:
            _FA2_AVAILABLE = False
    return _FA2_AVAILABLE


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention with automatic backend selection.

    Supports MHA (``num_kv_heads == num_heads``), GQA (``num_kv_heads < num_heads``),
    self-attention, cross-attention, and causal masking.

    Parameters
    ----------
    embed_dim:
        Model / embedding dimension.
    num_heads:
        Number of query attention heads.  ``embed_dim`` must be divisible by this.
    num_kv_heads:
        Number of key/value heads (for GQA).  Defaults to ``num_heads`` (standard MHA).
    dropout:
        Attention dropout probability.  Applied only in training mode.
    causal:
        If ``True``, apply a causal (lower-triangular) mask for self-attention.
        Causal masking is **not** applied during cross-attention even when ``True``.
    qkv_bias:
        If ``True``, add bias to the Q/K/V and output projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        causal: bool = False,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({kv_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads: int = kv_heads
        self.head_dim: int = embed_dim // num_heads
        self.dropout: float = dropout
        self.causal: bool = causal

        # Projections — GQA: K/V are smaller when num_kv_heads < num_heads
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=qkv_bias)

    # ------------------------------------------------------------------
    # Backend detection (property — re-evaluated on device/mode change)
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the active backend name for the current device and mode."""
        device = next(self.parameters()).device

        if device.type != "cuda":
            return "math"

        major, _ = torch.cuda.get_device_capability(device)

        # FA3: Hopper sm_90+
        if major >= 9 and _check_fa3():
            # FA3 has no dropout support — fall back during training if needed
            if self.training and self.dropout > 0.0:
                if _check_fa2():
                    return "flash_attn_2"
                return "sdpa"
            return "flash_attn_3"

        # FA2: Ampere sm_80+
        if major >= 8 and _check_fa2():
            return "flash_attn_2"

        # SDPA: all CUDA GPUs (V100 sm_70 etc.)
        return "sdpa"

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        key_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Parameters
        ----------
        x:
            Query input of shape ``(B, S_q, embed_dim)``.
        key_value:
            Optional key/value source of shape ``(B, S_kv, embed_dim)``.
            When ``None``, self-attention is performed (K/V derived from *x*).

        Returns
        -------
        torch.Tensor:
            Output of shape ``(B, S_q, embed_dim)``.
        """
        batch_size, seq_q, _ = x.shape

        # --- projections ---------------------------------------------------
        q = self.q_proj(x)  # (B, S_q, num_heads * head_dim)

        kv_input = key_value if key_value is not None else x
        k = self.k_proj(kv_input)  # (B, S_kv, num_kv_heads * head_dim)
        v = self.v_proj(kv_input)  # (B, S_kv, num_kv_heads * head_dim)

        seq_kv = k.shape[1]

        # Reshape to (B, S, H, D) — the canonical attention layout
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_kv, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_kv, self.num_kv_heads, self.head_dim)

        # Causal masking only for self-attention (not cross-attention)
        is_causal = self.causal and key_value is None

        # --- dispatch to backend --------------------------------------------
        backend = self.backend
        if backend == "flash_attn_3":
            attn_out = self._forward_fa3(q, k, v, is_causal)
        elif backend == "flash_attn_2":
            attn_out = self._forward_fa2(q, k, v, is_causal)
        elif backend == "sdpa":
            attn_out = self._forward_sdpa(q, k, v, is_causal)
        else:
            attn_out = self._forward_math(q, k, v, is_causal)

        # --- output projection ----------------------------------------------
        attn_out = attn_out.reshape(batch_size, seq_q, self.embed_dim)
        return cast(torch.Tensor, self.out_proj(attn_out))

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _expand_kv_heads(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat KV heads to match the number of query heads (for GQA).

        Input layout: ``(B, H_kv, S, D)`` — returns same layout with ``H_kv → H_q``.
        """
        if self.num_kv_heads == self.num_heads:
            return k, v
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        return k, v

    def _forward_fa3(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """FlashAttention 3 path (Hopper sm_90+).  No dropout support."""
        import flash_attn_interface as fa3  # type: ignore[import-untyped]

        # q, k, v: (B, S, H, D) — native FA layout
        out: torch.Tensor = fa3.flash_attn_func(q, k, v, causal=is_causal)
        return out

    def _forward_fa2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """FlashAttention 2 path (Ampere sm_80+).  Supports dropout."""
        import flash_attn as fa2  # type: ignore[import-untyped]

        dropout_p = self.dropout if self.training else 0.0
        # q, k, v: (B, S, H, D) — native FA layout
        out: torch.Tensor = fa2.flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=is_causal,
        )
        return out

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """PyTorch SDPA path — works on all CUDA GPUs."""
        # Transpose from (B, S, H, D) → (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: expand KV heads to match Q heads
        k, v = self._expand_kv_heads(k, v)

        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        # (B, H, S_q, D) → (B, S_q, H, D)
        return out.transpose(1, 2)

    def _forward_math(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """Naive math fallback for CPU."""
        batch_size, seq_q, _, head_dim = q.shape
        seq_kv = k.shape[1]

        # Transpose to (B, H, S, D)
        q = q.transpose(1, 2)  # (B, H_q, S_q, D)
        k = k.transpose(1, 2)  # (B, H_kv, S_kv, D)
        v = v.transpose(1, 2)  # (B, H_kv, S_kv, D)

        # GQA: expand KV heads
        k, v = self._expand_kv_heads(k, v)

        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S_q, S_kv)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_kv, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        out = torch.matmul(attn_weights, v)  # (B, H, S_q, D)
        return out.transpose(1, 2)  # (B, S_q, H, D)
