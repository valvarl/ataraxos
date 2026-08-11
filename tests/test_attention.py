"""Tests for stratego.networks.attention.MultiHeadAttention.

TDD: these tests are written before the implementation.
Covers: shape, causal masking, multi-head, GQA, dropout, backward,
bf16, variable seq lengths, batch size 1, finiteness, backend selection,
and cross-attention.
"""

from __future__ import annotations

import pytest
import torch

from stratego.networks.attention import MultiHeadAttention

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 not supported on this device",
)


# ---------------------------------------------------------------------------
# 1. Forward pass shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_output_shape_matches_input(self, device: torch.device) -> None:
        """Input (B, S, E) -> output (B, S, E)."""
        mha = MultiHeadAttention(embed_dim=512, num_heads=8).to(device)
        x = torch.randn(4, 100, 512, device=device)
        out = mha(x)
        assert out.shape == (4, 100, 512)

    def test_output_shape_setup_network_dims(self, device: torch.device) -> None:
        """Setup network: dim=512, heads=8, causal=True."""
        mha = MultiHeadAttention(embed_dim=512, num_heads=8, causal=True).to(device)
        x = torch.randn(2, 92, 512, device=device)
        out = mha(x)
        assert out.shape == (2, 92, 512)

    def test_output_shape_move_network_dims(self, device: torch.device) -> None:
        """Move network: dim=384, heads=8, causal=False."""
        mha = MultiHeadAttention(embed_dim=384, num_heads=8).to(device)
        x = torch.randn(2, 92, 384, device=device)
        out = mha(x)
        assert out.shape == (2, 92, 384)


# ---------------------------------------------------------------------------
# 2. Causal vs non-causal masking
# ---------------------------------------------------------------------------


class TestCausalMasking:
    def test_causal_output_differs_from_non_causal(self, device: torch.device) -> None:
        """Causal and non-causal should produce different outputs for the same input."""
        torch.manual_seed(0)
        x = torch.randn(2, 20, 64, device=device)

        mha_causal = MultiHeadAttention(embed_dim=64, num_heads=4, causal=True).to(device)
        mha_non_causal = MultiHeadAttention(embed_dim=64, num_heads=4, causal=False).to(device)

        # Copy weights so the only difference is the mask
        mha_non_causal.load_state_dict(mha_causal.state_dict())

        mha_causal.eval()
        mha_non_causal.eval()

        out_c = mha_causal(x)
        out_nc = mha_non_causal(x)
        assert not torch.allclose(out_c, out_nc, atol=1e-5)

    def test_causal_first_token_independent(self, device: torch.device) -> None:
        """With causal masking, the first output token depends only on the first input token."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4, causal=True).to(device)
        mha.eval()

        x1 = torch.randn(1, 10, 64, device=device)
        x2 = x1.clone()
        x2[0, 5:] = torch.randn(5, 64, device=device)  # change tokens 5-9

        out1 = mha(x1)
        out2 = mha(x2)
        # First token output should be identical since causal mask blocks future tokens
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Multi-head configuration
# ---------------------------------------------------------------------------


class TestMultiHead:
    def test_eight_heads_dim_64(self, device: torch.device) -> None:
        """8 heads with head_dim=64 -> embed_dim=512."""
        mha = MultiHeadAttention(embed_dim=512, num_heads=8).to(device)
        x = torch.randn(2, 16, 512, device=device)
        out = mha(x)
        assert out.shape == (2, 16, 512)
        assert mha.head_dim == 64
        assert mha.num_heads == 8


# ---------------------------------------------------------------------------
# 4. Grouped-query attention (GQA)
# ---------------------------------------------------------------------------


class TestGQA:
    def test_gqa_num_kv_heads(self, device: torch.device) -> None:
        """GQA: num_kv_heads=2, num_heads=8 -> output shape unchanged."""
        mha = MultiHeadAttention(embed_dim=512, num_heads=8, num_kv_heads=2).to(device)
        x = torch.randn(2, 16, 512, device=device)
        out = mha(x)
        assert out.shape == (2, 16, 512)
        assert mha.num_kv_heads == 2

    def test_gqa_kv_heads_less_than_heads(self, device: torch.device) -> None:
        """GQA with 4 KV heads and 8 query heads."""
        mha = MultiHeadAttention(embed_dim=256, num_heads=8, num_kv_heads=4).to(device)
        x = torch.randn(1, 10, 256, device=device)
        out = mha(x)
        assert out.shape == (1, 10, 256)


# ---------------------------------------------------------------------------
# 5. Dropout
# ---------------------------------------------------------------------------


class TestDropout:
    def test_dropout_in_training_mode(self, device: torch.device) -> None:
        """With dropout=0.5 in training mode, outputs should vary across runs."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.5).to(device)
        mha.train()
        x = torch.randn(2, 16, 64, device=device)

        outputs = [mha(x) for _ in range(5)]
        # At least some pairs should differ due to dropout
        any_differ = any(
            not torch.allclose(outputs[0], outputs[i], atol=1e-6)
            for i in range(1, len(outputs))
        )
        assert any_differ, "Dropout should cause variation in training mode"

    def test_no_dropout_in_eval_mode(self, device: torch.device) -> None:
        """In eval mode, outputs should be deterministic regardless of dropout setting."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.5).to(device)
        mha.eval()
        x = torch.randn(2, 16, 64, device=device)

        out1 = mha(x)
        out2 = mha(x)
        assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Backward pass
# ---------------------------------------------------------------------------


class TestBackwardPass:
    def test_gradients_flow(self, device: torch.device) -> None:
        """Gradients should flow through the attention module."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        x = torch.randn(2, 10, 64, device=device, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Check that at least some gradients are non-zero
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 7. bf16 on CUDA
# ---------------------------------------------------------------------------


class TestBFloat16:
    @requires_bf16
    def test_bf16_forward(self) -> None:
        """Forward pass in bf16 on CUDA produces finite outputs."""
        device = torch.device("cuda")
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device, dtype=torch.bfloat16)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        out = mha(x)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    @requires_bf16
    def test_bf16_backward(self) -> None:
        """Backward pass in bf16 on CUDA produces finite gradients."""
        device = torch.device("cuda")
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device, dtype=torch.bfloat16)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        out = mha(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 8. Different sequence lengths
# ---------------------------------------------------------------------------


class TestSequenceLengths:
    @pytest.mark.parametrize("seq_len", [1, 10, 100, 256])
    def test_variable_seq_lengths(self, seq_len: int, device: torch.device) -> None:
        """Forward pass works for various sequence lengths."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        x = torch.randn(2, seq_len, 64, device=device)
        out = mha(x)
        assert out.shape == (2, seq_len, 64)


# ---------------------------------------------------------------------------
# 9. Batch size 1
# ---------------------------------------------------------------------------


class TestBatchSizeOne:
    def test_batch_size_one(self, device: torch.device) -> None:
        """Forward pass works with batch size 1."""
        mha = MultiHeadAttention(embed_dim=128, num_heads=4).to(device)
        x = torch.randn(1, 20, 128, device=device)
        out = mha(x)
        assert out.shape == (1, 20, 128)


# ---------------------------------------------------------------------------
# 10. Output finiteness
# ---------------------------------------------------------------------------


class TestOutputFiniteness:
    def test_no_nan_or_inf(self, device: torch.device) -> None:
        """Output contains no NaN or Inf values."""
        mha = MultiHeadAttention(embed_dim=512, num_heads=8).to(device)
        x = torch.randn(4, 100, 512, device=device)
        out = mha(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_no_nan_or_inf_causal(self, device: torch.device) -> None:
        """Causal output contains no NaN or Inf values."""
        mha = MultiHeadAttention(embed_dim=512, num_heads=8, causal=True).to(device)
        x = torch.randn(4, 100, 512, device=device)
        out = mha(x)
        assert torch.isfinite(out).all(), "Causal output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 11. Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_backend_attribute_exists(self, device: torch.device) -> None:
        """Module exposes a backend attribute indicating the selected path."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        assert hasattr(mha, "backend")
        assert mha.backend in ("flash_attn_3", "flash_attn_2", "sdpa", "math")

    def test_backend_on_rtx3060_is_sdpa_or_fa2(self, device: torch.device) -> None:
        """On RTX 3060 (sm_86), backend should be sdpa or flash_attn_2 (not FA3 or math)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        # sm_86 is Ampere — should select FA2 (if installed) or SDPA
        assert mha.backend in ("flash_attn_2", "sdpa")

    def test_cpu_falls_back_to_math(self) -> None:
        """On CPU, backend should be 'math'."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4)
        # Force CPU
        mha = mha.to(torch.device("cpu"))
        # Re-detect backend for CPU
        assert mha.backend == "math"


# ---------------------------------------------------------------------------
# 12. Cross-attention
# ---------------------------------------------------------------------------


class TestCrossAttention:
    def test_cross_attention_shape(self, device: torch.device) -> None:
        """Cross-attention: Q from x, K/V from key_value with different seq len."""
        mha = MultiHeadAttention(embed_dim=128, num_heads=4).to(device)
        q_input = torch.randn(2, 10, 128, device=device)  # decoder
        kv_input = torch.randn(2, 50, 128, device=device)  # encoder
        out = mha(q_input, key_value=kv_input)
        assert out.shape == (2, 10, 128)

    def test_cross_attention_differs_from_self(self, device: torch.device) -> None:
        """Cross-attention output differs from self-attention on the same query."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        mha.eval()
        x = torch.randn(2, 10, 64, device=device)
        kv = torch.randn(2, 30, 64, device=device)

        out_self = mha(x)
        out_cross = mha(x, key_value=kv)
        assert not torch.allclose(out_self, out_cross, atol=1e-5)

    def test_cross_attention_no_causal(self, device: torch.device) -> None:
        """Cross-attention should not apply causal masking even if causal=True."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4, causal=True).to(device)
        mha.eval()
        q = torch.randn(2, 10, 64, device=device)
        kv = torch.randn(2, 30, 64, device=device)
        out = mha(q, key_value=kv)
        assert out.shape == (2, 10, 64)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 13. Belief network config (dropout=0.2)
# ---------------------------------------------------------------------------


class TestBeliefNetConfig:
    def test_belief_net_encoder_config(self, device: torch.device) -> None:
        """Belief encoder: dim=512, heads=8, dropout=0.2, causal=False."""
        mha = MultiHeadAttention(
            embed_dim=512, num_heads=8, dropout=0.2, causal=False
        ).to(device)
        x = torch.randn(2, 92, 512, device=device)
        out = mha(x)
        assert out.shape == (2, 92, 512)

    def test_belief_net_decoder_config(self, device: torch.device) -> None:
        """Belief decoder: dim=512, heads=8, dropout=0.2, causal=True."""
        mha = MultiHeadAttention(
            embed_dim=512, num_heads=8, dropout=0.2, causal=True
        ).to(device)
        x = torch.randn(2, 92, 512, device=device)
        out = mha(x)
        assert out.shape == (2, 92, 512)
