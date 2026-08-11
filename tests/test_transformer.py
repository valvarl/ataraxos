"""Tests for stratego.networks.transformer building blocks.

TDD: these tests are written before the implementation.
Covers: PositionalEmbedding (learned abs), PreLNTransformerBlock,
TransformerStack, ValueHead, EntropyHead, PieceDistributionHead,
backward pass, bf16 on CUDA, variable batch sizes and sequence lengths,
and the three paper network configs (setup / move / belief).
"""

from __future__ import annotations

import pytest
import torch

from stratego.networks.transformer import (
    EntropyHead,
    PieceDistributionHead,
    PositionalEmbedding,
    PreLNTransformerBlock,
    TransformerStack,
    ValueHead,
)

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 not supported on this device",
)

# ---------------------------------------------------------------------------
# 1. PositionalEmbedding
# ---------------------------------------------------------------------------


class TestPositionalEmbedding:
    def test_output_shape_matches_input(self, device: torch.device) -> None:
        """Input (B, S, E) -> output (B, S, E)."""
        pe = PositionalEmbedding(max_len=128, embed_dim=64).to(device)
        x = torch.randn(4, 10, 64, device=device)
        out = pe(x)
        assert out.shape == (4, 10, 64)

    def test_additive_not_replacement(self, device: torch.device) -> None:
        """Output should be input + pos_emb, not a replacement of input."""
        pe = PositionalEmbedding(max_len=128, embed_dim=64).to(device)
        pe.eval()
        x = torch.randn(2, 10, 64, device=device)
        out = pe(x)
        # The difference should equal the positional embedding broadcast
        diff = out - x
        assert diff.shape == x.shape
        # All positions should have a non-zero contribution (init std=0.1)
        assert diff.abs().sum() > 0

    def test_init_normal_std(self, device: torch.device) -> None:
        """init='normal' with std=0.1 produces embeddings with std near 0.1."""
        torch.manual_seed(0)
        pe = PositionalEmbedding(max_len=1000, embed_dim=512, init_std=0.1, init="normal").to(device)
        std = pe.embed.data.std().item()
        # With 1000*512 samples, std should be close to 0.1
        assert 0.08 < std < 0.12

    def test_init_kaiming_uniform(self, device: torch.device) -> None:
        """init='kaiming_uniform' produces a bounded uniform distribution, unlike normal."""
        torch.manual_seed(0)
        pe_k = PositionalEmbedding(max_len=256, embed_dim=128, init="kaiming_uniform").to(device)
        torch.manual_seed(0)
        pe_n = PositionalEmbedding(max_len=256, embed_dim=128, init_std=0.1, init="normal").to(device)
        k_data = pe_k.embed.data
        n_data = pe_n.embed.data
        # For (256,128) with a=sqrt(5): bound = sqrt(2/6) * sqrt(3/128) ≈ 0.0884.
        assert k_data.abs().max().item() < 0.1
        assert n_data.abs().max().item() > k_data.abs().max().item()
        assert not torch.allclose(k_data, n_data, atol=1e-6)

    def test_offset_shifts_positions(self, device: torch.device) -> None:
        """offset shifts which slice of the positional embedding is used."""
        pe = PositionalEmbedding(max_len=128, embed_dim=32).to(device)
        pe.eval()
        x = torch.randn(1, 4, 32, device=device)
        out_no_offset = pe(x, offset=0)
        out_offset = pe(x, offset=10)
        # Different offsets must yield different outputs for the same x
        assert not torch.allclose(out_no_offset, out_offset, atol=1e-6)

    def test_offset_uses_correct_slice(self, device: torch.device) -> None:
        """pe(x, offset=k) adds embed[k:k+S] to x."""
        pe = PositionalEmbedding(max_len=128, embed_dim=32).to(device)
        pe.eval()
        x = torch.randn(1, 4, 32, device=device)
        k = 7
        out = pe(x, offset=k)
        expected = x + pe.embed.data[k : k + 4].unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. PreLNTransformerBlock
# ---------------------------------------------------------------------------


class TestPreLNTransformerBlock:
    def test_forward_shape(self, device: torch.device) -> None:
        """(B, S, E) -> (B, S, E)."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self, device: torch.device) -> None:
        """With attn and ff outputs zeroed, output equals input (residual path)."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device)
        block.eval()
        # Zero out attention output projection -> attn output is 0
        with torch.no_grad():
            block.attn.out_proj.weight.zero_()
            if block.attn.out_proj.bias is not None:
                block.attn.out_proj.bias.zero_()
            # Zero out FFN final layer -> ff output is 0
            block.ff[2].weight.zero_()
            block.ff[2].bias.zero_()
        x = torch.randn(2, 16, 64, device=device)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_causal_masking_differs_from_non_causal(self, device: torch.device) -> None:
        """Causal and non-causal blocks produce different outputs for the same input."""
        torch.manual_seed(0)
        x = torch.randn(2, 20, 64, device=device)
        block_c = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256, causal=True).to(device)
        block_nc = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256, causal=False).to(device)
        block_nc.load_state_dict(block_c.state_dict())
        block_c.eval()
        block_nc.eval()
        out_c = block_c(x)
        out_nc = block_nc(x)
        assert not torch.allclose(out_c, out_nc, atol=1e-5)

    def test_causal_first_token_independent(self, device: torch.device) -> None:
        """With causal masking, the first output token is independent of later tokens."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256, causal=True).to(device)
        block.eval()
        x1 = torch.randn(1, 10, 64, device=device)
        x2 = x1.clone()
        x2[0, 5:] = torch.randn(5, 64, device=device)
        out1 = block(x1)
        out2 = block(x2)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)

    def test_pre_ln_submodule_layout(self, device: torch.device) -> None:
        """Block exposes ln1, attn, ln2, ff submodules in pre-LN order."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device)
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")
        assert hasattr(block, "attn")
        assert hasattr(block, "ff")
        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)


# ---------------------------------------------------------------------------
# 3. TransformerStack
# ---------------------------------------------------------------------------


class TestTransformerStack:
    def test_depth_creates_n_blocks(self, device: torch.device) -> None:
        """Stack with depth=N has exactly N PreLNTransformerBlocks."""
        stack = TransformerStack(depth=6, embed_dim=64, num_heads=4, ff_dim=256).to(device)
        assert len(stack.blocks) == 6
        for block in stack.blocks:
            assert isinstance(block, PreLNTransformerBlock)

    def test_forward_shape(self, device: torch.device) -> None:
        """(B, S, E) -> (B, S, E)."""
        stack = TransformerStack(depth=4, embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = stack(x)
        assert out.shape == (2, 16, 64)

    def test_setup_net_config(self, device: torch.device) -> None:
        """Setup net: dim=512, heads=8, ff=2048, depth=4, causal=True."""
        stack = TransformerStack(depth=4, embed_dim=512, num_heads=8, ff_dim=2048, causal=True).to(device)
        x = torch.randn(2, 92, 512, device=device)
        out = stack(x)
        assert out.shape == (2, 92, 512)
        assert len(stack.blocks) == 4

    def test_move_net_config(self, device: torch.device) -> None:
        """Move net: dim=384, heads=8, ff=1536, depth=8, causal=True."""
        stack = TransformerStack(depth=8, embed_dim=384, num_heads=8, ff_dim=1536, causal=True).to(device)
        x = torch.randn(2, 92, 384, device=device)
        out = stack(x)
        assert out.shape == (2, 92, 384)
        assert len(stack.blocks) == 8

    def test_belief_net_config(self, device: torch.device) -> None:
        """Belief net encoder: dim=512, heads=8, ff=2048, depth=6, dropout=0.2."""
        stack = TransformerStack(depth=6, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.2).to(device)
        x = torch.randn(2, 92, 512, device=device)
        out = stack(x)
        assert out.shape == (2, 92, 512)
        assert len(stack.blocks) == 6


# ---------------------------------------------------------------------------
# 4. ValueHead
# ---------------------------------------------------------------------------


class TestValueHead:
    def test_output_shape_3d(self, device: torch.device) -> None:
        """(B, S, E) -> (B, 3) for [win, loss, draw]."""
        head = ValueHead(embed_dim=64).to(device)
        x = torch.randn(4, 16, 64, device=device)
        out = head(x)
        assert out.shape == (4, 3)

    def test_output_shape_2d(self, device: torch.device) -> None:
        """(B, E) -> (B, 3)."""
        head = ValueHead(embed_dim=64).to(device)
        x = torch.randn(4, 64, device=device)
        out = head(x)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# 5. EntropyHead
# ---------------------------------------------------------------------------


class TestEntropyHead:
    def test_output_shape_3d(self, device: torch.device) -> None:
        """(B, S, E) -> (B,) scalar per batch element."""
        head = EntropyHead(embed_dim=64).to(device)
        x = torch.randn(4, 16, 64, device=device)
        out = head(x)
        assert out.shape == (4,)

    def test_output_shape_2d(self, device: torch.device) -> None:
        """(B, E) -> (B,)."""
        head = EntropyHead(embed_dim=64).to(device)
        x = torch.randn(4, 64, device=device)
        out = head(x)
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# 6. PieceDistributionHead
# ---------------------------------------------------------------------------


class TestPieceDistributionHead:
    def test_output_shape_default(self, device: torch.device) -> None:
        """(B, S, E) -> (B, S, 12) by default."""
        head = PieceDistributionHead(embed_dim=64).to(device)
        x = torch.randn(4, 16, 64, device=device)
        out = head(x)
        assert out.shape == (4, 16, 12)

    def test_output_shape_custom_num_types(self, device: torch.device) -> None:
        """Custom num_types is respected."""
        head = PieceDistributionHead(embed_dim=64, num_types=10).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = head(x)
        assert out.shape == (2, 8, 10)


# ---------------------------------------------------------------------------
# 7. Backward pass
# ---------------------------------------------------------------------------


class TestBackwardPass:
    def test_gradients_flow_block(self, device: torch.device) -> None:
        """Gradients flow through PreLNTransformerBlock."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(2, 10, 64, device=device, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().sum() > 0

    def test_gradients_flow_stack(self, device: torch.device) -> None:
        """Gradients flow through the full TransformerStack."""
        stack = TransformerStack(depth=4, embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(2, 10, 64, device=device, requires_grad=True)
        out = stack(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradients_flow_positional(self, device: torch.device) -> None:
        """Gradients flow into the positional embedding parameter."""
        pe = PositionalEmbedding(max_len=128, embed_dim=64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = pe(x)
        out.sum().backward()
        assert pe.embed.grad is not None
        assert pe.embed.grad.abs().sum() > 0

    def test_gradients_flow_heads(self, device: torch.device) -> None:
        """Gradients flow through ValueHead, EntropyHead, PieceDistributionHead."""
        x = torch.randn(4, 16, 64, device=device, requires_grad=True)
        vh = ValueHead(embed_dim=64).to(device)
        eh = EntropyHead(embed_dim=64).to(device)
        pdh = PieceDistributionHead(embed_dim=64).to(device)
        (vh(x).sum() + eh(x).sum() + pdh(x).sum()).backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 8. bf16 on CUDA
# ---------------------------------------------------------------------------


class TestBFloat16:
    @requires_bf16
    def test_bf16_forward_block(self) -> None:
        """PreLNTransformerBlock forward in bf16 produces finite outputs."""
        device = torch.device("cuda")
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device, dtype=torch.bfloat16)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        out = block(x)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    @requires_bf16
    def test_bf16_backward_stack(self) -> None:
        """TransformerStack backward in bf16 produces finite gradients."""
        device = torch.device("cuda")
        stack = TransformerStack(depth=2, embed_dim=64, num_heads=4, ff_dim=256).to(device, dtype=torch.bfloat16)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        out = stack(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 9. Different batch sizes
# ---------------------------------------------------------------------------


class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_block_batch_sizes(self, batch_size: int, device: torch.device) -> None:
        """Block forward works for batch sizes 1, 4, 16."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(batch_size, 16, 64, device=device)
        out = block(x)
        assert out.shape == (batch_size, 16, 64)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_stack_batch_sizes(self, batch_size: int, device: torch.device) -> None:
        """Stack forward works for batch sizes 1, 4, 16."""
        stack = TransformerStack(depth=4, embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(batch_size, 16, 64, device=device)
        out = stack(x)
        assert out.shape == (batch_size, 16, 64)


# ---------------------------------------------------------------------------
# 10. Different sequence lengths
# ---------------------------------------------------------------------------


class TestSequenceLengths:
    @pytest.mark.parametrize("seq_len", [10, 92, 100, 101])
    def test_block_seq_lengths(self, seq_len: int, device: torch.device) -> None:
        """Block forward works for sequence lengths 10, 92, 100, 101."""
        block = PreLNTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256).to(device)
        x = torch.randn(2, seq_len, 64, device=device)
        out = block(x)
        assert out.shape == (2, seq_len, 64)

    @pytest.mark.parametrize("seq_len", [10, 92, 100, 101])
    def test_positional_seq_lengths(self, seq_len: int, device: torch.device) -> None:
        """PositionalEmbedding works for sequence lengths up to 101."""
        pe = PositionalEmbedding(max_len=128, embed_dim=64).to(device)
        x = torch.randn(2, seq_len, 64, device=device)
        out = pe(x)
        assert out.shape == (2, seq_len, 64)
