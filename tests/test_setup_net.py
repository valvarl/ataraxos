"""Tests for stratego.networks.setup_net.SetupNetwork.

TDD: these tests are written before the implementation.
Covers: forward pass shapes, parameter count (~12.6M), causal masking,
batch size and sequence length variation, backward pass, bf16 on CUDA,
output finiteness, head output shapes, and default config matching the paper.
"""

from __future__ import annotations

import pytest
import torch

from stratego.constants import (
    SETUP_NET_DEPTH,
    SETUP_NET_DIM,
    SETUP_NET_FF,
    SETUP_NET_HEADS,
    SETUP_NET_POS_EMB_INIT_STD,
    TOTAL_PIECES,
)
from stratego.networks.setup_net import SetupNetwork
from stratego.networks.transformer import PreLNTransformerBlock
from stratego.types import NUM_PIECE_TYPES

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 not supported on this device",
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# 1. Forward pass shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_forward_output_shapes(self, device: torch.device) -> None:
        """Forward returns (value (B,3), entropy (B,), policy (B,S,12))."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (4, 20), device=device)
        value, entropy, policy = net(tokens)
        assert value.shape == (4, 3)
        assert entropy.shape == (4,)
        assert policy.shape == (4, 20, NUM_PIECE_TYPES)

    def test_forward_output_dtypes(self, device: torch.device) -> None:
        """All outputs are float32 by default."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, 10), device=device)
        value, entropy, policy = net(tokens)
        assert value.dtype == torch.float32
        assert entropy.dtype == torch.float32
        assert policy.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. Parameter count
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_param_count_approx_12_6m(self) -> None:
        """Default config has ~12.6M parameters (paper: tab:setup-network-hyper)."""
        net = SetupNetwork()
        count = sum(p.numel() for p in net.parameters())
        # Computed: 12,647,440 — assert within [12.4M, 12.9M] of the paper's 12.6M.
        assert 12_400_000 < count < 12_900_000, f"Param count {count} not ~12.6M"

    def test_param_count_exact(self) -> None:
        """Exact parameter count matches the architectural breakdown."""
        net = SetupNetwork()
        count = sum(p.numel() for p in net.parameters())
        # token_emb(12*512) + pos_emb(40*512) + 4 blocks + 3 heads
        # attn has qkv_bias=False: 4 * (512*512) = 1,048,576 per block (no bias)
        # block: ln1(1024) + attn(1048576) + ln2(1024) + ff(1050624+1049088)
        # value_head: 1024 + 1539; entropy_head: 1024 + 513; policy_head: 1024 + 6156
        expected = 12_639_248
        assert count == expected, f"Param count {count} != {expected}"


# ---------------------------------------------------------------------------
# 3. Causal masking
# ---------------------------------------------------------------------------


class TestCausalMasking:
    def test_policy_position_independence(self, device: torch.device) -> None:
        """Policy at position i does not depend on tokens at positions > i."""
        torch.manual_seed(0)
        net = SetupNetwork().to(device)
        net.eval()
        seq_len = 20
        tokens1 = torch.randint(0, NUM_PIECE_TYPES, (2, seq_len), device=device)
        tokens2 = tokens1.clone()
        tokens2[:, 10:] = torch.randint(0, NUM_PIECE_TYPES, (2, seq_len - 10), device=device)
        with torch.no_grad():
            _, _, policy1 = net(tokens1)
            _, _, policy2 = net(tokens2)
        # Causal: positions 0..9 don't attend to positions >= 10, so they match.
        assert torch.allclose(policy1[:, :10], policy2[:, :10], atol=1e-5)
        # Positions 10..seq_len-1 had their inputs changed, so they differ.
        assert not torch.allclose(policy1[:, 10:], policy2[:, 10:], atol=1e-5)

    def test_value_entropy_independent_of_future(self, device: torch.device) -> None:
        """Value and entropy (first-token heads) are independent of later tokens."""
        torch.manual_seed(0)
        net = SetupNetwork().to(device)
        net.eval()
        seq_len = 20
        tokens1 = torch.randint(0, NUM_PIECE_TYPES, (2, seq_len), device=device)
        tokens2 = tokens1.clone()
        tokens2[:, 1:] = torch.randint(0, NUM_PIECE_TYPES, (2, seq_len - 1), device=device)
        with torch.no_grad():
            v1, e1, _ = net(tokens1)
            v2, e2, _ = net(tokens2)
        # Value/entropy use first token; under causal mask they don't see positions > 0
        assert torch.allclose(v1, v2, atol=1e-5)
        assert torch.allclose(e1, e2, atol=1e-5)

    def test_causal_differs_from_non_causal(self, device: torch.device) -> None:
        """A causal net and a non-causal net with the same weights produce different policies."""
        torch.manual_seed(0)
        causal_net = SetupNetwork().to(device)
        noncausal_net = SetupNetwork(causal=False).to(device)
        noncausal_net.load_state_dict(causal_net.state_dict())
        causal_net.eval()
        noncausal_net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, 20), device=device)
        with torch.no_grad():
            _, _, p_causal = causal_net(tokens)
            _, _, p_noncausal = noncausal_net(tokens)
        assert not torch.allclose(p_causal, p_noncausal, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Different batch sizes
# ---------------------------------------------------------------------------


class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size(self, batch_size: int, device: torch.device) -> None:
        """Forward works for batch sizes 1, 4, 16."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (batch_size, 20), device=device)
        value, entropy, policy = net(tokens)
        assert value.shape == (batch_size, 3)
        assert entropy.shape == (batch_size,)
        assert policy.shape == (batch_size, 20, NUM_PIECE_TYPES)


# ---------------------------------------------------------------------------
# 5. Different sequence lengths
# ---------------------------------------------------------------------------


class TestSequenceLengths:
    @pytest.mark.parametrize("seq_len", [1, 10, 40])
    def test_seq_len(self, seq_len: int, device: torch.device) -> None:
        """Forward works for sequence lengths 1, 10, 40 (max setup length)."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, seq_len), device=device)
        value, entropy, policy = net(tokens)
        assert value.shape == (2, 3)
        assert entropy.shape == (2,)
        assert policy.shape == (2, seq_len, NUM_PIECE_TYPES)


# ---------------------------------------------------------------------------
# 6. Backward pass
# ---------------------------------------------------------------------------


class TestBackwardPass:
    def test_gradients_flow(self, device: torch.device) -> None:
        """Gradients flow through all components of the setup network."""
        net = SetupNetwork().to(device)
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, 10), device=device)
        value, entropy, policy = net(tokens)
        loss = value.sum() + entropy.sum() + policy.sum()
        loss.backward()
        # Token embedding gradient
        assert net.token_emb.weight.grad is not None
        assert net.token_emb.weight.grad.abs().sum() > 0
        # Positional embedding gradient
        assert net.pos_emb.embed.grad is not None
        assert net.pos_emb.embed.grad.abs().sum() > 0
        # At least one block's attention has gradient
        for block in net.blocks.blocks:
            assert block.attn.q_proj.weight.grad is not None
            assert block.attn.q_proj.weight.grad.abs().sum() > 0

    def test_gradients_flow_into_input_embedding(self, device: torch.device) -> None:
        """Gradients reach the token embedding table for the tokens used."""
        net = SetupNetwork().to(device)
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        _, _, policy = net(tokens)
        policy.sum().backward()
        grad = net.token_emb.weight.grad
        assert grad is not None
        # Only the 4 used token indices should have non-zero gradient rows
        used = {0, 1, 2, 3}
        for i in range(NUM_PIECE_TYPES):
            row_grad = grad[i]
            if i in used:
                assert row_grad.abs().sum() > 0, f"Token {i} used but no gradient"
            else:
                assert row_grad.abs().sum() == 0, f"Token {i} unused but has gradient"


# ---------------------------------------------------------------------------
# 7. bf16 on CUDA
# ---------------------------------------------------------------------------


class TestBFloat16:
    @requires_bf16
    def test_bf16_forward(self) -> None:
        """bf16 forward on CUDA produces finite outputs with bf16 dtype."""
        device = torch.device("cuda")
        net = SetupNetwork().to(device, dtype=torch.bfloat16)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, 20), device=device)
        value, entropy, policy = net(tokens)
        assert value.dtype == torch.bfloat16
        assert entropy.dtype == torch.bfloat16
        assert policy.dtype == torch.bfloat16
        assert torch.isfinite(value).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(policy).all()

    @requires_bf16
    def test_bf16_backward(self) -> None:
        """bf16 backward on CUDA produces finite gradients."""
        device = torch.device("cuda")
        net = SetupNetwork().to(device, dtype=torch.bfloat16)
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, 10), device=device)
        value, entropy, policy = net(tokens)
        (value.sum() + entropy.sum() + policy.sum()).backward()
        assert net.token_emb.weight.grad is not None
        assert torch.isfinite(net.token_emb.weight.grad).all()
        assert net.pos_emb.embed.grad is not None
        assert torch.isfinite(net.pos_emb.embed.grad).all()


# ---------------------------------------------------------------------------
# 8. Output finiteness
# ---------------------------------------------------------------------------


class TestOutputFiniteness:
    def test_outputs_finite(self, device: torch.device) -> None:
        """No NaN/Inf in any output."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (4, 40), device=device)
        value, entropy, policy = net(tokens)
        assert torch.isfinite(value).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(policy).all()

    def test_outputs_finite_full_length(self, device: torch.device) -> None:
        """Full 40-token sequence produces finite outputs."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (1, TOTAL_PIECES), device=device)
        value, entropy, policy = net(tokens)
        assert torch.isfinite(value).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(policy).all()
        assert policy.shape == (1, TOTAL_PIECES, NUM_PIECE_TYPES)


# ---------------------------------------------------------------------------
# 9. Head output shapes
# ---------------------------------------------------------------------------


class TestHeadShapes:
    def test_value_head_3_logits(self, device: torch.device) -> None:
        """Value head produces exactly 3 logits (win/loss/draw)."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (4, 20), device=device)
        value, _, _ = net(tokens)
        assert value.shape == (4, 3)
        assert value.shape[-1] == 3

    def test_policy_head_12_way(self, device: torch.device) -> None:
        """Policy head produces 12-way distribution per position."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (4, 20), device=device)
        _, _, policy = net(tokens)
        assert policy.shape == (4, 20, NUM_PIECE_TYPES)
        assert policy.shape[-1] == NUM_PIECE_TYPES

    def test_entropy_head_scalar(self, device: torch.device) -> None:
        """Entropy head produces one scalar per batch element."""
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (4, 20), device=device)
        _, entropy, _ = net(tokens)
        assert entropy.shape == (4,)
        assert entropy.dim() == 1


# ---------------------------------------------------------------------------
# 10. Default config matches paper
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_depth(self) -> None:
        """Default depth is 4 (SETUP_NET_DEPTH)."""
        net = SetupNetwork()
        assert len(net.blocks.blocks) == SETUP_NET_DEPTH

    def test_default_dim(self) -> None:
        """Default embedding dim is 512 (SETUP_NET_DIM)."""
        net = SetupNetwork()
        assert net.token_emb.embedding_dim == SETUP_NET_DIM

    def test_default_num_heads(self) -> None:
        """Default num_heads is 8 (SETUP_NET_HEADS)."""
        from stratego.networks.attention import MultiHeadAttention

        net = SetupNetwork()
        block = net.blocks.blocks[0]
        assert isinstance(block, PreLNTransformerBlock)
        attn = block.attn
        assert isinstance(attn, MultiHeadAttention)
        assert attn.num_heads == SETUP_NET_HEADS

    def test_default_ff(self) -> None:
        """Default ff_dim is 2048 (SETUP_NET_FF)."""
        net = SetupNetwork()
        block = net.blocks.blocks[0]
        assert isinstance(block, PreLNTransformerBlock)
        first_child = next(iter(block.ff.children()))
        assert isinstance(first_child, torch.nn.Linear)
        assert first_child.out_features == SETUP_NET_FF

    def test_default_pos_emb_std(self) -> None:
        """Default positional embedding init std is 0.1."""
        # Re-init with known seed and check std is near 0.1
        torch.manual_seed(0)
        net = SetupNetwork()
        std = net.pos_emb.embed.data.std().item()
        assert abs(std - SETUP_NET_POS_EMB_INIT_STD) < 0.02

    def test_default_num_piece_types(self) -> None:
        """Default vocabulary size is 12 (NUM_PIECE_TYPES)."""
        net = SetupNetwork()
        assert net.token_emb.num_embeddings == NUM_PIECE_TYPES
        assert net.num_piece_types == NUM_PIECE_TYPES

    def test_default_causal(self) -> None:
        """Default config uses causal self-attention."""
        net = SetupNetwork()
        for block in net.blocks.blocks:
            assert block.attn.causal is True

    def test_pos_emb_max_len_40(self) -> None:
        """Positional embedding table has 40 entries (TOTAL_PIECES)."""
        net = SetupNetwork()
        assert net.pos_emb.embed.shape[0] == TOTAL_PIECES


# ---------------------------------------------------------------------------
# 11. Determinism (eval mode)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_eval_mode_deterministic(self, device: torch.device) -> None:
        """In eval mode, the same input produces the same output."""
        torch.manual_seed(0)
        net = SetupNetwork().to(device)
        net.eval()
        tokens = torch.randint(0, NUM_PIECE_TYPES, (2, 20), device=device)
        with torch.no_grad():
            v1, e1, p1 = net(tokens)
            v2, e2, p2 = net(tokens)
        assert torch.allclose(v1, v2, atol=1e-6)
        assert torch.allclose(e1, e2, atol=1e-6)
        assert torch.allclose(p1, p2, atol=1e-6)
