"""Tests for stratego.networks.move_net.MoveNetwork.

TDD: these tests are written before the implementation.
Covers: KeyQueryPolicyHead shape/masking, MoveNetwork forward shapes, parameter
count (~14.7M), batch size variation, backward pass, bf16 on CUDA, output
finiteness, value head 3 logits, asymmetric Q·K^T matrix, default config, and
eval-mode determinism.
"""

from __future__ import annotations

import pytest
import torch

from stratego.constants import (
    MOVE_NET_DEPTH,
    MOVE_NET_DIM,
    MOVE_NET_FF,
    MOVE_NET_HEADS,
    MOVE_NET_POS_EMB_INIT_STD,
    NUM_INFOSTATE_CHANNELS,
    NUM_SQUARES,
)
from stratego.networks.move_net import KeyQueryPolicyHead, MoveNetwork
from stratego.networks.transformer import PreLNTransformerBlock

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 not supported on this device",
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# 1. KeyQueryPolicyHead output shape
# ---------------------------------------------------------------------------


class TestKeyQueryPolicyHeadShape:
    def test_output_shape(self, device: torch.device) -> None:
        """Head returns (B, 100*100) = (B, 10000) flattened logits."""
        head = KeyQueryPolicyHead(MOVE_NET_DIM).to(device)
        head.eval()
        x = torch.randn(4, NUM_SQUARES, MOVE_NET_DIM, device=device)
        out = head(x)
        assert out.shape == (4, NUM_SQUARES * NUM_SQUARES)

    def test_output_shape_batch1(self, device: torch.device) -> None:
        """Single batch element also yields (1, 10000)."""
        head = KeyQueryPolicyHead(MOVE_NET_DIM).to(device)
        head.eval()
        x = torch.randn(1, NUM_SQUARES, MOVE_NET_DIM, device=device)
        out = head(x)
        assert out.shape == (1, NUM_SQUARES * NUM_SQUARES)


# ---------------------------------------------------------------------------
# 2. KeyQueryPolicyHead masking
# ---------------------------------------------------------------------------


class TestKeyQueryPolicyHeadMasking:
    def test_illegal_moves_masked(self, device: torch.device) -> None:
        """Illegal positions are filled with -1e10."""
        head = KeyQueryPolicyHead(MOVE_NET_DIM).to(device)
        head.eval()
        x = torch.randn(2, NUM_SQUARES, MOVE_NET_DIM, device=device)
        # Only one (i, j) pair legal per batch element.
        mask = torch.zeros(2, NUM_SQUARES, NUM_SQUARES, dtype=torch.bool, device=device)
        mask[0, 3, 5] = True
        mask[1, 7, 2] = True
        out = head(x, legal_move_mask=mask)
        out_2d = out.view(2, NUM_SQUARES, NUM_SQUARES)
        # Legal positions are NOT -1e10.
        assert out_2d[0, 3, 5] > -1e9
        assert out_2d[1, 7, 2] > -1e9
        # All other positions ARE -1e10.
        illegal_count = (out_2d <= -1e9).sum().item()
        assert illegal_count == 2 * NUM_SQUARES * NUM_SQUARES - 2

    def test_no_mask_no_fill(self, device: torch.device) -> None:
        """Without a mask, no position is -1e10."""
        head = KeyQueryPolicyHead(MOVE_NET_DIM).to(device)
        head.eval()
        x = torch.randn(2, NUM_SQUARES, MOVE_NET_DIM, device=device)
        out = head(x)
        assert (out > -1e9).all()


# ---------------------------------------------------------------------------
# 3. MoveNetwork forward shapes
# ---------------------------------------------------------------------------


class TestMoveNetForwardShapes:
    def test_forward_output_shapes(self, device: torch.device) -> None:
        """Forward returns value (B, 3) and policy (B, 10000)."""
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(4, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        value, policy = net(infostate)
        assert value.shape == (4, 3)
        assert policy.shape == (4, NUM_SQUARES * NUM_SQUARES)

    def test_forward_output_dtypes(self, device: torch.device) -> None:
        """Outputs are float32 by default."""
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        value, policy = net(infostate)
        assert value.dtype == torch.float32
        assert policy.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. Parameter count
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_param_count_approx_14_7m(self) -> None:
        """Default config has ~14.7M parameters (paper: tab:move-network-hyper)."""
        net = MoveNetwork()
        count = sum(p.numel() for p in net.parameters())
        # Computed: ~14.86M — assert within [13.7M, 15.7M] (±1M of 14.7M).
        assert 13_700_000 < count < 15_700_000, f"Param count {count} not ~14.7M"


# ---------------------------------------------------------------------------
# 5. Different batch sizes
# ---------------------------------------------------------------------------


class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size(self, batch_size: int, device: torch.device) -> None:
        """Forward works for batch sizes 1, 4, 16."""
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(batch_size, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        value, policy = net(infostate)
        assert value.shape == (batch_size, 3)
        assert policy.shape == (batch_size, NUM_SQUARES * NUM_SQUARES)


# ---------------------------------------------------------------------------
# 6. Backward pass
# ---------------------------------------------------------------------------


class TestBackwardPass:
    def test_gradients_flow(self, device: torch.device) -> None:
        """Gradients flow through all components of the move network."""
        net = MoveNetwork().to(device)
        infostate = torch.randn(2, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        value, policy = net(infostate)
        loss = value.sum() + policy.sum()
        loss.backward()
        # Input projection gradient
        assert net.input_proj.weight.grad is not None
        assert net.input_proj.weight.grad.abs().sum() > 0
        # Positional embedding gradient
        assert net.pos_emb.embed.grad is not None
        assert net.pos_emb.embed.grad.abs().sum() > 0
        # Value token gradient
        assert net.value_token.grad is not None
        assert net.value_token.grad.abs().sum() > 0
        # At least one block's attention has gradient
        for block in net.blocks.blocks:
            assert isinstance(block, PreLNTransformerBlock)
            assert block.attn.q_proj.weight.grad is not None
            assert block.attn.q_proj.weight.grad.abs().sum() > 0
        # Policy head gradients
        assert net.policy_head.wq.weight.grad is not None
        assert net.policy_head.wq.weight.grad.abs().sum() > 0
        assert net.policy_head.wk.weight.grad is not None
        assert net.policy_head.wk.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 7. bf16 on CUDA
# ---------------------------------------------------------------------------


class TestBFloat16:
    @requires_bf16
    def test_bf16_forward(self) -> None:
        """bf16 forward on CUDA produces finite outputs with bf16 dtype."""
        device = torch.device("cuda")
        net = MoveNetwork().to(device, dtype=torch.bfloat16)
        net.eval()
        infostate = torch.randn(2, NUM_INFOSTATE_CHANNELS, 10, 10, device=device).to(torch.bfloat16)
        value, policy = net(infostate)
        assert value.dtype == torch.bfloat16
        assert policy.dtype == torch.bfloat16
        assert torch.isfinite(value).all()
        assert torch.isfinite(policy).all()

    @requires_bf16
    def test_bf16_backward(self) -> None:
        """bf16 backward on CUDA produces finite gradients."""
        device = torch.device("cuda")
        net = MoveNetwork().to(device, dtype=torch.bfloat16)
        infostate = torch.randn(2, NUM_INFOSTATE_CHANNELS, 10, 10, device=device).to(torch.bfloat16)
        value, policy = net(infostate)
        (value.sum() + policy.sum()).backward()
        assert net.input_proj.weight.grad is not None
        assert torch.isfinite(net.input_proj.weight.grad).all()
        assert net.pos_emb.embed.grad is not None
        assert torch.isfinite(net.pos_emb.embed.grad).all()


# ---------------------------------------------------------------------------
# 8. Output finiteness
# ---------------------------------------------------------------------------


class TestOutputFiniteness:
    def test_outputs_finite(self, device: torch.device) -> None:
        """No NaN/Inf in any output."""
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(4, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        value, policy = net(infostate)
        assert torch.isfinite(value).all()
        assert torch.isfinite(policy).all()

    def test_outputs_finite_with_mask(self, device: torch.device) -> None:
        """Outputs remain finite when a legal-move mask is applied."""
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        mask = torch.ones(2, NUM_SQUARES, NUM_SQUARES, dtype=torch.bool, device=device)
        value, policy = net(infostate, legal_move_mask=mask)
        assert torch.isfinite(value).all()
        assert torch.isfinite(policy).all()


# ---------------------------------------------------------------------------
# 9. Value head 3 logits
# ---------------------------------------------------------------------------


class TestValueHead:
    def test_value_head_3_logits(self, device: torch.device) -> None:
        """Value head produces exactly 3 logits (win/loss/draw)."""
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(4, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        value, _ = net(infostate)
        assert value.shape == (4, 3)
        assert value.shape[-1] == 3


# ---------------------------------------------------------------------------
# 10. Asymmetric matrix (Q·K^T not symmetric)
# ---------------------------------------------------------------------------


class TestAsymmetricMatrix:
    def test_logit_matrix_not_symmetric(self, device: torch.device) -> None:
        """Q·K^T is asymmetric because wq != wk."""
        torch.manual_seed(0)
        head = KeyQueryPolicyHead(MOVE_NET_DIM).to(device)
        head.eval()
        x = torch.randn(1, NUM_SQUARES, MOVE_NET_DIM, device=device)
        with torch.no_grad():
            out = head(x)
        matrix = out.view(NUM_SQUARES, NUM_SQUARES)
        # The matrix should not be symmetric: M != M^T somewhere.
        asymmetry = (matrix - matrix.t()).abs()
        assert asymmetry.max() > 1e-5, "Q·K^T matrix is symmetric (wq == wk?)"


# ---------------------------------------------------------------------------
# 11. Default config matches paper
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_depth(self) -> None:
        """Default depth is 8 (MOVE_NET_DEPTH)."""
        net = MoveNetwork()
        assert len(net.blocks.blocks) == MOVE_NET_DEPTH

    def test_default_dim(self) -> None:
        """Default embedding dim is 384 (MOVE_NET_DIM)."""
        net = MoveNetwork()
        assert net.input_proj.out_features == MOVE_NET_DIM

    def test_default_num_heads(self) -> None:
        """Default num_heads is 8 (MOVE_NET_HEADS)."""
        from stratego.networks.attention import MultiHeadAttention

        net = MoveNetwork()
        block = net.blocks.blocks[0]
        assert isinstance(block, PreLNTransformerBlock)
        attn = block.attn
        assert isinstance(attn, MultiHeadAttention)
        assert attn.num_heads == MOVE_NET_HEADS

    def test_default_ff(self) -> None:
        """Default ff_dim is 1536 (MOVE_NET_FF)."""
        net = MoveNetwork()
        block = net.blocks.blocks[0]
        assert isinstance(block, PreLNTransformerBlock)
        first_child = next(iter(block.ff.children()))
        assert isinstance(first_child, torch.nn.Linear)
        assert first_child.out_features == MOVE_NET_FF

    def test_default_pos_emb_std(self) -> None:
        """Default positional embedding init std is 0.1."""
        torch.manual_seed(0)
        net = MoveNetwork()
        std = net.pos_emb.embed.data.std().item()
        assert abs(std - MOVE_NET_POS_EMB_INIT_STD) < 0.02

    def test_default_non_causal(self) -> None:
        """Default config uses non-causal (bidirectional) self-attention."""
        net = MoveNetwork()
        for block in net.blocks.blocks:
            assert isinstance(block, PreLNTransformerBlock)
            assert block.attn.causal is False

    def test_pos_emb_max_len_101(self) -> None:
        """Positional embedding table has 101 entries (100 squares + 1 value token)."""
        net = MoveNetwork()
        assert net.pos_emb.embed.shape[0] == NUM_SQUARES + 1


# ---------------------------------------------------------------------------
# 12. Determinism (eval mode)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_eval_mode_deterministic(self, device: torch.device) -> None:
        """In eval mode, the same input produces the same output."""
        torch.manual_seed(0)
        net = MoveNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, NUM_INFOSTATE_CHANNELS, 10, 10, device=device)
        with torch.no_grad():
            v1, p1 = net(infostate)
            v2, p2 = net(infostate)
        assert torch.allclose(v1, v2, atol=1e-6)
        assert torch.allclose(p1, p2, atol=1e-6)
