"""Tests for stratego.networks.belief_net.BeliefNetwork.

TDD: these tests are written before the implementation.
Covers: encoder/decoder shapes, param count, dropout behavior, batch sizes,
sequence lengths, backward pass, bf16 on CUDA, output finiteness, Kaiming
uniform init, cross-attention dependency, and hidden mask effect.
"""

from __future__ import annotations

import pytest
import torch

from stratego.networks.belief_net import BeliefNetwork

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 not supported on this device",
)


# ---------------------------------------------------------------------------
# 1. Encoder forward (target_tokens=None)
# ---------------------------------------------------------------------------


class TestEncoderForward:
    def test_encoder_output_shape(self, device: torch.device) -> None:
        """When target_tokens is None, returns encoder output (B, 100, dim)."""
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        out = net(infostate, hidden_mask)
        assert out.shape == (2, 100, 512)


# ---------------------------------------------------------------------------
# 2. Decoder forward (with target_tokens)
# ---------------------------------------------------------------------------


class TestDecoderForward:
    def test_decoder_output_shape(self, device: torch.device) -> None:
        """With target_tokens, returns logits (B, S, 12)."""
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)
        out = net(infostate, hidden_mask, target)
        assert out.shape == (2, 10, 12)


# ---------------------------------------------------------------------------
# 3. Parameter count
# ---------------------------------------------------------------------------


class TestParamCount:
    def test_param_count(self, device: torch.device) -> None:
        """Architecture per spec yields ~36M params.

        Paper reports 57.1M for the full belief network; the difference likely
        reflects additional components (e.g. value head, larger decoder) not
        included in this implementation.
        """
        net = BeliefNetwork().to(device)
        n_params = sum(p.numel() for p in net.parameters())
        assert 35_000_000 < n_params < 37_000_000


# ---------------------------------------------------------------------------
# 4. Dropout behavior
# ---------------------------------------------------------------------------


class TestDropout:
    def test_training_mode_output_varies(self, device: torch.device) -> None:
        """In training mode, dropout causes outputs to vary across calls."""
        net = BeliefNetwork().to(device)
        net.train()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)
        with torch.no_grad():
            out1 = net(infostate, hidden_mask, target)
            out2 = net(infostate, hidden_mask, target)
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_eval_mode_output_deterministic(self, device: torch.device) -> None:
        """In eval mode, no dropout -- outputs are deterministic."""
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)
        with torch.no_grad():
            out1 = net(infostate, hidden_mask, target)
            out2 = net(infostate, hidden_mask, target)
        assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Different batch sizes
# ---------------------------------------------------------------------------


class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size: int, device: torch.device) -> None:
        """Forward works for batch sizes 1, 4, 16."""
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(batch_size, 488, 10, 10, device=device)
        hidden_mask = torch.ones(batch_size, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (batch_size, 10), device=device)
        out = net(infostate, hidden_mask, target)
        assert out.shape == (batch_size, 10, 12)


# ---------------------------------------------------------------------------
# 6. Different sequence lengths
# ---------------------------------------------------------------------------


class TestSequenceLengths:
    @pytest.mark.parametrize("seq_len", [1, 5, 10, 40])
    def test_seq_lengths(self, seq_len: int, device: torch.device) -> None:
        """Decoder handles sequence lengths 1, 5, 10, 40 (max = 40 pieces)."""
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, seq_len), device=device)
        out = net(infostate, hidden_mask, target)
        assert out.shape == (2, seq_len, 12)


# ---------------------------------------------------------------------------
# 7. Backward pass
# ---------------------------------------------------------------------------


class TestBackward:
    def test_backward_pass(self, device: torch.device) -> None:
        """Gradients flow through the full network back to infostate."""
        net = BeliefNetwork().to(device)
        infostate = torch.randn(2, 488, 10, 10, device=device, requires_grad=True)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)
        out = net(infostate, hidden_mask, target)
        out.sum().backward()
        assert infostate.grad is not None
        assert infostate.grad.shape == infostate.shape
        assert infostate.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 8. bf16 on CUDA
# ---------------------------------------------------------------------------


class TestBFloat16:
    @requires_bf16
    def test_bf16_forward(self) -> None:
        """bf16 forward on CUDA produces finite outputs."""
        device = torch.device("cuda")
        net = BeliefNetwork().to(device, dtype=torch.bfloat16)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device, dtype=torch.bfloat16)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)
        with torch.no_grad():
            out = net(infostate, hidden_mask, target)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 9. Output finiteness
# ---------------------------------------------------------------------------


class TestFiniteness:
    def test_output_finite(self, device: torch.device) -> None:
        """Output logits are finite (no NaN/Inf)."""
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)
        out = net(infostate, hidden_mask, target)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 10. Kaiming uniform init for positional embeddings
# ---------------------------------------------------------------------------


class TestKaimingInit:
    def test_pos_emb_kaiming_uniform(self, device: torch.device) -> None:
        """Encoder and decoder positional embeddings use Kaiming uniform init.

        Kaiming uniform with a=sqrt(5) for (N, 512) produces values bounded by
        sqrt(1/512) ~= 0.0442, unlike normal(0, 0.1) which has std=0.1 and
        unbounded tails.
        """
        net = BeliefNetwork().to(device)
        enc_data = net.enc_pos_emb.embed.data
        dec_data = net.dec_pos_emb.embed.data

        # Kaiming uniform bound ~= 0.0442 -- all values strictly within [-0.045, 0.045]
        assert enc_data.abs().max().item() < 0.045
        assert dec_data.abs().max().item() < 0.045

        # std ~= 0.0255, much less than normal's 0.1
        assert enc_data.std().item() < 0.04
        assert dec_data.std().item() < 0.04

    def test_pos_emb_differs_from_normal(self, device: torch.device) -> None:
        """Kaiming uniform embeddings differ from normal(0, 0.1) embeddings."""
        from stratego.networks.transformer import PositionalEmbedding

        torch.manual_seed(42)
        net = BeliefNetwork().to(device)
        enc_data = net.enc_pos_emb.embed.data

        torch.manual_seed(42)
        normal_pe = PositionalEmbedding(max_len=100, embed_dim=512, init_std=0.1, init="normal").to(device)
        normal_data = normal_pe.embed.data

        # Normal init has larger max (unbounded tails) and larger std
        assert normal_data.abs().max().item() > enc_data.abs().max().item()
        assert normal_data.std().item() > enc_data.std().item()
        assert not torch.allclose(enc_data, normal_data, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. Cross-attention dependency
# ---------------------------------------------------------------------------


class TestCrossAttention:
    def test_encoder_affects_decoder(self, device: torch.device) -> None:
        """Different encoder inputs produce different decoder outputs.

        This proves the decoder uses cross-attention to the encoder output.
        """
        net = BeliefNetwork().to(device)
        net.eval()
        target = torch.randint(0, 12, (2, 10), device=device)
        hidden_mask = torch.ones(2, 100, dtype=torch.bool, device=device)

        infostate1 = torch.randn(2, 488, 10, 10, device=device)
        infostate2 = torch.randn(2, 488, 10, 10, device=device)

        with torch.no_grad():
            out1 = net(infostate1, hidden_mask, target)
            out2 = net(infostate2, hidden_mask, target)

        assert not torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# 12. Hidden mask effect
# ---------------------------------------------------------------------------


class TestHiddenMask:
    def test_hidden_mask_affects_output(self, device: torch.device) -> None:
        """Different hidden masks produce different decoder outputs.

        The hidden_mask controls which encoder tokens the decoder attends to
        via memory_key_padding_mask.
        """
        net = BeliefNetwork().to(device)
        net.eval()
        infostate = torch.randn(2, 488, 10, 10, device=device)
        target = torch.randint(0, 12, (2, 10), device=device)

        # All squares hidden
        mask_all = torch.ones(2, 100, dtype=torch.bool, device=device)
        # Only first 40 squares hidden
        mask_partial = torch.zeros(2, 100, dtype=torch.bool, device=device)
        mask_partial[:, :40] = True

        with torch.no_grad():
            out_all = net(infostate, mask_all, target)
            out_partial = net(infostate, mask_partial, target)

        assert not torch.allclose(out_all, out_partial, atol=1e-5)
