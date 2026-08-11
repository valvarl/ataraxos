"""Tests for compute_infostate_tensor() in the StrategoRolloutBuffer CUDA extension.

Verifies the (N, 488, 10, 10) float32 infostate tensor matches the channel layout
described in stratego/env/infostate.py:
  0-11:   own piece type one-hot
  12-23:  opp piece type probs (uniform 1/12 hidden, one-hot revealed)
  24-35:  mirror (opp's view of own pieces)
  36:     own hidden
  37:     opp hidden
  38:     empty squares
  39:     own moved pieces
  40:     opp moved pieces
  41:     move_number / 4000.0
  42:     moves_since_attack / 100.0
  43-455: zeros (deferred)
  456-487: last 32 moves (+1 at dst, -1 at src)

All tests are skipped when CUDA is unavailable or the extension fails to build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"

PT_NONE = 0
PT_SPY = 1
PT_SCOUT = 2
PT_SERGEANT = 4
PT_MARSHAL = 10
PT_FLAG = 11
PT_BOMB = 12

PLAYER_RED = 0
PLAYER_BLUE = 1
PLAYER_EMPTY = -1

BOARD_ROWS = 10
BOARD_COLS = 10
NUM_SQUARES = BOARD_ROWS * BOARD_COLS
NUM_INFOSTATE_CHANNELS = 488
UNIFORM_PROB = 1.0 / 12.0


def _encode_action(src_r: int, src_c: int, dst_r: int, dst_c: int) -> int:
    return (src_r * BOARD_COLS + src_c) * NUM_SQUARES + (dst_r * BOARD_COLS + dst_c)


def _make_setup(pieces_red, pieces_blue, n_games=1):
    setup_red = torch.zeros((n_games, BOARD_ROWS, BOARD_COLS), dtype=torch.int8)
    setup_blue = torch.zeros((n_games, BOARD_ROWS, BOARD_COLS), dtype=torch.int8)
    for r, c, p in pieces_red:
        setup_red[:, r, c] = p
    for r, c, p in pieces_blue:
        setup_blue[:, r, c] = p
    return setup_red, setup_blue


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

_skip = pytest.mark.skipif(
    _stratego_cuda is None, reason="_stratego_cuda extension not built (no CUDA or build error)"
)


def _buf(n_games=1):
    return _stratego_cuda.StrategoRolloutBuffer(n_games, 0)


@_skip
class TestComputeInfostateTensor:
    """Tests for StrategoRolloutBuffer.compute_infostate_tensor()."""

    def test_shape_dtype_device(self):
        """compute_infostate_tensor must return (N, 488, 10, 10) float32 CUDA tensor."""
        buf = _buf(3)
        red, blue = _make_setup([(3, 0, PT_SPY)], [(6, 0, PT_FLAG)], n_games=3)
        buf.reset_all(red, blue)

        info = buf.compute_infostate_tensor()
        assert info.shape == (3, NUM_INFOSTATE_CHANNELS, BOARD_ROWS, BOARD_COLS)
        assert info.dtype == torch.float32
        assert info.is_cuda

    def test_initial_state_channels(self):
        """Fresh reset: own one-hot, opp uniform, hidden, empty, deferred zeros."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [(6, 0, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)

        info = buf.compute_infostate_tensor()
        # Red's perspective (current_player = RED after reset)
        # Channel 0 (SPY): own piece one-hot at (3,0)
        assert info[0, 0, 3, 0].item() == pytest.approx(1.0)
        # Channel 10 (FLAG): own piece one-hot — Red has no Flag → 0.0 everywhere
        assert info[0, 10].sum().item() == pytest.approx(0.0)

        # Channels 12-23: opp piece probs — Blue Flag at (6,0) is hidden → uniform 1/12
        for ch in range(12, 24):
            assert info[0, ch, 6, 0].item() == pytest.approx(UNIFORM_PROB)
        # Sum of opp probs at (6,0) should be 1.0
        assert info[0, 12:24, 6, 0].sum().item() == pytest.approx(1.0)

        # Channel 36: own hidden — Red Spy at (3,0) is hidden → 1.0
        assert info[0, 36, 3, 0].item() == pytest.approx(1.0)
        # Channel 37: opp hidden — Blue Flag at (6,0) is hidden → 1.0
        assert info[0, 37, 6, 0].item() == pytest.approx(1.0)
        # Channel 38: empty squares — (0,0) is empty → 1.0
        assert info[0, 38, 0, 0].item() == pytest.approx(1.0)
        # Channel 38: (3,0) is not empty → 0.0
        assert info[0, 38, 3, 0].item() == pytest.approx(0.0)

        # Channels 43-455: deferred — all zeros
        assert info[0, 43:456].abs().sum().item() == pytest.approx(0.0)

        # Channels 456-487: no moves yet — all zeros
        assert info[0, 456:488].abs().sum().item() == pytest.approx(0.0)

    def test_after_simple_move(self):
        """After Red Spy moves to empty square: channel 456 encodes the move."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [(6, 0, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64)
        buf.apply_actions(actions)

        info = buf.compute_infostate_tensor()
        # After Red's move, current_player = Blue → Blue's perspective
        # Channel 456 (most recent move): +1 at dst (4,0), -1 at src (3,0)
        assert info[0, 456, 4, 0].item() == pytest.approx(1.0)
        assert info[0, 456, 3, 0].item() == pytest.approx(-1.0)

        # Channel 40 (opp moved): Red Spy moved to (4,0), Red is opp from Blue's view → 1.0
        assert info[0, 40, 4, 0].item() == pytest.approx(1.0)
        # (3,0) is now empty → opp_mask false → 0.0
        assert info[0, 40, 3, 0].item() == pytest.approx(0.0)

        # Channel 37 (opp hidden): Red Spy at (4,0) is NOT revealed (simple move) → 1.0
        assert info[0, 37, 4, 0].item() == pytest.approx(1.0)

        # Channels 457-487: only 1 move in history → zeros
        assert info[0, 457:488].abs().sum().item() == pytest.approx(0.0)

    def test_after_attack_revealed(self):
        """After Red Spy attacks Blue Marshal: opp piece is revealed (one-hot)."""
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SPY)],
            [(4, 0, PT_MARSHAL), (6, 9, PT_SCOUT)],
            n_games=1,
        )
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64)
        buf.apply_actions(actions)

        info = buf.compute_infostate_tensor()
        # Spy wins vs Marshal, occupies (4,0). Now Blue's turn.
        # Red Spy at (4,0) is revealed (participated in combat).
        # Channel 37 (opp hidden): Red Spy at (4,0) is revealed → 0.0
        assert info[0, 37, 4, 0].item() == pytest.approx(0.0)

        # Channel 12 (opp SPY, ch=12 → pt=1=SPY): revealed → one-hot 1.0 at (4,0)
        assert info[0, 12, 4, 0].item() == pytest.approx(1.0)
        # Other opp piece type channels at (4,0) → 0.0
        for ch in range(13, 24):
            assert info[0, ch, 4, 0].item() == pytest.approx(0.0)

        # Channel 456: +1 at dst (4,0), -1 at src (3,0)
        assert info[0, 456, 4, 0].item() == pytest.approx(1.0)
        assert info[0, 456, 3, 0].item() == pytest.approx(-1.0)

    def test_scalar_channels(self):
        """Channel 41 = move_number/4000, channel 42 = moves_since_attack/100."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [(6, 0, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)

        # Before any move: move_number=0, moves_since_attack=0
        info0 = buf.compute_infostate_tensor()
        assert info0[0, 41].sum().item() / 100 == pytest.approx(0.0)
        assert info0[0, 42].sum().item() / 100 == pytest.approx(0.0)

        # After 1 non-attack move: move_number=1, moves_since_attack=1
        actions = torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64)
        buf.apply_actions(actions)

        info1 = buf.compute_infostate_tensor()
        expected_41 = 1.0 / 4000.0
        expected_42 = 1.0 / 100.0
        # Every square in channel 41 should have the same value
        assert torch.allclose(info1[0, 41], torch.full((10, 10), expected_41, device="cuda"))
        assert torch.allclose(info1[0, 42], torch.full((10, 10), expected_42, device="cuda"))

    def test_move_history_two_moves(self):
        """After 2 moves: channel 456 = most recent, channel 457 = second most recent."""
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SCOUT)],
            [(6, 0, PT_SCOUT)],
            n_games=1,
        )
        buf.reset_all(red, blue)

        # Move 1: Red Scout (3,0) → (2,0)
        buf.apply_actions(torch.tensor([_encode_action(3, 0, 2, 0)], dtype=torch.int64))
        # Move 2: Blue Scout (6,0) → (7,0)
        buf.apply_actions(torch.tensor([_encode_action(6, 0, 7, 0)], dtype=torch.int64))

        info = buf.compute_infostate_tensor()
        # Channel 456 (most recent = move 2): Blue (6,0)→(7,0)
        assert info[0, 456, 7, 0].item() == pytest.approx(1.0)
        assert info[0, 456, 6, 0].item() == pytest.approx(-1.0)

        # Channel 457 (second most recent = move 1): Red (3,0)→(2,0)
        assert info[0, 457, 2, 0].item() == pytest.approx(1.0)
        assert info[0, 457, 3, 0].item() == pytest.approx(-1.0)

        # Channels 458-487: zeros
        assert info[0, 458:488].abs().sum().item() == pytest.approx(0.0)

    def test_move_history_ring_buffer_overflow(self):
        """After 33 moves: only last 32 are in history (1st move overwritten)."""
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SCOUT)],
            [(6, 0, PT_SCOUT)],
            n_games=1,
        )
        buf.reset_all(red, blue)

        # 33 moves: Red Scout oscillates (3,0)↔(2,0), Blue Scout oscillates (6,0)↔(7,0)
        for i in range(33):
            if i % 2 == 0:
                # Red's turn: Scout at (3,0) or (2,0)
                r = 3 if i % 4 == 0 else 2
                dst = 2 if i % 4 == 0 else 3
                buf.apply_actions(torch.tensor([_encode_action(r, 0, dst, 0)], dtype=torch.int64))
            else:
                # Blue's turn: Scout at (6,0) or (7,0)
                r = 6 if i % 4 == 1 else 7
                dst = 7 if i % 4 == 1 else 6
                buf.apply_actions(torch.tensor([_encode_action(r, 0, dst, 0)], dtype=torch.int64))

        info = buf.compute_infostate_tensor()

        # Exactly 32 non-zero channels in 456-487 (33rd move overwrote 1st)
        move_channels = info[0, 456:488]
        nonzero_per_channel = (move_channels != 0).any(dim=(1, 2))
        assert nonzero_per_channel.sum().item() == 32

        # Channel 456 (most recent = 33rd move, index 32): 32%4==0 → Red (3,0)→(2,0)
        assert info[0, 456, 2, 0].item() == pytest.approx(1.0)
        assert info[0, 456, 3, 0].item() == pytest.approx(-1.0)

        # Channel 487 (32nd most recent = 2nd move, index 1): 1%4==1 → Blue (6,0)→(7,0)
        assert info[0, 487, 7, 0].item() == pytest.approx(1.0)
        assert info[0, 487, 6, 0].item() == pytest.approx(-1.0)

    def test_reset_clears_tracking(self):
        """After reset_all, tracking tensors are zeroed (no moved, no history)."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [(6, 0, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)

        # Make a move to populate tracking
        buf.apply_actions(torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64))
        info_before = buf.compute_infostate_tensor()
        assert info_before[0, 456].abs().sum().item() > 0

        # Reset and verify tracking is cleared
        buf.reset_all(red, blue)
        info_after = buf.compute_infostate_tensor()
        assert info_after[0, 456:488].abs().sum().item() == pytest.approx(0.0)
        # Channel 39/40 (moved) should be all zeros
        assert info_after[0, 39].sum().item() == pytest.approx(0.0)
        assert info_after[0, 40].sum().item() == pytest.approx(0.0)
        # Channel 41 (move_number) should be 0
        assert info_after[0, 41].sum().item() == pytest.approx(0.0)
