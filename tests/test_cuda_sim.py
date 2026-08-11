"""Tests for the StrategoRolloutBuffer CUDA simulator (csrc/stratego_buffer.*).

These tests JIT-compile the _stratego_cuda extension and exercise the full
GPU simulator: construction, apply_actions, legal-action mask, combat rules
(Spy>Marshal, Miner>Bomb, equal-rank mutual death), flag-capture win,
outcomes, terminated flags, and current_step.

All tests are skipped when CUDA is unavailable or the extension fails to build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

cuda_skip = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"

# ---------------------------------------------------------------------------
# Constants — mirror stratego/types.py and csrc/stratego_buffer.cuh
# ---------------------------------------------------------------------------
PT_NONE = 0
PT_SPY = 1
PT_SCOUT = 2
PT_MINER = 3
PT_SERGEANT = 4
PT_MARSHAL = 10
PT_FLAG = 11
PT_BOMB = 12

PLAYER_RED = 0
PLAYER_BLUE = 1
PLAYER_EMPTY = -1

OUTCOME_ONGOING = -1
OUTCOME_RED_WIN = 0
OUTCOME_BLUE_WIN = 1
OUTCOME_DRAW = 2

BOARD_ROWS = 10
BOARD_COLS = 10
NUM_SQUARES = BOARD_ROWS * BOARD_COLS  # 100


def _encode_action(src_r: int, src_c: int, dst_r: int, dst_c: int) -> int:
    """Encode (src_row, src_col, dst_row, dst_col) as src_idx*100 + dst_idx."""
    return (src_r * BOARD_COLS + src_c) * NUM_SQUARES + (dst_r * BOARD_COLS + dst_c)


def _make_setup(pieces_red, pieces_blue, n_games=1):
    """Build (N, 10, 10) int8 setup tensors from lists of (row, col, piece)."""
    setup_red = torch.zeros((n_games, BOARD_ROWS, BOARD_COLS), dtype=torch.int8)
    setup_blue = torch.zeros((n_games, BOARD_ROWS, BOARD_COLS), dtype=torch.int8)
    for r, c, p in pieces_red:
        setup_red[:, r, c] = p
    for r, c, p in pieces_blue:
        setup_blue[:, r, c] = p
    return setup_red, setup_blue


# ---------------------------------------------------------------------------
# JIT-compile the extension once at module import time.
# ---------------------------------------------------------------------------
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

cuda_or_build_skip = pytest.mark.skipif(
    _stratego_cuda is None, reason="_stratego_cuda extension not built (no CUDA or build error)"
)


def _buf(n_games=4):
    """Create a fresh StrategoRolloutBuffer."""
    return _stratego_cuda.StrategoRolloutBuffer(n_games, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@cuda_or_build_skip
class TestStrategoRolloutBuffer:
    """Tests for the StrategoRolloutBuffer CUDA simulator."""

    def test_buffer_construction(self):
        """Buffer with N=4 must expose n_games and device_id properties."""
        buf = _buf(4)
        assert buf.n_games == 4
        assert buf.device_id == 0

    def test_apply_actions_simple_move(self):
        """A Red Spy moving to an empty square must relocate the piece."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [], n_games=1)
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64)
        buf.apply_actions(actions)

        owner = buf.get_board_owner()
        piece = buf.get_board_piece()
        # Spy moved to (4, 0)
        assert owner[0, 4, 0].item() == PLAYER_RED
        assert piece[0, 4, 0].item() == PT_SPY
        # Source square now empty
        assert owner[0, 3, 0].item() == PLAYER_EMPTY
        assert piece[0, 3, 0].item() == PT_NONE

    def test_legal_action_mask_shape(self):
        """compute_legal_action_mask must return (N, 100, 100) bool tensor."""
        buf = _buf(4)
        mask = buf.compute_legal_action_mask()
        assert mask.shape == (4, NUM_SQUARES, NUM_SQUARES)
        assert mask.dtype == torch.bool
        # Empty board → no legal moves
        assert not mask.any().item()

    def test_legal_action_mask_with_pieces(self):
        """A Red Spy on the board must produce legal-move entries."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [], n_games=1)
        buf.reset_all(red, blue)

        mask = buf.compute_legal_action_mask()
        # Spy at (3,0) can move up (2,0), down (4,0), right (3,1)
        assert mask[0, 30, 20].item() is True  # up
        assert mask[0, 30, 40].item() is True  # down
        assert mask[0, 30, 31].item() is True  # right
        # Cannot move left (off-board)
        assert mask[0, 30, 39].item() is False

    def test_combat_spy_defeats_marshal(self):
        """Spy attacking Marshal must win (special rule)."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [(4, 0, PT_MARSHAL)], n_games=1)
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64)
        buf.apply_actions(actions)

        owner = buf.get_board_owner()
        piece = buf.get_board_piece()
        # Spy wins, occupies Marshal's square
        assert owner[0, 4, 0].item() == PLAYER_RED
        assert piece[0, 4, 0].item() == PT_SPY
        assert owner[0, 3, 0].item() == PLAYER_EMPTY

    def test_combat_miner_defeats_bomb(self):
        """Miner attacking Bomb must win (special rule)."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 1, PT_MINER)], [(4, 1, PT_BOMB)], n_games=1)
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 1, 4, 1)], dtype=torch.int64)
        buf.apply_actions(actions)

        owner = buf.get_board_owner()
        piece = buf.get_board_piece()
        # Miner defuses bomb, occupies square
        assert owner[0, 4, 1].item() == PLAYER_RED
        assert piece[0, 4, 1].item() == PT_MINER
        assert owner[0, 3, 1].item() == PLAYER_EMPTY

    def test_combat_equal_rank_both_die(self):
        """Equal-rank combat must destroy both pieces."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 4, PT_SERGEANT)], [(4, 4, PT_SERGEANT)], n_games=1)
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 4, 4, 4)], dtype=torch.int64)
        buf.apply_actions(actions)

        owner = buf.get_board_owner()
        piece = buf.get_board_piece()
        # Both squares empty
        assert owner[0, 3, 4].item() == PLAYER_EMPTY
        assert piece[0, 3, 4].item() == PT_NONE
        assert owner[0, 4, 4].item() == PLAYER_EMPTY
        assert piece[0, 4, 4].item() == PT_NONE

    def test_flag_capture_wins(self):
        """Capturing the Flag must end the game with a Red win."""
        buf = _buf(1)
        red, blue = _make_setup([(3, 5, PT_SPY)], [(4, 5, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)

        actions = torch.tensor([_encode_action(3, 5, 4, 5)], dtype=torch.int64)
        buf.apply_actions(actions)

        outcome = buf.get_outcomes()
        terminated = buf.get_terminated()
        assert outcome[0].item() == OUTCOME_RED_WIN
        assert terminated[0].item() is True

    def test_get_outcomes_initial(self):
        """Fresh buffer must report ONGOING for all games."""
        buf = _buf(4)
        outcome = buf.get_outcomes()
        assert outcome.shape == (4,)
        assert outcome.dtype == torch.int8
        assert torch.all(outcome == OUTCOME_ONGOING).item()

    def test_get_terminated_initial(self):
        """Fresh buffer must report not-terminated for all games."""
        buf = _buf(4)
        terminated = buf.get_terminated()
        assert terminated.shape == (4,)
        assert terminated.dtype == torch.bool
        assert not terminated.any().item()

    def test_current_step(self):
        """current_step must return the sum of move_numbers across games."""
        buf = _buf(2)
        red, blue = _make_setup(
            [(3, 0, PT_SPY), (3, 1, PT_SPY)],
            [(6, 0, PT_FLAG), (6, 1, PT_FLAG)],
            n_games=2,
        )
        buf.reset_all(red, blue)
        assert buf.current_step() == 0

        # Apply one move per game (Spy moves down, no combat)
        actions = torch.tensor(
            [_encode_action(3, 0, 4, 0), _encode_action(3, 1, 4, 1)],
            dtype=torch.int64,
        )
        buf.apply_actions(actions)
        assert buf.current_step() == 2

    def test_hello_world(self):
        """hello_world kernel must return a CUDA tensor with value 42."""
        buf = _buf(1)
        result = buf.hello_world()
        assert result.is_cuda
        assert result.shape == (1,)
        assert result.dtype == torch.int32
        assert result.item() == 42
