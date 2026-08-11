"""Tests for two-square rule and chasing rule GPU tracking.

Tests the simplified GPU-side tracking added to the apply_actions kernel:
  - Two-square rule: per-player consecutive boundary crossing count + violation
  - Chasing rule: Zobrist board hashing + per-player chase position ring buffer

All tests are skipped when CUDA is unavailable or the extension fails to build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

cuda_skip = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"

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

BOARD_ROWS = 10
BOARD_COLS = 10
NUM_SQUARES = BOARD_ROWS * BOARD_COLS

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
    _stratego_cuda = None

cuda_or_build_skip = pytest.mark.skipif(
    _stratego_cuda is None, reason="_stratego_cuda extension not built"
)


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


def _buf(n_games=1):
    return _stratego_cuda.StrategoRolloutBuffer(n_games, 0)


# Blue shuttle 4-cycle that avoids two-square violations: (8,8)->(8,9)->(9,9)->(9,8)->(8,8)
_SHUTTLE_CYCLE = [
    _encode_action(8, 8, 8, 9),
    _encode_action(8, 9, 9, 9),
    _encode_action(9, 9, 9, 8),
    _encode_action(9, 8, 8, 8),
]

# Blue shuttle 2-move oscillation: (8,8)->(8,9)->(8,8).  Count stays ≤ 3 for ≤6 moves.
_SHUTTLE_OSC = [
    _encode_action(8, 8, 8, 9),
    _encode_action(8, 9, 8, 8),
]


def _apply_alternating(buf, red_moves, blue_moves, n_games=1):
    """Apply red_moves and blue_moves in alternating fashion, starting with red."""
    for i in range(max(len(red_moves), len(blue_moves))):
        if i < len(red_moves):
            buf.apply_actions(torch.tensor([red_moves[i]] * n_games, dtype=torch.int64))
        if i < len(blue_moves):
            buf.apply_actions(torch.tensor([blue_moves[i]] * n_games, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Two-square rule tests
# ---------------------------------------------------------------------------


@cuda_or_build_skip
class TestTwoSquareRule:
    def test_initial_no_violation(self):
        buf = _buf(3)
        violation = buf.compute_two_square_rule_applies()
        assert violation.shape == (3,)
        assert violation.dtype == torch.bool
        assert not violation.any().item()

    def test_single_move_no_violation(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        buf.apply_actions(torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64))
        assert not buf.compute_two_square_rule_applies()[0].item()

    def test_three_crossings_no_violation(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        red_moves = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 4, 0),
        ]
        blue_moves = _SHUTTLE_CYCLE[:3]
        _apply_alternating(buf, red_moves, blue_moves)
        assert not buf.compute_two_square_rule_applies()[0].item()

    def test_four_crossings_violation(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        red_moves = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
        ]
        blue_moves = _SHUTTLE_CYCLE[:4]
        _apply_alternating(buf, red_moves, blue_moves)
        assert buf.compute_two_square_rule_applies()[0].item()

    def test_different_pair_no_violation(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        red_moves = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 4, 1),
            _encode_action(4, 1, 3, 1),
        ]
        blue_moves = _SHUTTLE_CYCLE[:3]
        _apply_alternating(buf, red_moves, blue_moves)
        assert not buf.compute_two_square_rule_applies()[0].item()

    def test_attack_resets_count(self):
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_MARSHAL)],
            [(5, 0, PT_SERGEANT), (8, 8, PT_SERGEANT)],
        )
        buf.reset_all(red, blue)
        red_moves = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 5, 0),
        ]
        blue_moves = _SHUTTLE_CYCLE[:4]
        _apply_alternating(buf, red_moves, blue_moves)
        assert not buf.compute_two_square_rule_applies()[0].item()

    def test_violation_is_sticky(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        red_moves = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 3, 1),
        ]
        blue_moves = _SHUTTLE_CYCLE[:5]
        _apply_alternating(buf, red_moves, blue_moves)
        assert buf.compute_two_square_rule_applies()[0].item()

    def test_reset_clears_violation(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        red_moves = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
        ]
        blue_moves = _SHUTTLE_CYCLE[:4]
        _apply_alternating(buf, red_moves, blue_moves)
        assert buf.compute_two_square_rule_applies()[0].item()
        buf.reset_all(red, blue)
        assert not buf.compute_two_square_rule_applies()[0].item()


# ---------------------------------------------------------------------------
# Chasing rule tests
# ---------------------------------------------------------------------------


@cuda_or_build_skip
class TestChasingRule:
    def test_initial_no_violation(self):
        buf = _buf(3)
        violation = buf.is_chasing_violation()
        assert violation.shape == (3,)
        assert violation.dtype == torch.bool
        assert not violation.any().item()

    def test_hash_board_consistency(self):
        buf = _buf(2)
        red, blue = _make_setup(
            [(3, 0, PT_SERGEANT)],
            [(6, 0, PT_SERGEANT)],
            n_games=2,
        )
        buf.reset_all(red, blue)
        h1 = buf.hash_board()
        h2 = buf.hash_board()
        assert h1.shape == (2,)
        assert h1.dtype == torch.int64
        assert torch.equal(h1, h2)

    def test_hash_board_changes_after_move(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(6, 0, PT_SERGEANT)])
        buf.reset_all(red, blue)
        h_before = buf.hash_board()
        buf.apply_actions(torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64))
        h_after = buf.hash_board()
        assert h_before[0].item() != h_after[0].item()

    def test_non_threatening_move_no_violation(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SERGEANT)], [(8, 8, PT_SERGEANT)])
        buf.reset_all(red, blue)
        buf.apply_actions(torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64))
        assert not buf.is_chasing_violation()[0].item()

    def test_single_threatening_move_no_violation(self):
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SERGEANT)],
            [(5, 0, PT_SERGEANT), (8, 8, PT_SERGEANT)],
        )
        buf.reset_all(red, blue)
        buf.apply_actions(torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64))
        assert not buf.is_chasing_violation()[0].item()

    def test_chasing_violation_4cycle(self):
        """Red completes a 4-cycle of threatening moves, returning to a
        previously-seen board position. The 5th threatening move triggers
        a chasing violation because the board hash matches and the move
        is not back to the preceding square (exception does not apply).

        Board setup:
          Red Sergeant at (3,0) — cycles (3,0)->(4,0)->(4,1)->(3,1)->(3,0)->(4,0)
          Blue Sergeants at (5,0),(5,1),(2,1),(2,0) — stationary, each adjacent
            to one square of Red's cycle so every Red move is threatening
          Blue Sergeant at (8,8) — shuttle, 4-cycles in sync so the full board
            hash recurs on Red's 5th threatening move
        """
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SERGEANT)],
            [
                (5, 0, PT_SERGEANT),
                (5, 1, PT_SERGEANT),
                (2, 1, PT_SERGEANT),
                (2, 0, PT_SERGEANT),
                (8, 8, PT_SERGEANT),
            ],
        )
        buf.reset_all(red, blue)

        red_cycle = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 4, 1),
            _encode_action(4, 1, 3, 1),
            _encode_action(3, 1, 3, 0),
            _encode_action(3, 0, 4, 0),
        ]
        blue_shuttle = _SHUTTLE_CYCLE[:4]

        _apply_alternating(buf, red_cycle, blue_shuttle)
        assert buf.is_chasing_violation()[0].item()

    def test_chasing_exception_back_to_preceding(self):
        """Red oscillates between two squares (2-cycle). The 3rd threatening
        move returns to a previously-seen board position, but the exception
        (dst == last_chaser_src) prevents a violation.

        Board setup:
          Red Sergeant at (3,0) — oscillates (3,0)->(4,0)->(3,0)->(4,0)
          Blue Sergeants at (5,0) and (2,0) — stationary, each adjacent to
            one of Red's positions so both directions are threatening
          Blue Sergeant at (8,8) — shuttle oscillates (8,8)->(8,9)->(8,8)
            so the full board hash recurs on Red's 3rd threatening move
        """
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SERGEANT)],
            [
                (5, 0, PT_SERGEANT),
                (2, 0, PT_SERGEANT),
                (8, 8, PT_SERGEANT),
            ],
        )
        buf.reset_all(red, blue)

        red_oscillation = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 3, 0),
            _encode_action(3, 0, 4, 0),
        ]
        blue_shuttle = _SHUTTLE_OSC[:2]

        _apply_alternating(buf, red_oscillation, blue_shuttle)
        assert not buf.is_chasing_violation()[0].item()

    def test_reset_clears_chasing_violation(self):
        buf = _buf(1)
        red, blue = _make_setup(
            [(3, 0, PT_SERGEANT)],
            [
                (5, 0, PT_SERGEANT),
                (5, 1, PT_SERGEANT),
                (2, 1, PT_SERGEANT),
                (2, 0, PT_SERGEANT),
                (8, 8, PT_SERGEANT),
            ],
        )
        buf.reset_all(red, blue)

        red_cycle = [
            _encode_action(3, 0, 4, 0),
            _encode_action(4, 0, 4, 1),
            _encode_action(4, 1, 3, 1),
            _encode_action(3, 1, 3, 0),
            _encode_action(3, 0, 4, 0),
        ]
        blue_shuttle = _SHUTTLE_CYCLE[:4]
        _apply_alternating(buf, red_cycle, blue_shuttle)
        assert buf.is_chasing_violation()[0].item()
        buf.reset_all(red, blue)
        assert not buf.is_chasing_violation()[0].item()


# ---------------------------------------------------------------------------
# Multi-game parallelism test
# ---------------------------------------------------------------------------


@cuda_or_build_skip
class TestMultiGameRules:
    def test_shapes_for_n_games(self):
        buf = _buf(4)
        ts = buf.compute_two_square_rule_applies()
        ch = buf.is_chasing_violation()
        hb = buf.hash_board()
        assert ts.shape == (4,)
        assert ch.shape == (4,)
        assert hb.shape == (4,)
        assert ts.dtype == torch.bool
        assert ch.dtype == torch.bool
        assert hb.dtype == torch.int64
        assert not ts.any().item()
        assert not ch.any().item()

    def test_two_square_violation_in_one_game(self):
        """Game 0 oscillates 4 times (violation); game 1 moves to a new
        square each time (no violation).  Both games receive the same
        action tensor, so game 1's piece must end up on a different
        square each Red turn to avoid repeating the same pair."""
        n = 2
        buf = _buf(n)
        red, blue = _make_setup(
            [(3, 0, PT_SERGEANT)],
            [(8, 8, PT_SERGEANT)],
            n_games=n,
        )
        buf.reset_all(red, blue)

        # Game 0: oscillate (3,0)<->(4,0) — pair (30,40) repeats 4 times.
        # Game 1: walk (3,0)->(4,0)->(4,1)->(3,1) — each pair is unique.
        red_actions = [
            torch.tensor([_encode_action(3, 0, 4, 0), _encode_action(3, 0, 4, 0)], dtype=torch.int64),
            torch.tensor([_encode_action(4, 0, 3, 0), _encode_action(4, 0, 4, 1)], dtype=torch.int64),
            torch.tensor([_encode_action(3, 0, 4, 0), _encode_action(4, 1, 3, 1)], dtype=torch.int64),
            torch.tensor([_encode_action(4, 0, 3, 0), _encode_action(3, 1, 3, 0)], dtype=torch.int64),
        ]
        blue_actions = [
            torch.tensor([_SHUTTLE_CYCLE[i]] * n, dtype=torch.int64)
            for i in range(4)
        ]

        for i in range(4):
            buf.apply_actions(red_actions[i])
            buf.apply_actions(blue_actions[i])

        violation = buf.compute_two_square_rule_applies()
        assert violation[0].item()
        assert not violation[1].item()
