"""Tests for new getter methods on StrategoRolloutBuffer.

Covers: get_num_moves, get_num_moves_since_last_attack, get_flag_captured,
get_has_legal_movement, get_terminated_since, board_strs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

cuda_skip = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"

PT_SPY = 1
PT_FLAG = 11
PT_MARSHAL = 10

PLAYER_RED = 0
PLAYER_BLUE = 1
PLAYER_EMPTY = -1

OUTCOME_ONGOING = -1
OUTCOME_RED_WIN = 0

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


def _encode_action(src_r, src_c, dst_r, dst_c):
    return (src_r * BOARD_COLS + src_c) * NUM_SQUARES + (dst_r * BOARD_COLS + dst_c)


def _make_setup(pieces_red, pieces_blue, n_games=1):
    setup_red = torch.zeros((n_games, BOARD_ROWS, BOARD_COLS), dtype=torch.int8)
    setup_blue = torch.zeros((n_games, BOARD_ROWS, BOARD_COLS), dtype=torch.int8)
    for r, c, p in pieces_red:
        setup_red[:, r, c] = p
    for r, c, p in pieces_blue:
        setup_blue[:, r, c] = p
    return setup_red, setup_blue


def _buf(n_games=2):
    return _stratego_cuda.StrategoRolloutBuffer(n_games, 0)


@cuda_or_build_skip
class TestNewGetters:
    def test_get_num_moves_initial(self):
        buf = _buf(3)
        nm = buf.get_num_moves()
        assert nm.shape == (3,)
        assert nm.dtype == torch.int32
        assert torch.all(nm == 0).item()

    def test_get_num_moves_since_last_attack_initial(self):
        buf = _buf(3)
        ms = buf.get_num_moves_since_last_attack()
        assert ms.shape == (3,)
        assert ms.dtype == torch.int32
        assert torch.all(ms == 0).item()

    def test_get_flag_captured_initial(self):
        buf = _buf(3)
        fc = buf.get_flag_captured()
        assert fc.shape == (3,)
        assert fc.dtype == torch.bool
        assert not fc.any().item()

    def test_get_flag_captured_after_flag_capture(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 5, PT_SPY)], [(4, 5, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)
        actions = torch.tensor([_encode_action(3, 5, 4, 5)], dtype=torch.int64)
        buf.apply_actions(actions)
        fc = buf.get_flag_captured()
        assert fc[0].item() is True

    def test_get_flag_captured_no_capture(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [], n_games=1)
        buf.reset_all(red, blue)
        actions = torch.tensor([_encode_action(3, 0, 4, 0)], dtype=torch.int64)
        buf.apply_actions(actions)
        fc = buf.get_flag_captured()
        assert not fc[0].item()

    def test_get_has_legal_movement_with_pieces(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [], n_games=1)
        buf.reset_all(red, blue)
        hlm = buf.get_has_legal_movement()
        assert hlm.shape == (1,)
        assert hlm.dtype == torch.bool
        assert hlm[0].item() is True

    def test_get_has_legal_movement_empty(self):
        buf = _buf(1)
        hlm = buf.get_has_legal_movement()
        assert not hlm[0].item()

    def test_get_terminated_since_initial(self):
        buf = _buf(3)
        ts = buf.get_terminated_since()
        assert ts.shape == (3,)
        assert ts.dtype == torch.int32
        assert torch.all(ts == 0).item()

    def test_get_terminated_since_after_termination(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 5, PT_SPY)], [(4, 5, PT_FLAG)], n_games=1)
        buf.reset_all(red, blue)
        actions = torch.tensor([_encode_action(3, 5, 4, 5)], dtype=torch.int64)
        buf.apply_actions(actions)
        ts = buf.get_terminated_since()
        assert ts[0].item() >= 1

    def test_board_strs(self):
        buf = _buf(1)
        red, blue = _make_setup([(3, 0, PT_SPY)], [], n_games=1)
        buf.reset_all(red, blue)
        strs = buf.board_strs()
        assert len(strs) == 1
        assert isinstance(strs[0], str)
        assert "R1" in strs[0]

    def test_board_strs_empty(self):
        buf = _buf(2)
        strs = buf.board_strs()
        assert len(strs) == 2
        assert all(isinstance(s, str) for s in strs)
