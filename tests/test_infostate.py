"""Tests for stratego/env/infostate.py — 488-channel infostate computation."""

from __future__ import annotations

import numpy as np

from stratego.constants import MAX_GAME_LENGTH, TRAINING_NO_ATTACK_LIMIT
from stratego.env.infostate import compute_infostate
from stratego.env.rules import StrategoState
from stratego.types import Action, PieceType, Player, Square

_UNIF = 1.0 / 12


def make_state_with_pieces(pieces: list[tuple[Square, int, PieceType]]) -> StrategoState:
    state = StrategoState()
    state.board_owner[:] = -1
    state.board_piece[:] = 0
    for sq, owner, pt in pieces:
        state.board_owner[sq.row, sq.col] = owner
        state.board_piece[sq.row, sq.col] = int(pt)
    return state


class TestShape:
    def test_shape(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        assert info.shape == (488, 10, 10)

    def test_dtype(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        assert info.dtype == np.float32


class TestEmptyBoard:
    def test_piece_channels_zero(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        for i in range(36):
            assert info[i].sum() == 0.0

    def test_empty_channel_all_ones(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        assert (info[38] == 1.0).all()


class TestOwnPieces:
    def test_own_piece_one_hot(self) -> None:
        state = make_state_with_pieces([(Square(0, 0), 0, PieceType.MARSHAL)])
        info = compute_infostate(state, Player.RED)
        assert info[9, 0, 0] == 1.0
        assert info[0, 0, 0] == 0.0

    def test_own_flag_channel(self) -> None:
        state = make_state_with_pieces([(Square(0, 1), 0, PieceType.FLAG)])
        info = compute_infostate(state, Player.RED)
        assert info[10, 0, 1] == 1.0

    def test_own_bomb_channel(self) -> None:
        state = make_state_with_pieces([(Square(0, 2), 0, PieceType.BOMB)])
        info = compute_infostate(state, Player.RED)
        assert info[11, 0, 2] == 1.0


class TestOpponentPieces:
    def test_hidden_opponent_uniform(self) -> None:
        state = make_state_with_pieces([(Square(9, 9), 1, PieceType.SPY)])
        info = compute_infostate(state, Player.RED)
        for i in range(12):
            assert info[12 + i, 9, 9] == _UNIF

    def test_revealed_opponent_one_hot(self) -> None:
        sq = Square(9, 9)
        state = make_state_with_pieces([(sq, 1, PieceType.MARSHAL)])
        info = compute_infostate(state, Player.RED, revealed_squares={sq})
        assert info[21, 9, 9] == 1.0
        assert info[12, 9, 9] == 0.0


class TestMirror:
    def test_mirror_own_revealed(self) -> None:
        sq = Square(0, 0)
        state = make_state_with_pieces([(sq, 0, PieceType.MARSHAL)])
        info = compute_infostate(state, Player.RED, revealed_squares={sq})
        assert info[33, 0, 0] == 1.0

    def test_mirror_own_hidden_uniform(self) -> None:
        state = make_state_with_pieces([(Square(0, 0), 0, PieceType.MARSHAL)])
        info = compute_infostate(state, Player.RED)
        for i in range(12):
            assert info[24 + i, 0, 0] == _UNIF


class TestHiddenAndMoved:
    def test_own_hidden(self) -> None:
        state = make_state_with_pieces([(Square(0, 0), 0, PieceType.SPY)])
        info = compute_infostate(state, Player.RED)
        assert info[36, 0, 0] == 1.0

    def test_opponent_hidden(self) -> None:
        state = make_state_with_pieces([(Square(9, 9), 1, PieceType.SPY)])
        info = compute_infostate(state, Player.RED)
        assert info[37, 9, 9] == 1.0

    def test_revealed_not_hidden(self) -> None:
        sq = Square(0, 0)
        state = make_state_with_pieces([(sq, 0, PieceType.SPY)])
        info = compute_infostate(state, Player.RED, revealed_squares={sq})
        assert info[36, 0, 0] == 0.0

    def test_own_moved(self) -> None:
        sq = Square(0, 0)
        state = make_state_with_pieces([(sq, 0, PieceType.SPY)])
        info = compute_infostate(state, Player.RED, moved_squares={sq})
        assert info[39, 0, 0] == 1.0

    def test_opponent_moved(self) -> None:
        sq = Square(9, 9)
        state = make_state_with_pieces([(sq, 1, PieceType.SPY)])
        info = compute_infostate(state, Player.RED, moved_squares={sq})
        assert info[40, 9, 9] == 1.0


class TestFractions:
    def test_move_fraction_zero_at_start(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        assert (info[41] == 0.0).all()

    def test_move_fraction_after_moves(self) -> None:
        state = StrategoState()
        state._move_number = 100
        info = compute_infostate(state, Player.RED)
        assert (info[41] == 100 / MAX_GAME_LENGTH).all()

    def test_no_attack_fraction_zero_at_start(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        assert (info[42] == 0.0).all()

    def test_no_attack_fraction_after_moves(self) -> None:
        state = StrategoState()
        state._moves_since_last_attack = 50
        info = compute_infostate(state, Player.RED)
        assert (info[42] == 50 / TRAINING_NO_ATTACK_LIMIT).all()


class TestLastMoves:
    def test_single_move(self) -> None:
        state = StrategoState()
        mv = Action(Square(3, 0), Square(3, 1))
        info = compute_infostate(state, Player.RED, move_history=[mv])
        assert info[456, 3, 1] == 1.0
        assert info[456, 3, 0] == -1.0

    def test_two_moves_ordering(self) -> None:
        state = StrategoState()
        mv1 = Action(Square(3, 0), Square(3, 1))
        mv2 = Action(Square(3, 1), Square(3, 2))
        info = compute_infostate(state, Player.RED, move_history=[mv1, mv2])
        assert info[456, 3, 2] == 1.0
        assert info[456, 3, 1] == -1.0
        assert info[457, 3, 1] == 1.0
        assert info[457, 3, 0] == -1.0

    def test_empty_history_zeros(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        for i in range(32):
            assert info[456 + i].sum() == 0.0

    def test_more_than_32_moves(self) -> None:
        state = StrategoState()
        moves = [Action(Square(i % 10, 0), Square(i % 10, 1)) for i in range(40)]
        info = compute_infostate(state, Player.RED, move_history=moves)
        last_move = moves[-1]
        assert info[456, last_move.dst.row, last_move.dst.col] == 1.0
        assert info[456, last_move.src.row, last_move.src.col] == -1.0
        oldest_kept = moves[-32]
        assert info[487, oldest_kept.dst.row, oldest_kept.dst.col] == 1.0


class TestPerspectives:
    def test_red_vs_blue(self) -> None:
        state = make_state_with_pieces([
            (Square(0, 0), 0, PieceType.MARSHAL),
            (Square(9, 9), 1, PieceType.SPY),
        ])
        red_info = compute_infostate(state, Player.RED)
        blue_info = compute_infostate(state, Player.BLUE)
        assert red_info[9, 0, 0] == 1.0
        assert blue_info[0, 9, 9] == 1.0  # Spy is channel 0 (PIECE_TYPES[0]=SPY)
        assert red_info[12, 9, 9] == _UNIF
        assert blue_info[12, 0, 0] == _UNIF


class TestLakes:
    def test_lake_zero_in_piece_channels(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        from stratego.constants import LAKES
        for sq in LAKES:
            for i in range(36):
                assert info[i, sq.row, sq.col] == 0.0


class TestComplexChannels:
    def test_complex_channels_zero(self) -> None:
        state = StrategoState()
        info = compute_infostate(state, Player.RED)
        for i in range(43, 456):
            assert info[i].sum() == 0.0
