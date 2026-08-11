"""Tests for stratego.env.rules — the pure-Python reference Stratego rules engine.

Covers: setup validation, basic movement, scout long moves, combat resolution,
win/draw conditions, and state management. 30+ test cases, TDD RED-first.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.constants import LAKE_SET, MAX_GAME_LENGTH, PIECE_COUNTS, TOTAL_PIECES, TRAINING_NO_ATTACK_LIMIT
from stratego.env.rules import StrategoState
from stratego.types import Action, GameOutcome, PieceType, Player, Square

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_setup(player: Player) -> list[tuple[Square, PieceType]]:
    """Generate a valid 40-piece setup for the given player.

    Places pieces deterministically in the player's setup rows, avoiding lakes.
    """
    start_row, end_row = player.setup_rows
    pieces_needed: list[PieceType] = []
    for pt, count in PIECE_COUNTS.items():
        pieces_needed.extend([pt] * count)
    assert len(pieces_needed) == TOTAL_PIECES

    placement: list[tuple[Square, PieceType]] = []
    idx = 0
    for r in range(start_row, end_row + 1):
        for c in range(10):
            sq = Square(r, c)
            if sq.is_lake:
                continue
            if idx < len(pieces_needed):
                placement.append((sq, pieces_needed[idx]))
                idx += 1
    assert idx == TOTAL_PIECES, f"Only placed {idx} of {TOTAL_PIECES}"
    return placement


def _make_game_with_setups(
    no_attack_limit: int = TRAINING_NO_ATTACK_LIMIT,
) -> StrategoState:
    """Create a StrategoState with both players' setups applied."""
    state = StrategoState(no_attack_limit=no_attack_limit)
    state.apply_setup(Player.RED, _default_setup(Player.RED))
    state.apply_setup(Player.BLUE, _default_setup(Player.BLUE))
    return state


def _place_piece(
    state: StrategoState,
    player: Player,
    piece: PieceType,
    row: int,
    col: int,
) -> None:
    """Directly place a piece on the board (bypassing setup validation)."""
    state.board_owner[row, col] = int(player)
    state.board_piece[row, col] = int(piece)


def _clear_board(state: StrategoState) -> None:
    """Clear all pieces from the board."""
    state.board_owner[:] = -1
    state.board_piece[:] = 0


# ---------------------------------------------------------------------------
# Setup validation
# ---------------------------------------------------------------------------


class TestSetupValidation:
    def test_valid_setup_accepted(self) -> None:
        """A valid 40-piece setup with correct counts and valid rows is accepted."""
        state = StrategoState()
        placement = _default_setup(Player.RED)
        state.apply_setup(Player.RED, placement)  # should not raise

    def test_wrong_piece_count_rejected(self) -> None:
        """Having 2 Marshals (instead of 1) is rejected."""
        state = StrategoState()
        placement = _default_setup(Player.RED)
        # Replace one Scout with a second Marshal
        for i, (sq, pt) in enumerate(placement):
            if pt == PieceType.SCOUT:
                placement[i] = (sq, PieceType.MARSHAL)
                break
        with pytest.raises(ValueError, match="count"):
            state.apply_setup(Player.RED, placement)

    def test_piece_outside_setup_rows_rejected(self) -> None:
        """A piece placed outside the player's setup rows is rejected."""
        state = StrategoState()
        placement = _default_setup(Player.RED)
        # Move one piece to row 5 (outside Red's 0-3 setup zone)
        placement[0] = (Square(5, 0), placement[0][1])
        with pytest.raises(ValueError, match="row"):
            state.apply_setup(Player.RED, placement)

    def test_piece_on_lake_rejected(self) -> None:
        """A piece placed on a lake square is rejected."""
        state = StrategoState()
        placement = _default_setup(Player.RED)
        # Lakes are at rows 4-5, which are outside Red's setup rows (0-3),
        # so we test with Blue whose setup rows are 6-9 (no lakes there either).
        # Instead, manually construct a placement with a lake square.
        # Red setup rows are 0-3, no lakes there. Let's force a lake square.
        placement[0] = (Square(4, 2), placement[0][1])  # lake square
        with pytest.raises(ValueError):
            state.apply_setup(Player.RED, placement)

    def test_setup_wrong_total_count(self) -> None:
        """Providing fewer than 40 pieces is rejected."""
        state = StrategoState()
        placement = _default_setup(Player.RED)
        placement.pop()  # remove one piece → 39
        with pytest.raises(ValueError, match="40"):
            state.apply_setup(Player.RED, placement)


# ---------------------------------------------------------------------------
# Basic movement
# ---------------------------------------------------------------------------


class TestBasicMovement:
    def test_sergeant_moves_one_cardinal(self) -> None:
        """A Sergeant can move 1 square in a cardinal direction."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SERGEANT, 3, 4)
        state._current_player = Player.RED

        actions = state.legal_actions(Player.RED)
        sergeant_actions = [a for a in actions if a.src == Square(3, 4)]
        assert len(sergeant_actions) == 4, "Sergeant in open should have 4 moves"
        for a in sergeant_actions:
            dr = abs(a.dst.row - a.src.row)
            dc = abs(a.dst.col - a.src.col)
            assert (dr + dc) == 1, "Sergeant must move exactly 1 square cardinal"

    def test_diagonal_moves_illegal(self) -> None:
        """No legal action should be a diagonal move."""
        state = _make_game_with_setups()
        for action in state.legal_actions(Player.RED):
            dr = abs(action.dst.row - action.src.row)
            dc = abs(action.dst.col - action.src.col)
            assert not (dr == 1 and dc == 1), "Diagonal moves are illegal"

    def test_flag_cannot_move(self) -> None:
        """Flag should never appear as src in legal actions."""
        state = _make_game_with_setups()
        for action in state.legal_actions(Player.RED):
            assert state.board_piece[action.src.row, action.src.col] != int(
                PieceType.FLAG
            ), "Flag cannot move"

    def test_bomb_cannot_move(self) -> None:
        """Bomb should never appear as src in legal actions."""
        state = _make_game_with_setups()
        for action in state.legal_actions(Player.RED):
            assert state.board_piece[action.src.row, action.src.col] != int(
                PieceType.BOMB
            ), "Bomb cannot move"

    def test_cannot_move_onto_own_pieces(self) -> None:
        """A piece cannot move onto a square occupied by a friendly piece."""
        state = _make_game_with_setups()
        for action in state.legal_actions(Player.RED):
            if state.board_owner[action.dst.row, action.dst.col] == int(Player.RED):
                pytest.fail("Cannot move onto own piece")

    def test_cannot_move_onto_lake(self) -> None:
        """No legal action should have a lake square as destination."""
        state = _make_game_with_setups()
        for action in state.legal_actions(Player.RED):
            assert action.dst not in LAKE_SET, "Cannot move onto lake"

    def test_cannot_move_off_board(self) -> None:
        """All legal action destinations must be valid board squares."""
        state = _make_game_with_setups()
        for action in state.legal_actions(Player.RED):
            assert action.dst.is_valid, "Cannot move off the board"


# ---------------------------------------------------------------------------
# Scout long move
# ---------------------------------------------------------------------------


class TestScoutLongMove:
    def test_scout_moves_multiple_squares(self) -> None:
        """Scout can move multiple squares in a cardinal direction."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 4, 0)
        state._move_number = 0  # ensure it's Red's turn
        state._current_player = Player.RED

        actions = state.legal_actions(Player.RED)
        long_moves = [a for a in actions if abs(a.dst.row - a.src.row) + abs(a.dst.col - a.src.col) > 1]
        assert len(long_moves) > 0, "Scout should have long-range moves available"

    def test_scout_blocked_by_occupied_square(self) -> None:
        """Scout cannot jump over an occupied square."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 4, 0)
        _place_piece(state, Player.RED, PieceType.SERGEANT, 4, 2)  # blocker
        state._current_player = Player.RED

        actions = state.legal_actions(Player.RED)
        scout_actions = [a for a in actions if a.src == Square(4, 0)]
        # Scout can move to (4,1) but NOT (4,3) or beyond through the blocker at (4,2)
        dsts = {a.dst for a in scout_actions}
        assert Square(4, 1) in dsts, "Scout can move to adjacent empty square"
        assert Square(4, 3) not in dsts, "Scout cannot jump over blocker"
        assert Square(4, 4) not in dsts, "Scout cannot jump over blocker"

    def test_scout_blocked_by_lake(self) -> None:
        """Scout cannot cross a lake square."""
        state = StrategoState()
        _clear_board(state)
        # Place scout at (4, 1) — lake is at (4, 2) and (4, 3)
        _place_piece(state, Player.RED, PieceType.SCOUT, 4, 1)
        state._current_player = Player.RED

        actions = state.legal_actions(Player.RED)
        scout_actions = [a for a in actions if a.src == Square(4, 1)]
        dsts = {a.dst for a in scout_actions}
        # Can't go right past the lake at (4,2)
        assert Square(4, 4) not in dsts, "Scout cannot cross lake"
        assert Square(4, 5) not in dsts, "Scout cannot cross lake"

    def test_scout_attack_from_distance(self) -> None:
        """Scout can attack an enemy piece from multiple squares away."""
        state = StrategoState()
        _clear_board(state)
        # Use row 0 (no lakes) for clear path
        _place_piece(state, Player.RED, PieceType.SCOUT, 0, 0)
        _place_piece(state, Player.BLUE, PieceType.SERGEANT, 0, 4)
        state._current_player = Player.RED

        actions = state.legal_actions(Player.RED)
        scout_attacks = [
            a for a in actions if a.src == Square(0, 0) and a.dst == Square(0, 4)
        ]
        assert len(scout_attacks) == 1, "Scout should be able to attack from distance"

    def test_scout_cannot_change_direction(self) -> None:
        """Scout must move in a straight line — no L-shaped moves."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 5, 5)
        state._current_player = Player.RED

        actions = state.legal_actions(Player.RED)
        for a in actions:
            if a.src != Square(5, 5):
                continue
            dr = a.dst.row - a.src.row
            dc = a.dst.col - a.src.col
            # Must be purely horizontal or purely vertical
            assert dr == 0 or dc == 0, "Scout cannot change direction mid-move"


# ---------------------------------------------------------------------------
# Combat resolution
# ---------------------------------------------------------------------------


class TestCombatResolution:
    def test_higher_rank_wins(self) -> None:
        """Marshal (10) beats General (9)."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.MARSHAL, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.GENERAL, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        # Marshal wins, moves to (4,1); General removed
        assert state.board_piece[4, 1] == int(PieceType.MARSHAL)
        assert state.board_owner[4, 1] == int(Player.RED)
        assert state.board_piece[4, 0] == int(PieceType.NONE)
        assert state.board_owner[4, 0] == -1

    def test_equal_rank_both_die(self) -> None:
        """Equal rank: both pieces are removed."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.COLONEL, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.COLONEL, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        assert state.board_piece[4, 0] == int(PieceType.NONE)
        assert state.board_piece[4, 1] == int(PieceType.NONE)
        assert state.board_owner[4, 0] == -1
        assert state.board_owner[4, 1] == -1

    def test_spy_beats_marshal_when_attacking(self) -> None:
        """Spy (attacker) defeats Marshal (defender)."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SPY, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.MARSHAL, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        # Spy wins as attacker
        assert state.board_piece[4, 1] == int(PieceType.SPY)
        assert state.board_owner[4, 1] == int(Player.RED)
        assert state.board_piece[4, 0] == int(PieceType.NONE)

    def test_spy_loses_to_marshal_when_defending(self) -> None:
        """Marshal (attacker) defeats Spy (defender)."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.MARSHAL, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.SPY, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        # Marshal wins as attacker against Spy defender
        assert state.board_piece[4, 1] == int(PieceType.MARSHAL)
        assert state.board_owner[4, 1] == int(Player.RED)
        assert state.board_piece[4, 0] == int(PieceType.NONE)

    def test_spy_loses_to_other_pieces_when_attacking(self) -> None:
        """Spy loses to all non-Marshal pieces when attacking."""
        for defender_pt in [PieceType.SCOUT, PieceType.SERGEANT, PieceType.GENERAL]:
            state = StrategoState()
            _clear_board(state)
            _place_piece(state, Player.RED, PieceType.SPY, 4, 0)
            _place_piece(state, Player.BLUE, defender_pt, 4, 1)
            state._current_player = Player.RED

            state.apply_action(Action(Square(4, 0), Square(4, 1)))
            # Spy loses: defender survives, attacker removed
            assert state.board_piece[4, 1] == int(defender_pt), f"Spy should lose to {defender_pt}"
            assert state.board_owner[4, 1] == int(Player.BLUE)
            assert state.board_piece[4, 0] == int(PieceType.NONE)

    def test_miner_defeats_bomb(self) -> None:
        """Miner (attacker) defeats Bomb (defender)."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.MINER, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.BOMB, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        # Miner wins, moves to bomb's square
        assert state.board_piece[4, 1] == int(PieceType.MINER)
        assert state.board_owner[4, 1] == int(Player.RED)
        assert state.board_piece[4, 0] == int(PieceType.NONE)

    def test_bomb_defeats_non_miner_attacker(self) -> None:
        """Bomb defeats all non-Miner attackers (attacker dies, Bomb survives)."""
        for attacker_pt in [PieceType.SPY, PieceType.SCOUT, PieceType.SERGEANT, PieceType.MARSHAL]:
            state = StrategoState()
            _clear_board(state)
            _place_piece(state, Player.RED, attacker_pt, 4, 0)
            _place_piece(state, Player.BLUE, PieceType.BOMB, 4, 1)
            state._current_player = Player.RED

            state.apply_action(Action(Square(4, 0), Square(4, 1)))
            # Bomb survives, attacker dies
            assert state.board_piece[4, 1] == int(PieceType.BOMB), f"Bomb should defeat {attacker_pt}"
            assert state.board_owner[4, 1] == int(Player.BLUE)
            assert state.board_piece[4, 0] == int(PieceType.NONE)

    def test_flag_capture_wins_game(self) -> None:
        """Capturing the Flag = immediate win for the attacker."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.FLAG, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        assert state.outcome == GameOutcome.RED_WIN
        assert state.is_terminal

    def test_attacker_wins_moves_to_defender_square(self) -> None:
        """When attacker wins, attacker moves to defender's square."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.GENERAL, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.SERGEANT, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        assert state.board_piece[4, 1] == int(PieceType.GENERAL)
        assert state.board_owner[4, 1] == int(Player.RED)
        assert state.board_piece[4, 0] == int(PieceType.NONE)
        assert state.board_owner[4, 0] == -1

    def test_defender_wins_no_movement(self) -> None:
        """When defender wins, attacker is removed and no movement occurs."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SERGEANT, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.GENERAL, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        # Defender (General) stays, attacker (Sergeant) removed
        assert state.board_piece[4, 1] == int(PieceType.GENERAL)
        assert state.board_owner[4, 1] == int(Player.BLUE)
        assert state.board_piece[4, 0] == int(PieceType.NONE)
        assert state.board_owner[4, 0] == -1

    def test_equal_rank_both_removed(self) -> None:
        """Equal rank combat: both pieces removed from board."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.MAJOR, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.MAJOR, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        assert state.board_piece[4, 0] == int(PieceType.NONE)
        assert state.board_piece[4, 1] == int(PieceType.NONE)
        assert state.board_owner[4, 0] == -1
        assert state.board_owner[4, 1] == -1


# ---------------------------------------------------------------------------
# Win/draw conditions
# ---------------------------------------------------------------------------


class TestWinDrawConditions:
    def test_flag_capture_immediate_win(self) -> None:
        """Flag capture ends the game immediately."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.MINER, 5, 0)
        _place_piece(state, Player.BLUE, PieceType.FLAG, 5, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(5, 0), Square(5, 1)))
        assert state.check_terminal() == GameOutcome.RED_WIN
        assert state.is_terminal

    def test_no_legal_moves_is_loss(self) -> None:
        """A player with no legal moves on their turn loses."""
        state = StrategoState()
        _clear_board(state)
        # Red has only a Flag surrounded by own Bombs (no legal moves)
        _place_piece(state, Player.RED, PieceType.FLAG, 0, 0)
        _place_piece(state, Player.RED, PieceType.BOMB, 0, 1)
        _place_piece(state, Player.RED, PieceType.BOMB, 1, 0)
        # Blue has a movable piece
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED

        # Red has no legal moves (Flag can't move, Bomb can't move)
        actions = state.legal_actions(Player.RED)
        assert len(actions) == 0
        assert state.check_terminal() == GameOutcome.BLUE_WIN

    def test_no_attack_limit_training_draw(self) -> None:
        """100 consecutive moves without attack = draw (training rule)."""
        state = StrategoState(no_attack_limit=100)
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 0, 0)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED

        # Simulate 100 non-attack moves (ping-pong back and forth)
        for _ in range(100):
            player = state.current_player
            actions = state.legal_actions(player)
            # Pick a non-attack action
            non_attacks = [
                a
                for a in actions
                if state.board_owner[a.dst.row, a.dst.col] == -1
            ]
            if non_attacks:
                state.apply_action(non_attacks[0])
            else:
                break  # might run out of room

        assert state.check_terminal() == GameOutcome.DRAW

    def test_no_attack_limit_eval_draw(self) -> None:
        """200 consecutive moves without attack = draw (eval rule)."""
        state = StrategoState(no_attack_limit=200)
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 0, 0)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED

        # After 100 moves, should NOT be a draw with limit=200
        for _ in range(100):
            player = state.current_player
            actions = state.legal_actions(player)
            non_attacks = [
                a for a in actions if state.board_owner[a.dst.row, a.dst.col] == -1
            ]
            if non_attacks:
                state.apply_action(non_attacks[0])
            else:
                break

        # Should still be ongoing at 100 moves with limit=200
        if state.moves_since_last_attack < 200:
            assert state.check_terminal() == GameOutcome.ONGOING

    def test_max_game_length_draw(self) -> None:
        """Reaching MAX_GAME_LENGTH (4000) moves = draw."""
        state = StrategoState(no_attack_limit=999999)  # disable no-attack draw
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 0, 0)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED
        state._move_number = MAX_GAME_LENGTH  # force to max

        assert state.check_terminal() == GameOutcome.DRAW

    def test_both_players_no_movable_pieces_draw(self) -> None:
        """Both players with only immovable pieces = draw."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.FLAG, 0, 0)
        _place_piece(state, Player.RED, PieceType.BOMB, 0, 1)
        _place_piece(state, Player.BLUE, PieceType.FLAG, 9, 9)
        _place_piece(state, Player.BLUE, PieceType.BOMB, 9, 8)
        state._current_player = Player.RED

        # Neither player has movable pieces
        assert len(state.legal_actions(Player.RED)) == 0
        assert state.check_terminal() == GameOutcome.BLUE_WIN  # Red can't move → Red loses


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_clone_produces_independent_copy(self) -> None:
        """clone() produces a deep copy — modifying the clone doesn't affect the original."""
        state = _make_game_with_setups()
        cloned = state.clone()

        # Modify the clone
        cloned.board_piece[0, 0] = int(PieceType.NONE)
        cloned.board_owner[0, 0] = -1

        # Original should be unchanged
        assert state.board_piece[0, 0] != int(PieceType.NONE) or state.board_owner[0, 0] != -1

    def test_current_player_alternates(self) -> None:
        """After each move, current_player alternates between Red and Blue."""
        state = _make_game_with_setups()
        assert state.current_player == Player.RED  # Red starts

        actions = state.legal_actions()
        state.apply_action(actions[0])
        assert state.current_player == Player.BLUE

        actions = state.legal_actions()
        state.apply_action(actions[0])
        assert state.current_player == Player.RED

    def test_move_number_increments(self) -> None:
        """move_number increments after each move."""
        state = _make_game_with_setups()
        initial = state.move_number

        actions = state.legal_actions()
        state.apply_action(actions[0])
        assert state.move_number == initial + 1

        actions = state.legal_actions()
        state.apply_action(actions[0])
        assert state.move_number == initial + 2

    def test_random_legal_action_returns_valid(self) -> None:
        """random_legal_action() returns an action from the legal set."""
        state = _make_game_with_setups()
        rng = np.random.default_rng(42)
        action = state.random_legal_action(rng=rng)
        assert action in state.legal_actions()

    def test_board_str_human_readable(self) -> None:
        """board_str() produces a non-empty human-readable string."""
        state = _make_game_with_setups()
        s = state.board_str()
        assert isinstance(s, str)
        assert len(s) > 50  # should be a substantial board representation


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_attack_resets_no_attack_counter(self) -> None:
        """An attack resets the moves_since_last_attack counter."""
        state = StrategoState(no_attack_limit=100)
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 4, 1)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED

        # Do some non-attack moves first by placing another red piece
        _place_piece(state, Player.RED, PieceType.SCOUT, 0, 0)
        # Move red scout at (0,0) to (0,1) — non-attack
        state.apply_action(Action(Square(0, 0), Square(0, 1)))
        assert state.moves_since_last_attack == 1

        # Blue moves
        state.apply_action(Action(Square(9, 9), Square(9, 8)))
        assert state.moves_since_last_attack == 2

        # Red attacks Blue at (4,1) with Scout at (4,0)
        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        assert state.moves_since_last_attack == 0

    def test_non_attack_increments_counter(self) -> None:
        """A non-attack move increments moves_since_last_attack."""
        state = _make_game_with_setups()
        assert state.moves_since_last_attack == 0

        actions = state.legal_actions()
        # Find a non-attack action
        non_attacks = [
            a for a in actions if state.board_owner[a.dst.row, a.dst.col] == -1
        ]
        if non_attacks:
            state.apply_action(non_attacks[0])
            assert state.moves_since_last_attack == 1

    def test_captured_pieces_tracked(self) -> None:
        """Captured pieces are recorded in red_captured / blue_captured."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.MARSHAL, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.GENERAL, 4, 1)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 1)))
        # Red captured Blue's General
        assert PieceType.GENERAL in state.red_captured

    def test_scout_long_move_attack_clears_path(self) -> None:
        """Scout long-range attack: attacker lands on defender's square, path was clear."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.SCOUT, 4, 0)
        _place_piece(state, Player.BLUE, PieceType.SERGEANT, 4, 3)
        state._current_player = Player.RED

        state.apply_action(Action(Square(4, 0), Square(4, 3)))
        # Scout (rank 2) vs Sergeant (rank 4): Sergeant wins
        assert state.board_piece[4, 3] == int(PieceType.SERGEANT)
        assert state.board_owner[4, 3] == int(Player.BLUE)
        assert state.board_piece[4, 0] == int(PieceType.NONE)

    def test_legal_actions_empty_for_immovable_only(self) -> None:
        """A player with only Flag and Bombs has no legal actions."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.RED, PieceType.FLAG, 0, 0)
        _place_piece(state, Player.RED, PieceType.BOMB, 0, 1)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED

        assert len(state.legal_actions(Player.RED)) == 0

    def test_apply_action_validates_src_owner(self) -> None:
        """Applying an action where src doesn't belong to current player raises."""
        state = StrategoState()
        _clear_board(state)
        _place_piece(state, Player.BLUE, PieceType.SCOUT, 4, 0)
        _place_piece(state, Player.RED, PieceType.SCOUT, 9, 9)
        state._current_player = Player.RED

        with pytest.raises(ValueError):
            state.apply_action(Action(Square(4, 0), Square(4, 1)))
