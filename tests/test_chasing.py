"""Tests for the ISF continuous-chasing rule (stratego/env/chasing.py).

TDD: these tests are written BEFORE the implementation. They should all fail
initially (RED), then pass once ChasingTracker and ZobristHasher are
implemented (GREEN).

ISF Rule Chapter 11: A player may not make a chasing move that leads to a
board position that already occurred during the current chase. Exception:
chasing back to the directly-preceding-turn square is always allowed.
"""

from __future__ import annotations

import numpy as np

from stratego.env.chasing import ChasingTracker, ZobristHasher
from stratego.types import Action, Player, Square

# ---------------------------------------------------------------------------
# Test squares — all on row 3 (no lakes), in a horizontal line
# ---------------------------------------------------------------------------
A = Square(3, 0)
B = Square(3, 1)
C = Square(3, 2)
D = Square(3, 3)
E = Square(3, 4)


def make_board(pieces: list[tuple[Square, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build (board_owner, board_piece) arrays from a list of (square, owner, piece_type).

    owner: 0 = Red, 1 = Blue. Unspecified squares are empty (-1 / 0).
    """
    owner = np.full((10, 10), -1, dtype=np.int8)
    piece = np.zeros((10, 10), dtype=np.int8)
    for sq, o, p in pieces:
        owner[sq.row, sq.col] = o
        piece[sq.row, sq.col] = p
    return owner, piece


# ===================================================================
# 1. THREAT DETECTION
# ===================================================================


class TestThreatDetection:
    """A piece threatens cardinal-adjacent opponent pieces."""

    def test_threat_detection(self) -> None:
        """Red moves piece to B(3,1), Blue piece at C(3,2). Red at B is adjacent to Blue at C."""
        board = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 1, 4)])
        tracker = ChasingTracker()
        threatened = tracker._find_threatened_pieces(
            Player.RED, Square(3, 1), board[0], board[1]
        )
        assert Square(3, 2) in threatened

    def test_no_threat_when_far(self) -> None:
        """Red at B(3,1), Blue at E(3,4). Not adjacent -> no threat."""
        board = make_board([(Square(3, 1), 0, 4), (Square(3, 4), 1, 4)])
        tracker = ChasingTracker()
        threatened = tracker._find_threatened_pieces(
            Player.RED, Square(3, 1), board[0], board[1]
        )
        assert len(threatened) == 0

    def test_threat_only_counts_opponent(self) -> None:
        """A piece adjacent to a friendly piece does not threaten it."""
        board = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 0, 4)])
        tracker = ChasingTracker()
        threatened = tracker._find_threatened_pieces(
            Player.RED, Square(3, 1), board[0], board[1]
        )
        assert len(threatened) == 0


# ===================================================================
# 2. EVASION DETECTION
# ===================================================================


class TestEvasionDetection:
    """A threatened piece moving away on the next move is an evasion."""

    def test_evasion_detected(self) -> None:
        """Red threatens Blue at C. Blue moves C->D. The move C->D is an evasion."""
        tracker = ChasingTracker()
        # Red move: A->B (threatens C)
        board_before = make_board([(Square(3, 0), 0, 2), (Square(3, 2), 1, 4)])
        board_after = make_board([(Square(3, 1), 0, 2), (Square(3, 2), 1, 4)])
        tracker.record_move(
            Player.RED,
            Action(Square(3, 0), Square(3, 1)),
            board_before[0],
            board_before[1],
            board_after[0],
            board_after[1],
        )
        # Blue move: C->D (evasion)
        board_before2 = board_after
        board_after2 = make_board([(Square(3, 1), 0, 2), (Square(3, 3), 1, 4)])
        tracker.record_move(
            Player.BLUE,
            Action(Square(3, 2), Square(3, 3)),
            board_before2[0],
            board_before2[1],
            board_after2[0],
            board_after2[1],
        )
        # Red's last threats should include C (the threatened piece)
        assert Square(3, 2) in tracker._last_threats[Player.RED]
        # Blue's last move should be detected as an evasion
        assert tracker._last_was_evasion[Player.BLUE]


# ===================================================================
# 3. CHASE LIFECYCLE
# ===================================================================


def _setup_chase() -> ChasingTracker:
    """Run Red threaten -> Blue evade -> Red threaten again. Returns tracker with active chase."""
    tracker = ChasingTracker()
    # Move 1 (Red): A->B, threatens Blue at C
    b1 = make_board([(Square(3, 0), 0, 2), (Square(3, 2), 1, 4)])
    a1 = make_board([(Square(3, 1), 0, 2), (Square(3, 2), 1, 4)])
    tracker.record_move(
        Player.RED,
        Action(Square(3, 0), Square(3, 1)),
        b1[0],
        b1[1],
        a1[0],
        a1[1],
    )
    # Move 2 (Blue): C->D, evades
    a2 = make_board([(Square(3, 1), 0, 2), (Square(3, 3), 1, 4)])
    tracker.record_move(
        Player.BLUE,
        Action(Square(3, 2), Square(3, 3)),
        a1[0],
        a1[1],
        a2[0],
        a2[1],
    )
    # Move 3 (Red): B->C, threatens Blue at D
    a3 = make_board([(Square(3, 2), 0, 2), (Square(3, 3), 1, 4)])
    tracker.record_move(
        Player.RED,
        Action(Square(3, 1), Square(3, 2)),
        a2[0],
        a2[1],
        a3[0],
        a3[1],
    )
    return tracker


class TestChaseLifecycle:
    """Chase starts after threaten -> evade -> threaten, ends when no threat."""

    def test_chase_starts_after_threat_evade_threat(self) -> None:
        """Full chase: Red threatens -> Blue evades -> Red threatens again -> chase active."""
        tracker = _setup_chase()
        assert len(tracker._chase_positions[Player.RED]) > 0

    def test_chase_ends_when_no_threat(self) -> None:
        """Red moves elsewhere (not threatening) -> chase ends."""
        tracker = _setup_chase()
        assert len(tracker._chase_positions[Player.RED]) > 0
        # Red moves C->A (away, no threat to any Blue piece)
        b_before = make_board([(Square(3, 2), 0, 2), (Square(3, 3), 1, 4)])
        b_after = make_board([(Square(3, 0), 0, 2), (Square(3, 3), 1, 4)])
        tracker.record_move(
            Player.RED,
            Action(Square(3, 2), Square(3, 0)),
            b_before[0],
            b_before[1],
            b_after[0],
            b_after[1],
        )
        # All Red chases should be cleared
        red_chases = len(tracker._chase_positions[Player.RED])
        assert red_chases == 0

    def test_no_violation_outside_chase(self) -> None:
        """Normal move not in a chase -> no violation."""
        tracker = ChasingTracker()
        board = make_board([(Square(3, 0), 0, 2), (Square(3, 4), 1, 4)])
        action = Action(Square(3, 0), Square(3, 1))
        assert not tracker.is_chasing_violation(Player.RED, action, board[0], board[1])


# ===================================================================
# 4. CHASE VIOLATION & EXCEPTION
# ===================================================================


class TestChaseViolation:
    """A chasing move that repeats a board position is a violation; back-to-preceding is exempt."""

    def test_chase_violation_position_repeat(self) -> None:
        """A chasing move that would repeat a board position seen during the chase is a violation.

        Setup: seed the tracker with an active chase and a known board hash,
        then check that a move producing the same hash is flagged as a violation.
        """
        tracker = ChasingTracker()
        board_target = make_board([(Square(8, 1), 0, 2), (Square(8, 2), 1, 4)])
        target_hash = tracker._hasher.hash_board(board_target[0], board_target[1])
        tracker._chase_positions[Player.RED].add(target_hash)
        tracker._last_was_evasion[Player.BLUE] = True
        tracker._last_chaser_src[Player.RED] = Square(8, 0)
        board_current = make_board([(Square(8, 0), 0, 2), (Square(8, 2), 1, 4)])
        action = Action(Square(8, 0), Square(8, 1))
        assert tracker.is_chasing_violation(Player.RED, action, board_current[0], board_current[1])

    def test_exception_back_to_preceding_square(self) -> None:
        """If chaser moves back to directly-preceding square -> allowed (no violation).

        Sequence: Red A->B (threat C) | Blue C->D (evade) | Red B->C (threat D, pos P1)
                  Blue D->E (evade) | Red C->B (back to preceding square B -> allowed)
        """
        tracker = ChasingTracker()
        # Move 1 (Red): A->B, threatens Blue at C
        b1 = make_board([(A, 0, 2), (C, 1, 4)])
        a1 = make_board([(B, 0, 2), (C, 1, 4)])
        tracker.record_move(Player.RED, Action(A, B), b1[0], b1[1], a1[0], a1[1])
        # Move 2 (Blue): C->D, evades
        a2 = make_board([(B, 0, 2), (D, 1, 4)])
        tracker.record_move(Player.BLUE, Action(C, D), a1[0], a1[1], a2[0], a2[1])
        # Move 3 (Red): B->C, threatens Blue at D (position P1)
        a3 = make_board([(C, 0, 2), (D, 1, 4)])
        tracker.record_move(Player.RED, Action(B, C), a2[0], a2[1], a3[0], a3[1])
        # Move 4 (Blue): D->E, evades
        a4 = make_board([(C, 0, 2), (E, 1, 4)])
        tracker.record_move(Player.BLUE, Action(D, E), a3[0], a3[1], a4[0], a4[1])
        # Move 5 (Red): C->B (back to preceding square B)
        # The chaser's last src was B (from move 3: B->C). Moving back to B is allowed.
        a5 = make_board([(B, 0, 2), (E, 1, 4)])
        tracker.record_move(Player.RED, Action(C, B), a4[0], a4[1], a5[0], a5[1])
        # Now Red wants to move B->C again (threaten E from C, position P1 again)
        # But wait - this is the exception case: dst C == last_chaser_src (B from move 5? No.)
        # Actually after move 5, last_chaser_src for the chase key (RED, C, D) was updated.
        # Let's check the simpler exception: Red moves back to where it just came from.
        # The chase key is (RED, dst, threatened). After move 5, Red is at B, not threatening.
        # So chase ended. Let's restart chase and test exception directly.
        # Move 6 (Blue): E->D (evade back)
        a6 = make_board([(B, 0, 2), (D, 1, 4)])
        tracker.record_move(Player.BLUE, Action(E, D), a5[0], a5[1], a6[0], a6[1])
        # Move 7 (Red): B->C, threatens Blue at D (position P1 again - but chase restarted)
        a7 = make_board([(C, 0, 2), (D, 1, 4)])
        tracker.record_move(Player.RED, Action(B, C), a6[0], a6[1], a7[0], a7[1])
        # Move 8 (Blue): D->E, evades
        a8 = make_board([(C, 0, 2), (E, 1, 4)])
        tracker.record_move(Player.BLUE, Action(D, E), a7[0], a7[1], a8[0], a8[1])
        # Move 9 (Red): C->B (back to preceding square B -> exception, allowed)
        # last_chaser_src for chase key (RED, C, D) should be B (from move 7: B->C)
        # Now Red wants to move C->B. dst=B == last_chaser_src=B -> exception
        action = Action(C, B)
        # This should NOT be a violation (exception: back to preceding square)
        assert not tracker.is_chasing_violation(Player.RED, action, a8[0], a8[1])


# ===================================================================
# 5. STATE MANAGEMENT
# ===================================================================


class TestStateManagement:
    """Clone, reset, and multi-chase tracking."""

    def test_clone_preserves_state(self) -> None:
        """Clone tracker, verify same active chases."""
        tracker = _setup_chase()
        cloned = tracker.clone()
        assert cloned._chase_positions == tracker._chase_positions
        assert cloned._last_threats == tracker._last_threats
        assert cloned._last_was_evasion == tracker._last_was_evasion
        assert cloned._last_chaser_src == tracker._last_chaser_src

    def test_clone_is_independent(self) -> None:
        """Mutations on clone do not affect original."""
        tracker = _setup_chase()
        cloned = tracker.clone()
        # Mutate clone
        cloned._chase_positions[Player.RED].clear()
        cloned._chase_positions[Player.BLUE].clear()
        # Original unaffected
        assert len(tracker._chase_positions[Player.RED]) > 0

    def test_reset_clears_state(self) -> None:
        """Reset, verify all state cleared."""
        tracker = _setup_chase()
        assert len(tracker._chase_positions[Player.RED]) > 0
        tracker.reset()
        assert len(tracker._chase_positions[Player.RED]) == 0
        assert len(tracker._chase_positions[Player.BLUE]) == 0
        assert tracker._last_chaser_src[Player.RED] is None
        assert tracker._last_chaser_src[Player.BLUE] is None
        assert not tracker._last_was_evasion[Player.RED]
        assert not tracker._last_was_evasion[Player.BLUE]
        assert len(tracker._last_threats[Player.RED]) == 0
        assert len(tracker._last_threats[Player.BLUE]) == 0

    def test_multiple_chases(self) -> None:
        """Two independent chases tracked separately."""
        tracker = ChasingTracker()
        # Chase 1: Red A->B threatens Blue at C
        b1 = make_board([(A, 0, 2), (C, 1, 4)])
        a1 = make_board([(B, 0, 2), (C, 1, 4)])
        tracker.record_move(Player.RED, Action(A, B), b1[0], b1[1], a1[0], a1[1])
        # Blue evades: C->D
        a2 = make_board([(B, 0, 2), (D, 1, 4)])
        tracker.record_move(Player.BLUE, Action(C, D), a1[0], a1[1], a2[0], a2[1])
        # Red threatens again: B->C
        a3 = make_board([(C, 0, 2), (D, 1, 4)])
        tracker.record_move(Player.RED, Action(B, C), a2[0], a2[1], a3[0], a3[1])
        # Now set up a second independent chase on a different part of the board
        # Use row 7 (no lakes): X=Square(7,0), Y=Square(7,1), Z=Square(7,2)
        X = Square(7, 0)
        Y = Square(7, 1)
        Z = Square(7, 2)
        # Blue threatens Red at Y from X
        b4 = make_board([(C, 0, 2), (D, 1, 4), (X, 1, 2), (Y, 0, 4)])
        a4 = make_board([(C, 0, 2), (D, 1, 4), (Y, 1, 2), (Y, 0, 4)])
        # Wait - Y is occupied. Use X->Y where Y was empty.
        b4 = make_board([(C, 0, 2), (D, 1, 4), (X, 1, 2), (Y, 0, 4)])
        # Actually let's place Red at Z and Blue at X, Blue moves X->Y to threaten Z
        b4 = make_board([(C, 0, 2), (D, 1, 4), (X, 1, 2), (Z, 0, 4)])
        a4 = make_board([(C, 0, 2), (D, 1, 4), (Y, 1, 2), (Z, 0, 4)])
        tracker.record_move(Player.BLUE, Action(X, Y), b4[0], b4[1], a4[0], a4[1])
        # Red evades: Z->Square(7,3)
        W = Square(7, 3)
        a5 = make_board([(C, 0, 2), (D, 1, 4), (Y, 1, 2), (W, 0, 4)])
        tracker.record_move(Player.RED, Action(Z, W), a4[0], a4[1], a5[0], a5[1])
        # Blue threatens again: Y->Z
        a6 = make_board([(C, 0, 2), (D, 1, 4), (Z, 1, 2), (W, 0, 4)])
        tracker.record_move(Player.BLUE, Action(Y, Z), a5[0], a5[1], a6[0], a6[1])
        # Red evaded on move 5 (didn't threaten) → Red's chase ended.
        # Blue's chase should be active (Blue threatened on move 6 after Red evaded).
        red_chases = len(tracker._chase_positions[Player.RED])
        blue_chases = len(tracker._chase_positions[Player.BLUE])
        assert red_chases == 0
        assert blue_chases > 0


# ===================================================================
# 6. ZOBRIST HASHER
# ===================================================================


class TestZobristHasher:
    """Zobrist hashing: same board -> same hash, different board -> different hash."""

    def test_same_board_same_hash(self) -> None:
        """Two identical boards produce the same hash."""
        hasher = ZobristHasher()
        board1 = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 1, 4)])
        board2 = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 1, 4)])
        h1 = hasher.hash_board(board1[0], board1[1])
        h2 = hasher.hash_board(board2[0], board2[1])
        assert h1 == h2

    def test_different_board_different_hash(self) -> None:
        """Two different boards produce different hashes."""
        hasher = ZobristHasher()
        board1 = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 1, 4)])
        board2 = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 1, 5)])  # different piece
        h1 = hasher.hash_board(board1[0], board1[1])
        h2 = hasher.hash_board(board2[0], board2[1])
        assert h1 != h2

    def test_empty_board_hash(self) -> None:
        """Empty board has a deterministic hash (zero)."""
        hasher = ZobristHasher()
        owner = np.full((10, 10), -1, dtype=np.int8)
        piece = np.zeros((10, 10), dtype=np.int8)
        assert hasher.hash_board(owner, piece) == 0

    def test_hash_is_deterministic_across_instances(self) -> None:
        """Two hashers with the same seed produce the same hash for the same board."""
        board = make_board([(Square(3, 1), 0, 4), (Square(3, 2), 1, 4)])
        h1 = ZobristHasher(seed=42).hash_board(board[0], board[1])
        h2 = ZobristHasher(seed=42).hash_board(board[0], board[1])
        assert h1 == h2
