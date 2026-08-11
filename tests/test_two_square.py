"""Tests for the ISF two-square rule tracker (stratego/env/two_square.py).

TDD: these tests are written BEFORE the implementation. They should all fail
initially (RED), then pass once TwoSquareTracker is implemented (GREEN).

ISF Rule Chapter 10: A piece may not be moved more than TWO_SQUARE_LIMIT (3)
times non-stop between the same two squares. Tracking is per-piece. Attacks
(strikes) are exempt.
"""

from __future__ import annotations

from stratego.constants import TWO_SQUARE_LIMIT
from stratego.env.two_square import TwoSquareTracker
from stratego.types import Action, Player, Square

# ---------------------------------------------------------------------------
# Test squares — all on row 3 (no lakes), in a horizontal line
# ---------------------------------------------------------------------------
A = Square(3, 0)
B = Square(3, 1)
C = Square(3, 2)
D = Square(3, 3)
E = Square(3, 4)

# A second oscillation pair for reset tests
X = Square(7, 0)
Y = Square(7, 1)


def _move(src: Square, dst: Square) -> Action:
    """Helper to create an Action."""
    return Action(src, dst)


# ===================================================================
# 1. BASIC RULE: 3 crossings OK, 4th is violation
# ===================================================================


class TestBasicRule:
    """Core two-square rule: limit of TWO_SQUARE_LIMIT consecutive crossings."""

    def test_one_crossing_no_violation(self) -> None:
        """A single A→B move is fine."""
        tracker = TwoSquareTracker()
        action = _move(A, B)
        assert not tracker.is_violation(Player.RED, A, action, False, [], False)

    def test_two_crossings_no_violation(self) -> None:
        """A→B→A (2 crossings) is fine."""
        tracker = TwoSquareTracker()
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        assert not tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)

    def test_three_crossings_no_violation(self) -> None:
        """A→B→A→B (3 crossings = TWO_SQUARE_LIMIT) is still OK."""
        tracker = TwoSquareTracker()
        # Crossing 1: A→B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Crossing 2: B→A
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        # Crossing 3: A→B — should NOT be a violation (exactly at limit)
        assert not tracker.is_violation(Player.RED, A, _move(A, B), False, [], False)

    def test_fourth_crossing_is_violation(self) -> None:
        """After 3 crossings A↔B, the 4th crossing IS a violation."""
        tracker = TwoSquareTracker()
        # Crossing 1: A→B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Crossing 2: B→A
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        # Crossing 3: A→B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Crossing 4: B→A — should be a violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)

    def test_limit_value_is_three(self) -> None:
        """Sanity: TWO_SQUARE_LIMIT constant is 3."""
        assert TWO_SQUARE_LIMIT == 3


# ===================================================================
# 2. RESET CONDITIONS
# ===================================================================


class TestResetConditions:
    """Counter resets when the piece moves to a DIFFERENT square pair."""

    def test_different_pair_resets_counter(self) -> None:
        """Moving the same piece to a different pair resets its counter."""
        tracker = TwoSquareTracker()
        # 3 crossings on A↔B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Now piece is at B. Move to C (different pair: B↔C)
        tracker.record_move(Player.RED, B, _move(B, C), False, [], False)
        # Now back to B↔A — counter should be reset, so no violation
        assert not tracker.is_violation(Player.RED, C, _move(C, B), False, [], False)
        # Move back to B
        tracker.record_move(Player.RED, C, _move(C, B), False, [], False)
        # Now try B→A: this is a fresh start on A↔B, count=1, no violation
        assert not tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)

    def test_opponent_moves_do_not_reset_counter(self) -> None:
        """Opponent's moves do NOT reset the counter (rule is per-piece, regardless of opponent)."""
        tracker = TwoSquareTracker()
        # Red piece oscillates A↔B: 3 crossings
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Blue makes a move somewhere else
        tracker.record_move(Player.BLUE, X, _move(X, Y), False, [], False)
        # Red's A↔B counter should NOT be reset — 4th crossing is still violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)

    def test_own_other_pieces_do_not_reset_counter(self) -> None:
        """Moving a DIFFERENT piece of the same player does NOT reset the counter."""
        tracker = TwoSquareTracker()
        # Red piece at A oscillates A↔B: 3 crossings
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Red moves a DIFFERENT piece (at X) to Y
        tracker.record_move(Player.RED, X, _move(X, Y), False, [], False)
        # Original piece at B: 4th crossing A↔B is still violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)


# ===================================================================
# 3. SCOUT SPECIAL CASE
# ===================================================================


class TestScoutSpecialCase:
    """Scouts track all (start, end) pairs from stepped-over squares."""

    def test_scout_long_move_tracks_all_pairs(self) -> None:
        """Scout A→D through B,C: all 6 pairs are tracked."""
        tracker = TwoSquareTracker()
        path = [B, C, D]  # exclusive of src, inclusive of dst
        action = _move(A, D)
        tracker.record_move(Player.RED, A, action, True, path, False)
        # Now moving D→A through C,B should check all pairs.
        # The pair {A,D} has count=1, so 2nd crossing is fine.
        reverse_path = [C, B, A]
        reverse_action = _move(D, A)
        assert not tracker.is_violation(Player.RED, D, reverse_action, True, reverse_path, False)

    def test_scout_three_long_moves_violation(self) -> None:
        """Scout oscillating A↔D 4 times triggers violation on shared pair {A,D}."""
        tracker = TwoSquareTracker()
        fwd_path = [B, C, D]
        rev_path = [C, B, A]
        # 3 full round-trips (6 crossings of {A,D})
        # Crossing 1: A→D
        tracker.record_move(Player.RED, A, _move(A, D), True, fwd_path, False)
        # Crossing 2: D→A
        tracker.record_move(Player.RED, D, _move(D, A), True, rev_path, False)
        # Crossing 3: A→D
        tracker.record_move(Player.RED, A, _move(A, D), True, fwd_path, False)
        # Crossing 4: D→A — pair {A,D} has count=3, this would be 4th → violation
        assert tracker.is_violation(Player.RED, D, _move(D, A), True, rev_path, False)

    def test_scout_intermediate_squares_tracked(self) -> None:
        """Scout A→D: pair {C,D} is also tracked for that scout piece."""
        tracker = TwoSquareTracker()
        fwd_path = [B, C, D]
        rev_path = [C, B, A]
        # 3 moves: A→D, D→A, A→D — all 6 pairs reach count=3
        tracker.record_move(Player.RED, A, _move(A, D), True, fwd_path, False)
        tracker.record_move(Player.RED, D, _move(D, A), True, rev_path, False)
        tracker.record_move(Player.RED, A, _move(A, D), True, fwd_path, False)
        # Scout is now at D. A normal move D→C checks pair {C,D} which has count=3.
        # 4th crossing → violation
        assert tracker.is_violation(Player.RED, D, _move(D, C), False, [], False)

    def test_scout_different_path_no_violation(self) -> None:
        """Scout long move on different squares doesn't trigger violation for unrelated pair."""
        tracker = TwoSquareTracker()
        # Scout moves A→B (single step, no intermediates)
        tracker.record_move(Player.RED, A, _move(A, B), True, [B], False)
        tracker.record_move(Player.RED, B, _move(B, A), True, [A], False)
        tracker.record_move(Player.RED, A, _move(A, B), True, [B], False)
        # Now a scout move D→E is on a completely different pair — no violation
        assert not tracker.is_violation(Player.RED, D, _move(D, E), True, [E], False)


# ===================================================================
# 4. STRIKES (ATTACKS) ARE EXEMPT
# ===================================================================


class TestStrikesExempt:
    """Attacks are exempt from the two-square rule and reset the counter."""

    def test_attack_not_a_violation(self) -> None:
        """An attack onto the oscillation square is legal even after 3 crossings."""
        tracker = TwoSquareTracker()
        # 3 crossings on A↔B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # 4th move is an ATTACK — should NOT be a violation
        assert not tracker.is_violation(Player.RED, B, _move(B, A), False, [], True)

    def test_attack_resets_counter(self) -> None:
        """After an attack, the two-square counter resets (oscillation broken)."""
        tracker = TwoSquareTracker()
        # 3 crossings on A↔B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Attack B→A (resets counter)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], True)
        # Now A→B again — counter was reset, so no violation
        assert not tracker.is_violation(Player.RED, A, _move(A, B), False, [], False)


# ===================================================================
# 5. STATE MANAGEMENT
# ===================================================================


class TestStateManagement:
    """Clone, reset, and multi-piece tracking."""

    def test_clone_produces_independent_copy(self) -> None:
        """clone() produces an independent copy — mutations don't affect original."""
        tracker = TwoSquareTracker()
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)

        cloned = tracker.clone()

        # Original: 4th crossing is violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)
        # Cloned: same state, also violation
        assert cloned.is_violation(Player.RED, B, _move(B, A), False, [], False)

        # Record attack on clone (resets its counter)
        cloned.record_move(Player.RED, B, _move(B, A), False, [], True)
        # Clone: counter reset, no violation
        assert not cloned.is_violation(Player.RED, A, _move(A, B), False, [], False)
        # Original: still violation (unaffected by clone mutation)
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)

    def test_reset_clears_all_state(self) -> None:
        """reset() clears all tracking state."""
        tracker = TwoSquareTracker()
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Before reset: violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)
        # Reset
        tracker.reset()
        # After reset: no violation (counter cleared)
        assert not tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)

    def test_multiple_pieces_tracked_independently(self) -> None:
        """Each piece has its own counter — multiple pieces tracked simultaneously."""
        tracker = TwoSquareTracker()
        # Piece 1 (at A): 3 crossings on A↔B
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Piece 2 (at X): 1 crossing on X↔Y
        tracker.record_move(Player.RED, X, _move(X, Y), False, [], False)

        # Piece 1: 4th crossing is violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)
        # Piece 2: 2nd crossing is NOT violation
        assert not tracker.is_violation(Player.RED, Y, _move(Y, X), False, [], False)

    def test_different_players_independent(self) -> None:
        """Red and Blue pieces on the same squares are tracked independently."""
        tracker = TwoSquareTracker()
        # Red piece oscillates A↔B: 3 crossings
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        tracker.record_move(Player.RED, B, _move(B, A), False, [], False)
        tracker.record_move(Player.RED, A, _move(A, B), False, [], False)
        # Blue piece on same squares: 0 crossings
        # Blue's A↔B should not be a violation
        assert not tracker.is_violation(Player.BLUE, A, _move(A, B), False, [], False)
        # Red's is still a violation
        assert tracker.is_violation(Player.RED, B, _move(B, A), False, [], False)
