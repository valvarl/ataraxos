"""Tests for stratego.training.selfplay — SelfPlayGenerator.

TDD: tests written before the implementation.
Covers: construction, generate_setups (valid piece counts, unique squares,
correct rows), generate_games (transitions, infostate shape, outcome values,
move_idx range, move_history tracking, setup recorded, smoke test),
_random_setup (valid for both players, different on re-call), and
_map_policy_to_actions (normalization, uniform fallback).
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.constants import (
    NUM_INFOSTATE_CHANNELS,
    NUM_SQUARES,
    PIECE_COUNTS,
    TOTAL_PIECES,
    TRAINING_NO_ATTACK_LIMIT,
)
from stratego.networks.move_net import MoveNetwork
from stratego.networks.setup_net import SetupNetwork
from stratego.training.selfplay import SelfPlayGame, SelfPlayGenerator, SelfPlayTransition
from stratego.types import Action, PieceType, Player, Square

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_setup_net() -> SetupNetwork:
    """Small SetupNetwork for fast tests."""
    return SetupNetwork(depth=1, dim=32, heads=2, ff=64)


def _make_tiny_move_net() -> MoveNetwork:
    """Small MoveNetwork for fast tests."""
    return MoveNetwork(depth=2, dim=64, heads=2, ff=128)


def _make_generator(no_attack_limit: int = 10) -> SelfPlayGenerator:
    """Create a SelfPlayGenerator with tiny networks and fast game termination."""
    return SelfPlayGenerator(
        setup_net=_make_tiny_setup_net(),
        move_net=_make_tiny_move_net(),
        num_envs=4,
        device="cpu",
        no_attack_limit=no_attack_limit,
    )


def _count_pieces(setup: list[tuple[Square, PieceType]]) -> dict[PieceType, int]:
    """Count pieces by type in a setup."""
    counts: dict[PieceType, int] = {}
    for _sq, pt in setup:
        counts[pt] = counts.get(pt, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_attributes_set_correctly(self) -> None:
        """Constructor stores all attributes."""
        gen = _make_generator()
        assert gen.num_envs == 4
        assert gen.device == "cpu"
        assert gen.no_attack_limit == 10
        assert gen.setup_pool == []

    def test_default_values(self) -> None:
        """Default num_envs=16 and no_attack_limit=TRAINING_NO_ATTACK_LIMIT."""
        gen = SelfPlayGenerator(
            setup_net=_make_tiny_setup_net(),
            move_net=_make_tiny_move_net(),
        )
        assert gen.num_envs == 16
        assert gen.no_attack_limit == TRAINING_NO_ATTACK_LIMIT


# ---------------------------------------------------------------------------
# generate_setups
# ---------------------------------------------------------------------------


class TestGenerateSetups:
    def test_produces_valid_piece_counts(self) -> None:
        """Each setup has 40 pieces with correct per-type counts."""
        gen = _make_generator()
        setups = gen.generate_setups(n_setups=3)
        assert len(setups) == 3
        for setup in setups:
            assert len(setup) == TOTAL_PIECES
            counts = _count_pieces(setup)
            for pt, expected in PIECE_COUNTS.items():
                assert counts.get(pt, 0) == expected, f"{pt.name}: got {counts.get(pt, 0)}"

    def test_setups_use_red_setup_rows(self) -> None:
        """All squares are in Red's setup zone (rows 0-3)."""
        gen = _make_generator()
        setups = gen.generate_setups(n_setups=2)
        for setup in setups:
            for sq, _pt in setup:
                assert 0 <= sq.row <= 3
                assert 0 <= sq.col < 10

    def test_setups_have_unique_squares(self) -> None:
        """No two pieces occupy the same square within a setup."""
        gen = _make_generator()
        setups = gen.generate_setups(n_setups=2)
        for setup in setups:
            squares = [sq for sq, _pt in setup]
            assert len(squares) == len(set(squares))


# ---------------------------------------------------------------------------
# _random_setup
# ---------------------------------------------------------------------------


class TestRandomSetup:
    def test_produces_valid_setups_for_both_players(self) -> None:
        """Both red and blue setups have 40 pieces, correct counts, correct rows."""
        gen = _make_generator()
        red, blue = gen._random_setup()
        assert len(red) == TOTAL_PIECES
        assert len(blue) == TOTAL_PIECES

        for setup, player in [(red, Player.RED), (blue, Player.BLUE)]:
            counts = _count_pieces(setup)
            for pt, expected in PIECE_COUNTS.items():
                assert counts.get(pt, 0) == expected
            start, end = player.setup_rows
            for sq, _pt in setup:
                assert start <= sq.row <= end

    def test_random_setups_are_different(self) -> None:
        """Two calls produce different piece orderings (with high probability)."""
        gen = _make_generator()
        red1, _ = gen._random_setup()
        red2, _ = gen._random_setup()
        pieces1 = [pt for _sq, pt in red1]
        pieces2 = [pt for _sq, pt in red2]
        assert pieces1 != pieces2


# ---------------------------------------------------------------------------
# _map_policy_to_actions
# ---------------------------------------------------------------------------


class TestMapPolicyToActions:
    def test_normalizes_correctly(self) -> None:
        """Output sums to 1 and has same length as legal_actions."""
        gen = _make_generator()
        policy_probs = np.random.rand(NUM_SQUARES * NUM_SQUARES)
        legal = [
            Action(Square(0, 0), Square(0, 1)),
            Action(Square(0, 0), Square(1, 0)),
        ]
        probs = gen._map_policy_to_actions(policy_probs, legal)
        assert len(probs) == len(legal)
        assert probs.sum() == pytest.approx(1.0)
        assert (probs >= 0).all()

    def test_uniform_when_all_zero(self) -> None:
        """When all policy probs are zero, returns uniform distribution."""
        gen = _make_generator()
        policy_probs = np.zeros(NUM_SQUARES * NUM_SQUARES)
        legal = [
            Action(Square(0, 0), Square(0, 1)),
            Action(Square(0, 0), Square(1, 0)),
        ]
        probs = gen._map_policy_to_actions(policy_probs, legal)
        assert len(probs) == len(legal)
        assert probs.sum() == pytest.approx(1.0)
        for p in probs:
            assert p == pytest.approx(1.0 / len(legal))


# ---------------------------------------------------------------------------
# generate_games
# ---------------------------------------------------------------------------


class TestGenerateGames:
    def test_produces_games_with_transitions(self) -> None:
        """Each game is a SelfPlayGame with at least one transition."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=2)
        assert len(games) == 2
        for game in games:
            assert isinstance(game, SelfPlayGame)
            assert len(game.transitions) > 0
            for t in game.transitions:
                assert isinstance(t, SelfPlayTransition)

    def test_transition_infostate_shape(self) -> None:
        """Every transition's infostate has shape (488, 10, 10)."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=1)
        for game in games:
            for t in game.transitions:
                assert t.infostate.shape == (NUM_INFOSTATE_CHANNELS, 10, 10)

    def test_outcome_in_valid_set(self) -> None:
        """Game outcome is in {-1, 0, 1}."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=3)
        for game in games:
            assert game.outcome in {-1, 0, 1}

    def test_move_idx_in_valid_range(self) -> None:
        """Every transition's move_idx is in [0, 10000)."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=1)
        for game in games:
            for t in game.transitions:
                assert 0 <= t.move_idx < NUM_SQUARES * NUM_SQUARES

    def test_move_history_tracking(self) -> None:
        """Each transition's move_idx matches action.src.idx*100+action.dst.idx."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=1)
        for game in games:
            for t in game.transitions:
                expected = t.action.src.idx * NUM_SQUARES + t.action.dst.idx
                assert t.move_idx == expected
                assert t.player in (Player.RED, Player.BLUE)

    def test_setup_recorded(self) -> None:
        """Each game records both players' 40-piece setups."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=1)
        for game in games:
            assert len(game.setup_red) == TOTAL_PIECES
            assert len(game.setup_blue) == TOTAL_PIECES

    def test_smoke_two_games_no_crash(self) -> None:
        """Smoke test: 2 games on tiny config completes without error."""
        gen = _make_generator(no_attack_limit=10)
        games = gen.generate_games(n_games=2)
        assert len(games) == 2
        for game in games:
            assert len(game.transitions) > 0
