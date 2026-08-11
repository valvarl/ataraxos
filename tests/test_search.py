"""Tests for stratego.search — test-time search (rollouts + magnetic mirror descent).

Covers: BeliefSampler autoregressive config sampling, RolloutEngine depth-limited
rollouts, compute_search_policy closed-form magnetic mirror descent, and
SearchEngine magnet probability computation.
"""

from __future__ import annotations

import numpy as np

from stratego.constants import PIECE_COUNTS, TOTAL_PIECES
from stratego.env.rules import StrategoState
from stratego.networks.belief_net import BeliefNetwork
from stratego.networks.move_net import MoveNetwork
from stratego.search.belief_sampler import BeliefSampler
from stratego.search.mirror_descent import SearchEngine, compute_search_policy
from stratego.search.rollouts import RolloutEngine
from stratego.types import PieceType, Player, Square

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_setup(player: Player) -> list[tuple[Square, PieceType]]:
    """Generate a valid 40-piece setup for the given player."""
    start_row, end_row = player.setup_rows
    pieces: list[PieceType] = []
    for pt, count in PIECE_COUNTS.items():
        pieces.extend([pt] * count)
    assert len(pieces) == TOTAL_PIECES
    placement: list[tuple[Square, PieceType]] = []
    idx = 0
    for r in range(start_row, end_row + 1):
        for c in range(10):
            sq = Square(r, c)
            if sq.is_lake:
                continue
            placement.append((sq, pieces[idx]))
            idx += 1
    return placement


def _make_minimal_state() -> StrategoState:
    """Create a state with a few pieces for fast testing."""
    state = StrategoState()
    state.board_owner[3, 0] = int(Player.RED)
    state.board_piece[3, 0] = int(PieceType.SCOUT)
    state.board_owner[6, 0] = int(Player.BLUE)
    state.board_piece[6, 0] = int(PieceType.SCOUT)
    state._current_player = Player.RED  # noqa: SLF001
    state._red_setup_done = True  # noqa: SLF001
    state._blue_setup_done = True  # noqa: SLF001
    return state


# ---------------------------------------------------------------------------
# 1. BeliefSampler
# ---------------------------------------------------------------------------


class TestBeliefSampler:
    def test_construction(self) -> None:
        """BeliefSampler wraps a BeliefNetwork and device string."""
        net = BeliefNetwork()
        sampler = BeliefSampler(net, device="cpu")
        assert sampler.belief_net is net
        assert sampler.device == "cpu"

    def test_sample_configs_returns_n_samples(self) -> None:
        """sample_configs returns exactly n_samples configs."""
        state = _make_minimal_state()
        net = BeliefNetwork()
        net.eval()
        sampler = BeliefSampler(net, device="cpu")
        configs = sampler.sample_configs(state, Player.RED, n_samples=3)
        assert len(configs) == 3

    def test_sample_configs_correct_types(self) -> None:
        """Each config maps Square -> PieceType with valid values (1-12)."""
        state = _make_minimal_state()
        net = BeliefNetwork()
        net.eval()
        sampler = BeliefSampler(net, device="cpu")
        configs = sampler.sample_configs(state, Player.RED, n_samples=2)
        for config in configs:
            assert len(config) == 1  # one hidden Blue piece
            for sq, pt in config.items():
                assert isinstance(sq, Square)
                assert isinstance(pt, PieceType)
                assert 1 <= int(pt) <= 12

    def test_sample_configs_no_hidden_pieces(self) -> None:
        """When opponent has no hidden pieces, returns empty dicts."""
        state = StrategoState()
        net = BeliefNetwork()
        net.eval()
        sampler = BeliefSampler(net, device="cpu")
        configs = sampler.sample_configs(state, Player.RED, n_samples=2)
        assert len(configs) == 2
        assert all(c == {} for c in configs)


# ---------------------------------------------------------------------------
# 2. RolloutEngine
# ---------------------------------------------------------------------------


class TestRolloutEngine:
    def test_construction(self) -> None:
        """RolloutEngine wraps a MoveNetwork and device string."""
        net = MoveNetwork()
        engine = RolloutEngine(net, device="cpu")
        assert engine.move_net is net
        assert engine.device == "cpu"
        assert engine.no_attack_limit == 200

    def test_rollout_returns_float(self) -> None:
        """rollout returns a float value estimate in [-1, 1]."""
        state = _make_minimal_state()
        net = MoveNetwork()
        net.eval()
        engine = RolloutEngine(net, device="cpu")
        val = engine.rollout(state, depth=2)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_rollout_for_move_applies_action(self) -> None:
        """rollout_for_move clones state, applies action, then rolls out."""
        state = _make_minimal_state()
        original_player = state.current_player
        legal = state.legal_actions()
        assert len(legal) > 0
        action = legal[0]
        net = MoveNetwork()
        net.eval()
        engine = RolloutEngine(net, device="cpu")
        val = engine.rollout_for_move(state, action, depth=3)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0
        # Original state must not be modified.
        assert state.current_player == original_player


# ---------------------------------------------------------------------------
# 3. compute_search_policy (closed-form magnetic mirror descent)
# ---------------------------------------------------------------------------


class TestComputeSearchPolicy:
    def test_sums_to_one(self) -> None:
        """Output distribution sums to 1 and is non-negative."""
        q = np.array([0.1, -0.2, 0.0, 0.3])
        net = np.array([0.3, 0.3, 0.2, 0.2])
        mag = np.array([0.25, 0.25, 0.25, 0.25])
        pi = compute_search_policy(q, net, mag)
        assert np.isclose(pi.sum(), 1.0)
        assert np.all(pi >= 0)

    def test_higher_q_higher_prob(self) -> None:
        """Higher q-value yields higher search probability."""
        q = np.array([1.0, -1.0])
        net = np.array([0.5, 0.5])
        mag = np.array([0.5, 0.5])
        pi = compute_search_policy(q, net, mag)
        assert pi[0] > pi[1]

    def test_alpha_zero_no_magnet_effect(self) -> None:
        """When alpha=0, changing magnet_probs does not change the output."""
        q = np.array([0.5, -0.5, 0.0])
        net = np.array([0.4, 0.3, 0.3])
        mag1 = np.array([0.9, 0.05, 0.05])
        mag2 = np.array([0.1, 0.8, 0.1])
        pi1 = compute_search_policy(q, net, mag1, alpha=0.0, beta=0.02)
        pi2 = compute_search_policy(q, net, mag2, alpha=0.0, beta=0.02)
        assert np.allclose(pi1, pi2)

    def test_beta_zero_no_net_effect(self) -> None:
        """When beta=0, changing net_probs does not change the output."""
        q = np.array([0.5, -0.5, 0.0])
        mag = np.array([0.4, 0.3, 0.3])
        net1 = np.array([0.9, 0.05, 0.05])
        net2 = np.array([0.1, 0.8, 0.1])
        pi1 = compute_search_policy(q, net1, mag, alpha=0.002, beta=0.0)
        pi2 = compute_search_policy(q, net2, mag, alpha=0.002, beta=0.0)
        assert np.allclose(pi1, pi2)

    def test_numerical_stability_extreme_values(self) -> None:
        """Output is finite and normalized even with extreme inputs."""
        q = np.array([1000.0, -1000.0, 500.0])
        net = np.array([1e-10, 1.0, 1e-5])
        mag = np.array([1e-10, 1e-10, 1.0])
        pi = compute_search_policy(q, net, mag)
        assert np.all(np.isfinite(pi))
        assert np.isclose(pi.sum(), 1.0)
        assert np.all(pi >= 0)

    def test_uniform_inputs_uniform_output(self) -> None:
        """When all inputs are uniform, output is uniform."""
        n = 4
        q = np.zeros(n)
        net = np.ones(n) / n
        mag = np.ones(n) / n
        pi = compute_search_policy(q, net, mag)
        assert np.allclose(pi, 1.0 / n)


# ---------------------------------------------------------------------------
# 4. SearchEngine
# ---------------------------------------------------------------------------


class TestSearchEngine:
    def test_construction(self) -> None:
        """SearchEngine wraps BeliefNetwork and MoveNetwork with correct defaults."""
        bn = BeliefNetwork()
        mn = MoveNetwork()
        engine = SearchEngine(bn, mn, device="cpu", n_rollouts=10, depth=2)
        assert engine.belief_sampler.belief_net is bn
        assert engine.rollout_engine.move_net is mn
        assert engine.device == "cpu"
        assert engine.n_rollouts == 10
        assert engine.depth == 2
        assert engine.alpha == 0.002
        assert engine.beta == 0.02

    def test_compute_magnet_probs_uniform_per_piece(self) -> None:
        """Magnet rho = uniform piece selection + uniform move for that piece."""
        state = _make_minimal_state()
        legal = state.legal_actions()
        assert len(legal) > 0
        bn = BeliefNetwork()
        mn = MoveNetwork()
        engine = SearchEngine(bn, mn, device="cpu")
        probs = engine._compute_magnet_probs(legal, state, Player.RED)
        # All probabilities are positive and sum to 1.
        assert np.all(probs > 0)
        assert np.isclose(probs.sum(), 1.0)
        # Group by source piece — each piece gets equal share.
        piece_probs: dict[Square, float] = {}
        for i, a in enumerate(legal):
            piece_probs[a.src] = piece_probs.get(a.src, 0.0) + probs[i]
        values = list(piece_probs.values())
        assert np.allclose(values, values[0])
