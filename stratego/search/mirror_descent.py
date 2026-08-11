"""Test-time search via magnetic mirror descent (Sokota et al. 2023/2024).

Implements the closed-form update-equivalence search policy:

    pi_search proportional to [exp(q_hat) * rho^alpha * pi_theta^beta]^(1/(alpha+beta))

where:
  * ``q_hat``  — averaged value estimates from depth-limited rollouts.
  * ``rho``    — magnet policy (uniform piece selection + uniform move for
                 that piece).
  * ``pi_theta`` — move network's policy head probabilities.
  * ``alpha``  — reverse-KL coefficient to the magnet (paper: 0.002).
  * ``beta``   — reverse-KL coefficient to the network (paper: 0.02).

The update is computed in log-space for numerical stability.
"""

from __future__ import annotations

import numpy as np
import torch

from stratego.env.infostate import compute_infostate
from stratego.env.rules import StrategoState
from stratego.networks.belief_net import BeliefNetwork
from stratego.networks.move_net import MoveNetwork
from stratego.search.belief_sampler import BeliefSampler
from stratego.search.rollouts import RolloutEngine
from stratego.types import Action, Player, Square

__all__ = ["SearchEngine", "compute_search_policy"]


def compute_search_policy(
    q_values: np.ndarray,
    net_probs: np.ndarray,
    magnet_probs: np.ndarray,
    alpha: float = 0.002,
    beta: float = 0.02,
) -> np.ndarray:
    """Compute ``pi_search`` via closed-form magnetic mirror descent.

    ``pi_search proportional to [exp(q_hat) * rho^alpha * pi_theta^beta]^(1/(alpha+beta))``

    In log-space::

        log_pi = (q_hat + alpha * log(rho) + beta * log(pi_theta)) / (alpha + beta)

    The max is subtracted before exponentiation for numerical stability,
    then the result is normalised to sum to 1.

    Parameters
    ----------
    q_values:
        Averaged value estimates, shape ``(n_legal_moves,)``.
    net_probs:
        Move-network policy probabilities, shape ``(n_legal_moves,)``.
    magnet_probs:
        Magnet policy probabilities, shape ``(n_legal_moves,)``.
    alpha:
        Reverse-KL coefficient to the magnet (paper: 0.002).
    beta:
        Reverse-KL coefficient to the network (paper: 0.02).

    Returns
    -------
    np.ndarray
        Search policy probabilities, shape ``(n_legal_moves,)``, summing to 1.
    """
    eps = 1e-8
    log_unnorm = (
        q_values
        + alpha * np.log(magnet_probs + eps)
        + beta * np.log(net_probs + eps)
    ) / (alpha + beta)
    log_unnorm -= np.max(log_unnorm)
    pi_search = np.exp(log_unnorm)
    pi_search /= np.sum(pi_search)
    return pi_search


class SearchEngine:
    """Test-time search via update-equivalence (magnetic mirror descent).

    Pipeline (paper: 1000 rollouts x depth 40, alpha=0.002, beta=0.02):

      1. Sample ``n_rollouts // n_legal`` opponent hidden configs from the
         belief network.
      2. For each legal move, run one rollout per sampled config (depth 40)
         and average the value estimates -> ``q_hat``.
      3. Query the move network for the policy probabilities ``pi_theta``.
      4. Compute the magnet policy ``rho`` (uniform piece + uniform move).
      5. ``pi_search = compute_search_policy(q_hat, pi_theta, rho, alpha, beta)``.
      6. Sample the final move from ``pi_search``.
    """

    def __init__(
        self,
        belief_net: BeliefNetwork,
        move_net: MoveNetwork,
        device: str = "cpu",
        n_rollouts: int = 1000,
        depth: int = 40,
        alpha: float = 0.002,
        beta: float = 0.02,
    ) -> None:
        self.belief_sampler = BeliefSampler(belief_net, device)
        self.rollout_engine = RolloutEngine(move_net, device)
        self.device = device
        self.n_rollouts = n_rollouts
        self.depth = depth
        self.alpha = alpha
        self.beta = beta

    def search(self, state: StrategoState, player: Player) -> Action:
        """Run test-time search and return the selected action."""
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions")

        # --- Move-network policy probabilities ---------------------------
        infostate = compute_infostate(state, player)
        info_t = torch.from_numpy(infostate).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, policy = self.rollout_engine.move_net(info_t)
            pol_probs = torch.softmax(policy, dim=-1).squeeze(0).cpu().numpy()
        net_probs = np.array(
            [pol_probs[a.src.idx * 100 + a.dst.idx] for a in legal]
        )
        net_probs = net_probs / (net_probs.sum() + 1e-8)

        # --- Magnet policy ------------------------------------------------
        magnet_probs = self._compute_magnet_probs(legal, state, player)

        # --- Sample opponent configs and run rollouts ---------------------
        n_configs = max(1, self.n_rollouts // len(legal))
        configs = self.belief_sampler.sample_configs(state, player, n_configs)

        q_values = np.zeros(len(legal))
        for i, action in enumerate(legal):
            total_value = 0.0
            for _ in configs:
                total_value += self.rollout_engine.rollout_for_move(
                    state, action, self.depth
                )
            q_values[i] = total_value / len(configs)

        # --- Closed-form magnetic mirror descent -------------------------
        pi_search = compute_search_policy(
            q_values, net_probs, magnet_probs, self.alpha, self.beta
        )

        action_idx = int(np.random.choice(len(legal), p=pi_search))
        return legal[action_idx]

    def _compute_magnet_probs(
        self,
        legal: list[Action],
        state: StrategoState,
        player: Player,
    ) -> np.ndarray:
        """Magnet policy ``rho`` = uniform piece + uniform move for that piece.

        Each movable piece gets ``1 / n_pieces`` of the probability mass,
        split equally among that piece's legal moves.
        """
        piece_moves: dict[Square, list[int]] = {}
        for i, action in enumerate(legal):
            piece_moves.setdefault(action.src, []).append(i)

        n_pieces = len(piece_moves)
        probs = np.zeros(len(legal))
        for indices in piece_moves.values():
            p = 1.0 / n_pieces / len(indices)
            for idx in indices:
                probs[idx] = p
        return probs
