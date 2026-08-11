"""Depth-limited rollouts using the move network for test-time search.

A rollout plays out the game from the current state for up to ``depth``
half-moves, sampling each move from the move network's policy head.  The
value of the terminal (or depth-limited) position is read from the move
network's value head and returned as a scalar in ``[-1, 1]``.
"""

from __future__ import annotations

import numpy as np
import torch

from stratego.env.infostate import compute_infostate
from stratego.env.rules import StrategoState
from stratego.networks.move_net import MoveNetwork
from stratego.types import Action, Square

__all__ = ["RolloutEngine"]


class RolloutEngine:
    """Runs depth-limited rollouts using the move network.

    Each rollout step:
      1. Compute the infostate for the current player (including move history
         and moved-square tracking).
      2. Forward through the move network to obtain value and policy logits.
      3. Mask the policy to legal actions, softmax, and sample one action.
      4. Apply the action and record it in the move history.

    After the loop (terminal or depth exhausted), the value head is queried
    one final time and ``P(win) - P(loss)`` is returned.
    """

    def __init__(
        self,
        move_net: MoveNetwork,
        device: str = "cpu",
        no_attack_limit: int = 200,
    ) -> None:
        self.move_net = move_net
        self.device = device
        self.no_attack_limit = no_attack_limit

    def rollout(self, state: StrategoState, depth: int = 40) -> float:
        """Run a single rollout from ``state``, return a value estimate.

        The state is modified in place.  Callers that need to preserve the
        original state should pass a clone (see :meth:`rollout_for_move`).

        Returns
        -------
        float
            ``P(win) - P(loss)`` from the move network's value head at the
            final position, in ``[-1, 1]``.
        """
        self.move_net.eval()
        move_history: list[Action] = []
        moved_squares: set[Square] = set()
        revealed_squares: set[Square] = set()

        for _ in range(depth):
            if state.is_terminal:
                break

            player = state.current_player
            infostate = compute_infostate(
                state,
                player,
                move_history=move_history,
                moved_squares=moved_squares,
                revealed_squares=revealed_squares,
            )
            legal = state.legal_actions()
            if not legal:
                break

            with torch.no_grad():
                info_t = torch.from_numpy(infostate).float().unsqueeze(0).to(self.device)
                _, policy = self.move_net(info_t)
                probs = torch.softmax(policy, dim=-1).squeeze(0).cpu().numpy()

            action_probs = np.array(
                [probs[a.src.idx * 100 + a.dst.idx] for a in legal]
            )
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            action_idx = int(np.random.choice(len(legal), p=action_probs))
            action = legal[action_idx]

            state.apply_action(action)
            move_history.append(action)
            moved_squares.add(action.src)
            moved_squares.add(action.dst)

        # Final value estimate from the move network's value head.
        player = state.current_player
        infostate = compute_infostate(state, player)
        with torch.no_grad():
            info_t = torch.from_numpy(infostate).float().unsqueeze(0).to(self.device)
            value, _ = self.move_net(info_t)

        value_probs = torch.softmax(value, dim=-1).squeeze(0).cpu().numpy()
        return float(value_probs[0] - value_probs[1])

    def rollout_for_move(
        self,
        state: StrategoState,
        action: Action,
        depth: int = 40,
    ) -> float:
        """Apply ``action`` to a clone, then roll out ``depth - 1`` more moves.

        The original ``state`` is not modified.
        """
        state = state.clone()
        state.apply_action(action)
        return self.rollout(state, depth - 1)
