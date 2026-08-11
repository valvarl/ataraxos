"""Self-play data generation for Stratego (arXiv:2511.07312).

Generates self-play games by direct policy sampling from the setup and move
networks. Paper: 1536 envs/GPU, 202 moves per env (101 per player), setup pool
of 1000 setups per player per GPU regenerated each iteration.

This module uses the pure-Python StrategoState for game simulation. CUDA
integration (batched parallel environments) is deferred to T23.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from stratego.constants import (
    NUM_SQUARES,
    PIECE_COUNTS,
    TOTAL_PIECES,
    TRAINING_NO_ATTACK_LIMIT,
)
from stratego.env.infostate import compute_infostate
from stratego.env.rules import StrategoState
from stratego.networks.move_net import MoveNetwork
from stratego.networks.setup_net import SetupNetwork
from stratego.types import (
    NUM_PIECE_TYPES,
    PIECE_TYPES,
    Action,
    GameOutcome,
    PieceType,
    Player,
    Square,
)

__all__ = ["SelfPlayTransition", "SelfPlayGame", "SelfPlayGenerator"]


@dataclass
class SelfPlayTransition:
    """A single transition (state, action, player) from self-play.

    Attributes:
        infostate: ``(488, 10, 10)`` float32 infostate tensor.
        action: The action taken at this transition.
        player: The player who took the action.
        move_idx: Flat index ``src*100+dst`` into the move network's 10000-dim
            policy output.
        is_attack: Whether this action was an attack (combat with opponent).
    """

    infostate: np.ndarray
    action: Action
    player: Player
    move_idx: int
    is_attack: bool


@dataclass
class SelfPlayGame:
    """A complete self-play game.

    Attributes:
        transitions: Ordered list of transitions from the game.
        outcome: ``1`` if Red wins, ``-1`` if Blue wins, ``0`` if draw.
        setup_red: Red's 40-piece setup placement.
        setup_blue: Blue's 40-piece setup placement.
    """

    transitions: list[SelfPlayTransition]
    outcome: int
    setup_red: list[tuple[Square, PieceType]]
    setup_blue: list[tuple[Square, PieceType]]


class SelfPlayGenerator:
    """Generates self-play data by sampling from policy networks.

    Paper: 1536 envs/GPU, 202 moves per env (101 per player), direct policy
    sampling (NO search in data gen), setup pool regenerated each iteration.

    Parameters:
        setup_net: The setup network for autoregressive setup generation.
        move_net: The move network for move selection.
        num_envs: Number of parallel environments (paper: 1536). Stored as
            an attribute; ``generate_games`` defaults to this count.
        device: Torch device for network inference.
        no_attack_limit: Draw threshold for consecutive moves without attack.
    """

    def __init__(
        self,
        setup_net: SetupNetwork,
        move_net: MoveNetwork,
        num_envs: int = 16,
        device: str = "cpu",
        no_attack_limit: int = TRAINING_NO_ATTACK_LIMIT,
    ) -> None:
        self.setup_net = setup_net
        self.move_net = move_net
        self.num_envs = num_envs
        self.device = device
        self.no_attack_limit = no_attack_limit
        self.setup_pool: list[list[tuple[Square, PieceType]]] = []

    # ------------------------------------------------------------------
    # Setup generation
    # ------------------------------------------------------------------

    def generate_setups(
        self, n_setups: int = 1000
    ) -> list[list[tuple[Square, PieceType]]]:
        """Sample setups from the setup network (autoregressive).

        Generates ``n_setups`` setups, each with exactly 40 pieces and correct
        piece-type counts. Piece types whose count is exhausted are masked out
        so the sampled distribution only places valid remaining pieces.

        Setups are placed in row-major order on Red's setup rows (0-3). For
        Blue, mirror the rows (``row -> 9 - row``) at consumption time.

        Args:
            n_setups: Number of setups to generate.

        Returns:
            List of setups, each a list of ``(Square, PieceType)`` tuples.
        """
        setups: list[list[tuple[Square, PieceType]]] = []
        self.setup_net.eval()
        with torch.no_grad():
            for _ in range(n_setups):
                tokens = torch.zeros(1, 0, dtype=torch.long, device=self.device)
                pieces: list[tuple[Square, PieceType]] = []
                remaining: dict[PieceType, int] = dict(PIECE_COUNTS)

                for step in range(TOTAL_PIECES):
                    probs = self._sample_next_piece(tokens, remaining)
                    next_idx = torch.multinomial(probs, 1)  # (1, 1)
                    tokens = torch.cat([tokens, next_idx], dim=1)

                    pt = PIECE_TYPES[int(next_idx.item())]
                    remaining[pt] -= 1
                    pieces.append((Square(step // 10, step % 10), pt))

                setups.append(pieces)
        return setups

    def _sample_next_piece(
        self, tokens: torch.Tensor, remaining: dict[PieceType, int]
    ) -> torch.Tensor:
        """Compute the next-piece probability distribution.

        For the first piece (empty ``tokens``), uses a uniform distribution
        over available piece types. For subsequent pieces, queries the setup
        network and masks exhausted types.

        Args:
            tokens: Sequence of piece-type indices placed so far, shape ``(1, S)``.
            remaining: Remaining count per piece type.

        Returns:
            Probability tensor of shape ``(1, NUM_PIECE_TYPES)``.
        """
        mask = torch.full(
            (NUM_PIECE_TYPES,), float("-inf"), device=self.device
        )
        for i, pt in enumerate(PIECE_TYPES):
            if remaining[pt] > 0:
                mask[i] = 0.0

        if tokens.size(1) == 0:
            # No context yet: uniform over available types.
            probs = torch.zeros(1, NUM_PIECE_TYPES, device=self.device)
            avail = mask == 0.0
            probs[0][avail] = 1.0
            return probs / probs.sum()

        _, _, policy = self.setup_net(tokens)
        logits = policy[:, -1]  # (1, num_piece_types)
        logits = logits + mask
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Game generation
    # ------------------------------------------------------------------

    def generate_games(self, n_games: int = 16) -> list[SelfPlayGame]:
        """Run ``n_games`` self-play games, sampling from the move network.

        Each game uses a random valid setup for both players (from
        ``_random_setup``). Moves are sampled directly from the move network
        policy — no search is performed during data generation.

        Args:
            n_games: Number of games to generate.

        Returns:
            List of :class:`SelfPlayGame` objects.
        """
        games: list[SelfPlayGame] = []
        self.move_net.eval()

        for _ in range(n_games):
            state = StrategoState(no_attack_limit=self.no_attack_limit)
            red_setup, blue_setup = self._random_setup()
            state.apply_setup(Player.RED, red_setup)
            state.apply_setup(Player.BLUE, blue_setup)

            transitions: list[SelfPlayTransition] = []
            move_history: list[Action] = []
            moved_squares: set[Square] = set()
            revealed_squares: set[Square] = set()

            while not state.is_terminal:
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

                policy_probs = self._query_move_policy(infostate)
                action_probs = self._map_policy_to_actions(policy_probs, legal)
                action_idx = int(np.random.choice(len(legal), p=action_probs))
                action = legal[action_idx]

                is_attack = bool(
                    state.board_owner[action.dst.row, action.dst.col] >= 0
                )

                transitions.append(
                    SelfPlayTransition(
                        infostate=infostate,
                        action=action,
                        player=player,
                        move_idx=action.src.idx * NUM_SQUARES + action.dst.idx,
                        is_attack=is_attack,
                    )
                )

                state.apply_action(action)
                move_history.append(action)
                moved_squares.add(action.src)
                moved_squares.add(action.dst)
                if is_attack:
                    revealed_squares.add(action.src)
                    revealed_squares.add(action.dst)

            outcome = self._get_outcome(state)
            games.append(
                SelfPlayGame(
                    transitions=transitions,
                    outcome=outcome,
                    setup_red=red_setup,
                    setup_blue=blue_setup,
                )
            )

        return games

    def _query_move_policy(self, infostate: np.ndarray) -> np.ndarray:
        """Run the move network on a single infostate.

        Args:
            infostate: ``(488, 10, 10)`` float32 array.

        Returns:
            Softmax policy probabilities of shape ``(10000,)``.
        """
        with torch.no_grad():
            info_tensor = (
                torch.from_numpy(infostate).float().unsqueeze(0).to(self.device)
            )
            _, policy = self.move_net(info_tensor)
            return torch.softmax(policy, dim=-1).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_setup(
        self,
    ) -> tuple[list[tuple[Square, PieceType]], list[tuple[Square, PieceType]]]:
        """Generate a random valid setup for Red and Blue.

        Shuffles the standard 40-piece roster and places pieces in row-major
        order within each player's setup zone (Red: rows 0-3, Blue: rows 6-9).

        Returns:
            Tuple of ``(red_setup, blue_setup)``, each a list of
            ``(Square, PieceType)`` tuples.
        """
        pieces: list[PieceType] = []
        for pt, count in PIECE_COUNTS.items():
            pieces.extend([pt] * count)

        red_pieces = pieces[:]
        random.shuffle(red_pieces)
        red_setup = [
            (Square(i // 10, i % 10), red_pieces[i]) for i in range(TOTAL_PIECES)
        ]

        blue_pieces = pieces[:]
        random.shuffle(blue_pieces)
        blue_setup = [
            (Square(6 + i // 10, i % 10), blue_pieces[i])
            for i in range(TOTAL_PIECES)
        ]

        return red_setup, blue_setup

    def _map_policy_to_actions(
        self, policy_probs: np.ndarray, legal_actions: list[Action]
    ) -> np.ndarray:
        """Map flat policy probabilities to legal action probabilities.

        Extracts the policy probability for each legal action from the flat
        10000-dim policy, then renormalizes. Falls back to a uniform
        distribution if all legal-action probabilities are near zero.

        Args:
            policy_probs: Flat policy probabilities of shape ``(10000,)``.
            legal_actions: List of legal actions.

        Returns:
            Normalized probability array of shape ``(len(legal_actions),)``.
        """
        probs: np.ndarray = np.array(
            [
                policy_probs[a.src.idx * NUM_SQUARES + a.dst.idx]
                for a in legal_actions
            ]
        )
        total = float(probs.sum())
        if total < 1e-8:
            return np.full(len(legal_actions), 1.0 / len(legal_actions))
        return probs / total

    def _get_outcome(self, state: StrategoState) -> int:
        """Map game outcome to a scalar reward.

        Args:
            state: The terminal (or current) game state.

        Returns:
            ``1`` if Red wins, ``-1`` if Blue wins, ``0`` for draw or ongoing.
        """
        if state.outcome == GameOutcome.RED_WIN:
            return 1
        if state.outcome == GameOutcome.BLUE_WIN:
            return -1
        return 0
