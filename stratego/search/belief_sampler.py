"""Sample opponent hidden piece configurations from the belief network.

Implements autoregressive decoding from the BeliefNetwork (arXiv:2511.07312)
to draw samples of opponent hidden piece placements consistent with the
current infostate.  These samples are consumed by the test-time search
engine to run rollouts over plausible opponent configurations.
"""

from __future__ import annotations

import torch

from stratego.env.infostate import compute_infostate
from stratego.env.rules import StrategoState
from stratego.networks.belief_net import BeliefNetwork
from stratego.types import PieceType, Player, Square

__all__ = ["BeliefSampler"]


class BeliefSampler:
    """Samples opponent hidden piece configurations from the belief network.

    The belief network is an encoder-decoder transformer that autoregressively
    predicts opponent hidden piece types in row-major order.  This class wraps
    the decoding loop: for each sample, it iterates over every hidden opponent
    square, feeds the tokens sampled so far (plus a placeholder for the current
    position) through the decoder, and draws the next piece type from the
    softmax of the last position's logits.
    """

    def __init__(self, belief_net: BeliefNetwork, device: str = "cpu") -> None:
        self.belief_net = belief_net
        self.device = device

    def sample_configs(
        self,
        state: StrategoState,
        player: Player,
        n_samples: int = 100,
    ) -> list[dict[Square, PieceType]]:
        """Sample ``n_samples`` configurations of opponent hidden pieces.

        Parameters
        ----------
        state:
            Current game state.
        player:
            The player doing the searching (opponent pieces are hidden).
        n_samples:
            Number of independent configurations to draw.

        Returns
        -------
        list[dict[Square, PieceType]]:
            A list of length ``n_samples``.  Each entry maps every hidden
            opponent square to a sampled ``PieceType``.  When the opponent
            has no hidden pieces, each entry is an empty dict.
        """
        configs: list[dict[Square, PieceType]] = []
        self.belief_net.eval()

        infostate = compute_infostate(state, player)
        info_tensor = torch.from_numpy(infostate).float().unsqueeze(0).to(self.device)

        opp = player.opponent
        hidden_squares: list[Square] = []
        for r in range(10):
            for c in range(10):
                if state.board_owner[r, c] == int(opp):
                    hidden_squares.append(Square(r, c))

        if not hidden_squares:
            return [{} for _ in range(n_samples)]

        hidden_squares_sorted = sorted(hidden_squares, key=lambda s: (s.row, s.col))

        hidden_mask = torch.zeros(1, 100, dtype=torch.bool, device=self.device)
        for sq in hidden_squares_sorted:
            hidden_mask[0, sq.idx] = True

        with torch.no_grad():
            for _ in range(n_samples):
                # sampled accumulates the piece-type tokens drawn so far.
                sampled = torch.zeros(1, 0, dtype=torch.long, device=self.device)
                config: dict[Square, PieceType] = {}
                for sq in hidden_squares_sorted:
                    # Append a placeholder token at the current position so
                    # the decoder produces logits for it.  The placeholder
                    # is replaced by the sampled token in the next iteration.
                    placeholder = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    tokens = torch.cat([sampled, placeholder], dim=1)

                    logits = self.belief_net(info_tensor, hidden_mask, tokens)
                    # logits: (1, S, 12) — take the last position's logits.
                    step_logits = logits[0:1, -1]  # (1, 12)
                    probs = torch.softmax(step_logits, dim=-1)  # (1, 12)
                    piece_idx = torch.multinomial(probs, num_samples=1)  # (1, 1)

                    sampled = torch.cat([sampled, piece_idx], dim=1)
                    config[sq] = PieceType(int(piece_idx.item()) + 1)

                configs.append(config)

        return configs
