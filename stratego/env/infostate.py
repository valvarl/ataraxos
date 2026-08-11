"""Infostate tensor computation for Stratego (arXiv:2511.07312 Appendix).

488-channel infostate = 456 board channels + 32 last-move channels.

Key channels (MUST be correct): 0-42 (piece types, hidden, empty, moved, move
fractions) and 456-487 (last 32 moves). Complex tracking channels (43-455:
threats, evasions, protections, causes of death, starting locations) are
zeros — the CUDA port (T11) will fill these from GPU-resident game state.
"""

from __future__ import annotations

import numpy as np

from stratego.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    MAX_GAME_LENGTH,
    NUM_INFOSTATE_CHANNELS,
    TRAINING_NO_ATTACK_LIMIT,
)
from stratego.env.rules import StrategoState
from stratego.types import PIECE_TYPES, Action, Player, Square

_UNIFORM = 1.0 / len(PIECE_TYPES)


def _squares_to_mask(squares: set[Square] | None) -> np.ndarray:
    mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
    if squares:
        for sq in squares:
            if 0 <= sq.row < BOARD_ROWS and 0 <= sq.col < BOARD_COLS:
                mask[sq.row, sq.col] = True
    return mask


def compute_infostate(
    state: StrategoState,
    player: Player,
    *,
    move_history: list[Action] | None = None,
    moved_squares: set[Square] | None = None,
    revealed_squares: set[Square] | None = None,
) -> np.ndarray:
    """Compute (488, 10, 10) float32 infostate tensor for ``player``."""
    tensor = np.zeros(
        (NUM_INFOSTATE_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32
    )

    opp = player.opponent
    own = state.board_owner == int(player)
    opp_mask = state.board_owner == int(opp)
    empty = state.board_owner == -1

    revealed = _squares_to_mask(revealed_squares)
    moved = _squares_to_mask(moved_squares)

    # 0-11: own piece type one-hot
    for i, pt in enumerate(PIECE_TYPES):
        tensor[i] = np.where(own & (state.board_piece == int(pt)), 1.0, 0.0)

    # 12-23: opponent piece type probabilities (uniform for hidden, one-hot for revealed)
    opp_rev = opp_mask & revealed
    opp_hid = opp_mask & ~revealed
    for i, pt in enumerate(PIECE_TYPES):
        ch = 12 + i
        tensor[ch] = np.where(opp_rev & (state.board_piece == int(pt)), 1.0, 0.0)
        tensor[ch] += np.where(opp_hid, _UNIFORM, 0.0)

    # 24-35: mirror — opponent's view of player's pieces
    own_rev = own & revealed
    own_hid = own & ~revealed
    for i, pt in enumerate(PIECE_TYPES):
        ch = 24 + i
        tensor[ch] = np.where(own_rev & (state.board_piece == int(pt)), 1.0, 0.0)
        tensor[ch] += np.where(own_hid, _UNIFORM, 0.0)

    # 36: own hidden pieces
    tensor[36] = np.where(own_hid, 1.0, 0.0)
    # 37: opponent hidden pieces
    tensor[37] = np.where(opp_hid, 1.0, 0.0)
    # 38: empty squares
    tensor[38] = np.where(empty, 1.0, 0.0)
    # 39: own moved pieces
    tensor[39] = np.where(own & moved, 1.0, 0.0)
    # 40: opponent moved pieces
    tensor[40] = np.where(opp_mask & moved, 1.0, 0.0)
    # 41: fraction of max game moves exhausted
    tensor[41] = np.full(
        (BOARD_ROWS, BOARD_COLS),
        state.move_number / MAX_GAME_LENGTH,
        dtype=np.float32,
    )
    # 42: fraction of max moves between attacks exhausted
    limit = getattr(state, "_no_attack_limit", TRAINING_NO_ATTACK_LIMIT)
    frac = state.moves_since_last_attack / limit if limit > 0 else 0.0
    tensor[42] = np.full((BOARD_ROWS, BOARD_COLS), frac, dtype=np.float32)

    # 43-455: complex tracking channels — zeros (CUDA port T11 fills these)

    # 456-487: last 32 moves (+1 at dst, -1 at src, most recent = 456)
    history = move_history or []
    recent = history[-32:]
    for idx, action in enumerate(reversed(recent)):
        ch = 456 + idx
        tensor[ch, action.dst.row, action.dst.col] = 1.0
        tensor[ch, action.src.row, action.src.col] = -1.0

    return tensor


__all__ = ["compute_infostate"]
