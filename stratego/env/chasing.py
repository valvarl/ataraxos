"""ISF continuous-chasing rule tracker (chapter 11).

A chase is per-CHASER player: the same player non-stop threatening evading
opponent pieces. All board positions during a player's active chase are
tracked in a single set. The chaser may not make a chasing move that leads
to a position already seen during this chase. Exception: chasing back to
the directly-preceding-turn square is always allowed.
"""

from __future__ import annotations

import numpy as np

from stratego.types import Action, Player, Square


class ZobristHasher:
    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._table: np.ndarray = rng.integers(
            low=0, high=2**63, size=(10, 10, 2, 13), dtype=np.int64
        )

    def hash_board(self, board_owner: np.ndarray, board_piece: np.ndarray) -> int:
        h: np.int64 = np.int64(0)
        mask = board_owner >= 0
        if mask.any():
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols, strict=False):
                h ^= self._table[r, c, board_owner[r, c], board_piece[r, c]]
        return int(h)


class ChasingTracker:
    def __init__(self) -> None:
        self._hasher = ZobristHasher()
        self._last_threats: dict[Player, set[Square]] = {
            Player.RED: set(),
            Player.BLUE: set(),
        }
        self._last_was_evasion: dict[Player, bool] = {
            Player.RED: False,
            Player.BLUE: False,
        }
        self._chase_positions: dict[Player, set[int]] = {
            Player.RED: set(),
            Player.BLUE: set(),
        }
        self._last_chaser_src: dict[Player, Square | None] = {
            Player.RED: None,
            Player.BLUE: None,
        }

    def record_move(
        self,
        player: Player,
        action: Action,
        board_owner_before: np.ndarray,
        board_piece_before: np.ndarray,
        board_owner_after: np.ndarray,
        board_piece_after: np.ndarray,
    ) -> None:
        opp = Player(int(player) ^ 1)
        threatened = self._find_threatened_pieces(
            player, action.dst, board_owner_after, board_piece_after
        )
        was_evasion = action.src in self._last_threats.get(opp, set())

        if threatened and self._last_was_evasion.get(opp, False):
            board_hash = self._hasher.hash_board(board_owner_after, board_piece_after)
            self._chase_positions[player].add(board_hash)
            self._last_chaser_src[player] = action.src
        elif not threatened:
            self._chase_positions[player].clear()
            self._last_chaser_src[player] = None

        self._last_threats[player] = threatened
        self._last_was_evasion[player] = was_evasion

    def is_chasing_violation(
        self,
        player: Player,
        action: Action,
        board_owner: np.ndarray,
        board_piece: np.ndarray,
    ) -> bool:
        sim_owner = board_owner.copy()
        sim_piece = board_piece.copy()
        sim_piece[action.dst.row, action.dst.col] = sim_piece[action.src.row, action.src.col]
        sim_owner[action.dst.row, action.dst.col] = int(player)
        sim_piece[action.src.row, action.src.col] = 0
        sim_owner[action.src.row, action.src.col] = -1

        threatened = self._find_threatened_pieces(player, action.dst, sim_owner, sim_piece)
        if not threatened:
            return False

        opp = Player(int(player) ^ 1)
        if not self._last_was_evasion.get(opp, False):
            return False

        result_hash = self._hasher.hash_board(sim_owner, sim_piece)
        if result_hash not in self._chase_positions.get(player, set()):
            return False

        last_src = self._last_chaser_src.get(player)
        if last_src is not None and action.dst == last_src:
            return False
        return True

    def _find_threatened_pieces(
        self, player: Player, moved_to: Square,
        board_owner: np.ndarray, board_piece: np.ndarray,
    ) -> set[Square]:
        opp = Player(int(player) ^ 1)
        threatened: set[Square] = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = moved_to.row + dr, moved_to.col + dc
            if 0 <= r < 10 and 0 <= c < 10 and board_owner[r, c] == int(opp):
                threatened.add(Square(r, c))
        return threatened

    def clone(self) -> ChasingTracker:
        new = ChasingTracker()
        new._hasher = self._hasher
        new._last_threats = {p: set(s) for p, s in self._last_threats.items()}
        new._last_was_evasion = dict(self._last_was_evasion)
        new._chase_positions = {p: set(s) for p, s in self._chase_positions.items()}
        new._last_chaser_src = dict(self._last_chaser_src)
        return new

    def reset(self) -> None:
        self._last_threats = {Player.RED: set(), Player.BLUE: set()}
        self._last_was_evasion = {Player.RED: False, Player.BLUE: False}
        self._chase_positions = {Player.RED: set(), Player.BLUE: set()}
        self._last_chaser_src = {Player.RED: None, Player.BLUE: None}


__all__ = ["ChasingTracker", "ZobristHasher"]
