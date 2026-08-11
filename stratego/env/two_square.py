"""ISF two-square rule tracker (chapter 10).

Tracks per-piece oscillation between square pairs. A piece may not cross the
same square boundary more than TWO_SQUARE_LIMIT (3) consecutive times.
Attacks (strikes) are exempt and reset the counter.
"""

from __future__ import annotations

import copy

from stratego.constants import TWO_SQUARE_LIMIT
from stratego.types import Action, Player, Square


class TwoSquareTracker:
    """Stateful tracker for the ISF two-square rule.

    Per-piece tracking: each piece (identified by player + current square)
    maintains its own set of (square_pair → consecutive_count) entries.
    Attacks reset the counter. Other pieces moving does not affect tracking.
    """

    def __init__(self) -> None:
        self._counters: dict[tuple[Player, Square], dict[frozenset[Square], int]] = {}

    def record_move(
        self,
        player: Player,
        piece_square: Square,
        action: Action,
        is_scout: bool,
        path: list[Square],
        is_attack: bool,
    ) -> None:
        old_key = (player, piece_square)
        old_counters = self._counters.pop(old_key, {})

        if is_attack:
            new_counters: dict[frozenset[Square], int] = {}
        else:
            move_pairs = self._compute_pairs(action.src, action.dst, is_scout, path)
            new_counters = {}
            for pair in move_pairs:
                new_counters[pair] = old_counters.get(pair, 0) + 1

        new_key = (player, action.dst)
        if new_counters:
            self._counters[new_key] = new_counters
        elif new_key in self._counters:
            del self._counters[new_key]

    def is_violation(
        self,
        player: Player,
        piece_square: Square,
        action: Action,
        is_scout: bool,
        path: list[Square],
        is_attack: bool,
    ) -> bool:
        if is_attack:
            return False

        counters = self._counters.get((player, piece_square), {})
        move_pairs = self._compute_pairs(action.src, action.dst, is_scout, path)

        return any(
            counters.get(pair, 0) + 1 > TWO_SQUARE_LIMIT for pair in move_pairs
        )

    def clone(self) -> TwoSquareTracker:
        new = TwoSquareTracker()
        new._counters = copy.deepcopy(self._counters)
        return new

    def reset(self) -> None:
        self._counters.clear()

    @staticmethod
    def _compute_pairs(
        src: Square, dst: Square, is_scout: bool, path: list[Square]
    ) -> frozenset[frozenset[Square]]:
        if is_scout and len(path) > 1:
            intermediates = set(path[:-1])
            start_positions = {src} | intermediates
            end_positions = {dst} | intermediates
            pairs: set[frozenset[Square]] = set()
            for s in start_positions:
                for e in end_positions:
                    if s != e:
                        pairs.add(frozenset((s, e)))
            return frozenset(pairs)
        return frozenset({frozenset((src, dst))})
