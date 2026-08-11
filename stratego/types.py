"""Stratego domain types: PieceType, Player, Square, Action, GameOutcome.

These types are the shared vocabulary for the entire codebase — both the pure-Python
reference rules engine (stratego/env/rules.py) and the CUDA C++ simulator
(csrc/stratego_buffer.cpp) mirror these definitions.

Conventions:
- Board is 10x10, 0-indexed by (row, col).
- Player 0 = Red (starts at the top, rows 0-3), Player 1 = Blue (rows 6-9).
- Piece ranks: 1=Spy, 2=Scout, ..., 10=Marshal. Flag and Bomb have no rank.
- Pieces are stored as positive integers (PieceType) regardless of owner; ownership
  is tracked separately.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class PieceType(IntEnum):
    """The 12 Stratego piece types plus a NONE sentinel for empty squares.

    Integer values are chosen so that the combat rank is recoverable from the value
    for the ranked pieces (Spy=1..Marshal=10). Flag and Bomb use values 11 and 12
    respectively, matching the paper's channel ordering (channels 0-11 use this order).
    """

    SPY = 1
    SCOUT = 2
    MINER = 3
    SERGEANT = 4
    LIEUTENANT = 5
    CAPTAIN = 6
    MAJOR = 7
    COLONEL = 8
    GENERAL = 9
    MARSHAL = 10
    FLAG = 11
    BOMB = 12
    NONE = 0  # Empty square sentinel

    @property
    def rank(self) -> int:
        """Combat rank for the piece (1-10 for ranked pieces; 0 for Flag/Bomb/NONE).

        Higher rank defeats lower rank in standard combat. Flag and Bomb do not
        participate in rank-based combat resolution (Flag loses to any attacker,
        Bomb defeats any attacker except Miner).
        """
        if self in (PieceType.FLAG, PieceType.BOMB, PieceType.NONE):
            return 0
        return int(self)

    @property
    def can_move(self) -> bool:
        """Whether this piece type can ever move. Flag and Bomb are immovable."""
        return self not in (PieceType.FLAG, PieceType.BOMB, PieceType.NONE)

    @property
    def is_scout(self) -> bool:
        """Scouts can move any number of squares in a cardinal direction."""
        return self == PieceType.SCOUT


# Ordered tuple of all 12 piece types (matches paper channel indexing 0-11)
PIECE_TYPES: tuple[PieceType, ...] = (
    PieceType.SPY,
    PieceType.SCOUT,
    PieceType.MINER,
    PieceType.SERGEANT,
    PieceType.LIEUTENANT,
    PieceType.CAPTAIN,
    PieceType.MAJOR,
    PieceType.COLONEL,
    PieceType.GENERAL,
    PieceType.MARSHAL,
    PieceType.FLAG,
    PieceType.BOMB,
)

NUM_PIECE_TYPES = len(PIECE_TYPES)  # 12


def piece_count(pt: PieceType) -> int:
    """Number of pieces of this type in a standard 40-piece Stratego setup."""
    counts = {
        PieceType.SPY: 1,
        PieceType.SCOUT: 8,
        PieceType.MINER: 5,
        PieceType.SERGEANT: 4,
        PieceType.LIEUTENANT: 4,
        PieceType.CAPTAIN: 4,
        PieceType.MAJOR: 3,
        PieceType.COLONEL: 2,
        PieceType.GENERAL: 1,
        PieceType.MARSHAL: 1,
        PieceType.FLAG: 1,
        PieceType.BOMB: 6,
    }
    return counts.get(pt, 0)


class Player(IntEnum):
    """Stratego player. Red starts the game (ISF rule, chapter 5.1)."""

    RED = 0
    BLUE = 1

    @property
    def opponent(self) -> Player:
        """Return the other player."""
        return Player.RED if self == Player.BLUE else Player.BLUE

    @property
    def setup_rows(self) -> tuple[int, int]:
        """Inclusive (start_row, end_row) for this player's setup zone."""
        return (0, 3) if self == Player.RED else (6, 9)

    @property
    def forward_direction(self) -> int:
        """Row delta for 'forward' movement (toward the opponent's side).

        Red moves downward (+1), Blue moves upward (-1).
        """
        return 1 if self == Player.RED else -1


class Square(NamedTuple):
    """A board square identified by (row, col) with 0-indexed coordinates.

    The Stratego board is 10x10. Lakes occupy (4,2),(4,3),(5,2),(5,3) (left lake)
    and (4,6),(4,7),(5,6),(5,7) (right lake). See constants.LAKES for the canonical list.
    """

    row: int
    col: int

    @property
    def idx(self) -> int:
        """Row-major flat index in [0, 99]."""
        return self.row * 10 + self.col

    @property
    def is_valid(self) -> bool:
        """True iff this square is within the 10x10 board."""
        return 0 <= self.row < 10 and 0 <= self.col < 10

    @property
    def is_lake(self) -> bool:
        """True iff this square is a lake (impassable)."""
        from stratego.constants import LAKE_SET

        return self in LAKE_SET

    def neighbors_cardinal(self) -> list[Square]:
        """The up to 4 cardinal-adjacent squares (up/down/left/right), excluding off-board."""
        candidates = [
            Square(self.row - 1, self.col),
            Square(self.row + 1, self.col),
            Square(self.row, self.col - 1),
            Square(self.row, self.col + 1),
        ]
        return [s for s in candidates if s.is_valid]

    @classmethod
    def from_idx(cls, idx: int) -> Square:
        """Inverse of .idx — build a Square from a row-major flat index."""
        return cls(row=idx // 10, col=idx % 10)


class Action(NamedTuple):
    """A Stratego move: piece moves from `src` to `dst`.

    If `dst` is occupied by an opponent piece, the move is an attack (battle).
    Scouts can move multiple squares in one action; all other movable pieces
    move exactly one cardinal square per action.
    """

    src: Square
    dst: Square

    @property
    def is_attack(self) -> bool:
        """Whether this action is an attack (dst != src and combat will occur).

        Note: whether the destination is actually occupied must be checked against
        the current board state. This property is a convenience for filtering in
        contexts where occupancy is known.
        """
        return self.src != self.dst

    def path_scout(self) -> list[Square]:
        """For a Scout long move, the list of squares traversed (exclusive of src, inclusive of dst).

        Assumes the move is a straight cardinal-line move (caller must validate).
        Returns an empty list for non-straight or unit-distance moves.
        """
        dr = self.dst.row - self.src.row
        dc = self.dst.col - self.src.col
        # Determine direction sign and length
        if dr == 0:
            steps = abs(dc)
            step_col = 1 if dc > 0 else -1
            return [Square(self.src.row, self.src.col + step_col * (i + 1)) for i in range(steps)]
        if dc == 0:
            steps = abs(dr)
            step_row = 1 if dr > 0 else -1
            return [Square(self.src.row + step_row * (i + 1), self.src.col) for i in range(steps)]
        return []

    def __str__(self) -> str:
        """Human-readable algebraic-like notation: 'src->dst' using (row,col)."""
        return f"({self.src.row},{self.src.col})->({self.dst.row},{self.dst.col})"


class GameOutcome(IntEnum):
    """Possible outcomes of a Stratego game."""

    ONGOING = -1
    RED_WIN = 0
    BLUE_WIN = 1
    DRAW = 2


def opponent(player: Player) -> Player:
    """Convenience function: return the other player."""
    return player.opponent


__all__ = [
    "PIECE_TYPES",
    "NUM_PIECE_TYPES",
    "Action",
    "GameOutcome",
    "PieceType",
    "Player",
    "Square",
    "opponent",
    "piece_count",
]
