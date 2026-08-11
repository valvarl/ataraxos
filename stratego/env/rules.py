"""Pure-Python reference Stratego rules engine.

This is the GROUND TRUTH implementation that the CUDA C++ simulator
will be validated against. Correctness > performance.

Board representation uses two numpy int8 arrays:
- board_owner: -1 = empty, 0 = Red, 1 = Blue
- board_piece: PieceType value (0 = NONE for empty squares)
"""

from __future__ import annotations

import numpy as np

from stratego.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    CARDINAL_DIRECTIONS,
    MAX_GAME_LENGTH,
    PIECE_COUNTS,
    TOTAL_PIECES,
    TRAINING_NO_ATTACK_LIMIT,
)
from stratego.types import Action, GameOutcome, PieceType, Player, Square


class StrategoState:
    """Pure-Python reference Stratego game state.

    This is the GROUND TRUTH implementation that the CUDA C++ simulator
    will be validated against. Correctness > performance.
    """

    def __init__(self, no_attack_limit: int = TRAINING_NO_ATTACK_LIMIT) -> None:
        self.board_owner: np.ndarray = np.full(
            (BOARD_ROWS, BOARD_COLS), -1, dtype=np.int8
        )
        self.board_piece: np.ndarray = np.zeros(
            (BOARD_ROWS, BOARD_COLS), dtype=np.int8
        )
        self._current_player: Player = Player.RED
        self._move_number: int = 0
        self._moves_since_last_attack: int = 0
        self._no_attack_limit: int = no_attack_limit
        self._outcome: GameOutcome = GameOutcome.ONGOING
        self._red_captured: list[PieceType] = []
        self._blue_captured: list[PieceType] = []
        self._red_setup_done: bool = False
        self._blue_setup_done: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> Player:
        """The player whose turn it is."""
        return self._current_player

    @property
    def move_number(self) -> int:
        """Total number of half-moves (simulator steps) played."""
        return self._move_number

    @property
    def moves_since_last_attack(self) -> int:
        """Consecutive moves without a combat (for draw detection)."""
        return self._moves_since_last_attack

    @property
    def outcome(self) -> GameOutcome:
        """Current game outcome."""
        return self._outcome

    @property
    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        return self._outcome != GameOutcome.ONGOING

    @property
    def red_captured(self) -> list[PieceType]:
        """Pieces captured by Red (Blue pieces that Red has killed)."""
        return list(self._red_captured)

    @property
    def blue_captured(self) -> list[PieceType]:
        """Pieces captured by Blue (Red pieces that Blue has killed)."""
        return list(self._blue_captured)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def apply_setup(
        self, player: Player, placement: list[tuple[Square, PieceType]]
    ) -> None:
        """Place a player's 40 pieces on the board during setup phase.

        Validates:
        - Exactly 40 pieces
        - Correct piece counts per type
        - All pieces within the player's setup rows
        - No pieces on lake squares
        """
        if len(placement) != TOTAL_PIECES:
            raise ValueError(
                f"Setup requires exactly {TOTAL_PIECES} pieces, got {len(placement)}"
            )

        # Check piece counts
        counts: dict[PieceType, int] = {}
        for _sq, pt in placement:
            counts[pt] = counts.get(pt, 0) + 1

        for pt, expected in PIECE_COUNTS.items():
            actual = counts.get(pt, 0)
            if actual != expected:
                raise ValueError(
                    f"Piece count mismatch for {pt.name}: "
                    f"expected {expected}, got {actual}"
                )

        start_row, end_row = player.setup_rows
        for sq, _pt in placement:
            if sq.is_lake:
                raise ValueError(
                    f"Cannot place piece on lake square ({sq.row},{sq.col})"
                )
            if sq.row < start_row or sq.row > end_row:
                raise ValueError(
                    f"Piece at row {sq.row} outside {player.name} setup rows "
                    f"({start_row}-{end_row})"
                )
            if not sq.is_valid:
                raise ValueError(f"Invalid square ({sq.row},{sq.col})")

        # Apply placement
        for sq, pt in placement:
            self.board_owner[sq.row, sq.col] = int(player)
            self.board_piece[sq.row, sq.col] = int(pt)

        if player == Player.RED:
            self._red_setup_done = True
        else:
            self._blue_setup_done = True

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def legal_actions(self, player: Player | None = None) -> list[Action]:
        """Return all legal actions for the given player (default: current_player)."""
        if player is None:
            player = self._current_player

        actions: list[Action] = []
        player_int = int(player)

        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.board_owner[r, c] != player_int:
                    continue
                pt_val = self.board_piece[r, c]
                if pt_val == int(PieceType.NONE):
                    continue

                pt = PieceType(pt_val)
                if not pt.can_move:
                    continue  # Flag and Bomb cannot move

                src = Square(r, c)

                if pt.is_scout:
                    # Scout: can move N squares in one cardinal direction
                    actions.extend(self._scout_moves(src, player))
                else:
                    # Normal piece: move 1 square cardinal
                    actions.extend(self._single_moves(src, player))

        return actions

    def _single_moves(self, src: Square, player: Player) -> list[Action]:
        """Generate 1-square cardinal moves for a non-Scout piece."""
        actions: list[Action] = []
        player_int = int(player)

        for dr, dc in CARDINAL_DIRECTIONS:
            nr, nc = src.row + dr, src.col + dc
            dst = Square(nr, nc)
            if not dst.is_valid:
                continue
            if dst.is_lake:
                continue
            owner = self.board_owner[nr, nc]
            if owner == player_int:
                continue  # can't move onto own piece
            # Empty or enemy: legal move (enemy = attack)
            actions.append(Action(src, dst))
        return actions

    def _scout_moves(self, src: Square, player: Player) -> list[Action]:
        """Generate all legal Scout moves (1..N squares in one cardinal direction)."""
        actions: list[Action] = []
        player_int = int(player)

        for dr, dc in CARDINAL_DIRECTIONS:
            # Walk in this direction until blocked
            nr, nc = src.row + dr, src.col + dc
            while 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                dst = Square(nr, nc)
                if dst.is_lake:
                    break  # lake blocks scout path
                owner = self.board_owner[nr, nc]
                if owner == player_int:
                    break  # own piece blocks
                if owner == -1:
                    # Empty square: can move here, continue further
                    actions.append(Action(src, dst))
                else:
                    # Enemy piece: can attack, but cannot continue past
                    actions.append(Action(src, dst))
                    break
                nr += dr
                nc += dc

        return actions

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> None:
        """Apply a move/attack action for the current player.

        Updates board state, resolves combat if attacking, advances turn.
        """
        src = action.src
        dst = action.dst

        # Validate source belongs to current player
        src_owner = self.board_owner[src.row, src.col]
        if src_owner != int(self._current_player):
            raise ValueError(
                f"Source square ({src.row},{src.col}) does not belong to "
                f"{self._current_player.name}"
            )

        src_piece = PieceType(int(self.board_piece[src.row, src.col]))
        dst_owner = self.board_owner[dst.row, dst.col]

        is_attack = dst_owner != -1 and dst_owner != int(self._current_player)

        if is_attack:
            dst_piece = PieceType(int(self.board_piece[dst.row, dst.col]))
            self._resolve_combat(src, src_piece, dst, dst_piece)
            self._moves_since_last_attack = 0
        else:
            # Simple move to empty square
            self.board_owner[dst.row, dst.col] = src_owner
            self.board_piece[dst.row, dst.col] = int(src_piece)
            self.board_owner[src.row, src.col] = -1
            self.board_piece[src.row, src.col] = int(PieceType.NONE)
            self._moves_since_last_attack += 1

        # Advance turn
        self._move_number += 1
        self._current_player = self._current_player.opponent

        # Check terminal after move
        self._check_terminal_internal()

    def _resolve_combat(
        self,
        src: Square,
        attacker: PieceType,
        dst: Square,
        defender: PieceType,
    ) -> None:
        """Resolve combat between attacker (at src) and defender (at dst).

        Rules (ISF 5.3-5.4):
        - Flag: any attacker captures it → attacker wins game
        - Bomb vs Miner: Miner wins (Bomb removed, Miner moves to dst)
        - Bomb vs non-Miner: Bomb wins (attacker removed, Bomb stays)
        - Spy vs Marshal (Spy attacking): Spy wins
        - Otherwise: higher rank wins; equal rank → both die
        """
        attacker_player = self.board_owner[src.row, src.col]

        # Flag capture: attacker wins
        if defender == PieceType.FLAG:
            # Attacker moves to flag's square
            self.board_owner[dst.row, dst.col] = attacker_player
            self.board_piece[dst.row, dst.col] = int(attacker)
            self.board_owner[src.row, src.col] = -1
            self.board_piece[src.row, src.col] = int(PieceType.NONE)
            # Record capture
            if attacker_player == int(Player.RED):
                self._red_captured.append(PieceType.FLAG)
            else:
                self._blue_captured.append(PieceType.FLAG)
            # Set game outcome
            if attacker_player == int(Player.RED):
                self._outcome = GameOutcome.RED_WIN
            else:
                self._outcome = GameOutcome.BLUE_WIN
            return

        # Bomb defender
        if defender == PieceType.BOMB:
            if attacker == PieceType.MINER:
                # Miner defuses bomb: attacker wins
                self.board_owner[dst.row, dst.col] = attacker_player
                self.board_piece[dst.row, dst.col] = int(attacker)
                self.board_owner[src.row, src.col] = -1
                self.board_piece[src.row, src.col] = int(PieceType.NONE)
                if attacker_player == int(Player.RED):
                    self._red_captured.append(PieceType.BOMB)
                else:
                    self._blue_captured.append(PieceType.BOMB)
            else:
                # Bomb kills attacker
                self.board_owner[src.row, src.col] = -1
                self.board_piece[src.row, src.col] = int(PieceType.NONE)
                defender_player = self.board_owner[dst.row, dst.col]
                if defender_player == int(Player.RED):
                    self._blue_captured.append(attacker)
                else:
                    self._red_captured.append(attacker)
            return

        # Spy attacking Marshal: Spy wins
        if attacker == PieceType.SPY and defender == PieceType.MARSHAL:
            self.board_owner[dst.row, dst.col] = attacker_player
            self.board_piece[dst.row, dst.col] = int(attacker)
            self.board_owner[src.row, src.col] = -1
            self.board_piece[src.row, src.col] = int(PieceType.NONE)
            if attacker_player == int(Player.RED):
                self._red_captured.append(defender)
            else:
                self._blue_captured.append(defender)
            return

        # Standard rank-based combat
        attacker_rank = attacker.rank
        defender_rank = defender.rank

        defender_player = self.board_owner[dst.row, dst.col]

        if attacker_rank > defender_rank:
            # Attacker wins
            self.board_owner[dst.row, dst.col] = attacker_player
            self.board_piece[dst.row, dst.col] = int(attacker)
            self.board_owner[src.row, src.col] = -1
            self.board_piece[src.row, src.col] = int(PieceType.NONE)
            if attacker_player == int(Player.RED):
                self._red_captured.append(defender)
            else:
                self._blue_captured.append(defender)
        elif attacker_rank < defender_rank:
            # Defender wins
            self.board_owner[src.row, src.col] = -1
            self.board_piece[src.row, src.col] = int(PieceType.NONE)
            if defender_player == int(Player.RED):
                self._blue_captured.append(attacker)
            else:
                self._red_captured.append(attacker)
        else:
            # Equal rank: both die
            self.board_owner[src.row, src.col] = -1
            self.board_piece[src.row, src.col] = int(PieceType.NONE)
            self.board_owner[dst.row, dst.col] = -1
            self.board_piece[dst.row, dst.col] = int(PieceType.NONE)
            # Both captured
            if attacker_player == int(Player.RED):
                self._red_captured.append(defender)
                self._blue_captured.append(attacker)
            else:
                self._blue_captured.append(defender)
                self._red_captured.append(attacker)

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def check_terminal(self) -> GameOutcome:
        """Check and return the current game outcome."""
        self._check_terminal_internal()
        return self._outcome

    def _check_terminal_internal(self) -> None:
        """Internal terminal check — updates _outcome if game is over."""
        if self._outcome != GameOutcome.ONGOING:
            return  # already terminal

        # Max game length draw
        if self._move_number >= MAX_GAME_LENGTH:
            self._outcome = GameOutcome.DRAW
            return

        # No-attack draw
        if self._moves_since_last_attack >= self._no_attack_limit:
            self._outcome = GameOutcome.DRAW
            return

        # Current player has no legal moves → current player loses
        current_actions = self.legal_actions(self._current_player)
        if len(current_actions) == 0:
            # Current player loses
            if self._current_player == Player.RED:
                self._outcome = GameOutcome.BLUE_WIN
            else:
                self._outcome = GameOutcome.RED_WIN

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> StrategoState:
        """Return a deep copy of this state."""
        new = StrategoState.__new__(StrategoState)
        new.board_owner = self.board_owner.copy()
        new.board_piece = self.board_piece.copy()
        new._current_player = self._current_player
        new._move_number = self._move_number
        new._moves_since_last_attack = self._moves_since_last_attack
        new._no_attack_limit = self._no_attack_limit
        new._outcome = self._outcome
        new._red_captured = list(self._red_captured)
        new._blue_captured = list(self._blue_captured)
        new._red_setup_done = self._red_setup_done
        new._blue_setup_done = self._blue_setup_done
        return new

    # ------------------------------------------------------------------
    # Random legal action
    # ------------------------------------------------------------------

    def random_legal_action(
        self, rng: np.random.Generator | None = None
    ) -> Action:
        """Return a uniformly random legal action."""
        actions = self.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.integers(0, len(actions)))
        return actions[idx]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def board_str(self) -> str:
        """Human-readable board string for debugging."""
        piece_chars: dict[int, str] = {
            int(PieceType.NONE): ".",
            int(PieceType.SPY): "S",
            int(PieceType.SCOUT): "s",
            int(PieceType.MINER): "m",
            int(PieceType.SERGEANT): "4",
            int(PieceType.LIEUTENANT): "5",
            int(PieceType.CAPTAIN): "6",
            int(PieceType.MAJOR): "7",
            int(PieceType.COLONEL): "8",
            int(PieceType.GENERAL): "G",
            int(PieceType.MARSHAL): "M",
            int(PieceType.FLAG): "F",
            int(PieceType.BOMB): "B",
        }

        lines: list[str] = []
        lines.append(f"Move {self._move_number} | {self._current_player.name}'s turn")
        lines.append(f"No-attack streak: {self._moves_since_last_attack}")
        lines.append("  " + " ".join(f"{c}" for c in range(10)))
        for r in range(BOARD_ROWS):
            row_str = f"{r} "
            for c in range(BOARD_COLS):
                sq = Square(r, c)
                if sq.is_lake:
                    row_str += "~ "
                elif self.board_owner[r, c] == -1:
                    row_str += ". "
                else:
                    ch = piece_chars.get(int(self.board_piece[r, c]), "?")
                    owner = self.board_owner[r, c]
                    if owner == int(Player.RED):
                        row_str += ch.upper() + " "
                    else:
                        row_str += ch.lower() + " "
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"StrategoState(move={self._move_number}, "
            f"player={self._current_player.name}, "
            f"outcome={self._outcome.name})"
        )
