"""Tests for stratego.types and stratego.constants.

These tests pin the domain vocabulary that every downstream module depends on.
They verify:
- Piece counts sum to 40 (standard Stratego setup)
- Lake positions match the ISF standard (8 squares, two 2x2 blocks)
- Player/flip/opponent semantics are correct
- Ranks order Spy < Scout < ... < Marshal
- Constants from the paper (MAX_GAME_LENGTH, infostate channel counts, etc.) are exact
"""

from __future__ import annotations

from stratego import constants as C  # noqa: N812
from stratego.types import (
    NUM_PIECE_TYPES,
    PIECE_TYPES,
    Action,
    GameOutcome,
    PieceType,
    Player,
    Square,
    opponent,
    piece_count,
)

# ---------------------------------------------------------------------------
# PieceType enum
# ---------------------------------------------------------------------------


class TestPieceType:
    def test_enum_values(self) -> None:
        assert int(PieceType.NONE) == 0
        assert int(PieceType.SPY) == 1
        assert int(PieceType.SCOUT) == 2
        assert int(PieceType.MARSHAL) == 10
        assert int(PieceType.FLAG) == 11
        assert int(PieceType.BOMB) == 12

    def test_piece_types_tuple_order(self) -> None:
        # Paper channels 0-11 follow this exact ordering.
        assert PIECE_TYPES[0] == PieceType.SPY
        assert PIECE_TYPES[9] == PieceType.MARSHAL
        assert PIECE_TYPES[10] == PieceType.FLAG
        assert PIECE_TYPES[11] == PieceType.BOMB

    def test_num_piece_types(self) -> None:
        assert NUM_PIECE_TYPES == 12
        assert len(PIECE_TYPES) == 12

    def test_rank(self) -> None:
        assert PieceType.SPY.rank == 1
        assert PieceType.SCOUT.rank == 2
        assert PieceType.MARSHAL.rank == 10
        assert PieceType.FLAG.rank == 0
        assert PieceType.BOMB.rank == 0
        assert PieceType.NONE.rank == 0

    def test_can_move(self) -> None:
        assert PieceType.SPY.can_move
        assert PieceType.SCOUT.can_move
        assert PieceType.MARSHAL.can_move
        assert not PieceType.FLAG.can_move
        assert not PieceType.BOMB.can_move
        assert not PieceType.NONE.can_move

    def test_is_scout(self) -> None:
        assert PieceType.SCOUT.is_scout
        assert not PieceType.SPY.is_scout
        assert not PieceType.MARSHAL.is_scout


# ---------------------------------------------------------------------------
# Piece counts
# ---------------------------------------------------------------------------


class TestPieceCounts:
    def test_spy_count(self) -> None:
        assert piece_count(PieceType.SPY) == 1

    def test_scout_count(self) -> None:
        assert piece_count(PieceType.SCOUT) == 8

    def test_miner_count(self) -> None:
        assert piece_count(PieceType.MINER) == 5

    def test_bomb_count(self) -> None:
        assert piece_count(PieceType.BOMB) == 6

    def test_flag_count(self) -> None:
        assert piece_count(PieceType.FLAG) == 1

    def test_marshal_count(self) -> None:
        assert piece_count(PieceType.MARSHAL) == 1

    def test_general_count(self) -> None:
        assert piece_count(PieceType.GENERAL) == 1

    def test_total_counts_sum_to_40(self) -> None:
        total = sum(piece_count(pt) for pt in PIECE_TYPES)
        assert total == C.TOTAL_PIECES == 40

    def test_piece_counts_dict(self) -> None:
        # Verify the constants module agrees with the type-level helper.
        for pt in PIECE_TYPES:
            assert C.PIECE_COUNTS[pt] == piece_count(pt)

    def test_each_piece_count_positive(self) -> None:
        for pt in PIECE_TYPES:
            assert piece_count(pt) > 0, f"{pt.name} has zero count"


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


class TestPlayer:
    def test_red_starts(self) -> None:
        # ISF rule 5.1: "Red begins."
        assert int(Player.RED) == 0

    def test_opponent(self) -> None:
        assert Player.RED.opponent == Player.BLUE
        assert Player.BLUE.opponent == Player.RED

    def test_opponent_function(self) -> None:
        assert opponent(Player.RED) == Player.BLUE
        assert opponent(Player.BLUE) == Player.RED

    def test_setup_rows_red(self) -> None:
        start, end = Player.RED.setup_rows
        assert (start, end) == (0, 3)

    def test_setup_rows_blue(self) -> None:
        start, end = Player.BLUE.setup_rows
        assert (start, end) == (6, 9)

    def test_forward_direction(self) -> None:
        assert Player.RED.forward_direction == 1
        assert Player.BLUE.forward_direction == -1


# ---------------------------------------------------------------------------
# Square
# ---------------------------------------------------------------------------


class TestSquare:
    def test_idx_row_major(self) -> None:
        assert Square(0, 0).idx == 0
        assert Square(0, 9).idx == 9
        assert Square(1, 0).idx == 10
        assert Square(9, 9).idx == 99

    def test_from_idx_roundtrip(self) -> None:
        for idx in (0, 1, 10, 55, 99):
            assert Square.from_idx(idx).idx == idx

    def test_is_valid_in_bounds(self) -> None:
        assert Square(0, 0).is_valid
        assert Square(9, 9).is_valid
        assert Square(5, 5).is_valid

    def test_is_valid_out_of_bounds(self) -> None:
        assert not Square(-1, 0).is_valid
        assert not Square(0, -1).is_valid
        assert not Square(10, 0).is_valid
        assert not Square(0, 10).is_valid

    def test_is_lake(self) -> None:
        # Left lake
        assert Square(4, 2).is_lake
        assert Square(4, 3).is_lake
        assert Square(5, 2).is_lake
        assert Square(5, 3).is_lake
        # Right lake
        assert Square(4, 6).is_lake
        assert Square(4, 7).is_lake
        assert Square(5, 6).is_lake
        assert Square(5, 7).is_lake

    def test_is_lake_false_for_non_lake(self) -> None:
        assert not Square(0, 0).is_lake
        assert not Square(5, 5).is_lake
        assert not Square(9, 9).is_lake

    def test_neighbors_cardinal_corner(self) -> None:
        # Top-left corner has 2 neighbors (down, right)
        nbrs = Square(0, 0).neighbors_cardinal()
        assert len(nbrs) == 2
        assert Square(1, 0) in nbrs
        assert Square(0, 1) in nbrs

    def test_neighbors_cardinal_center(self) -> None:
        nbrs = Square(5, 5).neighbors_cardinal()
        assert len(nbrs) == 4
        assert Square(4, 5) in nbrs
        assert Square(6, 5) in nbrs
        assert Square(5, 4) in nbrs
        assert Square(5, 6) in nbrs


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TestAction:
    def test_is_attack_true_when_src_ne_dst(self) -> None:
        a = Action(Square(0, 0), Square(1, 0))
        assert a.is_attack

    def test_path_scout_horizontal(self) -> None:
        # Scout moves from (0,0) to (0,5): path is (0,1),(0,2),(0,3),(0,4),(0,5)
        a = Action(Square(0, 0), Square(0, 5))
        path = a.path_scout()
        assert path == [Square(0, 1), Square(0, 2), Square(0, 3), Square(0, 4), Square(0, 5)]

    def test_path_scout_vertical(self) -> None:
        a = Action(Square(0, 0), Square(4, 0))
        path = a.path_scout()
        assert path == [Square(1, 0), Square(2, 0), Square(3, 0), Square(4, 0)]

    def test_path_scout_negative_direction(self) -> None:
        a = Action(Square(5, 5), Square(5, 0))
        path = a.path_scout()
        assert path == [Square(5, 4), Square(5, 3), Square(5, 2), Square(5, 1), Square(5, 0)]

    def test_path_scout_unit_distance(self) -> None:
        a = Action(Square(0, 0), Square(0, 1))
        path = a.path_scout()
        assert path == [Square(0, 1)]

    def test_path_scout_diagonal_empty(self) -> None:
        a = Action(Square(0, 0), Square(1, 1))
        assert a.path_scout() == []


# ---------------------------------------------------------------------------
# GameOutcome
# ---------------------------------------------------------------------------


class TestGameOutcome:
    def test_values(self) -> None:
        assert int(GameOutcome.ONGOING) == -1
        assert int(GameOutcome.RED_WIN) == 0
        assert int(GameOutcome.BLUE_WIN) == 1
        assert int(GameOutcome.DRAW) == 2


# ---------------------------------------------------------------------------
# Board constants
# ---------------------------------------------------------------------------


class TestBoardConstants:
    def test_board_dims(self) -> None:
        assert C.BOARD_ROWS == 10
        assert C.BOARD_COLS == 10
        assert C.NUM_SQUARES == 100

    def test_lake_count(self) -> None:
        assert C.NUM_LAKE_SQUARES == 8
        assert len(C.LAKES) == 8

    def test_occupiable_count(self) -> None:
        assert C.NUM_OCCUPIABLE == 92

    def test_lakes_form_two_2x2_blocks(self) -> None:
        # Left lake: rows 4-5, cols 2-3
        left = {Square(4, 2), Square(4, 3), Square(5, 2), Square(5, 3)}
        # Right lake: rows 4-5, cols 6-7
        right = {Square(4, 6), Square(4, 7), Square(5, 6), Square(5, 7)}
        assert left.issubset(set(C.LAKES))
        assert right.issubset(set(C.LAKES))
        assert left | right == set(C.LAKES)

    def test_lake_set_lookup(self) -> None:
        for sq in C.LAKES:
            assert sq in C.LAKE_SET
        assert Square(0, 0) not in C.LAKE_SET


# ---------------------------------------------------------------------------
# Infostate channel counts (from paper appendix)
# ---------------------------------------------------------------------------


class TestInfostateConstants:
    def test_board_channels(self) -> None:
        assert C.NUM_BOARD_CHANNELS == 456

    def test_move_channels(self) -> None:
        assert C.NUM_MOVE_CHANNELS == 32
        assert C.MOVE_HISTORY_LENGTH == 32

    def test_total_infostate_channels(self) -> None:
        assert C.NUM_INFOSTATE_CHANNELS == 488

    def test_starting_location_channels(self) -> None:
        assert C.NUM_STARTING_LOCATION_CHANNELS == 101


# ---------------------------------------------------------------------------
# Training hyperparameters (from paper appendix tables)
# ---------------------------------------------------------------------------


class TestTrainingConstants:
    def test_data_collection(self) -> None:
        assert C.NUM_ENVS_PER_GPU == 1536
        assert C.NUM_MOVES_BETWEEN_ITERATIONS == 202
        assert C.NUM_GENERATED_SETUPS_PER_PLAYER_PER_GPU == 1000

    def test_setup_hyperparams(self) -> None:
        assert C.SETUP_LR == 5e-5
        assert C.SETUP_BATCH_SIZE_PER_GPU == 1024
        assert C.SETUP_EPOCHS_PER_ITER == 5
        assert C.SETUP_PPO_CLIP == 0.2
        assert C.SETUP_ENTROPY_COEFF == 1.0
        assert C.SETUP_KL_COEFF == 0.1
        assert C.SETUP_VALUE_COEFF == 0.5
        assert C.SETUP_REG_TEMP_NUMERATOR == 0.1
        assert C.SETUP_REG_TEMP_EXPONENT == 0.3
        assert C.SETUP_MAX_GRAD_NORM == 0.5
        assert C.SETUP_EMA_SMOOTHING == 0.999
        assert C.SETUP_ENTROPY_NORMALIZER == 10.0

    def test_move_hyperparams(self) -> None:
        assert C.MOVE_PPO_CLIP == 0.2
        assert C.MOVE_ADV_QUANTILE == 0.75
        assert C.MOVE_ADV_MAGNITUDE_THRESHOLD == 0.01
        assert C.MOVE_EMA_SMOOTHING == 0.999
        assert C.MOVE_ADVANTAGE_LAMBDA == 0.5
        assert C.MOVE_OUTCOME_LAMBDA == 0.8
        assert C.MOVE_KL_COEFF == 0.1
        assert C.MOVE_LR_NUMERATOR == 0.5
        assert C.MOVE_LR_EXPONENT == 1.1
        assert C.MOVE_LR_MIN == 5e-6
        assert C.MOVE_LR_MAX == 1e-4
        assert C.MOVE_MAGNET_KL_NUMERATOR == 0.05
        assert C.MOVE_MAGNET_KL_EXPONENT == 0.3
        assert C.MOVE_MAX_GRAD_NORM == 0.267
        assert C.MOVE_EPOCHS_PER_ITER == 1
        assert C.MOVE_VALUE_COEFF == 1.0
        assert C.MOVE_BATCH_SIZE_PER_GPU == 1536

    def test_belief_hyperparams(self) -> None:
        assert C.BELIEF_DROPOUT == 0.2

    def test_search_hyperparams(self) -> None:
        assert C.SEARCH_MAGNET_KL_COEFF == 0.002
        assert C.SEARCH_NET_KL_COEFF == 0.02
        assert C.SEARCH_NUM_ROLLOUTS == 1000
        assert C.SEARCH_ROLLOUT_DEPTH == 40


# ---------------------------------------------------------------------------
# Network architecture constants
# ---------------------------------------------------------------------------


class TestNetworkConstants:
    def test_setup_net(self) -> None:
        assert C.SETUP_NET_DEPTH == 4
        assert C.SETUP_NET_DIM == 512
        assert C.SETUP_NET_HEADS == 8
        assert C.SETUP_NET_FF == 2048

    def test_move_net(self) -> None:
        assert C.MOVE_NET_DEPTH == 8
        assert C.MOVE_NET_DIM == 384
        assert C.MOVE_NET_HEADS == 8
        assert C.MOVE_NET_FF == 1536

    def test_belief_net(self) -> None:
        assert C.BELIEF_NET_ENCODER_DEPTH == 6
        assert C.BELIEF_NET_DECODER_BLOCKS == 4
        assert C.BELIEF_NET_HEADS == 8
        assert C.BELIEF_NET_DIM == 512
        assert C.BELIEF_NET_FF == 2048

    def test_game_rules_constants(self) -> None:
        assert C.TRAINING_NO_ATTACK_LIMIT == 100
        assert C.EVAL_NO_ATTACK_LIMIT == 200
        assert C.MAX_GAME_LENGTH == 4000
        assert C.TWO_SQUARE_LIMIT == 3
