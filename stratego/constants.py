"""Project-wide constants for the Ataraxos Stratego environment.

All magic numbers from the paper (arXiv:2511.07312) are centralized here so that
both the pure-Python reference rules engine and the CUDA C++ simulator reference
the same source of truth.
"""

from __future__ import annotations

from stratego.types import NUM_PIECE_TYPES, PieceType, Square, piece_count

# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------

BOARD_ROWS = 10
BOARD_COLS = 10
NUM_SQUARES = BOARD_ROWS * BOARD_COLS  # 100
NUM_LAKE_SQUARES = 8
NUM_OCCUPIABLE = NUM_SQUARES - NUM_LAKE_SQUARES  # 92

# Lake positions (0-indexed row, col). Two 2x2 lakes in the middle rows.
# Confirmed by ISF rules PDF + cross-referenced with benletchford/stratego.io
# and TextArena/TextArena.
LAKES: list[Square] = [
    Square(4, 2), Square(4, 3), Square(5, 2), Square(5, 3),  # left lake
    Square(4, 6), Square(4, 7), Square(5, 6), Square(5, 7),  # right lake
]
LAKE_SET: frozenset[Square] = frozenset(LAKES)

# Setup zones — first 4 rows of each player's side.
RED_SETUP_ROWS: tuple[int, int] = (0, 3)  # inclusive
BLUE_SETUP_ROWS: tuple[int, int] = (6, 9)  # inclusive

# ---------------------------------------------------------------------------
# Piece roster
# ---------------------------------------------------------------------------

TOTAL_PIECES = 40  # per player

PIECE_COUNTS: dict[PieceType, int] = {pt: piece_count(pt) for pt in PieceType if pt != PieceType.NONE}

# Sanity check: counts sum to 40
assert sum(PIECE_COUNTS.values()) == TOTAL_PIECES, "Stratego piece counts must sum to 40"

# Cardinal directions as (drow, dcol) deltas: up, down, left, right.
CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

# ---------------------------------------------------------------------------
# Game rules (from app:game rules in methods.tex)
# ---------------------------------------------------------------------------

# Number of consecutive battleless moves after which a draw is declared.
# Training uses 100; evaluation uses 200 (Strategus online rule).
TRAINING_NO_ATTACK_LIMIT = 100
EVAL_NO_ATTACK_LIMIT = 200

# Maximum game length (in half-moves / simulator steps) before a forced draw.
# Not an actual Stratego rule — edge case safety from the paper.
MAX_GAME_LENGTH = 4000

# Two-square rule: a piece may not cross the same square boundary more than
# TWO_SQUARE_LIMIT times in a row (ISF rule chapter 10). The limit is 3 crossings.
TWO_SQUARE_LIMIT = 3

# ---------------------------------------------------------------------------
# Infostate representation (from app:infostate in methods.tex)
# ---------------------------------------------------------------------------

# Board state channels: 0-455 (456 total). See stratego/env/infostate.py for full spec.
NUM_BOARD_CHANNELS = 456

# Last-move channels: 456-487 (32 total). Each channel encodes one past move:
# +1.0 at dst, -1.0 at src, 0.0 elsewhere.
NUM_MOVE_CHANNELS = 32
MOVE_HISTORY_LENGTH = NUM_MOVE_CHANNELS  # 32 past moves encoded

# Total infostate depth: 488 channels, each 10x10.
NUM_INFOSTATE_CHANNELS = NUM_BOARD_CHANNELS + NUM_MOVE_CHANNELS  # 488

# One-hot starting-location channels: 355-455 (101 squares = 100 board + 1 'not on board').
NUM_STARTING_LOCATION_CHANNELS = 101

# ---------------------------------------------------------------------------
# Training hyperparameters (from methods.tex appendix tables)
# ---------------------------------------------------------------------------

# Data collection (from tab:data-hyper)
NUM_ENVS_PER_GPU = 1536
NUM_MOVES_BETWEEN_ITERATIONS = 202
NUM_GENERATED_SETUPS_PER_PLAYER_PER_GPU = 1000

# Setup learning (from tab:setup-learning-hyper)
SETUP_LR = 5e-5
SETUP_BATCH_SIZE_PER_GPU = 1024
SETUP_EPOCHS_PER_ITER = 5
SETUP_PPO_CLIP = 0.2  # range [1 - 0.2, 1 + 0.2] = [0.8, 1.2]
SETUP_ENTROPY_COEFF = 1.0
SETUP_KL_COEFF = 0.1
SETUP_VALUE_COEFF = 0.5
SETUP_REG_TEMP_NUMERATOR = 0.1
SETUP_REG_TEMP_EXPONENT = 0.3  # alpha = 0.1 / iter^0.3
SETUP_MAX_GRAD_NORM = 0.5
SETUP_EMA_SMOOTHING = 0.999
SETUP_ENTROPY_NORMALIZER = 10.0  # H is divided by 10 before MSE

# Move learning (from tab:move-learning-hyper)
MOVE_PPO_CLIP = 0.2
MOVE_ADV_QUANTILE = 0.75
MOVE_ADV_MAGNITUDE_THRESHOLD = 0.01
MOVE_EMA_SMOOTHING = 0.999
MOVE_ADVANTAGE_LAMBDA = 0.5
MOVE_OUTCOME_LAMBDA = 0.8
MOVE_KL_COEFF = 0.1  # reverse KL to data collection policy
MOVE_LR_NUMERATOR = 0.5
MOVE_LR_EXPONENT = 1.1  # lr = clip(0.5 / iter^1.1, 5e-6, 1e-4)
MOVE_LR_MIN = 5e-6
MOVE_LR_MAX = 1e-4
MOVE_MAGNET_KL_NUMERATOR = 0.05  # alpha = 0.05 / iter^0.3 (reverse KL to magnet policy)
MOVE_MAGNET_KL_EXPONENT = 0.3
MOVE_MAX_GRAD_NORM = 0.267
MOVE_EPOCHS_PER_ITER = 1
MOVE_VALUE_COEFF = 1.0
MOVE_BATCH_SIZE_PER_GPU = 1536  # before advantage filtering

# Belief learning (from tab:belief-network-hyper)
BELIEF_DROPOUT = 0.2
BELIEF_EMA_SMOOTHING = 0.999  # assumed; not explicitly stated

# Search (from tab:search-hyper)
SEARCH_MAGNET_KL_COEFF = 0.002  # alpha
SEARCH_NET_KL_COEFF = 0.02  # beta
SEARCH_NUM_ROLLOUTS = 1000
SEARCH_ROLLOUT_DEPTH = 40

# ---------------------------------------------------------------------------
# Network architecture (from tab:setup-network-hyper, tab:move-network-hyper, tab:belief-network-hyper)
# ---------------------------------------------------------------------------

# Setup network
SETUP_NET_DEPTH = 4
SETUP_NET_DIM = 512
SETUP_NET_HEADS = 8
SETUP_NET_FF = 2048
SETUP_NET_POS_EMB_INIT_STD = 0.1

# Move network
MOVE_NET_DEPTH = 8
MOVE_NET_DIM = 384
MOVE_NET_HEADS = 8
MOVE_NET_FF = 1536
MOVE_NET_POS_EMB_INIT_STD = 0.1

# Belief network
BELIEF_NET_ENCODER_DEPTH = 6
BELIEF_NET_DECODER_BLOCKS = 4
BELIEF_NET_HEADS = 8
BELIEF_NET_DIM = 512
BELIEF_NET_FF = 2048

# Common
NUM_OCCUPIABLE_SQUARES = 92  # token count for move/belief encoder input

# ---------------------------------------------------------------------------
# Time controls (for evaluation, not training)
# ---------------------------------------------------------------------------

EVAL_TIME_BUFFER_SECONDS = 15 * 60  # 15 minutes
EVAL_TIME_INCREMENT_SECONDS = 3  # 3 free seconds per move

__all__ = [
    "BELIEF_DROPOUT",
    "BELIEF_EMA_SMOOTHING",
    "BELIEF_NET_DECODER_BLOCKS",
    "BELIEF_NET_DIM",
    "BELIEF_NET_ENCODER_DEPTH",
    "BELIEF_NET_FF",
    "BELIEF_NET_HEADS",
    "BOARD_COLS",
    "BOARD_ROWS",
    "CARDINAL_DIRECTIONS",
    "EVAL_NO_ATTACK_LIMIT",
    "EVAL_TIME_BUFFER_SECONDS",
    "EVAL_TIME_INCREMENT_SECONDS",
    "LAKE_SET",
    "LAKES",
    "MAX_GAME_LENGTH",
    "MOVE_ADV_MAGNITUDE_THRESHOLD",
    "MOVE_ADV_QUANTILE",
    "MOVE_ADVANTAGE_LAMBDA",
    "MOVE_BATCH_SIZE_PER_GPU",
    "MOVE_EPOCHS_PER_ITER",
    "MOVE_EMA_SMOOTHING",
    "MOVE_KL_COEFF",
    "MOVE_LR_EXPONENT",
    "MOVE_LR_MAX",
    "MOVE_LR_MIN",
    "MOVE_LR_NUMERATOR",
    "MOVE_MAGNET_KL_EXPONENT",
    "MOVE_MAGNET_KL_NUMERATOR",
    "MOVE_MAX_GRAD_NORM",
    "MOVE_NET_DEPTH",
    "MOVE_NET_DIM",
    "MOVE_NET_FF",
    "MOVE_NET_HEADS",
    "MOVE_NET_POS_EMB_INIT_STD",
    "MOVE_OUTCOME_LAMBDA",
    "MOVE_PPO_CLIP",
    "MOVE_VALUE_COEFF",
    "NUM_BOARD_CHANNELS",
    "NUM_ENVS_PER_GPU",
    "NUM_GENERATED_SETUPS_PER_PLAYER_PER_GPU",
    "NUM_INFOSTATE_CHANNELS",
    "NUM_LAKE_SQUARES",
    "NUM_MOVE_CHANNELS",
    "NUM_MOVES_BETWEEN_ITERATIONS",
    "NUM_OCCUPIABLE",
    "NUM_OCCUPIABLE_SQUARES",
    "NUM_PIECE_TYPES",
    "NUM_SQUARES",
    "NUM_STARTING_LOCATION_CHANNELS",
    "PIECE_COUNTS",
    "RED_SETUP_ROWS",
    "BLUE_SETUP_ROWS",
    "SEARCH_MAGNET_KL_COEFF",
    "SEARCH_NET_KL_COEFF",
    "SEARCH_NUM_ROLLOUTS",
    "SEARCH_ROLLOUT_DEPTH",
    "SETUP_BATCH_SIZE_PER_GPU",
    "SETUP_EMA_SMOOTHING",
    "SETUP_ENTROPY_COEFF",
    "SETUP_ENTROPY_NORMALIZER",
    "SETUP_EPOCHS_PER_ITER",
    "SETUP_KL_COEFF",
    "SETUP_LR",
    "SETUP_MAX_GRAD_NORM",
    "SETUP_NET_DEPTH",
    "SETUP_NET_DIM",
    "SETUP_NET_FF",
    "SETUP_NET_HEADS",
    "SETUP_NET_POS_EMB_INIT_STD",
    "SETUP_PPO_CLIP",
    "SETUP_REG_TEMP_EXPONENT",
    "SETUP_REG_TEMP_NUMERATOR",
    "SETUP_VALUE_COEFF",
    "TOTAL_PIECES",
    "TRAINING_NO_ATTACK_LIMIT",
    "TWO_SQUARE_LIMIT",
]
