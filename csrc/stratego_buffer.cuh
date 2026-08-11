#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace stratego {

// ---------------------------------------------------------------------------
// PieceType values (mirror stratego/types.py PieceType IntEnum)
// ---------------------------------------------------------------------------
static constexpr int8_t PT_NONE = 0;
static constexpr int8_t PT_SPY = 1;
static constexpr int8_t PT_SCOUT = 2;
static constexpr int8_t PT_MINER = 3;
static constexpr int8_t PT_SERGEANT = 4;
static constexpr int8_t PT_LIEUTENANT = 5;
static constexpr int8_t PT_CAPTAIN = 6;
static constexpr int8_t PT_MAJOR = 7;
static constexpr int8_t PT_COLONEL = 8;
static constexpr int8_t PT_GENERAL = 9;
static constexpr int8_t PT_MARSHAL = 10;
static constexpr int8_t PT_FLAG = 11;
static constexpr int8_t PT_BOMB = 12;

// ---------------------------------------------------------------------------
// Player values (mirror stratego/types.py Player IntEnum)
// ---------------------------------------------------------------------------
static constexpr int8_t PLAYER_RED = 0;
static constexpr int8_t PLAYER_BLUE = 1;
static constexpr int8_t PLAYER_EMPTY = -1;

// ---------------------------------------------------------------------------
// GameOutcome values (mirror stratego/types.py GameOutcome IntEnum)
// ---------------------------------------------------------------------------
static constexpr int8_t OUTCOME_ONGOING = -1;
static constexpr int8_t OUTCOME_RED_WIN = 0;
static constexpr int8_t OUTCOME_BLUE_WIN = 1;
static constexpr int8_t OUTCOME_DRAW = 2;

// ---------------------------------------------------------------------------
// Board geometry (mirror stratego/constants.py)
// ---------------------------------------------------------------------------
static constexpr int BOARD_ROWS = 10;
static constexpr int BOARD_COLS = 10;
static constexpr int NUM_SQUARES = BOARD_ROWS * BOARD_COLS;  // 100
static constexpr int NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES;  // 10000

// ---------------------------------------------------------------------------
// Game rules (mirror stratego/constants.py)
// ---------------------------------------------------------------------------
static constexpr int TRAINING_NO_ATTACK_LIMIT = 100;
static constexpr int MAX_GAME_LENGTH = 4000;
static constexpr int TWO_SQUARE_LIMIT = 3;  // max consecutive crossings allowed

// ---------------------------------------------------------------------------
// Two-square + chasing rule tracking (GPU-side state)
// ---------------------------------------------------------------------------
static constexpr int NUM_PLAYERS = 2;
static constexpr int CHASE_HASH_BUFFER_SIZE = 64;   // ring buffer per player per game
static constexpr int NUM_ZOBRIST_PIECES = 13;       // PT_NONE(0) .. PT_BOMB(12)

// ---------------------------------------------------------------------------
// Infostate representation (mirror stratego/constants.py)
// ---------------------------------------------------------------------------
static constexpr int NUM_PIECE_TYPES = 12;
static constexpr int NUM_MOVE_HISTORY = 32;
static constexpr int NUM_BOARD_CHANNELS = 456;
static constexpr int NUM_INFOSTATE_CHANNELS = 488;  // 456 + 32

// ---------------------------------------------------------------------------
// Hello-world kernel (kept for build verification)
// ---------------------------------------------------------------------------
__global__ void hello_world_kernel(int* out, int n);
void launch_hello_world(int* out, int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// apply_actions_kernel
//   For each game: decode action, validate, apply movement/combat, check terminal.
//   actions[gid] encodes (src_idx * 100 + dst_idx) where idx = row*10 + col.
// ---------------------------------------------------------------------------
__global__ void apply_actions_kernel(
    int8_t* board_owner,       // (N, 10, 10)
    int8_t* board_piece,       // (N, 10, 10)
    const int64_t* actions,    // (N,)
    int8_t* current_player,    // (N,)
    int32_t* move_number,      // (N,)
    int32_t* moves_since_attack, // (N,)
    int8_t* outcome,           // (N,)
    bool* terminated,          // (N,)
    bool* moved_squares,       // (N, 100)
    bool* revealed_squares,    // (N, 100)
    int32_t* move_history,    // (N, 32)
    int32_t* move_history_head, // (N,)
    int32_t* move_history_len, // (N,)
    // --- two-square rule tracking ---
    int32_t* two_square_count,    // (N, 2)
    int32_t* two_square_pair_lo,  // (N, 2)
    int32_t* two_square_pair_hi,  // (N, 2)
    bool* two_square_violation,   // (N,)
    // --- chasing rule tracking ---
    const int64_t* zobrist_table,  // (10, 10, 2, 13)
    int64_t* chase_hashes,         // (N, 2, 64)
    int32_t* chase_hash_head,      // (N, 2)
    int32_t* chase_hash_len,       // (N, 2)
    int32_t* chase_last_src,       // (N, 2)
    bool* chasing_violation,       // (N,)
    int N);

void launch_apply_actions(
    int8_t* board_owner,
    int8_t* board_piece,
    const int64_t* actions,
    int8_t* current_player,
    int32_t* move_number,
    int32_t* moves_since_attack,
    int8_t* outcome,
    bool* terminated,
    bool* moved_squares,
    bool* revealed_squares,
    int32_t* move_history,
    int32_t* move_history_head,
    int32_t* move_history_len,
    int32_t* two_square_count,
    int32_t* two_square_pair_lo,
    int32_t* two_square_pair_hi,
    bool* two_square_violation,
    const int64_t* zobrist_table,
    int64_t* chase_hashes,
    int32_t* chase_hash_head,
    int32_t* chase_hash_len,
    int32_t* chase_last_src,
    bool* chasing_violation,
    int N,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// legal_action_mask_kernel
//   For each (game, src_square): enumerate legal moves, set mask entries.
//   mask is (N, 100, 100) bool, flat index = game*10000 + src_idx*100 + dst_idx.
// ---------------------------------------------------------------------------
__global__ void legal_action_mask_kernel(
    const int8_t* board_owner,   // (N, 10, 10)
    const int8_t* board_piece,   // (N, 10, 10)
    const int8_t* current_player, // (N,)
    bool* mask,                   // (N, 100, 100)
    int N);

void launch_legal_action_mask(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int8_t* current_player,
    bool* mask,
    int N,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// reset_kernel
//   For each terminated game: copy setup_red/setup_blue into board, reset scalars.
// ---------------------------------------------------------------------------
__global__ void reset_kernel(
    int8_t* board_owner,         // (N, 10, 10)
    int8_t* board_piece,         // (N, 10, 10)
    const int8_t* setup_red,    // (N, 10, 10)
    const int8_t* setup_blue,    // (N, 10, 10)
    int8_t* current_player,      // (N,)
    int32_t* move_number,        // (N,)
    int32_t* moves_since_attack, // (N,)
    int8_t* outcome,             // (N,)
    bool* terminated,            // (N,)
    bool* moved_squares,         // (N, 100)
    bool* revealed_squares,      // (N, 100)
    int32_t* move_history,      // (N, 32)
    int32_t* move_history_head,  // (N,)
    int32_t* move_history_len,  // (N,)
    // --- two-square rule tracking ---
    int32_t* two_square_count,
    int32_t* two_square_pair_lo,
    int32_t* two_square_pair_hi,
    bool* two_square_violation,
    // --- chasing rule tracking ---
    int64_t* chase_hashes,
    int32_t* chase_hash_head,
    int32_t* chase_hash_len,
    int32_t* chase_last_src,
    bool* chasing_violation,
    int N);

void launch_reset(
    int8_t* board_owner,
    int8_t* board_piece,
    const int8_t* setup_red,
    const int8_t* setup_blue,
    int8_t* current_player,
    int32_t* move_number,
    int32_t* moves_since_attack,
    int8_t* outcome,
    bool* terminated,
    bool* moved_squares,
    bool* revealed_squares,
    int32_t* move_history,
    int32_t* move_history_head,
    int32_t* move_history_len,
    int32_t* two_square_count,
    int32_t* two_square_pair_lo,
    int32_t* two_square_pair_hi,
    bool* two_square_violation,
    int64_t* chase_hashes,
    int32_t* chase_hash_head,
    int32_t* chase_hash_len,
    int32_t* chase_last_src,
    bool* chasing_violation,
    int N,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// compute_infostate_kernel
//   For each (game, channel, square): compute the infostate tensor value.
//   Output is (N, 488, 10, 10) float32. Channels 43-455 are zeros (deferred).
//   Perspective: current_player[game] determines "own" vs "opp".
// ---------------------------------------------------------------------------
__global__ void compute_infostate_kernel(
    const int8_t* board_owner,        // (N, 100)
    const int8_t* board_piece,        // (N, 100)
    const int8_t* current_player,     // (N,)
    const int32_t* move_number,       // (N,)
    const int32_t* moves_since_attack, // (N,)
    const bool* moved_squares,        // (N, 100)
    const bool* revealed_squares,     // (N, 100)
    const int32_t* move_history,     // (N, 32)
    const int32_t* move_history_head, // (N,)
    const int32_t* move_history_len, // (N,)
    float* out,                       // (N, 488, 100)
    int N);

void launch_compute_infostate(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int8_t* current_player,
    const int32_t* move_number,
    const int32_t* moves_since_attack,
    const bool* moved_squares,
    const bool* revealed_squares,
    const int32_t* move_history,
    const int32_t* move_history_head,
    const int32_t* move_history_len,
    float* out,
    int N,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// hash_board_kernel
//   One thread per game. Computes Zobrist hash of the current board.
//   hash = XOR of zobrist_table[r, c, owner, piece] for all occupied squares.
// ---------------------------------------------------------------------------
__global__ void hash_board_kernel(
    const int8_t* board_owner,   // (N, 10, 10)
    const int8_t* board_piece,   // (N, 10, 10)
    const int64_t* zobrist_table, // (10, 10, 2, 13)
    int64_t* out,                 // (N,)
    int N);

void launch_hash_board(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int64_t* zobrist_table,
    int64_t* out,
    int N,
    cudaStream_t stream);

}  // namespace stratego
