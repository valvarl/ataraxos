#include "stratego_buffer.cuh"
#include "utils.cuh"

namespace stratego {

// ---------------------------------------------------------------------------
// Device helper functions
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool is_lake(int r, int c) {
    // Lakes: (4,2),(4,3),(5,2),(5,3) left; (4,6),(4,7),(5,6),(5,7) right
    if (r != 4 && r != 5) return false;
    if (c >= 2 && c <= 3) return true;
    if (c >= 6 && c <= 7) return true;
    return false;
}

__device__ __forceinline__ int piece_rank(int8_t pt) {
    // Ranked pieces: SPY=1..MARSHAL=10. Flag/Bomb/NONE have rank 0.
    if (pt >= PT_SPY && pt <= PT_MARSHAL) return static_cast<int>(pt);
    return 0;
}

__device__ __forceinline__ bool piece_can_move(int8_t pt) {
    return pt >= PT_SPY && pt <= PT_MARSHAL;
}

// Check if the given player has at least one legal move on the board.
// For the "has any legal move" check, it suffices to examine the 4 cardinal
// neighbours of each movable piece — a Scout that can move at all can move
// to an adjacent square.
__device__ bool has_any_legal_move(
    const int8_t* game_owner,  // 100 elements
    const int8_t* game_piece,  // 100 elements
    int8_t player)
{
    for (int src_idx = 0; src_idx < NUM_SQUARES; ++src_idx) {
        if (game_owner[src_idx] != player) continue;
        int8_t pt = game_piece[src_idx];
        if (!piece_can_move(pt)) continue;

        int src_r = src_idx / BOARD_COLS;
        int src_c = src_idx % BOARD_COLS;

        // 4 cardinal directions: up, down, left, right
        const int drs[4] = {-1, 1, 0, 0};
        const int dcs[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; ++d) {
            int r = src_r + drs[d];
            int c = src_c + dcs[d];
            if (r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS) continue;
            if (is_lake(r, c)) continue;
            int idx = r * BOARD_COLS + c;
            if (game_owner[idx] == player) continue;  // own piece
            return true;  // empty or enemy = legal move
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Hello-world kernel (kept for build verification)
// ---------------------------------------------------------------------------

__global__ void hello_world_kernel(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 42;
    }
}

void launch_hello_world(int* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ATARAXOS_CUDA_LAUNCH(hello_world_kernel, blocks, threads, stream, out, n);
}

// ---------------------------------------------------------------------------
// apply_actions_kernel
// ---------------------------------------------------------------------------

__global__ void apply_actions_kernel(
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
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // Skip already-terminated games
    if (terminated[gid]) return;

    int8_t player = current_player[gid];
    int8_t* game_owner = board_owner + gid * NUM_SQUARES;
    int8_t* game_piece = board_piece + gid * NUM_SQUARES;
    bool* game_moved = moved_squares + gid * NUM_SQUARES;
    bool* game_revealed = revealed_squares + gid * NUM_SQUARES;
    int32_t* game_history = move_history + gid * NUM_MOVE_HISTORY;

    int64_t raw_action = actions[gid];
    if (raw_action < 0 || raw_action >= NUM_ACTIONS) return;  // invalid

    int src_idx = static_cast<int>(raw_action / NUM_SQUARES);
    int dst_idx = static_cast<int>(raw_action % NUM_SQUARES);
    if (src_idx == dst_idx) return;  // no-op

    int src_r = src_idx / BOARD_COLS;
    int src_c = src_idx % BOARD_COLS;
    int dst_r = dst_idx / BOARD_COLS;
    int dst_c = dst_idx % BOARD_COLS;

    int8_t src_owner = game_owner[src_idx];
    int8_t src_piece = game_piece[src_idx];
    int8_t dst_owner = game_owner[dst_idx];
    int8_t dst_piece = game_piece[dst_idx];

    // Validate: source belongs to current player
    if (src_owner != player) return;
    // Validate: piece can move
    if (!piece_can_move(src_piece)) return;
    // Validate: destination is not own piece
    if (dst_owner == player) return;
    // Validate: destination is not a lake
    if (is_lake(dst_r, dst_c)) return;

    int dr = dst_r - src_r;
    int dc = dst_c - src_c;

    // Must be cardinal (straight line)
    if (dr != 0 && dc != 0) return;

    bool is_scout = (src_piece == PT_SCOUT);
    if (!is_scout) {
        // Non-scout: must move exactly 1 square
        int dist = (dr != 0) ? (dr > 0 ? dr : -dr) : (dc > 0 ? dc : -dc);
        if (dist != 1) return;
    } else {
        // Scout: path must be clear (no jumping over pieces or lakes)
        int step_r = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
        int step_c = (dc == 0) ? 0 : (dc > 0 ? 1 : -1);
        int steps = (dr != 0) ? (dr > 0 ? dr : -dr) : (dc > 0 ? dc : -dc);
        int r = src_r + step_r;
        int c = src_c + step_c;
        for (int s = 1; s < steps; ++s) {
            if (is_lake(r, c)) return;  // path blocked by lake
            int idx = r * BOARD_COLS + c;
            if (game_owner[idx] != PLAYER_EMPTY) return;  // path blocked
            r += step_r;
            c += step_c;
        }
        // dst square itself: already validated not own piece, not lake
    }

    bool is_attack = (dst_owner != PLAYER_EMPTY && dst_owner != player);

    if (is_attack) {
        int8_t attacker = src_piece;
        int8_t defender = dst_piece;

        if (defender == PT_FLAG) {
            // Flag capture: attacker wins game
            game_owner[dst_idx] = player;
            game_piece[dst_idx] = attacker;
            game_owner[src_idx] = PLAYER_EMPTY;
            game_piece[src_idx] = PT_NONE;
            outcome[gid] = (player == PLAYER_RED) ? OUTCOME_RED_WIN : OUTCOME_BLUE_WIN;
            moves_since_attack[gid] = 0;
        } else if (defender == PT_BOMB) {
            if (attacker == PT_MINER) {
                // Miner defuses bomb
                game_owner[dst_idx] = player;
                game_piece[dst_idx] = attacker;
                game_owner[src_idx] = PLAYER_EMPTY;
                game_piece[src_idx] = PT_NONE;
            } else {
                // Bomb kills attacker
                game_owner[src_idx] = PLAYER_EMPTY;
                game_piece[src_idx] = PT_NONE;
                // Bomb stays
            }
            moves_since_attack[gid] = 0;
        } else if (attacker == PT_SPY && defender == PT_MARSHAL) {
            // Spy attacking Marshal: Spy wins
            game_owner[dst_idx] = player;
            game_piece[dst_idx] = attacker;
            game_owner[src_idx] = PLAYER_EMPTY;
            game_piece[src_idx] = PT_NONE;
            moves_since_attack[gid] = 0;
        } else {
            // Standard rank-based combat
            int atk_rank = piece_rank(attacker);
            int def_rank = piece_rank(defender);

            if (atk_rank > def_rank) {
                // Attacker wins
                game_owner[dst_idx] = player;
                game_piece[dst_idx] = attacker;
                game_owner[src_idx] = PLAYER_EMPTY;
                game_piece[src_idx] = PT_NONE;
            } else if (atk_rank < def_rank) {
                // Defender wins
                game_owner[src_idx] = PLAYER_EMPTY;
                game_piece[src_idx] = PT_NONE;
            } else {
                // Equal rank: both die
                game_owner[src_idx] = PLAYER_EMPTY;
                game_piece[src_idx] = PT_NONE;
                game_owner[dst_idx] = PLAYER_EMPTY;
                game_piece[dst_idx] = PT_NONE;
            }
            moves_since_attack[gid] = 0;
        }
    } else {
        // Simple move to empty square
        game_owner[dst_idx] = player;
        game_piece[dst_idx] = src_piece;
        game_owner[src_idx] = PLAYER_EMPTY;
        game_piece[src_idx] = PT_NONE;
        moves_since_attack[gid] += 1;
    }

    // Track moved/revealed squares and push to move history ring buffer
    if (is_attack) {
        game_revealed[src_idx] = true;
        game_revealed[dst_idx] = true;
    } else {
        game_moved[src_idx] = true;
        game_moved[dst_idx] = true;
        bool src_revealed = game_revealed[src_idx];
        game_revealed[dst_idx] = src_revealed;
        game_revealed[src_idx] = false;
    }

    int32_t head = move_history_head[gid];
    game_history[head] = static_cast<int32_t>(raw_action);
    move_history_head[gid] = (head + 1) % NUM_MOVE_HISTORY;
    if (move_history_len[gid] < NUM_MOVE_HISTORY) {
        move_history_len[gid] += 1;
    }

    // Advance turn
    move_number[gid] += 1;
    current_player[gid] = (player == PLAYER_RED) ? PLAYER_BLUE : PLAYER_RED;

    // -----------------------------------------------------------------------
    // Two-square rule tracking (simplified per-player)
    //   - non-attack move: update pair + count, set violation if count > limit
    //   - attack move: reset count = 0 for this player
    //   - opponent moves do NOT reset (per-piece, simplified to per-player)
    // -----------------------------------------------------------------------
    {
        int pidx = static_cast<int>(player);
        int32_t* ts_count = two_square_count + gid * NUM_PLAYERS;
        int32_t* ts_lo = two_square_pair_lo + gid * NUM_PLAYERS;
        int32_t* ts_hi = two_square_pair_hi + gid * NUM_PLAYERS;

        if (is_attack) {
            ts_count[pidx] = 0;
            ts_lo[pidx] = -1;
            ts_hi[pidx] = -1;
        } else {
            int lo = (src_idx < dst_idx) ? src_idx : dst_idx;
            int hi = (src_idx < dst_idx) ? dst_idx : src_idx;
            if (ts_lo[pidx] == lo && ts_hi[pidx] == hi) {
                ts_count[pidx] += 1;
            } else {
                ts_count[pidx] = 1;
                ts_lo[pidx] = lo;
                ts_hi[pidx] = hi;
            }
            if (ts_count[pidx] > TWO_SQUARE_LIMIT) {
                two_square_violation[gid] = true;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Chasing rule tracking (simplified)
    //   - threatening move (piece at dst adjacent to opponent):
    //       compute board hash, check if hash in chase set,
    //       violation if found AND dst != last_chaser_src (exception),
    //       add hash to ring buffer, update last_chaser_src = src
    //   - non-threatening move: clear chase set + last_chaser_src
    // -----------------------------------------------------------------------
    {
        int pidx = static_cast<int>(player);
        int8_t opp = (player == PLAYER_RED) ? PLAYER_BLUE : PLAYER_RED;
        int64_t* game_chase_hashes =
            chase_hashes + (gid * NUM_PLAYERS + pidx) * CHASE_HASH_BUFFER_SIZE;
        int32_t* game_chase_head = chase_hash_head + gid * NUM_PLAYERS;
        int32_t* game_chase_len = chase_hash_len + gid * NUM_PLAYERS;
        int32_t* game_chase_last_src = chase_last_src + gid * NUM_PLAYERS;

        // Detect threat: piece at dst_idx is adjacent to an opponent piece
        bool threatening = false;
        const int drs[4] = {-1, 1, 0, 0};
        const int dcs[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; ++d) {
            int r = dst_r + drs[d];
            int c = dst_c + dcs[d];
            if (r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS) continue;
            if (is_lake(r, c)) continue;
            int idx = r * BOARD_COLS + c;
            if (game_owner[idx] == opp) {
                threatening = true;
                break;
            }
        }

        if (threatening) {
            // Compute Zobrist hash of the current (post-move) board
            int64_t h = 0;
            for (int sq = 0; sq < NUM_SQUARES; ++sq) {
                int8_t ow = game_owner[sq];
                if (ow >= 0) {
                    int r = sq / BOARD_COLS;
                    int c = sq % BOARD_COLS;
                    h ^= zobrist_table[
                        ((r * BOARD_COLS + c) * NUM_PLAYERS + ow) *
                        NUM_ZOBRIST_PIECES + game_piece[sq]];
                }
            }

            // Check if hash is already in chase set
            int32_t len = game_chase_len[pidx];
            bool found = false;
            for (int i = 0; i < len; ++i) {
                if (game_chase_hashes[i] == h) {
                    found = true;
                    break;
                }
            }

            // Exception: dst == last_chaser_src (back to preceding square)
            int32_t last_src = game_chase_last_src[pidx];
            bool is_exception = (last_src >= 0 && dst_idx == last_src);

            if (found && !is_exception) {
                chasing_violation[gid] = true;
            }

            // Add hash to ring buffer
            if (len < CHASE_HASH_BUFFER_SIZE) {
                game_chase_hashes[len] = h;
                game_chase_len[pidx] = len + 1;
            } else {
                game_chase_hashes[game_chase_head[pidx]] = h;
                game_chase_head[pidx] =
                    (game_chase_head[pidx] + 1) % CHASE_HASH_BUFFER_SIZE;
            }

            // Update last chaser src to this move's source
            game_chase_last_src[pidx] = src_idx;
        } else {
            // Non-threatening move: clear chase positions
            game_chase_len[pidx] = 0;
            game_chase_head[pidx] = 0;
            game_chase_last_src[pidx] = -1;
        }
    }

    // Check terminal (only if not already set by flag capture)
    if (outcome[gid] == OUTCOME_ONGOING) {
        if (move_number[gid] >= MAX_GAME_LENGTH) {
            outcome[gid] = OUTCOME_DRAW;
        } else if (moves_since_attack[gid] >= TRAINING_NO_ATTACK_LIMIT) {
            outcome[gid] = OUTCOME_DRAW;
        } else {
            int8_t new_player = current_player[gid];
            if (!has_any_legal_move(game_owner, game_piece, new_player)) {
                outcome[gid] = (new_player == PLAYER_RED) ? OUTCOME_BLUE_WIN : OUTCOME_RED_WIN;
            }
        }
    }

    if (outcome[gid] != OUTCOME_ONGOING) {
        terminated[gid] = true;
    }
}

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
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks == 0) return;
    ATARAXOS_CUDA_LAUNCH(apply_actions_kernel, blocks, threads, stream,
        board_owner, board_piece, actions, current_player,
        move_number, moves_since_attack, outcome, terminated,
        moved_squares, revealed_squares,
        move_history, move_history_head, move_history_len,
        two_square_count, two_square_pair_lo, two_square_pair_hi,
        two_square_violation,
        zobrist_table,
        chase_hashes, chase_hash_head, chase_hash_len,
        chase_last_src, chasing_violation, N);
}

// ---------------------------------------------------------------------------
// legal_action_mask_kernel
// ---------------------------------------------------------------------------

__global__ void legal_action_mask_kernel(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int8_t* current_player,
    bool* mask,
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * NUM_SQUARES;
    if (gid >= total) return;

    int game = gid / NUM_SQUARES;
    int src_idx = gid % NUM_SQUARES;

    const int8_t* game_owner = board_owner + game * NUM_SQUARES;
    const int8_t* game_piece = board_piece + game * NUM_SQUARES;
    int8_t player = current_player[game];
    bool* game_mask = mask + game * NUM_ACTIONS;

    if (game_owner[src_idx] != player) return;
    int8_t pt = game_piece[src_idx];
    if (!piece_can_move(pt)) return;

    int src_r = src_idx / BOARD_COLS;
    int src_c = src_idx % BOARD_COLS;
    bool is_scout = (pt == PT_SCOUT);

    const int drs[4] = {-1, 1, 0, 0};
    const int dcs[4] = {0, 0, -1, 1};

    for (int d = 0; d < 4; ++d) {
        int dr = drs[d];
        int dc = dcs[d];

        if (is_scout) {
            int r = src_r + dr;
            int c = src_c + dc;
            while (r >= 0 && r < BOARD_ROWS && c >= 0 && c < BOARD_COLS) {
                if (is_lake(r, c)) break;
                int idx = r * BOARD_COLS + c;
                int8_t owner = game_owner[idx];
                if (owner == player) break;  // own piece blocks
                // Empty or enemy: legal move
                game_mask[src_idx * NUM_SQUARES + idx] = true;
                if (owner != PLAYER_EMPTY) break;  // enemy: can attack but can't continue
                r += dr;
                c += dc;
            }
        } else {
            int r = src_r + dr;
            int c = src_c + dc;
            if (r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS) continue;
            if (is_lake(r, c)) continue;
            int idx = r * BOARD_COLS + c;
            int8_t owner = game_owner[idx];
            if (owner == player) continue;
            game_mask[src_idx * NUM_SQUARES + idx] = true;
        }
    }
}

void launch_legal_action_mask(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int8_t* current_player,
    bool* mask,
    int N,
    cudaStream_t stream)
{
    int total = N * NUM_SQUARES;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks == 0) return;
    ATARAXOS_CUDA_LAUNCH(legal_action_mask_kernel, blocks, threads, stream,
        board_owner, board_piece, current_player, mask, N);
}

// ---------------------------------------------------------------------------
// reset_kernel
// ---------------------------------------------------------------------------

__global__ void reset_kernel(
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
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    if (!terminated[gid]) return;

    int8_t* game_owner = board_owner + gid * NUM_SQUARES;
    int8_t* game_piece = board_piece + gid * NUM_SQUARES;
    const int8_t* game_red = setup_red + gid * NUM_SQUARES;
    const int8_t* game_blue = setup_blue + gid * NUM_SQUARES;
    bool* game_moved = moved_squares + gid * NUM_SQUARES;
    bool* game_revealed = revealed_squares + gid * NUM_SQUARES;
    int32_t* game_history = move_history + gid * NUM_MOVE_HISTORY;

    for (int sq = 0; sq < NUM_SQUARES; ++sq) {
        int8_t red_piece = game_red[sq];
        int8_t blue_piece = game_blue[sq];
        if (red_piece != PT_NONE) {
            game_owner[sq] = PLAYER_RED;
            game_piece[sq] = red_piece;
        } else if (blue_piece != PT_NONE) {
            game_owner[sq] = PLAYER_BLUE;
            game_piece[sq] = blue_piece;
        } else {
            game_owner[sq] = PLAYER_EMPTY;
            game_piece[sq] = PT_NONE;
        }
        game_moved[sq] = false;
        game_revealed[sq] = false;
    }

    for (int i = 0; i < NUM_MOVE_HISTORY; ++i) {
        game_history[i] = 0;
    }

    current_player[gid] = PLAYER_RED;
    move_number[gid] = 0;
    moves_since_attack[gid] = 0;
    outcome[gid] = OUTCOME_ONGOING;
    terminated[gid] = false;
    move_history_head[gid] = 0;
    move_history_len[gid] = 0;

    // Zero two-square tracking for both players
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        two_square_count[gid * NUM_PLAYERS + p] = 0;
        two_square_pair_lo[gid * NUM_PLAYERS + p] = -1;
        two_square_pair_hi[gid * NUM_PLAYERS + p] = -1;
    }
    two_square_violation[gid] = false;

    // Zero chasing tracking for both players
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        chase_hash_head[gid * NUM_PLAYERS + p] = 0;
        chase_hash_len[gid * NUM_PLAYERS + p] = 0;
        chase_last_src[gid * NUM_PLAYERS + p] = -1;
        // Zero the ring buffer contents
        int64_t* game_hashes =
            chase_hashes + (gid * NUM_PLAYERS + p) * CHASE_HASH_BUFFER_SIZE;
        for (int i = 0; i < CHASE_HASH_BUFFER_SIZE; ++i) {
            game_hashes[i] = 0;
        }
    }
    chasing_violation[gid] = false;
}

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
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks == 0) return;
    ATARAXOS_CUDA_LAUNCH(reset_kernel, blocks, threads, stream,
        board_owner, board_piece, setup_red, setup_blue,
        current_player, move_number, moves_since_attack,
        outcome, terminated,
        moved_squares, revealed_squares,
        move_history, move_history_head, move_history_len,
        two_square_count, two_square_pair_lo, two_square_pair_hi,
        two_square_violation,
        chase_hashes, chase_hash_head, chase_hash_len,
        chase_last_src, chasing_violation, N);
}

// ---------------------------------------------------------------------------
// compute_infostate_kernel
//   One thread per (game, channel, square). Channels 43-455 are zeros (deferred).
//   Perspective: current_player[game] determines "own" vs "opp".
// ---------------------------------------------------------------------------

__global__ void compute_infostate_kernel(
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
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * NUM_INFOSTATE_CHANNELS * NUM_SQUARES;
    if (gid >= total) return;

    int sq = gid % NUM_SQUARES;
    int ch = (gid / NUM_SQUARES) % NUM_INFOSTATE_CHANNELS;
    int game = gid / (NUM_INFOSTATE_CHANNELS * NUM_SQUARES);

    const int8_t* game_owner = board_owner + game * NUM_SQUARES;
    const int8_t* game_piece = board_piece + game * NUM_SQUARES;
    const bool* game_moved = moved_squares + game * NUM_SQUARES;
    const bool* game_revealed = revealed_squares + game * NUM_SQUARES;
    const int32_t* game_history = move_history + game * NUM_MOVE_HISTORY;

    int8_t player = current_player[game];
    int8_t owner = game_owner[sq];
    int8_t piece = game_piece[sq];
    bool is_own = (owner == player);
    bool is_opp = (owner != player && owner != PLAYER_EMPTY);
    bool is_empty = (owner == PLAYER_EMPTY);
    bool is_revealed = game_revealed[sq];
    bool is_moved = game_moved[sq];

    float value = 0.0f;

    if (ch >= 43 && ch <= 455) {
        // Deferred complex tracking channels — remain zero
    } else if (ch < NUM_PIECE_TYPES) {
        // 0-11: own piece type one-hot
        int8_t pt = static_cast<int8_t>(ch + 1);
        if (is_own && piece == pt) value = 1.0f;
    } else if (ch < 2 * NUM_PIECE_TYPES) {
        // 12-23: opp piece type probs (uniform for hidden, one-hot for revealed)
        int8_t pt = static_cast<int8_t>(ch - NUM_PIECE_TYPES + 1);
        if (is_opp) {
            if (is_revealed) {
                if (piece == pt) value = 1.0f;
            } else {
                value = 1.0f / static_cast<float>(NUM_PIECE_TYPES);
            }
        }
    } else if (ch < 3 * NUM_PIECE_TYPES) {
        // 24-35: mirror — opponent's view of own pieces
        int8_t pt = static_cast<int8_t>(ch - 2 * NUM_PIECE_TYPES + 1);
        if (is_own) {
            if (is_revealed) {
                if (piece == pt) value = 1.0f;
            } else {
                value = 1.0f / static_cast<float>(NUM_PIECE_TYPES);
            }
        }
    } else if (ch == 36) {
        // own hidden
        if (is_own && !is_revealed) value = 1.0f;
    } else if (ch == 37) {
        // opp hidden
        if (is_opp && !is_revealed) value = 1.0f;
    } else if (ch == 38) {
        // empty squares
        if (is_empty) value = 1.0f;
    } else if (ch == 39) {
        // own moved pieces
        if (is_own && is_moved) value = 1.0f;
    } else if (ch == 40) {
        // opp moved pieces
        if (is_opp && is_moved) value = 1.0f;
    } else if (ch == 41) {
        // move_number / MAX_GAME_LENGTH
        value = static_cast<float>(move_number[game]) / static_cast<float>(MAX_GAME_LENGTH);
    } else if (ch == 42) {
        // moves_since_attack / TRAINING_NO_ATTACK_LIMIT
        value = static_cast<float>(moves_since_attack[game]) / static_cast<float>(TRAINING_NO_ATTACK_LIMIT);
    } else if (ch >= NUM_BOARD_CHANNELS) {
        // 456-487: last 32 moves (+1 at dst, -1 at src, most recent = 456)
        int hist_idx = ch - NUM_BOARD_CHANNELS;
        int hist_len = move_history_len[game];
        if (hist_idx < hist_len) {
            int head = move_history_head[game];
            int slot = (head - 1 - hist_idx + NUM_MOVE_HISTORY * 2) % NUM_MOVE_HISTORY;
            int32_t action = game_history[slot];
            int src = action / NUM_SQUARES;
            int dst = action % NUM_SQUARES;
            if (sq == dst) value = 1.0f;
            else if (sq == src) value = -1.0f;
        }
    }

    out[game * NUM_INFOSTATE_CHANNELS * NUM_SQUARES + ch * NUM_SQUARES + sq] = value;
}

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
    cudaStream_t stream)
{
    int total = N * NUM_INFOSTATE_CHANNELS * NUM_SQUARES;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks == 0) return;
    ATARAXOS_CUDA_LAUNCH(compute_infostate_kernel, blocks, threads, stream,
        board_owner, board_piece, current_player,
        move_number, moves_since_attack,
        moved_squares, revealed_squares,
        move_history, move_history_head, move_history_len,
        out, N);
}

// ---------------------------------------------------------------------------
// hash_board_kernel
//   One thread per game. Computes Zobrist hash of the current board.
//   hash = XOR of zobrist_table[r, c, owner, piece] for all occupied squares.
// ---------------------------------------------------------------------------

__global__ void hash_board_kernel(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int64_t* zobrist_table,
    int64_t* out,
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    const int8_t* game_owner = board_owner + gid * NUM_SQUARES;
    const int8_t* game_piece = board_piece + gid * NUM_SQUARES;

    int64_t h = 0;
    for (int sq = 0; sq < NUM_SQUARES; ++sq) {
        int8_t ow = game_owner[sq];
        if (ow >= 0) {
            int r = sq / BOARD_COLS;
            int c = sq % BOARD_COLS;
            h ^= zobrist_table[
                ((r * BOARD_COLS + c) * NUM_PLAYERS + ow) *
                NUM_ZOBRIST_PIECES + game_piece[sq]];
        }
    }
    out[gid] = h;
}

void launch_hash_board(
    const int8_t* board_owner,
    const int8_t* board_piece,
    const int64_t* zobrist_table,
    int64_t* out,
    int N,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks == 0) return;
    ATARAXOS_CUDA_LAUNCH(hash_board_kernel, blocks, threads, stream,
        board_owner, board_piece, zobrist_table, out, N);
}

}  // namespace stratego
