#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "stratego_buffer.cuh"

namespace stratego {

class StrategoRolloutBuffer {
public:
    explicit StrategoRolloutBuffer(int n_games = 1, int device_id = 0)
        : n_games_(n_games), device_id_(device_id) {
        auto int8_opt = torch::TensorOptions()
                            .dtype(torch::kInt8)
                            .device(torch::kCUDA, device_id_);
        auto int32_opt = torch::TensorOptions()
                             .dtype(torch::kInt32)
                             .device(torch::kCUDA, device_id_);
        auto int64_opt = torch::TensorOptions()
                             .dtype(torch::kInt64)
                             .device(torch::kCUDA, device_id_);
        auto bool_opt = torch::TensorOptions()
                            .dtype(torch::kBool)
                            .device(torch::kCUDA, device_id_);

        board_owner_ = torch::full({n_games, BOARD_ROWS, BOARD_COLS}, -1, int8_opt);
        board_piece_ = torch::zeros({n_games, BOARD_ROWS, BOARD_COLS}, int8_opt);
        current_player_ = torch::zeros({n_games}, int8_opt);
        move_number_ = torch::zeros({n_games}, int32_opt);
        moves_since_attack_ = torch::zeros({n_games}, int32_opt);
        outcome_ = torch::full({n_games}, static_cast<int8_t>(OUTCOME_ONGOING), int8_opt);
        terminated_ = torch::zeros({n_games}, bool_opt);
        flag_captured_ = torch::zeros({n_games}, bool_opt);
        terminated_since_ = torch::zeros({n_games}, int32_opt);
        moved_squares_ = torch::zeros({n_games, NUM_SQUARES}, bool_opt);
        revealed_squares_ = torch::zeros({n_games, NUM_SQUARES}, bool_opt);
        move_history_ = torch::zeros({n_games, NUM_MOVE_HISTORY}, int32_opt);
        move_history_head_ = torch::zeros({n_games}, int32_opt);
        move_history_len_ = torch::zeros({n_games}, int32_opt);

        // --- Two-square rule tracking ---
        two_square_count_ = torch::zeros({n_games, NUM_PLAYERS}, int32_opt);
        two_square_pair_lo_ = torch::full({n_games, NUM_PLAYERS}, -1, int32_opt);
        two_square_pair_hi_ = torch::full({n_games, NUM_PLAYERS}, -1, int32_opt);
        two_square_violation_ = torch::zeros({n_games}, bool_opt);

        // --- Chasing rule tracking ---
        zobrist_table_ = torch::randint(
            0, (1LL << 62),
            {BOARD_ROWS, BOARD_COLS, NUM_PLAYERS, NUM_ZOBRIST_PIECES},
            int64_opt);
        chase_hashes_ = torch::zeros(
            {n_games, NUM_PLAYERS, CHASE_HASH_BUFFER_SIZE}, int64_opt);
        chase_hash_head_ = torch::zeros({n_games, NUM_PLAYERS}, int32_opt);
        chase_hash_len_ = torch::zeros({n_games, NUM_PLAYERS}, int32_opt);
        chase_last_src_ = torch::full({n_games, NUM_PLAYERS}, -1, int32_opt);
        chasing_violation_ = torch::zeros({n_games}, bool_opt);
    }

    // --- Hello-world (kept for build verification) ---
    torch::Tensor hello_world() {
        auto options = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .device(torch::kCUDA, device_id_);
        auto out = torch::zeros({1}, options);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id_).stream();
        launch_hello_world(static_cast<int*>(out.data_ptr()), 1, stream);
        return out;
    }

    // --- apply_actions ---
    void apply_actions(torch::Tensor actions) {
        TORCH_CHECK(actions.dim() == 1, "actions must be 1D (N,)");
        TORCH_CHECK(actions.size(0) == n_games_, "actions size mismatch: expected ",
                    n_games_, ", got ", actions.size(0));
        TORCH_CHECK(actions.dtype() == torch::kInt64, "actions must be int64");
        auto actions_dev = actions.to(torch::Device(torch::kCUDA, device_id_),
                                       torch::kInt64, /*non_blocking=*/false);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id_).stream();

        auto blue_flag_count_before = ((board_owner_ == PLAYER_BLUE) & (board_piece_ == PT_FLAG))
            .view({n_games_, -1}).sum(1).to(torch::kInt32);
        auto red_flag_count_before = ((board_owner_ == PLAYER_RED) & (board_piece_ == PT_FLAG))
            .view({n_games_, -1}).sum(1).to(torch::kInt32);

        launch_apply_actions(
            static_cast<int8_t*>(board_owner_.data_ptr()),
            static_cast<int8_t*>(board_piece_.data_ptr()),
            static_cast<int64_t*>(actions_dev.data_ptr()),
            static_cast<int8_t*>(current_player_.data_ptr()),
            static_cast<int32_t*>(move_number_.data_ptr()),
            static_cast<int32_t*>(moves_since_attack_.data_ptr()),
            static_cast<int8_t*>(outcome_.data_ptr()),
            static_cast<bool*>(terminated_.data_ptr()),
            static_cast<bool*>(moved_squares_.data_ptr()),
            static_cast<bool*>(revealed_squares_.data_ptr()),
            static_cast<int32_t*>(move_history_.data_ptr()),
            static_cast<int32_t*>(move_history_head_.data_ptr()),
            static_cast<int32_t*>(move_history_len_.data_ptr()),
            static_cast<int32_t*>(two_square_count_.data_ptr()),
            static_cast<int32_t*>(two_square_pair_lo_.data_ptr()),
            static_cast<int32_t*>(two_square_pair_hi_.data_ptr()),
            static_cast<bool*>(two_square_violation_.data_ptr()),
            static_cast<int64_t*>(zobrist_table_.data_ptr()),
            static_cast<int64_t*>(chase_hashes_.data_ptr()),
            static_cast<int32_t*>(chase_hash_head_.data_ptr()),
            static_cast<int32_t*>(chase_hash_len_.data_ptr()),
            static_cast<int32_t*>(chase_last_src_.data_ptr()),
            static_cast<bool*>(chasing_violation_.data_ptr()),
            n_games_, stream);

        auto blue_flag_count_after = ((board_owner_ == PLAYER_BLUE) & (board_piece_ == PT_FLAG))
            .view({n_games_, -1}).sum(1).to(torch::kInt32);
        auto red_flag_count_after = ((board_owner_ == PLAYER_RED) & (board_piece_ == PT_FLAG))
            .view({n_games_, -1}).sum(1).to(torch::kInt32);

        auto flag_captured = (blue_flag_count_before > blue_flag_count_after) |
                             (red_flag_count_before > red_flag_count_after);
        flag_captured_ = flag_captured_ | flag_captured;
        terminated_since_ += terminated_.to(torch::kInt32);
    }

    // --- compute_legal_action_mask ---
    torch::Tensor compute_legal_action_mask() {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kBool)
                        .device(torch::kCUDA, device_id_);
        // (N, 100, 100): mask[game, src_idx, dst_idx] = true if move is legal.
        // Flat memory layout matches kernel indexing: game*10000 + src*100 + dst.
        auto mask = torch::zeros({n_games_, NUM_SQUARES, NUM_SQUARES}, opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id_).stream();
        launch_legal_action_mask(
            static_cast<int8_t*>(board_owner_.data_ptr()),
            static_cast<int8_t*>(board_piece_.data_ptr()),
            static_cast<int8_t*>(current_player_.data_ptr()),
            static_cast<bool*>(mask.data_ptr()),
            n_games_, stream);
        return mask;
    }

    // --- compute_reward_pl0 ---
    torch::Tensor compute_reward_pl0() {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA, device_id_);
        auto reward = torch::zeros({n_games_}, opts);
        // RED_WIN (0) → +1, BLUE_WIN (1) → -1, DRAW/ONGOING → 0
        reward.masked_fill_(outcome_ == static_cast<int8_t>(OUTCOME_RED_WIN), 1.0f);
        reward.masked_fill_(outcome_ == static_cast<int8_t>(OUTCOME_BLUE_WIN), -1.0f);
        return reward;
    }

    // --- compute_infostate_tensor ---
    torch::Tensor compute_infostate_tensor() {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA, device_id_);
        auto out = torch::zeros(
            {n_games_, NUM_INFOSTATE_CHANNELS, BOARD_ROWS, BOARD_COLS}, opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id_).stream();
        launch_compute_infostate(
            static_cast<int8_t*>(board_owner_.data_ptr()),
            static_cast<int8_t*>(board_piece_.data_ptr()),
            static_cast<int8_t*>(current_player_.data_ptr()),
            static_cast<int32_t*>(move_number_.data_ptr()),
            static_cast<int32_t*>(moves_since_attack_.data_ptr()),
            static_cast<bool*>(moved_squares_.data_ptr()),
            static_cast<bool*>(revealed_squares_.data_ptr()),
            static_cast<int32_t*>(move_history_.data_ptr()),
            static_cast<int32_t*>(move_history_head_.data_ptr()),
            static_cast<int32_t*>(move_history_len_.data_ptr()),
            static_cast<float*>(out.data_ptr()),
            n_games_, stream);
        return out;
    }

    // --- compute_two_square_rule_applies ---
    torch::Tensor compute_two_square_rule_applies() {
        return two_square_violation_.clone();
    }

    // --- is_chasing_violation ---
    torch::Tensor is_chasing_violation() {
        return chasing_violation_.clone();
    }

    // --- hash_board ---
    torch::Tensor hash_board() {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(torch::kCUDA, device_id_);
        auto out = torch::zeros({n_games_}, opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id_).stream();
        launch_hash_board(
            static_cast<int8_t*>(board_owner_.data_ptr()),
            static_cast<int8_t*>(board_piece_.data_ptr()),
            static_cast<int64_t*>(zobrist_table_.data_ptr()),
            static_cast<int64_t*>(out.data_ptr()),
            n_games_, stream);
        return out;
    }

    // --- Getters ---
    torch::Tensor get_terminated() { return terminated_.clone(); }
    torch::Tensor get_outcomes() { return outcome_.clone(); }
    torch::Tensor get_board_owner() { return board_owner_.clone(); }
    torch::Tensor get_board_piece() { return board_piece_.clone(); }
    torch::Tensor get_current_player() { return current_player_.clone(); }
    torch::Tensor get_move_number() { return move_number_.clone(); }

    torch::Tensor get_num_moves() { return move_number_.clone(); }
    torch::Tensor get_num_moves_since_last_attack() { return moves_since_attack_.clone(); }
    torch::Tensor get_flag_captured() { return flag_captured_.clone(); }
    torch::Tensor get_has_legal_movement() {
        auto mask = compute_legal_action_mask();
        return mask.view({n_games_, -1}).any(-1);
    }
    torch::Tensor get_terminated_since() { return terminated_since_.clone(); }

    std::vector<std::string> board_strs() {
        auto owner_cpu = board_owner_.to(torch::kCPU);
        auto piece_cpu = board_piece_.to(torch::kCPU);
        auto o = owner_cpu.accessor<int8_t, 3>();
        auto p = piece_cpu.accessor<int8_t, 3>();
        std::vector<std::string> result;
        for (int g = 0; g < n_games_; ++g) {
            std::string s;
            for (int r = 0; r < 10; ++r) {
                for (int c = 0; c < 10; ++c) {
                    int8_t ow = o[g][r][c];
                    int8_t pc = p[g][r][c];
                    if (ow < 0) { s += ". "; }
                    else { s += (ow == 0 ? "R" : "B"); s += std::to_string((int)pc); if (pc < 10) s += " "; }
                }
                s += "\n";
            }
            result.push_back(s);
        }
        return result;
    }

    // --- reset_terminated ---
    void reset_terminated(torch::Tensor setup_red, torch::Tensor setup_blue) {
        TORCH_CHECK(setup_red.dim() == 3, "setup_red must be (N, 10, 10)");
        TORCH_CHECK(setup_red.size(0) == n_games_, "setup_red N mismatch");
        TORCH_CHECK(setup_red.size(1) == BOARD_ROWS && setup_red.size(2) == BOARD_COLS,
                    "setup_red must be (N, 10, 10)");
        TORCH_CHECK(setup_red.dtype() == torch::kInt8, "setup_red must be int8");
        TORCH_CHECK(setup_blue.dim() == 3, "setup_blue must be (N, 10, 10)");
        TORCH_CHECK(setup_blue.size(0) == n_games_, "setup_blue N mismatch");
        TORCH_CHECK(setup_blue.size(1) == BOARD_ROWS && setup_blue.size(2) == BOARD_COLS,
                    "setup_blue must be (N, 10, 10)");
        TORCH_CHECK(setup_blue.dtype() == torch::kInt8, "setup_blue must be int8");

        auto red_dev = setup_red.to(torch::Device(torch::kCUDA, device_id_), torch::kInt8, false);
        auto blue_dev = setup_blue.to(torch::Device(torch::kCUDA, device_id_), torch::kInt8, false);

        auto was_terminated = terminated_.clone();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id_).stream();
        launch_reset(
            static_cast<int8_t*>(board_owner_.data_ptr()),
            static_cast<int8_t*>(board_piece_.data_ptr()),
            static_cast<int8_t*>(red_dev.data_ptr()),
            static_cast<int8_t*>(blue_dev.data_ptr()),
            static_cast<int8_t*>(current_player_.data_ptr()),
            static_cast<int32_t*>(move_number_.data_ptr()),
            static_cast<int32_t*>(moves_since_attack_.data_ptr()),
            static_cast<int8_t*>(outcome_.data_ptr()),
            static_cast<bool*>(terminated_.data_ptr()),
            static_cast<bool*>(moved_squares_.data_ptr()),
            static_cast<bool*>(revealed_squares_.data_ptr()),
            static_cast<int32_t*>(move_history_.data_ptr()),
            static_cast<int32_t*>(move_history_head_.data_ptr()),
            static_cast<int32_t*>(move_history_len_.data_ptr()),
            static_cast<int32_t*>(two_square_count_.data_ptr()),
            static_cast<int32_t*>(two_square_pair_lo_.data_ptr()),
            static_cast<int32_t*>(two_square_pair_hi_.data_ptr()),
            static_cast<bool*>(two_square_violation_.data_ptr()),
            static_cast<int64_t*>(chase_hashes_.data_ptr()),
            static_cast<int32_t*>(chase_hash_head_.data_ptr()),
            static_cast<int32_t*>(chase_hash_len_.data_ptr()),
            static_cast<int32_t*>(chase_last_src_.data_ptr()),
            static_cast<bool*>(chasing_violation_.data_ptr()),
            n_games_, stream);

        auto reset_mask = was_terminated & ~terminated_;
        flag_captured_.masked_fill_(reset_mask, false);
        terminated_since_.masked_fill_(reset_mask, 0);
    }

    // --- reset_all (force-reset every game regardless of terminated state) ---
    void reset_all(torch::Tensor setup_red, torch::Tensor setup_blue) {
        terminated_.fill_(true);
        reset_terminated(setup_red, setup_blue);
    }

    // --- current_step ---
    int64_t current_step() {
        return move_number_.sum().item().toLong();
    }

    int n_games() const { return n_games_; }
    int device_id() const { return device_id_; }

private:
    int n_games_;
    int device_id_;
    torch::Tensor board_owner_;
    torch::Tensor board_piece_;
    torch::Tensor current_player_;
    torch::Tensor move_number_;
    torch::Tensor moves_since_attack_;
    torch::Tensor outcome_;
    torch::Tensor terminated_;
    torch::Tensor flag_captured_;
    torch::Tensor terminated_since_;
    torch::Tensor moved_squares_;
    torch::Tensor revealed_squares_;
    torch::Tensor move_history_;
    torch::Tensor move_history_head_;
    torch::Tensor move_history_len_;
    // Two-square rule tracking
    torch::Tensor two_square_count_;
    torch::Tensor two_square_pair_lo_;
    torch::Tensor two_square_pair_hi_;
    torch::Tensor two_square_violation_;
    // Chasing rule tracking
    torch::Tensor zobrist_table_;
    torch::Tensor chase_hashes_;
    torch::Tensor chase_hash_head_;
    torch::Tensor chase_hash_len_;
    torch::Tensor chase_last_src_;
    torch::Tensor chasing_violation_;
};

}  // namespace stratego

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Stratego CUDA acceleration — rollout buffer and game kernels";

    pybind11::class_<stratego::StrategoRolloutBuffer>(m, "StrategoRolloutBuffer")
        .def(pybind11::init<int, int>(),
             pybind11::arg("n_games") = 1,
             pybind11::arg("device_id") = 0)
        .def("hello_world", &stratego::StrategoRolloutBuffer::hello_world,
             "Run hello-world CUDA kernel, returns tensor with value 42")
        .def("apply_actions", &stratego::StrategoRolloutBuffer::apply_actions,
             pybind11::arg("actions"),
             "Apply actions (N,) int64 to all games in parallel")
        .def("compute_legal_action_mask",
             &stratego::StrategoRolloutBuffer::compute_legal_action_mask,
             "Returns (N, 100, 100) bool tensor of legal move mask")
        .def("compute_reward_pl0",
             &stratego::StrategoRolloutBuffer::compute_reward_pl0,
             "Returns (N,) float32 reward for player 0 (Red)")
        .def("compute_infostate_tensor",
             &stratego::StrategoRolloutBuffer::compute_infostate_tensor,
             "Returns (N, 488, 10, 10) float32 infostate tensor from current player perspective")
        .def("compute_two_square_rule_applies",
             &stratego::StrategoRolloutBuffer::compute_two_square_rule_applies,
             "Returns (N,) bool tensor — true if two-square rule violation occurred")
        .def("is_chasing_violation",
             &stratego::StrategoRolloutBuffer::is_chasing_violation,
             "Returns (N,) bool tensor — true if chasing rule violation occurred")
        .def("hash_board",
             &stratego::StrategoRolloutBuffer::hash_board,
             "Returns (N,) int64 tensor of Zobrist hashes for current board state")
        .def("get_terminated", &stratego::StrategoRolloutBuffer::get_terminated,
             "Returns (N,) bool tensor of terminated flags")
        .def("get_outcomes", &stratego::StrategoRolloutBuffer::get_outcomes,
             "Returns (N,) int8 tensor of game outcomes")
        .def("get_board_owner", &stratego::StrategoRolloutBuffer::get_board_owner,
             "Returns (N, 10, 10) int8 tensor of board owners")
        .def("get_board_piece", &stratego::StrategoRolloutBuffer::get_board_piece,
             "Returns (N, 10, 10) int8 tensor of board pieces")
        .def("get_current_player",
             &stratego::StrategoRolloutBuffer::get_current_player,
             "Returns (N,) int8 tensor of current players")
        .def("get_move_number", &stratego::StrategoRolloutBuffer::get_move_number,
             "Returns (N,) int32 tensor of move numbers")
        .def("get_num_moves", &stratego::StrategoRolloutBuffer::get_num_moves,
             "Returns (N,) int32 tensor of move numbers")
        .def("get_num_moves_since_last_attack",
             &stratego::StrategoRolloutBuffer::get_num_moves_since_last_attack,
             "Returns (N,) int32 tensor of moves since last attack")
        .def("get_flag_captured", &stratego::StrategoRolloutBuffer::get_flag_captured,
             "Returns (N,) bool tensor of flag-captured flags")
        .def("get_has_legal_movement",
             &stratego::StrategoRolloutBuffer::get_has_legal_movement,
             "Returns (N,) bool tensor — true if current player has any legal move")
        .def("get_terminated_since",
             &stratego::StrategoRolloutBuffer::get_terminated_since,
             "Returns (N,) int32 tensor of steps since termination")
        .def("board_strs", &stratego::StrategoRolloutBuffer::board_strs,
             "Returns list of string representations, one per game")
        .def("reset_terminated", &stratego::StrategoRolloutBuffer::reset_terminated,
             pybind11::arg("setup_red"), pybind11::arg("setup_blue"),
             "Reset terminated games with new setups")
        .def("reset_all", &stratego::StrategoRolloutBuffer::reset_all,
             pybind11::arg("setup_red"), pybind11::arg("setup_blue"),
             "Force-reset all games with new setups")
        .def("current_step", &stratego::StrategoRolloutBuffer::current_step,
             "Returns total steps across all games")
        .def_property_readonly("device_id",
             &stratego::StrategoRolloutBuffer::device_id)
        .def_property_readonly("n_games",
             &stratego::StrategoRolloutBuffer::n_games);
}
