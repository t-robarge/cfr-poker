#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct NodeState {
    std::vector<double> regrets;
    std::vector<double> strategy_sums;
};

struct SolverContext {
    int combo_count;
    int node_count;
    int root_index;
    std::vector<int> current_player;
    std::vector<int> terminal_type;
    std::vector<int> folded_player;
    std::vector<double> pot;
    std::vector<double> contrib0;
    std::vector<double> contrib1;
    std::vector<int> num_actions;
    std::vector<int> action_codes;
    std::vector<int> child_index;
    int max_actions;
    std::vector<int64_t> hand_strengths;
    std::vector<uint64_t> combo_masks;
    std::vector<int8_t> pair_result;
    std::vector<NodeState> nodes;
    double reward_unit;
};

auto current_strategy(const std::vector<double>& regrets, int combo_count, int num_actions)
    -> std::vector<double> {
    std::vector<double> strategy(combo_count * num_actions, 0.0);
    if (num_actions <= 0) {
        return strategy;
    }
    const double uniform = 1.0 / static_cast<double>(num_actions);
    for (int combo = 0; combo < combo_count; ++combo) {
        double positive_sum = 0.0;
        for (int action = 0; action < num_actions; ++action) {
            const double value = std::max(0.0, regrets[combo * num_actions + action]);
            strategy[combo * num_actions + action] = value;
            positive_sum += value;
        }
        if (positive_sum > 0.0) {
            for (int action = 0; action < num_actions; ++action) {
                strategy[combo * num_actions + action] /= positive_sum;
            }
        } else {
            for (int action = 0; action < num_actions; ++action) {
                strategy[combo * num_actions + action] = uniform;
            }
        }
    }
    return strategy;
}

auto average_strategy(
    const std::vector<double>& strategy_sums,
    const std::vector<double>& fallback,
    int combo_count,
    int num_actions
) -> std::vector<double> {
    std::vector<double> average(combo_count * num_actions, 0.0);
    for (int combo = 0; combo < combo_count; ++combo) {
        double row_sum = 0.0;
        for (int action = 0; action < num_actions; ++action) {
            row_sum += strategy_sums[combo * num_actions + action];
        }
        if (row_sum > 0.0) {
            for (int action = 0; action < num_actions; ++action) {
                average[combo * num_actions + action] = strategy_sums[combo * num_actions + action] / row_sum;
            }
        } else {
            for (int action = 0; action < num_actions; ++action) {
                average[combo * num_actions + action] = fallback[combo * num_actions + action];
            }
        }
    }
    return average;
}

void apply_dcfr_discount(SolverContext& ctx, int iteration) {
    const double alpha = static_cast<double>(iteration) / (static_cast<double>(iteration) + 1.0);
    for (int node = 0; node < ctx.node_count; ++node) {
        auto& state = ctx.nodes[node];
        for (double& value : state.regrets) {
            value *= alpha;
        }
        for (double& value : state.strategy_sums) {
            value *= alpha;
        }
    }
}

auto terminal_values(
    const SolverContext& ctx,
    int node_index,
    int traverser,
    const std::vector<double>& reach0,
    const std::vector<double>& reach1
) -> std::vector<double> {
    std::vector<double> result(ctx.combo_count, 0.0);
    if (ctx.terminal_type[node_index] == 0) {
        const int folded = ctx.folded_player[node_index];
        const double p0_share = folded == 0 ? 0.0 : 1.0;
        const double payoff_p0 =
            (ctx.pot[node_index] * p0_share - ctx.contrib0[node_index]) / ctx.reward_unit;
        if (traverser == 0) {
            for (int combo = 0; combo < ctx.combo_count; ++combo) {
                double total = 0.0;
                for (int opponent = 0; opponent < ctx.combo_count; ++opponent) {
                    if (ctx.pair_result[combo * ctx.combo_count + opponent] == 2) {
                        continue;
                    }
                    total += reach1[opponent];
                }
                result[combo] = payoff_p0 * total;
            }
        } else {
            for (int combo = 0; combo < ctx.combo_count; ++combo) {
                double total = 0.0;
                for (int opponent = 0; opponent < ctx.combo_count; ++opponent) {
                    if (ctx.pair_result[opponent * ctx.combo_count + combo] == 2) {
                        continue;
                    }
                    total += reach0[opponent];
                }
                result[combo] = -payoff_p0 * total;
            }
        }
        return result;
    }

    const double base = -ctx.contrib0[node_index] / ctx.reward_unit;
    const double scale = ctx.pot[node_index] / ctx.reward_unit;
    if (traverser == 0) {
        for (int combo = 0; combo < ctx.combo_count; ++combo) {
            double value = 0.0;
            for (int opponent = 0; opponent < ctx.combo_count; ++opponent) {
                const int8_t relation = ctx.pair_result[combo * ctx.combo_count + opponent];
                if (relation == 2) {
                    continue;
                }
                double share = 0.0;
                if (relation > 0) {
                    share = 1.0;
                } else if (relation == 0) {
                    share = 0.5;
                }
                value += reach1[opponent] * (base + scale * share);
            }
            result[combo] = value;
        }
    } else {
        for (int combo = 0; combo < ctx.combo_count; ++combo) {
            double value = 0.0;
            for (int opponent = 0; opponent < ctx.combo_count; ++opponent) {
                const int8_t relation = ctx.pair_result[opponent * ctx.combo_count + combo];
                if (relation == 2) {
                    continue;
                }
                double share = 0.0;
                if (relation > 0) {
                    share = 1.0;
                } else if (relation == 0) {
                    share = 0.5;
                }
                value += reach0[opponent] * -(base + scale * share);
            }
            result[combo] = value;
        }
    }
    return result;
}

auto cfr(
    SolverContext& ctx,
    int node_index,
    int traverser,
    const std::vector<double>& reach0,
    const std::vector<double>& reach1,
    const std::string& algorithm,
    double average_weight
) -> std::vector<double> {
    if (ctx.terminal_type[node_index] != -1) {
        return terminal_values(ctx, node_index, traverser, reach0, reach1);
    }

    const int actor = ctx.current_player[node_index];
    const int actions = ctx.num_actions[node_index];
    auto& node_state = ctx.nodes[node_index];
    std::vector<double> strategy = current_strategy(node_state.regrets, ctx.combo_count, actions);
    const std::vector<double>& actor_reach = actor == 0 ? reach0 : reach1;
    for (int combo = 0; combo < ctx.combo_count; ++combo) {
        for (int action = 0; action < actions; ++action) {
            node_state.strategy_sums[combo * actions + action] +=
                average_weight * actor_reach[combo] * strategy[combo * actions + action];
        }
    }

    std::vector<std::vector<double>> child_values;
    child_values.reserve(actions);
    for (int action = 0; action < actions; ++action) {
        std::vector<double> next_reach0 = reach0;
        std::vector<double> next_reach1 = reach1;
        if (actor == 0) {
            for (int combo = 0; combo < ctx.combo_count; ++combo) {
                next_reach0[combo] *= strategy[combo * actions + action];
            }
        } else {
            for (int combo = 0; combo < ctx.combo_count; ++combo) {
                next_reach1[combo] *= strategy[combo * actions + action];
            }
        }
        child_values.push_back(
            cfr(
                ctx,
                ctx.child_index[node_index * ctx.max_actions + action],
                traverser,
                next_reach0,
                next_reach1,
                algorithm,
                average_weight
            )
        );
    }

    std::vector<double> node_values(ctx.combo_count, 0.0);
    if (actor != traverser) {
        for (int action = 0; action < actions; ++action) {
            for (int combo = 0; combo < ctx.combo_count; ++combo) {
                node_values[combo] += child_values[action][combo];
            }
        }
        return node_values;
    }

    for (int combo = 0; combo < ctx.combo_count; ++combo) {
        double value = 0.0;
        for (int action = 0; action < actions; ++action) {
            value += strategy[combo * actions + action] * child_values[action][combo];
        }
        node_values[combo] = value;
    }
    for (int combo = 0; combo < ctx.combo_count; ++combo) {
        for (int action = 0; action < actions; ++action) {
            double updated =
                node_state.regrets[combo * actions + action] + child_values[action][combo] - node_values[combo];
            if (algorithm == "cfr_plus") {
                updated = std::max(0.0, updated);
            }
            node_state.regrets[combo * actions + action] = updated;
        }
    }
    return node_values;
}

auto to_vector_i32(const py::array_t<int, py::array::c_style | py::array::forcecast>& array)
    -> std::vector<int> {
    auto info = array.request();
    const auto* data = static_cast<const int*>(info.ptr);
    return std::vector<int>(data, data + info.size);
}

auto to_vector_f64(const py::array_t<double, py::array::c_style | py::array::forcecast>& array)
    -> std::vector<double> {
    auto info = array.request();
    const auto* data = static_cast<const double*>(info.ptr);
    return std::vector<double>(data, data + info.size);
}

auto to_vector_i64(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& array)
    -> std::vector<int64_t> {
    auto info = array.request();
    const auto* data = static_cast<const int64_t*>(info.ptr);
    return std::vector<int64_t>(data, data + info.size);
}

auto solve_river_root(
    py::dict tree_payload,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> combo_cards,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> hand_strengths,
    py::array_t<double, py::array::c_style | py::array::forcecast> range0,
    py::array_t<double, py::array::c_style | py::array::forcecast> range1,
    int iterations,
    const std::string& algorithm
) -> py::dict {
    auto combo_info = combo_cards.request();
    if (combo_info.ndim != 2 || combo_info.shape[1] != 2) {
        throw std::runtime_error("combo_cards must be a contiguous [num_combos, 2] uint8 array.");
    }
    const int combo_count = static_cast<int>(combo_info.shape[0]);

    SolverContext ctx;
    ctx.combo_count = combo_count;
    ctx.root_index = tree_payload["root_index"].cast<int>();
    ctx.current_player = to_vector_i32(tree_payload["current_player"].cast<py::array_t<int>>());
    ctx.terminal_type = to_vector_i32(tree_payload["terminal_type"].cast<py::array_t<int>>());
    ctx.folded_player = to_vector_i32(tree_payload["folded_player"].cast<py::array_t<int>>());
    ctx.pot = to_vector_f64(tree_payload["pot"].cast<py::array_t<double>>());
    ctx.contrib0 = to_vector_f64(tree_payload["contrib0"].cast<py::array_t<double>>());
    ctx.contrib1 = to_vector_f64(tree_payload["contrib1"].cast<py::array_t<double>>());
    ctx.num_actions = to_vector_i32(tree_payload["num_actions"].cast<py::array_t<int>>());
    ctx.action_codes = to_vector_i32(tree_payload["action_codes"].cast<py::array_t<int>>());
    ctx.child_index = to_vector_i32(tree_payload["child_index"].cast<py::array_t<int>>());
    ctx.hand_strengths = to_vector_i64(hand_strengths);
    ctx.node_count = static_cast<int>(ctx.current_player.size());
    ctx.max_actions = tree_payload["action_codes"].cast<py::array_t<int>>().request().shape[1];
    ctx.reward_unit = tree_payload["reward_unit"].cast<double>();

    if (static_cast<int>(ctx.pot.size()) != ctx.node_count || static_cast<int>(ctx.hand_strengths.size()) != combo_count) {
        throw std::runtime_error("Inconsistent native river payload shapes.");
    }

    auto combo_ptr = static_cast<const uint8_t*>(combo_info.ptr);
    ctx.combo_masks.resize(combo_count);
    for (int combo = 0; combo < combo_count; ++combo) {
        const uint8_t left = combo_ptr[combo * 2];
        const uint8_t right = combo_ptr[combo * 2 + 1];
        ctx.combo_masks[combo] = (uint64_t{1} << left) | (uint64_t{1} << right);
    }
    ctx.pair_result.assign(combo_count * combo_count, 2);
    for (int left = 0; left < combo_count; ++left) {
        for (int right = 0; right < combo_count; ++right) {
            if (ctx.combo_masks[left] & ctx.combo_masks[right]) {
                continue;
            }
            if (ctx.hand_strengths[left] > ctx.hand_strengths[right]) {
                ctx.pair_result[left * combo_count + right] = 1;
            } else if (ctx.hand_strengths[left] < ctx.hand_strengths[right]) {
                ctx.pair_result[left * combo_count + right] = -1;
            } else {
                ctx.pair_result[left * combo_count + right] = 0;
            }
        }
    }

    ctx.nodes.resize(ctx.node_count);
    for (int node = 0; node < ctx.node_count; ++node) {
        const int actions = ctx.num_actions[node];
        ctx.nodes[node].regrets.assign(combo_count * actions, 0.0);
        ctx.nodes[node].strategy_sums.assign(combo_count * actions, 0.0);
    }

    std::vector<double> root_reach0 = to_vector_f64(range0);
    std::vector<double> root_reach1 = to_vector_f64(range1);
    if (static_cast<int>(root_reach0.size()) != combo_count || static_cast<int>(root_reach1.size()) != combo_count) {
        throw std::runtime_error("range arrays must match combo_cards length.");
    }

    for (int iteration = 1; iteration <= std::max(1, iterations); ++iteration) {
        if (algorithm == "dcfr") {
            apply_dcfr_discount(ctx, iteration);
        }
        const double average_weight = algorithm == "dcfr" ? 1.0 : static_cast<double>(iteration);
        for (int traverser = 0; traverser < 2; ++traverser) {
            cfr(ctx, ctx.root_index, traverser, root_reach0, root_reach1, algorithm, average_weight);
        }
    }

    const int root_actions = ctx.num_actions[ctx.root_index];
    std::vector<double> current = current_strategy(
        ctx.nodes[ctx.root_index].regrets,
        combo_count,
        root_actions
    );
    std::vector<double> average = average_strategy(
        ctx.nodes[ctx.root_index].strategy_sums,
        current,
        combo_count,
        root_actions
    );

    py::array_t<double> average_out({combo_count, root_actions});
    py::array_t<double> current_out({combo_count, root_actions});
    std::copy(average.begin(), average.end(), static_cast<double*>(average_out.request().ptr));
    std::copy(current.begin(), current.end(), static_cast<double*>(current_out.request().ptr));

    py::dict result;
    result["average_strategy"] = average_out;
    result["current_strategy"] = current_out;
    return result;
}

}  // namespace

PYBIND11_MODULE(_native_river, module) {
    module.doc() = "Native exact river resolver for hulhe_bot.";
    module.def("solve_river_root", &solve_river_root, py::arg("tree_payload"), py::arg("combo_cards"),
               py::arg("hand_strengths"), py::arg("range0"), py::arg("range1"), py::arg("iterations"),
               py::arg("algorithm"));
}
