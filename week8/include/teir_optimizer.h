#pragma once
#include "teir.h"
#include <memory>
#include <string>
#include <vector>

class TEIROptimizer {
public:
    explicit TEIROptimizer(TEIRProgram& p) : prog(p) {}

    // Implementiert die TEIR Transformationen (Week 8 Task 1)
    void split_iteration(const std::string& parent_axis, const std::string& new_inner_axis, int split_factor);
    void fuse_iteration_nodes(const std::string& ax1, const std::string& ax2, const std::string& fused_ax);
    void reorder_schedule_chain(std::shared_ptr<IterNode> parent, std::vector<std::string> desired_order);
    void set_policy(const std::string& axis, const std::string& policy);
    void promote_to_primitive(const std::string& axis, const std::string& prim_name);

    // High Level Heuristik-Passes
    void expose_parallelism();
    void apply_cache_blocking_matmul();
    void apply_cache_blocking_contraction();
    void apply_cache_blocking_einsum();
    void apply_cache_blocking_transposition();

private:
    TEIRProgram& prog;

    std::shared_ptr<IterNode> find_iter_node(const std::vector<std::shared_ptr<ScheduleNode>>& nodes, const std::string& axis);
    std::shared_ptr<IterNode> find_iter_node(const std::string& axis) { return find_iter_node(prog.roots, axis); }
};
