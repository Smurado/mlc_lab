#include "include/teir_optimizer.h"
#include <iostream>
#include <algorithm>

std::shared_ptr<IterNode> TEIROptimizer::find_iter_node(const std::vector<std::shared_ptr<ScheduleNode>>& nodes, const std::string& axis) {
    for (auto& n : nodes) {
        if (n->type == NodeType::Iter) {
            auto inode = std::static_pointer_cast<IterNode>(n);
            if (inode->axis == axis || inode->axis == "@" + axis) return inode;
            auto child_res = find_iter_node(inode->children, axis);
            if (child_res) return child_res;
        }
    }
    return nullptr;
}

void TEIROptimizer::split_iteration(const std::string& parent_axis, const std::string& new_inner_axis, int split_factor) {
    auto node = find_iter_node(parent_axis);
    if (!node) return;
    
    std::string raw_parent = (parent_axis[0] == '@') ? parent_axis.substr(1) : parent_axis;
    std::string raw_inner = (new_inner_axis[0] == '@') ? new_inner_axis.substr(1) : new_inner_axis;
    
    int original_extent = prog.axes[raw_parent].extent;
    prog.axes[raw_parent].extent = original_extent / split_factor;
    
    Axis inner_ax = prog.axes[raw_parent]; 
    inner_ax.name = raw_inner;
    inner_ax.extent = split_factor;
    for (auto& [tname, val] : prog.axes[raw_parent].strides) {
        val *= split_factor;
    }
    prog.axes[raw_inner] = inner_ax;
    
    auto inner_node = std::make_shared<IterNode>();
    inner_node->axis = "@" + raw_inner;
    inner_node->name = "iter_" + raw_inner;
    inner_node->policy = "sequential";
    
    inner_node->children = std::move(node->children);
    node->children.clear();
    node->children.push_back(inner_node);
}

void TEIROptimizer::fuse_iteration_nodes(const std::string& ax1, const std::string& ax2, const std::string& fused_ax) {
}

void TEIROptimizer::reorder_schedule_chain(std::shared_ptr<IterNode> parent, std::vector<std::string> desired_order) {
}

void TEIROptimizer::set_policy(const std::string& axis, const std::string& policy) {
    auto node = find_iter_node(axis);
    if (node) node->policy = policy;
}

void TEIROptimizer::promote_to_primitive(const std::string& axis, const std::string& prim_name) {
}

void TEIROptimizer::apply_cache_blocking_matmul() {
    split_iteration("m0", "m_in", 4);
    split_iteration("n0", "n_in", 16);
            
    auto k0 = find_iter_node("k0");
    auto m0 = find_iter_node("m0");
    auto n0 = find_iter_node("n0");
    auto m_in = find_iter_node("m_in");
    auto n_in = find_iter_node("n_in");
            
    if (!k0 || !m0 || !n0 || !m_in || !n_in) return;
            
    auto inv_zero = k0->children[0];
    auto inv_gemm = n_in->children[0];

    prog.roots = { m0 };
    m0->children = { n0 };
    n0->children = { k0 };
    k0->children = { m_in };
    m_in->children = { n_in };
    n_in->children = { inv_zero, inv_gemm }; 

    auto invoke_z = std::static_pointer_cast<InvokeNode>(inv_zero);
    invoke_z->guard = "first(@k0)"; 
}

void TEIROptimizer::expose_parallelism() {
    for (auto root : prog.roots) {
        if (root->type == NodeType::Iter) {
            if (prog.name == "matmul" || prog.name == "matmul_small") {
                set_policy("m0", "parallel");
            } else if (prog.name == "contraction") {
                set_policy("p", "parallel");
                set_policy("r", "parallel");
            } else if (prog.name == "einsum") {
                set_policy("a", "parallel");
                set_policy("b", "parallel");
                set_policy("c", "parallel");
            } else if (prog.name == "transposition") {
                set_policy("b", "parallel");
            } else {
                root->name = "parallel"; 
            }
        }
    }
}


void TEIROptimizer::apply_cache_blocking_contraction() {
    split_iteration("p", "p_in", 4);
    split_iteration("r", "r_in", 4);
    
    auto p0 = find_iter_node("p");
    auto r0 = find_iter_node("r");
    auto t0 = find_iter_node("t");
    auto p_in = find_iter_node("p_in");
    auto r_in = find_iter_node("r_in");
    
    if (!p0 || !r0 || !t0 || !p_in || !r_in) return;
    
    auto inv_zero = t0->children[0];
    auto inv_gemm = t0->children[1];
    
    prog.roots = { p0 };
    p0->children = { r0 };
    r0->children = { t0 };
    t0->children = { p_in };
    p_in->children = { r_in };
    r_in->children = { inv_zero, inv_gemm };
}

void TEIROptimizer::apply_cache_blocking_einsum() {
    split_iteration("a", "a_in", 2);
    split_iteration("b", "b_in", 2);
    
    auto a0 = find_iter_node("a");
    auto a_in = find_iter_node("a_in");
    auto b0 = find_iter_node("b");
    auto b_in = find_iter_node("b_in");
    auto c0 = find_iter_node("c");
    auto s0 = find_iter_node("s");
    
    if (!a0 || !b0 || !c0 || !a_in || !b_in || !s0) return;
    
    auto inv_zero = s0->children[0];
    auto inv_gemm = s0->children[1];
    
    prog.roots = { a0 };
    a0->children = { b0 };
    b0->children = { c0 };
    c0->children = { s0 };
    s0->children = { a_in };
    a_in->children = { b_in };
    
    b_in->children = { inv_zero, inv_gemm };
}

void TEIROptimizer::apply_cache_blocking_transposition() {
    split_iteration("a", "a_in", 4);
    split_iteration("b", "b_in", 4);

    auto a0 = find_iter_node("a");
    auto b0 = find_iter_node("b");
    auto a_in = find_iter_node("a_in");
    auto b_in = find_iter_node("b_in");

    if (!a0 || !b0 || !a_in || !b_in) return;

    auto inv_copy = b_in->children[0];

    prog.roots = { a0 };
    a0->children = { b0 };
    b0->children = { a_in };
    a_in->children = { b_in };
    b_in->children = { inv_copy };
}

