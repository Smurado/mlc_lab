import re

with open("week8/teir_optimizer.cpp", "r") as f:
    content = f.read()

heuristics = """
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
"""

content = re.sub(r'void TEIROptimizer::apply_cache_blocking_contraction\(\) \{\n\}\nvoid TEIROptimizer::apply_cache_blocking_einsum\(\) \{\n\}\nvoid TEIROptimizer::apply_cache_blocking_transposition\(\) \{\n\}', heuristics, content)

with open("week8/teir_optimizer.cpp", "w") as f:
    f.write(content)

