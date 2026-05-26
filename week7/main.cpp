#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <dlfcn.h>
#include <cassert>
#include <chrono>
#include <omp.h>

// ============================================================================
// 1. TEIR Abstract Syntax Tree (AST) & Representation
// ============================================================================

struct Axis {
    std::string name;
    int extent;
    std::unordered_map<std::string, int> strides; 
};

struct Primitive {
    std::string name;
    std::string kind; 
    std::unordered_map<std::string, std::vector<std::string>> axes_map;
};

enum class NodeType { Iter, Invoke };

struct ScheduleNode {
    NodeType type;
    std::string name;
    virtual ~ScheduleNode() = default;
};

struct InvokeNode : public ScheduleNode {
    std::string primitive;
    std::string guard; 
    InvokeNode() { type = NodeType::Invoke; }
};

struct IterNode : public ScheduleNode {
    std::string axis;
    std::string policy; 
    std::vector<std::shared_ptr<ScheduleNode>> children;
    IterNode() { type = NodeType::Iter; }
};

struct TEIRProgram {
    std::string name;
    std::vector<std::string> tensors; 
    std::unordered_map<std::string, Axis> axes;
    std::unordered_map<std::string, Primitive> primitives;
    std::vector<std::shared_ptr<ScheduleNode>> roots;
};

using TEIRKernelPtr = void(*)(void**);

struct BenchmarkResult {
    double avg_ms;
    double min_ms;
    double median_ms;
};

static BenchmarkResult benchmark_kernel(TEIRKernelPtr kernel, void** args, int iterations,
                                       const std::function<void()>& reset) {
    if (!kernel || iterations <= 0) return {0.0, 0.0, 0.0};

    if (reset) reset();
    kernel(args); 

    std::vector<double> run_times;
    run_times.reserve(iterations);
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::infinity();

    for (int i = 0; i < iterations; ++i) {
        if (reset) reset();
        auto it_start = std::chrono::high_resolution_clock::now();
        kernel(args);
        auto it_end = std::chrono::high_resolution_clock::now();
        double run_ms = std::chrono::duration<double, std::milli>(it_end - it_start).count();
        run_times.push_back(run_ms);
        total_ms += run_ms;
        min_ms = std::min(min_ms, run_ms);
    }

    std::sort(run_times.begin(), run_times.end());
    double median_ms = run_times[iterations / 2];
    if (iterations % 2 == 0) {
        median_ms = (run_times[iterations / 2 - 1] + run_times[iterations / 2]) * 0.5;
    }

    return { total_ms / iterations, min_ms, median_ms };
}

// ============================================================================
// 2. Just-In-Time Code Generator & Compiler
// ============================================================================

class TEIRCompiler {
private:
    TEIRProgram prog;
    std::stringstream src;
    int indent_level = 0;

    void emit(const std::string& line) {
        src << std::string(indent_level * 4, ' ') << line << "\n";
    }

    std::string get_offset_expr(const std::string& tensor_name, 
                                const std::unordered_map<std::string, std::string>& active_loops) {
        std::vector<std::string> terms;
        for (auto const& [ax_name, loop_var] : active_loops) {
            auto it = prog.axes.find(ax_name);
            if (it != prog.axes.end()) {
                auto stride_it = it->second.strides.find(tensor_name);
                if (stride_it != it->second.strides.end() && stride_it->second != 0) {
                    terms.push_back("((long long)" + loop_var + " * " + std::to_string(stride_it->second) + ")");
                }
            }
        }
        if (terms.empty()) return "0";
        std::string expr = terms[0];
        for (size_t i = 1; i < terms.size(); ++i) {
            expr += " + " + terms[i];
        }
        return expr;
    }

    void emit_kernel_body(const Primitive& prim, const std::unordered_map<std::string, std::string>& all_loops) {
        if (prim.kind == "Zero") {
            std::string out_offset = get_offset_expr("out", all_loops);
            emit("*(float*)((char*)out + " + out_offset + ") = 0.0f;");
        } 
        else if (prim.kind == "Contraction") {
            std::string in0_offset = get_offset_expr("in0", all_loops);
            std::string in1_offset = get_offset_expr("in1", all_loops);
            std::string out_offset = get_offset_expr("out", all_loops);
            emit("*(float*)((char*)out + " + out_offset + ") += "
                 "(*(float*)((char*)in0 + " + in0_offset + ")) * (*(float*)((char*)in1 + " + in1_offset + "));");
        } 
        else if (prim.kind == "Copy") {
            std::string in_offset = get_offset_expr("in", all_loops);
            std::string out_offset = get_offset_expr("out", all_loops);
            emit("*(float*)((char*)out + " + out_offset + ") = *(float*)((char*)in + " + in_offset + ");");
        }
    }

    void lower_primitive(const Primitive& prim, std::unordered_map<std::string, std::string> active_loops) {
        std::vector<std::string> prim_axes;
        for (auto const& [key, list] : prim.axes_map) {
            for (auto const& ax : list) {
                std::string clean_ax = (ax[0] == '@') ? ax.substr(1) : ax;
                if (std::find(prim_axes.begin(), prim_axes.end(), clean_ax) == prim_axes.end()) {
                    prim_axes.push_back(clean_ax);
                }
            }
        }

        auto emit_inner_loops = [&](auto& self, size_t depth) -> void {
            if (depth == prim_axes.size()) {
                emit_kernel_body(prim, active_loops);
                return;
            }
            std::string ax_name = prim_axes[depth];
            
            // Falls die Achse bereits durch eine äußere Schleife im Schedule kontrolliert wird, 
            // dürfen wir hier KEINE neue Schleife generieren! Sonst gibt es Race Conditions.
            if (active_loops.find(ax_name) != active_loops.end()) {
                self(self, depth + 1);
            } else {
                int extent = prog.axes[ax_name].extent;
                std::string loop_var = "i_" + ax_name;
                active_loops[ax_name] = loop_var;

                emit("for (int " + loop_var + " = 0; " + loop_var + " < " + std::to_string(extent) + "; ++" + loop_var + ") {");
                indent_level++;
                self(self, depth + 1);
                indent_level--;
                emit("}");
            }
        };

        emit_inner_loops(emit_inner_loops, 0);
    }

void lower_node(const std::shared_ptr<ScheduleNode>& node, std::unordered_map<std::string, std::string> active_loops) {
        if (node->type == NodeType::Iter) {
            auto iter = std::static_pointer_cast<IterNode>(node);
            Axis axis_info = prog.axes[iter->axis];
            std::string loop_var = "i_" + axis_info.name;

            // FIX: Sämtlichen fehlerhaften Pseudocode entfernt
            if (iter->policy == "parallel") {
                emit("#pragma omp parallel for");
            }
            emit("for (int " + loop_var + " = 0; " + loop_var + " < " + std::to_string(axis_info.extent) + "; ++" + loop_var + ") {");
            indent_level++;

            active_loops[axis_info.name] = loop_var;
            for (auto const& child : iter->children) {
                lower_node(child, active_loops);
            }

            indent_level--;
            emit("}");
        } 
        else if (node->type == NodeType::Invoke) {
            auto inv = std::static_pointer_cast<InvokeNode>(node);
            bool has_guard = false;

            if (!inv->guard.empty() && inv->guard.find("first") != std::string::npos) {
                size_t start = inv->guard.find("@") + 1;
                size_t end = inv->guard.find(")");
                std::string g_axis = inv->guard.substr(start, end - start);
                emit("if (i_" + g_axis + " == 0) {");
                indent_level++;
                has_guard = true;
            }

            lower_primitive(prog.primitives[inv->primitive], active_loops);

            if (has_guard) {
                indent_level--;
                emit("}");
            }
        }
    }

public:
    explicit TEIRCompiler(TEIRProgram p) : prog(std::move(p)) {}

    std::string generate_cpp_source() {
        src.str(""); src.clear();
        emit("#include <omp.h>");
        emit("#include <iostream>");
        emit("\nextern \"C\" {");
        
        std::string sig = "void kernel_" + prog.name + "(void** args) {";
        emit(sig);
        indent_level++;

        for (size_t i = 0; i < prog.tensors.size(); ++i) {
            emit("float* " + prog.tensors[i] + " = (float*)args[" + std::to_string(i) + "];");
        }

        std::unordered_map<std::string, std::string> active_loops;
        for (auto const& root : prog.roots) {
            lower_node(root, active_loops);
        }

        indent_level--;
        emit("}");
        emit("}");
        return src.str();
    }

    TEIRKernelPtr compile() {
        std::string source = generate_cpp_source();
        std::string cpp_filename = "teir_jit_" + prog.name + ".cpp";
        std::string so_filename = "./teir_jit_" + prog.name + ".so";

        std::ofstream out(cpp_filename);
        out << source;
        out.close();

        // Dein clang++ Aufruf
        std::string cmd = "clang++ -O3 -Xpreprocessor -fopenmp -shared -fPIC "
                          "-I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp "
                          + cpp_filename + " -o " + so_filename;
                          
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "Compilation failed for program: " << prog.name << "\n";
            return nullptr;
        }

        void* handle = dlopen(so_filename.c_str(), RTLD_NOW);
        if (!handle) {
            std::cerr << "Failed to load DSO: " << dlerror() << "\n";
            return nullptr;
        }

        std::string sym_name = "kernel_" + prog.name;
        auto func = (TEIRKernelPtr)dlsym(handle, sym_name.c_str());
        if (!func) {
            std::cerr << "Failed to locate entry point: " << dlerror() << "\n";
            return nullptr;
        }

        return func;
    }
};

// ============================================================================
// 3. Mathematisch korrekte Repräsentationen der .teir Spezifikationen
// ============================================================================

TEIRProgram create_transposition_program() {
    TEIRProgram prog;
    prog.name = "transposition";
    prog.tensors = {"in", "out"};

    prog.axes["a"] = {"a", 96, {{"in", 786432}, {"out", 192}}};
    prog.axes["b"] = {"b", 128, {{"in", 6144}, {"out", 17664}}};
    prog.axes["c"] = {"c", 48, {{"in", 128}, {"out", 4}}};
    prog.axes["d"] = {"d", 32, {{"in", 4}, {"out", 2260992}}};

    Primitive copy_prim;
    copy_prim.name = "copy"; copy_prim.kind = "Copy";
    copy_prim.axes_map["M"] = {"@c"}; copy_prim.axes_map["N"] = {"@d"};
    prog.primitives["copy"] = copy_prim;

    // Transposition profitiert massiv von Parallelisierung auf Achse A
    auto iter_a = std::make_shared<IterNode>();
    iter_a->axis = "a"; iter_a->policy = "parallel"; // HIER PARALLELISIERT!
    
    auto iter_b = std::make_shared<IterNode>();
    iter_b->axis = "b"; iter_b->policy = "sequential";
    
    auto inv_copy = std::make_shared<InvokeNode>();
    inv_copy->primitive = "copy";

    iter_b->children.push_back(inv_copy);
    iter_a->children.push_back(iter_b);
    prog.roots.push_back(iter_a);

    return prog;
}

TEIRProgram create_matmul_program() {
    TEIRProgram prog;
    prog.name = "matmul";
    prog.tensors = {"in0", "in1", "out"};

    prog.axes["m0"] = {"m0", 256, {{"in0", 1048576}, {"out", 1048576}}};
    prog.axes["m1"] = {"m1", 32,  {{"in0", 2048},    {"out", 256}}};
    prog.axes["n0"] = {"n0", 128, {{"in1", 131072},  {"out", 8192}}};
    prog.axes["n1"] = {"n1", 64,  {{"in1", 4},       {"out", 4}}};
    prog.axes["k0"] = {"k0", 16,  {{"in0", 65536},   {"in1", 16777216}}};
    prog.axes["k1"] = {"k1", 512, {{"in0", 4},       {"in1", 256}}};

    Primitive zero_prim;
    zero_prim.name = "zero"; zero_prim.kind = "Zero";
    zero_prim.axes_map["M"] = {"@m1"}; zero_prim.axes_map["N"] = {"@n1"};
    prog.primitives["zero"] = zero_prim;

    Primitive gemm_prim;
    gemm_prim.name = "gemm"; gemm_prim.kind = "Contraction";
    gemm_prim.axes_map["M"] = {"@m1"}; gemm_prim.axes_map["N"] = {"@n1"}; gemm_prim.axes_map["K"] = {"@k1"};
    prog.primitives["gemm"] = gemm_prim;

    // Lineare Auflösung der Matmul-Schnittstelle ohne False Sharing
    auto iter_m0 = std::make_shared<IterNode>();
    iter_m0->axis = "m0"; iter_m0->policy = "parallel"; 

    auto iter_n0 = std::make_shared<IterNode>();
    iter_n0->axis = "n0"; iter_n0->policy = "parallel"; // Beide Raumachsen parallelisieren!

    auto iter_k0 = std::make_shared<IterNode>();
    iter_k0->axis = "k0"; iter_k0->policy = "sequential"; 

    auto inv_zero = std::make_shared<InvokeNode>();
    inv_zero->primitive = "zero"; 
    inv_zero->guard = "first(@k0)"; 

    auto inv_gemm = std::make_shared<InvokeNode>();
    inv_gemm->primitive = "gemm";

    iter_k0->children.push_back(inv_zero);
    iter_k0->children.push_back(inv_gemm);
    iter_n0->children.push_back(iter_k0);
    iter_m0->children.push_back(iter_n0);
    
    prog.roots.push_back(iter_m0); 

    return prog;
}

TEIRProgram create_contraction_program() {
    TEIRProgram prog;
    prog.name = "contraction";
    prog.tensors = {"in0", "in1", "out"};

    prog.axes["p"] = {"p", 128, {{"in0", 3145728}, {"out", 2359296}}};
    prog.axes["q"] = {"q", 96,  {{"in0", 32768},   {"out", 24576}}};
    prog.axes["r"] = {"r", 96,  {{"in1", 65536},   {"out", 256}}};
    prog.axes["s"] = {"s", 64,  {{"in1", 4},       {"out", 4}}};
    prog.axes["t"] = {"t", 32,  {{"in0", 1024},    {"in1", 6291456}}};
    prog.axes["u"] = {"u", 256, {{"in0", 4},       {"in1", 256}}};

    Primitive zero_prim;
    zero_prim.name = "zero"; zero_prim.kind = "Zero";
    zero_prim.axes_map["M"] = {"@q"}; zero_prim.axes_map["N"] = {"@s"};
    prog.primitives["zero"] = zero_prim;

    Primitive gemm_prim;
    gemm_prim.name = "gemm"; gemm_prim.kind = "Contraction";
    gemm_prim.axes_map["M"] = {"@q"}; gemm_prim.axes_map["N"] = {"@s"}; gemm_prim.axes_map["K"] = {"@u"};
    prog.primitives["gemm"] = gemm_prim;

    auto iter_p = std::make_shared<IterNode>();
    iter_p->axis = "p"; iter_p->policy = "parallel"; 

    auto iter_r = std::make_shared<IterNode>();
    iter_r->axis = "r"; iter_r->policy = "parallel"; 

    auto iter_t = std::make_shared<IterNode>();
    iter_t->axis = "t"; iter_t->policy = "sequential"; 

    auto inv_zero = std::make_shared<InvokeNode>();
    inv_zero->primitive = "zero"; 
    inv_zero->guard = "first(@t)"; 

    auto inv_gemm = std::make_shared<InvokeNode>();
    inv_gemm->primitive = "gemm";

    iter_t->children.push_back(inv_zero);
    iter_t->children.push_back(inv_gemm);
    iter_r->children.push_back(iter_t);
    iter_p->children.push_back(iter_r);
    
    prog.roots.push_back(iter_p); 

    return prog;
}

// ============================================================================
// 4. Execution Routine
// ============================================================================

int main() {
    std::cout << "[TEIR Runtime System Initializing Evaluation Execution Backend]\n\n";

    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts = {4, 8, 10};

    if (thread_counts.back() != max_threads) {
        thread_counts.push_back(max_threads);
    }

    std::cout << "[OpenMP max threads] " << max_threads << "\n\n";

    auto report = [&](const std::string& name, TEIRKernelPtr kernel, void** args, int iterations,
                      const std::function<void()>& reset) {
        BenchmarkResult result = benchmark_kernel(kernel, args, iterations, reset);
        std::cout << " -> Benchmark [" << name << "] (threads=" << omp_get_max_threads() << "): average "
                  << result.avg_ms << " ms, median " << result.median_ms
                  << " ms, best " << result.min_ms << " ms over "
                  << iterations << " run(s).\n";
    };

    auto run_benchmark = [&](const std::string& name, TEIRKernelPtr kernel, void** args,
                             const std::function<void()>& reset, int iterations) {
        if (!kernel) return;
        std::cout << " -> " << name << " kernel ready. Sweeping thread counts...\n";
        for (int threads : thread_counts) {
            omp_set_num_threads(threads);
            std::cout << "   threads=" << threads << " ... " << std::flush;
            report(name, kernel, args, iterations, reset);
        }
        std::cout << "\n";
    };

    // --- Test 1: Transposition ---
    TEIRProgram trans_prog = create_transposition_program();
    TEIRCompiler compiler_trans(trans_prog);
    TEIRKernelPtr trans_kernel = compiler_trans.compile();
    std::vector<float> input_tensor(96 * 128 * 48 * 32, 1.23f);
    std::vector<float> output_tensor(32 * 128 * 96 * 48, 0.0f);
    void* trans_args[] = { input_tensor.data(), output_tensor.data() };
    auto reset_trans = [&]() { std::fill(output_tensor.begin(), output_tensor.end(), 0.0f); };

    if(trans_kernel) {
        std::cout << " -> Compilation Successful: Transposition Kernel Pointer Ready.\n";
        run_benchmark("transposition", trans_kernel, trans_args, reset_trans, 10);
    }

    // --- Test 2: Matmul ---
    TEIRProgram matmul_prog = create_matmul_program();
    TEIRCompiler compiler_matmul(matmul_prog);
    TEIRKernelPtr matmul_kernel = compiler_matmul.compile();
    std::vector<float> in0_tensor_mm(256LL * 32 * 16 * 512, 1.0f);
    std::vector<float> in1_tensor_mm(16LL * 512 * 128 * 64, 1.0f);
    std::vector<float> out_tensor_mm(256LL * 32 * 128 * 64, 0.0f);
    void* matmul_args[] = { in0_tensor_mm.data(), in1_tensor_mm.data(), out_tensor_mm.data() };
    auto reset_matmul = [&]() { std::fill(out_tensor_mm.begin(), out_tensor_mm.end(), 0.0f); };

    if(matmul_kernel) {
         std::cout << " -> Compilation Successful: Matmul Kernel Pointer Ready.\n";
         run_benchmark("matmul", matmul_kernel, matmul_args, reset_matmul, 3);
    }

    // --- Test 3: Contraction ---
    TEIRProgram contraction_prog = create_contraction_program();
    TEIRCompiler compiler_contract(contraction_prog);
    TEIRKernelPtr contract_kernel = compiler_contract.compile();
    if(contract_kernel) {
         std::cout << " -> Compilation Successful: Contraction Kernel Pointer Ready.\n";
         std::vector<float> in0_tensor_co(128LL * 96 * 32 * 256, 1.0f);
         std::vector<float> in1_tensor_co(32LL * 96 * 256 * 64, 1.0f);
         std::vector<float> out_tensor_co(128LL * 96 * 96 * 64, 0.0f);
         void* contraction_args[] = { in0_tensor_co.data(), in1_tensor_co.data(), out_tensor_co.data() };
         auto reset_contraction = [&]() { std::fill(out_tensor_co.begin(), out_tensor_co.end(), 0.0f); };
         run_benchmark("contraction", contract_kernel, contraction_args, reset_contraction, 3);
    }

    return 0;
}