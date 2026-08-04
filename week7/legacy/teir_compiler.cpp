#include "include/teir_compiler.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>

void TEIRCompiler::emit(const std::string& line) {
    src << std::string(indent_level * 4, ' ') << line << "\n";
}

std::string TEIRCompiler::get_offset_expr(
        const std::string& tensor_name,
        const std::unordered_map<std::string, std::string>& active_loops) {
    std::vector<std::string> terms;
    for (auto const& [ax_name, loop_var] : active_loops) {
        auto it = prog.axes.find(ax_name);
        if (it != prog.axes.end()) {
            auto stride_it = it->second.strides.find(tensor_name);
            if (stride_it != it->second.strides.end() && stride_it->second != 0) {
                terms.push_back("((long long)" + loop_var + " * " +
                                std::to_string(stride_it->second) + ")");
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

void TEIRCompiler::emit_kernel_body(
        const Primitive& prim,
        const std::unordered_map<std::string, std::string>& all_loops) {
    if (prim.kind == "Zero") {
        std::string out_offset = get_offset_expr("out", all_loops);
        emit("*(float*)((char*)out + " + out_offset + ") = 0.0f;");
    } else if (prim.kind == "Contraction") {
        std::string in0_offset = get_offset_expr("in0", all_loops);
        std::string in1_offset = get_offset_expr("in1", all_loops);
        std::string out_offset = get_offset_expr("out", all_loops);
        emit("*(float*)((char*)out + " + out_offset + ") += "
             "(*(float*)((char*)in0 + " + in0_offset + ")) * (*(float*)((char*)in1 + " + in1_offset + "));");
    } else if (prim.kind == "Copy") {
        std::string in_offset = get_offset_expr("in", all_loops);
        std::string out_offset = get_offset_expr("out", all_loops);
        emit("*(float*)((char*)out + " + out_offset + ") = *(float*)((char*)in + " + in_offset + ");");
    }
}

// Versucht, fuer ein Contraction-Primitive einen NEON-Microkernel zu emittieren.
// Bedingungen (sonst false -> Fallback auf Skalar-Codegen):
//   - Genau eine M-, N- und K-Achse im Primitive
//   - N-Achse hat Stride 4 (kontiguierlich) auf in1 und out
//   - K-Achse hat Stride 4 (kontiguierlich) auf in0
//   - M/N/K sind noch nicht durch Schedule-Loops aktiv (keine Race-Conditions)
//   - M-Extent durch M_R, N-Extent durch N_R teilbar
// Tile: 4 M-Zeilen x 4 NEON-Vektoren (= 16 Floats breit), broadcast-FMA ueber K.
bool TEIRCompiler::try_emit_neon_contraction(
        const Primitive& prim,
        const std::unordered_map<std::string, std::string>& active_loops) {
    if (prim.kind != "Contraction") return false;
    auto itM = prim.axes_map.find("M");
    auto itN = prim.axes_map.find("N");
    auto itK = prim.axes_map.find("K");
    if (itM == prim.axes_map.end() || itN == prim.axes_map.end() || itK == prim.axes_map.end()) return false;
    if (itM->second.size() != 1 || itN->second.size() != 1 || itK->second.size() != 1) return false;

    auto strip = [](const std::string& s){ return (!s.empty() && s[0]=='@') ? s.substr(1) : s; };
    std::string mAx = strip(itM->second[0]);
    std::string nAx = strip(itN->second[0]);
    std::string kAx = strip(itK->second[0]);

    if (active_loops.count(mAx) || active_loops.count(nAx) || active_loops.count(kAx)) return false;

    auto axIt_m = prog.axes.find(mAx);
    auto axIt_n = prog.axes.find(nAx);
    auto axIt_k = prog.axes.find(kAx);
    if (axIt_m == prog.axes.end() || axIt_n == prog.axes.end() || axIt_k == prog.axes.end()) return false;
    const Axis& aM = axIt_m->second;
    const Axis& aN = axIt_n->second;
    const Axis& aK = axIt_k->second;

    auto getStride = [](const Axis& a, const std::string& t)->int {
        auto it = a.strides.find(t);
        return it == a.strides.end() ? 0 : it->second;
    };
    int sM_in0 = getStride(aM, "in0");
    int sM_out = getStride(aM, "out");
    int sN_in1 = getStride(aN, "in1");
    int sN_out = getStride(aN, "out");
    int sK_in0 = getStride(aK, "in0");
    int sK_in1 = getStride(aK, "in1");

    if (sN_in1 != 4 || sN_out != 4) return false; // N muss kontiguierlich sein
    if (sK_in0 != 4) return false;                 // K muss in in0 kontiguierlich sein

    constexpr int M_R = 4;
    constexpr int N_R = 16; // 4 NEON-Vektoren breit (4 * 4 Floats)
    constexpr int N_V = N_R / 4; // Anzahl NEON-Vektoren pro M-Zeile
    if (aM.extent % M_R != 0) return false;
    if (aN.extent % N_R != 0) return false;

    std::string base_out = get_offset_expr("out", active_loops);
    std::string base_in0 = get_offset_expr("in0", active_loops);
    std::string base_in1 = get_offset_expr("in1", active_loops);

    emit("// === NEON microkernel (M_R=" + std::to_string(M_R) +
         ", N_R=" + std::to_string(N_R) + ") ===");
    emit("{");
    indent_level++;
    emit("long long b_out = " + base_out + ";");
    emit("long long b_in0 = " + base_in0 + ";");
    emit("long long b_in1 = " + base_in1 + ";");
    emit("for (int im = 0; im < " + std::to_string(aM.extent) + "; im += " + std::to_string(M_R) + ") {");
    indent_level++;
    emit("for (int in_ = 0; in_ < " + std::to_string(aN.extent) + "; in_ += " + std::to_string(N_R) + ") {");
    indent_level++;
    // Akkumulator-Register laden: M_R Zeilen x N_V Vektoren
    for (int i = 0; i < M_R; ++i) {
        for (int v = 0; v < N_V; ++v) {
            emit("float32x4_t c" + std::to_string(i) + "_" + std::to_string(v) +
                 " = vld1q_f32((float*)((char*)out + b_out + (long long)(im+" + std::to_string(i) +
                 ")*" + std::to_string(sM_out) + " + (long long)(in_+" + std::to_string(v*4) + ")*4));");
        }
    }
    emit("for (int k = 0; k < " + std::to_string(aK.extent) + "; ++k) {");
    indent_level++;
    // N_V Vektor-Loads aus in1
    for (int v = 0; v < N_V; ++v) {
        emit("float32x4_t b" + std::to_string(v) +
             " = vld1q_f32((float*)((char*)in1 + b_in1 + (long long)k*" +
             std::to_string(sK_in1) + " + (long long)(in_+" + std::to_string(v*4) + ")*4));");
    }
    // M_R Skalar-Loads aus in0
    for (int i = 0; i < M_R; ++i) {
        emit("float a" + std::to_string(i) +
             " = *(float*)((char*)in0 + b_in0 + (long long)(im+" + std::to_string(i) +
             ")*" + std::to_string(sM_in0) + " + (long long)k*4);");
    }
    // M_R * N_V FMAs (broadcast-multiply-add)
    for (int i = 0; i < M_R; ++i) {
        for (int v = 0; v < N_V; ++v) {
            emit("c" + std::to_string(i) + "_" + std::to_string(v) +
                 " = vfmaq_n_f32(c" + std::to_string(i) + "_" + std::to_string(v) +
                 ", b" + std::to_string(v) + ", a" + std::to_string(i) + ");");
        }
    }
    indent_level--;
    emit("}");
    // Akkumulatoren zurueckschreiben
    for (int i = 0; i < M_R; ++i) {
        for (int v = 0; v < N_V; ++v) {
            emit("vst1q_f32((float*)((char*)out + b_out + (long long)(im+" + std::to_string(i) +
                 ")*" + std::to_string(sM_out) + " + (long long)(in_+" + std::to_string(v*4) + ")*4), c" +
                 std::to_string(i) + "_" + std::to_string(v) + ");");
        }
    }
    indent_level--;
    emit("}");
    indent_level--;
    emit("}");
    indent_level--;
    emit("}");
    return true;
}

void TEIRCompiler::lower_primitive(
        const Primitive& prim,
        std::unordered_map<std::string, std::string> active_loops) {
    // Eigener Code-Generator fuer den innersten Kernel: bei Contractions
    // versuchen wir zuerst den NEON-Microkernel; passt der nicht, fallen wir
    // auf den generischen Skalar-Codegen zurueck.
    if (try_emit_neon_contraction(prim, active_loops)) return;

    std::vector<std::string> prim_axes;
    for (auto const& [key, list] : prim.axes_map) {
        (void)key;
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

        // Falls die Achse bereits durch eine aeussere Schleife im Schedule kontrolliert wird,
        // duerfen wir hier KEINE neue Schleife generieren! Sonst gibt es Race Conditions.
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

void TEIRCompiler::lower_node(
        const std::shared_ptr<ScheduleNode>& node,
        std::unordered_map<std::string, std::string> active_loops) {
    if (node->type == NodeType::Iter) {
        auto iter = std::static_pointer_cast<IterNode>(node);

        // Falls dieser Knoten parallel ist, sammle die maximale Kette
        // aufeinanderfolgender parallel-IterNodes (linear, jeweils genau ein Kind),
        // damit wir ein einziges `#pragma omp parallel for collapse(N)` emittieren
        // statt verschachtelter parallel-Regionen.
        if (iter->policy == "parallel") {
            std::vector<std::shared_ptr<IterNode>> chain;
            chain.push_back(iter);
            while (chain.back()->children.size() == 1 &&
                   chain.back()->children[0]->type == NodeType::Iter) {
                auto next = std::static_pointer_cast<IterNode>(chain.back()->children[0]);
                if (next->policy != "parallel") break;
                chain.push_back(next);
            }

            if (chain.size() > 1) {
                emit("#pragma omp parallel for collapse(" + std::to_string(chain.size()) + ")");
            } else {
                emit("#pragma omp parallel for");
            }

            for (auto const& link : chain) {
                Axis ax = prog.axes[link->axis];
                std::string lv = "i_" + ax.name;
                emit("for (int " + lv + " = 0; " + lv + " < " +
                     std::to_string(ax.extent) + "; ++" + lv + ") {");
                indent_level++;
                active_loops[ax.name] = lv;
            }

            // Kinder des innersten Glieds der Kette rekursiv emittieren
            for (auto const& child : chain.back()->children) {
                lower_node(child, active_loops);
            }

            for (size_t i = 0; i < chain.size(); ++i) {
                indent_level--;
                emit("}");
            }
            return;
        }

        Axis axis_info = prog.axes[iter->axis];
        std::string loop_var = "i_" + axis_info.name;
        emit("for (int " + loop_var + " = 0; " + loop_var + " < " + std::to_string(axis_info.extent) + "; ++" + loop_var + ") {");
        indent_level++;

        active_loops[axis_info.name] = loop_var;
        for (auto const& child : iter->children) {
            lower_node(child, active_loops);
        }

        indent_level--;
        emit("}");
    } else if (node->type == NodeType::Invoke) {
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

std::string TEIRCompiler::generate_cpp_source() {
    src.str(""); src.clear();
    emit("#include <omp.h>");
    emit("#include <iostream>");
    emit("#include <arm_neon.h>");
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

TEIRKernelPtr TEIRCompiler::compile() {
    std::string source = generate_cpp_source();
    std::string cpp_filename = "teir_jit_" + prog.name + ".cpp";
    std::string so_filename = "./teir_jit_" + prog.name + ".so";

    std::ofstream out(cpp_filename);
    out << source;
    out.close();

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
