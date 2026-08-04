#include "include/teir_evaluator.h"
#include <iostream>
#include <algorithm>
#include <omp.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

long long TEIREvaluator::get_offset(const std::string& tensor_name,
                                    const std::unordered_map<std::string, int>& active_loops) {
    long long offset = 0;
    for (auto const& [ax_name, loop_val] : active_loops) {
        auto it = prog.axes.find(ax_name);
        if (it != prog.axes.end()) {
            auto stride_it = it->second.strides.find(tensor_name);
            if (stride_it != it->second.strides.end() && stride_it->second != 0) {
                offset += (long long)loop_val * stride_it->second;
            }
        }
    }
    return offset;
}

static int get_tensor_idx(const TEIRProgram& prog, const std::string& name) {
    for (size_t i = 0; i < prog.tensors.size(); ++i) {
        if (prog.tensors[i] == name) return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// SME-Kernel aus Woche 6 (mini_jit) statt eigener Rechenschleifen.
//
// Das Feedback zu Woche 8 lautet "des Weiteren wird erneut kein SME verwendet".
// Vorher gab es hier eine handgeschriebene NEON-Kachel (4x16) fuer Contraction
// und sonst elementweise Skalarschleifen. Beides bleibt als Rueckfallweg
// erhalten, denn nicht jedes Layout passt zum SME-Kernel -- siehe
// prepare_primitive.
// ---------------------------------------------------------------------------

namespace {

std::string strip_at_ev(const std::string& s) {
    return (!s.empty() && s[0] == '@') ? s.substr(1) : s;
}

// Erste Achse einer Rolle (M/N/K), ohne fuehrendes '@'.
std::string role_axis(const Primitive& p, const std::string& role) {
    auto it = p.axes_map.find(role);
    if (it == p.axes_map.end() || it->second.empty()) return {};
    return strip_at_ev(it->second[0]);
}

// Gepackter A-Kachel-Puffer je Thread.
//
// mini_jit::Gemm erwartet A M-zusammenhaengend; die TEIR-Beispiele liefern in0
// K-zusammenhaengend. Der Puffer merkt sich die zuletzt gepackte Quelladresse,
// damit ueber eine unveraenderte A-Kachel nicht bei jedem Aufruf neu gepackt
// wird. Identisch zu week7/teir_runtime.cpp.
struct PackBuffer {
    std::vector<float> data;
    const void* lastSrc = nullptr;
    long lastM = -1, lastK = -1, lastSm = -1, lastSk = -1;

    const float* get(const float* src, long m, long k, long stride_m, long stride_k) {
        if (src == lastSrc && m == lastM && k == lastK &&
            stride_m == lastSm && stride_k == lastSk) {
            return data.data();
        }
        data.resize(static_cast<size_t>(m) * k);
        for (long l = 0; l < k; ++l)
            for (long i = 0; i < m; ++i)
                data[static_cast<size_t>(l) * m + i] = src[i * stride_m + l * stride_k];
        lastSrc = src; lastM = m; lastK = k; lastSm = stride_m; lastSk = stride_k;
        return data.data();
    }
};

PackBuffer& packBuffer() {
    static thread_local PackBuffer buf;
    return buf;
}

} // namespace

long TEIREvaluator::stride_of(const std::string& axis, const std::string& tensor) const {
    auto a = prog.axes.find(axis);
    if (a == prog.axes.end()) return 0;
    auto s = a->second.strides.find(tensor);
    if (s == a->second.strides.end()) return 0;
    return s->second / 4;   // .teir gibt Bytes an, gerechnet wird in Elementen
}

void TEIREvaluator::prepare_primitive(const Primitive& prim, PrimPlan& p) {
    p.mAxis = role_axis(prim, "M");
    p.nAxis = role_axis(prim, "N");
    p.kAxis = role_axis(prim, "K");

    auto extent_of = [&](const std::string& ax) -> int {
        auto it = prog.axes.find(ax);
        return it == prog.axes.end() ? 0 : it->second.extent;
    };
    p.mExt = extent_of(p.mAxis);
    p.nExt = extent_of(p.nAxis);
    p.kExt = extent_of(p.kAxis);

    p.outC_m = stride_of(p.mAxis, "out");
    p.outC_n = stride_of(p.nAxis, "out");

    if (prim.kind == "Zero") {
        // Der Unary-Kernel schreibt 16 zusammenhaengende Floats je Schritt --
        // seine M-Richtung muss die zusammenhaengende Achse sein. Bei den
        // TEIR-Beispielen ist out N-zusammenhaengend, deshalb der Rollentausch.
        const long contig = (p.outC_n == 1) ? p.nExt   : p.mExt;
        const long other  = (p.outC_n == 1) ? p.mExt   : p.nExt;
        const long ld     = (p.outC_n == 1) ? p.outC_m : p.outC_n;

        // ld muss mindestens die zusammenhaengende Kante ueberspringen, sonst
        // ueberlappen die Spalten und der Kernel nullt nur einen Bruchteil der
        // Kachel. Bei einsum.teir tritt genau das auf: dort deklarieren BEIDE
        // Primitive-Achsen (@y und @x) Stride 4 auf `out`, woraus ld = 1 folgt
        // -- der Kernel haette 2686 statt 1769472 Elementen genullt. Aufgefallen
        // waere es nicht, weil main.cpp `out` ohnehin vorher per memset nullt.
        // Ein Kernel, der stillschweigend zu wenig tut, ist schlechter als ein
        // ehrlicher Rueckfall.
        if (contig % 16 == 0 && ld >= contig) {
            p.unary = std::make_shared<mini_jit::Unary>();
            if (p.unary->generate(static_cast<uint32_t>(contig), static_cast<uint32_t>(other),
                                  0, mini_jit::Unary::dtype_t::fp32,
                                  mini_jit::Unary::ptype_t::zero)
                == mini_jit::Unary::error_t::success) {
                p.unaryFn = p.unary->get_kernel();
                p.kind    = PrimPlan::Kind::ZeroJit;
                p.uLdB    = ld;
                plan_lines.push_back("  " + prim.name + " (Zero) -> mini_jit::Unary "
                                     + std::to_string(contig) + "x" + std::to_string(other));
                return;
            }
        }
        plan_lines.push_back(
            "  " + prim.name + " (Zero) -> Rueckfall ("
            + (contig % 16 != 0
                   ? "Kante " + std::to_string(contig) + " nicht durch 16 teilbar"
                   : "ld " + std::to_string(ld) + " < Kante " + std::to_string(contig)
                     + ", Kernel wuerde die Kachel nur teilweise nullen")
            + ")");
        return;
    }

    if (prim.kind == "Copy") {
        const long inM = stride_of(p.mAxis, "in");
        const long inN = stride_of(p.nAxis, "in");

        // Der Unary-Kernel rechnet B := op(A) mit
        //   A(i,j) = a[i + j*ld_a]                (A ist i-zusammenhaengend)
        //   B(i,j) = b[i + j*ld_b]                bei trans_b = 0
        //   B(i,j) = b[i*ld_b + j]                bei trans_b = 1
        // Gesucht ist also die Zuordnung der beiden TEIR-Achsen zu i und j, bei
        // der beide Tensoren mit ihrer jeweils zusammenhaengenden Achse passen.
        // i := die in `in` zusammenhaengende Achse. Ob transponiert werden muss,
        // entscheidet sich daran, ob `out` auf DERSELBEN Achse zusammenhaengt.
        uint32_t trans_b = 2;   // 2 = kein passendes Layout gefunden
        if (inM == 1) {
            p.uM = p.mExt; p.uN = p.nExt; p.uLdA = inN;
            if      (p.outC_m == 1) { trans_b = 0; p.uLdB = p.outC_n; }
            else if (p.outC_n == 1) { trans_b = 1; p.uLdB = p.outC_m; }
        } else if (inN == 1) {
            p.uM = p.nExt; p.uN = p.mExt; p.uLdA = inM;
            if      (p.outC_n == 1) { trans_b = 0; p.uLdB = p.outC_m; }
            else if (p.outC_m == 1) { trans_b = 1; p.uLdB = p.outC_n; }
        }

        if (trans_b <= 1 && p.uM % 16 == 0 && p.uN % 16 == 0 && p.uLdB > 0) {
            p.unary = std::make_shared<mini_jit::Unary>();
            if (p.unary->generate(static_cast<uint32_t>(p.uM), static_cast<uint32_t>(p.uN),
                                  trans_b, mini_jit::Unary::dtype_t::fp32,
                                  mini_jit::Unary::ptype_t::identity)
                == mini_jit::Unary::error_t::success) {
                p.unaryFn = p.unary->get_kernel();
                p.kind    = PrimPlan::Kind::CopyJit;
                plan_lines.push_back("  " + prim.name + " (Copy) -> mini_jit::Unary identity "
                                     + std::to_string(p.uM) + "x" + std::to_string(p.uN)
                                     + (trans_b ? " (transponierend)" : ""));
                return;
            }
        }
        plan_lines.push_back("  " + prim.name + " (Copy) -> Rueckfall "
                             "(Layout oder Groesse passt nicht zum Unary-Kernel)");
        return;
    }

    if (prim.kind == "Contraction") {
        p.inA_m = stride_of(p.mAxis, "in0");
        p.inA_k = stride_of(p.kAxis, "in0");
        p.inB_n = stride_of(p.nAxis, "in1");
        p.inB_k = stride_of(p.kAxis, "in1");

        // Durch Rollentausch (i<->n, j<->m) passen B und C zum Kernel; A wird
        // gepackt. Bedingung: out und in1 muessen N-zusammenhaengend sein.
        const bool swapped = (p.outC_n == 1 && p.inB_n == 1);
        if (swapped && p.mExt % 16 == 0 && p.nExt % 16 == 0) {
            p.gemm = std::make_shared<mini_jit::Gemm>();
            if (p.gemm->generate(static_cast<uint32_t>(p.nExt), static_cast<uint32_t>(p.mExt),
                                 static_cast<uint32_t>(p.kExt), 0, 0, 0,
                                 mini_jit::Gemm::dtype_t::fp32)
                == mini_jit::Gemm::error_t::success) {
                p.gemmFn = p.gemm->get_kernel();
                p.kind   = PrimPlan::Kind::GemmJit;
                plan_lines.push_back("  " + prim.name + " (Contraction) -> mini_jit::Gemm "
                                     + std::to_string(p.nExt) + "x" + std::to_string(p.mExt)
                                     + "x" + std::to_string(p.kExt) + " (A wird gepackt)");
                return;
            }
        }
        plan_lines.push_back("  " + prim.name + " (Contraction) -> Rueckfall NEON/skalar "
                             "(Layout oder Groesse passt nicht zum SME-Kernel)");
        return;
    }

    plan_lines.push_back("  " + prim.name + " (" + prim.kind + ") -> Rueckfall");
}

void TEIREvaluator::build_plans() {
    plans.clear();
    plan_lines.clear();
    plan_lines.push_back("Ausfuehrungsplan fuer @" + prog.name + ":");
    for (auto const& entry : prog.primitives) {
        PrimPlan p;
        prepare_primitive(entry.second, p);
        plans[entry.first] = std::move(p);
    }
}

bool TEIREvaluator::try_evaluate_jit(const Primitive& prim, void** args,
                                     const std::unordered_map<std::string, int>& active_loops) {
    auto pit = plans.find(prim.name);
    if (pit == plans.end()) return false;
    const PrimPlan& p = pit->second;
    if (p.kind == PrimPlan::Kind::Fallback) return false;

    // Die Primitive-eigenen Achsen uebernimmt der Kernel selbst -- laeuft eine
    // davon noch als Schleife aussen herum, passt der Plan nicht.
    if (active_loops.count(p.mAxis) || active_loops.count(p.nAxis) ||
        active_loops.count(p.kAxis)) {
        return false;
    }

    const int iOut = get_tensor_idx(prog, "out");
    if (iOut < 0) return false;
    float* out = static_cast<float*>(args[iOut]) + get_offset("out", active_loops) / 4;

    if (p.kind == PrimPlan::Kind::ZeroJit) {
        // Signatur: (a, b, ld_a, ld_b); a wird beim Zero-Kernel nicht gelesen.
        p.unaryFn(nullptr, out, p.uLdB, p.uLdB);
        return true;
    }

    if (p.kind == PrimPlan::Kind::CopyJit) {
        const int iIn = get_tensor_idx(prog, "in");
        if (iIn < 0) return false;
        const float* in = static_cast<const float*>(args[iIn])
                        + get_offset("in", active_loops) / 4;
        p.unaryFn(in, out, p.uLdA, p.uLdB);
        return true;
    }

    const int iA = get_tensor_idx(prog, "in0");
    const int iB = get_tensor_idx(prog, "in1");
    if (iA < 0 || iB < 0) return false;
    const float* a0 = static_cast<const float*>(args[iA]) + get_offset("in0", active_loops) / 4;
    const float* b0 = static_cast<const float*>(args[iB]) + get_offset("in1", active_loops) / 4;

    // Rollentausch: der Kernel rechnet C'(n,m) += A'(n,k) * B'(k,m).
    const float* packed = packBuffer().get(a0, p.mExt, p.kExt, p.inA_m, p.inA_k);
    p.gemmFn(b0, packed, out, p.inB_k, p.mExt, p.outC_m);
    return true;
}

void TEIREvaluator::evaluate_primitive(const Primitive& prim, void** args,
                                       const std::unordered_map<std::string, int>& active_loops) {
    if (prim.kind == "Zero") {
        float* out = (float*)args[get_tensor_idx(prog, "out")];
        out[get_offset("out", active_loops) / 4] = 0.0f; // get_offset gives bytes!
    } else if (prim.kind == "Contraction") {
        float* out = (float*)args[get_tensor_idx(prog, "out")];
        float* in0 = (float*)args[get_tensor_idx(prog, "in0")];
        float* in1 = (float*)args[get_tensor_idx(prog, "in1")];
        out[get_offset("out", active_loops) / 4] += in0[get_offset("in0", active_loops) / 4] * in1[get_offset("in1", active_loops) / 4];
    } else if (prim.kind == "Copy") {
        float* out = (float*)args[get_tensor_idx(prog, "out")];
        float* in = (float*)args[get_tensor_idx(prog, "in")];
        out[get_offset("out", active_loops) / 4] = in[get_offset("in", active_loops) / 4];
    }
}

bool TEIREvaluator::try_evaluate_neon_contraction(const Primitive& prim, void** args,
                                                  const std::unordered_map<std::string, int>& active_loops) {
#if defined(__ARM_NEON) || defined(__aarch64__)
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
        auto it = a.strides.find(t); return it == a.strides.end() ? 0 : it->second;
    };
    int sM_out = getStride(aM, "out");
    int sN_in1 = getStride(aN, "in1");
    int sN_out = getStride(aN, "out");
    int sK_in0 = getStride(aK, "in0");

    if (sN_in1 != 4 || sN_out != 4 || sK_in0 != 4) return false;
    constexpr int M_R = 4;
    constexpr int N_R = 16;
    constexpr int N_V = 4; // N_R / 4
    if (aM.extent % M_R != 0 || aN.extent % N_R != 0) return false;

    float* out = (float*)((char*)args[get_tensor_idx(prog, "out")] + get_offset("out", active_loops));
    float* in0 = (float*)((char*)args[get_tensor_idx(prog, "in0")] + get_offset("in0", active_loops));
    float* in1 = (float*)((char*)args[get_tensor_idx(prog, "in1")] + get_offset("in1", active_loops));

    for (int im = 0; im < aM.extent; im += M_R) {
        for (int in_ = 0; in_ < aN.extent; in_ += N_R) {
            float32x4_t c[M_R][N_V];
            for (int i = 0; i < M_R; ++i) {
                for (int v = 0; v < N_V; ++v) {
                    c[i][v] = vld1q_f32((float*)((char*)out + (im+i)*sM_out + (in_+v*4)*4));
                }
            }
            for (int k = 0; k < aK.extent; ++k) {
                float32x4_t b_vec[N_V];
                for (int v = 0; v < N_V; ++v) {
                    b_vec[v] = vld1q_f32((float*)((char*)in1 + k*getStride(aK, "in1") + (in_+v*4)*4));
                }
                for (int i = 0; i < M_R; ++i) {
                    float a_val = *(float*)((char*)in0 + (im+i)*getStride(aM, "in0") + k*4);
                    for (int v = 0; v < N_V; ++v) {
                        c[i][v] = vfmaq_n_f32(c[i][v], b_vec[v], a_val);
                    }
                }
            }
            for (int i = 0; i < M_R; ++i) {
                for (int v = 0; v < N_V; ++v) {
                    vst1q_f32((float*)((char*)out + (im+i)*sM_out + (in_+v*4)*4), c[i][v]);
                }
            }
        }
    }
    return true;
#else
    return false;
#endif
}

void TEIREvaluator::lower_primitive_recursive(const Primitive& prim, void** args,
                                              std::unordered_map<std::string, int>& active_loops,
                                              const std::vector<std::string>& prim_axes,
                                              size_t depth) {
    if (depth == prim_axes.size()) {
        evaluate_primitive(prim, args, active_loops);
        return;
    }
    std::string ax_name = prim_axes[depth];
    if (active_loops.find(ax_name) != active_loops.end()) {
        lower_primitive_recursive(prim, args, active_loops, prim_axes, depth + 1);
    } else {
        int extent = prog.axes[ax_name].extent;
        for (int i = 0; i < extent; ++i) {
            active_loops[ax_name] = i;
            lower_primitive_recursive(prim, args, active_loops, prim_axes, depth + 1);
        }
        active_loops.erase(ax_name);
    }
}

void TEIREvaluator::evaluate_node(const std::shared_ptr<ScheduleNode>& node, void** args,
                                  std::unordered_map<std::string, int> active_loops) {
    if (node->type == NodeType::Iter) {
        auto iter = std::static_pointer_cast<IterNode>(node);
        std::string ax_name = (iter->axis[0] == '@') ? iter->axis.substr(1) : iter->axis;
        int extent = prog.axes[ax_name].extent;

        if (iter->policy == "parallel") {
            // Kette direkt geschachtelter paralleler Achsen einsammeln und zu
            // EINER Schleife ueber das Produkt abflachen. Ohne das bleibt jede
            // innere parallel-Region wirkungslos: OpenMP schaltet geschachtelte
            // Parallelitaet standardmaessig ab (omp_get_max_active_levels() == 1).
            // Programme, deren Parallelitaet auf mehrere kleine Achsen verteilt
            // ist, liefen dadurch faktisch einkernig.
            std::vector<std::string> axes    { ax_name };
            std::vector<int>         extents { extent };
            auto deepest = iter;
            while (deepest->children.size() == 1 &&
                   deepest->children[0]->type == NodeType::Iter) {
                auto ch = std::static_pointer_cast<IterNode>(deepest->children[0]);
                if (ch->policy != "parallel") break;
                std::string cax = (ch->axis[0] == '@') ? ch->axis.substr(1) : ch->axis;
                axes.push_back(cax);
                extents.push_back(prog.axes[cax].extent);
                deepest = ch;
            }

            long total = 1;
            for (int e : extents) total *= e;
            const int  ndim  = static_cast<int>(axes.size());
            auto&      leafs = deepest->children;

            #pragma omp parallel for
            for (long lin = 0; lin < total; ++lin) {
                auto local_loops = active_loops;
                long rem = lin;
                for (int d = ndim - 1; d >= 0; --d) {   // letzte Achse laeuft am schnellsten
                    local_loops[axes[d]] = static_cast<int>(rem % extents[d]);
                    rem /= extents[d];
                }
                for (auto& child : leafs) {
                    evaluate_node(child, args, local_loops);
                }
            }
        } else {
            for (int i = 0; i < extent; ++i) {
                active_loops[ax_name] = i;
                for (auto& child : iter->children) {
                    evaluate_node(child, args, active_loops);
                }
            }
        }
    } else if (node->type == NodeType::Invoke) {
        auto invoke = std::static_pointer_cast<InvokeNode>(node);
        // Einfache Evaluierung des Guards "first(@k0)" etc.
        if (!invoke->guard.empty() && invoke->guard.find("first") != std::string::npos) {
            size_t start = invoke->guard.find("(@") + 2;
            size_t end = invoke->guard.find(")", start);
            std::string guard_ax = invoke->guard.substr(start, end - start);
            if (active_loops.count(guard_ax) && active_loops[guard_ax] != 0) {
                return; // Guard greift, wir führen Prime nicht aus
            }
        }

        std::string prim_name = (invoke->primitive[0] == '@') ? invoke->primitive.substr(1) : invoke->primitive;
        const Primitive& prim = prog.primitives[prim_name];


        // Reihenfolge: erst der generierte SME-Kernel, dann die NEON-Kachel,
        // dann die Skalarschleifen. Jede Stufe prueft selbst, ob sie passt.
        if (try_evaluate_jit(prim, args, active_loops)) {
            return;
        }

        if (try_evaluate_neon_contraction(prim, args, active_loops)) {
            return;
        }

        // Fast Fallback for Contraction
        if (prim.kind == "Contraction" && prim.axes_map.count("M") && prim.axes_map.count("N") && prim.axes_map.count("K")) {
            auto strip = [](const std::string& s){ return (!s.empty() && s[0]=='@') ? s.substr(1) : s; };
            std::string aM = strip(prim.axes_map.at("M")[0]);
            std::string aN = strip(prim.axes_map.at("N")[0]);
            std::string aK = strip(prim.axes_map.at("K")[0]);
            
            if (!active_loops.count(aM) && !active_loops.count(aN) && !active_loops.count(aK)) {
                int extM = prog.axes[aM].extent;
                int extN = prog.axes[aN].extent;
                int extK = prog.axes[aK].extent;
                
                int sM_out = prog.axes[aM].strides.count("out") ? prog.axes[aM].strides["out"]/4 : 0;
                int sN_out = prog.axes[aN].strides.count("out") ? prog.axes[aN].strides["out"]/4 : 0;
                
                int sM_in0 = prog.axes[aM].strides.count("in0") ? prog.axes[aM].strides["in0"]/4 : 0;
                int sN_in0 = prog.axes[aN].strides.count("in0") ? prog.axes[aN].strides["in0"]/4 : 0;
                int sK_in0 = prog.axes[aK].strides.count("in0") ? prog.axes[aK].strides["in0"]/4 : 0;
                
                int sM_in1 = prog.axes[aM].strides.count("in1") ? prog.axes[aM].strides["in1"]/4 : 0;
                int sN_in1 = prog.axes[aN].strides.count("in1") ? prog.axes[aN].strides["in1"]/4 : 0;
                int sK_in1 = prog.axes[aK].strides.count("in1") ? prog.axes[aK].strides["in1"]/4 : 0;
                
                int base_out = get_offset("out", active_loops)/4;
                int base_in0 = get_offset("in0", active_loops)/4;
                int base_in1 = get_offset("in1", active_loops)/4;
                
                float* out = (float*)args[get_tensor_idx(prog, "out")];
                float* in0 = (float*)args[get_tensor_idx(prog, "in0")];
                float* in1 = (float*)args[get_tensor_idx(prog, "in1")];
                
                for (int m = 0; m < extM; ++m) {
                    for (int n = 0; n < extN; ++n) {
                        float sum = 0.0f;
                        int out_idx = base_out + m * sM_out + n * sN_out;
                        int in0_mn = base_in0 + m * sM_in0 + n * sN_in0;
                        int in1_mn = base_in1 + m * sM_in1 + n * sN_in1;
                        for (int k = 0; k < extK; ++k) {
                            sum += in0[in0_mn + k * sK_in0] * in1[in1_mn + k * sK_in1];
                        }
                        out[out_idx] += sum;
                    }
                }
                return;
            }
        }
        
        // Fast Fallback for Zero
        if (prim.kind == "Zero" && prim.axes_map.count("M") && prim.axes_map.count("N")) {
            auto strip = [](const std::string& s){ return (!s.empty() && s[0]=='@') ? s.substr(1) : s; };
            std::string aM = strip(prim.axes_map.at("M")[0]);
            std::string aN = strip(prim.axes_map.at("N")[0]);
            if (!active_loops.count(aM) && !active_loops.count(aN)) {
                int extM = prog.axes[aM].extent;
                int extN = prog.axes[aN].extent;
                int sM_out = prog.axes[aM].strides.count("out") ? prog.axes[aM].strides["out"]/4 : 0;
                int sN_out = prog.axes[aN].strides.count("out") ? prog.axes[aN].strides["out"]/4 : 0;
                int base_out = get_offset("out", active_loops)/4;
                float* out = (float*)args[get_tensor_idx(prog, "out")];
                
                for (int m = 0; m < extM; ++m) {
                    for (int n = 0; n < extN; ++n) {
                        out[base_out + m * sM_out + n * sN_out] = 0.0f;
                    }
                }
                return;
            }
        }


        std::vector<std::string> prim_axes;
        for (auto const& [key, list] : prim.axes_map) {
            for (auto const& ax : list) {
                std::string clean_ax = (ax[0] == '@') ? ax.substr(1) : ax;
                if (std::find(prim_axes.begin(), prim_axes.end(), clean_ax) == prim_axes.end()) {
                    prim_axes.push_back(clean_ax);
                }
            }
        }
        lower_primitive_recursive(prim, args, active_loops, prim_axes, 0);
    }
}

void TEIREvaluator::evaluate(void** args) {
    // Kernel einmal vor der ersten Ausfuehrung erzeugen. Danach sind `plans`
    // nur noch lesend im Zugriff -- das ist die Voraussetzung dafuer, dass
    // evaluate_node aus mehreren Threads darauf zugreifen darf.
    if (!plans_built) {
        build_plans();
        plans_built = true;
    }
    std::unordered_map<std::string, int> active_loops;
    for (auto& root : prog.roots) {
        evaluate_node(root, args, active_loops);
    }
}
