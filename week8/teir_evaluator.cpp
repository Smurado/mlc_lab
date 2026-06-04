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
            #pragma omp parallel for
            for (int i = 0; i < extent; ++i) {
                auto local_loops = active_loops;
                local_loops[ax_name] = i;
                for (auto& child : iter->children) {
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
    std::unordered_map<std::string, int> active_loops;
    for (auto& root : prog.roots) {
        evaluate_node(root, args, active_loops);
    }
}
