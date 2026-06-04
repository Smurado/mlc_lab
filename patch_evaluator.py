import re

with open("week8/teir_evaluator.cpp", "r") as f:
    text = f.read()

fallback_code = """
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
"""

text = text.replace("        if (try_evaluate_neon_contraction(prim, args, active_loops)) {\n            return;\n        }", fallback_code)

with open("week8/teir_evaluator.cpp", "w") as f:
    f.write(text)

