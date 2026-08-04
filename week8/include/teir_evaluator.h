#pragma once
#include "teir.h"
#include "../../week6/Unary.h"
#include "../../week6/Gemm.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class TEIREvaluator {
public:
    explicit TEIREvaluator(TEIRProgram p) : prog(std::move(p)) {}

    // Führt das TEIR-Programm rekursiv für die gegebenen Tensoren (args) aus.
    void evaluate(void** args);

    // Wie die Primitive tatsächlich ausgeführt wurden — eine Zeile je Primitive.
    // Macht im Protokoll sichtbar, ob der SME-Kernel gegriffen hat oder ob auf
    // NEON/skalar zurückgefallen wurde.
    const std::vector<std::string>& plan() const { return plan_lines; }

    // Nur die Kernel-Auswahl treffen, ohne zu rechnen. Fuer Tests, die die
    // Auswahl pruefen wollen, ohne ein minutenlanges Modell auszufuehren.
    void plan_only() { if (!plans_built) { build_plans(); plans_built = true; } }

private:
    TEIRProgram prog;

    // Wie ein Primitive ausgeführt wird. Übernommen aus week7/teir_runtime.cpp,
    // wo derselbe Aufbau bereits gemessen ist.
    struct PrimPlan {
        enum class Kind { ZeroJit, CopyJit, GemmJit, Fallback } kind = Kind::Fallback;

        std::string mAxis, nAxis, kAxis;
        int mExt = 0, nExt = 0, kExt = 0;

        // Strides in ELEMENTEN (die .teir-Datei gibt Bytes an).
        long inA_m = 0, inA_k = 0;      // in0
        long inB_n = 0, inB_k = 0;      // in1
        long outC_m = 0, outC_n = 0;    // out

        // Parameter für den Unary-Kernel. Getrennt von mExt/nExt, weil die
        // Rollen je nach Layout vertauscht werden — bei einer Transposition ist
        // die in `in` zusammenhängende Achse eine andere als die in `out`.
        int  uM = 0, uN = 0;
        long uLdA = 0, uLdB = 0;

        // Generierte Kernel — müssen so lange leben wie der Evaluator.
        std::shared_ptr<mini_jit::Unary> unary;
        std::shared_ptr<mini_jit::Gemm>  gemm;
        mini_jit::Unary::kernel_t unaryFn = nullptr;
        mini_jit::Gemm::kernel_t  gemmFn  = nullptr;
    };

    std::unordered_map<std::string, PrimPlan> plans;
    std::vector<std::string> plan_lines;
    bool plans_built = false;

    void build_plans();
    void prepare_primitive(const Primitive& prim, PrimPlan& p);
    long stride_of(const std::string& axis, const std::string& tensor) const;

    // Führt prim mit dem generierten Kernel aus; false = kein Kernel anwendbar.
    bool try_evaluate_jit(const Primitive& prim, void** args,
                          const std::unordered_map<std::string, int>& active_loops);

    long long get_offset(const std::string& tensor_name,
                         const std::unordered_map<std::string, int>& active_loops);

    void evaluate_primitive(const Primitive& prim, void** args,
                            const std::unordered_map<std::string, int>& active_loops);

    void evaluate_node(const std::shared_ptr<ScheduleNode>& node, void** args,
                       std::unordered_map<std::string, int> active_loops);

    bool try_evaluate_neon_contraction(const Primitive& prim, void** args,
                                       const std::unordered_map<std::string, int>& active_loops);

    // Helpers für Primitiveauswertung mit noch fehlenden Achsen
    void lower_primitive_recursive(const Primitive& prim, void** args,
                                   std::unordered_map<std::string, int>& active_loops,
                                   const std::vector<std::string>& prim_axes,
                                   size_t depth);
};
