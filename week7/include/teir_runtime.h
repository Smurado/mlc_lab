// Rekursive Laufzeitumgebung fuer TEIR-Programme.
//
// Ersetzt den Weg "C++-Quelltext erzeugen -> clang++ aufrufen -> .so laden".
// Stattdessen wird das Schleifennest zur Laufzeit REKURSIV ueber die
// Iterationsknoten abgelaufen, und der innerste Kernel kommt aus dem EIGENEN
// Codegenerator (mini_jit aus Woche 5/6), der Maschinencode direkt im
// Arbeitsspeicher erzeugt.
//
// Damit entfaellt jede Datei und jeder externe Compiler-Aufruf.
#pragma once

#include "teir.h"

#include "../../week6/Unary.h"
#include "../../week6/Gemm.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class TEIRRuntime {
public:
    explicit TEIRRuntime(TEIRProgram p) : prog(std::move(p)) {}

    // Wiederverwendbares Callable: nimmt das Array der Tensor-Pointer entgegen,
    // genau wie der frueher per dlopen geladene Funktionszeiger.
    using Kernel = std::function<void(void**)>;

    // Bereitet die Ausfuehrung vor (Kernel generieren, Primitive analysieren)
    // und liefert das Callable. Leeres Callable bei Fehler.
    Kernel build();

    // Diagnose: welcher Weg wurde je Primitive gewaehlt?
    const std::vector<std::string>& plan() const { return plan_lines; }

private:
    // Wie ein Primitive ausgefuehrt wird.
    struct PrimPlan {
        enum class Kind { ZeroJit, CopyJit, GemmJit, GemmScalar, Scalar } kind{};

        // Achsen des Primitives (aus axes_map), in Elementen umgerechnet.
        std::string mAxis, nAxis, kAxis;
        int mExt = 0, nExt = 0, kExt = 0;

        // Strides in ELEMENTEN (die .teir-Datei gibt Bytes an).
        long inA_m = 0, inA_k = 0;      // in0
        long inB_n = 0, inB_k = 0;      // in1
        long outC_m = 0, outC_n = 0;    // out
        long inCopy_m = 0, inCopy_n = 0; // in (Copy)

        // Generierte Kernel (leben so lange wie die Runtime).
        std::shared_ptr<mini_jit::Unary> unary;
        std::shared_ptr<mini_jit::Gemm>  gemm;
        mini_jit::Unary::kernel_t unaryFn = nullptr;
        mini_jit::Gemm::kernel_t  gemmFn  = nullptr;

        // Muss A vor dem GEMM in M-zusammenhaengendes Layout gepackt werden?
        bool packA = false;
    };

    TEIRProgram prog;
    std::unordered_map<std::string, PrimPlan> plans;
    std::vector<std::string> plan_lines;

    bool preparePrimitive(const Primitive& prim, PrimPlan& out);
    int argIndex(const std::string& tensor) const;
    long strideOf(const std::string& axis, const std::string& tensor) const;

    void run(const std::shared_ptr<ScheduleNode>& node,
             std::unordered_map<std::string, long>& idx,
             void** args) const;
    void runInvoke(const InvokeNode& inv,
                   const std::unordered_map<std::string, long>& idx,
                   void** args) const;
};
