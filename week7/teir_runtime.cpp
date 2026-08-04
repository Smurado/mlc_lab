#include "include/teir_runtime.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>

namespace {

// Achsenname aus "@m1" bzw. "m1".
std::string stripAt(const std::string& s) {
    return (!s.empty() && s[0] == '@') ? s.substr(1) : s;
}

// Erste Achse einer Rolle (M/N/K) aus der axes_map, ohne fuehrendes '@'.
std::string roleAxis(const Primitive& p, const std::string& role) {
    auto it = p.axes_map.find(role);
    if (it == p.axes_map.end() || it->second.empty()) return {};
    return stripAt(it->second[0]);
}

// Gepackter A-Kachel-Puffer je Thread.
//
// mini_jit::Gemm erwartet A M-zusammenhaengend (Spalten-Layout). Die
// TEIR-Beispiele liefern in0 dagegen K-zusammenhaengend. Deshalb wird die
// Kachel einmal umkopiert.
//
// Wichtig fuer die Laufzeit: der Puffer merkt sich, welche Quelladresse zuletzt
// gepackt wurde. Da die Schleifenordnung der Beispiele k0 -> m0 -> n0 lautet,
// bleibt A ueber die gesamte n0-Schleife gleich -- es wird also nur einmal je
// (k0, m0) gepackt statt einmal je Aufruf.
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
        // Ziel: A(i,l) = data[l * m + i]  (M zusammenhaengend)
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

// Position eines Tensors in der Argumentliste.
//
// Frueher stand hier fest args[0]=in0, args[1]=in1, args[2]=out. Das stimmt nur
// fuer Programme mit drei Tensoren -- transposition.teir hat nur `in` und `out`,
// dort ist out also args[1]. Die Reihenfolge steht in prog.tensors.
int TEIRRuntime::argIndex(const std::string& tensor) const {
    for (size_t i = 0; i < prog.tensors.size(); ++i)
        if (prog.tensors[i] == tensor) return static_cast<int>(i);
    return -1;
}

long TEIRRuntime::strideOf(const std::string& axis, const std::string& tensor) const {
    auto a = prog.axes.find(axis);
    if (a == prog.axes.end()) return 0;
    auto s = a->second.strides.find(tensor);
    if (s == a->second.strides.end()) return 0;
    return s->second / 4; // .teir gibt Bytes an, wir rechnen in float-Elementen
}

bool TEIRRuntime::preparePrimitive(const Primitive& prim, PrimPlan& p) {
    p.mAxis = roleAxis(prim, "M");
    p.nAxis = roleAxis(prim, "N");
    p.kAxis = roleAxis(prim, "K");

    auto extentOf = [&](const std::string& ax) -> int {
        auto it = prog.axes.find(ax);
        return it == prog.axes.end() ? 0 : it->second.extent;
    };
    p.mExt = extentOf(p.mAxis);
    p.nExt = extentOf(p.nAxis);
    p.kExt = extentOf(p.kAxis);

    p.outC_m = strideOf(p.mAxis, "out");
    p.outC_n = strideOf(p.nAxis, "out");

    if (prim.kind == "Zero") {
        // Der Unary-Kernel schreibt 16 zusammenhaengende Floats je Schritt --
        // seine "M"-Richtung muss also die zusammenhaengende Achse sein.
        // Bei den TEIR-Beispielen ist out N-zusammenhaengend, deshalb werden
        // die Rollen getauscht. Fuer "alles auf null" ist das unerheblich.
        const long contig = (p.outC_n == 1) ? p.nExt : p.mExt;
        const long other  = (p.outC_n == 1) ? p.mExt : p.nExt;
        const long ld     = (p.outC_n == 1) ? p.outC_m : p.outC_n;
        if (contig % 16 == 0 && ld > 0) {
            p.unary = std::make_shared<mini_jit::Unary>();
            if (p.unary->generate(static_cast<uint32_t>(contig), static_cast<uint32_t>(other),
                                  0, mini_jit::Unary::dtype_t::fp32,
                                  mini_jit::Unary::ptype_t::zero)
                == mini_jit::Unary::error_t::success) {
                p.unaryFn = p.unary->get_kernel();
                p.kind = PrimPlan::Kind::ZeroJit;
                p.inA_m = contig; p.inA_k = other; p.inB_n = ld; // Parameter merken
                plan_lines.push_back("  " + prim.name + " (Zero) -> mini_jit::Unary "
                                     + std::to_string(contig) + "x" + std::to_string(other));
                return true;
            }
        }
        p.kind = PrimPlan::Kind::Scalar;
        plan_lines.push_back("  " + prim.name + " (Zero) -> skalar (Groesse nicht 16-teilbar)");
        return true;
    }

    if (prim.kind == "Copy") {
        p.inCopy_m = strideOf(p.mAxis, "in");
        p.inCopy_n = strideOf(p.nAxis, "in");
        p.kind = PrimPlan::Kind::Scalar;
        plan_lines.push_back("  " + prim.name + " (Copy) -> skalar");
        return true;
    }

    if (prim.kind == "Contraction") {
        p.inA_m = strideOf(p.mAxis, "in0");
        p.inA_k = strideOf(p.kAxis, "in0");
        p.inB_n = strideOf(p.nAxis, "in1");
        p.inB_k = strideOf(p.kAxis, "in1");

        // mini_jit::Gemm erwartet:
        //   A  M-zusammenhaengend, k-Schritt = ld_a
        //   B  N-zusammenhaengend, k-Schritt = ld_b
        //   C  M-zusammenhaengend, n-Schritt = ld_c
        // Die TEIR-Beispiele liefern C und B N-zusammenhaengend, A dagegen
        // K-zusammenhaengend. Durch Rollentausch (i<->n, j<->m) passen B und C;
        // A muss gepackt werden.
        const bool swapped = (p.outC_n == 1 && p.inB_n == 1);
        if (swapped && p.mExt % 16 == 0 && p.nExt % 16 == 0) {
            p.gemm = std::make_shared<mini_jit::Gemm>();
            if (p.gemm->generate(static_cast<uint32_t>(p.nExt), static_cast<uint32_t>(p.mExt),
                                 static_cast<uint32_t>(p.kExt), 0, 0, 0,
                                 mini_jit::Gemm::dtype_t::fp32)
                == mini_jit::Gemm::error_t::success) {
                p.gemmFn = p.gemm->get_kernel();
                p.kind = PrimPlan::Kind::GemmJit;
                p.packA = true;
                plan_lines.push_back("  " + prim.name + " (Contraction) -> mini_jit::Gemm "
                                     + std::to_string(p.nExt) + "x" + std::to_string(p.mExt)
                                     + "x" + std::to_string(p.kExt) + " (A wird gepackt)");
                return true;
            }
        }
        p.kind = PrimPlan::Kind::GemmScalar;
        plan_lines.push_back("  " + prim.name + " (Contraction) -> skalar "
                             "(Layout oder Groesse passt nicht zum SME-Kernel)");
        return true;
    }

    p.kind = PrimPlan::Kind::Scalar;
    return true;
}

TEIRRuntime::Kernel TEIRRuntime::build() {
    plan_lines.clear();
    plan_lines.push_back("Ausfuehrungsplan fuer @" + prog.name + ":");
    for (auto const& [name, prim] : prog.primitives) {
        PrimPlan p;
        if (!preparePrimitive(prim, p)) return {};
        plans[name] = std::move(p);
    }

    return [this](void** args) {
        std::unordered_map<std::string, long> idx;
        for (auto const& root : prog.roots) run(root, idx, args);
    };
}

void TEIRRuntime::run(const std::shared_ptr<ScheduleNode>& node,
                      std::unordered_map<std::string, long>& idx,
                      void** args) const {
    if (node->type == NodeType::Invoke) {
        runInvoke(static_cast<const InvokeNode&>(*node), idx, args);
        return;
    }

    const auto& it = static_cast<const IterNode&>(*node);
    auto ax = prog.axes.find(it.axis);
    const long extent = (ax == prog.axes.end()) ? 1 : ax->second.extent;

    if (it.policy == "parallel") {
        // Jeder Thread braucht eine eigene Index-Abbildung.
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < extent; ++i) {
            std::unordered_map<std::string, long> local = idx;
            local[it.axis] = i;
            for (auto const& c : it.children) run(c, local, args);
        }
    } else {
        for (long i = 0; i < extent; ++i) {
            idx[it.axis] = i;
            for (auto const& c : it.children) run(c, idx, args);
        }
        idx.erase(it.axis);
    }
}

void TEIRRuntime::runInvoke(const InvokeNode& inv,
                            const std::unordered_map<std::string, long>& idx,
                            void** args) const {
    // Guard "first(@axis)": nur im ersten Durchlauf dieser Achse ausfuehren.
    if (!inv.guard.empty()) {
        const size_t a = inv.guard.find('@');
        const size_t b = inv.guard.find(')');
        if (a != std::string::npos && b != std::string::npos) {
            const std::string gAxis = inv.guard.substr(a + 1, b - a - 1);
            auto g = idx.find(gAxis);
            if (g != idx.end() && g->second != 0) return;
        }
    }

    auto pit = plans.find(inv.primitive);
    auto prit = prog.primitives.find(inv.primitive);
    if (pit == plans.end() || prit == prog.primitives.end()) return;
    const PrimPlan& p = pit->second;

    // Basisversatz aus den AKTIVEN Schleifen (die Primitive-eigenen Achsen
    // uebernimmt der Kernel selbst).
    auto baseOffset = [&](const char* tensor) -> long {
        long off = 0;
        for (auto const& [axName, i] : idx) {
            if (axName == p.mAxis || axName == p.nAxis || axName == p.kAxis) continue;
            off += i * strideOf(axName, tensor);
        }
        return off;
    };

    const int iOut = argIndex("out");
    if (iOut < 0) return;
    float* out = static_cast<float*>(args[iOut]) + baseOffset("out");

    switch (p.kind) {
    case PrimPlan::Kind::ZeroJit:
        // Signatur: (a, b, ld_a, ld_b); a wird beim Zero-Kernel nicht gelesen.
        p.unaryFn(nullptr, out, p.inB_n, p.inB_n);
        return;

    case PrimPlan::Kind::GemmJit: {
        const float* a0 = static_cast<const float*>(args[argIndex("in0")]) + baseOffset("in0");
        const float* b0 = static_cast<const float*>(args[argIndex("in1")]) + baseOffset("in1");
        // Rollentausch: der Kernel rechnet C'(n,m) += A'(n,k) * B'(k,m).
        //   A' = in1 (N-zusammenhaengend)        -> ld = k-Schritt von in1
        //   B' = gepacktes in0 (M-zusammenhaengend, k-Schritt = mExt)
        //   C' = out (N-zusammenhaengend)        -> ld = m-Schritt von out
        const float* packed = packBuffer().get(a0, p.mExt, p.kExt, p.inA_m, p.inA_k);
        p.gemmFn(b0, packed, out, p.inB_k, p.mExt, p.outC_m);
        return;
    }

    case PrimPlan::Kind::GemmScalar: {
        const float* a0 = static_cast<const float*>(args[argIndex("in0")]) + baseOffset("in0");
        const float* b0 = static_cast<const float*>(args[argIndex("in1")]) + baseOffset("in1");
        for (long i = 0; i < p.mExt; ++i)
            for (long j = 0; j < p.nExt; ++j) {
                float acc = 0.0f;
                for (long l = 0; l < p.kExt; ++l)
                    acc += a0[i * p.inA_m + l * p.inA_k] * b0[j * p.inB_n + l * p.inB_k];
                out[i * p.outC_m + j * p.outC_n] += acc;
            }
        return;
    }

    case PrimPlan::Kind::CopyJit:
    case PrimPlan::Kind::Scalar:
    default:
        if (prit->second.kind == "Zero") {
            for (long i = 0; i < p.mExt; ++i)
                for (long j = 0; j < p.nExt; ++j)
                    out[i * p.outC_m + j * p.outC_n] = 0.0f;
        } else if (prit->second.kind == "Copy") {
            const float* in = static_cast<const float*>(args[argIndex("in")]) + baseOffset("in");
            // Schleifenreihenfolge nach dem AUSGABE-Stride waehlen: innen laeuft
            // die Achse, die in `out` den kleineren Schritt hat.
            //
            // Gemessen an transposition.teir: 1,92 -> 16,45 GB/s (Faktor 8,6).
            // Zusaetzliches Kacheln in 16x16-Bloecken wurde probiert und war
            // LANGSAMER (15,0 GB/s) -- die Kachel ist mit 48x32 = 6 KB ohnehin
            // L1-resident, das Blocken fuegt nur Schleifenaufwand hinzu.
            //
            // Warum das entscheidend ist: In transposition.teir hat die d-Achse
            // in `out` einen Schritt von 565248 Elementen = 35328 Cache-Zeilen.
            // Das ist ein glattes Vielfaches der Set-Anzahl -- alle Adressen der
            // inneren Schleife landen dadurch im SELBEN Cache-Set und verdraengen
            // sich gegenseitig. Mit der d-Achse aussen schreibt die innere
            // Schleife dagegen zusammenhaengend.
            if (p.outC_m <= p.outC_n) {
                for (long j = 0; j < p.nExt; ++j)
                    for (long i = 0; i < p.mExt; ++i)
                        out[i * p.outC_m + j * p.outC_n] = in[i * p.inCopy_m + j * p.inCopy_n];
            } else {
                for (long i = 0; i < p.mExt; ++i)
                    for (long j = 0; j < p.nExt; ++j)
                        out[i * p.outC_m + j * p.outC_n] = in[i * p.inCopy_m + j * p.inCopy_n];
            }
        }
        return;
    }
}
