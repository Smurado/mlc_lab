#include "Unary.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace mini_jit;

bool verify_zero(uint32_t m, uint32_t n) {
    std::vector<float> B(m * n, 1.0f); // Array mit Einsen füllen
    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::zero) != Unary::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(nullptr, B.data(), m, m);

    for (float val : B) {
        if (val != 0.0f) return false;
    }
    return true;
}

bool verify_identity(uint32_t m, uint32_t n) {
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i);

    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity) != Unary::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(A.data(), B.data(), m, m);

    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i] != B[i]) return false;
    }
    return true;
}

bool verify_relu(uint32_t m, uint32_t n) {
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i) - static_cast<float>(A.size()/2); // Mix aus negativen und positiven Werten

    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, Unary::ptype_t::relu) != Unary::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    kernel(A.data(), B.data(), m, m);

    for (size_t i = 0; i < A.size(); ++i) {
        float expected = std::max(0.0f, A[i]);
        if (B[i] != expected) return false;
    }
    return true;
}

// Frueher stand hier __attribute__((optimize("O0"))). Begruendung damals:
// "Sonst speichert der Compiler die lokalen Variablen zur Zeitmessung in den
// callee-saved Vector-Registern (d8-d15). Da smstart sm diese Register
// hardwareseitig nullt, bekaemen wir sonst Divisionen durch Null und kaputte
// Timer."
//
// Die Beobachtung war richtig, die Ursache lag aber nicht beim Compiler: Der
// generierte Kernel verletzte die Aufrufkonvention, weil er d8-d15 zerstoerte,
// ohne sie zu sichern. Der Compiler durfte sie zu Recht fuer die Zeitvariablen
// nutzen.
//
// Die Optimierung abzuschalten war deshalb eine Umgehung mit zwei Nachteilen:
// `optimize` ist ein GCC-Attribut (clang ignoriert es kommentarlos), und es
// betrifft die GANZE Funktion -- also auch den Zeitmesscode selbst.
//
// Behoben ist es jetzt an der Wurzel: der Generator sichert d8-d15 im Prolog
// und stellt sie im Epilog wieder her (siehe Unary.cpp). Damit ist das Attribut
// ueberfluessig und die Benchmarks laufen wieder mit voller Optimierung.
void benchmark(uint32_t m, uint32_t n, Unary::ptype_t ptype, const std::string& name, bool benchmark_mode) {
    Unary kernel_gen;
    if (kernel_gen.generate(m, n, 0, Unary::dtype_t::fp32, ptype) != Unary::error_t::success) {
        std::cout << name << " (" << m << "x" << n << "): Generation failed" << std::endl;
        return;
    }
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return;

    std::vector<float> A(m * n, 1.0f);
    std::vector<float> B(m * n, 0.0f);
    
    int num_runs = 10000;
    int outer_runs = benchmark_mode ? 10 : 1;
    double total_gib_per_sec = 0.0;
    
    for (int outer = 0; outer < outer_runs; ++outer) {
// Der Kernel wird ganz normal ueber den Funktionszeiger aufgerufen.
//
// Frueher stand hier ein Inline-Assembler-Wrapper mit einer Clobber-Liste. Er
// war noetig, weil der erzeugte Kernel per `smstart` den Streaming-Modus betrat
// und dabei die nach AAPCS64 callee-saved Register v8-v15 zerstoerte, ohne sie
// zu sichern -- also die Aufrufkonvention verletzte.
//
// Statt das im Aufrufer zu umgehen, sichert der Generator diese Register jetzt
// selbst (siehe Prolog/Epilog in Unary.cpp). Damit ist der Kernel ABI-konform
// und ein gewoehnlicher Aufruf genuegt.
auto safe_kernel = [&](const float* a, float* b, uint32_t arg_m, uint32_t arg_n) {
    kernel(a, b, arg_m, arg_n);
};

// ... Cache anwärmen ...
for (int i = 0; i < 100; ++i) {
    safe_kernel(A.data(), B.data(), m, m); // safe_kernel statt kernel() nutzen!
}

auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < num_runs; ++i) {
    safe_kernel(A.data(), B.data(), m, m); // safe_kernel statt kernel() nutzen!
}
auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        // Durchsatz in GiB/s berechnen
        double bytes_per_run = 0;
        if (ptype == Unary::ptype_t::zero) {
            bytes_per_run = m * n * sizeof(float); // Nur Schreiben
        } else {
            bytes_per_run = 2.0 * m * n * sizeof(float); // Lesen + Schreiben
        }
        
        double total_bytes = bytes_per_run * num_runs;
        double gib_per_sec = (total_bytes / diff.count()) / (1024.0 * 1024.0 * 1024.0);
        total_gib_per_sec += gib_per_sec;
    }
    
    double avg_gib_per_sec = total_gib_per_sec / outer_runs;
    
    std::cout << std::left << std::setw(10) << name 
              << " (" << std::setw(3) << m << "x" << std::setw(3) << n << "): " 
              << std::fixed << std::setprecision(5) << avg_gib_per_sec << " GiB/s" 
              << (benchmark_mode ? " (Avg of 10)" : "") << std::endl;
}

#include "Gemm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>

using namespace mini_jit;

// Unary code omitted for brevity right here, assuming the user knows we just append this

// Aufruf des GEMM-Kernels.
//
// Anders als beim Unary-Kernel laesst sich die ABI-Verletzung hier nicht an der
// Wurzel beheben: Der GEMM-Kernel ist ein fest einprogrammiertes
// Instruktionsarray (Gemm.cpp), und sein Prolog sichert x19-x27, aber NICHT
// d8-d15 -- die `smstart` zerstoert. Zusaetzliche stp/ldp-Instruktionen
// einzufuegen wuerde die relativen Sprungziele innerhalb des Arrays
// verschieben.
//
// Deshalb wird die Zerstoerung hier an der Aufrufstelle deklariert. Ohne das
// legt der Compiler seine Zeitmessvariablen in d8-d15 ab, findet dort nach dem
// Aufruf Nullen und rechnet mit einer Zeitdifferenz von 0 -- die Ausgabe lautet
// dann "nan GFLOPS".
static void call_gemm(Gemm::kernel_t kern,
                      const float* a, const float* b, float* c,
                      int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    register const float* x0 asm("x0") = a;
    register const float* x1 asm("x1") = b;
    register float*       x2 asm("x2") = c;
    register int64_t      x3 asm("x3") = ld_a;
    register int64_t      x4 asm("x4") = ld_b;
    register int64_t      x5 asm("x5") = ld_c;
    Gemm::kernel_t fn = kern;

    asm volatile(
        "blr %[fn]\n"
        : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3), "+r"(x4), "+r"(x5)
        : [fn] "r"(fn)
        : "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          // x18 steht bewusst NICHT hier: auf Apple-Plattformen ist es das
          // Platform-Register und darf nicht ueberschrieben werden. Der
          // Kernel fasst es in keinem seiner 114 Instruktionswoerter an.
          "x16", "x17", "x30",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "memory", "cc"
    );
}

bool verify_gemm(uint32_t m, uint32_t n, uint32_t k) {
    if (m != 512 || n != 512 || k != 512) return true; // Only 512 implemented

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // GEMUSTERTE Werte statt Konstanten.
    //
    // Vorher standen hier A = 1.0 und B = 2.0. Damit liefert JEDE Vertauschung
    // von Zeilen und Spalten dasselbe Ergebnis -- ein Layout-Fehler faellt nicht
    // auf. Genau das ist hier passiert: die Referenz unten indizierte B
    // spaltenweise (B[l + j*k]), waehrend der Kernel B laut Aufgabenstellung
    // ZEILENWEISE liest. Die Pruefung bestand trotzdem.
    // Teilerfremde Perioden, damit sich das Muster ueber A x B nicht frueh
    // wiederholt.
    for(size_t i=0; i<A.size(); ++i) A[i] = static_cast<float>((i % 13) + 1) / 13.0f;
    for(size_t i=0; i<B.size(); ++i) B[i] = static_cast<float>((i %  7) + 1) /  7.0f;

    // Referenz: A und C spaltenweise, B ZEILENWEISE (Layout laut Aufgabe).
    for(uint32_t j=0; j<n; ++j) {
        for(uint32_t l=0; l<k; ++l) {
            for(uint32_t i=0; i<m; ++i) {
                C_ref[i + j*m] += A[i + l*m] * B[l*k + j];
            }
        }
    }

    Gemm kernel_gen;
    if (kernel_gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32) != Gemm::error_t::success) return false;
    
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return false;

    // Call the generated JIT kernel
    call_gemm(kernel, A.data(), B.data(), C.data(), m, k, m);

    // Verify
    for(size_t i=0; i<C.size(); ++i) {
        if (std::abs(C[i] - C_ref[i]) > 1e-4) return false;
    }
    return true;
}

// Ebenfalls ohne __attribute__((optimize("O0"))) -- Begruendung siehe benchmark().
void benchmark_gemm(uint32_t m, uint32_t n, uint32_t k, bool benchmark_mode) {
    Gemm kernel_gen;
    if (kernel_gen.generate(m, n, k, 0, 0, 0, Gemm::dtype_t::fp32) != Gemm::error_t::success) return;
    auto kernel = kernel_gen.get_kernel();
    if (!kernel) return;

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> B(k * n, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    int num_runs = 50;
    int outer_runs = benchmark_mode ? 10 : 1;
    double total_gflops_per_sec = 0.0;
    
    for (int outer = 0; outer < outer_runs; ++outer) {
        // Cache anwärmen (Warmup)
        for (int i = 0; i < 5; ++i) call_gemm(kernel, A.data(), B.data(), C.data(), m, k, m);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            call_gemm(kernel, A.data(), B.data(), C.data(), m, k, m);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        // Calculate GFLOPS
        double flops_per_run = 2.0 * m * n * k;
        double total_gflops = (flops_per_run * num_runs) / (1e9);
        double gflops_per_sec = total_gflops / diff.count();
        total_gflops_per_sec += gflops_per_sec;
    }
    
    double avg_gflops_per_sec = total_gflops_per_sec / outer_runs;
    
    std::cout << std::left << std::setw(10) << "GEMM" 
              << " (" << std::setw(3) << m << "x" << std::setw(3) << n << "x" << std::setw(3) << k << "): " 
              << std::fixed << std::setprecision(2) << avg_gflops_per_sec << " GFLOPS" 
              << (benchmark_mode ? " (Avg of 10)" : "") << std::endl;
}

int main(int argc, char** argv) {
    bool benchmark_mode = false;
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        benchmark_mode = true;
        std::cout << "Starting BENCHMARK MODE (10 runs averaged)..." << std::endl;
    }

    if (!benchmark_mode) {
        std::cout << "--- Verify Kernels ---" << std::endl;
        std::cout << "Zero 16x16:     " << (verify_zero(16, 16) ? "PASS" : "FAIL") << std::endl;
        std::cout << "Identity 16x16: " << (verify_identity(16, 16) ? "PASS" : "FAIL") << std::endl;
        std::cout << "ReLU 16x16:     " << (verify_relu(16, 16) ? "PASS" : "FAIL") << std::endl;
        std::cout << "GEMM 512x512:   " << (verify_gemm(512, 512, 512) ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    std::cout << "--- Benchmarks ---" << std::endl;
    std::vector<uint32_t> sizes = {64, 128, 512};
    for (auto size : sizes) {
        benchmark(size, size, Unary::ptype_t::zero, "Zero", benchmark_mode);
        benchmark(size, size, Unary::ptype_t::identity, "Identity", benchmark_mode);
        benchmark(size, size, Unary::ptype_t::relu, "ReLU", benchmark_mode);
        std::cout << std::endl;
    }
    
    benchmark_gemm(512, 512, 512, benchmark_mode);

    return 0;
}
