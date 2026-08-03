// A6: Tests fuer die Messschleife (bench_loop.hpp).
//
// Getestet wird mit einer SIMULIERTEN Uhr: runChunk(n) laesst die Uhr um
// n * perCall vorlaufen. Damit durchlaeuft der Test exakt dieselbe Schleife wie
// der Produktivcode (runBlockLoop ist der einzige Schleifenkoerper, beide
// Aufrufer benutzen ihn) -- nur eben mit vorgegebenen Kernel-Zeiten statt
// echten Messungen. So sind die Grenzfaelle deterministisch pruefbar, ohne
// minutenlange Benchmarks.
//
// Die Zahlen stammen aus der realen Analyse:
//   763 ms  = gemessene Kernel-Zeit von abc-dca-bd @ 384er (parallel)
//   0.0001 ms = ~0.1 us, der 6out-Mikrokernel aus dem C1-Kommentar
#include "bench_loop.hpp"

#include <cstdio>
#include <cmath>
#include <string>

static int g_fail = 0;
static int g_run = 0;

static void check(bool ok, const std::string& name, const std::string& detail = "")
{
    ++g_run;
    if (ok) {
        std::printf("  [ok]   %s\n", name.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s   %s\n", name.c_str(), detail.c_str());
    }
}

// Simulierte Uhr: jeder Aufruf kostet exakt perCall Millisekunden.
struct FakeClock {
    double perCall;
    double now = 0.0;
    void   run(long n) { now += static_cast<double>(n) * perCall; }
    double elapsed() const { return now; }
};

static BlockResult oneRep(const BlockPolicy& p, double perCall)
{
    FakeClock c{perCall};
    return runBlockLoop(p, [&](long n) { c.run(n); }, [&] { return c.elapsed(); });
}

// Mehrere Wiederholungen (die Uhr startet je Wiederholung neu, wie im Original).
static BlockResult nReps(const BlockPolicy& p, double perCall, int reps)
{
    BlockResult tot;
    for (int i = 0; i < reps; ++i) {
        BlockResult r = oneRep(p, perCall);
        tot.calls += r.calls;
        tot.elapsedMs += r.elapsedMs;
        tot.reads += r.reads;
    }
    return tot;
}

int main()
{
    // Parametersaetze wie im Produktivcode.
    BlockPolicy search;                       // benchmark.cpp: Such-Messung
    search.minWindowMs = 20.0;  search.capMs = 300.0;
    BlockPolicy final_;                       // main.cpp: [PERFORMANCE]-Messung
    final_.minWindowMs = 50.0;  final_.capMs = 1000.0;

    const double SLOW = 763.0;      // langsamer Kernel (ms/Aufruf)
    const double FAST = 0.0001;     // Mikrokernel (ms/Aufruf)

    std::printf("\n=== A6: Messschleife ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- Default ist das alte Verhalten (Reproduzierbarkeit) --\n");
    // ---------------------------------------------------------------
    check(BlockPolicy{}.adaptive == false,
          "Default-Policy ist NICHT adaptiv");

    {
        BlockPolicy p = search;              // adaptive = false
        BlockResult r = oneRep(p, SLOW);
        check(r.calls == 64,
              "alt: langsamer Kernel -> genau ein Block a 64 Aufrufe",
              "calls=" + std::to_string(r.calls));
        BlockResult f = oneRep(p, FAST);
        check(f.calls % 64 == 0,
              "alt: Aufrufzahl ist immer ein Vielfaches von 64",
              "calls=" + std::to_string(f.calls));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Der Fehler: fixer Block sprengt den Deckel --\n");
    // ---------------------------------------------------------------
    {
        BlockPolicy p = search;
        BlockResult r = oneRep(p, SLOW);
        // 64 x 763 ms = 48832 ms bei einem Deckel von 300 ms.
        check(std::fabs(r.elapsedMs - 64 * SLOW) < 1e-6,
              "alt: Wiederholung dauert 64 x Kernel-Zeit",
              "elapsed=" + std::to_string(r.elapsedMs));
        check(r.elapsedMs > 100.0 * p.capMs,
              "alt: Deckel wird um mehr als Faktor 100 ueberschritten",
              "elapsed=" + std::to_string(r.elapsedMs) +
              " cap=" + std::to_string(p.capMs));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Adaptiv haelt den Deckel ein --\n");
    // ---------------------------------------------------------------
    {
        BlockPolicy p = search; p.adaptive = true;
        BlockResult r = oneRep(p, SLOW);
        check(r.calls == 1,
              "adaptiv: langsamer Kernel -> genau EIN Aufruf",
              "calls=" + std::to_string(r.calls));
        // Der Deckel kann nur zwischen Bloecken greifen; mehr als einen
        // ueberzaehligen Aufruf darf er nie kosten.
        check(r.elapsedMs <= p.capMs + SLOW + 1e-9,
              "adaptiv: Deckel hoechstens um EINEN Aufruf ueberschritten",
              "elapsed=" + std::to_string(r.elapsedMs));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Adaptiv veraendert den MESSWERT nicht --\n");
    // ---------------------------------------------------------------
    {
        for (double perCall : {SLOW, 1.0, 0.05, FAST}) {
            BlockPolicy a = search; a.adaptive = false;
            BlockPolicy b = search; b.adaptive = true;
            const double va = oneRep(a, perCall).perCallMs();
            const double vb = oneRep(b, perCall).perCallMs();
            const double rel = std::fabs(vb - va) / va;
            check(rel < 1e-9,
                  "gleicher Messwert bei " + std::to_string(perCall) + " ms/Aufruf",
                  "alt=" + std::to_string(va) + " adaptiv=" + std::to_string(vb));
        }
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Adaptiv erreicht bei schnellen Kernels das Messfenster --\n");
    // ---------------------------------------------------------------
    {
        BlockPolicy p = search; p.adaptive = true;
        BlockResult r = oneRep(p, FAST);
        check(r.elapsedMs >= p.minWindowMs,
              "adaptiv: Mindest-Messfenster erreicht",
              "elapsed=" + std::to_string(r.elapsedMs));
        check(r.calls >= 200000,
              "adaptiv: Mikrokernel wird oft genug wiederholt",
              "calls=" + std::to_string(r.calls));
        // Timer-Overhead: ~25 ns je Ablesung gegen die Gesamtzeit.
        const double timerCostMs = r.reads * 0.000025;
        check(timerCostMs / r.elapsedMs < 0.01,
              "adaptiv: Timer-Overhead unter 1 %",
              "reads=" + std::to_string(r.reads));
        check(r.reads <= 20,
              "adaptiv: wenige Timer-Ablesungen (geometrisches Wachstum)",
              "reads=" + std::to_string(r.reads));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Blockgroesse: Wachstum und Timer-Untergrenze --\n");
    // ---------------------------------------------------------------
    {
        BlockPolicy p = search; p.adaptive = true;
        // Wachstum je Runde hoechstens Faktor 8.
        check(nextChunkSize(p, 1, 1, 1e-9) <= 8,
              "Wachstum pro Runde auf Faktor 8 begrenzt",
              "chunk=" + std::to_string(nextChunkSize(p, 1, 1, 1e-9)));
        // Timer-Untergrenze: kurz vor dem Fenster fehlen rechnerisch nur ~10
        // Aufrufe (0.001 ms / 0.0001 ms) -- so ein Block waere zu kurz, um die
        // Timer-Ablesung zu amortisieren. Die Untergrenze hebt ihn an.
        // Geprueft wird die EIGENSCHAFT, nicht die exakte Zahl: ceil() auf einer
        // Fliesskomma-Grenze liefert je nach Rundung 25 oder 26.
        const long done = 199990;
        const double blockMs = 19.999;
        const double perCall = blockMs / static_cast<double>(done);
        const long got = nextChunkSize(p, 100000, done, blockMs);
        check(got * perCall >= p.timerFloorMs,
              "Timer-Untergrenze: Block dauert mind. 2,5 us",
              "chunk=" + std::to_string(got) +
              " blockMs=" + std::to_string(got * perCall));
        check(got > 10,
              "Timer-Untergrenze hebt den Block ueber die reine Fensterluecke",
              "chunk=" + std::to_string(got));
        // Nie kleiner als 1.
        check(nextChunkSize(p, 1, 1, 1e9) >= 1, "Blockgroesse nie unter 1");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Finale [PERFORMANCE]-Messung (main.cpp, 5 Wiederholungen) --\n");
    // ---------------------------------------------------------------
    {
        BlockPolicy a = final_; a.adaptive = false;
        BlockPolicy b = final_; b.adaptive = true;
        BlockResult ra = nReps(a, SLOW, 5);
        BlockResult rb = nReps(b, SLOW, 5);
        // Das ist die real beobachtete Zahl: 5 x 64 x 763 ms = 244 s.
        check(ra.calls == 320,
              "alt: 5 Wiederholungen a 64 Aufrufe = 320",
              "calls=" + std::to_string(ra.calls));
        check(std::fabs(ra.elapsedMs / 1000.0 - 244.16) < 0.1,
              "alt: Endmessung dauert ~244 s (real gemessen: 238-252 s)",
              "s=" + std::to_string(ra.elapsedMs / 1000.0));
        check(rb.calls == 5,
              "adaptiv: 5 Wiederholungen a 1 Aufruf = 5",
              "calls=" + std::to_string(rb.calls));
        check(std::fabs(rb.elapsedMs / 1000.0 - 3.815) < 0.01,
              "adaptiv: Endmessung dauert ~3,8 s",
              "s=" + std::to_string(rb.elapsedMs / 1000.0));
        check(std::fabs(ra.elapsedMs / rb.elapsedMs - 64.0) < 1e-6,
              "Ersparnis exakt Faktor 64 (= fixe Blockgroesse)",
              "faktor=" + std::to_string(ra.elapsedMs / rb.elapsedMs));
        check(std::fabs(ra.perCallMs() - rb.perCallMs()) < 1e-9,
              "trotzdem derselbe Messwert");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Grenzfaelle --\n");
    // ---------------------------------------------------------------
    {
        BlockPolicy p = search; p.adaptive = true;
        // Kernel exakt auf dem Deckel.
        BlockResult r = oneRep(p, p.capMs);
        check(r.calls == 1, "adaptiv: Kernel genau auf Deckel -> ein Aufruf",
              "calls=" + std::to_string(r.calls));
        // Kernel zwischen Messfenster und Deckel: ein Aufruf reicht fuer das
        // Fenster, der Deckel ist nicht erreicht.
        BlockResult m = oneRep(p, 25.0);
        check(m.calls == 1 && m.elapsedMs >= p.minWindowMs,
              "adaptiv: Kernel ueber dem Fenster -> ein Aufruf genuegt",
              "calls=" + std::to_string(m.calls));
        // Safety-Ceiling greift bei perCall = 0.
        BlockPolicy z = search; z.adaptive = true; z.safetyCeil = 1024;
        BlockResult zr = oneRep(z, 0.0);
        check(zr.calls >= z.safetyCeil,
              "Safety-Ceiling beendet die Schleife bei Kernel-Zeit 0",
              "calls=" + std::to_string(zr.calls));
    }

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
