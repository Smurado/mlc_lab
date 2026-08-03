#pragma once
// A6: Gemeinsame Messschleife fuer benchmark.cpp (Such-Messung) und main.cpp
// (finale [PERFORMANCE]-Messung).
//
// Warum ein eigener Header: die Schleife stand zweimal inline in zwei
// Funktionen, mit unterschiedlichen Konstanten (20/300/3 Wiederholungen vs.
// 50/1000/5 Wiederholungen) und einem Unterschied im Abbruchverhalten. Ein Fix
// an einer Stelle liess die andere unangetastet.
//
// Das Problem, das die adaptive Blockgroesse loest:
// Der Zeit-Deckel (capMs) kann nur ZWISCHEN zwei Bloecken geprueft werden. Bei
// fixer Blockgroesse 64 laeuft ein Block bei einem langsamen Kernel weit ueber
// den Deckel hinaus -- 64 x 763 ms = 48,8 s bei einem Deckel von 300 ms. Der
// Block existiert aber nur, um die Timer-Ablesung (~25 ns) zu amortisieren; bei
// einem 763-ms-Kernel genuegt dafuer EIN Aufruf. Adaptiv wird die Blockgroesse
// aus der gemessenen Kernel-Zeit bestimmt: gross bei schnellen Kernels, klein
// bei langsamen.
//
// Default bleibt der fixe Block (adaptive=false), damit frueher gemessene
// Zahlen reproduzierbar bleiben. Umschalten ueber TEIR_BENCH_ADAPTIVE=1.

#include <algorithm>
#include <cmath>

struct BlockPolicy {
    bool   adaptive     = false;
    double minWindowMs  = 20.0;    // Mindest-Messfenster je Wiederholung
    double capMs        = 300.0;   // Obergrenze je Wiederholung
    long   fixedChunk   = 64;      // Blockgroesse im nicht-adaptiven Modus
    // Timer-Budget: eine Ablesung (~25 ns) soll unter 1 % der Blockzeit
    // bleiben -> ein Block muss mindestens 100 x 25 ns = 2,5 us dauern.
    double timerFloorMs = 0.0025;
    long   safetyCeil   = 50000000L;  // Notbremse gegen pathologische Schleifen
};

struct BlockResult {
    long   calls     = 0;    // tatsaechlich ausgefuehrte Kernel-Aufrufe
    double elapsedMs = 0.0;  // Zeit fuer diese Wiederholung
    int    reads     = 0;    // Timer-Ablesungen (fuer Overhead-Abschaetzung)

    // Gemessene Zeit pro Aufruf -- der eigentliche Messwert.
    double perCallMs() const { return calls > 0 ? elapsedMs / calls : elapsedMs; }
};

// Naechste Blockgroesse aus der bisher gemessenen Rate.
inline long nextChunkSize(const BlockPolicy& p, long chunk, long done, double blockMs)
{
    const double perCall = blockMs / static_cast<double>(done);
    if (perCall <= 0.0) {
        // Noch unter Timer-Aufloesung -> geometrisch wachsen.
        return std::min(chunk * 8, p.safetyCeil);
    }
    // So viele Aufrufe, wie bis zum Messfenster noch fehlen ...
    long want = static_cast<long>(std::ceil((p.minWindowMs - blockMs) / perCall));
    // ... aber genug, dass die Timer-Ablesung nicht ins Gewicht faellt ...
    const long timerMin = static_cast<long>(std::ceil(p.timerFloorMs / perCall));
    want = std::max(want, timerMin);
    // ... und hoechstens Faktor 8 Wachstum pro Runde (begrenzt Ueberschiessen).
    return std::max(1L, std::min(want, chunk * 8));
}

// Eine Mess-Wiederholung.
//   runChunk(n)  fuehrt n Kernel-Aufrufe aus
//   elapsedMs()  liefert die seit Beginn der Wiederholung vergangene Zeit
// Beide werden injiziert, damit der Test EXAKT diese Schleife mit einer
// simulierten Uhr durchlaufen kann statt einer nachgebauten Kopie.
template <class RunChunk, class ElapsedMs>
inline BlockResult runBlockLoop(const BlockPolicy& p, RunChunk runChunk,
                                ElapsedMs elapsedMs)
{
    BlockResult r;
    long chunk = p.adaptive ? 1L : p.fixedChunk;
    while (r.elapsedMs < p.minWindowMs && r.calls < p.safetyCeil) {
        runChunk(chunk);
        r.calls += chunk;
        r.elapsedMs = elapsedMs();
        ++r.reads;
        if (r.elapsedMs >= p.capMs) break;
        if (p.adaptive) chunk = nextChunkSize(p, chunk, r.calls, r.elapsedMs);
    }
    return r;
}
