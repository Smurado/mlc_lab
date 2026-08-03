#pragma once
// OOM-Schutz fuer das Benchmark-Harness: Tensoren oberhalb einer Elementgrenze
// werden nicht allokiert.
//
// Warum ein eigener Header: die Grenze stand zweimal im Code -- einmal in
// benchmark.cpp (Such-Messung) und einmal in main.cpp (Endmessung). Beim
// Env-Toggle wurde nur die erste angepasst, wodurch drei GETT-Faelle die Suche
// zwar durchliefen, aber vor der Endmessung abbrachen (Status "partial" ohne
// [PERFORMANCE]-Zeile). Eine Kopie, ein Verhalten, ein Test.
//
// Betroffen waren abcde-efbad-cf / -ecbfa-fd / -efcad-bf: deren in0 hat
// 107-143 Mio. Elemente und liegt damit ueber dem Default von 100 Mio.
// Ueber TEIR_MAX_TENSOR laesst sich die Grenze anheben (Angabe in ELEMENTEN,
// ein float-Element = 4 Byte, 100 Mio. entsprechen also ~400 MB pro Tensor).

#include <algorithm>
#include <cstdlib>

// Default: 100 Mio. Elemente (~400 MB als float).
constexpr int kDefaultMaxTensorElements = 100000000;

// Auswertung des Env-Werts als reine Funktion -- so ist sie testbar, ohne im
// Test Umgebungsvariablen setzen zu muessen.
// Hinweis: ein nicht parsbarer Wert ergibt ueber atoi() 0 und wird auf 1
// geklemmt, die Grenze ist dann faktisch "alles zu gross". Das entspricht der
// Konvention der uebrigen TEIR_*-Schalter (std::max(1, atoi(...))).
inline int parseMaxTensorElements(const char* env)
{
    return env ? std::max(1, std::atoi(env)) : kDefaultMaxTensorElements;
}

// Aktuelle Grenze aus der Umgebung. Einmalig ausgewertet (Prozesslaufzeit).
inline int maxTensorElements()
{
    static const int value = parseMaxTensorElements(std::getenv("TEIR_MAX_TENSOR"));
    return value;
}

// Reine Pruefung gegen eine explizite Grenze (fuer Tests).
inline bool tensorTooLarge(int in0Elems, int in1Elems, int outElems, int limit)
{
    return in0Elems > limit || in1Elems > limit || outElems > limit;
}

// Pruefung gegen die Grenze aus der Umgebung (Produktivpfad).
inline bool tensorTooLarge(int in0Elems, int in1Elems, int outElems)
{
    return tensorTooLarge(in0Elems, in1Elems, outElems, maxTensorElements());
}
