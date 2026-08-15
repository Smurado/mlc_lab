// Uebersetzt die transformierte TEIR in kompilierbaren C++-Quelltext.
// Backend-Wahl (Scalar/NEON/SME) steckt in der IR; das Ergebnis wird von
// main.cpp bzw. benchmark.cpp mit dem Systemcompiler zur .so gebaut.
#pragma once
#include "teir.hpp"
#include <string>

// Generiert C++ Quellcode aus der optimierten TEIR-Struktur
std::string generateSourceCode(const TEIR& ir);

// Schreibt den generierten Code in eine Datei auf die Festplatte
void writeCodeToFile(const std::string& filename, const std::string& code);