#pragma once
#include "teir.hpp"
#include <string>
#include <vector>

// Liest eine .teir Datei ein und gibt die strukturierte IR zurück
TEIR parseTEIR(const std::string& filename);

// Liest eine Liste von Einsum-Kernelbeschreibungen und gibt alle TEIR-Kerne zurück
std::vector<TEIR> parseKernelCatalog(const std::string& filename);
