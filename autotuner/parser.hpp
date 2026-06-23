#pragma once
#include "teir.hpp"
#include <string>

// Liest eine .teir Datei ein und gibt die strukturierte IR zurück
TEIR parseTEIR(const std::string& filename);