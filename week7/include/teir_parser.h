// TEIR-Parser: liest .teir-Dateien und baut ein TEIRProgram-AST.
#pragma once

#include "teir.h"

#include <string>

// Komfort-Funktion: Datei einlesen und parsen.
TEIRProgram load_teir(const std::string& path);
