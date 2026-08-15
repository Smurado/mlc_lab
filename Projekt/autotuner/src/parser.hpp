// Eingangspunkt der Pipeline: liest die Kontraktions-Beschreibung aus der
// CSV-Zeile und baut daraus die TEIR (Details und Fehlerfaelle: test_parser).
#pragma once

#include "teir.hpp"

#include <string>


TEIR parseCSV(const std::string& filename);