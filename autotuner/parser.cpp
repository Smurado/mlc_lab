#include "parser.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

// Hilfsfunktion: Entfernt Leerzeichen/Tabs am Anfang und Ende einer Zeile
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Hilfsfunktion: Entfernt IR-spezifische Syntax-Präfixe/Suffixe für saubere Strings
std::string cleanName(const std::string& str) {
    std::string res = str;
    res.erase(std::remove_if(res.begin(), res.end(), [](char c) {
        return c == '@' || c == '%' || c == ':' || c == ',' || c == '{' || c == '}';
    }), res.end());
    return res;
}

TEIR parseTEIR(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Datei konnte nicht geoeffnet werden: " + filename);
    }

    TEIR ir;
    std::string line;
    bool inSchedule = false;

    while (std::getline(infile, line)) {
        line = trim(line);
        
        // Ignoriere leere Zeilen oder Kommentare
        if (line.empty() || line.starts_with("//")) continue;

        std::stringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "teir") {
            std::string name;
            ss >> name;
            ir.name = cleanName(name);
        } 
        else if (token == "tensor") {
            std::string name, colon, type;
            ss >> name >> colon >> type;
            ir.tensors.push_back({cleanName(name), type});
        } 
        else if (token == "axis") {
            std::string name, extentKw;
            int extent = 0;
            ss >> name >> extentKw >> extent;
            ir.axes.push_back({cleanName(name), extent});
        } 
        else if (token == "primitive") {
            std::string name;
            ss >> name;
            ir.primitives.push_back(cleanName(name));
        } 
        else if (token == "schedule") {
            inSchedule = true;
        } 
        else if (token == "iter" && inSchedule) {
            std::string name, policyKw, policyVal;
            ss >> name >> policyKw >> policyVal;
            Policy p = (policyVal == "parallel") ? Policy::Parallel : Policy::Sequential;
            ir.schedule.push_back({cleanName(name), p});
        } 
        else if (token == "invoke" && inSchedule) {
            std::string name;
            ss >> name;
            ir.invokes.push_back(cleanName(name));
        } 
        else if (token == "}") {
            if (inSchedule) {
                inSchedule = false; // Schedule-Block geschlossen
            }
        }
    }
    return ir;
}