#include "parser.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<std::string> split(const std::string& str, char delim)
{
    std::vector<std::string> result;

    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, delim))
    {
        if (!item.empty())
            result.push_back(item);
    }

    return result;
}

std::string removeQuotes(std::string s)
{
    if (!s.empty() && s.front() == '"')
        s.erase(s.begin());

    if (!s.empty() && s.back() == '"')
        s.pop_back();

    return s;
}

std::vector<std::string> parseCSVLine(const std::string& line)
{
    std::vector<std::string> cols;

    bool inQuotes = false;
    std::string current;

    for (char c : line)
    {
        if (c == '"')
        {
            inQuotes = !inQuotes;
        }
        else if (c == ',' && !inQuotes)
        {
            cols.push_back(current);
            current.clear();
        }
        else
        {
            current += c;
        }
    }

    cols.push_back(current);

    return cols;
}

} // namespace

std::vector<TEIR> parseCSV(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Cannot open CSV file: " + filename);

    std::vector<TEIR> kernels;

    std::string line;

    // Header überspringen
    std::getline(file, line);

    // Alle Datenzeilen lesen
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto cols = parseCSVLine(line);

        if (cols.size() != 6)
            throw std::runtime_error(
                "Expected exactly 6 CSV columns.");

        TEIR ir;

        //--------------------------------------------------
        // Name
        //--------------------------------------------------

        ir.name = cols[0];

        //--------------------------------------------------
        // Tensoren
        //--------------------------------------------------

        for (const auto& t : split(removeQuotes(cols[1]), ';'))
        {
            auto parts = split(t, ':');

            if (parts.size() != 2)
                throw std::runtime_error(
                    "Invalid tensor: " + t);

            ir.tensors.push_back(
            {
                parts[0],
                parts[1]
            });
        }

        //--------------------------------------------------
        // Achsen
        //--------------------------------------------------

        for (const auto& a : split(removeQuotes(cols[2]), ';'))
        {
            auto parts = split(a, ':');

            if (parts.size() != 2)
                throw std::runtime_error(
                    "Invalid axis: " + a);

            ir.axes.push_back(
            {
                parts[0],
                std::stoi(parts[1])
            });
        }

        //--------------------------------------------------
        // Primitives
        //--------------------------------------------------

        for (const auto& p : split(removeQuotes(cols[3]), ';'))
        {
            ir.primitives.push_back(p);
        }

        //--------------------------------------------------
        // Schedule
        //--------------------------------------------------

        for (const auto& s : split(removeQuotes(cols[4]), ';'))
        {
            auto parts = split(s, ':');

            if (parts.size() != 2)
                throw std::runtime_error(
                    "Invalid schedule entry: " + s);

            Policy pol =
                (parts[1] == "parallel")
                    ? Policy::Parallel
                    : Policy::Sequential;

            ir.schedule.push_back(
            {
                parts[0],
                pol
            });
        }

        //--------------------------------------------------
        // Invokes
        //--------------------------------------------------

        for (const auto& inv : split(removeQuotes(cols[5]), ';'))
        {
            ir.invokes.push_back(inv);
        }

        kernels.push_back(std::move(ir));
    }

    return kernels;
}