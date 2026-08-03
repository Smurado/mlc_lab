// Tests fuer den CSV-Parser (parser.cpp).
//
// parseCSV() ist der Eingangspunkt des gesamten Systems: alles, was danach
// passiert (Codegen, Suche, Messung), haengt an der hier erzeugten IR. Die Tests
// schreiben kleine CSVs in ein temporaeres Verzeichnis und rufen parseCSV()
// unveraendert auf -- also genau den Pfad, der auch produktiv laeuft.
#include "parser.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

static const char* HEADER =
    "name,tensors,axes,primitives,schedule,invokes,einsum\n";

// Schreibt eine CSV und liefert den Pfad.
static std::string writeCsv(const std::string& stem, const std::string& content)
{
    const std::filesystem::path p =
        std::filesystem::temp_directory_path() / ("_teir_test_" + stem + ".csv");
    std::ofstream f(p);
    f << content;
    f.close();
    return p.string();
}

// Liefert true, wenn parseCSV bei dieser Datei wirft.
static bool throwsOn(const std::string& stem, const std::string& content)
{
    const std::string path = writeCsv(stem, content);
    bool threw = false;
    try {
        parseCSV(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::filesystem::remove(path);
    return threw;
}

int main()
{
    std::printf("\n=== CSV-Parser (parser.cpp) ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- Vollstaendige Zeile mit Einsum (7 Spalten) --\n");
    // ---------------------------------------------------------------
    {
        const std::string path = writeCsv("full",
            std::string(HEADER) +
            "mycase,in0:f32;in1:f32;out:f32,a:32;b:16;c:8,zero;gemm,"
            "a:parallel;b:sequential;c:sequential,zero;gemm,ab-ac-cb\n");
        TEIR ir = parseCSV(path);

        check(ir.name == "mycase", "Name", ir.name);
        check(ir.einsum == "ab-ac-cb", "Einsum (7. Spalte)", ir.einsum);

        check(ir.tensors.size() == 3, "3 Tensoren",
              std::to_string(ir.tensors.size()));
        check(ir.tensors[0].name == "in0" && ir.tensors[0].type == "f32",
              "Tensor wird in Name und Typ zerlegt");

        check(ir.axes.size() == 3, "3 Achsen", std::to_string(ir.axes.size()));
        check(ir.axes[0].name == "a" && ir.axes[0].extent == 32,
              "Achsen-Extent wird als Zahl geparst");
        check(ir.axes[2].name == "c" && ir.axes[2].extent == 8,
              "Achsen-Reihenfolge bleibt erhalten");

        check(ir.primitives.size() == 2 && ir.primitives[0] == "zero" &&
              ir.primitives[1] == "gemm", "Primitives");
        check(ir.invokes.size() == 2, "Invokes",
              std::to_string(ir.invokes.size()));

        check(ir.schedule.size() == 3, "3 Schedule-Eintraege",
              std::to_string(ir.schedule.size()));
        // "parallel" ist das einzige Schluesselwort; alles andere = sequentiell.
        check(ir.schedule[0].policy == Policy::Parallel,
              "a:parallel wird zu Policy::Parallel");
        check(ir.schedule[1].policy == Policy::Sequential,
              "b:sequential wird zu Policy::Sequential");

        // Defaults, die nicht aus der CSV kommen.
        check(ir.unrollFactor == 1, "Default Unroll-Faktor ist 1");
        check(ir.backend == Backend::Scalar, "Default-Backend ist Scalar");

        std::filesystem::remove(path);
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Ohne Einsum (6 Spalten, alter GEMM-Pfad) --\n");
    // ---------------------------------------------------------------
    {
        const std::string path = writeCsv("six",
            "name,tensors,axes,primitives,schedule,invokes\n"
            "gemmcase,in0:f32;in1:f32;out:f32,p:128;r:96;t:64,zero;gemm,"
            "p:parallel;r:sequential,zero;gemm\n");
        TEIR ir = parseCSV(path);
        check(ir.name == "gemmcase", "6-Spalten-Zeile wird akzeptiert");
        check(ir.einsum.empty(), "ohne 7. Spalte bleibt einsum leer",
              "'" + ir.einsum + "'");
        check(ir.axes.size() == 3, "Achsen trotzdem geparst");
        std::filesystem::remove(path);
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Anfuehrungszeichen --\n");
    // ---------------------------------------------------------------
    {
        // data/input.csv verwendet Quotes um die Semikolon-Felder.
        const std::string path = writeCsv("quoted",
            std::string(HEADER) +
            "q,\"in0:f32;in1:f32;out:f32\",\"a:4;b:5\",\"zero;gemm\","
            "\"a:sequential;b:sequential\",\"zero;gemm\",ab-ab-ab\n");
        TEIR ir = parseCSV(path);
        check(ir.tensors.size() == 3, "Quotes um Semikolon-Felder werden entfernt",
              std::to_string(ir.tensors.size()));
        check(ir.axes.size() == 2 && ir.axes[0].name == "a",
              "Achsen aus gequotetem Feld korrekt");
        check(ir.tensors[0].name == "in0",
              "kein uebrig gebliebenes Anfuehrungszeichen im Namen",
              ir.tensors[0].name);
        std::filesystem::remove(path);
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Dokumentierte Grenze: nur die ERSTE Datenzeile --\n");
    // ---------------------------------------------------------------
    {
        // Bekannte Einschraenkung (Roadmap: "Mehrere Kernel pro Aufruf").
        // Hier festgehalten, damit eine spaetere Aenderung bewusst erfolgt.
        const std::string path = writeCsv("multi",
            std::string(HEADER) +
            "first,in0:f32;in1:f32;out:f32,a:4;b:4;c:4,zero;gemm,"
            "a:sequential,zero;gemm,ab-ac-cb\n"
            "second,in0:f32;in1:f32;out:f32,a:8;b:8;c:8,zero;gemm,"
            "a:sequential,zero;gemm,ab-ac-cb\n");
        TEIR ir = parseCSV(path);
        check(ir.name == "first", "zweite Zeile wird ignoriert (bekannte Grenze)",
              ir.name);
        check(ir.axes[0].extent == 4, "Werte stammen aus der ersten Zeile");
        std::filesystem::remove(path);
    }

    // ---------------------------------------------------------------
    std::printf("\n-- Fehlerfaelle --\n");
    // ---------------------------------------------------------------
    {
        bool threw = false;
        try {
            parseCSV("/nonexistent/pfad/gibt/es/nicht.csv");
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "nicht existierende Datei wirft");
    }
    check(throwsOn("empty", HEADER),
          "Datei nur mit Header wirft (keine Datenzeile)");
    check(throwsOn("fewcols", std::string(HEADER) + "a,b,c\n"),
          "zu wenige Spalten werfen");
    check(throwsOn("badtensor",
          std::string(HEADER) +
          "x,in0,a:4;b:4;c:4,zero,a:sequential,zero,ab-ac-cb\n"),
          "Tensor ohne Typ wirft");
    check(throwsOn("badaxis",
          std::string(HEADER) +
          "x,in0:f32;in1:f32;out:f32,a,zero,a:sequential,zero,ab-ac-cb\n"),
          "Achse ohne Extent wirft");
    check(throwsOn("badsched",
          std::string(HEADER) +
          "x,in0:f32;in1:f32;out:f32,a:4;b:4;c:4,zero,a,zero,ab-ac-cb\n"),
          "Schedule-Eintrag ohne Policy wirft");

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
