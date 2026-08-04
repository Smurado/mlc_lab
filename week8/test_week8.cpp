// Tests fuer Woche 8: Optimizer-Transformationen und Evaluator.
//
// WARUM ES DIESE DATEI GIBT
// main.cpp prueft mit `verify_tensors(baseline, optimiert)` nur, ob beide Seiten
// UEBEREINSTIMMEN. Seit beide durch dieselben SME-Kernel laufen, ist das ein
// Selbstvergleich: ein falscher Kernel macht beide Seiten gleich falsch und die
// Pruefung meldet trotzdem PASSED. Genau dieser Fehlertyp ist im Projekt schon
// zweimal aufgetreten (week5/verify_gemm mit konstanter Fuellung, week6/trans_b
// nie getestet und deshalb nicht implementiert).
//
// Die Referenz hier kennt WEDER den Schedule NOCH die Kernel: sie laeuft ueber
// das kartesische Produkt aller Achsen und wendet die Primitive-Semantik direkt
// auf die deklarierten Strides an. Ein Fehler im Evaluator kann sich darin nicht
// spiegeln.

#include "include/teir_parser.h"
#include "include/teir_evaluator.h"
#include "include/teir_optimizer.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "  [FAIL] " << what << "\n";
    }
}

void section(const std::string& title) {
    std::cout << "\n== " << title << " ==\n";
}

// ---------------------------------------------------------------------------
// Hilfsmittel
// ---------------------------------------------------------------------------

size_t tensor_bytes(const TEIRProgram& prog, const std::string& t) {
    size_t max_offset = 0;
    for (auto const& entry : prog.axes) {
        auto it = entry.second.strides.find(t);
        if (it != entry.second.strides.end())
            max_offset += static_cast<size_t>(entry.second.extent - 1) * it->second;
    }
    return max_offset + 4;
}

long stride_of(const TEIRProgram& prog, const std::string& axis, const std::string& tensor) {
    auto a = prog.axes.find(axis);
    if (a == prog.axes.end()) return 0;
    auto s = a->second.strides.find(tensor);
    return s == a->second.strides.end() ? 0 : s->second;
}

// Unabhaengige Referenz: kartesisches Produkt ueber ALLE Achsen, Offsets direkt
// aus den Strides. Kein Schedule, kein Kernel, keine Kachelung.
void reference_evaluate(const TEIRProgram& prog,
                        const std::vector<std::string>& tensor_names,
                        const std::vector<std::vector<float>>& in_data,
                        std::vector<float>& out) {
    std::vector<std::string> axis_names;
    std::vector<int> extents;
    for (auto const& entry : prog.axes) {
        axis_names.push_back(entry.first);
        extents.push_back(entry.second.extent);
    }

    std::fill(out.begin(), out.end(), 0.0f);

    // Welches Primitive rechnet? Zero wird von der Nullung oben erledigt.
    std::string kind;
    for (auto const& entry : prog.primitives)
        if (entry.second.kind != "Zero") kind = entry.second.kind;

    const size_t ndim = axis_names.size();
    std::vector<int> idx(ndim, 0);
    long total = 1;
    for (int e : extents) total *= e;

    auto data_of = [&](const std::string& name) -> const std::vector<float>* {
        for (size_t i = 0; i < tensor_names.size(); ++i)
            if (tensor_names[i] == name) return &in_data[i];
        return nullptr;
    };

    for (long lin = 0; lin < total; ++lin) {
        long rem = lin;
        for (size_t d = 0; d < ndim; ++d) {
            idx[d] = static_cast<int>(rem % extents[d]);
            rem /= extents[d];
        }

        auto offset = [&](const std::string& tensor) -> long {
            long off = 0;
            for (size_t d = 0; d < ndim; ++d)
                off += static_cast<long>(idx[d]) * stride_of(prog, axis_names[d], tensor);
            return off / 4;   // Bytes -> Elemente
        };

        if (kind == "Contraction") {
            const auto* a = data_of("in0");
            const auto* b = data_of("in1");
            out[offset("out")] += (*a)[offset("in0")] * (*b)[offset("in1")];
        } else if (kind == "Copy") {
            const auto* in = data_of("in");
            out[offset("out")] = (*in)[offset("in")];
        }
    }
}

// Programm laden, mit festen Zufallswerten fuellen, auswerten und gegen die
// Referenz vergleichen. `optimize` darf die Schedule vorher umbauen.
void run_against_reference(const std::string& file, const std::string& label,
                           void (*optimize)(TEIROptimizer&)) {
    TEIRProgram prog = load_teir(file);

    std::mt19937 rng(20260804u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::string> names = prog.tensors;
    std::vector<std::vector<float>> data;
    std::vector<void*> args;
    size_t out_elems = 0;

    for (auto const& t : names) {
        const size_t elems = tensor_bytes(prog, t) / 4;
        data.emplace_back(elems);
        if (t == "out") {
            out_elems = elems;
            // Genullt, weil die .teir-Modelle das voraussetzen: der @zero-Aufruf
            // haengt dort UNTER der k-Schleife, aber AUSSERHALB der m/n-Schleifen
            // und deckt deshalb nur die Kachel am Basisversatz ab. main.cpp
            // gleicht das mit einem memset aus -- dieselbe Zusage gilt hier.
            // (Mit einer Wachfuellung statt Null schlagen 768 von 1024 Werten
            // fehl; das ist eine Eigenschaft der vorgegebenen Modelle, kein
            // Fehler des Evaluators.)
            std::fill(data.back().begin(), data.back().end(), 0.0f);
        } else {
            for (auto& v : data.back()) v = dist(rng);
        }
    }
    for (auto& d : data) args.push_back(d.data());

    std::vector<float> ref(out_elems);
    reference_evaluate(prog, names, data, ref);

    if (optimize) {
        TEIROptimizer opt(prog);
        optimize(opt);
    }
    TEIREvaluator eval(prog);
    eval.evaluate(args.data());

    // out steht in data an derselben Position wie in names.
    size_t out_idx = 0;
    for (size_t i = 0; i < names.size(); ++i) if (names[i] == "out") out_idx = i;
    const std::vector<float>& got = data[out_idx];

    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < out_elems; ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - ref[i]);
        const double tol  = 1e-3 * std::max(1.0, std::fabs(static_cast<double>(ref[i])));
        if (diff > tol) { ++bad; worst = std::max(worst, diff); }
    }
    check(bad == 0, label + ": " + std::to_string(bad) + " von " +
                    std::to_string(out_elems) + " Werten falsch (groesste Abweichung "
                    + std::to_string(worst) + ")");
}

std::shared_ptr<IterNode> find_iter(const std::vector<std::shared_ptr<ScheduleNode>>& nodes,
                                    const std::string& axis) {
    for (auto& n : nodes) {
        if (n->type != NodeType::Iter) continue;
        auto it = std::static_pointer_cast<IterNode>(n);
        const std::string ax = (it->axis[0] == '@') ? it->axis.substr(1) : it->axis;
        if (ax == axis) return it;
        if (auto found = find_iter(it->children, axis)) return found;
    }
    return nullptr;
}

} // namespace

// ---------------------------------------------------------------------------

int main() {
    std::cout << "======================================\n"
                 "Woche 8 -- Tests\n"
                 "======================================\n";

    // -----------------------------------------------------------------------
    section("1. Evaluator gegen unabhaengige Referenz");
    // Das ist der Test, der die SME-Kernel absichert. main.cpp kann das nicht:
    // dort laufen beide Vergleichsseiten durch denselben Kernel.

    run_against_reference("data/test_matmul.teir",
                          "GEMM (mini_jit::Gemm) gegen Referenz", nullptr);
    run_against_reference("data/test_transposition.teir",
                          "Transposition (mini_jit::Unary) gegen Referenz", nullptr);

    // Dieselbe Pruefung, nachdem der Optimizer die Schedule umgebaut hat. Eine
    // Transformation, die die Semantik verletzt, faellt nur hier auf.
    run_against_reference("data/test_matmul.teir",
                          "GEMM nach expose_parallelism gegen Referenz",
                          [](TEIROptimizer& o) { o.expose_parallelism(); });
    run_against_reference("data/test_transposition.teir",
                          "Transposition nach expose_parallelism gegen Referenz",
                          [](TEIROptimizer& o) { o.expose_parallelism(); });
    run_against_reference("data/test_matmul.teir",
                          "GEMM nach apply_cache_blocking gegen Referenz",
                          [](TEIROptimizer& o) { o.apply_cache_blocking(); });
    run_against_reference("data/test_matmul.teir",
                          "GEMM nach beiden Passes gegen Referenz",
                          [](TEIROptimizer& o) { o.expose_parallelism(); o.apply_cache_blocking(); });

    // -----------------------------------------------------------------------
    section("2. Kernel-Auswahl");
    // Ohne diese Pruefung wuerde ein stiller Rueckfall auf die Skalarschleife
    // unbemerkt bleiben -- die Ergebnisse waeren ja weiterhin richtig, nur
    // hundertfach langsamer. Genau das war der Zustand vor dem SME-Einbau.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        std::vector<std::vector<float>> data;
        std::vector<void*> args;
        for (auto const& t : prog.tensors) {
            data.emplace_back(tensor_bytes(prog, t) / 4, 0.0f);
        }
        for (auto& d : data) args.push_back(d.data());

        TEIREvaluator eval(prog);
        eval.evaluate(args.data());

        bool gemm_jit = false, zero_jit = false;
        for (auto const& line : eval.plan()) {
            if (line.find("mini_jit::Gemm") != std::string::npos)  gemm_jit = true;
            if (line.find("mini_jit::Unary") != std::string::npos) zero_jit = true;
        }
        check(gemm_jit, "Contraction waehlt mini_jit::Gemm (kein stiller Rueckfall)");
        check(zero_jit, "Zero waehlt mini_jit::Unary");
    }
    {
        TEIRProgram prog = load_teir("data/test_transposition.teir");
        std::vector<std::vector<float>> data;
        std::vector<void*> args;
        for (auto const& t : prog.tensors) data.emplace_back(tensor_bytes(prog, t) / 4, 0.0f);
        for (auto& d : data) args.push_back(d.data());

        TEIREvaluator eval(prog);
        eval.evaluate(args.data());

        bool transposing = false;
        for (auto const& line : eval.plan())
            if (line.find("transponierend") != std::string::npos) transposing = true;
        check(transposing, "Copy waehlt den transponierenden Unary-Kernel");
    }

    // Gegenprobe: ein Kernel, der nur einen Bruchteil der Kachel abdecken
    // wuerde, muss ABGELEHNT werden. Bei einsum.teir deklarieren beide
    // Primitive-Achsen (@y und @x) Stride 4 auf `out`; daraus folgt ld = 1 und
    // der Zero-Kernel haette 2686 statt 1769472 Elementen genullt. Sichtbar
    // geworden waere das nie, weil main.cpp `out` vorher per memset nullt.
    {
        TEIRProgram prog = load_teir("data/einsum.teir");
        std::vector<std::vector<float>> data;
        std::vector<void*> args;
        for (auto const& t : prog.tensors) data.emplace_back(tensor_bytes(prog, t) / 4, 0.0f);
        for (auto& d : data) args.push_back(d.data());

        // Nur den Plan erzeugen, nicht rechnen -- einsum laeuft ueber zwei Minuten.
        TEIREvaluator eval(prog);
        eval.plan_only();

        bool zero_rejected = false;
        for (auto const& line : eval.plan())
            if (line.find("(Zero)") != std::string::npos &&
                line.find("Rueckfall") != std::string::npos) zero_rejected = true;
        check(zero_rejected,
              "Zero mit ld < zusammenhaengender Kante faellt zurueck statt still zu wenig zu nullen");
    }

    // -----------------------------------------------------------------------
    section("3. Transformationen");

    // split: Extent geht auf, Strides passen zur neuen Schachtelung.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        const int orig_extent = prog.axes["m0"].extent;
        const int orig_stride = prog.axes["m0"].strides["out"];

        TEIROptimizer opt(prog);
        opt.split_iteration("@m0", "@m0i", 2);

        check(prog.axes["m0"].extent == orig_extent / 2, "split: aeusserer Extent halbiert");
        check(prog.axes["m0i"].extent == 2,              "split: innerer Extent = Faktor");
        check(prog.axes["m0"].strides["out"] == orig_stride * 2,
              "split: aeusserer Stride mit Faktor multipliziert");
        check(prog.axes["m0i"].strides["out"] == orig_stride,
              "split: innerer Stride unveraendert");
        check(find_iter(prog.roots, "m0i") != nullptr, "split: innerer Knoten eingehaengt");
    }

    // split gefolgt von fuse muss den Ausgangszustand wiederherstellen.
    // Das prueft beide Richtungen gegeneinander, ohne die Erwartung von Hand
    // hinschreiben zu muessen.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        const int orig_extent = prog.axes["m0"].extent;
        const int orig_stride = prog.axes["m0"].strides["out"];

        TEIROptimizer opt(prog);
        opt.split_iteration("@m0", "@m0i", 2);
        opt.fuse_iteration_nodes("@m0", "@m0i", "@m0f");

        check(prog.axes.count("m0f") == 1, "fuse: neue Achse angelegt");
        if (prog.axes.count("m0f")) {
            check(prog.axes["m0f"].extent == orig_extent,
                  "fuse nach split: Extent wiederhergestellt");
            check(prog.axes["m0f"].strides["out"] == orig_stride,
                  "fuse nach split: Stride wiederhergestellt");
        }
    }

    // fuse muss ABLEHNEN, wenn die Achsen verschiedene Tensoren beruehren.
    // Genau das ist bei einsum der Fall (a adressiert in0, b adressiert in1) --
    // wuerde hier faelschlich fusioniert, waeren die Ergebnisse still falsch.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        TEIROptimizer opt(prog);
        opt.fuse_iteration_nodes("@m0", "@n0", "@mn");
        check(prog.axes.count("mn") == 0,
              "fuse lehnt ab: m0 und n0 beruehren verschiedene Tensoren");
    }

    // fuse muss ABLEHNEN, wenn die Achsen nicht direkt geschachtelt sind.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        TEIROptimizer opt(prog);
        opt.fuse_iteration_nodes("@k0", "@n0", "@kn");
        check(prog.axes.count("kn") == 0,
              "fuse lehnt ab: n0 ist kein direktes Kind von k0");
    }

    // set_policy / expose_parallelism
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        TEIROptimizer opt(prog);
        opt.set_policy("@m0", "parallel");
        auto n = find_iter(prog.roots, "m0");
        check(n && n->policy == "parallel", "set_policy setzt die Policy");
    }

    // expose_parallelism darf die Reduktionsachse NIE parallel schalten --
    // mehrere Threads wuerden dieselbe Ausgabezelle akkumulieren.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        TEIROptimizer opt(prog);
        opt.expose_parallelism();
        auto k = find_iter(prog.roots, "k0");
        check(k && k->policy != "parallel",
              "expose_parallelism laesst die Reduktionsachse k0 sequenziell");
    }

    // Das Produkt-Kriterium: einsum hat aeussere Achsen 4, 4 und 3 -- keine
    // erreicht die Kernzahl, zusammen aber 48. Mit der alten Einzelschwelle
    // blieb das Programm komplett sequenziell (gemessen 1,0014x).
    {
        TEIRProgram prog = load_teir("data/einsum.teir");
        TEIROptimizer opt(prog);
        opt.expose_parallelism();

        long exposed = 1;
        for (const char* ax : {"a", "b", "c"}) {
            auto n = find_iter(prog.roots, ax);
            if (n && n->policy == "parallel") exposed *= prog.axes[ax].extent;
        }
        check(exposed >= 12,
              "expose_parallelism erreicht bei einsum >= 12 Iterationen (erreicht: "
              + std::to_string(exposed) + ")");

        auto s = find_iter(prog.roots, "s");
        check(s && s->policy != "parallel",
              "expose_parallelism laesst einsums Reduktionsachse s sequenziell");
    }

    // expose_parallelism muss eine Achse ablehnen, deren Iterationen einander
    // ueberschreiben. transposition.teir ist so ein Fall: die Indexabbildung
    // auf `out` ist nicht injektiv (18874368 gelesene Werte auf 18088128
    // Zellen), weil `a` mit `c` zusammen 0..4607 ueberdeckt, `b` aber nur um
    // 4416 springt. Parallel lieferte das in einem von drei Laeufen falsche
    // Ergebnisse.
    {
        TEIRProgram prog = load_teir("data/transposition.teir");
        TEIROptimizer opt(prog);
        opt.expose_parallelism();
        auto a = find_iter(prog.roots, "a");
        check(a && a->policy != "parallel",
              "expose_parallelism lehnt transposition/a ab (Schreibbereiche ueberlappen)");
    }

    // Gegenprobe: das kleine Testmodell IST injektiv und muss parallel laufen,
    // sonst wuerde die neue Pruefung einfach alles ablehnen.
    {
        TEIRProgram prog = load_teir("data/test_transposition.teir");
        TEIROptimizer opt(prog);
        opt.expose_parallelism();
        auto a = find_iter(prog.roots, "a");
        check(a && a->policy == "parallel",
              "expose_parallelism erlaubt test_transposition/a (disjunkte Bereiche)");
    }

    // promote_to_primitive: Achse wandert in die axes_map und die Schleife
    // verschwindet aus dem Nest.
    {
        TEIRProgram prog = load_teir("data/test_matmul.teir");
        TEIROptimizer opt(prog);
        opt.promote_to_primitive("@n0", "@gemm");

        auto& slot = prog.primitives["gemm"].axes_map["N"];
        check(std::find(slot.begin(), slot.end(), "@n0") != slot.end(),
              "promote: Achse in axes_map aufgenommen");
        check(find_iter(prog.roots, "n0") == nullptr,
              "promote: Schleife aus dem Nest entfernt");
    }

    // -----------------------------------------------------------------------
    std::cout << "\n======================================\n";
    std::cout << (failures == 0 ? "ALLE TESTS BESTANDEN" : "TESTS FEHLGESCHLAGEN")
              << " -- " << (checks - failures) << " von " << checks << " Pruefungen ok\n";
    std::cout << "======================================\n";
    return failures == 0 ? 0 : 1;
}
