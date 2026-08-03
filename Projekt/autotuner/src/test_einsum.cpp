// Tests fuer die Einstein-Notation (einsum.cpp).
//
// Warum diese Funktionen kritisch sind:
//   parseEinsum        - Grundlage von allem; leitet Reduktionsachsen her
//   extentOfChar       - muss GESPLITTETE Achsen (a -> a0/a1) zusammenrechnen,
//                        sonst stimmen nach einem Split alle Groessen nicht mehr
//   computeStrides     - hier leben Layout-Bugs
//   isGEMMForm         - entscheidet, ob NEON/SME ueberhaupt feuern
//   isReduceAxis       - muss auch die gesplitteten Namen ("p0"/"p1") erkennen
//
// referenceEinsum und validateEinsumSample sind bereits durch
// test_kernel_validation.cpp bzw. test_c2_validator.cpp abgedeckt.
#include "einsum.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

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

static std::string chars(const std::vector<char>& v)
{
    return std::string(v.begin(), v.end());
}

static TEIR makeIr(const std::string& einsum, const std::vector<Axis>& axes)
{
    TEIR ir;
    ir.name = "t";
    ir.einsum = einsum;
    ir.axes = axes;
    return ir;
}

int main()
{
    std::printf("\n=== Einstein-Notation (einsum.cpp) ===\n");

    // ---------------------------------------------------------------
    std::printf("\n-- parseEinsum: Zerlegung --\n");
    // ---------------------------------------------------------------
    {
        // Reihenfolge der Notation ist out-in0-in1.
        EinsumSpec s = parseEinsum("ab-ac-cb");
        check(s.out_idx == "ab", "out_idx", s.out_idx);
        check(s.in0_idx == "ac", "in0_idx", s.in0_idx);
        check(s.in1_idx == "cb", "in1_idx", s.in1_idx);
        check(chars(s.out_axes) == "ab", "out_axes", chars(s.out_axes));
        // c kommt in beiden Eingaben, aber nicht in out -> Reduktionsachse.
        check(chars(s.reduce_axes) == "c", "reduce_axes = c", chars(s.reduce_axes));
        // all_axes = erst Output-, dann Reduktionsachsen.
        check(chars(s.all_axes) == "abc", "all_axes = out + reduce", chars(s.all_axes));
    }
    {
        // Mehrere Reduktionsachsen, hoehere Ordnung.
        EinsumSpec s = parseEinsum("abc-bda-dc");
        check(chars(s.out_axes) == "abc", "3D: out_axes", chars(s.out_axes));
        check(chars(s.reduce_axes) == "d", "3D: reduce_axes = d", chars(s.reduce_axes));
    }
    {
        // Achse in BEIDEN Eingaben und nicht in out -> nur EINMAL als Reduktion.
        EinsumSpec s = parseEinsum("ab-acd-dcb");
        check(chars(s.reduce_axes) == "cd" || chars(s.reduce_axes) == "dc",
              "Achse aus beiden Eingaben nur einmal in reduce_axes",
              chars(s.reduce_axes));
        check(s.reduce_axes.size() == 2, "genau 2 Reduktionsachsen",
              std::to_string(s.reduce_axes.size()));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- parseEinsum: Fehlerfaelle --\n");
    // ---------------------------------------------------------------
    {
        bool threw = false;
        try { parseEinsum("ab-ac"); } catch (const std::exception&) { threw = true; }
        check(threw, "zu wenige Teile werfen eine Ausnahme");
        threw = false;
        try { parseEinsum("ab-ac-cb-xy"); } catch (const std::exception&) { threw = true; }
        check(threw, "zu viele Teile werfen eine Ausnahme");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- extentOfChar: auch fuer gesplittete Achsen --\n");
    // ---------------------------------------------------------------
    {
        TEIR ir = makeIr("ab-ac-cb", {{"a", 32}, {"b", 16}, {"c", 8}});
        check(extentOfChar(ir, 'a') == 32, "direkte Achse");
        check(extentOfChar(ir, 'z') == 1, "unbekannte Achse -> 1 (neutral)");
    }
    {
        // Nach splitOuterAxis("c", 4) heisst c nicht mehr c, sondern c0/c1.
        // extentOfChar MUSS daraus wieder das Gesamt-Extent bilden.
        TEIR ir = makeIr("ab-ac-cb", {{"a", 32}, {"b", 16}, {"c0", 2}, {"c1", 4}});
        check(extentOfChar(ir, 'c') == 8,
              "gesplittete Achse c0*c1 wird zusammengerechnet",
              std::to_string(extentOfChar(ir, 'c')));
        // Nur eine Haelfte vorhanden -> kein gueltiger Split -> 1.
        TEIR half = makeIr("ab-ac-cb", {{"a", 32}, {"b", 16}, {"c0", 2}});
        check(extentOfChar(half, 'c') == 1, "halber Split zaehlt nicht als Split");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- computeStrides / tensorElements --\n");
    // ---------------------------------------------------------------
    {
        TEIR ir = makeIr("abc-bda-dc", {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 5}});
        // Row-major: letzte Achse hat Stride 1, davor Produkt der rechten Extents.
        auto st = computeStrides("abc", ir);
        check(st.size() == 3 && st[2] == 1 && st[1] == 4 && st[0] == 12,
              "row-major Strides fuer abc (2,3,4) = [12,4,1]",
              std::to_string(st[0]) + "," + std::to_string(st[1]) + "," + std::to_string(st[2]));
        auto st1 = computeStrides("bda", ir);
        check(st1[2] == 1 && st1[1] == 2 && st1[0] == 10,
              "Strides folgen der Index-Reihenfolge, nicht der Achsen-Definition",
              std::to_string(st1[0]) + "," + std::to_string(st1[1]) + "," + std::to_string(st1[2]));
        check(computeStrides("", ir).empty(), "leerer Index -> leere Strides");

        check(tensorElements("abc", ir) == 2 * 3 * 4, "tensorElements abc");
        check(tensorElements("bda", ir) == 3 * 5 * 2, "tensorElements bda");
        check(tensorElements("", ir) == 1, "leerer Index -> 1 Element (Skalar)");
    }

    // ---------------------------------------------------------------
    std::printf("\n-- einsumFlops --\n");
    // ---------------------------------------------------------------
    {
        TEIR ir = makeIr("ab-ac-cb", {{"a", 2}, {"b", 3}, {"c", 4}});
        // 2 Flops (mul+add) je Kombination aller Achsen.
        check(einsumFlops(ir) == 2.0 * 2 * 3 * 4, "FLOPs = 2 * Produkt aller Achsen",
              std::to_string(einsumFlops(ir)));
        TEIR split = makeIr("ab-ac-cb", {{"a", 2}, {"b", 3}, {"c0", 2}, {"c1", 2}});
        check(einsumFlops(split) == einsumFlops(ir),
              "Split aendert die FLOPs nicht", std::to_string(einsumFlops(split)));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- isGEMMForm: gated NEON/SME --\n");
    // ---------------------------------------------------------------
    {
        // C[a,b] = sum_c A[a,c] * B[c,b] -- die einzige Form, die die
        // spezialisierten Backends bedienen koennen.
        check(isGEMMForm(parseEinsum("ab-ac-cb")), "ab-ac-cb ist GEMM-Form");
        // Vertauschte Reduktionsposition -> kein GEMM.
        check(!isGEMMForm(parseEinsum("ab-ca-cb")), "ab-ca-cb ist KEINE GEMM-Form");
        check(!isGEMMForm(parseEinsum("ab-ac-bc")), "ab-ac-bc ist KEINE GEMM-Form");
        // Hoehere Ordnung -> kein GEMM (faellt auf den generischen Kernel zurueck).
        check(!isGEMMForm(parseEinsum("abc-bda-dc")), "abc-bda-dc ist KEINE GEMM-Form");
        // Zwei Reduktionsachsen -> kein GEMM.
        check(!isGEMMForm(parseEinsum("ab-acd-dcb")), "zwei Reduktionsachsen -> kein GEMM");
    }
    {
        GEMMMapping m = extractGEMMMapping(parseEinsum("ab-ac-cb"));
        check(m.out0_axis == 'a' && m.out1_axis == 'b' && m.reduce_axis == 'c',
              "extractGEMMMapping liefert M=a, N=b, K=c",
              std::string(1, m.out0_axis) + std::string(1, m.out1_axis) +
              std::string(1, m.reduce_axis));
    }

    // ---------------------------------------------------------------
    std::printf("\n-- isReduceAxis: auch fuer gesplittete Namen --\n");
    // ---------------------------------------------------------------
    {
        EinsumSpec s = parseEinsum("ab-ac-cb");   // reduce = c
        check(isReduceAxis("c", s), "c ist Reduktionsachse");
        check(!isReduceAxis("a", s), "a ist keine Reduktionsachse");
        // Nach einem Split heisst die Achse c0/c1 -- muss weiterhin erkannt werden,
        // sonst behandelt der Codegen die innere Schleife falsch.
        check(isReduceAxis("c0", s), "c0 (gesplittet) ist Reduktionsachse");
        check(isReduceAxis("c1", s), "c1 (gesplittet) ist Reduktionsachse");
        check(!isReduceAxis("a0", s), "a0 (gesplittet) ist keine Reduktionsachse");
        check(!isReduceAxis("c2", s), "c2 ist kein gueltiger Split-Name");
    }

    std::printf("\n=== %d Pruefungen, %d Fehler ===\n\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
