#include "include/teir_optimizer.h"
#include <iostream>
#include <algorithm>
#include <functional>

// Achsenname ohne fuehrendes '@'. Weiter unten nochmal als Datei-lokale
// Funktion vorhanden; hier vorgezogen, weil find_iter_node sie bereits braucht.
static std::string strip_at(const std::string& s);

std::shared_ptr<IterNode> TEIROptimizer::find_iter_node(const std::vector<std::shared_ptr<ScheduleNode>>& nodes, const std::string& axis) {
    // Beide Schreibweisen zulassen. Der Parser legt die Achse OHNE '@' ab,
    // die Transformationen werden aber mit "@m0" aufgerufen -- ohne das
    // Abschneiden fand split_iteration/set_policy den Knoten nie und tat
    // stillschweigend nichts (von test_week8.cpp aufgedeckt).
    const std::string want = strip_at(axis);
    for (auto& n : nodes) {
        if (n->type == NodeType::Iter) {
            auto inode = std::static_pointer_cast<IterNode>(n);
            if (strip_at(inode->axis) == want) return inode;
            auto child_res = find_iter_node(inode->children, axis);
            if (child_res) return child_res;
        }
    }
    return nullptr;
}

void TEIROptimizer::split_iteration(const std::string& parent_axis, const std::string& new_inner_axis, int split_factor) {
    auto node = find_iter_node(parent_axis);
    if (!node) return;
    
    std::string raw_parent = (parent_axis[0] == '@') ? parent_axis.substr(1) : parent_axis;
    std::string raw_inner = (new_inner_axis[0] == '@') ? new_inner_axis.substr(1) : new_inner_axis;
    
    int original_extent = prog.axes[raw_parent].extent;
    prog.axes[raw_parent].extent = original_extent / split_factor;
    
    Axis inner_ax = prog.axes[raw_parent]; 
    inner_ax.name = raw_inner;
    inner_ax.extent = split_factor;
    for (auto& [tname, val] : prog.axes[raw_parent].strides) {
        val *= split_factor;
    }
    prog.axes[raw_inner] = inner_ax;
    
    auto inner_node = std::make_shared<IterNode>();
    inner_node->axis = "@" + raw_inner;
    inner_node->name = "iter_" + raw_inner;
    inner_node->policy = "sequential";
    
    inner_node->children = std::move(node->children);
    node->children.clear();
    node->children.push_back(inner_node);
}

// ---------------------------------------------------------------------------
// Hilfsfunktionen
// ---------------------------------------------------------------------------

// "@m0" -> "m0"
static std::string strip_at(const std::string& s) {
    return (!s.empty() && s[0] == '@') ? s.substr(1) : s;
}

// Liefert den Elternknoten von `target` oder nullptr, wenn `target` eine Wurzel ist.
std::shared_ptr<IterNode> TEIROptimizer::find_parent_of(
        const std::vector<std::shared_ptr<ScheduleNode>>& nodes,
        const std::shared_ptr<ScheduleNode>& target) {
    for (auto& n : nodes) {
        if (n->type != NodeType::Iter) continue;
        auto it = std::static_pointer_cast<IterNode>(n);
        for (auto& c : it->children)
            if (c == target) return it;
        if (auto found = find_parent_of(it->children, target)) return found;
    }
    return nullptr;
}

// Ersetzt `target` in der Kinderliste seines Elternknotens (bzw. in den Wurzeln)
// durch `replacement`. Ist `replacement` leer, wird `target` entfernt.
void TEIROptimizer::replace_node(const std::shared_ptr<ScheduleNode>& target,
                                 const std::vector<std::shared_ptr<ScheduleNode>>& replacement) {
    auto parent = find_parent_of(prog.roots, target);
    auto& list = parent ? parent->children : prog.roots;
    for (size_t i = 0; i < list.size(); ++i) {
        if (list[i] == target) {
            list.erase(list.begin() + i);
            list.insert(list.begin() + i, replacement.begin(), replacement.end());
            return;
        }
    }
}

// Rolle einer Achse in einem Primitive aus den Strides ableiten.
//   in out UND in0, nicht in in1  -> M
//   in out UND in1, nicht in in0  -> N
//   in in0 UND in1, nicht in out  -> K
// Liefert "" wenn nicht eindeutig. Das ersetzt fest eingetragene Achsennamen.
std::string TEIROptimizer::infer_role(const std::string& axis) const {
    auto it = prog.axes.find(strip_at(axis));
    if (it == prog.axes.end()) return "";
    auto has = [&](const char* t) {
        auto s = it->second.strides.find(t);
        return s != it->second.strides.end() && s->second != 0;
    };
    const bool o = has("out"), a = has("in0") || has("in"), b = has("in1");
    if (o && a && !b) return "M";
    if (o && b && !a) return "N";
    if (a && b && !o) return "K";
    return "";
}

// ---------------------------------------------------------------------------
// Transformation: Fuse Iteration Nodes
// ---------------------------------------------------------------------------
//
// Fasst zwei DIREKT verschachtelte Iterationsknoten zu einem zusammen.
// Zulaessig ist das nur, wenn die beiden Achsen im Speicher zusammenhaengen,
// also fuer JEDEN Tensor gilt:  stride(aussen) == extent(innen) * stride(innen).
// Sonst waere der fusionierte Index nicht mehr linear abbildbar und der
// Codegen wuerde falsche Adressen berechnen. Die Pruefung ist der Grund, warum
// die Funktion einen Zustand zurueckweisen darf, statt blind zu fusionieren.
void TEIROptimizer::fuse_iteration_nodes(const std::string& ax1, const std::string& ax2, const std::string& fused_ax) {
    const std::string outer = strip_at(ax1), inner = strip_at(ax2), fused = strip_at(fused_ax);

    auto n_outer = find_iter_node(outer);
    auto n_inner = find_iter_node(inner);
    if (!n_outer || !n_inner) return;

    // n_inner muss direktes Kind von n_outer sein.
    bool nested = false;
    for (auto& c : n_outer->children) if (c == n_inner) nested = true;
    if (!nested) return;

    auto a_it = prog.axes.find(outer), b_it = prog.axes.find(inner);
    if (a_it == prog.axes.end() || b_it == prog.axes.end()) return;

    // Zusammenhang pruefen -- fuer jeden Tensor, den beide Achsen beruehren.
    for (auto const& [tensor, s_outer] : a_it->second.strides) {
        auto s_in = b_it->second.strides.find(tensor);
        if (s_in == b_it->second.strides.end()) return;          // nur eine Achse -> nicht fusionierbar
        if (s_outer != b_it->second.extent * s_in->second) return; // Luecke -> nicht fusionierbar
    }

    // Neue Achse: Extent multipliziert, Strides von der INNEREN Achse.
    Axis fused_axis = b_it->second;
    fused_axis.name   = fused;
    fused_axis.extent = a_it->second.extent * b_it->second.extent;
    prog.axes[fused]  = fused_axis;

    auto fused_node = std::make_shared<IterNode>();
    fused_node->axis     = "@" + fused;
    fused_node->name     = "iter_" + fused;
    fused_node->policy   = n_outer->policy;      // aeussere Policy gewinnt
    fused_node->children = n_inner->children;

    replace_node(n_outer, { fused_node });
}

// ---------------------------------------------------------------------------
// Transformation: Reorder Schedule Chain
// ---------------------------------------------------------------------------
//
// Ordnet eine LINEARE Kette verschachtelter Iterationsknoten um. Die Kette
// beginnt bei `parent` und laeuft so weit, wie jeder Knoten genau ein
// Iter-Kind hat. Achsen, die in `desired_order` fehlen, bleiben in ihrer
// urspruenglichen Reihenfolge hinten stehen -- so geht keine Schleife verloren.
void TEIROptimizer::reorder_schedule_chain(std::shared_ptr<IterNode> parent, std::vector<std::string> desired_order) {
    if (!parent) return;

    // Kette einsammeln.
    std::vector<std::shared_ptr<IterNode>> chain;
    auto cur = parent;
    while (cur) {
        chain.push_back(cur);
        std::shared_ptr<IterNode> next = nullptr;
        int iter_children = 0;
        for (auto& c : cur->children) {
            if (c->type == NodeType::Iter) {
                ++iter_children;
                next = std::static_pointer_cast<IterNode>(c);
            }
        }
        cur = (iter_children == 1) ? next : nullptr;   // Verzweigung beendet die Kette
    }
    if (chain.size() < 2) return;

    // Der letzte Knoten haelt die eigentliche Nutzlast (Invokes).
    std::vector<std::shared_ptr<ScheduleNode>> payload;
    for (auto& c : chain.back()->children)
        if (c->type != NodeType::Iter) payload.push_back(c);

    // Gewuenschte Reihenfolge, danach die nicht genannten in Originalreihenfolge.
    std::vector<std::shared_ptr<IterNode>> ordered;
    for (auto const& want : desired_order) {
        const std::string w = strip_at(want);
        for (auto& n : chain)
            if (strip_at(n->axis) == w &&
                std::find(ordered.begin(), ordered.end(), n) == ordered.end())
                ordered.push_back(n);
    }
    for (auto& n : chain)
        if (std::find(ordered.begin(), ordered.end(), n) == ordered.end())
            ordered.push_back(n);

    // Neu verketten.
    for (size_t i = 0; i < ordered.size(); ++i) {
        ordered[i]->children.clear();
        if (i + 1 < ordered.size()) ordered[i]->children.push_back(ordered[i + 1]);
        else                        ordered[i]->children = payload;
    }
    replace_node(parent, { ordered.front() });
}

void TEIROptimizer::set_policy(const std::string& axis, const std::string& policy) {
    auto node = find_iter_node(axis);
    if (node) node->policy = policy;
}

// ---------------------------------------------------------------------------
// Transformation: Promote to Primitive
// ---------------------------------------------------------------------------
//
// Zieht eine Achse aus dem Schleifennest in das Primitive hinein: Statt einer
// Schleife uebernimmt der Kernel diese Dimension selbst. Sinnvoll fuer die
// innersten Achsen, weil der Kernel sie vektorisieren kann.
//
// Die Rolle (M/N/K) wird aus den Strides ABGELEITET, nicht vorgegeben -- das
// ist der Unterschied zu einem fest eingetragenen Achsennamen.
void TEIROptimizer::promote_to_primitive(const std::string& axis, const std::string& prim_name) {
    const std::string ax = strip_at(axis);
    const std::string pn = strip_at(prim_name);

    auto prim_it = prog.primitives.find(pn);
    if (prim_it == prog.primitives.end()) return;

    auto node = find_iter_node(ax);
    if (!node) return;

    const std::string role = infer_role(ax);
    if (role.empty()) return;   // nicht eindeutig -> lieber nichts tun

    // Schon vorhanden? Dann nur die Schleife entfernen.
    auto& slot = prim_it->second.axes_map[role];
    if (std::find(slot.begin(), slot.end(), "@" + ax) == slot.end())
        slot.push_back("@" + ax);

    // Schleife aus dem Nest herausloesen: Kinder ruecken an ihre Stelle.
    replace_node(node, node->children);
}

void TEIROptimizer::apply_cache_blocking_matmul() {
    split_iteration("m0", "m_in", 4);
    split_iteration("n0", "n_in", 16);
            
    auto k0 = find_iter_node("k0");
    auto m0 = find_iter_node("m0");
    auto n0 = find_iter_node("n0");
    auto m_in = find_iter_node("m_in");
    auto n_in = find_iter_node("n_in");
            
    if (!k0 || !m0 || !n0 || !m_in || !n_in) return;
            
    auto inv_zero = k0->children[0];
    auto inv_gemm = n_in->children[0];

    prog.roots = { m0 };
    m0->children = { n0 };
    n0->children = { k0 };
    k0->children = { m_in };
    m_in->children = { n_in };
    n_in->children = { inv_zero, inv_gemm }; 

    auto invoke_z = std::static_pointer_cast<InvokeNode>(inv_zero);
    invoke_z->guard = "first(@k0)"; 
}

// Zielplattform-Kennwerte. Ausgelesen auf dem Referenzrechner (Apple M4 Max)
// via sysctl: hw.perflevel0/1.physicalcpu, hw.l1dcachesize, hw.l2cachesize,
// hw.cachelinesize. Die ZA-Kachel ergibt sich aus SVL=512 bit -> 16 fp32-Lanes,
// vier Kacheln bilden den 32x32-Akkumulator.
TEIROptimizer::HardwareParams TEIROptimizer::hardware() {
    return HardwareParams{};   // Defaults siehe teir_optimizer.h
}

// ---------------------------------------------------------------------------
// Heuristik: Parallelitaet freilegen
// ---------------------------------------------------------------------------
//
// Vorher war das eine Fallunterscheidung ueber prog.name ("matmul" -> m0,
// "contraction" -> p und r, ...). Damit wurde jedes andere Programm gar nicht
// parallelisiert -- der Rueckfallzweig setzte ausserdem root->name statt
// root->policy und war wirkungslos.
//
// Jetzt aus der IR abgeleitet:
//   - Reduktionsachsen (Rolle K) werden NIE parallelisiert: mehrere Threads
//     wuerden dieselbe Ausgabezelle akkumulieren (Schreibkonflikt).
//   - Es zaehlt nicht der Extent einer einzelnen Achse, sondern das PRODUKT der
//     parallel geschalteten Achsen: es wird abgestiegen, bis genug unabhaengige
//     Iterationen fuer alle P-Kerne zusammenkommen. Eine einzelne Schwelle
//     "Extent >= p_cores" laesst Programme wie einsum (aeussere Achsen 4, 4, 3)
//     komplett sequenziell -- gemessen 1,00x, obwohl 48 unabhaengige
//     Iterationen vorliegen.
//   - Hoechstens drei Ebenen, damit die Kette ueberschaubar bleibt.
//
// Der Evaluator flacht eine Kette direkt geschachtelter paralleler Achsen zu
// EINER OpenMP-Schleife ab (siehe teir_evaluator.cpp). Das ist die Voraussetzung
// dafuer, dass mehrere Ebenen ueberhaupt etwas bringen: geschachtelte
// parallel-Regionen ignoriert OpenMP standardmaessig.
void TEIROptimizer::expose_parallelism() {
    const auto hw = hardware();
    const int  kMaxLevels = 3;

    // Erstes Iter-Kind einer Knotenliste; Invokes werden uebersprungen.
    auto first_iter = [](const std::vector<std::shared_ptr<ScheduleNode>>& nodes)
                      -> std::shared_ptr<IterNode> {
        for (auto& n : nodes)
            if (n->type == NodeType::Iter) return std::static_pointer_cast<IterNode>(n);
        return nullptr;
    };

    long exposed = 1;   // Produkt der bereits parallelen Extents
    int  levels  = 0;

    // Bereits durchlaufene (weiter aussen liegende) Achsen.
    std::vector<std::string> outer;

    // Schreiben verschiedene Iterationen von `ax` disjunkte Bereiche in `out`?
    //
    // Notwendige Bedingung: der Stride von `ax` muss GROESSER sein als die
    // Spanne, die alle weiter innen liegenden Achsen zusammen ueberdecken.
    // Sonst ueberlappen sich die Schreibbereiche, das Ergebnis haengt von der
    // Reihenfolge ab -- sequenziell gewinnt deterministisch der letzte
    // Schreiber, parallel entscheidet der Zufall.
    //
    // transposition.teir verletzt das: `a` hat Stride 48, waehrend b, c und d
    // zusammen 18083567 Elemente ueberdecken. Das Modell liest 18874368 Werte
    // und schreibt sie in 18088128 Zellen -- Kollisionen sind unvermeidlich.
    // Ohne diese Pruefung lieferte Stufe 1 in einem von drei Laeufen falsche
    // Ergebnisse (Mismatch an Index 4464), und die Verifikation in main.cpp
    // meldete das nur sporadisch.
    auto writes_disjoint = [&](const std::string& ax) {
        auto out_stride = [&](const std::string& n) -> long {
            auto it = prog.axes.find(n);
            if (it == prog.axes.end()) return 0;
            auto s = it->second.strides.find("out");
            return s == it->second.strides.end() ? 0 : s->second / 4;
        };
        const long s_ax = out_stride(ax);
        if (s_ax == 0) return false;         // schreibt gar nicht -> Reduktion

        long span_inner = 0;
        for (auto const& [name, axis] : prog.axes) {
            if (name == ax) continue;
            if (std::find(outer.begin(), outer.end(), name) != outer.end()) continue;
            span_inner += static_cast<long>(axis.extent - 1) * out_stride(name);
        }
        return s_ax > span_inner;
    };

    // Entlang der aeusseren Schleifenkette absteigen. Reduktionsachsen werden
    // uebersprungen, aber durchlaufen -- unter einer sequenziellen K-Achse kann
    // sehr wohl eine parallelisierbare Achse liegen (so ist matmul aufgebaut).
    for (auto node = first_iter(prog.roots);
         node && levels < kMaxLevels && exposed < hw.p_cores;
         node = first_iter(node->children)) {

        const std::string ax = strip_at(node->axis);
        auto a = prog.axes.find(ax);
        const int extent = (a == prog.axes.end()) ? 0 : a->second.extent;

        if (infer_role(ax) != "K" && extent > 1 && writes_disjoint(ax)) {
            node->policy = "parallel";
            exposed *= extent;
            ++levels;
        }
        outer.push_back(ax);
    }
}

// ---------------------------------------------------------------------------
// Heuristik: Cache-Blocking
// ---------------------------------------------------------------------------
//
// Ersetzt die vier fallspezifischen apply_cache_blocking_*-Funktionen, die
// Achsennamen und Faktoren fest eingetragen hatten.
//
// Modell: Der innerste Block arbeitet auf einer M_t x N_t grossen Kachel von C
// und braucht dafuer zusaetzlich je einen Streifen aus A und B. Bei einem
// K-Streifen der Laenge k betraegt der Arbeitssatz
//     (M_t*N_t + M_t*k + k*N_t) * 4 Byte.
// Gesucht ist die groesste Kachel, die noch in den L1-Datencache passt.
//
// Die Kachelkanten sind Vielfache der ZA-Kachel (16), damit der SME-Kernel sie
// ohne Rest verarbeiten kann. Ein Split wird nur eingebaut, wenn er die Achse
// echt teilt und der innere Teil kleiner als das Original ist.
void TEIROptimizer::apply_cache_blocking() {
    const auto hw = hardware();

    // Groesste Kachelkante t (Vielfaches von za_tile), sodass der Arbeitssatz
    // mit einem K-Streifen von einer Kachelbreite in den L1 passt.
    auto fits = [&](long t) {
        const long ws = (t * t + 2 * t * hw.za_tile) * 4;
        return ws <= hw.l1d_bytes;
    };
    long tile = hw.za_tile;
    while (fits(tile + hw.za_tile)) tile += hw.za_tile;

    // Kandidaten: Achsen mit Rolle M oder N, deren Extent groesser als die
    // Kachel ist und die sich sauber teilen lassen.
    std::vector<std::pair<std::string,int>> splits;
    for (auto const& [name, ax] : prog.axes) {
        const std::string role = infer_role(name);
        if (role != "M" && role != "N") continue;
        if (ax.extent <= tile) continue;
        if (ax.extent % tile != 0) continue;
        if (!find_iter_node(name)) continue;      // Achse hat keine Schleife
        splits.emplace_back(name, static_cast<int>(tile));
    }

    for (auto const& [name, inner_extent] : splits)
        split_iteration(name, name + "_in", inner_extent);
}

void TEIROptimizer::apply_cache_blocking_contraction() {
    split_iteration("p", "p_in", 4);
    split_iteration("r", "r_in", 4);
    
    auto p0 = find_iter_node("p");
    auto r0 = find_iter_node("r");
    auto t0 = find_iter_node("t");
    auto p_in = find_iter_node("p_in");
    auto r_in = find_iter_node("r_in");
    
    if (!p0 || !r0 || !t0 || !p_in || !r_in) return;
    
    auto inv_zero = t0->children[0];
    auto inv_gemm = t0->children[1];
    
    prog.roots = { p0 };
    p0->children = { r0 };
    r0->children = { t0 };
    t0->children = { p_in };
    p_in->children = { r_in };
    r_in->children = { inv_zero, inv_gemm };
}

void TEIROptimizer::apply_cache_blocking_einsum() {
    split_iteration("a", "a_in", 2);
    split_iteration("b", "b_in", 2);
    
    auto a0 = find_iter_node("a");
    auto a_in = find_iter_node("a_in");
    auto b0 = find_iter_node("b");
    auto b_in = find_iter_node("b_in");
    auto c0 = find_iter_node("c");
    auto s0 = find_iter_node("s");
    
    if (!a0 || !b0 || !c0 || !a_in || !b_in || !s0) return;
    
    auto inv_zero = s0->children[0];
    auto inv_gemm = s0->children[1];
    
    prog.roots = { a0 };
    a0->children = { b0 };
    b0->children = { c0 };
    c0->children = { s0 };
    s0->children = { a_in };
    a_in->children = { b_in };
    
    b_in->children = { inv_zero, inv_gemm };
}

void TEIROptimizer::apply_cache_blocking_transposition() {
    split_iteration("a", "a_in", 4);
    split_iteration("b", "b_in", 4);

    auto a0 = find_iter_node("a");
    auto b0 = find_iter_node("b");
    auto a_in = find_iter_node("a_in");
    auto b_in = find_iter_node("b_in");

    if (!a0 || !b0 || !a_in || !b_in) return;

    auto inv_copy = b_in->children[0];

    prog.roots = { a0 };
    a0->children = { b0 };
    b0->children = { a_in };
    a_in->children = { b_in };
    b_in->children = { inv_copy };
}

