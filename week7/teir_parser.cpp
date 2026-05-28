#include "include/teir_parser.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

class TEIRParser {
    std::vector<std::string> toks;
    size_t pos = 0;
    std::unordered_map<std::string, std::vector<std::string>> pending_children;

    static std::string strip_sigil(const std::string& s) {
        if (!s.empty() && (s[0] == '@' || s[0] == '%')) return s.substr(1);
        return s;
    }

    void tokenize(const std::string& src) {
        std::string buf;
        auto flush = [&]() { if (!buf.empty()) { toks.push_back(buf); buf.clear(); } };
        bool in_line_comment = false;
        for (size_t i = 0; i < src.size(); ++i) {
            char c = src[i];
            if (in_line_comment) {
                if (c == '\n') in_line_comment = false;
                continue;
            }
            if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') {
                flush();
                in_line_comment = true;
                ++i;
                continue;
            }
            if (c == '{' || c == '}' || c == '[' || c == ']' ||
                c == '(' || c == ')' || c == ',' || c == ':') {
                flush();
                toks.push_back(std::string(1, c));
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                flush();
            } else {
                buf += c;
            }
        }
        flush();
    }

    std::string consume() { return toks[pos++]; }
    void expect(const std::string& s) {
        if (pos >= toks.size() || toks[pos] != s) {
            throw std::runtime_error("TEIR parse error: expected '" + s +
                                     "', got '" + (pos < toks.size() ? toks[pos] : "<eof>") + "'");
        }
        ++pos;
    }
    bool accept(const std::string& s) {
        if (pos < toks.size() && toks[pos] == s) { ++pos; return true; }
        return false;
    }

    // Überspringt einen Block `{ ... }` (für unbekannte/optionale Sektionen wie metadata).
    void skip_brace_block() {
        expect("{");
        int depth = 1;
        while (depth > 0 && pos < toks.size()) {
            const std::string& t = toks[pos++];
            if (t == "{") ++depth;
            else if (t == "}") --depth;
        }
    }

    void parse_strides(Axis& ax) {
        expect("{");
        while (!accept("}")) {
            std::string tname = consume();
            expect(":");
            ax.strides[tname] = std::stoi(consume());
            accept(",");
        }
    }

    void parse_primitive_axes(Primitive& p) {
        expect("{");
        while (!accept("}")) {
            std::string label = consume();
            expect(":");
            expect("[");
            while (!accept("]")) {
                p.axes_map[label].push_back(consume()); // behält "@" Präfix
                accept(",");
            }
            accept(",");
        }
    }

    void parse_schedule(TEIRProgram& prog,
                       std::unordered_map<std::string, std::shared_ptr<ScheduleNode>>& nodes,
                       std::vector<std::string>& root_ids) {
        (void)prog;
        expect("{");
        while (!accept("}")) {
            std::string kw = consume();
            if (kw == "roots") {
                expect("[");
                while (!accept("]")) {
                    root_ids.push_back(strip_sigil(consume()));
                    accept(",");
                }
            } else if (kw == "iter") {
                auto node = std::make_shared<IterNode>();
                node->name = strip_sigil(consume());
                expect("axis");
                node->axis = strip_sigil(consume());
                expect("policy");
                node->policy = consume();
                expect("children");
                expect("[");
                std::vector<std::string> child_ids;
                while (!accept("]")) {
                    child_ids.push_back(strip_sigil(consume()));
                    accept(",");
                }
                pending_children[node->name] = child_ids;
                nodes[node->name] = node;
            } else if (kw == "invoke") {
                auto node = std::make_shared<InvokeNode>();
                node->name = strip_sigil(consume());
                expect("primitive");
                node->primitive = strip_sigil(consume());
                if (accept("guard")) {
                    std::string fname = consume(); // z.B. "first"
                    expect("(");
                    std::string ax = consume();    // z.B. "@t"
                    expect(")");
                    node->guard = fname + "(" + ax + ")";
                }
                nodes[node->name] = node;
            } else {
                throw std::runtime_error("TEIR parse error: unknown schedule entry '" + kw + "'");
            }
        }
    }

public:
    TEIRProgram parse_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open TEIR file: " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        return parse(ss.str());
    }

    TEIRProgram parse(const std::string& src) {
        toks.clear(); pos = 0; pending_children.clear();
        tokenize(src);

        TEIRProgram prog;
        std::unordered_map<std::string, std::shared_ptr<ScheduleNode>> nodes;
        std::vector<std::string> root_ids;

        expect("teir");
        prog.name = strip_sigil(consume());
        expect("{");

        while (!accept("}")) {
            std::string kw = consume();
            if (kw == "tensor") {
                std::string tname = strip_sigil(consume());
                expect(":"); consume(); // dtype (f32)
                prog.tensors.push_back(tname);
            } else if (kw == "axis") {
                Axis ax;
                ax.name = strip_sigil(consume());
                expect("extent");
                ax.extent = std::stoi(consume());
                expect("strides");
                parse_strides(ax);
                prog.axes[ax.name] = ax;
            } else if (kw == "primitive") {
                Primitive p;
                p.name = strip_sigil(consume());
                expect(":");
                p.kind = consume();
                expect("axes");
                parse_primitive_axes(p);
                if (accept("metadata")) skip_brace_block();
                prog.primitives[p.name] = p;
            } else if (kw == "schedule") {
                parse_schedule(prog, nodes, root_ids);
            } else {
                throw std::runtime_error("TEIR parse error: unknown top-level keyword '" + kw + "'");
            }
        }

        // Kinder verlinken
        for (auto const& [id, ids] : pending_children) {
            auto it_node = std::static_pointer_cast<IterNode>(nodes.at(id));
            for (auto const& cid : ids) {
                auto child_it = nodes.find(cid);
                if (child_it == nodes.end())
                    throw std::runtime_error("TEIR parse error: unresolved child '" + cid + "'");
                it_node->children.push_back(child_it->second);
            }
        }
        for (auto const& rid : root_ids) {
            auto root_it = nodes.find(rid);
            if (root_it == nodes.end())
                throw std::runtime_error("TEIR parse error: unresolved root '" + rid + "'");
            prog.roots.push_back(root_it->second);
        }
        return prog;
    }
};

} // namespace

TEIRProgram load_teir(const std::string& path) {
    TEIRParser parser;
    return parser.parse_file(path);
}
