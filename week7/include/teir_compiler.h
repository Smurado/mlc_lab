// JIT-Codegen + Compiler/Loader fuer TEIR-Programme.
#pragma once

#include "teir.h"

#include <sstream>
#include <string>
#include <unordered_map>

class TEIRCompiler {
public:
    explicit TEIRCompiler(TEIRProgram p) : prog(std::move(p)) {}

    // Erzeugt den C++-Source fuer den Kernel.
    std::string generate_cpp_source();

    // Schreibt den Source, ruft clang++ auf, laedt das .so und liefert den
    // Function-Pointer auf den Kernel. nullptr bei Fehler.
    TEIRKernelPtr compile();

private:
    TEIRProgram prog;
    std::stringstream src;
    int indent_level = 0;

    void emit(const std::string& line);
    std::string get_offset_expr(const std::string& tensor_name,
                                const std::unordered_map<std::string, std::string>& active_loops);
    void emit_kernel_body(const Primitive& prim,
                          const std::unordered_map<std::string, std::string>& all_loops);
    bool try_emit_neon_contraction(const Primitive& prim,
                                   const std::unordered_map<std::string, std::string>& active_loops);
    void lower_primitive(const Primitive& prim,
                         std::unordered_map<std::string, std::string> active_loops);
    void lower_node(const std::shared_ptr<ScheduleNode>& node,
                    std::unordered_map<std::string, std::string> active_loops);
};
