#pragma once
#include "teir.h"
#include <unordered_map>
#include <vector>

class TEIREvaluator {
public:
    explicit TEIREvaluator(TEIRProgram p) : prog(std::move(p)) {}

    // Führt das TEIR-Programm rekursiv für die gegebenen Tensoren (args) aus.
    void evaluate(void** args);

private:
    TEIRProgram prog;

    long long get_offset(const std::string& tensor_name,
                         const std::unordered_map<std::string, int>& active_loops);

    void evaluate_primitive(const Primitive& prim, void** args,
                            const std::unordered_map<std::string, int>& active_loops);

    void evaluate_node(const std::shared_ptr<ScheduleNode>& node, void** args,
                       std::unordered_map<std::string, int> active_loops);

    bool try_evaluate_neon_contraction(const Primitive& prim, void** args,
                                       const std::unordered_map<std::string, int>& active_loops);

    // Helpers für Primitiveauswertung mit noch fehlenden Achsen
    void lower_primitive_recursive(const Primitive& prim, void** args,
                                   std::unordered_map<std::string, int>& active_loops,
                                   const std::vector<std::string>& prim_axes,
                                   size_t depth);
};
