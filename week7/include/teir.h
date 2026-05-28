// TEIR Abstract Syntax Tree (AST) & gemeinsame Typen.
// Wird sowohl vom Parser als auch vom JIT-Compiler verwendet.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct Axis {
    std::string name;
    int extent;
    std::unordered_map<std::string, int> strides;
};

struct Primitive {
    std::string name;
    std::string kind; // "Zero" | "Copy" | "Contraction"
    std::unordered_map<std::string, std::vector<std::string>> axes_map;
};

enum class NodeType { Iter, Invoke };

struct ScheduleNode {
    NodeType type;
    std::string name;
    virtual ~ScheduleNode() = default;
};

struct InvokeNode : public ScheduleNode {
    std::string primitive;
    std::string guard;
    InvokeNode() { type = NodeType::Invoke; }
};

struct IterNode : public ScheduleNode {
    std::string axis;
    std::string policy; // "parallel" | "sequential"
    std::vector<std::shared_ptr<ScheduleNode>> children;
    IterNode() { type = NodeType::Iter; }
};

struct TEIRProgram {
    std::string name;
    std::vector<std::string> tensors;
    std::unordered_map<std::string, Axis> axes;
    std::unordered_map<std::string, Primitive> primitives;
    std::vector<std::shared_ptr<ScheduleNode>> roots;
};

// Signatur des per JIT erzeugten Kernels (args = Array von Tensor-Pointern).
using TEIRKernelPtr = void(*)(void**);
