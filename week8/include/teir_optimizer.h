#pragma once
#include "teir.h"
#include <memory>
#include <string>
#include <vector>

class TEIROptimizer {
public:
    explicit TEIROptimizer(TEIRProgram& p) : prog(p) {}

    // Implementiert die TEIR Transformationen (Week 8 Task 1)
    void split_iteration(const std::string& parent_axis, const std::string& new_inner_axis, int split_factor);
    void fuse_iteration_nodes(const std::string& ax1, const std::string& ax2, const std::string& fused_ax);
    void reorder_schedule_chain(std::shared_ptr<IterNode> parent, std::vector<std::string> desired_order);
    void set_policy(const std::string& axis, const std::string& policy);
    void promote_to_primitive(const std::string& axis, const std::string& prim_name);

    // High Level Heuristik-Passes
    void expose_parallelism();
    void apply_cache_blocking();          // generisch, aus Hardware abgeleitet
    void apply_cache_blocking_matmul();
    void apply_cache_blocking_contraction();
    void apply_cache_blocking_einsum();
    void apply_cache_blocking_transposition();

    // Zielplattform-Parameter, aus denen die Heuristiken ihre Faktoren ableiten.
    // Werte fuer Apple M4 Max (ausgelesen via sysctl), siehe HardwareParams.
    struct HardwareParams {
        int  p_cores        = 12;      // hw.perflevel0.physicalcpu
        int  e_cores        = 4;       // hw.perflevel1.physicalcpu
        long l1d_bytes      = 65536;   // hw.l1dcachesize
        long l2_bytes       = 4194304; // hw.l2cachesize
        int  cacheline      = 128;     // hw.cachelinesize
        int  za_tile        = 16;      // SME: 16x16 fp32 je Kachel (SVL 512 bit)
        int  za_tiles       = 4;       // vier Kacheln -> 32x32-Akkumulator
    };
    static HardwareParams hardware();

private:
    TEIRProgram& prog;

    std::shared_ptr<IterNode> find_iter_node(const std::vector<std::shared_ptr<ScheduleNode>>& nodes, const std::string& axis);
    std::shared_ptr<IterNode> find_iter_node(const std::string& axis) { return find_iter_node(prog.roots, axis); }

    static std::shared_ptr<IterNode> find_parent_of(
        const std::vector<std::shared_ptr<ScheduleNode>>& nodes,
        const std::shared_ptr<ScheduleNode>& target);
    void replace_node(const std::shared_ptr<ScheduleNode>& target,
                      const std::vector<std::shared_ptr<ScheduleNode>>& replacement);
    std::string infer_role(const std::string& axis) const;
};
