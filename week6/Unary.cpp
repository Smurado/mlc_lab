#include "Unary.h"
#include <sys/mman.h>
#include <iostream>
#include <vector>

namespace mini_jit {

Unary::~Unary() {
    if (m_code) {
        munmap(m_code, m_size);
    }
}

Unary::kernel_t Unary::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_code);
}

Unary::error_t Unary::generate(uint32_t m, uint32_t n, uint32_t trans_b, dtype_t dtype, ptype_t ptype) {
    if (m % 16 != 0 || n % 16 != 0 || dtype != dtype_t::fp32) {
        return error_t::success; // unsupported
    }

    size_t size = 4096;
    void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        return error_t::success;
    }

    uint32_t* code = reinterpret_cast<uint32_t*>(mem);
    int idx = 0;

    auto emit = [&](uint32_t inst) { code[idx++] = inst; };

    auto encode_mov_k_x = [](int reg, uint16_t imm16) -> uint32_t {
        return 0xD2800000 | (imm16 << 5) | reg;
    };

    auto encode_b_ne = [&](int target_idx, int current_idx) -> uint32_t {
        int offset = target_idx - current_idx;
        return 0x54000001 | ((offset & 0x7FFFF) << 5);
    };

    // ------------------------------------------------------------------
    // Prologue: save callee-saved FP regs d8..d15 (AAPCS64).
    // 'smstart sm' nullt die Streaming-SVE-Register inkl. v8..v15.
    // Damit der C++-Aufrufer unter -O3 seine Locals (die der Compiler
    // gern in d8..d15 ablegt) nicht verliert, sichern wir sie hier.
    // ------------------------------------------------------------------
    emit(0x6DBC27E8); // stp d8,  d9,  [sp, #-64]!
    emit(0x6D012FEA); // stp d10, d11, [sp, #16]
    emit(0x6D0237EC); // stp d12, d13, [sp, #32]
    emit(0x6D033FEE); // stp d14, d15, [sp, #48]

if (ptype == ptype_t::zero) {
    emit(0xd503437f); // smstart sm
    emit(0x2598e120); // ptrue p0.s, vl16
    emit(0xd37ef463); // lsl x3, x3, #2 (ld_b in Bytes)
    emit(0x25b8c000); // mov z0.s, #0

    // RETTEN: Kopiere den Basispointer x1 einmalig vor den Schleifen
    emit(0xaa0103e6); // mov x6, x1

    emit(encode_mov_k_x(9, n));       // mov x9, n
    
    int outer_start = idx;
    emit(encode_mov_k_x(10, m / 16)); // mov x10, m / 16
    emit(0xaa0603e4);                 // mov x4, x6 (Nutze geretteten Pointer!)

    int inner_start = idx;
    emit(0xe540e080);                    // st1w {z0.s}, p0, [x4]
    emit(0x91010084);                    // add x4, x4, #64
    emit(0xf100054a);                    // subs x10, x10, #1
    emit(encode_b_ne(inner_start, idx)); // b.ne inner
    
    emit(0x8b0300c6);                    // add x6, x6, x3 (Erhöhe den geretteten Basispointer!)
    emit(0xf1000529);                    // subs x9, x9, #1
    emit(encode_b_ne(outer_start, idx)); // b.ne outer

    emit(0xd503427f); // smstop sm
}
else if (ptype == ptype_t::identity) {
    emit(0xd503437f); // smstart sm
    emit(0x2598e120); // ptrue p0.s, vl16
    emit(0xd37ef442); // lsl x2, x2, #2
    emit(0xd37ef463); // lsl x3, x3, #2

    if (trans_b == 0) {
        // RETTEN: Basispointer vor den Schleifen wegsichern
        emit(0xaa0003e6); // mov x6, x0
        emit(0xaa0103e7); // mov x7, x1

        emit(encode_mov_k_x(9, n));       // mov x9, n
        int outer_start = idx;
        emit(encode_mov_k_x(10, m / 16)); // mov x10, m / 16
        emit(0xaa0603e4);                 // mov x4, x6
        emit(0xaa0703e5);                 // mov x5, x7
        
        int inner_start = idx;
        emit(0xa540a080);                    // ld1w {z0.s}, p0/z, [x4]
        emit(0xe540e0a0);                    // st1w {z0.s}, p0, [x5]
        emit(0x91010084);                    // add x4, x4, #64
        emit(0x910100a5);                    // add x5, x5, #64
        emit(0xf100054a);                    // subs x10, x10, #1
        emit(encode_b_ne(inner_start, idx));
        
        emit(0x8b0200c6);                    // add x6, x6, x2
        emit(0x8b0300e7);                    // add x7, x7, x3
        emit(0xf1000529);                    // subs x9, x9, #1
        emit(encode_b_ne(outer_start, idx));
    }

    emit(0xd503427f); // smstop sm
}
else if (ptype == ptype_t::relu) {
    // Register-Layout: x0 = a, x1 = b, x2 = ld_a, x3 = ld_b
    emit(0xd503437f); // smstart sm
    emit(0x2598e120); // ptrue p0.s, vl16
    emit(0x25b8c01f); // mov z31.s, #0.0
    emit(0xd37ef442); // lsl x2, x2, #2
    emit(0xd37ef463); // lsl x3, x3, #2

    if (trans_b == 0) {
        // RETTEN: Basispointer vor dem Eintritt in die Schleifen wegsichern
        emit(0xaa0003e6); // mov x6, x0
        emit(0xaa0103e7); // mov x7, x1

        emit(encode_mov_k_x(9, n));       // mov x9, n
        int outer_start = idx;
        emit(encode_mov_k_x(10, m / 16)); // mov x10, m / 16
        emit(0xaa0603e4);                 // mov x4, x6
        emit(0xaa0703e5);                 // mov x5, x7
        
        int inner_start = idx;
        emit(0xa540a080);                    // ld1w {z0.s}, p0/z, [x4]
        emit(0x658683e0);                    // fmax z0.s, p0/m, z0.s, z31.s
        emit(0xe540e0a0);                    // st1w {z0.s}, p0, [x5]
        emit(0x91010084);                    // add x4, x4, #64
        emit(0x910100a5);                    // add x5, x5, #64
        emit(0xf100054a);                    // subs x10, x10, #1
        emit(encode_b_ne(inner_start, idx));
        
        emit(0x8b0200c6);                    // add x6, x6, x2 (Erhöhe geretteten Pointer A)
        emit(0x8b0300e7);                    // add x7, x7, x3 (Erhöhe geretteten Pointer B)
        emit(0xf1000529);                    // subs x9, x9, #1
        emit(encode_b_ne(outer_start, idx));
    }

    emit(0xd503427f); // smstop sm
}

    // ------------------------------------------------------------------
    // Epilogue: d8..d15 wiederherstellen (in umgekehrter Reihenfolge).
    // ------------------------------------------------------------------
    emit(0x6D433FEE); // ldp d14, d15, [sp, #48]
    emit(0x6D4237EC); // ldp d12, d13, [sp, #32]
    emit(0x6D412FEA); // ldp d10, d11, [sp, #16]
    emit(0x6CC427E8); // ldp d8,  d9,  [sp], #64
    emit(0xd65f03c0); // ret

    mprotect(mem, size, PROT_READ | PROT_EXEC);
    
    this->m_code = mem;
    this->m_size = size;

    return error_t::success;
}

} // namespace mini_jit
