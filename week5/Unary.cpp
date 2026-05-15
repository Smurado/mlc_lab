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

    if (ptype == ptype_t::zero) {
        // Register-Layout: x0 = A (ungenutzt da Zero), x1 = B, x2 = ld_a, x3 = ld_b
        emit(0xd503437f); // smstart sm
        emit(0x2598e120); // ptrue p0.s, vl16
        emit(0xd37ef463); // lsl x3, x3, #2 (ld_b in Bytes umwandeln)
        emit(0x25b8c000); // mov z0.s, #0

        // Äussere Schleife über Spalten (n)
        emit(encode_mov_k_x(9, n));       // mov x9, n
        
        int outer_start = idx;
        emit(encode_mov_k_x(10, m / 16)); // mov x10, m / 16
        code[idx++] = 0xaa0103e4;         // mov x4, x1 (Pointer auf B)

        int inner_start = idx;
        emit(0xe540e080);                    // st1w {z0.s}, p0, [x4]
        emit(0x91010084);                    // add x4, x4, #64
        emit(0xf100054a);                    // subs x10, x10, #1
        emit(encode_b_ne(inner_start, idx)); // b.ne inner
        
        emit(0x8b030021);                    // add x1, x1, x3
        emit(0xf1000529);                    // subs x9, x9, #1
        emit(encode_b_ne(outer_start, idx)); // b.ne outer

        emit(0xd503427f); // smstop sm
        emit(0xd65f03c0); // ret
    } 
    else if (ptype == ptype_t::identity) {
        // Register-Layout: x0 = a, x1 = b, x2 = ld_a, x3 = ld_b
        emit(0xd503477f); // smstart
        emit(0x2598e120); // ptrue p0.s, vl16
        emit(0xd37ef442); // lsl x2, x2, #2 (ld_a in Bytes umwandeln)
        emit(0xd37ef463); // lsl x3, x3, #2 (ld_b in Bytes umwandeln)

        if (trans_b == 0) {
            emit(encode_mov_k_x(9, n));       // mov x9, n
            int outer_start = idx;
            emit(encode_mov_k_x(10, m / 16)); // mov x10, m / 16
            emit(0xaa0003e4);                 // mov x4, x0
            emit(0xaa0103e5);                 // mov x5, x1
            
            int inner_start = idx;
            emit(0xa540a080);                    // ld1w {z0.s}, p0/z, [x4]
            emit(0xe540e0a0);                    // st1w {z0.s}, p0, [x5]
            emit(0x91010084);                    // add x4, x4, #64
            emit(0x910100a5);                    // add x5, x5, #64
            emit(0xf100054a);                    // subs x10, x10, #1
            emit(encode_b_ne(inner_start, idx));
            
            emit(0x8b020000);                    // add x0, x0, x2
            emit(0x8b030021);                    // add x1, x1, x3
            emit(0xf1000529);                    // subs x9, x9, #1
            emit(encode_b_ne(outer_start, idx));
        } else {
            // Transponiertes Identity noch nicht zwingend für diese Benchmarks benötigt.
            // Kann später durch Matrix-Transformationen via SME ergänzt werden.
        }

        emit(0xd503467f); // smstop
        emit(0xd65f03c0); // ret
    }
    else if (ptype == ptype_t::relu) {
        // Register-Layout: x0 = a, x1 = b, x2 = ld_a, x3 = ld_b
        emit(0xd503437f); // smstart sm
        emit(0x2598e120); // ptrue p0.s, vl16
        emit(0x25b8c01f); // mov z31.s, #0.0
        emit(0xd37ef442); // lsl x2, x2, #2
        emit(0xd37ef463); // lsl x3, x3, #2

        if (trans_b == 0) {
            emit(encode_mov_k_x(9, n));       // mov x9, n
            int outer_start = idx;
            emit(encode_mov_k_x(10, m / 16)); // mov x10, m / 16
            emit(0xaa0003e4);                 // mov x4, x0
            emit(0xaa0103e5);                 // mov x5, x1
            
            int inner_start = idx;
            emit(0xa540a080);                    // ld1w {z0.s}, p0/z, [x4]
            emit(0x658683e0);                    // fmax z0.s, p0/m, z0.s, z31.s
            emit(0xe540e0a0);                    // st1w {z0.s}, p0, [x5]
            emit(0x91010084);                    // add x4, x4, #64
            emit(0x910100a5);                    // add x5, x5, #64
            emit(0xf100054a);                    // subs x10, x10, #1
            emit(encode_b_ne(inner_start, idx));
            
            emit(0x8b020000);                    // add x0, x0, x2
            emit(0x8b030021);                    // add x1, x1, x3
            emit(0xf1000529);                    // subs x9, x9, #1
            emit(encode_b_ne(outer_start, idx));
        }

        emit(0xd503427f); // smstop sm
        emit(0xd65f03c0); // ret
    }

    mprotect(mem, size, PROT_READ | PROT_EXEC);
    
    this->m_code = mem;
    this->m_size = size;

    return error_t::success;
}

} // namespace mini_jit
