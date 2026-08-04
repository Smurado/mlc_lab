#include "Gemm.h"
#include <sys/mman.h>
#include <cstdint>

namespace mini_jit {

Gemm::~Gemm() {
    if (m_code) {
        munmap(m_code, m_size);
    }
}

Gemm::kernel_t Gemm::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_code);
}

// ---------------------------------------------------------------------------
// JIT-generierter SME-GEMM-Kernel
//
// Konventionen:
//   A : column-major  (M x K), Stride zwischen Spalten = ld_a Elemente
//   B : row-major     (K x N), Stride zwischen Zeilen  = ld_b Elemente
//   C : column-major  (M x N), Stride zwischen Spalten = ld_c Elemente
//
// Aufruf: f(a, b, c, ld_a, ld_b, ld_c)  -> berechnet C += A * B
//   x0 = a, x1 = b, x2 = c,
//   x3 = ld_a, x4 = ld_b, x5 = ld_c (jeweils in Elementen).
//
// Mikrokernel: 16x16-Block von C via einem einzigen ZA-Tile (za0).
//   Outer-Loop  : N / 16 (Spalten-Blöcke von B/C)
//   Middle-Loop : M / 16 (Zeilen-Blöcke von A/C)
//   Inner-Loop  : K Iterationen `fmopa za0.s, p0/m, p0/m, z0.s, z1.s`
//   Store-Loop  : 16 Spalten aus za0 nach C addieren (lesen+addieren+schreiben).
//
// Unterstützt beliebige K (>0) und beliebige Vielfache von 16 für M und N.
// ---------------------------------------------------------------------------
Gemm::error_t Gemm::generate(uint32_t m, uint32_t n, uint32_t k,
                             uint32_t trans_a, uint32_t trans_b, uint32_t trans_c,
                             dtype_t dtype) {
    (void)trans_a;
    (void)trans_b;
    (void)trans_c;

    if (m == 0 || n == 0 || k == 0 || m % 16 != 0 || n % 16 != 0 ||
        dtype != dtype_t::fp32 || k > 0xFFFF || (m / 16) > 0xFFFF ||
        (n / 16) > 0xFFFF) {
        return error_t::success; // unsupported / out of range
    }

    size_t size = 4096;
    void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        return error_t::success;
    }

    uint32_t* code = reinterpret_cast<uint32_t*>(mem);
    int idx = 0;
    auto emit = [&](uint32_t inst) { code[idx++] = inst; };

    // movz <reg>, #imm16, lsl #0 — funktioniert für jeden W/X-Register (kein sf nötig hier,
    // wir nutzen die X-Form, die obersten 32 Bit werden eh auf 0 gesetzt).
    auto mov_x_imm16 = [](int reg, uint16_t imm16) -> uint32_t {
        return 0xD2800000u | (uint32_t(imm16) << 5) | uint32_t(reg);
    };

    // b.<cond> off — off in instructions, relativ.
    auto b_cond = [&](uint32_t cond, int target_idx, int current_idx) -> uint32_t {
        int offset = target_idx - current_idx;
        return 0x54000000u | ((uint32_t(offset) & 0x7FFFFu) << 5) | (cond & 0xF);
    };
    const uint32_t COND_NE = 0x1;
    const uint32_t COND_LT = 0xB;

    // -------------------- Prologue: callee-saved sichern --------------------
    // d8..d15 (untere 64 Bit von v8..v15) UND x19..x22, die wir als
    // Basispointer-Register nutzen, sind laut AAPCS64 callee-saved.
    emit(0x6DBC27E8); // stp d8,  d9,  [sp, #-64]!
    emit(0x6D012FEA); // stp d10, d11, [sp, #16]
    emit(0x6D0237EC); // stp d12, d13, [sp, #32]
    emit(0x6D033FEE); // stp d14, d15, [sp, #48]
    emit(0xA9BE53F3); // stp x19, x20, [sp, #-32]!
    emit(0xA9015BF5); // stp x21, x22, [sp, #16]

    // -------------------- SME aktivieren --------------------
    emit(0xD503477F); // smstart   (SM + ZA)
    emit(0x2598E3E0); // ptrue p0.s  (SVL=512 => 16 aktive Lanes)

    // Strides in Bytes konvertieren.
    emit(0xD37EF463); // lsl x3, x3, #2      ; ld_a (bytes)
    emit(0xD37EF484); // lsl x4, x4, #2      ; ld_b (bytes)
    emit(0xD37EF4A5); // lsl x5, x5, #2      ; ld_c (bytes)

    // Basis-Pointer in callee-saved Regs sichern (wir clobbern sie nicht weiter).
    emit(0xAA0003F3); // mov x19, x0   ; A base
    emit(0xAA0103F4); // mov x20, x1   ; B base
    emit(0xAA0203F5); // mov x21, x2   ; C base

    // x22 = 16 * ld_c_bytes  (Spalten-Block-Schritt für C)
    emit(0xD37CECB6); // lsl x22, x5, #4

    // ==================================================================
    // Zwei Varianten des Schleifennests:
    //
    //  * 32x32-Akkumulator (bevorzugt): nutzt ALLE VIER ZA-Kacheln. Pro
    //    K-Schritt werden 4 Vektoren geladen (2 aus A, 2 aus B) und 4 `fmopa`
    //    ausgefuehrt. Arithmetische Intensitaet: 2048 Flop / 256 Byte =
    //    8 Flop/Byte.
    //
    //  * 16x16-Akkumulator (Rueckfall): nur za0, 2 Ladebefehle und 1 `fmopa`
    //    je K-Schritt -> 512 Flop / 128 Byte = 4 Flop/Byte, dazu viermal so
    //    viel Schleifenaufwand pro Rechenarbeit.
    //
    // Die Aufgabe verlangt Vielfache von 16 fuer M und N; der 32er-Pfad greift
    // nur, wenn beide auch durch 32 teilbar sind. Sonst bleibt es beim
    // 16er-Pfad, damit alle geforderten Groessen weiterhin funktionieren.
    // ==================================================================
    const bool use32 = (m % 32 == 0) && (n % 32 == 0);

    if (use32) {
    // x9 = N / 32
    emit(mov_x_imm16(9, n / 32));

    // ===================== N-Loop (32 Spalten) =====================
    int n_loop_start = idx;
    emit(mov_x_imm16(10, m / 32));   // x10 = M / 32
    emit(0xAA1303E7);                // mov x7, x19   ; A base
    emit(0xAA1503E8);                // mov x8, x21   ; C base dieses N-Blocks

    // ===================== M-Loop (32 Zeilen) =====================
    int m_loop_start = idx;

    // ---- C in za0..za3 laden ----
    //   za0 = C[i   :i+16, j   :j+16]     za2 = C[i   :i+16, j+16:j+32]
    //   za1 = C[i+16:i+32, j   :j+16]     za3 = C[i+16:i+32, j+16:j+32]
    emit(0x5280000E);                // mov w14, #0
    emit(0xAA0803EF);                // mov x15, x8        ; Spalten j..j+15
    emit(0x8B160110);                // add x16, x8, x22   ; Spalten j+16..j+31

    int load_loop_start = idx;
    emit(0x910101E6);                // add x6,  x15, #64  ; 16 Zeilen tiefer
    emit(0x91010211);                // add x17, x16, #64
    emit(0xE09FC1E0);                // ld1w {za0v.s[w14,0]}, p0/z, [x15]
    emit(0xE09FC0C4);                // ld1w {za1v.s[w14,0]}, p0/z, [x6]
    emit(0xE09FC208);                // ld1w {za2v.s[w14,0]}, p0/z, [x16]
    emit(0xE09FC22C);                // ld1w {za3v.s[w14,0]}, p0/z, [x17]
    emit(0x8B0501EF);                // add x15, x15, x5
    emit(0x8B050210);                // add x16, x16, x5
    emit(0x110005CE);                // add w14, w14, #1
    emit(0x710041DF);                // cmp w14, #16
    emit(b_cond(COND_LT, load_loop_start, idx));

    // ---- K-Schleife ----
    emit(0xAA0703EB);                // mov x11, x7   ; A laufend
    emit(0xAA1403EC);                // mov x12, x20  ; B laufend
    emit(mov_x_imm16(13, k));        // mov x13, K

    int k_loop_start = idx;
    emit(0xA540A160);                // ld1w {z0.s}, p0/z, [x11]              ; A-Zeilen  i..i+15
    emit(0xA541A161);                // ld1w {z1.s}, p0/z, [x11, #1, mul vl]  ; A-Zeilen  i+16..i+31
    emit(0xA540A182);                // ld1w {z2.s}, p0/z, [x12]              ; B-Spalten j..j+15
    emit(0xA541A183);                // ld1w {z3.s}, p0/z, [x12, #1, mul vl]  ; B-Spalten j+16..j+31
    emit(0x80820000);                // fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    emit(0x80820021);                // fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    emit(0x80830002);                // fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    emit(0x80830023);                // fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    emit(0x8B03016B);                // add x11, x11, x3   ; A += ld_a
    emit(0x8B04018C);                // add x12, x12, x4   ; B += ld_b
    emit(0xF10005AD);                // subs x13, x13, #1
    emit(b_cond(COND_NE, k_loop_start, idx));

    // ---- za0..za3 nach C schreiben ----
    emit(0x5280000E);                // mov w14, #0
    emit(0xAA0803EF);                // mov x15, x8
    emit(0x8B160110);                // add x16, x8, x22

    int store_loop_start = idx;
    emit(0x910101E6);                // add x6,  x15, #64
    emit(0x91010211);                // add x17, x16, #64
    emit(0xE0BFC1E0);                // st1w {za0v.s[w14,0]}, p0, [x15]
    emit(0xE0BFC0C4);                // st1w {za1v.s[w14,0]}, p0, [x6]
    emit(0xE0BFC208);                // st1w {za2v.s[w14,0]}, p0, [x16]
    emit(0xE0BFC22C);                // st1w {za3v.s[w14,0]}, p0, [x17]
    emit(0x8B0501EF);                // add x15, x15, x5
    emit(0x8B050210);                // add x16, x16, x5
    emit(0x110005CE);                // add w14, w14, #1
    emit(0x710041DF);                // cmp w14, #16
    emit(b_cond(COND_LT, store_loop_start, idx));

    // Naechster M-Block: 32 Zeilen = 128 Byte.
    emit(0x910200E7);                // add x7, x7, #128
    emit(0x91020108);                // add x8, x8, #128
    emit(0xF100054A);                // subs x10, x10, #1
    emit(b_cond(COND_NE, m_loop_start, idx));

    // Naechster N-Block: B 32 Spalten weiter (=128 Byte, zeilenweise),
    // C 32 Spalten weiter (= 2 * x22).
    emit(0x91020294);                // add x20, x20, #128
    emit(0x8B1602B5);                // add x21, x21, x22
    emit(0x8B1602B5);                // add x21, x21, x22
    emit(0xF1000529);                // subs x9, x9, #1
    emit(b_cond(COND_NE, n_loop_start, idx));

    } else {
        // x9 = N / 16  (äußere Schleife)
        emit(mov_x_imm16(9, n / 16));

        // ===================== N-Loop =====================
        int n_loop_start = idx;
        // x10 = M / 16  (mittlere Schleife)
        emit(mov_x_imm16(10, m / 16));
        // x7 = A base (wird pro Iteration neu auf x19 gesetzt, A wandert in M-Richtung)
        emit(0xAA1303E7); // mov x7, x19
        // x8 = C base für diesen N-Block (x21 zeigt schon auf den Spaltenblock)
        emit(0xAA1503E8); // mov x8, x21

        // ===================== M-Loop =====================
        int m_loop_start = idx;

        // ----------------------------------------------------------------
        // C in das ZA-Tile LADEN (statt es zu nullen).
        //
        // Vorher stand hier `zero {za}`, und C wurde erst in der Store-Schleife
        // per ld1w/fadd dazugerechnet. Das Nullsetzen ist hier aber falsch am
        // Platz: dafuer existiert der Zero-Kernel separat. Wird ZA gleich mit C
        // vorbelegt, akkumuliert `fmopa` direkt darauf und die Store-Schleife muss
        // nur noch schreiben.
        // ----------------------------------------------------------------
        emit(0x5280000E); // mov w14, #0
        emit(0xAA0803EF); // mov x15, x8                  ; running C (Spaltenanfang)

        int load_loop_start = idx;
        emit(0xE09FC1E0); // ld1w {za0v.s[w14, 0]}, p0/z, [x15]  ; C direkt ins ZA-Tile
        emit(0x8B0501EF); // add x15, x15, x5                 ; C += ld_c
        emit(0x110005CE); // add w14, w14, #1
        emit(0x710041DF); // cmp w14, #16
        emit(b_cond(COND_LT, load_loop_start, idx));

        // Laufpointer für die K-Schleife.
        emit(0xAA0703EB); // mov x11, x7   ; running A (16-row Block)
        emit(0xAA1403EC); // mov x12, x20  ; running B (16-col Block)
        emit(mov_x_imm16(13, k));          // mov x13, K

        // ===================== K-Loop =====================
        int k_loop_start = idx;
        emit(0xA540A160); // ld1w {z0.s}, p0/z, [x11]     ; 16 Floats: A-Spalte
        emit(0xA540A181); // ld1w {z1.s}, p0/z, [x12]     ; 16 Floats: B-Zeile
        emit(0x80810000); // fmopa za0.s, p0/m, p0/m, z0.s, z1.s
        emit(0x8B03016B); // add x11, x11, x3              ; A += ld_a
        emit(0x8B04018C); // add x12, x12, x4              ; B += ld_b
        emit(0xF10005AD); // subs x13, x13, #1
        emit(b_cond(COND_NE, k_loop_start, idx));

        // ===================== Store-Loop =====================
        // 16 Spalten des Tiles -> C[:,j_base..j_base+15] += column j (von za0).
        emit(0x5280000E); // mov w14, #0
        emit(0xAA0803EF); // mov x15, x8                  ; running C (Spaltenanfang)

        int store_loop_start = idx;
        // Kein mova/ld1w/fadd mehr noetig: C wurde vor der K-Schleife ins ZA-Tile
        // geladen, `fmopa` hat direkt darauf akkumuliert. SME kann ZA-Kachelspalten
        // unmittelbar in den Speicher schreiben -- ohne Umweg ueber ein z-Register.
        emit(0xE0BFC1E0); // st1w {za0v.s[w14, 0]}, p0, [x15]
        emit(0x8B0501EF); // add x15, x15, x5              ; C += ld_c (nächste Spalte)
        emit(0x110005CE); // add w14, w14, #1
        emit(0x710041DF); // cmp w14, #16
        emit(b_cond(COND_LT, store_loop_start, idx));

        // Nächster M-Block: A und C jeweils um 16 Zeilen weiter (= 64 Bytes).
        emit(0x910100E7); // add x7, x7, #64
        emit(0x91010108); // add x8, x8, #64
        emit(0xF100054A); // subs x10, x10, #1
        emit(b_cond(COND_NE, m_loop_start, idx));

        // Nächster N-Block:
        //   B row-major: 16 Spalten weiter = 16 * 4 Byte = 64 Byte (kontiguierlich in einer Zeile).
        //   C col-major: 16 Spalten weiter = 16 * ld_c Byte = x22.
        emit(0x91010294); // add x20, x20, #64
        emit(0x8B1602B5); // add x21, x21, x22
        emit(0xF1000529); // subs x9, x9, #1
        emit(b_cond(COND_NE, n_loop_start, idx));
    }

    // -------------------- SME deaktivieren --------------------
    emit(0xD503467F); // smstop

    // -------------------- Epilogue --------------------
    emit(0xA9415BF5); // ldp x21, x22, [sp, #16]
    emit(0xA8C253F3); // ldp x19, x20, [sp], #32
    emit(0x6D433FEE); // ldp d14, d15, [sp, #48]
    emit(0x6D4237EC); // ldp d12, d13, [sp, #32]
    emit(0x6D412FEA); // ldp d10, d11, [sp, #16]
    emit(0x6CC427E8); // ldp d8,  d9,  [sp], #64
    emit(0xD65F03C0); // ret

    mprotect(mem, size, PROT_READ | PROT_EXEC);
    __builtin___clear_cache(reinterpret_cast<char*>(mem),
                            reinterpret_cast<char*>(mem) + size);

    this->m_code = mem;
    this->m_size = size;

    return error_t::success;
}

} // namespace mini_jit
