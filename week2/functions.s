    .global _micro_benchmark
_micro_benchmark:

    

    // Stack Frame speichern
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

.loop_start:
    cbz x0, .loop_end

    .rep 50 // N = 50 * 2 * 28

        fmadd s0, s29, s30, s31
        fmadd s1, s29, s30, s31;
        fmadd s2, s29, s30, s31;
        fmadd s3, s29, s30, s31;

        fmadd s4, s29, s30, s31;
        fmadd s5, s29, s30, s31;
        fmadd s6, s29, s30, s31;
        fmadd s7, s29, s30, s31;

        fmadd s8, s29, s30, s31;
        fmadd s9, s29, s30, s31;
        fmadd s10, s29, s30, s31;
        fmadd s11, s29, s30, s31;

        fmadd s12, s29, s30, s31;
        fmadd s13, s29, s30, s31;
        fmadd s14, s29, s30, s31;
        fmadd s15, s29, s30, s31;

        fmadd s16, s29, s30, s31;
        fmadd s17, s29, s30, s31;
        fmadd s18, s29, s30, s31;
        fmadd s19, s29, s30, s31;

        fmadd s20, s29, s30, s31;
        fmadd s21, s29, s30, s31;
        fmadd s22, s29, s30, s31;
        fmadd s23, s29, s30, s31;

        fmadd s24, s29, s30, s31;
        fmadd s25, s29, s30, s31;
        fmadd s26, s29, s30, s31;
        fmadd s27, s29, s30, s31;
    
    
    .endr

    sub x0, x0, #1
    b .loop_start

.loop_end:
    // Stack Frame wiederherstellen
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16

    ret


.global _perm_neon_abc_cba
_perm_neon_abc_cba:
    // x0 = size_c
    // x1 = abc pointer
    // x2 = cba pointer

    mov x3, x0          // size_c
    mov x4, x1          // abc base
    mov x5, x2          // cba base

    mov x6, #0          // a = 0

// ---------------- a loop ----------------
a_loop:
    cmp x6, #8
    bge done

    mov x7, #0          // b = 0

// ---------------- b loop ----------------
b_loop:
    cmp x7, #4
    bge next_a

    mov x8, #0          // c = 0

// ---------------- c loop ----------------
c_loop:
    cmp x8, x3
    bge next_b

    // =====================================================
    // LOAD: abc[a][b][c]
    // index = ((a * 4) + b) * size_c + c
    // =====================================================

    lsl x9, x6, #2              // a * 4
    add x9, x9, x7              // + b
    mul x9, x9, x3              // * size_c
    add x9, x9, x8              // + c

    add x9, x4, x9, lsl #2      // abc_base + index * 4 (bytes base)
    ldr s0, [x9]

    // =====================================================
    // STORE: cba[c][b][a]
    // index = ((c * 4) + b) * 8 + a
    // =====================================================

    lsl x10, x8, #5             // c * (B * A = 32)
    add x10, x10, x7, lsl #3    // + b * 8
    add x10, x10, x6            // + a

    add x10, x5, x10, lsl #2    // cba_base + index * 4 (bytes base)
    str s0, [x10]

    // c++
    add x8, x8, #1
    b c_loop

// ---------------- next b ----------------
next_b:
    // b++
    add x7, x7, #1
    b b_loop

// ---------------- next a ----------------
next_a:
    // a++
    add x6, x6, #1
    b a_loop

done:
    ret