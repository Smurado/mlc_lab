.global _micro_benchmark, _micro_benchmark_fmla_4s, _micro_benchmark_fmla_2s, _perm_neon_abc_cba


// Microbenchmark: FMADD (scalar, FP32)
// Pro Iteration: 50 (Unroll) * 28 (unabhaengige Zielregister) FMADDs
_micro_benchmark:
    // Callee-saved FP-Register sichern (d8..d15)
    stp d8,  d9,  [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

fmadd_loop_start:
    cbz x0, fmadd_loop_end      // iterations == 0 -> fertig

    .rep 50                     // Unroll-Faktor
        fmadd s0,  s29, s30, s31
        fmadd s1,  s29, s30, s31
        fmadd s2,  s29, s30, s31
        fmadd s3,  s29, s30, s31
        fmadd s4,  s29, s30, s31
        fmadd s5,  s29, s30, s31
        fmadd s6,  s29, s30, s31
        fmadd s7,  s29, s30, s31
        fmadd s8,  s29, s30, s31
        fmadd s9,  s29, s30, s31
        fmadd s10, s29, s30, s31
        fmadd s11, s29, s30, s31
        fmadd s12, s29, s30, s31
        fmadd s13, s29, s30, s31
        fmadd s14, s29, s30, s31
        fmadd s15, s29, s30, s31
        fmadd s16, s29, s30, s31
        fmadd s17, s29, s30, s31
        fmadd s18, s29, s30, s31
        fmadd s19, s29, s30, s31
        fmadd s20, s29, s30, s31
        fmadd s21, s29, s30, s31
        fmadd s22, s29, s30, s31
        fmadd s23, s29, s30, s31
        fmadd s24, s29, s30, s31
        fmadd s25, s29, s30, s31
        fmadd s26, s29, s30, s31
        fmadd s27, s29, s30, s31
    .endr

    sub x0, x0, #1              // iterations--
    b fmadd_loop_start

fmadd_loop_end:
    // FP-Register wiederherstellen
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8,  d9,  [sp], #16
    ret


// Microbenchmark: FMLA (vector, Arrangement 4S)
// Gleiche Struktur wie fmadd, aber Vektor-FMLA auf 4 Lanes.
_micro_benchmark_fmla_4s:
    stp d8,  d9,  [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

fmla4s_loop_start:
    cbz x0, fmla4s_loop_end

    .rep 50
        fmla v0.4s,  v29.4s, v30.4s
        fmla v1.4s,  v29.4s, v30.4s
        fmla v2.4s,  v29.4s, v30.4s
        fmla v3.4s,  v29.4s, v30.4s
        fmla v4.4s,  v29.4s, v30.4s
        fmla v5.4s,  v29.4s, v30.4s
        fmla v6.4s,  v29.4s, v30.4s
        fmla v7.4s,  v29.4s, v30.4s
        fmla v8.4s,  v29.4s, v30.4s
        fmla v9.4s,  v29.4s, v30.4s
        fmla v10.4s, v29.4s, v30.4s
        fmla v11.4s, v29.4s, v30.4s
        fmla v12.4s, v29.4s, v30.4s
        fmla v13.4s, v29.4s, v30.4s
        fmla v14.4s, v29.4s, v30.4s
        fmla v15.4s, v29.4s, v30.4s
        fmla v16.4s, v29.4s, v30.4s
        fmla v17.4s, v29.4s, v30.4s
        fmla v18.4s, v29.4s, v30.4s
        fmla v19.4s, v29.4s, v30.4s
        fmla v20.4s, v29.4s, v30.4s
        fmla v21.4s, v29.4s, v30.4s
        fmla v22.4s, v29.4s, v30.4s
        fmla v23.4s, v29.4s, v30.4s
        fmla v24.4s, v29.4s, v30.4s
        fmla v25.4s, v29.4s, v30.4s
        fmla v26.4s, v29.4s, v30.4s
        fmla v27.4s, v29.4s, v30.4s
    .endr

    sub x0, x0, #1
    b fmla4s_loop_start

fmla4s_loop_end:
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8,  d9,  [sp], #16
    ret


// Microbenchmark: FMLA (vector, Arrangement 2S)
_micro_benchmark_fmla_2s:
    stp d8,  d9,  [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

fmla2s_loop_start:
    cbz x0, fmla2s_loop_end

    .rep 50
        fmla v0.2s,  v29.2s, v30.2s
        fmla v1.2s,  v29.2s, v30.2s
        fmla v2.2s,  v29.2s, v30.2s
        fmla v3.2s,  v29.2s, v30.2s
        fmla v4.2s,  v29.2s, v30.2s
        fmla v5.2s,  v29.2s, v30.2s
        fmla v6.2s,  v29.2s, v30.2s
        fmla v7.2s,  v29.2s, v30.2s
        fmla v8.2s,  v29.2s, v30.2s
        fmla v9.2s,  v29.2s, v30.2s
        fmla v10.2s, v29.2s, v30.2s
        fmla v11.2s, v29.2s, v30.2s
        fmla v12.2s, v29.2s, v30.2s
        fmla v13.2s, v29.2s, v30.2s
        fmla v14.2s, v29.2s, v30.2s
        fmla v15.2s, v29.2s, v30.2s
        fmla v16.2s, v29.2s, v30.2s
        fmla v17.2s, v29.2s, v30.2s
        fmla v18.2s, v29.2s, v30.2s
        fmla v19.2s, v29.2s, v30.2s
        fmla v20.2s, v29.2s, v30.2s
        fmla v21.2s, v29.2s, v30.2s
        fmla v22.2s, v29.2s, v30.2s
        fmla v23.2s, v29.2s, v30.2s
        fmla v24.2s, v29.2s, v30.2s
        fmla v25.2s, v29.2s, v30.2s
        fmla v26.2s, v29.2s, v30.2s
        fmla v27.2s, v29.2s, v30.2s
    .endr

    sub x0, x0, #1
    b fmla2s_loop_start

fmla2s_loop_end:
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8,  d9,  [sp], #16
    ret


// Permutation abc -> cba  (A=8, B=4, C=size_c, row-major)
//   abc-Index: a*B*C + b*C + c
//   cba-Index: c*B*A + b*A + a
//
// Pro b laufen wir ueber c in 4er-Bloecken. Wir laden fuer jede der
// 8 a-Zeilen einen 4-Float-Vektor (4 aufeinanderfolgende c-Werte) und
// transponieren zwei 4x4-Tiles. Dadurch enthaelt jedes Ziel-Register
// 4 a-Werte fuer ein festes c, sodass wir pro c alle 8 a-Werte mit
// einem STP q,q schreiben koennen. Rest (|c| % 4) wird skalar kopiert.
_perm_neon_abc_cba:
    // x0 = size_c (C), x1 = abc, x2 = cba
    cbz x0, perm_end            // C == 0 -> nichts zu tun

    lsl x3, x0, #2              // C * 4    (Stride zw. b-Slabs in abc)
    lsl x4, x0, #4              // B * C * 4 (Stride zw. a-Zeilen in abc)

    mov x5, #0                  // b = 0

perm_b_loop:
    cmp x5, #4
    b.ge perm_end

    // Basiszeiger fuer aktuelles b
    madd x6, x5, x3, x1         // abc_b = abc + b * C * 4
    add  x7, x2, x5, lsl #5     // cba_b = cba + b * A * 4 (A=8 -> <<5)

    mov x8, #0                  // c = 0

perm_c_block:
    // Solange mind. 4 c-Werte uebrig: vektorisierter Pfad
    sub x9, x0, x8
    cmp x9, #4
    b.lt perm_c_tail

    // 8 Vektoren laden: je 4 c-Werte fuer a = 0..7
    add x10, x6, x8, lsl #2     // src = abc_b + c * 4
    mov x11, x10

    ld1 {v0.4s}, [x11], x4      // a=0
    ld1 {v1.4s}, [x11], x4      // a=1
    ld1 {v2.4s}, [x11], x4      // a=2
    ld1 {v3.4s}, [x11], x4      // a=3
    ld1 {v4.4s}, [x11], x4      // a=4
    ld1 {v5.4s}, [x11], x4      // a=5
    ld1 {v6.4s}, [x11], x4      // a=6
    ld1 {v7.4s}, [x11]          // a=7

    // 4x4-Transpose von v0..v3 -> v20..v23 (Zeilen: c0..c3, a=0..3)
    trn1 v16.4s, v0.4s, v1.4s
    trn2 v17.4s, v0.4s, v1.4s
    trn1 v18.4s, v2.4s, v3.4s
    trn2 v19.4s, v2.4s, v3.4s
    zip1 v20.2d, v16.2d, v18.2d
    zip1 v21.2d, v17.2d, v19.2d
    zip2 v22.2d, v16.2d, v18.2d
    zip2 v23.2d, v17.2d, v19.2d

    // 4x4-Transpose von v4..v7 -> v28..v31 (Zeilen: c0..c3, a=4..7)
    trn1 v24.4s, v4.4s, v5.4s
    trn2 v25.4s, v4.4s, v5.4s
    trn1 v26.4s, v6.4s, v7.4s
    trn2 v27.4s, v6.4s, v7.4s
    zip1 v28.2d, v24.2d, v26.2d
    zip1 v29.2d, v25.2d, v27.2d
    zip2 v30.2d, v24.2d, v26.2d
    zip2 v31.2d, v25.2d, v27.2d

    // Stores: pro c-Schritt 8 floats (a0..a7). Stride zw. c-Werten
    // in cba ist B*A*4 = 128 Bytes.
    add x12, x7, x8, lsl #7     // dst = cba_b + c * 128
    stp q20, q28, [x12]         // c = x8 + 0
    stp q21, q29, [x12, #128]   // c = x8 + 1
    stp q22, q30, [x12, #256]   // c = x8 + 2
    stp q23, q31, [x12, #384]   // c = x8 + 3

    add x8, x8, #4
    b perm_c_block

perm_c_tail:
    // Skalarer Rest: einzelne c-Werte kopieren
    cmp x8, x0
    b.ge perm_next_b

    add x11, x6, x8, lsl #2     // src = abc_b + c * 4
    add x12, x7, x8, lsl #7     // dst = cba_b + c * 128

    ldr s0, [x11]
    add x11, x11, x4
    ldr s1, [x11]
    add x11, x11, x4
    ldr s2, [x11]
    add x11, x11, x4
    ldr s3, [x11]
    add x11, x11, x4
    ldr s4, [x11]
    add x11, x11, x4
    ldr s5, [x11]
    add x11, x11, x4
    ldr s6, [x11]
    add x11, x11, x4
    ldr s7, [x11]

    str s0, [x12]
    str s1, [x12, #4]
    str s2, [x12, #8]
    str s3, [x12, #12]
    str s4, [x12, #16]
    str s5, [x12, #20]
    str s6, [x12, #24]
    str s7, [x12, #28]

    add x8, x8, #1
    b perm_c_tail

perm_next_b:
    add x5, x5, #1
    b perm_b_loop

perm_end:
    ret
