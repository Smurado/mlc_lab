// SME GEMM Kernels
//
// Konventionen (laut README):
//   A: column-major, M x K
//   B: row-major,    K x N
//   C: column-major, M x N
//   ld_a, ld_b, ld_c sind in Elementen (Floats).
//
// Hardware-Annahme: SVL = 512 bit => 16 Floats pro Z-Register.
// Der 32x32-Akkumulator besteht aus 4 ZA-Tiles (za0..za3), je 16x16:
//
//        +----------------+----------------+
//        | za0 (rows 0-15,| za2 (rows 0-15,|
//        |  cols 0-15)    |  cols 16-31)   |
//        +----------------+----------------+
//        | za1 (rows16-31,| za3 (rows16-31,|
//        |  cols 0-15)    |  cols 16-31)   |
//        +----------------+----------------+


//==============================================================================
// gemm_32_32_1 : C(32x32) += A(32x1) * B(1x32)
//==============================================================================
.text
.align 4
.global _gemm_32_32_1

_gemm_32_32_1:
    smstart
    ptrue   p0.s

    ld1w    {z0.s}, p0/z, [x0]
    add     x6, x0, #64
    ld1w    {z1.s}, p0/z, [x6]

    ld1w    {z2.s}, p0/z, [x1]
    add     x6, x1, #64
    ld1w    {z3.s}, p0/z, [x6]

    zero    {za}

    fmopa   za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa   za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa   za3.s, p0/m, p0/m, z1.s, z3.s

    lsl     x5, x5, #2
    lsl     x6, x5, #4          // 16 * ld_c (Bytes)
    mov     w12, #0
.gemm32_store:
    mova    z4.s, p0/m, za0v.s[w12, 0]
    mova    z5.s, p0/m, za1v.s[w12, 0]
    add     x10, x2, #64
    ld1w    {z6.s}, p0/z, [x2]
    ld1w    {z7.s}, p0/z, [x10]
    fadd    z6.s, z6.s, z4.s
    fadd    z7.s, z7.s, z5.s
    st1w    {z6.s}, p0, [x2]
    st1w    {z7.s}, p0, [x10]

    add     x11, x2, x6
    add     x13, x11, #64
    mova    z4.s, p0/m, za2v.s[w12, 0]
    mova    z5.s, p0/m, za3v.s[w12, 0]
    ld1w    {z6.s}, p0/z, [x11]
    ld1w    {z7.s}, p0/z, [x13]
    fadd    z6.s, z6.s, z4.s
    fadd    z7.s, z7.s, z5.s
    st1w    {z6.s}, p0, [x11]
    st1w    {z7.s}, p0, [x13]

    add     x2, x2, x5
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    .gemm32_store

    smstop
    ret


//==============================================================================
// gemm_32_32_512 : C(32x32) += A(32x512) * B(512x32)   (K-Loop)
//==============================================================================
.text
.align 4
.global _gemm_32_32_512

_gemm_32_32_512:
    smstart
    ptrue   p0.s

    zero    {za}

    lsl     x3, x3, #2          // ld_a in Bytes
    lsl     x4, x4, #2          // ld_b in Bytes

    mov     x7, x0              // running A
    mov     x8, x1              // running B
    mov     w9, #512            // K
.gemm_32_32_512_kloop:
    ld1w    {z0.s}, p0/z, [x7]
    add     x10, x7, #64
    ld1w    {z1.s}, p0/z, [x10]

    ld1w    {z2.s}, p0/z, [x8]
    add     x10, x8, #64
    ld1w    {z3.s}, p0/z, [x10]

    fmopa   za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa   za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa   za3.s, p0/m, p0/m, z1.s, z3.s

    add     x7, x7, x3
    add     x8, x8, x4
    subs    w9, w9, #1
    b.ne    .gemm_32_32_512_kloop

    lsl     x5, x5, #2
    lsl     x6, x5, #4
    mov     w12, #0
.gemm_32_32_512_store:
    mova    z4.s, p0/m, za0v.s[w12, 0]
    mova    z5.s, p0/m, za1v.s[w12, 0]
    add     x10, x2, #64
    ld1w    {z6.s}, p0/z, [x2]
    ld1w    {z7.s}, p0/z, [x10]
    fadd    z6.s, z6.s, z4.s
    fadd    z7.s, z7.s, z5.s
    st1w    {z6.s}, p0, [x2]
    st1w    {z7.s}, p0, [x10]

    add     x11, x2, x6
    add     x13, x11, #64
    mova    z4.s, p0/m, za2v.s[w12, 0]
    mova    z5.s, p0/m, za3v.s[w12, 0]
    ld1w    {z6.s}, p0/z, [x11]
    ld1w    {z7.s}, p0/z, [x13]
    fadd    z6.s, z6.s, z4.s
    fadd    z7.s, z7.s, z5.s
    st1w    {z6.s}, p0, [x11]
    st1w    {z7.s}, p0, [x13]

    add     x2, x2, x5
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    .gemm_32_32_512_store

    smstop
    ret


//==============================================================================
// gemm_512_32_512 : C(512x32) += A(512x512) * B(512x32)   (M-Loop)
//
// 16 Blöcke à 32 Zeilen. Ruft gemm_32_32_512 pro Block auf.
//==============================================================================
.text
.align 4
.global _gemm_512_32_512

_gemm_512_32_512:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]

    mov     x19, x0     // a base
    mov     x20, x1     // b base
    mov     x21, x2     // c base
    mov     x22, x3     // ld_a
    mov     x23, x4     // ld_b
    mov     x24, x5     // ld_c

    mov     x25, #0     // m_start (Elemente)
    mov     w26, #16
.gemm_512_32_512_mloop:
    add     x0, x19, x25, lsl #2
    mov     x1, x20
    add     x2, x21, x25, lsl #2
    mov     x3, x22
    mov     x4, x23
    mov     x5, x24
    bl      _gemm_32_32_512

    add     x25, x25, #32
    subs    w26, w26, #1
    b.ne    .gemm_512_32_512_mloop

    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret


//==============================================================================
// gemm_512_512_512 : C(512x512) += A(512x512) * B(512x512)   (N-Loop)
//
// 16 Blöcke à 32 Spalten. Ruft gemm_512_32_512 pro Block auf.
//==============================================================================
.text
.align 4
.global _gemm_512_512_512

_gemm_512_512_512:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    str     x27,      [sp, #80]

    smstart

    mov     x19, x0     // a base
    mov     x20, x1     // b base
    mov     x21, x2     // c base
    mov     x22, x3     // ld_a
    mov     x23, x4     // ld_b
    mov     x24, x5     // ld_c

    mov     x25, #0     // n_start (Elemente)
    mov     w26, #16
.gemm_512_512_512_nloop:
    mov     x0, x19
    // B row-major: Spalten-Offset n_start Floats
    add     x1, x20, x25, lsl #2
    // C column-major: Offset n_start * ld_c Floats
    mul     x27, x25, x24
    add     x2, x21, x27, lsl #2
    mov     x3, x22
    mov     x4, x23
    mov     x5, x24
    bl      _gemm_512_32_512_fast

    add     x25, x25, #32
    subs    w26, w26, #1
    b.ne    .gemm_512_512_512_nloop

    smstop

    ldr     x27,      [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

//==============================================================================
// FAST VARIANTS W/O SMSTART/SMSTOP for nested loops
//==============================================================================
.text
.align 4

_gemm_32_32_512_fast:
    ptrue   p0.s
    zero    {za}

    lsl     x3, x3, #2
    lsl     x4, x4, #2

    mov     x7, x0
    mov     x8, x1
    mov     w9, #512
.gemm_32_32_512_kloop_fast:
    ld1w    {z0.s}, p0/z, [x7]
    add     x10, x7, #64
    ld1w    {z1.s}, p0/z, [x10]

    ld1w    {z2.s}, p0/z, [x8]
    add     x10, x8, #64
    ld1w    {z3.s}, p0/z, [x10]

    fmopa   za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa   za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa   za3.s, p0/m, p0/m, z1.s, z3.s

    add     x7, x7, x3
    add     x8, x8, x4
    subs    w9, w9, #1
    b.ne    .gemm_32_32_512_kloop_fast

    lsl     x5, x5, #2
    lsl     x6, x5, #4
    mov     w12, #0
.gemm_32_32_512_store_fast:
    mova    z4.s, p0/m, za0v.s[w12, 0]
    mova    z5.s, p0/m, za1v.s[w12, 0]
    add     x10, x2, #64
    ld1w    {z6.s}, p0/z, [x2]
    ld1w    {z7.s}, p0/z, [x10]
    fadd    z6.s, z6.s, z4.s
    fadd    z7.s, z7.s, z5.s
    st1w    {z6.s}, p0, [x2]
    st1w    {z7.s}, p0, [x10]

    add     x11, x2, x6
    add     x13, x11, #64
    mova    z4.s, p0/m, za2v.s[w12, 0]
    mova    z5.s, p0/m, za3v.s[w12, 0]
    ld1w    {z6.s}, p0/z, [x11]
    ld1w    {z7.s}, p0/z, [x13]
    fadd    z6.s, z6.s, z4.s
    fadd    z7.s, z7.s, z5.s
    st1w    {z6.s}, p0, [x11]
    st1w    {z7.s}, p0, [x13]

    add     x2, x2, x5
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    .gemm_32_32_512_store_fast

    ret


_gemm_512_32_512_fast:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]

    mov     x19, x0     // a base
    mov     x20, x1     // b base
    mov     x21, x2     // c base
    mov     x22, x3     // ld_a
    mov     x23, x4     // ld_b
    mov     x24, x5     // ld_c

    mov     x25, #0     // m_start (Elemente)
    mov     w26, #16
.gemm_512_32_512_mloop_fast:
    add     x0, x19, x25, lsl #2
    mov     x1, x20
    add     x2, x21, x25, lsl #2
    mov     x3, x22
    mov     x4, x23
    mov     x5, x24
    bl      _gemm_32_32_512_fast

    add     x25, x25, #32
    subs    w26, w26, #1
    b.ne    .gemm_512_32_512_mloop_fast

    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret

