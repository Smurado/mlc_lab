// SSVE (Streaming SVE) Unary Kernels
// - zero_16_16
// - relu_16_16

// void zero_16_16(float const * a, int64_t ld_a)
// x0 = a (Pointer)
// x1 = ld_a (Leading Dimension)

.text
.align 4
.global _zero_16_16

_zero_16_16:
    // 1. Einstieg in den Streaming SVE Mode
    smstart sm

    // 2. Predicate für 16 Floats (512-bit Vektor)
    ptrue   p0.s, vl16

    // 3. ld_a in Bytes
    lsl     x1, x1, #2

    // 4. Z0 = 0
    mov     z0.s, #0

    // 5. 16 Spalten schreiben
    mov     x2, #16
.zero_loop:
    st1w    {z0.s}, p0, [x0]
    add     x0, x0, x1
    subs    x2, x2, #1
    b.ne    .zero_loop

    smstop sm
    ret


// void relu_16_16(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b)
// x0 = a, x1 = b, x2 = ld_a, x3 = ld_b, w4 = trans_b

.text
.align 4
.global _relu_16_16

_relu_16_16:
    smstart sm

    ptrue   p0.s, vl16
    fmov    z31.s, #0.0

    lsl     x2, x2, #2      // ld_a in Bytes
    lsl     x3, x3, #2      // ld_b in Bytes

    cmp     w4, #1
    b.eq    .relu_transpose

// --- PFAD 1: Column-Major -> Column-Major ---
.relu_linear:
    mov     x9, #16
.relu_loop_linear:
    ld1w    {z0.s}, p0/z, [x0]
    fmax    z0.s, p0/m, z0.s, z31.s
    st1w    {z0.s}, p0, [x1]
    add     x0, x0, x2
    add     x1, x1, x3
    subs    x9, x9, #1
    b.ne    .relu_loop_linear
    b       .relu_exit

// --- PFAD 2: Column-Major -> Row-Major (Scatter Store) ---
.relu_transpose:
    index   z1.s, #0, w3    // z1 = [0, 1*ld_b, 2*ld_b, ...] (Bytes)

    mov     x9, #16
.relu_loop_transpose:
    ld1w    {z0.s}, p0/z, [x0]
    fmax    z0.s, p0/m, z0.s, z31.s
    st1w    {z0.s}, p0, [x1, z1.s, uxtw]
    add     x0, x0, x2
    add     x1, x1, #4
    subs    x9, x9, #1
    b.ne    .relu_loop_transpose

.relu_exit:
    smstop sm
    ret
