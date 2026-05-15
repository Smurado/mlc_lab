// SSVE (Streaming SVE) Unary Kernels
// - zero_16_16
// - relu_16_16

// void identity_16_16(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b)
// x0 = a (Source)
// x1 = b (Destination)
// x2 = ld_a (Leading Dimension A)
// x3 = ld_b (Leading Dimension B)
// w4 = trans_b (0 = Kopieren, 1 = Transponieren)

.text
.align 4
.global _identity_16_16
_identity_16_16:
    // We use 'za' here because permutation is best done via the matrix tile
    smstart              
    ptrue   p0.s, vl16
    lsl     x2, x2, #2
    lsl     x3, x3, #2

    cmp     w4, #1
    b.eq    .id_permute

.id_copy:
    mov     x9, #16
.id_copy_loop:
    ld1w    {z0.s}, p0/z, [x0]
    st1w    {z0.s}, p0, [x1]
    add     x0, x0, x2
    add     x1, x1, x3
    subs    x9, x9, #1
    b.ne    .id_copy_loop
    b       .id_end

.id_permute:
    // 1. Lade 16 Spalten von A horizontal in das ZA0-Tile
    mov     w12, #0
.id_load_tile:
    // Syntax: [Basis-Register, Immediate-Offset]
    ld1w    {za0h.s[w12, 0]}, p0/z, [x0]
    add     x0, x0, x2
    add     w12, w12, #1
    cmp     w12, #16
    b.ne    .id_load_tile

    // 2. Speichere 16 Zeilen von ZA0 vertikal nach B
    mov     w12, #0
.id_store_tile:
    // Auch hier: , 0 hinzufügen
    st1w    {za0v.s[w12, 0]}, p0, [x1]
    add     x1, x1, x3
    add     w12, w12, #1
    cmp     w12, #16
    b.ne    .id_store_tile

.id_end:
    smstop
    ret

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
