// void zero_16_16(float const * a, int64_t ld_a)
// x0 = a (Pointer)
// x1 = ld_a (Leading Dimension)

.text
.align 4
.global _zero_16_16

_zero_16_16:
    // 1. Einstieg in den Streaming SVE Mode
    // 'sm' aktiviert sowohl Streaming Mode als auch die ZA Storage (falls nötig)
    smstart sm

    // 2. Erzeuge ein Zero-Prädikat
    // P0 wird so gesetzt, dass es alle Elemente anspricht
    ptrue   p0.s

    // 3. Berechne Byte-Offset für ld_a (ld_a * 4 Bytes)
    lsl     x1, x1, #2

    // 4. Initialisiere Z0 mit Nullen
    mov     z0.s, #0

    // 5. Spaltenweises Schreiben (16 Spalten)
    // Da ein SVE-Register auf dem M4 (256-bit / 32-byte) bei 16x16 Matrix 
    // oft anders skaliert, nutzen wir hier die feste Matrix-Logik:
    
    // Wir schreiben 16 Spalten. Jede Spalte hat 16 Floats = 64 Bytes.
    // Ein Z-Register im M4 SVE ist 512-bit (64 Bytes) oder 256-bit (32 Bytes).
    // Der M4 nutzt typischerweise 512-bit im SSVE Mode.

    mov     x2, #16         // Counter für 16 Spalten
    
.loop:
    st1w    {z0.s}, p0, [x0] // Speichere 512-bit (16 Floats) Null
    add     x0, x0, x1       // Gehe zur nächsten Spalte (A + ld_a)
    subs    x2, x2, #1       // Dekrementiere Counter
    b.ne    .loop

    // 6. Streaming Mode verlassen
    smstop sm
    ret






// void relu_16_16(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b)

.text
.align 4
.global _relu_16_16

_relu_16_16:
    smstart sm              // Streaming Mode ein

    ptrue   p0.s, vl16      // 16 Floats (512-bit)
    fmov    z31.s, #0.0     // Referenz für ReLU
    
    lsl     x2, x2, #2      // ld_a in Bytes
    lsl     x3, x3, #2      // ld_b in Bytes

    cmp     w4, #1
    b.eq    .transpose_sve

// --- PFAD 1: Column-Major -> Column-Major ---
.linear_path:
    mov     x9, #16
.loop_linear:
    ld1w    {z0.s}, p0/z, [x0]
    fmax    z0.s, p0/m, z0.s, z31.s
    st1w    {z0.s}, p0, [x1]
    add     x0, x0, x2
    add     x1, x1, x3
    subs    x9, x9, #1
    b.ne    .loop_linear
    b       .exit

// --- PFAD 2: Column-Major -> Row-Major (Scatter Store) ---
.transpose_sve:
    // Wir erstellen einen Offset-Vektor in Z1: {0, 1*ld_b, 2*ld_b, ..., 15*ld_b}
    // Damit schreiben wir die Spalte von A direkt in die "Zeile" (verteilt) von B.
    index   z1.s, #0, w3    // z1 = [0*x3, 1*x3, 2*x3, ...] (Offsets in Bytes)
    
    mov     x9, #16         // 16 Spalten
.loop_transpose:
    ld1w    {z0.s}, p0/z, [x0]
    fmax    z0.s, p0/m, z0.s, z31.s
    
    // Scatter Store: Speichere Elemente von Z0 an [x1 + Offsets in Z1]
    st1w    {z0.s}, p0, [x1, z1.s, uxtw]
    
    add     x0, x0, x2      // Nächste Spalte in A
    add     x1, x1, #4      // Nächstes Element in der Zeile von B
    subs    x9, x9, #1
    b.ne    .loop_transpose

.exit:
    smstop sm               // Streaming Mode aus
    ret