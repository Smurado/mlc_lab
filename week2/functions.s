    .global _micro_benchmark
_micro_benchmark:
    // Stack Frame speichern
    stp x8, x9, [sp, #-16]!
    stp x10, x11, [sp, #-16]!
    stp x12, x13, [sp, #-16]!
    stp x14, x15, [sp, #-16]!
    
    .rep 50 // N = 50 * 2 * 28

        fmadd x0, x29, x30, x31;
        fmadd x1, x29, x30, x31;
        fmadd x2, x29, x30, x31;
        fmadd x3, x29, x30, x31;

        fmadd x4, x29, x30, x31;
        fmadd x5, x29, x30, x31;
        fmadd x6, x29, x30, x31;
        fmadd x7, x29, x30, x31;

        fmadd x8, x29, x30, x31;
        fmadd x9, x29, x30, x31;
        fmadd x10, x29, x30, x31;
        fmadd x11, x29, x30, x31;

        fmadd x12, x29, x30, x31;
        fmadd x13, x29, x30, x31;
        fmadd x14, x29, x30, x31;
        fmadd x15, x29, x30, x31;

        fmadd x16, x29, x30, x31;
        fmadd x17, x29, x30, x31;
        fmadd x18, x29, x30, x31;
        fmadd x19, x29, x30, x31;

        fmadd x20, x29, x30, x31;
        fmadd x21, x29, x30, x31;
        fmadd x22, x29, x30, x31;
        fmadd x23, x29, x30, x31;

        fmadd x24, x29, x30, x31;
        fmadd x25, x29, x30, x31;
        fmadd x26, x29, x30, x31;
        fmadd x27, x29, x30, x31;
    
    .endr
    
    // Stack Frame speichern
    ldp x8, x9, [sp, #-16]!
    ldp x10, x11, [sp, #-16]!
    ldp x12, x13, [sp, #-16]!
    ldp x14, x15, [sp, #-16]!

    ret


    .global _perm_neon_abc_cba
_perm_neon_abc_cba:
    // Parameter:
    // x0 = size_c (int64_t)
    // x1 = abc (float const *)
    // x2 = cba (float *)
    
    // Stack Frame speichern (x29 = FP, x30 = LR)
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    
    // Callee-saved Register speichern (x19-x28 bei Bedarf)
    stp x19, x20, [sp, #16]
    
    // Funktion hier implementieren
    
    // Callee-saved Register restoren
    ldp x19, x20, [sp, #16]
    
    // Stack Frame restoren
    ldp x29, x30, [sp], #32
    ret