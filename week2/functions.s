    .global _micro_benchmark
_micro_benchmark:

    

    // Stack Frame speichern
    stp s8, s9, [sp, #-16]!
    stp s10, s11, [sp, #-16]!
    stp s12, s13, [sp, #-16]!
    stp s14, s15, [sp, #-16]!
    
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
    
    // Stack Frame speichern
    ldp s8, s9, [sp, #-16]!
    ldp s10, s11, [sp, #-16]!
    ldp s12, s13, [sp, #-16]!
    ldp s14, s15, [sp, #-16]!

    ret


    .global _perm_neon_abc_cba
_perm_neon_abc_cba:

    ret