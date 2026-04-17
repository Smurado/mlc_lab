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

    ret