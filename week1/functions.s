.global _inner_product

_inner_product_cpp:
    # x0 -> a
    # x1 -> b
    # w2 -> i_size

    # w4 -> counter
    # x8 -> sum

    mov x8, #0          # Summe mit 0 initialisieren
    mov w4, #0          # Counter mit 0 initialisieren

    loop_start:
        cmp w4, w2
        b.ge loop_end           # Springe ans Ende der Schleife, falls counter >= i

        ldr w5, [x0, w4, uxtw 2]    # w5 = a[counter] (Basisadresse + counter * 4)
        ldr w6, [x1, w4, uxtw 2]    # w6 = b[counter]

        umull x5, w5, w6        # a[counter] * b[counter] (Zielregister 64bit)

        add x8, x8, x5          # sum += x5

        add w4, w4, #1          # counter++

        b loop_start            # Springe zurück zum Anfang
    loop_end:
    mov x0, x8          # Rückgabewert = sum       
    ret                 # Return


    