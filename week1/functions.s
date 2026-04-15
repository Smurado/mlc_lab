.global _outer_product, _inner_product, outer_product, inner_product

_inner_product:
inner_product:
    mov x8, #0                  // Summe initialisieren
    mov w4, #0                  // Zaehler initialisieren

loop_start:
    cmp w4, w2
    b.ge loop_end               // Springe zum Ende, falls Zaehler >= Groesse

    ldr w5, [x0, w4, uxtw 2]    // Lade a[counter]
    ldr w6, [x1, w4, uxtw 2]    // Lade b[counter]

    umull x5, w5, w6            // Multipliziere a[counter] * b[counter] (64-Bit Ergebnis)
    add x8, x8, x5              // Zur Summe addieren

    add w4, w4, #1              // Zaehler inkrementieren
    b loop_start                // Zurueck zum Schleifenanfang

loop_end:
    mov x0, x8                  // Rueckgabewert (Summe) setzen
    ret                         // Zum Aufrufer zurueckkehren

_outer_product:
outer_product:
    mov x8, x1                  // Zeiger auf i_b sichern
    mov w4, #0                  // Initiiere l_i = 0
outerloop:
    cmp w4, w2
    b.ge outer_end              // Pruefe ob l_i >= i_size, Schleife verlassen

    mov w5, #0                  // Initiiere l_j = 0
    ldr w6, [x0]                // Aktuelles Element von i_a laden
    
    mov x1, x8                  // Zeiger auf i_b fuer innere Schleife wiederherstellen
    innerloop:
        cmp w5, w2
        b.ge inner_end          // Pruefe ob l_j >= i_size, innere Schleife verlassen

        ldr w7, [x1]            // Aktuelles Element von i_b laden

        umull x9, w6, w7        // Multipliziere i_a[l_i] * i_b[l_j] (64-Bit Ergebnis)
        str x9, [x3]            // Ergebnis in o_c speichern
        add x3, x3, #8          // o_c Zeiger inkrementieren (8 Bytes pro uint64_t)

        add x1, x1, #4          // i_b Zeiger inkrementieren
        add w5, w5, #1          // l_j inkrementieren
        b innerloop

inner_end:
    add x0, x0, #4              // i_a Zeiger inkrementieren
    add w4, w4, #1              // l_i inkrementieren
    b outerloop

outer_end:
    ret                         // Zum Aufrufer zurueckkehren
