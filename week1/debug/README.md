# Aufgabe 2 — Beispielaufruf von `inner_product` im Debugger

> Use the GNU Project Debugger (GDB) to step through an example call to the inner
> product function.

## Werkzeugwahl: warum lldb statt gdb

Die Aufgabe nennt GDB. Auf der Zielplattform (Apple Silicon, arm64) ist mainline-GDB
dafür nicht verwendbar:

```
$ gdb --batch -ex "show architecture"
The target architecture is set to "auto" (currently "i386").

$ gdb --batch -ex "file ./test_run_debug" -ex "info functions inner_product"
DW_FORM_GNU_str_index or DW_FORM_strx used without .debug_str section in CU at offset 0x0
...
```

GDB erkennt das arm64-Binary als i386 und kann die DWARF-Debug-Informationen nicht
lesen — `aarch64-darwin` wird nicht unterstützt. `lldb` ist auf macOS das äquivalente
Werkzeug und liest dasselbe Binary korrekt (`(arm64)`).

Die entsprechenden GDB-Befehle stehen unten daneben, falls die Sitzung auf einer
Linux-Maschine nachvollzogen werden soll.

## Durchführung

```bash
make debug
lldb --batch -s debug/inner_product.lldb ./test_run_debug
```

Das vollständige Protokoll liegt in [`session_output.txt`](session_output.txt).

Der erste Aufruf von `inner_product` stammt aus dem Testfall
„Inner Product wird korrekt berechnet" mit
`a = {1,2,3,4}`, `b = {5,6,7,8}`, `size = 4`.
Erwartet: 1·5 + 2·6 + 3·7 + 4·8 = **70**.

## Beobachtungen

### Eintritt — Argumentübergabe nach AAPCS64

```
      x0 = 0x000000016fdfda70      ; Zeiger auf a
      x1 = 0x000000016fdfda60      ; Zeiger auf b
      w2 = 0x00000004              ; size = 4

0x16fdfda70: 1   0x16fdfda74: 2   0x16fdfda78: 3   0x16fdfda7c: 4      ; a
0x16fdfda60: 5   0x16fdfda64: 6   0x16fdfda68: 7   0x16fdfda6c: 8      ; b
```

Die ersten drei Ganzzahl-Argumente liegen in `x0`, `x1`, `x2` — wie von der
Aufrufkonvention vorgegeben. `size` wird als 32-Bit-Wert in `w2` gelesen.

### Prolog

```
->  0x1000dd1ec <+0>: mov    x8, #0x0     ; Summe
    0x1000dd1f0 <+4>: mov    w4, #0x0     ; Zaehler
```

Nach beiden Schritten: `x8 = 0`.

### Erster Schleifendurchlauf

```
    cmp    w4, w2                     ; 0 < 4  -> weiter
    ldr    w5, [x0, w4, uxtw #2]      ; a[0] = 1
    ldr    w6, [x1, w4, uxtw #2]      ; b[0] = 5
    umull  x5, w5, w6                 ; 1 * 5 = 5   (32x32 -> 64 Bit)
    add    x8, x8, x5

      w6 = 0x00000005
      x8 = 0x0000000000000005
```

Die Adressierung `[x0, w4, uxtw #2]` skaliert den Zähler mit 4 (`#2` = 2² Bytes) und
erweitert ihn vorzeichenlos — damit wird `a[w4]` ohne separate Adressrechnung geladen.

### Zweiter Durchlauf

```
      w4 = 0x00000001                 ; Zaehler steht auf 1
      w5 = 0x0000000c                 ; Ergebnis von umull: 2 * 6 = 12
      w6 = 0x00000006                 ; b[1] = 6
      x8 = 0x0000000000000011         ; 5 + 12 = 17
```

Die Summe wächst wie erwartet: 5 → 17. Nach den beiden verbleibenden Durchläufen
(+3·7 = 21, +4·8 = 32) ergibt sich 17 + 21 + 32 = **70**.

### Rücksprung

```
    mov    x0, x8      ; Rueckgabewert
    ret
```

`finish` kehrt in den Testfall zurück:

```
    frame #0: test_run_debug`C_A_T_C_H_T_E_S_T_0() at main.cpp:31:5
->  31  	    REQUIRE(inner_product(a, b, size) == 70);
```

Der Rückgabewert steht in `x0` — dieselbe Konvention wie bei der Argumentübergabe.

## Befehlsentsprechung lldb ↔ gdb

| Zweck | lldb | gdb |
|---|---|---|
| Haltepunkt setzen | `breakpoint set --name inner_product` | `break inner_product` |
| Starten | `run` | `run` |
| Register lesen | `register read x0 x1 w2` | `info registers x0 x1 w2` |
| Speicher lesen | `memory read --size 4 --format u --count 4 $x0` | `x/4wu $x0` |
| Einzelinstruktion | `thread step-inst` | `stepi` |
| N Instruktionen | `thread step-inst -c 6` | `stepi 6` |
| Bis Rücksprung | `finish` | `finish` |
| Disassemblieren | `disassemble --frame --count 8` | `x/8i $pc` |
