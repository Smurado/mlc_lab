#!/usr/bin/env bash
# Ende-zu-Ende-Rauchtest: laeuft die komplette Kette Parser -> IR -> Codegen ->
# JIT -> Validierung wirklich durch?
#
# Bewusst KEINE Leistungspruefung. Auf einem geteilten CI-Runner waeren GFLOPS
# wertlos (fremde Last), und genau diese Kontamination vermeiden wir sonst
# strikt. Geprueft wird nur, ob ein Kernel erzeugt, kompiliert, geladen und als
# KORREKT validiert wurde. Die Zahl selbst ist egal.
#
# Ergaenzt die Unit-Tests: die pruefen die Bausteine einzeln, hier laeuft
# erstmals ein echter JIT-Kernel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO="$(dirname "$HERE")"
COMPILER="$AUTO/src/teir_compiler"

if [[ ! -x "$COMPILER" ]]; then
    echo "FEHLER: $COMPILER nicht gefunden. Erst 'make -C src' ausfuehren." >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 48^3: gross genug, um oberhalb der Mikro-Workload-Schwelle (1e5 FLOPs) zu
# liegen, klein genug fuer wenige Sekunden.
cat > "$WORK/smoke.csv" <<'CSV'
name,tensors,axes,primitives,schedule,invokes,einsum
smoke,in0:f32;in1:f32;out:f32,a:48;b:48;c:48,zero;gemm,a:sequential;b:sequential;c:sequential,zero;gemm,ab-ac-cb
CSV

echo "== Rauchtest: ab-ac-cb @ 48, 3 Trials =="
set +e
TEIR_INPUT="$WORK/smoke.csv" \
TEIR_MAX_TRIALS=3 \
TEIR_TIME_BUDGET_MS=60000 \
"$COMPILER" > "$WORK/out.txt" 2>&1
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
    echo "FEHLGESCHLAGEN: Exit-Code $RC" >&2
    tail -30 "$WORK/out.txt" >&2
    exit 1
fi

fail() {
    echo "FEHLGESCHLAGEN: $1" >&2
    tail -30 "$WORK/out.txt" >&2
    exit 1
}

grep -q "\[JIT SUCCESS\]"          "$WORK/out.txt" || fail "kein JIT-Erfolg"
grep -q "\[SUCCESS\] Validation passed" "$WORK/out.txt" || fail "Validierung nicht bestanden"
grep -q "\[PERFORMANCE\]"          "$WORK/out.txt" || fail "keine Endmessung"

echo "  [ok]   JIT-Kompilierung erfolgreich"
echo "  [ok]   Kernel als korrekt validiert"
echo "  [ok]   Endmessung durchgefuehrt (Zahl bewusst nicht geprueft)"
echo "== Rauchtest bestanden =="
