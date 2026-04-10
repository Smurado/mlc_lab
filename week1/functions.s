.global _add

_add:
    movl %edi, %eax   # erstes Argument -> eax
    addl %esi, %eax   # zweites Argument addieren
    ret