#ifndef MINI_JIT_GEMM_H
#define MINI_JIT_GEMM_H

#include <cstdint>
#include <cstddef>

namespace mini_jit {
  class Gemm;
}

class mini_jit::Gemm {
  public:
    /// Datentyp der Matrizen
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };

    /// Fehlercodes
    enum class error_t : int32_t {
      success = 0
    };

    /**
     * @brief Generiert einen Kernel für Matrizen-Multiplikation.
     * @param m       Anzahl der Zeilen in A und C.
     * @param n       Anzahl der Spalten in B und C.
     * @param k       Anzahl der Spalten in A und Zeilen in B.
     * @param trans_a 0 wenn A spaltenbasiert (column-major) vorliegt, 1 wenn zeilenbasiert.
     * @param trans_b 0 wenn B spaltenbasiert vorliegt, 1 wenn zeilenbasiert.
     * @param trans_c 0 wenn C spaltenbasiert vorliegt, 1 wenn zeilenbasiert.
     * @param dtype   Datentyp der Matrizen.
     * @return error_t::success bei Erfolg, ansonsten passender Fehlercode.
     **/
    error_t generate( uint32_t m,
                      uint32_t n,
                      uint32_t k,
                      uint32_t trans_a,
                      uint32_t trans_b,
                      uint32_t trans_c,
                      dtype_t  dtype );

    /*
     * C-Signatur des Kernels, um ihn via Pointer aufrufen zu können.
     * Nimmt folgende Parameter entgegen:
     * - a:           Pointer auf Matrix A.
     * - b:           Pointer auf Matrix B.
     * - c:           Pointer auf Matrix C.
     * - ld_a:        Leading Dimension von A.
     * - ld_b:        Leading Dimension von B.
     * - ld_c:        Leading Dimension von C.
     */
    using kernel_t = void (*)( void    const * a,
                               void    const * b,
                               void          * c,
                               int64_t         ld_a,
                               int64_t         ld_b,
                               int64_t         ld_c);

    /**
     * @brief Liefert den ausführbaren JIT Kernel: C += A * B.
     * @return Zeiger auf die im Buffer liegende Maschinencode-Funktion.
     **/
    kernel_t get_kernel() const;

    ~Gemm();

  private:
    void*  m_code = nullptr;
    size_t m_size = 0;
};

#endif