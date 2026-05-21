#ifndef MINI_JIT_UNARY_H
#define MINI_JIT_UNARY_H

#include <cstdint>
#include <cstddef>

namespace mini_jit {
  class Unary;
}

class mini_jit::Unary {
  public:
    /// Datentyp der Matrizen
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };

    /// Typ der unären Operation
    enum class ptype_t : uint32_t {
      zero     = 0,
      identity = 1,
      relu     = 2     
    };

    /// Fehlercodes
    enum class error_t : int32_t {
      success = 0
    };

    /**
     * @brief Generiert einen Kernel für eine unäre Matrixoperation.
     * @param m       Anzahl der Zeilen in A und B.
     * @param n       Anzahl der Spalten in A und B.
     * @param trans_b 0 wenn B formatmäßig spaltenbasiert vorliegt (column-major), 1 wenn zeilenbasiert.
     * @param dtype   Datentyp der Matrizen.
     * @param ptype   Gewünschte Operation (zero, identity, relu).
     * @return error_t::success bei Erfolg, ansonsten ein entsprechender Fehlercode.
     **/
    error_t generate( uint32_t m,
                      uint32_t n,
                      uint32_t trans_b,
                      dtype_t  dtype,
                      ptype_t  ptype );

    /*
     * Ein Kernel ist eine Funktion, die folgende Parameter entgegennimmt:
     * - a:    Pointer auf Matrix A (nullptr, falls "zero" Kernel).
     * - b:    Pointer auf Matrix B.
     * - ld_a: Leading Dimension von A.
     * - ld_b: Leading Dimension von B.
     */
    using kernel_t = void (*)( void    const * a,
                               void          * b,
                               int64_t         ld_a,
                               int64_t         ld_b );

    /**
     * @brief Liefert den generierten Kernel: B := op(A).
     * @return Zeiger auf die ausführbare Machine-Code Funktion.
     **/
    kernel_t get_kernel() const;

    ~Unary();

  private:
    void*  m_code = nullptr;
    size_t m_size = 0;
};

#endif