// Referenzimplementierung der Aufgabe Woche 1.
//
// HINWEIS: Diese Datei wird laut Aufgabenstellung vom Kurs im Verzeichnis `data`
// bereitgestellt, war im Repository aber nicht vorhanden. Sie ist hier anhand der
// Aufgabenbeschreibung und der Semantik der Assembly-Implementierung
// (`../functions.s`) rekonstruiert. Falls die Originaldatei verfuegbar wird,
// sollte sie diese ersetzen.
//
// Semantik bewusst identisch zur Assembly:
//   - Multiplikation ist UNSIGNED 32x32 -> 64 Bit (`umull`), daher die
//     Erweiterung auf uint64_t vor der Multiplikation.
//   - inner_product summiert in 64 Bit und liefert int64_t zurueck.
//   - outer_product schreibt zeilenweise: c[i * size + j] = a[i] * b[j].

#include <cstdint>

/**
 * @brief Inneres Produkt zweier Vektoren.
 * @param i_a    Zeiger auf den ersten Vektor.
 * @param i_b    Zeiger auf den zweiten Vektor.
 * @param i_size Anzahl der Elemente je Vektor.
 * @return Summe ueber i_a[k] * i_b[k].
 */
int64_t inner_product_cpp(uint32_t const *i_a,
                          uint32_t const *i_b,
                          uint32_t        i_size)
{
    uint64_t l_sum = 0;

    for (uint32_t l_k = 0; l_k < i_size; ++l_k)
    {
        // Wie `umull`: auf 64 Bit erweitern, dann multiplizieren.
        l_sum += static_cast<uint64_t>(i_a[l_k]) * static_cast<uint64_t>(i_b[l_k]);
    }

    return static_cast<int64_t>(l_sum);
}

/**
 * @brief Aeusseres Produkt zweier Vektoren.
 * @param i_a    Zeiger auf den ersten Vektor.
 * @param i_b    Zeiger auf den zweiten Vektor.
 * @param i_size Anzahl der Elemente je Vektor.
 * @param o_c    Zeiger auf die Ergebnismatrix (i_size x i_size, zeilenweise).
 */
void outer_product_cpp(uint32_t const *i_a,
                       uint32_t const *i_b,
                       uint32_t        i_size,
                       uint64_t       *o_c)
{
    for (uint32_t l_i = 0; l_i < i_size; ++l_i)
    {
        for (uint32_t l_j = 0; l_j < i_size; ++l_j)
        {
            o_c[l_i * i_size + l_j] =
                static_cast<uint64_t>(i_a[l_i]) * static_cast<uint64_t>(i_b[l_j]);
        }
    }
}
