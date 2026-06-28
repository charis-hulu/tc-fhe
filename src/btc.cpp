#include <btc/algorithms.hpp>

#include <stdexcept>

namespace btc {

// i-k-j loop order: the inner j-loop walks contiguous memory in both B and C.
// Short-circuit on A[i][k] == 0 keeps the inner loop from running on sparse rows.
BoolMatrix PlaintextBackend::matmul(const BoolMatrix& A, const BoolMatrix& B) const {
    const std::size_t rows  = A.rows();
    const std::size_t inner = A.cols();
    const std::size_t cols  = B.cols();

    if (inner != B.rows())
        throw std::invalid_argument("PlaintextBackend::matmul: incompatible dimensions");

    BoolMatrix C(rows, cols, false);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < inner; ++k) {
            if (!A.get(i, k)) continue;
            const uint8_t* b_row = B.row_data(k);
            uint8_t*       c_row = C.row_data(i);
            for (std::size_t j = 0; j < cols; ++j)
                c_row[j] |= b_row[j];
        }
    }
    return C;
}

BoolMatrix floyd_warshall(const BoolMatrix& A) {
    const std::size_t n = A.rows();
    if (n != A.cols())
        throw std::invalid_argument("floyd_warshall: matrix must be square");

    BoolMatrix R = A;

    // Standard Warshall transitive-closure algorithm.
    // R[i][j] = 1 iff there is a path from i to j through intermediate nodes {0..k}.
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!R.get(i, k)) continue;
            const uint8_t* rk = R.row_data(k);
            uint8_t*       ri = R.row_data(i);
            for (std::size_t j = 0; j < n; ++j)
                ri[j] |= rk[j];
        }
    }
    return R;
}

} // namespace btc
