#pragma once

#include "matrix.hpp"

#include <stdexcept>

namespace btc {

// PlaintextBackend satisfies the Backend concept for plain boolean arithmetic.
//
// ── Backend concept (informally; C++17 lacks concepts) ───────────────────────
//
//   typename Backend::matrix_type               — matrix element type
//   matrix_type matmul(A, B) const              — boolean matrix multiplication
//   matrix_type or_mat(A, B) const              — element-wise OR
//   matrix_type copy(const matrix_type&) const  — deep copy
//   std::size_t dim(const matrix_type&) const   — square dimension
//
// ── Adding a TFHE backend ────────────────────────────────────────────────────
//
//   class TFHEBackend {
//   public:
//       using matrix_type = TFHEMatrix;   // wraps TFHE-rs ciphertext matrix
//       TFHEMatrix matmul(const TFHEMatrix& A, const TFHEMatrix& B) const;
//       TFHEMatrix or_mat(const TFHEMatrix& A, const TFHEMatrix& B) const;
//       TFHEMatrix copy(const TFHEMatrix& A) const;
//       std::size_t dim(const TFHEMatrix& A) const;
//   };
//
//   Then pass TFHEBackend{} to sum_powers_recursive / bounded_transitive_closure
//   — no changes to algorithms.hpp are required.
// ─────────────────────────────────────────────────────────────────────────────

class PlaintextBackend {
public:
    using matrix_type = BoolMatrix;

    // C[i][j] = OR_k ( A[i][k] AND B[k][j] )
    BoolMatrix matmul(const BoolMatrix& A, const BoolMatrix& B) const;

    BoolMatrix or_mat(const BoolMatrix& A, const BoolMatrix& B) const {
        return A | B;
    }

    BoolMatrix copy(const BoolMatrix& A) const { return A; }

    std::size_t dim(const BoolMatrix& A) const { return A.rows(); }
};

} // namespace btc
