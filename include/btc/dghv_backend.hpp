#pragma once

#ifdef TC_WITH_DGHV

#include "dghv.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace btc {

// Matrix of DGHV ciphertexts (row-major, square). Mirrors TFHEMatrix in
// tfhe_backend.hpp so the two backends are structurally interchangeable.
class DGHVMatrix {
public:
    explicit DGHVMatrix(std::size_t n) : n_(n), data_(n * n) {}

    std::size_t dim() const noexcept { return n_; }

    const dghv::Ciphertext& get(std::size_t i, std::size_t j) const {
        return data_[i * n_ + j];
    }

    void set(std::size_t i, std::size_t j, dghv::Ciphertext ct) {
        data_[i * n_ + j] = std::move(ct);
    }

private:
    std::size_t n_;
    std::vector<dghv::Ciphertext> data_;
};

// ── Multiplicative depth of the transitive-closure circuit ─────────────────
//
// Every AND and every OR (via a*b inside OR(a,b)=a+b-ab) costs exactly one
// DGHV multiplication, i.e. one level of depth. Two design choices below
// determine the total:
//
//  1. matmul's dot product: for an N-node graph, C[i][j] = OR_k(A[i][k] AND
//     B[k][j]) combines N AND-terms. AND-ing one pair costs depth 1. The N
//     terms are then OR-reduced; a *balanced* binary tree does this in
//     ceil(log2 N) sequential OR levels (vs. TFHEBackend's simple serial
//     accumulation, which is fine there because every TFHE gate bootstraps
//     away its noise and "depth" is not a resource). DGHVBackend below uses
//     a balanced tree specifically so depth stays O(log N) instead of O(N).
//
//       depth(matmul) = 1 (AND) + ceil(log2(N)) (OR-tree) = 1 + ceil(log2 N)
//
//  2. sum_powers_recursive (algorithms.hpp) halves T each level and, per
//     level, computes P_new = matmul(P,P) and S_new = S | matmul(P,S) --
//     i.e. one matmul then one more OR on top of it. So each recursion
//     level adds depth(matmul) + 1. There are ceil(log2 T) levels.
//
//       depth(sum_powers_recursive) = ceil(log2 T) * (depth(matmul) + 1)
//                                    = ceil(log2 T) * (2 + ceil(log2 N))
//
// bounded_transitive_closure calls sum_powers_recursive directly, so this
// is also the depth of the full encrypted transitive-closure computation.
inline int matmul_depth(std::size_t n) {
    int levels = 0;
    std::size_t remaining = n;
    while (remaining > 1) {
        remaining = (remaining + 1) / 2; // ceil(remaining / 2)
        ++levels;
    }
    return 1 + levels; // +1 for the AND before the OR-tree
}

inline int transitive_closure_depth(std::size_t n, int T) {
    if (T < 1)
        throw std::invalid_argument("transitive_closure_depth: T must be >= 1");
    int levels = 0;
    int remaining = T;
    while (remaining > 1) {
        remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
        ++levels;
    }
    return levels * (matmul_depth(n) + 1);
}

// DGHVBackend -- satisfies the Backend concept (see backend.hpp) using
// btc::dghv ciphertexts. Unlike TFHEBackend, there is no server key: every
// homomorphic op here is pure integer arithmetic on the ciphertext, so the
// backend object carries no state at all.
//
// Usage:
//   auto params = btc::dghv::experiment_params(btc::transitive_closure_depth(N, T));
//   btc::dghv::require_depth(params, btc::transitive_closure_depth(N, T));
//   std::mt19937_64 rng(seed);
//   auto key = btc::dghv::keygen(params, rng);
//
//   btc::DGHVBackend backend;
//   auto enc_A = btc::DGHVBackend::encrypt(plain_A, key, params, rng);
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = btc::DGHVBackend::decrypt(enc_S, key);
class DGHVBackend {
public:
    using matrix_type = DGHVMatrix;

    // C[i][j] = OR_k ( A[i][k] AND B[k][j] ), OR-reduced as a balanced tree
    // so depth stays O(log N) -- see matmul_depth above.
    DGHVMatrix matmul(const DGHVMatrix& A, const DGHVMatrix& B) const {
        const std::size_t n = A.dim();
        if (B.dim() != n)
            throw std::invalid_argument("DGHVBackend::matmul: dimension mismatch");
        DGHVMatrix C(n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<dghv::Ciphertext> terms;
                terms.reserve(n);
                for (std::size_t k = 0; k < n; ++k)
                    terms.push_back(dghv::encrypted_and(A.get(i, k), B.get(k, j)));
                C.set(i, j, or_reduce_tree(terms));
            }
        }
        return C;
    }

    DGHVMatrix or_mat(const DGHVMatrix& A, const DGHVMatrix& B) const {
        const std::size_t n = A.dim();
        DGHVMatrix C(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                C.set(i, j, dghv::encrypted_or(A.get(i, j), B.get(i, j)));
        return C;
    }

    DGHVMatrix copy(const DGHVMatrix& A) const {
        const std::size_t n = A.dim();
        DGHVMatrix C(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                C.set(i, j, A.get(i, j)); // Ciphertext is a plain mpz_class wrapper: copy is a deep copy
        return C;
    }

    std::size_t dim(const DGHVMatrix& A) const { return A.dim(); }

    // Encrypt a plaintext BoolMatrix into a DGHVMatrix.
    static DGHVMatrix encrypt(const BoolMatrix& plain, const dghv::SecretKey& key,
                               const dghv::Params& params, std::mt19937_64& rng) {
        const std::size_t n = plain.rows();
        DGHVMatrix enc(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                enc.set(i, j, dghv::encrypt_bit(key, params, plain.get(i, j), rng));
        return enc;
    }

    // Decrypt a DGHVMatrix back to a plaintext BoolMatrix.
    static BoolMatrix decrypt(const DGHVMatrix& enc, const dghv::SecretKey& key) {
        const std::size_t n = enc.dim();
        BoolMatrix plain(n, n, false);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                plain.set(i, j, dghv::decrypt_bit(key, enc.get(i, j)));
        return plain;
    }

private:
    // Combines `terms` with encrypted_or in a balanced binary tree
    // (ceil(log2 terms.size()) sequential levels) rather than a linear
    // chain, so multiplicative depth grows logarithmically in N.
    static dghv::Ciphertext or_reduce_tree(std::vector<dghv::Ciphertext> terms) {
        if (terms.empty())
            throw std::invalid_argument("DGHVBackend::or_reduce_tree: empty input");
        while (terms.size() > 1) {
            std::vector<dghv::Ciphertext> next;
            next.reserve((terms.size() + 1) / 2);
            for (std::size_t i = 0; i + 1 < terms.size(); i += 2)
                next.push_back(dghv::encrypted_or(terms[i], terms[i + 1]));
            if (terms.size() % 2 == 1)
                next.push_back(std::move(terms.back()));
            terms = std::move(next);
        }
        return terms[0];
    }
};

} // namespace btc

#endif // TC_WITH_DGHV
