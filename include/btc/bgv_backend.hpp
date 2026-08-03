#pragma once

#ifdef TC_WITH_BGV

#include "bgv.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace btc {

// Matrix of BGV ciphertexts (row-major, square). Mirrors DGHVMatrix in
// dghv_backend.hpp / TFHEMatrix in tfhe_backend.hpp so all three backends
// are structurally interchangeable.
//
// One ciphertext per matrix element, exactly as for DGHV/TFHE. See the
// "Future batching" note on BGVBackend::matmul below for how SIMD slot
// packing could reduce this to O(1) ciphertexts per row/matrix later.
class BGVMatrix {
public:
    explicit BGVMatrix(std::size_t n) : n_(n), data_(n * n) {}

    std::size_t dim() const noexcept { return n_; }

    const bgv::Ciphertext& get(std::size_t i, std::size_t j) const {
        return data_[i * n_ + j];
    }

    void set(std::size_t i, std::size_t j, bgv::Ciphertext ct) {
        data_[i * n_ + j] = std::move(ct);
    }

private:
    std::size_t n_;
    std::vector<bgv::Ciphertext> data_;
};

// ── Multiplicative depth of the transitive-closure circuit ─────────────────
//
// Identical accounting to DGHVBackend's matmul_depth / transitive_closure_depth
// (dghv_backend.hpp) -- the circuit shape is backend-independent, only the
// per-multiplication cost differs between DGHV's raw GMP integers and BGV's
// RNS ciphertexts. Repeated here (rather than shared) so this file stays
// self-contained and BGV-specific documentation stays next to BGV-specific
// code, per the "keep cryptographic setup separate" / "avoid large
// abstraction changes" guidance.
//
//  1. matmul's dot product: for an N-node graph, C[i][j] = OR_k(A[i][k] AND
//     B[k][j]) combines N AND-terms. AND-ing one pair costs depth 1 (one
//     EvalMult). The N terms are then OR-reduced in a *balanced* binary
//     tree (ceil(log2 N) sequential OR levels, each OR costing one more
//     EvalMult for its a*b term) rather than a linear chain, so depth stays
//     O(log N) instead of O(N):
//
//       depth(matmul) = 1 (AND) + ceil(log2(N)) (OR-tree) = 1 + ceil(log2 N)
//
//  2. sum_powers_recursive (algorithms.hpp) halves T each level and, per
//     level, computes P_new = matmul(P,P) and S_new = S | matmul(P,S) --
//     i.e. one matmul then one more OR (one more EvalMult) on top of it.
//     So each recursion level adds depth(matmul) + 1. There are
//     ceil(log2 T) levels.
//
//       depth(sum_powers_recursive) = ceil(log2 T) * (depth(matmul) + 1)
//                                    = ceil(log2 T) * (2 + ceil(log2 N))
//
// bounded_transitive_closure calls sum_powers_recursive directly, so this
// is also the depth of the full encrypted transitive-closure computation.
// EstimateBGVDepth below is exactly transitive_closure_depth(n, T) --
// spelled out under its own name so callers building a BGV crypto context
// don't need to reach into the DGHV file to size multiplicative_depth.
inline int BGVMatmulDepth(std::size_t n) {
    int levels = 0;
    std::size_t remaining = n;
    while (remaining > 1) {
        remaining = (remaining + 1) / 2; // ceil(remaining / 2)
        ++levels;
    }
    return 1 + levels; // +1 for the AND before the OR-tree
}

// Estimated multiplicative depth required to run bounded_transitive_closure
// on an N-node graph for T rounds of repeated squaring under BGVBackend.
// This is what Params::multiplicative_depth should be set to (or exceed)
// before calling bgv::setup -- see the CCParams<CryptoContextBGVRNS> usage
// in examples/bgv_closure.cpp. The estimate is exact for this circuit
// shape (not merely an upper bound), since every AND/OR here costs exactly
// one EvalMult and no other multiplications occur.
inline uint32_t EstimateBGVDepth(std::size_t numberOfNodes, int T) {
    if (T < 1)
        throw std::invalid_argument("EstimateBGVDepth: T must be >= 1");
    int levels = 0;
    int remaining = T;
    while (remaining > 1) {
        remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
        ++levels;
    }
    return static_cast<uint32_t>(levels * (BGVMatmulDepth(numberOfNodes) + 1));
}

// BGVBackend -- satisfies the Backend concept (see backend.hpp) using
// btc::bgv ciphertexts. Unlike DGHVBackend (which is stateless: DGHV
// homomorphic ops need only the ciphertexts themselves), BGV's EvalAdd/
// EvalSub/EvalMult are methods on the CryptoContext, and EvalMult also
// needs the relinearization key installed on that context by bgv::setup.
// So BGVBackend carries a reference to the bgv::Context it was built from.
//
// Usage:
//   auto params = btc::bgv::Params{};
//   params.multiplicative_depth = btc::EstimateBGVDepth(N, T);
//   auto ctx = btc::bgv::setup(params);
//
//   btc::BGVBackend backend(ctx);
//   auto enc_A = btc::BGVBackend::encrypt(plain_A, ctx);
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = btc::BGVBackend::decrypt(enc_S, ctx);
class BGVBackend {
public:
    using matrix_type = BGVMatrix;

    explicit BGVBackend(const bgv::Context& ctx) : ctx_(&ctx) {}

    // C[i][j] = OR_k ( A[i][k] AND B[k][j] ), OR-reduced as a balanced tree
    // so depth stays O(log N) -- see BGVMatmulDepth above.
    //
    // ── Future batching ──────────────────────────────────────────────────
    // This implementation uses one ciphertext per matrix element (N^2
    // ciphertexts per matrix, N per dot product), matching DGHVBackend and
    // TFHEBackend for readability and cross-backend comparability. OpenFHE
    // BGV supports SIMD slot packing (a single ciphertext holds up to
    // GetBatchSize() plaintext slots that all homomorphic ops act on in
    // parallel via EvalAdd/EvalMult/rotations). A future optimization could
    // pack each row of A (or each AND-term column) into slots of one
    // ciphertext and use EvalRotate-based reductions to cut ciphertext
    // count roughly by the batch size -- left undone here since the
    // codebase has no existing packing abstraction to build on, and
    // correctness/readability were prioritized for this first BGV backend.
    BGVMatrix matmul(const BGVMatrix& A, const BGVMatrix& B) const {
        const std::size_t n = A.dim();
        if (B.dim() != n)
            throw std::invalid_argument("BGVBackend::matmul: dimension mismatch");
        BGVMatrix C(n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<bgv::Ciphertext> terms;
                terms.reserve(n);
                for (std::size_t k = 0; k < n; ++k)
                    terms.push_back(bgv::encrypted_and(*ctx_, A.get(i, k), B.get(k, j)));
                C.set(i, j, or_reduce_tree(terms));
            }
        }
        return C;
    }

    BGVMatrix or_mat(const BGVMatrix& A, const BGVMatrix& B) const {
        const std::size_t n = A.dim();
        BGVMatrix C(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                C.set(i, j, bgv::encrypted_or(*ctx_, A.get(i, j), B.get(i, j)));
        return C;
    }

    // OpenFHE ciphertext handles are shared_ptr-backed and immutable under
    // homomorphic ops (Eval* always return a new Ciphertext), so a shallow
    // copy of the handle is already a safe, independent logical copy.
    BGVMatrix copy(const BGVMatrix& A) const {
        const std::size_t n = A.dim();
        BGVMatrix C(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                C.set(i, j, A.get(i, j));
        return C;
    }

    std::size_t dim(const BGVMatrix& A) const { return A.dim(); }

    // Encrypt a plaintext BoolMatrix into a BGVMatrix.
    static BGVMatrix encrypt(const BoolMatrix& plain, const bgv::Context& ctx) {
        const std::size_t n = plain.rows();
        BGVMatrix enc(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                enc.set(i, j, bgv::encrypt_bit(ctx, plain.get(i, j)));
        return enc;
    }

    // Decrypt a BGVMatrix back to a plaintext BoolMatrix. Propagates
    // bgv::decrypt_bit's correctness-error exception (non-Boolean decrypted
    // value) rather than swallowing it.
    static BoolMatrix decrypt(const BGVMatrix& enc, const bgv::Context& ctx) {
        const std::size_t n = enc.dim();
        BoolMatrix plain(n, n, false);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                plain.set(i, j, bgv::decrypt_bit(ctx, enc.get(i, j)));
        return plain;
    }

private:
    // Combines `terms` with encrypted_or in a balanced binary tree
    // (ceil(log2 terms.size()) sequential levels) rather than a linear
    // chain, so multiplicative depth grows logarithmically in N.
    bgv::Ciphertext or_reduce_tree(std::vector<bgv::Ciphertext> terms) const {
        if (terms.empty())
            throw std::invalid_argument("BGVBackend::or_reduce_tree: empty input");
        while (terms.size() > 1) {
            std::vector<bgv::Ciphertext> next;
            next.reserve((terms.size() + 1) / 2);
            for (std::size_t i = 0; i + 1 < terms.size(); i += 2)
                next.push_back(bgv::encrypted_or(*ctx_, terms[i], terms[i + 1]));
            if (terms.size() % 2 == 1)
                next.push_back(std::move(terms.back()));
            terms = std::move(next);
        }
        return terms[0];
    }

    const bgv::Context* ctx_; // non-owning
};

} // namespace btc

#endif // TC_WITH_BGV
