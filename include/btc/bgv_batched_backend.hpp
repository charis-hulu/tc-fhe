#pragma once

#ifdef TC_WITH_BGV_BATCHED

#include "bgv_batched.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace btc {

// ── Full-matrix row-major SIMD packing ───────────────────────────────────
//
// One ciphertext holds one complete N x N Boolean matrix:
//
//   slot(i, j) = i * N + j
//
// Only the first N*N slots are matrix slots; unused slots are zero. This
// wraps a single bgv_batched::Ciphertext -- unlike BGVMatrix (bgv_backend.hpp),
// which is a grid of N*N separate ciphertexts, BGVBatchedMatrix is one
// ciphertext (plus the dimension N).
class BGVBatchedMatrix {
public:
    BGVBatchedMatrix() : n_(0) {}
    BGVBatchedMatrix(bgv_batched::Ciphertext ct, std::size_t n) : ct_(std::move(ct)), n_(n) {}

    std::size_t dim() const noexcept { return n_; }
    const bgv_batched::Ciphertext& ciphertext() const { return ct_; }
    void set_ciphertext(bgv_batched::Ciphertext ct) { ct_ = std::move(ct); }

private:
    bgv_batched::Ciphertext ct_;
    std::size_t n_;
};

// ── Multiplicative depth of the packed transitive-closure circuit ─────────
//
// IMPORTANT: in OpenFHE's default scaling technique (FIXEDAUTO/FLEXIBLEAUTO,
// what bgv_batched::setup uses), ciphertext-PLAINTEXT EvalMult (used for
// mask_mult) consumes exactly one multiplicative-depth level, the SAME as
// ciphertext-ciphertext EvalMult -- both trigger a ModReduce that drops one
// RNS tower once the multiplied result's noise-scale degree reaches 2. This
// is NOT "free" the way EvalAdd/EvalSub/EvalRotate are (those never call
// ModReduce and never consume a level). Under-counting mask multiplications
// causes a real "Too few towers to support ModReduce" failure at runtime,
// not merely a conservative estimate -- verified empirically against this
// project's linked OpenFHE build while developing this backend.
//
// Packed matmul (see BooleanMatrixMultiplyPacked below) computes, for each
// k in [0, N), two broadcasts run independently before being AND-ed:
//
//   L_k = BroadcastColumnHorizontally(A, k):
//           mask_mult (1 level) -> rotate+add (free) -> mask_mult (1 level)
//         = 2 levels
//   Q_k = BroadcastRowVertically(B, k): symmetric, also 2 levels
//
//   T_k = EvalMult(L_k, Q_k)   -- ciphertext-ciphertext AND, 1 more level
//
// (L_k and Q_k are computed independently/in parallel, so their 2-level
// costs do not add to each other -- only the final AND on top of them does.)
// So T_k costs 2 (broadcast masks) + 1 (AND) = 3 levels. Then OR-reducing
// the N terms T_0..T_{N-1} in a balanced binary tree costs ceil(log2 N) more
// levels (one EvalMult per OR for its a*b term):
//
//   depth(matmul) = 2 (broadcast masks) + 1 (AND) + ceil(log2 N) (OR-tree)
//                 = 3 + ceil(log2 N)
//
// Packing changes ciphertext COUNT and rotation COUNT relative to
// BGVMatmulDepth (bgv_backend.hpp, "= 1 + ceil(log2 N)"), not just the
// asymptotic shape -- the two extra mask-multiplication levels are the
// price of full-matrix SIMD broadcast via row/column masks, and are exactly
// why this constant differs from the unpacked backend's.
//
// sum_powers_recursive (algorithms.hpp) then contributes the same
// ceil(log2 T) * (depth(matmul) + 1) recursion-depth term as BGV/DGHV.
inline int BGVBatchedMatmulDepth(std::size_t n) {
    int levels = 0;
    std::size_t remaining = n;
    while (remaining > 1) {
        remaining = (remaining + 1) / 2;
        ++levels;
    }
    return 3 + levels;
}

inline uint32_t EstimateBGVBatchedDepth(std::size_t numberOfNodes, int T) {
    if (T < 1)
        throw std::invalid_argument("EstimateBGVBatchedDepth: T must be >= 1");
    int levels = 0;
    int remaining = T;
    while (remaining > 1) {
        remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
        ++levels;
    }
    // T == 1 needs zero recursion levels (sum_powers_recursive's T==1 case is
    // a plain copy, no EvalMult at all), so `levels` can legitimately be 0
    // here. bgv_batched::setup still requires multiplicative_depth >= 1
    // (OpenFHE's BGV RNS parameter generation needs at least one RNS tower
    // budgeted for multiplication even if the caller's circuit never uses
    // it), so the estimate is floored at 1 rather than returned as 0.
    return std::max<uint32_t>(1, static_cast<uint32_t>(levels * (BGVBatchedMatmulDepth(numberOfNodes) + 1)));
}

// BGVBatchedBackend -- satisfies the Backend concept (see backend.hpp) using
// one packed btc::bgv_batched::Ciphertext per whole N x N matrix instead of
// one ciphertext per matrix element (contrast BGVBackend, bgv_backend.hpp).
//
// Precomputation (row masks, column masks, the valid-matrix-region mask, and
// rotation keys) is done once in the constructor and reused across every
// matmul call and TC round -- masks and keys must never be rebuilt inside a
// round (see the task's precomputation requirement).
//
// Usage:
//   auto params = btc::bgv_batched::Params{};
//   params.multiplicative_depth = btc::EstimateBGVBatchedDepth(N, T);
//   auto ctx = btc::bgv_batched::setup(params);
//
//   btc::BGVBatchedBackend backend(ctx, N);
//   auto enc_A = backend.PackMatrix_Encrypt(plain_A);   // see below
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = backend.Decrypt_UnpackMatrix(enc_S);
class BGVBatchedBackend {
public:
    using matrix_type = BGVBatchedMatrix;

    // `n` is fixed for the lifetime of this backend instance: masks and
    // rotation keys are sized for one specific N (this is the "one backend
    // instance per N" convention already implicit in BGVBackend/DGHVBackend,
    // made explicit here because packed masks are genuinely N-dependent
    // precomputed state rather than being derived fresh per call).
    BGVBatchedBackend(bgv_batched::Context& ctx, std::size_t n) : ctx_(&ctx), n_(n) {
        const std::size_t needed = n_ * n_;
        const uint32_t slots = bgv_batched::available_slots(ctx);
        if (needed > slots) {
            throw std::invalid_argument(
                "BGVBatchedBackend: N*N (" + std::to_string(needed) + ", N=" +
                std::to_string(n_) + ") exceeds available packed slots (" +
                std::to_string(slots) + "). Full-matrix row-major packing "
                "requires the entire N x N matrix to fit in one ciphertext; "
                "tiled/multi-ciphertext packing is not implemented in this "
                "backend. Reduce N, increase --bgv-batched-ring-dim (larger "
                "ring dimension raises the batch size), or use the legacy "
                "'bgv' or 'legacy' backend instead.");
        }

        PrecomputeMasks();
        GenerateRequiredRotationKeys();
    }

    std::size_t dim(const BGVBatchedMatrix& A) const { return A.dim(); }

    BGVBatchedMatrix copy(const BGVBatchedMatrix& A) const {
        // OpenFHE ciphertext handles are shared_ptr-backed and immutable
        // under homomorphic ops, so a shallow handle copy is already a safe,
        // independent logical copy (same reasoning as BGVBackend::copy).
        return BGVBatchedMatrix(A.ciphertext(), A.dim());
    }

    BGVBatchedMatrix or_mat(const BGVBatchedMatrix& A, const BGVBatchedMatrix& B) const {
        RequireSameDim(A, B, "or_mat");
        auto result = bgv_batched::encrypted_or(*ctx_, A.ciphertext(), B.ciphertext());
        return BGVBatchedMatrix(std::move(result), n_);
    }

    // C[i][j] = OR_k ( A[i][k] AND B[k][j] ), evaluated across ALL (i,j)
    // simultaneously for each k via SIMD broadcast + slot-wise multiply, then
    // OR-reduced over k as a balanced tree. See BooleanMatrixMultiplyPacked.
    BGVBatchedMatrix matmul(const BGVBatchedMatrix& A, const BGVBatchedMatrix& B) const {
        RequireSameDim(A, B, "matmul");
        return BooleanMatrixMultiplyPacked(A, B);
    }

    // ── Packing / encryption convenience wrappers ────────────────────────

    lbcrypto::Plaintext PackMatrix(const BoolMatrix& matrix) const {
        if (matrix.rows() != n_ || matrix.cols() != n_)
            throw std::invalid_argument("BGVBatchedBackend::PackMatrix: dimension mismatch");
        std::vector<int64_t> slots(bgv_batched::available_slots(*ctx_), 0);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                slots[i * n_ + j] = matrix.get(i, j) ? 1 : 0;
        return bgv_batched::make_packed_plaintext(*ctx_, slots);
    }

    BoolMatrix UnpackMatrix(const std::vector<int64_t>& slots) const {
        BoolMatrix result(n_, n_, false);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j) {
                int64_t v = slots[i * n_ + j];
                if (v != 0 && v != 1)
                    throw std::runtime_error(
                        "BGVBatchedBackend::UnpackMatrix: slot (" + std::to_string(i) + "," +
                        std::to_string(j) + ") = " + std::to_string(v) +
                        " is not a valid Boolean (0 or 1).");
                result.set(i, j, v != 0);
            }
        return result;
    }

    BGVBatchedMatrix Encrypt(const BoolMatrix& plain) const {
        auto pt = PackMatrix(plain);
        auto ct = ctx_->crypto_context->Encrypt(ctx_->public_key, pt);
        return BGVBatchedMatrix(std::move(ct), n_);
    }

    BoolMatrix Decrypt(const BGVBatchedMatrix& enc) const {
        auto raw = bgv_batched::decrypt_packed_raw(*ctx_, enc.ciphertext(), n_ * n_);
        return UnpackMatrix(raw);
    }

    // ── Layout-test entry points ──────────────────────────────────────────
    //
    // Exposed (not private) specifically so tests/test_bgv_batched_packing.cpp
    // can independently verify column masking, horizontal broadcast, row
    // masking, and vertical broadcast in isolation, per the task's explicit
    // testing requirements -- these are not part of the public Backend
    // concept surface used by algorithms.hpp.

    const lbcrypto::Plaintext& ColumnMask(std::size_t k) const { return column_masks_.at(k); }
    const lbcrypto::Plaintext& RowMask(std::size_t k) const { return row_masks_.at(k); }
    const lbcrypto::Plaintext& ValidRegionMask() const { return valid_region_mask_; }

    bgv_batched::Ciphertext BroadcastColumnHorizontally(const bgv_batched::Ciphertext& matrix, std::size_t k) const {
        return BroadcastColumnHorizontallyImpl(matrix, k);
    }

    bgv_batched::Ciphertext BroadcastRowVertically(const bgv_batched::Ciphertext& matrix, std::size_t k) const {
        return BroadcastRowVerticallyImpl(matrix, k);
    }

    bgv_batched::Ciphertext BooleanOr(const bgv_batched::Ciphertext& x, const bgv_batched::Ciphertext& y) const {
        return bgv_batched::encrypted_or(*ctx_, x, y);
    }

    bgv_batched::Ciphertext BooleanOrReduce(std::vector<bgv_batched::Ciphertext> terms) const {
        return OrReduceTree(std::move(terms));
    }

private:
    static void RequireSameDim(const BGVBatchedMatrix& A, const BGVBatchedMatrix& B, const char* who) {
        if (A.dim() != B.dim())
            throw std::invalid_argument(std::string("BGVBatchedBackend::") + who + ": dimension mismatch");
    }

    // Builds the plaintext masks used throughout packed matmul: N column
    // masks, N row masks, and the single valid-matrix-region mask. Built
    // once per backend instance (constructor) and reused for every matmul /
    // TC round -- never rebuilt per-round.
    void PrecomputeMasks() {
        const uint32_t slots = bgv_batched::available_slots(*ctx_);

        std::vector<int64_t> region(slots, 0);
        for (std::size_t idx = 0; idx < n_ * n_; ++idx)
            region[idx] = 1;
        valid_region_mask_ = bgv_batched::make_packed_plaintext(*ctx_, region);

        column_masks_.reserve(n_);
        row_masks_.reserve(n_);
        for (std::size_t k = 0; k < n_; ++k) {
            std::vector<int64_t> col_mask(slots, 0);
            std::vector<int64_t> row_mask(slots, 0);
            for (std::size_t i = 0; i < n_; ++i) {
                col_mask[i * n_ + k] = 1; // M_col_k[i,j] = 1 iff j == k
                row_mask[k * n_ + i] = 1; // M_row_k[i,j] = 1 iff i == k
            }
            column_masks_.push_back(bgv_batched::make_packed_plaintext(*ctx_, col_mask));
            row_masks_.push_back(bgv_batched::make_packed_plaintext(*ctx_, row_mask));
        }
    }

    // Horizontal offsets: -(N-1)..-1, 1..(N-1). Vertical offsets:
    // -(N-1)*N..-N step N, N..(N-1)*N step N. Deduplicated and generated
    // once (constructor), never inside a matmul/TC round.
    void GenerateRequiredRotationKeys() {
        std::vector<int32_t> offsets;
        offsets.reserve(4 * n_);
        for (std::size_t d = 1; d < n_; ++d) {
            offsets.push_back(static_cast<int32_t>(d));
            offsets.push_back(-static_cast<int32_t>(d));
            offsets.push_back(static_cast<int32_t>(d * n_));
            offsets.push_back(-static_cast<int32_t>(d * n_));
        }
        bgv_batched::generate_rotation_keys(*ctx_, offsets);
    }

    // L_k[i,j] = A[i,k], for all (i,j) simultaneously.
    //
    // 1. Mask out everything except column k: columnK = EvalMult(A, colMask_k)
    //    (ciphertext-PLAINTEXT multiplication -- see mask_mult). columnK's
    //    only nonzero content per row i sits at flat slot i*N+k.
    // 2. Broadcast each A[i,k] across its whole row via rotations. OpenFHE's
    //    EvalRotate(ct, offset) convention (pinned down empirically in
    //    tests/test_bgv_batched_rotation.cpp and wrapped by rotateToOffset)
    //    is a LEFT cyclic shift for offset > 0: rotated[idx] = orig[idx +
    //    offset]. To move content FROM flat slot i*N+k TO flat slot i*N+j we
    //    need rotated[i*N+j] = columnK[i*N+k], i.e. (i*N+j)+offset = i*N+k,
    //    so offset = k - j:
    //
    //        L_k = sum_j Rotate(columnK, k - j)   for j in [0, N).
    //
    //    j == k contributes the unrotated columnK term (offset 0).
    bgv_batched::Ciphertext BroadcastColumnHorizontallyImpl(const bgv_batched::Ciphertext& A, std::size_t k) const {
        auto column_k = bgv_batched::mask_mult(*ctx_, A, column_masks_[k]);

        bgv_batched::Ciphertext acc = column_k; // offset 0 (j == k): no rotation call
        for (std::size_t j = 0; j < n_; ++j) {
            if (j == k)
                continue;
            int32_t offset = static_cast<int32_t>(k) - static_cast<int32_t>(j);
            auto rotated = bgv_batched::rotateToOffset(*ctx_, column_k, offset);
            acc = bgv_batched::add(*ctx_, acc, rotated);
        }
        return bgv_batched::mask_mult(*ctx_, acc, valid_region_mask_);
    }

    // Q_k[i,j] = B[k,j], for all (i,j) simultaneously.
    //
    // 1. Mask out everything except row k: rowK = EvalMult(B, rowMask_k).
    //    rowK's only nonzero content sits at flat slots k*N+j for j in
    //    [0, N).
    // 2. Broadcast row k into every output row. Using the same left-shift
    //    convention as above: to move content FROM flat slot k*N+j TO flat
    //    slot i*N+j we need rotated[i*N+j] = rowK[k*N+j], i.e.
    //    (i*N+j)+offset = k*N+j, so offset = (k - i) * N:
    //
    //        Q_k = sum_i Rotate(rowK, (k - i) * N)   for i in [0, N).
    //
    //    i == k contributes the unrotated rowK term (offset 0).
    bgv_batched::Ciphertext BroadcastRowVerticallyImpl(const bgv_batched::Ciphertext& B, std::size_t k) const {
        auto row_k = bgv_batched::mask_mult(*ctx_, B, row_masks_[k]);

        bgv_batched::Ciphertext acc = row_k; // offset 0 (i == k): no rotation call
        for (std::size_t i = 0; i < n_; ++i) {
            if (i == k)
                continue;
            int32_t offset = (static_cast<int32_t>(k) - static_cast<int32_t>(i)) * static_cast<int32_t>(n_);
            auto rotated = bgv_batched::rotateToOffset(*ctx_, row_k, offset);
            acc = bgv_batched::add(*ctx_, acc, rotated);
        }
        return bgv_batched::mask_mult(*ctx_, acc, valid_region_mask_);
    }

    // Combines `terms` with encrypted_or in a balanced binary tree
    // (ceil(log2 terms.size()) sequential levels), matching
    // BGVBackend::or_reduce_tree (bgv_backend.hpp) -- see BGVBatchedMatmulDepth.
    bgv_batched::Ciphertext OrReduceTree(std::vector<bgv_batched::Ciphertext> terms) const {
        if (terms.empty())
            throw std::invalid_argument("BGVBatchedBackend::OrReduceTree: empty input");
        while (terms.size() > 1) {
            std::vector<bgv_batched::Ciphertext> next;
            next.reserve((terms.size() + 1) / 2);
            for (std::size_t i = 0; i + 1 < terms.size(); i += 2)
                next.push_back(bgv_batched::encrypted_or(*ctx_, terms[i], terms[i + 1]));
            if (terms.size() % 2 == 1)
                next.push_back(std::move(terms.back()));
            terms = std::move(next);
        }
        return terms[0];
    }

    // C = OR_{k=0}^{N-1} ( L_k AND Q_k ), where L_k[i,j] = A[i,k] and
    // Q_k[i,j] = B[k,j] (broadcast so that L_k AND Q_k, at slot (i,j),
    // equals A[i,k] * B[k,j] for every output position simultaneously).
    BGVBatchedMatrix BooleanMatrixMultiplyPacked(const BGVBatchedMatrix& A, const BGVBatchedMatrix& B) const {
        std::vector<bgv_batched::Ciphertext> terms;
        terms.reserve(n_);
        for (std::size_t k = 0; k < n_; ++k) {
            auto L_k = BroadcastColumnHorizontallyImpl(A.ciphertext(), k);
            auto Q_k = BroadcastRowVerticallyImpl(B.ciphertext(), k);
            terms.push_back(bgv_batched::encrypted_and(*ctx_, L_k, Q_k));
        }
        auto reduced = OrReduceTree(std::move(terms));
        return BGVBatchedMatrix(std::move(reduced), n_);
    }

    bgv_batched::Context* ctx_; // non-owning
    std::size_t n_;

    std::vector<lbcrypto::Plaintext> column_masks_; // column_masks_[k] = M^col_k
    std::vector<lbcrypto::Plaintext> row_masks_;     // row_masks_[k]    = M^row_k
    lbcrypto::Plaintext valid_region_mask_;          // 1 on [0, N*N), 0 elsewhere
};

} // namespace btc

#endif // TC_WITH_BGV_BATCHED
