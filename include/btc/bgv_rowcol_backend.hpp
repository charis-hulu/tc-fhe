#pragma once

#ifdef TC_WITH_BGV_BATCHED

// ── BGV row/column-packed Boolean matrix multiplication (OpenMP) ────────────
//
// A THIRD BGV backend alongside bgv_backend.hpp (one ciphertext per matrix
// element) and bgv_batched_backend.hpp (one ciphertext per whole N x N
// matrix). This one packs one ciphertext per matrix ROW and one ciphertext
// per matrix COLUMN, so a dot product C[i][j] = OR_k(A[i][k] AND B[k][j])
// is a single slot-wise EvalMult between an already-aligned row and column,
// no per-element rotation needed to bring A[i][k] and B[k][j] into the same
// slot. This file does not modify, depend on being enabled alongside, or
// replace bgv_backend.hpp / bgv_batched_backend.hpp -- all three are
// independently selectable backends satisfying the same Backend concept
// (see backend.hpp).
//
// Builds on top of bgv_batched.hpp's crypto-context primitives (Context,
// Ciphertext, rotateToOffset's verified rotation-sign convention,
// encrypt_packed/decrypt_packed_bits) rather than re-deriving them.
//
// ── Why NOT arithmetic RotateAndSum ──────────────────────────────────────
//
// The transitive-closure algorithm (algorithms.hpp, sum_powers_recursive)
// requires a BOOLEAN semiring matmul: C[i][j] = OR_k(A[i][k] AND B[k][j]).
// A naive packed dot product computed as
//
//     products = EvalMult(row, col)          // A[i][k] * B[k][j] per slot
//     cij      = RotateAndSum(products)       // sum_k A[i][k] * B[k][j]
//
// computes a WALK COUNT (how many k satisfy both terms), not a Boolean
// existence result. Fed into sum_powers_recursive's repeated squaring, that
// walk count would (a) stop being idempotent 0/1, so the recursion no
// longer computes transitive closure at all, and (b) risk silently
// exceeding the plaintext modulus t after enough rounds, since t is only
// sized here assuming {0,1} semantics. This file therefore replaces
// "rotate-and-sum" with "rotate-and-OR-reduce" (SumOR below), using the
// same {0,1}-ring OR identity a+b-ab already used throughout this project
// (bgv.hpp, bgv_batched.hpp).
//
// ── batch_size == N, and why EvalRotate still needs masking ──────────────
//
// This backend requires the crypto context's packed batch size to equal
// EXACTLY N (see BGVRowColBackend's constructor check) so a rotation
// schedule of just log2(N) steps can be reasoned about at all. It is
// tempting to assume OpenFHE replicates a size-N logical block across the
// whole ring when SetBatchSize(N) is smaller than the ring's natural slot
// count, making a plain EvalRotate(ct, shift) behave as an exact cyclic
// rotation modulo N for free -- THIS IS WRONG, verified empirically (an
// earlier version of this file assumed it and produced silently incorrect
// results, e.g. an OR of two witnesses decrypting to 2 instead of 1).
// OpenFHE's packed encoding at batch_size < ring capacity is ZERO-PADDED,
// not replicated: encrypting N values leaves every slot from N up to the
// ring's full ~ring_dim/2 capacity as an honest zero, and EvalRotate(ct, k)
// is a plain physical rotation over that entire (huge) slot range -- it
// does NOT wrap the content back into slot 0 when it falls off the end of
// our logical N-slot window; it just walks off into the zero padding.
//
// So a correct "rotate the logical N-slot window by `shift`, wrapping
// within N" (RotateModN below) has to be built from two physical
// rotations whose valid ranges are disjoint and together cover [0, N):
//
//   lo = EvalRotate(ct, shift)        -- correct on slots [0, N-shift)
//   hi = EvalRotate(ct, shift - N)    -- correct on slots [N-shift, N)
//   combined = EvalAdd(lo, hi)        -- free; correct on all of [0, N)...
//
// ...but NOT clean beyond slot N: `hi`'s invalid region (for slots outside
// [N-shift, N)) lands close to the window (around slot N to 2N-shift) --
// close enough that the NEXT step's rotation reads it right back into the
// window, corrupting later steps (this is exactly the same "wrap-around
// leaking into the matrix region" hazard bgv_batched_backend.hpp's
// valid_region_mask_ defends against, at whole-matrix scale instead of a
// single log-doubling reduction). So `combined` must be re-masked down to
// exactly [0, N) after every step -- see valid_region_mask_ and RotateModN.
// That mask is a ciphertext-PLAINTEXT EvalMult, which (like the repacking
// MaskMult below) costs one multiplicative-depth level in this OpenFHE
// build's default scaling technique -- so each SumOR step costs TWO
// EvalMults (one for the OR, one for the re-mask), not one; see
// BGVRowColMatmulDepth. N must still be a power of two (SumOR's
// log-doubling schedule assumes it to fully mix all N slots in log2(N)
// steps), and padding to the next power of two is deliberately NOT
// implemented (the caller must supply power-of-two N; see IsPowerOfTwo).
//
// ── Nested-OpenMP note ────────────────────────────────────────────────────
//
// The linked OpenFHE build (see OpenFHEConfig.cmake: OpenFHE_OPENMP=ON) has
// DCRTPolyImpl (the RNS polynomial type backing every ciphertext) run its
// own per-RNS-tower arithmetic under `#pragma omp parallel for
// num_threads(OpenFHEParallelControls.GetThreadLimit(size))` (e.g.
// EvalAdd/EvalSub, see dcrtpoly-impl.h), and OpenFHE's ParallelControls
// constructor leaves nested-parallelism disabling calls commented out
// (parallel.h). Left unchecked, wrapping EvalMult/EvalAdd calls inside our
// own outer `#pragma omp parallel for` would spawn P outer threads each
// spawning their own inner tower-parallel threads -- genuine
// oversubscription, not a hypothetical. BGVRowColBackend's constructor
// therefore caps `omp_set_max_active_levels(1)` once at construction (the
// "application initialization" point), so nested OpenFHE-internal parallel
// regions execute serially instead of spawning more threads.
//
// ─────────────────────────────────────────────────────────────────────────

#include "bgv_batched.hpp"
#include "matrix.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace btc {

// ── Power-of-two helpers ──────────────────────────────────────────────────

inline bool IsPowerOfTwo(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Rotation offsets required by SumOR's log-doubling reduction. For each
// power-of-two shift in {1, 2, 4, ..., N/2}, RotateModN needs BOTH the
// forward rotation `shift` AND the wrap-compensating rotation `shift - N`
// (see this file's header comment for why a single EvalRotate isn't
// enough at batch_size == N). Generated once and installed as rotation
// keys before any matmul (see BGVRowColBackend's constructor) -- never
// regenerated inside a matmul call.
inline std::vector<int32_t> BuildRequiredRotationIndices(std::size_t n) {
    std::vector<int32_t> indices;
    for (std::size_t shift = 1; shift < n; shift *= 2) {
        indices.push_back(static_cast<int32_t>(shift));
        indices.push_back(static_cast<int32_t>(shift) - static_cast<int32_t>(n));
    }
    return indices;
}

// ── Multiplicative depth ──────────────────────────────────────────────────
//
// Each SumOR step costs TWO EvalMults, not one: one for the Boolean OR
// (a+b-ab's a*b term) and one for RotateModN's re-mask back to [0, N) (a
// ciphertext-PLAINTEXT EvalMult, which in this OpenFHE build's default
// scaling technique costs a level exactly like ciphertext-ciphertext
// EvalMult -- see this file's header comment). So SumOR costs
// 2*ceil(log2 N) levels, not ceil(log2 N).
//
// On top of that, the row/col output ciphertext isn't cij itself -- it's
// cij AFTER the repacking MaskMult (isolates cij into its target slot
// before EvalAdd-ing it into a row/column accumulator; see MaskMult),
// which costs one more level for the same ciphertext-plaintext reason. The
// SAME undercounting pitfall (assuming ciphertext-plaintext EvalMult is
// "free") is already documented for the full-matrix-packed backend's
// mask_mult, see bgv_batched_backend.hpp's BGVBatchedMatmulDepth comment;
// an earlier version of this function undercounted BOTH the SumOR re-mask
// and this repacking mask and produced non-Boolean decrypted values at
// runtime (ran out of levels mid-circuit) before being corrected.
//
// So one full matmul costs:
//   1 (slot-wise AND) + 2*ceil(log2 N) (SumOR: OR + re-mask per step)
//   + 1 (repacking MaskMult) = 2 + 2*ceil(log2 N)
//
// sum_powers_recursive then contributes the same ceil(log2 T) *
// (depth(matmul) + 1) recursion-depth term as every other backend in this
// project (see EstimateBGVDepth / EstimateBGVBatchedDepth).
inline int BGVRowColMatmulDepth(std::size_t n) {
    if (!IsPowerOfTwo(n))
        throw std::invalid_argument("BGVRowColMatmulDepth: N must be a power of two, got " +
                                     std::to_string(n));
    int levels = 0;
    for (std::size_t shift = 1; shift < n; shift *= 2)
        ++levels;
    return 2 + 2 * levels;
}

inline uint32_t EstimateBGVRowColDepth(std::size_t numberOfNodes, int T) {
    if (T < 1)
        throw std::invalid_argument("EstimateBGVRowColDepth: T must be >= 1");
    int levels = 0;
    int remaining = T;
    while (remaining > 1) {
        remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
        ++levels;
    }
    return std::max<uint32_t>(1, static_cast<uint32_t>(levels * (BGVRowColMatmulDepth(numberOfNodes) + 1)));
}

// ── Nested-OpenMP level override ─────────────────────────────────────────
//
// Defaults to 1 (see the "Nested-OpenMP note" above -- serializes OpenFHE's
// internal per-RNS-tower parallel regions so they don't spawn additional
// threads underneath this backend's own outer parallel-for). Set
// TC_OMP_MAX_ACTIVE_LEVELS=2 in the environment to allow one level of
// nesting instead.
//
// This knob does NOT create extra core budget -- oversubscription is a
// per-node thing (bounded by that node's physical core count), not a
// per-job one, so running across more MPI ranks/nodes doesn't buy any
// headroom here: each rank is still confined to its own node's cores
// regardless of how many other nodes are in the job. Raising this to 2
// without also lowering OMP_NUM_THREADS (so outer x inner stays within one
// node's actual core count) causes real P-outer x Q-inner oversubscription,
// not a hypothetical one.
inline int GetConfiguredMaxActiveLevels() {
    if (const char* env = std::getenv("TC_OMP_MAX_ACTIVE_LEVELS")) {
        try {
            int levels = std::stoi(env);
            if (levels >= 1) return levels;
        } catch (...) {
        }
    }
    return 1;
}

// ── Matrix multiplication implementation selector ────────────────────────
//
// All three compute the identical Boolean-semiring result; they differ only
// in ciphertext-op count and OpenMP usage, so benchmarks can hold N/T/crypto
// parameters fixed and vary only this. See docs/bgv_rowcol.md.
enum class BGVMatMulBackend {
    // No OpenMP. Never materializes an N x N grid of live ciphertexts --
    // only 2N row/column accumulators plus one temporary cij at a time.
    Serial,
    // Race-free two-pass OpenMP: Stage 1 parallelizes over output rows,
    // Stage 2 (separately) parallelizes over output columns. Each thread
    // owns one whole accumulator, so no two threads ever write the same
    // ciphertext -- but every BooleanDotProduct is computed twice (once per
    // stage). Deliberate first-cut correctness-over-efficiency tradeoff.
    OpenMPTwoPass,
    // Computes every cij exactly once: threads parallelize over output rows
    // and accumulate into thread-local column accumulators, reduced into
    // the final columns after the parallel region. Uses O(threads * N)
    // temporary ciphertext accumulators instead of O(N) -- see MatMulOpenMPOptimized.
    OpenMPOptimized,
};

// ── Packed matrix representation ─────────────────────────────────────────
//
// rows[i] = [M[i][0], M[i][1], ..., M[i][N-1]]  (one ciphertext per row)
// cols[j] = [M[0][j], M[1][j], ..., M[N-1][j]]  (one ciphertext per column)
//
// Both must represent the same logical matrix -- BGVRowColBackend::Decrypt
// cross-checks this rather than trusting rows alone (see its doc comment).
class PackedBooleanMatrix {
public:
    PackedBooleanMatrix() : n_(0) {}
    PackedBooleanMatrix(std::size_t n, std::vector<bgv_batched::Ciphertext> rows,
                         std::vector<bgv_batched::Ciphertext> cols)
        : n_(n), rows_(std::move(rows)), cols_(std::move(cols)) {}

    std::size_t dim() const noexcept { return n_; }

    const bgv_batched::Ciphertext& row(std::size_t i) const { return rows_.at(i); }
    const bgv_batched::Ciphertext& col(std::size_t j) const { return cols_.at(j); }
    const std::vector<bgv_batched::Ciphertext>& rows() const { return rows_; }
    const std::vector<bgv_batched::Ciphertext>& cols() const { return cols_; }

private:
    std::size_t n_;
    std::vector<bgv_batched::Ciphertext> rows_;
    std::vector<bgv_batched::Ciphertext> cols_;
};

// BGVRowColBackend -- satisfies the Backend concept (see backend.hpp) using
// PackedBooleanMatrix. One backend instance is built for one specific N and
// one specific BGVMatMulBackend variant (masks, rotation keys, and the
// nested-OpenMP cap are all one-time construction-time state, matching the
// "one backend instance per N" convention already used by
// BGVBackend/BGVBatchedBackend).
//
// Usage:
//   auto params = btc::bgv_batched::Params{};
//   params.multiplicative_depth = btc::EstimateBGVRowColDepth(N, T);
//   params.batch_size = N;   // required -- see the batch_size == N invariant above
//   auto ctx = btc::bgv_batched::setup(params);
//
//   btc::BGVRowColBackend backend(ctx, N, btc::BGVMatMulBackend::OpenMPTwoPass);
//   auto enc_A = backend.Encrypt(plain_A);
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = backend.Decrypt(enc_S);
class BGVRowColBackend {
public:
    using matrix_type = PackedBooleanMatrix;

    // Snapshot of operation counters -- returned by value from counters()
    // (see AtomicCounters below for the mutable, thread-safe internal
    // storage these are read from).
    struct OpCounters {
        std::size_t cc_mult = 0;      // ciphertext-ciphertext EvalMult
        std::size_t cp_mult = 0;      // ciphertext-plaintext EvalMult (one-hot masks)
        std::size_t add = 0;          // EvalAdd
        std::size_t sub = 0;          // EvalSub
        std::size_t rotate = 0;       // EvalRotate
        std::size_t dot_products = 0; // BooleanDotProduct calls
        std::size_t sumor_calls = 0;  // SumOR calls
    };

    // `generate_rotation_keys` calls OpenFHE's EvalRotateKeyGen, which
    // requires ctx.secret_key -- fine for every existing (single-process)
    // caller, since ctx always carries the secret key there. The MPI
    // distributed backend (bgv_distributed_rowcol_backend.hpp) is the one
    // exception: compute ranks deliberately hold a secret-key-less Context
    // (deserialized from bgv_mpi::DeserializeContextBundle, see that file's
    // "secret key must not be required by compute ranks" requirement) whose
    // rotation keys were generated ONCE on the root rank and shipped over
    // MPI as already-installed EvalAutomorphismKey state -- regenerating
    // them here would both crash (null secret_key) and be redundant.
    enum class RotationKeys {
        GenerateHere,     // default: call generate_rotation_keys (needs ctx.secret_key)
        AssumeAlreadyInstalled, // skip generation -- caller guarantees keys already exist in ctx
    };

    BGVRowColBackend(bgv_batched::Context& ctx, std::size_t n,
                      BGVMatMulBackend variant = BGVMatMulBackend::OpenMPTwoPass,
                      RotationKeys rotation_keys = RotationKeys::GenerateHere)
        : ctx_(&ctx), n_(n), variant_(variant) {
        if (!IsPowerOfTwo(n_))
            throw std::invalid_argument(
                "BGVRowColBackend: N must be a power of two, got " + std::to_string(n_) +
                " -- this backend does not implement padding to the next power of two "
                "(see IsPowerOfTwo/BuildRequiredRotationIndices doc comments).");

        const uint32_t slots = bgv_batched::available_slots(*ctx_);
        if (slots != n_)
            throw std::invalid_argument(
                "BGVRowColBackend: crypto context batch size (" + std::to_string(slots) +
                ") must equal N (" + std::to_string(n_) + ") exactly. RotateModN's rotation "
                "schedule and the one-hot/valid-region masks are all sized for a logical [0,N) "
                "window -- build the context with Params::batch_size = N (see "
                "bgv_rowcol_backend.hpp's file header for the batch_size == N discussion).");

        BuildOneHotMasks();
        if (rotation_keys == RotationKeys::GenerateHere)
            bgv_batched::generate_rotation_keys(*ctx_, BuildRequiredRotationIndices(n_));

        // See the "Nested-OpenMP note" in this file's header comment: cap
        // active OpenMP nesting once, here, before any parallel region runs,
        // so OpenFHE's internal per-RNS-tower parallel loops execute
        // serially inside our own outer parallel-for instead of spawning
        // additional threads on top of it. Overridable via
        // TC_OMP_MAX_ACTIVE_LEVELS -- see GetConfiguredMaxActiveLevels.
        omp_set_dynamic(0);
        omp_set_max_active_levels(GetConfiguredMaxActiveLevels());
    }

    std::size_t dim(const PackedBooleanMatrix& A) const { return A.dim(); }

    PackedBooleanMatrix copy(const PackedBooleanMatrix& A) const {
        // Ciphertext handles are shared_ptr-backed and immutable under
        // homomorphic ops (Eval* always return a new Ciphertext), so a
        // shallow copy of every row/column handle is already a safe,
        // independent logical copy (same reasoning as BGVBackend::copy /
        // BGVBatchedMatrix's copy semantics).
        return PackedBooleanMatrix(A.dim(), A.rows(), A.cols());
    }

    PackedBooleanMatrix or_mat(const PackedBooleanMatrix& A, const PackedBooleanMatrix& B) const {
        RequireSameDim(A, B, "or_mat");
        std::vector<bgv_batched::Ciphertext> rows(n_), cols(n_);
        for (std::size_t i = 0; i < n_; ++i)
            rows[i] = EvalBooleanOR(A.row(i), B.row(i));
        for (std::size_t j = 0; j < n_; ++j)
            cols[j] = EvalBooleanOR(A.col(j), B.col(j));
        return PackedBooleanMatrix(n_, std::move(rows), std::move(cols));
    }

    // C[i][j] = OR_k(A[i][k] AND B[k][j]), dispatched to whichever variant
    // this backend instance was constructed with (see BGVMatMulBackend).
    PackedBooleanMatrix matmul(const PackedBooleanMatrix& A, const PackedBooleanMatrix& B) const {
        RequireSameDim(A, B, "matmul");
        switch (variant_) {
            case BGVMatMulBackend::Serial:
                return MatMulSerial(A, B);
            case BGVMatMulBackend::OpenMPTwoPass:
                return MatMulOpenMPTwoPass(A, B);
            case BGVMatMulBackend::OpenMPOptimized:
                return MatMulOpenMPOptimized(A, B);
        }
        throw std::logic_error("BGVRowColBackend::matmul: unknown BGVMatMulBackend variant");
    }

    // ── Packing / encryption convenience wrappers ────────────────────────

    PackedBooleanMatrix Encrypt(const BoolMatrix& plain) const {
        if (plain.rows() != n_ || plain.cols() != n_)
            throw std::invalid_argument("BGVRowColBackend::Encrypt: dimension mismatch");

        std::vector<bgv_batched::Ciphertext> rows(n_), cols(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            std::vector<int64_t> values(n_);
            for (std::size_t k = 0; k < n_; ++k)
                values[k] = plain.get(i, k) ? 1 : 0;
            rows[i] = bgv_batched::encrypt_packed(*ctx_, values);
        }
        for (std::size_t j = 0; j < n_; ++j) {
            std::vector<int64_t> values(n_);
            for (std::size_t k = 0; k < n_; ++k)
                values[k] = plain.get(k, j) ? 1 : 0;
            cols[j] = bgv_batched::encrypt_packed(*ctx_, values);
        }
        return PackedBooleanMatrix(n_, std::move(rows), std::move(cols));
    }

    // Decrypts using the row-packed representation, then cross-checks every
    // entry against the column-packed representation -- catches a
    // row/column desync bug (e.g. a matmul path that updates rows but not
    // columns) as a loud runtime_error instead of silently returning
    // internally-inconsistent results.
    BoolMatrix Decrypt(const PackedBooleanMatrix& enc) const {
        if (enc.dim() != n_)
            throw std::invalid_argument("BGVRowColBackend::Decrypt: dimension mismatch");

        BoolMatrix result(n_, n_, false);
        for (std::size_t i = 0; i < n_; ++i) {
            auto bits = bgv_batched::decrypt_packed_bits(*ctx_, enc.row(i), n_);
            for (std::size_t j = 0; j < n_; ++j)
                result.set(i, j, bits[j]);
        }
        for (std::size_t j = 0; j < n_; ++j) {
            auto bits = bgv_batched::decrypt_packed_bits(*ctx_, enc.col(j), n_);
            for (std::size_t i = 0; i < n_; ++i) {
                if (bits[i] != result.get(i, j))
                    throw std::runtime_error(
                        "BGVRowColBackend::Decrypt: row/column representations disagree at (" +
                        std::to_string(i) + "," + std::to_string(j) + ") -- rows say " +
                        std::to_string(result.get(i, j)) + ", columns say " + std::to_string(bits[i]));
            }
        }
        return result;
    }

    OpCounters counters() const {
        OpCounters c;
        c.cc_mult = counters_.cc_mult.load(std::memory_order_relaxed);
        c.cp_mult = counters_.cp_mult.load(std::memory_order_relaxed);
        c.add = counters_.add.load(std::memory_order_relaxed);
        c.sub = counters_.sub.load(std::memory_order_relaxed);
        c.rotate = counters_.rotate.load(std::memory_order_relaxed);
        c.dot_products = counters_.dot_products.load(std::memory_order_relaxed);
        c.sumor_calls = counters_.sumor_calls.load(std::memory_order_relaxed);
        return c;
    }

    void reset_counters() const {
        counters_.cc_mult.store(0, std::memory_order_relaxed);
        counters_.cp_mult.store(0, std::memory_order_relaxed);
        counters_.add.store(0, std::memory_order_relaxed);
        counters_.sub.store(0, std::memory_order_relaxed);
        counters_.rotate.store(0, std::memory_order_relaxed);
        counters_.dot_products.store(0, std::memory_order_relaxed);
        counters_.sumor_calls.store(0, std::memory_order_relaxed);
    }

    // ── Layout-test entry points ──────────────────────────────────────────
    //
    // Exposed so tests/test_bgv_rowcol_matmul.cpp can independently verify
    // EvalBooleanOR / SumOR / BooleanDotProduct in isolation, before they're
    // used inside a full matmul -- not part of the Backend concept surface
    // used by algorithms.hpp.

    bgv_batched::Ciphertext EvalBooleanORPublic(const bgv_batched::Ciphertext& a,
                                                 const bgv_batched::Ciphertext& b) const {
        return EvalBooleanOR(a, b);
    }

    bgv_batched::Ciphertext SumORPublic(const bgv_batched::Ciphertext& input) const { return SumOR(input); }

    bgv_batched::Ciphertext BooleanDotProductPublic(const bgv_batched::Ciphertext& row,
                                                     const bgv_batched::Ciphertext& col) const {
        return BooleanDotProduct(row, col);
    }

    // Exposed additionally for bgv_distributed_rowcol_backend.hpp: the
    // distributed backend repacks a locally-computed cij into row/column
    // pieces exactly the way MatMulSerial/MatMulOpenMP* do (one-hot mask,
    // ciphertext-plaintext EvalMult), just across MPI ranks instead of
    // OpenMP threads -- reusing this rather than re-deriving the one-hot
    // mask logic in a second place.
    bgv_batched::Ciphertext MaskMultPublic(const bgv_batched::Ciphertext& ct, std::size_t pos) const {
        return MaskMult(ct, pos);
    }

    // Exposed for the same reason as MaskMultPublic: the distributed
    // backend accumulates row/column pieces (locally, and again after
    // combining pieces received over MPI) using the exact same counted
    // EvalAdd this file's own MatMulSerial/MatMulOpenMP* use, rather than
    // calling ctx.crypto_context->EvalAdd directly and losing the op count.
    bgv_batched::Ciphertext EvalAddPublic(const bgv_batched::Ciphertext& a, const bgv_batched::Ciphertext& b) const {
        return EvalAddCounted(a, b);
    }

    // Exposed so the distributed backend can build its own zero-initialized
    // row/column accumulators the same way MatMulSerial/MatMulOpenMP* do.
    bgv_batched::Ciphertext EncryptZeroPublic() const { return EncryptZero(); }

    // Read-only access to N -- the distributed backend needs the same N
    // this instance was constructed with in order to size its local blocks
    // and validate the 2D process grid.
    std::size_t n() const noexcept { return n_; }

private:
    // Atomics so MatMulOpenMPTwoPass/MatMulOpenMPOptimized can increment
    // these concurrently from multiple OpenMP threads without a data race.
    // (counters() above copies out plain std::size_t snapshots since
    // std::atomic itself isn't copyable/returnable by value.)
    struct AtomicCounters {
        std::atomic<std::size_t> cc_mult{0};
        std::atomic<std::size_t> cp_mult{0};
        std::atomic<std::size_t> add{0};
        std::atomic<std::size_t> sub{0};
        std::atomic<std::size_t> rotate{0};
        std::atomic<std::size_t> dot_products{0};
        std::atomic<std::size_t> sumor_calls{0};
    };

    static void RequireSameDim(const PackedBooleanMatrix& A, const PackedBooleanMatrix& B, const char* who) {
        if (A.dim() != B.dim())
            throw std::invalid_argument(std::string("BGVRowColBackend::") + who + ": dimension mismatch");
    }

    // One-hot masks: mask[pos] = [0,...,1,...,0] with the 1 at slot `pos`.
    // Plus the valid-region mask ([1]*N, used by RotateModN to re-clean the
    // logical [0,N) window after every rotate-combine -- see this file's
    // header comment). Both built once (constructor) and reused for every
    // matmul call -- never rebuilt per-round.
    void BuildOneHotMasks() {
        one_hot_masks_.reserve(n_);
        for (std::size_t pos = 0; pos < n_; ++pos) {
            std::vector<int64_t> values(n_, 0);
            values[pos] = 1;
            one_hot_masks_.push_back(bgv_batched::make_packed_plaintext(*ctx_, values));
        }
        valid_region_mask_ = bgv_batched::make_packed_plaintext(*ctx_, std::vector<int64_t>(n_, 1));
    }

    bgv_batched::Ciphertext EncryptZero() const {
        std::vector<int64_t> zeros(n_, 0);
        return bgv_batched::encrypt_packed(*ctx_, zeros);
    }

    // OR(a,b) = a + b - a*b. Costs one EvalMult (the a*b term) -- NOT free
    // like EvalAdd/EvalRotate, see this file's header comment.
    bgv_batched::Ciphertext EvalBooleanOR(const bgv_batched::Ciphertext& a, const bgv_batched::Ciphertext& b) const {
        auto sum = ctx_->crypto_context->EvalAdd(a, b);
        auto product = ctx_->crypto_context->EvalMult(a, b);
        auto result = ctx_->crypto_context->EvalSub(sum, product);
        counters_.add.fetch_add(1, std::memory_order_relaxed);
        counters_.cc_mult.fetch_add(1, std::memory_order_relaxed);
        counters_.sub.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    // Rotates `ct`'s logical [0,N) window left by `shift`, wrapping within
    // N (NOT within OpenFHE's full physical slot count) -- see this file's
    // header comment for why a single EvalRotate can't do this at
    // batch_size == N. Combines the in-range rotation with the
    // wrap-compensating rotation (shift - N), then re-masks to [0,N) so the
    // result is clean for the NEXT rotation in SumOR's schedule. Costs one
    // EvalMult (the re-mask) and two EvalRotate calls; requires rotation
    // keys for both `shift` and `shift - N` (see BuildRequiredRotationIndices).
    bgv_batched::Ciphertext RotateModN(const bgv_batched::Ciphertext& ct, int32_t shift) const {
        auto lo = ctx_->crypto_context->EvalRotate(ct, shift);
        auto hi = ctx_->crypto_context->EvalRotate(ct, shift - static_cast<int32_t>(n_));
        counters_.rotate.fetch_add(2, std::memory_order_relaxed);
        auto combined = ctx_->crypto_context->EvalAdd(lo, hi);
        counters_.add.fetch_add(1, std::memory_order_relaxed);
        auto cleaned = ctx_->crypto_context->EvalMult(combined, valid_region_mask_);
        counters_.cp_mult.fetch_add(1, std::memory_order_relaxed);
        return cleaned;
    }

    // SIMD Boolean OR-reduction of all N slots via a log-doubling
    // rotate-and-OR schedule (shifts 1, 2, 4, ..., N/2), replicating the
    // final OR result across every slot -- NOT arithmetic RotateAndSum, see
    // this file's header comment for why. Costs 2*ceil(log2 N) EvalMult
    // calls (one for the OR, one for RotateModN's re-mask, per step).
    bgv_batched::Ciphertext SumOR(const bgv_batched::Ciphertext& input) const {
        auto result = input;
        for (std::size_t shift = 1; shift < n_; shift *= 2) {
            auto rotated = RotateModN(result, static_cast<int32_t>(shift));
            result = EvalBooleanOR(result, rotated);
        }
        counters_.sumor_calls.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    // BooleanDotProduct(A.row(i), B.col(j)) = OR_k(A[i][k] AND B[k][j]),
    // replicated across all N slots (SumOR's output form). Costs
    // 1 + 2*ceil(log2 N) EvalMult calls total (see BGVRowColMatmulDepth).
    bgv_batched::Ciphertext BooleanDotProduct(const bgv_batched::Ciphertext& packedRow,
                                               const bgv_batched::Ciphertext& packedCol) const {
        auto products = ctx_->crypto_context->EvalMult(packedRow, packedCol);
        counters_.cc_mult.fetch_add(1, std::memory_order_relaxed);
        auto replicated = SumOR(products);
        counters_.dot_products.fetch_add(1, std::memory_order_relaxed);
        return replicated;
    }

    // Ciphertext-plaintext multiplication with a one-hot mask -- isolates
    // cij (currently replicated across all N slots) down to a single slot,
    // ready to be EvalAdd-ed into a row or column accumulator. Must use
    // ciphertext-PLAINTEXT EvalMult, not ciphertext-ciphertext.
    bgv_batched::Ciphertext MaskMult(const bgv_batched::Ciphertext& ct, std::size_t pos) const {
        auto result = ctx_->crypto_context->EvalMult(ct, one_hot_masks_[pos]);
        counters_.cp_mult.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    bgv_batched::Ciphertext EvalAddCounted(const bgv_batched::Ciphertext& a, const bgv_batched::Ciphertext& b) const {
        counters_.add.fetch_add(1, std::memory_order_relaxed);
        return ctx_->crypto_context->EvalAdd(a, b);
    }

    // ── Serial reference implementation ──────────────────────────────────
    //
    // Never materializes an N x N grid of ciphertexts -- only 2N
    // accumulators (out_rows/out_cols) plus one temporary cij live at a
    // time, per the task's explicit memory requirement.
    PackedBooleanMatrix MatMulSerial(const PackedBooleanMatrix& A, const PackedBooleanMatrix& B) const {
        auto zero = EncryptZero();
        std::vector<bgv_batched::Ciphertext> out_rows(n_, zero), out_cols(n_, zero);

        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                auto cij = BooleanDotProduct(A.row(i), B.col(j));
                out_rows[i] = EvalAddCounted(out_rows[i], MaskMult(cij, j));
                out_cols[j] = EvalAddCounted(out_cols[j], MaskMult(cij, i));
            }
        }
        return PackedBooleanMatrix(n_, std::move(out_rows), std::move(out_cols));
    }

    // ── Race-free two-pass OpenMP ─────────────────────────────────────────
    //
    // Stage 1 parallelizes over output ROWS; each iteration writes only to
    // out_rows[i], so no two threads ever touch the same accumulator.
    // Stage 2 does the same for columns. This recomputes every
    // BooleanDotProduct twice (once per stage) -- a deliberate first-cut
    // tradeoff (see BGVMatMulBackend::OpenMPTwoPass's doc comment); prefer
    // OpenMPOptimized when that recomputation matters for benchmarking.
    PackedBooleanMatrix MatMulOpenMPTwoPass(const PackedBooleanMatrix& A, const PackedBooleanMatrix& B) const {
        std::vector<bgv_batched::Ciphertext> out_rows(n_), out_cols(n_);
        auto zero = EncryptZero();
        const auto n = static_cast<std::int64_t>(n_);

#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) {
            auto acc = zero;
            for (std::size_t j = 0; j < n_; ++j) {
                auto cij = BooleanDotProduct(A.row(static_cast<std::size_t>(i)), B.col(j));
                acc = EvalAddCounted(acc, MaskMult(cij, j));
            }
            out_rows[static_cast<std::size_t>(i)] = std::move(acc);
        }

#pragma omp parallel for schedule(static)
        for (std::int64_t j = 0; j < n; ++j) {
            auto acc = zero;
            for (std::size_t i = 0; i < n_; ++i) {
                auto cij = BooleanDotProduct(A.row(i), B.col(static_cast<std::size_t>(j)));
                acc = EvalAddCounted(acc, MaskMult(cij, i));
            }
            out_cols[static_cast<std::size_t>(j)] = std::move(acc);
        }

        return PackedBooleanMatrix(n_, std::move(out_rows), std::move(out_cols));
    }

    // ── Optimized OpenMP: every cij computed exactly once ────────────────
    //
    // Threads parallelize over output rows; each thread accumulates its row
    // result directly, and ALSO accumulates column contributions into its
    // own thread-local column accumulator array (no shared mutable state
    // touched by more than one thread during the parallel region). After
    // the region, thread-local column accumulators are reduced serially
    // into the final columns. Uses O(threads * N) temporary ciphertext
    // accumulators instead of two-pass's O(N) -- measure memory before
    // making this the default at large N (see docs/bgv_rowcol.md).
    PackedBooleanMatrix MatMulOpenMPOptimized(const PackedBooleanMatrix& A, const PackedBooleanMatrix& B) const {
        std::vector<bgv_batched::Ciphertext> out_rows(n_);
        auto zero = EncryptZero();
        const auto n = static_cast<std::int64_t>(n_);

        const int thread_count = std::max(1, omp_get_max_threads());
        std::vector<std::vector<bgv_batched::Ciphertext>> thread_cols(
            static_cast<std::size_t>(thread_count), std::vector<bgv_batched::Ciphertext>(n_, zero));

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < n; ++i) {
                auto row_acc = zero;
                for (std::size_t j = 0; j < n_; ++j) {
                    auto cij = BooleanDotProduct(A.row(static_cast<std::size_t>(i)), B.col(j));
                    row_acc = EvalAddCounted(row_acc, MaskMult(cij, j));
                    auto& slot = thread_cols[static_cast<std::size_t>(tid)][j];
                    slot = EvalAddCounted(slot, MaskMult(cij, static_cast<std::size_t>(i)));
                }
                out_rows[static_cast<std::size_t>(i)] = std::move(row_acc);
            }
        }

        std::vector<bgv_batched::Ciphertext> out_cols(n_, zero);
        for (std::size_t j = 0; j < n_; ++j)
            for (int t = 0; t < thread_count; ++t)
                out_cols[j] = EvalAddCounted(out_cols[j], thread_cols[static_cast<std::size_t>(t)][j]);

        return PackedBooleanMatrix(n_, std::move(out_rows), std::move(out_cols));
    }

    bgv_batched::Context* ctx_; // non-owning
    std::size_t n_;
    BGVMatMulBackend variant_;

    std::vector<lbcrypto::Plaintext> one_hot_masks_; // one_hot_masks_[pos] = M_pos
    lbcrypto::Plaintext valid_region_mask_;          // [1]*N -- see RotateModN
    mutable AtomicCounters counters_;
};

} // namespace btc

#endif // TC_WITH_BGV_BATCHED
