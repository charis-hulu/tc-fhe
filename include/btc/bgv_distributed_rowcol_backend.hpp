#pragma once

#ifdef TC_WITH_BGV_MPI

// ── Distributed BGV row/column-packed Boolean matrix multiplication ────────
// (MPI across nodes, OpenMP within a node) ─────────────────────────────────
//
// A FOURTH BGV backend, alongside bgv_backend.hpp (per-element),
// bgv_batched_backend.hpp (whole-matrix packing), and
// bgv_rowcol_backend.hpp (single-process row/column packing, OpenMP only).
// This file does not modify or replace any of those -- BGVRowColBackend in
// particular is reused directly (see below) rather than re-derived.
//
// ── Why this backend exists ────────────────────────────────────────────────
//
// bgv_rowcol_backend.hpp's OpenMPOptimized variant already established that
// the N^2 encrypted dot products C[i][j] = SumOR(A.row(i) (*) B.col(j)) are
// independent of each other. Independence within one process is exploitable
// with OpenMP threads; independence ACROSS the whole matrix is exploitable
// with MPI ranks across many compute nodes -- the row/column-packed layout
// is what makes this decomposition possible at all (the whole-matrix-packed
// backend, bgv_batched_backend.hpp, has no analogous per-cell independence:
// every output cell lives inside the SAME ciphertext).
//
// ── 2D process grid and the ownership invariant ─────────────────────────────
//
// P = Pr * Pc MPI ranks are arranged in a 2D Cartesian grid. Rank (r,c):
//   - owns (and keeps resident) every COMPLETE row ciphertext whose global
//     row index falls in this rank's row-block I_r;
//   - owns (and keeps resident) every COMPLETE column ciphertext whose
//     global column index falls in this rank's column-block J_c.
// Row blocks are replicated across every rank sharing the same process row
// (row_comm); column blocks are replicated across every rank sharing the
// same process column (col_comm). This is the SAME invariant the matrix
// must satisfy both BEFORE and AFTER every distributed matmul/or_mat call,
// so sum_powers_recursive (algorithms.hpp) can chain calls without any
// whole-matrix gather in between -- see DistributedMatMul/DistributedOR.
//
// ── What is reused from bgv_rowcol_backend.hpp, verbatim ──────────────────
//
// Every rank constructs its own LOCAL BGVRowColBackend instance (over the
// SAME shared CryptoContext, see bgv_mpi_serial.hpp) and calls its PUBLIC
// wrappers -- BooleanDotProductPublic (AND + SumOR, itself built from
// EvalBooleanOR + RotateModN, UNCHANGED) and MaskMultPublic (one-hot-mask
// repacking, UNCHANGED) -- for every locally-owned (i,j) pair. SumOR and
// RotateModN are not redesigned or reimplemented here, per the task's
// explicit constraint; this file only decides WHICH (i,j) pairs a given
// rank computes and how partial row/column pieces get combined across
// ranks afterward.
//
// ── Secret key handling ─────────────────────────────────────────────────────
//
// Rank 0 alone calls bgv_batched::setup() (which generates the secret key)
// and BGVRowColBackend's normal constructor path (which needs the secret
// key to call EvalRotateKeyGen once). Every other rank receives a
// SERIALIZED bundle (crypto context + public key + eval-mult key +
// rotation/automorphism keys, see bgv_mpi_serial.hpp) with NO secret key,
// and constructs its local BGVRowColBackend with
// RotationKeys::AssumeAlreadyInstalled so it never attempts to touch a null
// secret key. Only rank 0 can decrypt; see Decrypt below.
//
// ── Stage scope (see claude_distributed_bgv_rowcol_mpi.md) ─────────────────
//
// Stage 1 (correct distributed matmul + row/column assembly) and Stage 2
// (distributed bounded transitive closure via the Backend concept, no
// whole-matrix gather between recursion levels) are both implemented, as
// before. Of Stage 3's items, three are now also implemented:
//
//   - Communication batching: AssembleRows/AssembleCols/or_mat/
//     GatherToRankWithinGrid/ScatterFromRankToGrid all exchange ONE batched
//     message per (source, owner) pair or per broadcast root, instead of one
//     message per ciphertext (see bgv_mpi_serial.hpp's
//     SerializeCiphertextBatch/SendCiphertextBatch/RecvCiphertextBatch/
//     BroadcastCiphertextBatch and this file's AssembleOwnedBucket/
//     GatherOwnedToAll). Still not a true MPI_Alltoallv/Gatherv collective
//     (that remains a possible further step) -- this reduces MESSAGE COUNT,
//     the bottleneck this project's N=4/N=8 test sizes actually hit.
//   - or_mat owner-compute dedup: a single deterministic owner per
//     row/column computes EvalBooleanOR once and broadcasts it (batched),
//     instead of every rank sharing a replicated row/column block
//     recomputing the same OR redundantly (see or_mat's doc comment).
//   - Concurrent recursion branches: matmul_pair splits grid_'s world
//     communicator into two halves so the bounded-transitive-closure
//     recurrence's two mutually-independent matmul calls per even-T level
//     run concurrently on disjoint MPI ranks, not just sequentially on the
//     same ranks (see matmul_pair's doc comment and
//     DistributedSumPowersRecursive/DistributedBoundedTransitiveClosure,
//     which use it in place of algorithms.hpp's generic
//     sum_powers_recursive -- that file is deliberately left unmodified).
//
// Still deferred: nonblocking communication (everything above is still
// blocking send/recv/broadcast) and tiled/streaming repacking (the local
// dot-product phase still materializes the whole local C[i][j] block before
// assembly).
//
// ─────────────────────────────────────────────────────────────────────────

#include "bgv_batched.hpp"
#include "bgv_mpi_serial.hpp"
#include "bgv_rowcol_backend.hpp"
#include "matrix.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Debug trace gated by TC_BGV_MPI_DEBUG=1 -- prints "[rank R] <what>" to
// stderr, flushed immediately, from constructor/matmul checkpoints. Off by
// default (adds a getenv + branch per call site, negligible). Temporary
// diagnostic aid for isolating exactly where a distributed run stalls;
// candidate for removal once Stage 1/2 are established as reliably fast at
// the rank counts this project tests.
#define TC_BGV_MPI_TRACE(rank, ...)                                                                          \
    do {                                                                                                      \
        if (std::getenv("TC_BGV_MPI_DEBUG")) {                                                                \
            std::fprintf(stderr, "[rank %d] ", (rank));                                                       \
            std::fprintf(stderr, __VA_ARGS__);                                                                 \
            std::fprintf(stderr, "\n");                                                                       \
            std::fflush(stderr);                                                                              \
        }                                                                                                      \
    } while (0)

namespace btc {

// ── Block partitioning ──────────────────────────────────────────────────────
//
// Splits [0, n) into `parts` contiguous blocks as evenly as possible without
// assuming n is divisible by parts -- the first (n % parts) blocks get one
// extra element. Returns (counts, offsets) with counts.size() == offsets.size()
// == parts, offsets[k] + counts[k] == offsets[k+1].
struct BlockPartition {
    std::vector<std::size_t> counts;
    std::vector<std::size_t> offsets;
};

inline BlockPartition ComputeBlockPartition(std::size_t n, int parts) {
    if (parts <= 0)
        throw std::invalid_argument("ComputeBlockPartition: parts must be positive");
    BlockPartition part;
    part.counts.resize(static_cast<std::size_t>(parts));
    part.offsets.resize(static_cast<std::size_t>(parts));
    const std::size_t base = n / static_cast<std::size_t>(parts);
    const std::size_t remainder = n % static_cast<std::size_t>(parts);
    std::size_t offset = 0;
    for (int k = 0; k < parts; ++k) {
        const std::size_t count = base + (static_cast<std::size_t>(k) < remainder ? 1 : 0);
        part.counts[static_cast<std::size_t>(k)] = count;
        part.offsets[static_cast<std::size_t>(k)] = offset;
        offset += count;
    }
    return part;
}

// ── 2D MPI process grid ─────────────────────────────────────────────────────
struct ProcessGrid2D {
    MPI_Comm world = MPI_COMM_NULL;
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm row_comm = MPI_COMM_NULL; // all ranks sharing this rank's process ROW
    MPI_Comm col_comm = MPI_COMM_NULL; // all ranks sharing this rank's process COLUMN
    int Pr = 1;
    int Pc = 1;
    int process_row = 0;
    int process_col = 0;
    int world_rank = 0;
    int world_size = 1;

    ~ProcessGrid2D() {
        TC_BGV_MPI_TRACE(world_rank, "ProcessGrid2D::~dtor: freeing comms...");
        if (row_comm != MPI_COMM_NULL) MPI_Comm_free(&row_comm);
        TC_BGV_MPI_TRACE(world_rank, "ProcessGrid2D::~dtor: row_comm freed.");
        if (col_comm != MPI_COMM_NULL) MPI_Comm_free(&col_comm);
        TC_BGV_MPI_TRACE(world_rank, "ProcessGrid2D::~dtor: col_comm freed.");
        if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
        TC_BGV_MPI_TRACE(world_rank, "ProcessGrid2D::~dtor: cart freed, done.");
    }
    ProcessGrid2D() = default;
    ProcessGrid2D(const ProcessGrid2D&) = delete;
    ProcessGrid2D& operator=(const ProcessGrid2D&) = delete;

    // Move-only: BuildProcessGrid2D returns one of these by value, and
    // BGVDistributedRowColBackend's constructor assigns the result into
    // grid_ -- both need a move (not a deep MPI_Comm_dup copy) so the
    // MPI_Comm_free calls above only ever run once per underlying
    // communicator. Moved-from communicators are reset to MPI_COMM_NULL so
    // the moved-from object's destructor is a no-op.
    ProcessGrid2D(ProcessGrid2D&& other) noexcept { *this = std::move(other); }
    ProcessGrid2D& operator=(ProcessGrid2D&& other) noexcept {
        if (this == &other) return *this;
        if (row_comm != MPI_COMM_NULL) MPI_Comm_free(&row_comm);
        if (col_comm != MPI_COMM_NULL) MPI_Comm_free(&col_comm);
        if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);

        world = other.world;
        cart = other.cart;
        row_comm = other.row_comm;
        col_comm = other.col_comm;
        Pr = other.Pr;
        Pc = other.Pc;
        process_row = other.process_row;
        process_col = other.process_col;
        world_rank = other.world_rank;
        world_size = other.world_size;

        other.cart = MPI_COMM_NULL;
        other.row_comm = MPI_COMM_NULL;
        other.col_comm = MPI_COMM_NULL;
        return *this;
    }
};

// Builds a balanced Pr x Pc grid (Pr ~= Pc) over `comm`, with Pr, Pc <= n
// wherever possible (see the task's "unless empty blocks are explicitly
// supported" allowance -- for P > n*n this falls back to whatever
// MPI_Dims_create produces, which may leave some ranks with empty blocks;
// callers with 1/2/4 ranks against N=4/N=8 in this project's test matrix
// never hit that case).
inline ProcessGrid2D BuildProcessGrid2D(MPI_Comm comm, std::size_t n) {
    ProcessGrid2D grid;
    grid.world = comm;
    MPI_Comm_rank(comm, &grid.world_rank);
    MPI_Comm_size(comm, &grid.world_size);

    int dims[2] = {0, 0};
    MPI_Dims_create(grid.world_size, 2, dims);
    // MPI_Dims_create has no preference for which factor goes first; prefer
    // Pr <= Pc (matches the task's Pr ~= Pc balance goal deterministically).
    grid.Pr = std::min(dims[0], dims[1]);
    grid.Pc = std::max(dims[0], dims[1]);
    (void)n; // block partitioning tolerates Pr/Pc > n (see ComputeBlockPartition)

    int periods[2] = {0, 0};
    int mpi_dims[2] = {grid.Pr, grid.Pc};
    MPI_Cart_create(comm, 2, mpi_dims, periods, /*reorder=*/0, &grid.cart);

    int coords[2] = {0, 0};
    MPI_Cart_coords(grid.cart, grid.world_rank, 2, coords);
    grid.process_row = coords[0];
    grid.process_col = coords[1];

    // row_comm: fix process_row, vary process_col (MPI_Cart_sub keeps the
    // dimension marked `true`) -- so row_comm groups ranks sharing this
    // rank's process ROW, matching the doc comment above.
    int keep_col_varying[2] = {0, 1};
    MPI_Cart_sub(grid.cart, keep_col_varying, &grid.row_comm);
    int keep_row_varying[2] = {1, 0};
    MPI_Cart_sub(grid.cart, keep_row_varying, &grid.col_comm);

    return grid;
}

// ── Distributed packed-matrix representation ────────────────────────────────
//
// Unlike PackedBooleanMatrix (bgv_rowcol_backend.hpp), which holds ALL N rows
// and N columns, a DistributedRowColMatrix holds only the rows in this rank's
// row-block I_r and the columns in this rank's column-block J_c -- see this
// file's header comment for the full ownership invariant.
class DistributedRowColMatrix {
public:
    DistributedRowColMatrix() : n_(0) {}
    DistributedRowColMatrix(std::size_t n, std::vector<std::size_t> global_row_indices,
                             std::vector<bgv_batched::Ciphertext> rows,
                             std::vector<std::size_t> global_col_indices,
                             std::vector<bgv_batched::Ciphertext> cols)
        : n_(n),
          global_row_indices_(std::move(global_row_indices)),
          rows_(std::move(rows)),
          global_col_indices_(std::move(global_col_indices)),
          cols_(std::move(cols)) {}

    std::size_t dim() const noexcept { return n_; }
    const std::vector<std::size_t>& global_row_indices() const { return global_row_indices_; }
    const std::vector<std::size_t>& global_col_indices() const { return global_col_indices_; }
    const std::vector<bgv_batched::Ciphertext>& rows() const { return rows_; }
    const std::vector<bgv_batched::Ciphertext>& cols() const { return cols_; }

    const bgv_batched::Ciphertext& row_at(std::size_t local_i) const { return rows_.at(local_i); }
    const bgv_batched::Ciphertext& col_at(std::size_t local_j) const { return cols_.at(local_j); }

private:
    std::size_t n_;
    std::vector<std::size_t> global_row_indices_;
    std::vector<bgv_batched::Ciphertext> rows_;
    std::vector<std::size_t> global_col_indices_;
    std::vector<bgv_batched::Ciphertext> cols_;
};

// BGVDistributedRowColBackend -- satisfies the Backend concept (see
// backend.hpp) using DistributedRowColMatrix. One instance is built for one
// specific N, over one specific MPI communicator (typically MPI_COMM_WORLD).
//
// Usage (every rank runs this identically -- SPMD):
//   MPI_Init(&argc, &argv);
//   btc::BGVDistributedRowColBackend backend(MPI_COMM_WORLD, N, T);
//   auto enc_A = backend.EncryptFromRoot(plain_A);   // plain_A ignored on non-root ranks
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = backend.DecryptOnRoot(enc_S);      // valid only on rank 0
//   MPI_Finalize();
class BGVDistributedRowColBackend {
public:
    using matrix_type = DistributedRowColMatrix;

    struct OpCounters {
        std::size_t cc_mult = 0;
        std::size_t cp_mult = 0;
        std::size_t add = 0;
        std::size_t sub = 0;
        std::size_t rotate = 0;
        std::size_t dot_products = 0;
        std::size_t sumor_calls = 0;
        std::size_t bytes_sent = 0;
        std::size_t bytes_received = 0;
    };

private:
    // dot_products/bytes_sent/bytes_received are written from inside
    // matmul()'s `#pragma omp parallel for` (dot_products) or from
    // AssembleRows/AssembleCols (bytes_*, called after the parallel region
    // has already joined, but kept atomic for uniformity and because a
    // future Alltoallv-batched Stage-3 rewrite may parallelize assembly
    // too) -- so these three need atomic increments; cc_mult/cp_mult/add/
    // sub/rotate/sumor_calls are only ever copied wholesale from
    // local_backend_'s own counters (SyncLocalBackendCounters), never
    // incremented directly here, so they stay plain in the snapshot struct
    // above and don't need a mutable-atomics shadow copy.
    struct AtomicCounters {
        std::atomic<std::size_t> dot_products{0};
        std::atomic<std::size_t> bytes_sent{0};
        std::atomic<std::size_t> bytes_received{0};
    };

public:

    static constexpr int kRootRank = 0;

    // Builds (on root) or receives (on all other ranks) the shared crypto
    // context, then builds this rank's local 2D-grid position and local
    // BGVRowColBackend wrapper. `depth` should come from
    // EstimateBGVRowColDepth(n, T) exactly as the single-process benchmark
    // computes it (see bench_bgv_rowcol.cpp) -- unchanged cryptographic
    // parameters, per the task's "do not change cryptographic parameters
    // merely to make the benchmark faster" constraint.
    BGVDistributedRowColBackend(MPI_Comm comm, std::size_t n, uint32_t depth,
                                 uint64_t plaintext_modulus = 786433)
        : n_(n) {
        if (!IsPowerOfTwo(n_))
            throw std::invalid_argument("BGVDistributedRowColBackend: N must be a power of two, got " +
                                         std::to_string(n_));

        grid_ = BuildProcessGrid2D(comm, n_);
        TC_BGV_MPI_TRACE(grid_.world_rank, "grid built: Pr=%d Pc=%d process_row=%d process_col=%d", grid_.Pr,
                          grid_.Pc, grid_.process_row, grid_.process_col);

        if (grid_.world_rank == kRootRank) {
            TC_BGV_MPI_TRACE(grid_.world_rank, "root: calling bgv_batched::setup (keygen)...");
            bgv_batched::Params params;
            params.plaintext_modulus = plaintext_modulus;
            params.multiplicative_depth = depth;
            params.batch_size = static_cast<uint32_t>(n_);
            ctx_ = bgv_batched::setup(params);
            TC_BGV_MPI_TRACE(grid_.world_rank, "root: setup done, constructing local BGVRowColBackend "
                                                "(generates rotation keys)...");
            // Rotation keys need the secret key -- generate them on root
            // BEFORE serializing the bundle for the other ranks (see
            // BGVRowColBackend's constructor, called just below, which does
            // this generation as a side effect via RotationKeys::GenerateHere).
            local_backend_ = std::make_unique<BGVRowColBackend>(
                ctx_, n_, BGVMatMulBackend::OpenMPOptimized,
                BGVRowColBackend::RotationKeys::GenerateHere);
            TC_BGV_MPI_TRACE(grid_.world_rank, "root: local backend constructed, serializing context bundle...");

            bundle_ = bgv_mpi::SerializeContextBundle(ctx_);
            TC_BGV_MPI_TRACE(grid_.world_rank,
                              "root: bundle serialized (cc=%zu bytes, pk=%zu, evalmult=%zu, evalauto=%zu)",
                              bundle_.crypto_context.size(), bundle_.public_key.size(),
                              bundle_.eval_mult_key.size(), bundle_.eval_automorphism_key.size());
        }

        TC_BGV_MPI_TRACE(grid_.world_rank, "entering BroadcastContextBundle...");
        BroadcastContextBundle();
        TC_BGV_MPI_TRACE(grid_.world_rank, "BroadcastContextBundle done.");

        if (grid_.world_rank != kRootRank) {
            TC_BGV_MPI_TRACE(grid_.world_rank, "non-root: deserializing context bundle...");
            ctx_ = bgv_mpi::DeserializeContextBundle(bundle_);
            TC_BGV_MPI_TRACE(grid_.world_rank, "non-root: constructing local BGVRowColBackend "
                                                "(AssumeAlreadyInstalled)...");
            local_backend_ = std::make_unique<BGVRowColBackend>(
                ctx_, n_, BGVMatMulBackend::OpenMPOptimized,
                BGVRowColBackend::RotationKeys::AssumeAlreadyInstalled);
            TC_BGV_MPI_TRACE(grid_.world_rank, "non-root: local backend constructed.");
        }

        row_partition_ = ComputeBlockPartition(n_, grid_.Pr);
        col_partition_ = ComputeBlockPartition(n_, grid_.Pc);

        omp_set_dynamic(0);
        omp_set_max_active_levels(GetConfiguredMaxActiveLevels());
        TC_BGV_MPI_TRACE(grid_.world_rank, "constructor complete.");
    }

    // Constructs directly from an ALREADY-BUILT Context -- crypto context,
    // public key, and eval-mult/rotation keys already installed, secret key
    // absent -- instead of generating a fresh keypair on rank 0 and
    // broadcasting it. Used by the offline-artifact workflow
    // (bgv_distributed_rowcol_artifacts.hpp's "compute" mode): every rank
    // independently loads the SAME serialized artifacts from a shared
    // filesystem (see that file's doc comment for why this is safe --
    // OpenFHE's CryptoContext deserialization interns by parameters, so
    // every rank ends up with an equivalent context object), so there is no
    // root/non-root asymmetry and no MPI broadcast here -- this constructor
    // issues zero MPI calls of its own beyond `BuildProcessGrid2D`'s.
    //
    // Every rank must pass an equivalent `ctx` (same rotation indices
    // installed, same eval-mult key) -- mismatched contexts across ranks
    // will fail loudly the first time a mismatched key is used (OpenFHE
    // throws), not silently compute a wrong result.
    BGVDistributedRowColBackend(MPI_Comm comm, std::size_t n, bgv_batched::Context ctx) : n_(n), ctx_(std::move(ctx)) {
        if (!IsPowerOfTwo(n_))
            throw std::invalid_argument("BGVDistributedRowColBackend: N must be a power of two, got " +
                                         std::to_string(n_));

        grid_ = BuildProcessGrid2D(comm, n_);
        local_backend_ = std::make_unique<BGVRowColBackend>(ctx_, n_, BGVMatMulBackend::OpenMPOptimized,
                                                              BGVRowColBackend::RotationKeys::AssumeAlreadyInstalled);
        row_partition_ = ComputeBlockPartition(n_, grid_.Pr);
        col_partition_ = ComputeBlockPartition(n_, grid_.Pc);

        omp_set_dynamic(0);
        omp_set_max_active_levels(GetConfiguredMaxActiveLevels());
    }

    std::size_t dim(const DistributedRowColMatrix& A) const { return A.dim(); }

    DistributedRowColMatrix copy(const DistributedRowColMatrix& A) const {
        return DistributedRowColMatrix(A.dim(), A.global_row_indices(), A.rows(), A.global_col_indices(),
                                        A.cols());
    }

    // ── Distributed matrix multiplication ────────────────────────────────
    //
    // Phase 1 (no communication): rank (r,c) computes C[i][j] for every
    // i in I_r, j in J_c EXACTLY ONCE (BooleanDotProductPublic), then
    // repacks it into both a row-piece (masked to slot j) and a
    // column-piece (masked to slot i) via MaskMultPublic -- the SAME single-
    // computation-reused-twice pattern OpenMPOptimized uses locally
    // (bgv_rowcol_backend.hpp:606-637), just with MPI ranks standing in for
    // OpenMP threads as the unit of "owns a disjoint (i,j) block".
    //
    // Phase 2/3 (communication): row assembly within row_comm, column
    // assembly within col_comm -- see AssembleRows/AssembleCols.
    DistributedRowColMatrix matmul(const DistributedRowColMatrix& A, const DistributedRowColMatrix& B) const {
        return MatMulOnGrid(A, B, grid_, row_partition_, col_partition_);
    }

    // ── Concurrent recursion branches (Stage 3) ──────────────────────────
    //
    // sum_powers_recursive's even-T step (algorithms.hpp) issues two matmul
    // calls that are mutually independent -- both P(2h) = matmul(P_h, P_h)
    // and the intermediate matmul(P_h, S_h) depend only on already-computed
    // P_h/S_h, never on each other. The generic algorithm still calls them
    // sequentially (by design -- algorithms.hpp is backend-agnostic and
    // deliberately not touched here, see backend.hpp's "no changes to
    // algorithms.hpp are required" comment). matmul_pair gives this backend
    // a way to run BOTH concurrently: split grid_'s world communicator into
    // two disjoint halves, each with its OWN full 2D process grid over ALL
    // N rows/columns, and have one half compute matmul(A1,B1) while the
    // other computes matmul(A2,B2) -- genuinely concurrently, on disjoint
    // MPI ranks, not just interleaved on the same ranks.
    //
    // A sub-grid built over half the ranks generally has DIFFERENT Pr'/Pc'
    // than grid_ (MPI_Dims_create factors a different rank count), so it
    // partitions N rows/columns across ranks differently -- A1/B1/A2/B2 come
    // in shaped to grid_'s partition (each rank holding grid_'s row/column
    // block), but MatMulOnGrid indexes its operands by the SUB-grid's
    // partition. Both the INPUT operands and the OUTPUT result therefore
    // need reshaping between grid_'s partition and the sub-grid's: gather
    // each operand/result to a known rank (GatherToRankWithinGrid) and
    // re-scatter it across the other grid from there (ScatterFromRankToGrid).
    // Each reshape moves O(N) ciphertexts (N rows + N columns), not O(N^2)
    // -- the same order of magnitude as EncryptFromRoot/DecryptOnRoot's
    // existing boundary cost -- so this does not reintroduce the "gather the
    // whole matrix after every matmul" cost matmul() itself avoids.
    //
    // Falls back to plain sequential matmul()+matmul() when there are too
    // few ranks to split (world_size < 2) -- still correct, just not
    // concurrent.
    std::pair<DistributedRowColMatrix, DistributedRowColMatrix>
    matmul_pair(const DistributedRowColMatrix& A1, const DistributedRowColMatrix& B1,
                const DistributedRowColMatrix& A2, const DistributedRowColMatrix& B2) const {
        RequireSameDim(A1, B1, "matmul_pair(A1,B1)");
        RequireSameDim(A2, B2, "matmul_pair(A2,B2)");

        if (grid_.world_size < 2) {
            // Nothing to split -- one rank cannot run two things at once.
            return {matmul(A1, B1), matmul(A2, B2)};
        }

        // color 0 = world ranks [0, half); color 1 = world ranks [half,
        // world_size). key = world_rank preserves relative order within
        // each half, so sub-communicator rank 0 of color 0 is ALWAYS world
        // rank 0, and sub-communicator rank 0 of color 1 is ALWAYS world
        // rank `half` -- both computed identically (no communication
        // needed) on every rank, which the reconciliation step below relies
        // on to pick its ScatterFromRankToGrid root.
        const int half = grid_.world_size / 2;
        const int color = (grid_.world_rank < half) ? 0 : 1;
        MPI_Comm sub_comm = MPI_COMM_NULL;
        MPI_Comm_split(grid_.world, color, grid_.world_rank, &sub_comm);

        ProcessGrid2D sub_grid = BuildProcessGrid2D(sub_comm, n_);
        // sub_grid's own cart/row_comm/col_comm were derived FROM sub_comm
        // via MPI_Cart_create/MPI_Cart_sub, which produce independent
        // communicator handles -- freeing sub_comm here does not invalidate
        // them (standard MPI semantics), and sub_grid does not treat
        // `world` as owned (matches grid_'s own contract, see
        // ProcessGrid2D's destructor), so this must be freed explicitly.
        MPI_Comm_free(&sub_comm);

        BlockPartition sub_row_partition = ComputeBlockPartition(n_, sub_grid.Pr);
        BlockPartition sub_col_partition = ComputeBlockPartition(n_, sub_grid.Pc);

        TC_BGV_MPI_TRACE(grid_.world_rank, "matmul_pair: color=%d sub Pr=%d Pc=%d", color, sub_grid.Pr,
                          sub_grid.Pc);

        // ── Input reshape: grid_'s partition -> each half's sub-grid partition ──
        //
        // Gather each operand to a FIXED world rank (0 for the A1/B1 pair
        // color 0 will use, `half` for the A2/B2 pair color 1 will use) --
        // collective over the FULL grid_.world, so every rank participates
        // in all four calls regardless of its own color (unconditional,
        // same order on every rank). Only the two operands THIS rank's
        // color actually needs get re-scattered into sub_grid's shape below.
        PackedBooleanMatrix A1_full = GatherToRankWithinGrid(A1, grid_, /*local_root=*/0);
        PackedBooleanMatrix B1_full = GatherToRankWithinGrid(B1, grid_, /*local_root=*/0);
        PackedBooleanMatrix A2_full = GatherToRankWithinGrid(A2, grid_, /*local_root=*/half);
        PackedBooleanMatrix B2_full = GatherToRankWithinGrid(B2, grid_, /*local_root=*/half);

        // Re-scatter is collective over sub_grid.cart, i.e. only within THIS
        // rank's own half -- color 0 ranks scatter A1/B1 (their local rank 0
        // = world rank 0, which is exactly where A1_full/B1_full just
        // landed); color 1 ranks scatter A2/B2 (their local rank 0 = world
        // rank `half`, where A2_full/B2_full landed). Each half must NOT
        // call the other half's Scatter -- that communicator doesn't even
        // contain this rank.
        DistributedRowColMatrix A_local, B_local;
        if (color == 0) {
            A_local = ScatterFromRankToGrid(A1_full, sub_grid, sub_row_partition, sub_col_partition, 0);
            B_local = ScatterFromRankToGrid(B1_full, sub_grid, sub_row_partition, sub_col_partition, 0);
        } else {
            A_local = ScatterFromRankToGrid(A2_full, sub_grid, sub_row_partition, sub_col_partition, 0);
            B_local = ScatterFromRankToGrid(B2_full, sub_grid, sub_row_partition, sub_col_partition, 0);
        }

        DistributedRowColMatrix local_result = MatMulOnGrid(A_local, B_local, sub_grid, sub_row_partition,
                                                              sub_col_partition);

        // ── Output reshape: sub-grid's partition -> grid_'s partition ──────
        //
        // Symmetric to the input reshape: gather this half's result to ITS
        // local rank 0 (world rank 0 for color 0, world rank `half` for
        // color 1), then re-scatter each into grid_'s (the FULL grid's)
        // per-rank blocks. Both ScatterFromRankToGrid calls here are
        // collective over grid_.cart and must run identically on every rank
        // in the SAME order, hence unconditional (not guarded by `color`).
        PackedBooleanMatrix gathered = GatherToRankWithinGrid(local_result, sub_grid, /*local_root=*/0);

        PackedBooleanMatrix result1_full = (color == 0) ? std::move(gathered) : PackedBooleanMatrix{};
        PackedBooleanMatrix result2_full = (color == 1) ? std::move(gathered) : PackedBooleanMatrix{};

        DistributedRowColMatrix result1 =
            ScatterFromRankToGrid(result1_full, grid_, row_partition_, col_partition_, /*local_root=*/0);
        DistributedRowColMatrix result2 =
            ScatterFromRankToGrid(result2_full, grid_, row_partition_, col_partition_, /*local_root=*/half);

        // SyncLocalBackendCounters already ran inside MatMulOnGrid.
        return {std::move(result1), std::move(result2)};
    }

    // ── Distributed element-wise OR ──────────────────────────────────────
    //
    // Rows/columns are already co-located identically for A and B (same
    // ownership invariant), so no all-to-all communication is needed. Owner-
    // compute dedup (Stage 3): rather than every rank in a row_comm/col_comm
    // redundantly computing EvalBooleanOR for the SAME replicated row/column
    // (e.g. every rank in a 2x2 grid's row_comm computing row i's OR
    // independently), a single deterministic owner per row/column
    // (rowOwner(i) = i % row_comm_size, colOwner(j) = j % col_comm_size --
    // the SAME scheme AssembleRows/AssembleCols use, so ownership is
    // consistent across matmul/or_mat) computes it once, then broadcasts the
    // batch to the rest of its row_comm/col_comm. This trades EvalBooleanOR
    // compute (cheap: one EvalAdd-based OR, no rotations) for a broadcast
    // (also cheap: O(local block size) ciphertexts, batched into one
    // message) -- worthwhile because a redundant EvalBooleanOR call is pure
    // waste on every non-owner rank, not because the broadcast itself is
    // free.
    DistributedRowColMatrix or_mat(const DistributedRowColMatrix& A, const DistributedRowColMatrix& B) const {
        RequireSameDim(A, B, "or_mat");
        const std::size_t local_rows = A.rows().size();
        const std::size_t local_cols = A.cols().size();
        const auto& row_idx = A.global_row_indices();
        const auto& col_idx = A.global_col_indices();

        int row_comm_size = 0, row_comm_rank = 0;
        MPI_Comm_size(grid_.row_comm, &row_comm_size);
        MPI_Comm_rank(grid_.row_comm, &row_comm_rank);
        int col_comm_size = 0, col_comm_rank = 0;
        MPI_Comm_size(grid_.col_comm, &col_comm_size);
        MPI_Comm_rank(grid_.col_comm, &col_comm_rank);

        std::vector<bgv_batched::Ciphertext> rows(local_rows);
        for (std::size_t li = 0; li < local_rows; ++li) {
            const int owner = static_cast<int>(row_idx[li] % static_cast<std::size_t>(row_comm_size));
            if (row_comm_rank == owner)
                rows[li] = local_backend_->EvalBooleanORPublic(A.row_at(li), B.row_at(li));
        }
        BroadcastOwnedBatch(rows, row_idx, row_comm_size, row_comm_rank, grid_.row_comm);

        std::vector<bgv_batched::Ciphertext> cols(local_cols);
        for (std::size_t lj = 0; lj < local_cols; ++lj) {
            const int owner = static_cast<int>(col_idx[lj] % static_cast<std::size_t>(col_comm_size));
            if (col_comm_rank == owner)
                cols[lj] = local_backend_->EvalBooleanORPublic(A.col_at(lj), B.col_at(lj));
        }
        BroadcastOwnedBatch(cols, col_idx, col_comm_size, col_comm_rank, grid_.col_comm);

        SyncLocalBackendCounters();
        return DistributedRowColMatrix(n_, A.global_row_indices(), std::move(rows), A.global_col_indices(),
                                        std::move(cols));
    }

    // ── Root-driven encryption / decryption ──────────────────────────────
    //
    // Root encrypts the full plaintext matrix (needs the secret-key-bearing
    // context) and scatters row/column blocks to their owning process
    // row/column groups. `plain` is ignored on non-root ranks (every rank
    // must call this identically -- SPMD -- but only rank 0's argument is
    // meaningful).
    DistributedRowColMatrix EncryptFromRoot(const BoolMatrix& plain) const {
        if (grid_.world_rank == kRootRank && (plain.rows() != n_ || plain.cols() != n_))
            throw std::invalid_argument("BGVDistributedRowColBackend::EncryptFromRoot: dimension mismatch");

        // Root encrypts ALL N rows and N columns once (reusing
        // BGVRowColBackend::Encrypt), then every rank -- including root --
        // extracts just the row/column block it owns via a broadcast-and-
        // slice (the ciphertexts are the same size regardless of N, so this
        // is not a Stage-3-scale bottleneck at the N this project tests).
        PackedBooleanMatrix full;
        if (grid_.world_rank == kRootRank)
            full = local_backend_->Encrypt(plain);

        return ScatterFullMatrixToBlocks(full);
    }

    // Valid only on rank 0 -- gathers this rank's row block (root already
    // owns row block I_0, so only cross-row-block rows need fetching) into
    // a full PackedBooleanMatrix, then decrypts + cross-checks via
    // BGVRowColBackend::Decrypt exactly like the single-process backend.
    BoolMatrix DecryptOnRoot(const DistributedRowColMatrix& enc) const {
        PackedBooleanMatrix full = GatherToRoot(enc);
        if (grid_.world_rank != kRootRank)
            return BoolMatrix(n_, n_, false); // meaningless on non-root ranks
        return local_backend_->Decrypt(full);
    }

    // Same gather as DecryptOnRoot, but stops BEFORE decryption -- returns
    // the full result still as ciphertexts. Used by the offline-artifact
    // workflow's "compute" mode (bgv_distributed_rowcol_artifacts.hpp) to
    // serialize the encrypted result for a later, separate decrypt step;
    // compute ranks never hold a secret key (see this class's doc comment),
    // so gathering-without-decrypting is the only option available to them.
    // Meaningful only on rank 0, same convention as GatherToRoot/DecryptOnRoot.
    PackedBooleanMatrix GatherEncryptedResultToRoot(const DistributedRowColMatrix& enc) const {
        return GatherToRoot(enc);
    }

    OpCounters counters() const {
        OpCounters snapshot = plain_counters_;
        snapshot.dot_products = atomic_counters_.dot_products.load(std::memory_order_relaxed);
        snapshot.bytes_sent = atomic_counters_.bytes_sent.load(std::memory_order_relaxed);
        snapshot.bytes_received = atomic_counters_.bytes_received.load(std::memory_order_relaxed);
        return snapshot;
    }
    void reset_counters() const {
        plain_counters_ = OpCounters{};
        atomic_counters_.dot_products.store(0, std::memory_order_relaxed);
        atomic_counters_.bytes_sent.store(0, std::memory_order_relaxed);
        atomic_counters_.bytes_received.store(0, std::memory_order_relaxed);
        local_backend_->reset_counters();
    }

    int world_rank() const { return grid_.world_rank; }
    int world_size() const { return grid_.world_size; }
    int Pr() const { return grid_.Pr; }
    int Pc() const { return grid_.Pc; }
    int process_row() const { return grid_.process_row; }
    int process_col() const { return grid_.process_col; }

private:
    static void RequireSameDim(const DistributedRowColMatrix& A, const DistributedRowColMatrix& B,
                                const char* who) {
        if (A.dim() != B.dim())
            throw std::invalid_argument(std::string("BGVDistributedRowColBackend::") + who +
                                         ": dimension mismatch");
    }

    void BroadcastContextBundle() {
        bgv_mpi::BroadcastBytes(bundle_.crypto_context, kRootRank, grid_.world);
        bgv_mpi::BroadcastBytes(bundle_.public_key, kRootRank, grid_.world);
        bgv_mpi::BroadcastBytes(bundle_.eval_mult_key, kRootRank, grid_.world);
        bgv_mpi::BroadcastBytes(bundle_.eval_automorphism_key, kRootRank, grid_.world);
    }

    void SyncLocalBackendCounters() const {
        auto c = local_backend_->counters();
        plain_counters_.cc_mult = c.cc_mult;
        plain_counters_.cp_mult = c.cp_mult;
        plain_counters_.add = c.add;
        plain_counters_.sub = c.sub;
        plain_counters_.rotate = c.rotate;
        plain_counters_.sumor_calls = c.sumor_calls;
        // dot_products is tracked independently above (BGVRowColBackend's
        // own dot_products counter would also be correct here, but this
        // backend's counter is incremented at the exact call site for
        // clarity when reading this file in isolation).
    }

    // ── matmul, parameterized over an explicit grid ──────────────────────
    //
    // The body matmul() used to run directly against grid_/row_partition_/
    // col_partition_; factored out so matmul_pair can run the SAME logic
    // against a temporary sub-grid covering only half the world (see
    // matmul_pair's doc comment). `grid` supplies the row_comm/col_comm the
    // assembly phase communicates over; `row_partition`/`col_partition` must
    // be ComputeBlockPartition(n_, grid.Pr)/ComputeBlockPartition(n_,
    // grid.Pc) for that SAME grid (grid_'s own or a sub-grid's).
    DistributedRowColMatrix MatMulOnGrid(const DistributedRowColMatrix& A, const DistributedRowColMatrix& B,
                                          const ProcessGrid2D& grid, const BlockPartition& row_partition,
                                          const BlockPartition& col_partition) const {
        RequireSameDim(A, B, "matmul");
        TC_BGV_MPI_TRACE(grid.world_rank, "MatMulOnGrid: entered (A rows=%zu cols=%zu)", A.rows().size(),
                          A.cols().size());

        const std::size_t local_rows = row_partition.counts[static_cast<std::size_t>(grid.process_row)];
        const std::size_t row_offset = row_partition.offsets[static_cast<std::size_t>(grid.process_row)];
        const std::size_t local_cols = col_partition.counts[static_cast<std::size_t>(grid.process_col)];
        const std::size_t col_offset = col_partition.offsets[static_cast<std::size_t>(grid.process_col)];

        auto zero = local_backend_->EncryptZeroPublic();
        std::vector<bgv_batched::Ciphertext> partial_rows(local_rows, zero);
        std::vector<bgv_batched::Ciphertext> partial_cols(local_cols, zero);
        TC_BGV_MPI_TRACE(grid.world_rank, "MatMulOnGrid: starting local dot-product phase (local_rows=%zu "
                                           "local_cols=%zu)",
                          local_rows, local_cols);

        const auto local_rows_i = static_cast<std::int64_t>(local_rows);
        const auto local_cols_i = static_cast<std::int64_t>(local_cols);

#pragma omp parallel for collapse(2) schedule(static)
        for (std::int64_t li = 0; li < local_rows_i; ++li) {
            for (std::int64_t lj = 0; lj < local_cols_i; ++lj) {
                const std::size_t global_i = row_offset + static_cast<std::size_t>(li);
                const std::size_t global_j = col_offset + static_cast<std::size_t>(lj);

                // A.row_at(li) is valid because A's row block (replicated
                // across grid.row_comm) is indexed by the SAME row
                // partition this rank belongs to; symmetrically for B's
                // column block.
                auto cij = local_backend_->BooleanDotProductPublic(A.row_at(static_cast<std::size_t>(li)),
                                                                     B.col_at(static_cast<std::size_t>(lj)));
                auto row_piece = local_backend_->MaskMultPublic(cij, global_j);
                auto col_piece = local_backend_->MaskMultPublic(cij, global_i);

#pragma omp critical(distributed_rowcol_accumulate)
                {
                    partial_rows[static_cast<std::size_t>(li)] =
                        local_backend_->EvalAddPublic(partial_rows[static_cast<std::size_t>(li)], row_piece);
                    partial_cols[static_cast<std::size_t>(lj)] =
                        local_backend_->EvalAddPublic(partial_cols[static_cast<std::size_t>(lj)], col_piece);
                }

                atomic_counters_.dot_products.fetch_add(1, std::memory_order_relaxed);
            }
        }
        // BooleanDotProductPublic/MaskMultPublic internally increment
        // local_backend_'s OWN counters (cc_mult/cp_mult/rotate/...); those
        // are pulled into this backend's counters() via
        // SyncLocalBackendCounters, called once per matmul/or_mat below --
        // not per-cell, to avoid an atomic-add storm inside the OpenMP loop.
        TC_BGV_MPI_TRACE(grid.world_rank, "MatMulOnGrid: local dot-product phase done, entering AssembleRows...");

        auto assembled_rows = AssembleRows(partial_rows, row_offset, local_rows, grid.row_comm);
        TC_BGV_MPI_TRACE(grid.world_rank, "MatMulOnGrid: AssembleRows done, entering AssembleCols...");
        auto assembled_cols = AssembleCols(partial_cols, col_offset, local_cols, grid.col_comm);
        TC_BGV_MPI_TRACE(grid.world_rank, "MatMulOnGrid: AssembleCols done.");
        SyncLocalBackendCounters();

        return DistributedRowColMatrix(n_, assembled_rows.first, assembled_rows.second, assembled_cols.first,
                                        assembled_cols.second);
    }

    // ── Row assembly (within an explicit row communicator) ────────────────
    //
    // Every rank sharing `comm` holds a DIFFERENT partial piece of the same
    // row block (nonzero only in the column-slots that rank's column-block
    // owns) -- `comm` is grid_.row_comm for the top-level matmul(), or a
    // sub-grid's row_comm when called from within matmul_pair. Every rank in
    // `comm` enumerates the SAME global row indices (row_offset/local_rows
    // come from the SAME partition entry for every rank sharing a row
    // block), so ownership -- rowOwner(i) = i % comm_size -- is identical on
    // every rank without any handshake.
    //
    // Stage 3 batching: instead of one MPI message (send/recv or broadcast)
    // per row, every source rank buckets ITS local rows by owner and sends
    // ONE message per (source, owner) pair; each owner then broadcasts its
    // whole assembled batch in ONE message instead of one broadcast per row.
    std::pair<std::vector<std::size_t>, std::vector<bgv_batched::Ciphertext>>
    AssembleRows(const std::vector<bgv_batched::Ciphertext>& partial_rows, std::size_t row_offset,
                 std::size_t local_rows, MPI_Comm comm) const {
        int comm_size = 0, comm_rank = 0;
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);

        std::vector<std::size_t> global_indices(local_rows);
        for (std::size_t li = 0; li < local_rows; ++li) global_indices[li] = row_offset + li;

        std::vector<std::size_t> owned_tags;
        std::vector<bgv_batched::Ciphertext> owned_values;
        AssembleOwnedBucket(partial_rows, global_indices, comm_size, comm_rank, comm, kAssembleRowsTag, owned_tags,
                             owned_values);

        std::vector<bgv_batched::Ciphertext> assembled;
        GatherOwnedToAll(assembled, global_indices, owned_tags, owned_values, comm_size, comm_rank, comm);
        return {std::move(global_indices), std::move(assembled)};
    }

    // Symmetric to AssembleRows, over an explicit column communicator;
    // colOwner(j) = j % comm_size.
    std::pair<std::vector<std::size_t>, std::vector<bgv_batched::Ciphertext>>
    AssembleCols(const std::vector<bgv_batched::Ciphertext>& partial_cols, std::size_t col_offset,
                 std::size_t local_cols, MPI_Comm comm) const {
        int comm_size = 0, comm_rank = 0;
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);

        std::vector<std::size_t> global_indices(local_cols);
        for (std::size_t lj = 0; lj < local_cols; ++lj) global_indices[lj] = col_offset + lj;

        std::vector<std::size_t> owned_tags;
        std::vector<bgv_batched::Ciphertext> owned_values;
        AssembleOwnedBucket(partial_cols, global_indices, comm_size, comm_rank, comm, kAssembleColsTag, owned_tags,
                             owned_values);

        std::vector<bgv_batched::Ciphertext> assembled;
        GatherOwnedToAll(assembled, global_indices, owned_tags, owned_values, comm_size, comm_rank, comm);
        return {std::move(global_indices), std::move(assembled)};
    }

    // Shared gather-with-accumulate phase for AssembleRows/AssembleCols:
    // buckets `partial_values` (this rank's contribution for each of
    // `global_indices`, all in `comm`) by deterministic owner, exchanges one
    // batched message per (source, owner) pair, and -- only on this call's
    // owner slots -- EvalAdds every source's contribution together. Returns
    // this rank's owned (tag, assembled-value) pairs via `owned_tags`/
    // `owned_values` (empty if this rank owns nothing).
    void AssembleOwnedBucket(const std::vector<bgv_batched::Ciphertext>& partial_values,
                              const std::vector<std::size_t>& global_indices, int comm_size, int comm_rank,
                              MPI_Comm comm, int tag, std::vector<std::size_t>& owned_tags,
                              std::vector<bgv_batched::Ciphertext>& owned_values) const {
        std::vector<std::vector<std::size_t>> bucket_tags(static_cast<std::size_t>(comm_size));
        std::vector<std::vector<bgv_batched::Ciphertext>> bucket_values(static_cast<std::size_t>(comm_size));
        for (std::size_t k = 0; k < global_indices.size(); ++k) {
            const std::size_t g = global_indices[k];
            const auto owner = static_cast<std::size_t>(g % static_cast<std::size_t>(comm_size));
            bucket_tags[owner].push_back(g);
            bucket_values[owner].push_back(partial_values[k]);
        }

        // Non-blocking sends: this rank may need to BOTH send to other
        // ranks AND receive from them within this same call (a genuine
        // peer-to-peer exchange, not a root-centric gather), so posting
        // blocking sends for every owner before this rank's own receive
        // loop below risks a rendezvous-protocol deadlock once a batched
        // message is large enough to cross MPI's eager threshold -- see
        // IsendBytes's doc comment (bgv_mpi_serial.hpp). `pending` keeps
        // each serialized buffer alive until WaitAllSends below.
        std::deque<bgv_mpi::PendingSend> pending;
        for (int owner = 0; owner < comm_size; ++owner) {
            const auto& tags = bucket_tags[static_cast<std::size_t>(owner)];
            if (tags.empty() || owner == comm_rank) continue;
            const auto& values = bucket_values[static_cast<std::size_t>(owner)];
            bgv_mpi::IsendCiphertextBatch(tags, values, owner, tag, comm, pending);
            atomic_counters_.bytes_sent.fetch_add(EstimateCiphertextBatchBytes(tags, values),
                                                   std::memory_order_relaxed);
        }

        owned_tags.clear();
        owned_values.clear();
        if (!bucket_tags[static_cast<std::size_t>(comm_rank)].empty()) {
            owned_tags = bucket_tags[static_cast<std::size_t>(comm_rank)];
            owned_values = bucket_values[static_cast<std::size_t>(comm_rank)];
            std::unordered_map<std::size_t, std::size_t> tag_to_slot;
            for (std::size_t k = 0; k < owned_tags.size(); ++k) tag_to_slot[owned_tags[k]] = k;

            for (int src = 0; src < comm_size; ++src) {
                if (src == comm_rank) continue;
                std::vector<std::size_t> recv_tags;
                std::vector<bgv_batched::Ciphertext> recv_values;
                bgv_mpi::RecvCiphertextBatch(src, tag, comm, recv_tags, recv_values);
                atomic_counters_.bytes_received.fetch_add(EstimateCiphertextBatchBytes(recv_tags, recv_values),
                                                           std::memory_order_relaxed);
                for (std::size_t k = 0; k < recv_tags.size(); ++k) {
                    const auto slot = tag_to_slot.at(recv_tags[k]);
                    owned_values[slot] = local_backend_->EvalAddPublic(owned_values[slot], recv_values[k]);
                }
            }
        }

        bgv_mpi::WaitAllSends(pending);
    }

    // Broadcasts each owner's slice of `global_indices` (owner(g) = g %
    // comm_size) to the whole `comm`, one batched message per owner, so
    // every rank ends with `out[k]` == the value for `global_indices[k]`
    // regardless of which rank originally owned/computed it. `owned_tags`/
    // `owned_values` are this rank's own contribution (empty if this rank
    // owns nothing for this comm) -- used both by AssembleRows/AssembleCols
    // (owned values = post-accumulation result) and by or_mat's
    // BroadcastOwnedBatch (owned values = a single EvalBooleanOR, no
    // accumulation).
    void GatherOwnedToAll(std::vector<bgv_batched::Ciphertext>& out, const std::vector<std::size_t>& global_indices,
                           const std::vector<std::size_t>& owned_tags,
                           const std::vector<bgv_batched::Ciphertext>& owned_values, int comm_size, int comm_rank,
                           MPI_Comm comm) const {
        std::unordered_map<std::size_t, std::size_t> global_to_local;
        for (std::size_t k = 0; k < global_indices.size(); ++k) global_to_local[global_indices[k]] = k;
        out.assign(global_indices.size(), bgv_batched::Ciphertext{});

        for (int owner = 0; owner < comm_size; ++owner) {
            std::vector<std::size_t> tags;
            std::vector<bgv_batched::Ciphertext> values;
            if (comm_rank == owner) {
                tags = owned_tags;
                values = owned_values;
            }
            bgv_mpi::BroadcastCiphertextBatch(tags, values, owner, comm);
            for (std::size_t k = 0; k < tags.size(); ++k) out[global_to_local.at(tags[k])] = values[k];
        }
    }

    // or_mat's owner-compute-dedup broadcast: `values[k]` must already hold
    // the correct result for every k whose owner(global_indices[k]) ==
    // comm_rank; every other entry is ignored on input and overwritten on
    // output. Recomputes the (cheap, communication-free) owner assignment
    // locally rather than threading it through from or_mat, to keep that
    // call site simple.
    void BroadcastOwnedBatch(std::vector<bgv_batched::Ciphertext>& values,
                              const std::vector<std::size_t>& global_indices, int comm_size, int comm_rank,
                              MPI_Comm comm) const {
        std::vector<std::size_t> owned_tags;
        std::vector<bgv_batched::Ciphertext> owned_values;
        for (std::size_t k = 0; k < global_indices.size(); ++k) {
            const auto owner = static_cast<int>(global_indices[k] % static_cast<std::size_t>(comm_size));
            if (owner == comm_rank) {
                owned_tags.push_back(global_indices[k]);
                owned_values.push_back(values[k]);
            }
        }
        GatherOwnedToAll(values, global_indices, owned_tags, owned_values, comm_size, comm_rank, comm);
    }

    static std::size_t EstimateCiphertextBatchBytes(const std::vector<std::size_t>& tags,
                                                      const std::vector<bgv_batched::Ciphertext>& cts) {
        return bgv_mpi::SerializeCiphertextBatch(tags, cts).size();
    }

    // ── Gather/scatter, parameterized over an explicit grid + root ────────
    //
    // Generalized versions of the root-relative gather/scatter used by
    // EncryptFromRoot/DecryptOnRoot, so matmul_pair can also use them to
    // reconcile a sub-grid's result back into grid_'s shape (see
    // matmul_pair's doc comment) -- `grid`/`local_root` there is a
    // temporary sub-grid and ITS rank 0, not grid_/kRootRank. Both batch all
    // N rows (and all N columns) into a single message instead of N
    // messages, same rationale as AssembleRows/AssembleCols above.

    // Meaningful only on `grid`-rank `local_root`; every other rank's return
    // value is a default-constructed PackedBooleanMatrix.
    //
    // Communicates over `grid.cart`, NOT `grid.world` -- `grid.world` is
    // just whatever communicator BuildProcessGrid2D was originally called
    // with, which the CALLER owns and may free as soon as BuildProcessGrid2D
    // returns (matmul_pair does exactly this with its temporary sub_comm,
    // see its doc comment). `grid.cart` is derived from it via
    // MPI_Cart_create with reorder=0, which guarantees identical rank
    // numbering to the input communicator (so grid.world_rank is equally
    // valid as "rank within cart"), and is owned/freed by ProcessGrid2D
    // itself -- always safe to use for as long as `grid` is alive.
    PackedBooleanMatrix GatherToRankWithinGrid(const DistributedRowColMatrix& enc, const ProcessGrid2D& grid,
                                                int local_root) const {
        std::vector<bgv_batched::Ciphertext> all_rows(n_), all_cols(n_);
        const auto& row_idx = enc.global_row_indices();
        const auto& col_idx = enc.global_col_indices();

        if (grid.world_rank == local_root) {
            for (std::size_t li = 0; li < row_idx.size(); ++li) all_rows[row_idx[li]] = enc.row_at(li);
            for (std::size_t lj = 0; lj < col_idx.size(); ++lj) all_cols[col_idx[lj]] = enc.col_at(lj);
            // Rows/cols are replicated within row_comm/col_comm, so
            // multiple source ranks may send the same global row/col
            // redundantly -- root simply keeps the last one received (all
            // are byte-identical in exact arithmetic; see or_mat's analogous
            // redundant-recompute tradeoff).
            for (int src = 0; src < grid.world_size; ++src) {
                if (src == local_root) continue;
                std::vector<std::size_t> recv_row_tags, recv_col_tags;
                std::vector<bgv_batched::Ciphertext> recv_rows, recv_cols;
                bgv_mpi::RecvCiphertextBatch(src, kGatherRowsTag, grid.cart, recv_row_tags, recv_rows);
                bgv_mpi::RecvCiphertextBatch(src, kGatherColsTag, grid.cart, recv_col_tags, recv_cols);
                for (std::size_t k = 0; k < recv_row_tags.size(); ++k) all_rows[recv_row_tags[k]] = recv_rows[k];
                for (std::size_t k = 0; k < recv_col_tags.size(); ++k) all_cols[recv_col_tags[k]] = recv_cols[k];
            }
        } else {
            bgv_mpi::SendCiphertextBatch(row_idx, enc.rows(), local_root, kGatherRowsTag, grid.cart);
            bgv_mpi::SendCiphertextBatch(col_idx, enc.cols(), local_root, kGatherColsTag, grid.cart);
        }

        if (grid.world_rank != local_root)
            return PackedBooleanMatrix{};
        return PackedBooleanMatrix(n_, std::move(all_rows), std::move(all_cols));
    }

    // `full_maybe_root_only` is meaningful only on `grid`-rank `local_root`.
    // Communicates over `grid.cart` -- see GatherToRankWithinGrid's doc
    // comment on why not `grid.world`.
    DistributedRowColMatrix ScatterFromRankToGrid(const PackedBooleanMatrix& full_maybe_root_only,
                                                   const ProcessGrid2D& grid, const BlockPartition& row_partition,
                                                   const BlockPartition& col_partition, int local_root) const {
        std::vector<std::size_t> row_tags, col_tags;
        std::vector<bgv_batched::Ciphertext> all_rows, all_cols;
        if (grid.world_rank == local_root) {
            row_tags.resize(n_);
            col_tags.resize(n_);
            std::iota(row_tags.begin(), row_tags.end(), std::size_t{0});
            std::iota(col_tags.begin(), col_tags.end(), std::size_t{0});
            all_rows = full_maybe_root_only.rows();
            all_cols = full_maybe_root_only.cols();
        }
        bgv_mpi::BroadcastCiphertextBatch(row_tags, all_rows, local_root, grid.cart);
        bgv_mpi::BroadcastCiphertextBatch(col_tags, all_cols, local_root, grid.cart);

        const std::size_t row_offset = row_partition.offsets[static_cast<std::size_t>(grid.process_row)];
        const std::size_t local_rows = row_partition.counts[static_cast<std::size_t>(grid.process_row)];
        const std::size_t col_offset = col_partition.offsets[static_cast<std::size_t>(grid.process_col)];
        const std::size_t local_cols = col_partition.counts[static_cast<std::size_t>(grid.process_col)];

        std::vector<std::size_t> row_idx(local_rows), col_idx(local_cols);
        std::vector<bgv_batched::Ciphertext> rows(local_rows), cols(local_cols);
        for (std::size_t li = 0; li < local_rows; ++li) {
            row_idx[li] = row_offset + li;
            rows[li] = all_rows[row_offset + li]; // positionally tag-ordered 0..n_-1, see broadcast above
        }
        for (std::size_t lj = 0; lj < local_cols; ++lj) {
            col_idx[lj] = col_offset + lj;
            cols[lj] = all_cols[col_offset + lj];
        }
        return DistributedRowColMatrix(n_, std::move(row_idx), std::move(rows), std::move(col_idx),
                                        std::move(cols));
    }

    // Root serializes each full row/column, broadcasts it to everyone, and
    // every rank keeps only the slice its (process_row, process_col) owns.
    // Only used once, at initial encryption, so its O(N) broadcast cost is
    // not on the recursive BTC hot path.
    DistributedRowColMatrix ScatterFullMatrixToBlocks(const PackedBooleanMatrix& full_maybe_root_only) const {
        return ScatterFromRankToGrid(full_maybe_root_only, grid_, row_partition_, col_partition_, kRootRank);
    }

    // Inverse of ScatterFullMatrixToBlocks, over grid_/kRootRank.
    PackedBooleanMatrix GatherToRoot(const DistributedRowColMatrix& enc) const {
        return GatherToRankWithinGrid(enc, grid_, kRootRank);
    }

    static constexpr int kAssembleRowsTag = 200;
    static constexpr int kAssembleColsTag = 201;
    static constexpr int kGatherRowsTag = 300;
    static constexpr int kGatherColsTag = 301;

    std::size_t n_;
    ProcessGrid2D grid_;
    bgv_batched::Context ctx_;
    bgv_mpi::SerializedContextBundle bundle_;
    std::unique_ptr<BGVRowColBackend> local_backend_;
    BlockPartition row_partition_;
    BlockPartition col_partition_;
    mutable OpCounters plain_counters_;
    mutable AtomicCounters atomic_counters_;
};

// ── Concurrent-branch bounded transitive closure ────────────────────────────
//
// Mirrors algorithms.hpp's sum_powers_recursive/bounded_transitive_closure
// EXACTLY (same recurrence, same base case), but is specific to
// BGVDistributedRowColBackend rather than generic over the Backend concept.
// algorithms.hpp is deliberately left unmodified -- it's shared by every
// backend in this project (see backend.hpp: "no changes to algorithms.hpp
// are required" when adding a backend) -- so it still issues the even-T
// step's two independent matmul calls (P(2h) = matmul(P_h,P_h) and the
// intermediate matmul(P_h,S_h)) sequentially through the generic Backend
// interface. This function is the backend-specific alternative that
// replaces that pair with ONE matmul_pair call, so the two run concurrently
// on disjoint MPI ranks (see matmul_pair's doc comment on
// BGVDistributedRowColBackend). The odd-T step has only one matmul, so
// there's no pairing opportunity there -- it's identical to
// sum_powers_recursive's.
inline std::pair<DistributedRowColMatrix, DistributedRowColMatrix>
DistributedSumPowersRecursive(const DistributedRowColMatrix& A, int T, const BGVDistributedRowColBackend& backend) {
    if (T < 1)
        throw std::invalid_argument("DistributedSumPowersRecursive: T must be >= 1");

    if (T == 1)
        return {backend.copy(A), backend.copy(A)};

    if (T % 2 == 0) {
        const int h = T / 2;
        auto [S_h, P_h] = DistributedSumPowersRecursive(A, h, backend);

        // second_half = matmul(P_h, S_h); P_T = matmul(P_h, P_h) -- both
        // depend only on P_h/S_h, never on each other, so matmul_pair runs
        // them concurrently instead of sequentially.
        auto [second_half, P_T] = backend.matmul_pair(P_h, S_h, P_h, P_h);
        auto S_T = backend.or_mat(S_h, second_half);
        return {std::move(S_T), std::move(P_T)};
    } else {
        auto [S_prev, P_prev] = DistributedSumPowersRecursive(A, T - 1, backend);

        auto P_T = backend.matmul(P_prev, A);
        auto S_T = backend.or_mat(S_prev, P_T);
        return {std::move(S_T), std::move(P_T)};
    }
}

inline DistributedRowColMatrix DistributedBoundedTransitiveClosure(const DistributedRowColMatrix& A, int T,
                                                                     const BGVDistributedRowColBackend& backend) {
    auto [S_T, P_T] = DistributedSumPowersRecursive(A, T, backend);
    (void)P_T;
    return std::move(S_T);
}

} // namespace btc

#endif // TC_WITH_BGV_MPI
