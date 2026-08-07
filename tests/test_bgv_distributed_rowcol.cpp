#ifdef TC_WITH_BGV_MPI

// ── Correctness test for BGVDistributedRowColBackend (MPI + OpenMP) ────────
//
// Run under mpirun at 1, 2, and 4 ranks (see docs/bgv_rowcol.md and
// claude_distributed_bgv_rowcol_mpi.md's "Correctness Requirements"):
//
//   mpirun -n 1 ./build/tests/test_bgv_distributed_rowcol
//   mpirun -n 2 ./build/tests/test_bgv_distributed_rowcol
//   mpirun -n 4 ./build/tests/test_bgv_distributed_rowcol
//
// Not registered with ctest/add_btc_test (tests/CMakeLists.txt) because
// ctest invokes binaries directly, not through mpirun -- see
// tests/CMakeLists.txt's comment on this binary for how it's wired instead.
//
// Verifies, for N=4 T=4 (the task's required minimum coverage):
//   1. distributed decrypted result equals the plaintext reference;
//   2. distributed result equals the single-node OpenMPOptimized result;
//   3. row and column representations agree after every distributed matmul
//      (BGVRowColBackend::Decrypt's cross-check, reused via DecryptOnRoot);
//   4/5. covered by (1)/(2) -- a mismatch there implies a row or column
//      decrypted to the wrong values;
//   6. no Cij computed twice: cross-checked via operation counters (total
//      dot_products across all ranks must equal N^2 per matmul call, see
//      the reduction below), not by re-deriving correctness independently.
//
// Also verifies the Stage 3 concurrent-branch path (matmul_pair via
// DistributedBoundedTransitiveClosure) reproduces the exact same decrypted
// result as the sequential distributed path, with the same total dot_products
// -- i.e. splitting grid_'s world in half to run the recurrence's two
// independent even-T matmuls concurrently changes WHICH ranks do the work,
// not WHAT gets computed.

#include <btc/algorithms.hpp>
#include <btc/bgv_distributed_rowcol_backend.hpp>
#include <btc/bgv_rowcol_backend.hpp>
#include <btc/graph_io.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

void check(bool cond, const char* what, int rank) {
    if (!cond) {
        std::fprintf(stderr, "[rank %d] FAIL: %s\n", rank, what);
        ++g_failures;
    } else {
        std::fprintf(stderr, "[rank %d] ok: %s\n", rank, what);
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::size_t N = 4;
    const int T = 4;

    btc::BoolMatrix A;
    if (rank == 0) {
        A = btc::load_graph("data/graph_N4.txt");
    }

    const uint32_t depth = btc::EstimateBGVRowColDepth(N, T);

    // Everything that touches MPI communicators (BGVDistributedRowColBackend
    // owns a ProcessGrid2D whose destructor calls MPI_Comm_free) must be
    // destroyed BEFORE MPI_Finalize() runs below -- C++ destructs locals at
    // the end of their enclosing scope, not at the MPI_Finalize() call site,
    // so without this nested block `backend`'s destructor would run AFTER
    // MPI_Finalize() (at main's closing brace) and crash with "Attempting to
    // use an MPI routine ... after finalizing MPICH". Verified empirically:
    // an earlier version of this test didn't scope this and crashed exactly
    // that way after all correctness checks had already passed.
    {
        // Every rank constructs the backend identically (SPMD) -- see
        // BGVDistributedRowColBackend's constructor doc comment for why only
        // rank 0's Params actually drive context creation.
        btc::BGVDistributedRowColBackend backend(MPI_COMM_WORLD, N, depth);

        check(backend.world_size() == world_size, "world_size matches MPI_COMM_WORLD size", rank);
        if (rank == 0) {
            std::fprintf(stderr, "[rank 0] grid: Pr=%d Pc=%d (world_size=%d)\n", backend.Pr(), backend.Pc(),
                         world_size);
        }

        auto enc_A = backend.EncryptFromRoot(A);
        backend.reset_counters();
        auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
        auto dist_result = backend.DecryptOnRoot(enc_S);

        // Reduce dot_products across all ranks -- must equal N^2 PER matmul
        // call. sum_powers_recursive(A, 4) issues exactly 4 matmul calls (see
        // the T=4 trace this project's earlier analysis derived), so the
        // cross-rank total should be 4 * N^2 = 4*16 = 64, matching
        // OpenMPOptimized's per-process dot_products count for the same N,T
        // (see docs -- OpenMPOptimized already established N^2 per matmul, not
        // 2*N^2).
        auto local_counters = backend.counters();
        std::size_t local_dot_products = local_counters.dot_products;
        std::size_t total_dot_products = 0;
        MPI_Reduce(&local_dot_products, &total_dot_products, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto plain_S = btc::bounded_transitive_closure(A, T);
            check(dist_result == plain_S, "distributed result matches plaintext reference", rank);

            const std::size_t expected_dot_products = 4 * N * N; // 4 matmul calls at N=4,T=4
            std::fprintf(stderr, "[rank 0] total dot_products across all ranks = %zu (expected %zu)\n",
                         total_dot_products, expected_dot_products);
            check(total_dot_products == expected_dot_products, "no Cij computed more than once globally", rank);
        }

        // Cross-check against the single-node OpenMPOptimized backend, using
        // a SEPARATE independent crypto context (not shared with the
        // distributed one) -- only rank 0 needs to run this, since it's a
        // plaintext-visible comparison after decryption.
        if (rank == 0) {
            btc::bgv_batched::Params params;
            params.plaintext_modulus = 786433;
            params.multiplicative_depth = depth;
            params.batch_size = static_cast<uint32_t>(N);
            auto ctx = btc::bgv_batched::setup(params);
            btc::BGVRowColBackend single_node(ctx, N, btc::BGVMatMulBackend::OpenMPOptimized);
            auto single_enc_A = single_node.Encrypt(A);
            auto single_enc_S = btc::bounded_transitive_closure(single_enc_A, T, single_node);
            auto single_result = single_node.Decrypt(single_enc_S);
            check(dist_result == single_result, "distributed result matches single-node OpenMPOptimized result",
                  rank);
        }

        // ── Stage 3: concurrent-branch recursion (matmul_pair) ────────────
        //
        // DistributedSumPowersRecursive/DistributedBoundedTransitiveClosure
        // (bgv_distributed_rowcol_backend.hpp) replace the sequential
        // algorithms.hpp recurrence's two independent even-T matmul calls
        // with ONE matmul_pair call that splits grid_'s world communicator
        // in half and runs them concurrently. Must produce EXACTLY the same
        // result as the sequential path above -- matmul_pair's world-split
        // and grid-reconciliation machinery must not change what gets
        // computed, only how many MPI ranks work on it at once. Every rank
        // must call this identically (SPMD, matmul_pair is collective over
        // grid_.world).
        backend.reset_counters();
        auto enc_S_concurrent = btc::DistributedBoundedTransitiveClosure(enc_A, T, backend);
        auto dist_result_concurrent = backend.DecryptOnRoot(enc_S_concurrent);

        auto concurrent_counters = backend.counters();
        std::size_t local_dot_products_concurrent = concurrent_counters.dot_products;
        std::size_t total_dot_products_concurrent = 0;
        MPI_Reduce(&local_dot_products_concurrent, &total_dot_products_concurrent, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        if (rank == 0) {
            check(dist_result_concurrent == dist_result,
                  "matmul_pair concurrent-branch result matches sequential distributed result", rank);
            const std::size_t expected_dot_products = 4 * N * N; // same 4 matmul-equivalents as the sequential path
            std::fprintf(stderr,
                         "[rank 0] concurrent-branch total dot_products across all ranks = %zu (expected %zu)\n",
                         total_dot_products_concurrent, expected_dot_products);
            check(total_dot_products_concurrent == expected_dot_products,
                  "matmul_pair: no Cij computed more than once globally, across both branches", rank);
        }
    }

    std::fprintf(stderr, "[rank %d] scope exited (backend destroyed), entering final MPI_Reduce...\n", rank);
    std::fflush(stderr);

    int global_failures = 0;
    MPI_Reduce(&g_failures, &global_failures, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_failures, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::fprintf(stderr, "[rank %d] final reduce/bcast done, calling MPI_Finalize...\n", rank);
    std::fflush(stderr);
    MPI_Finalize();
    std::fprintf(stderr, "[rank %d] MPI_Finalize done, returning.\n", rank);
    std::fflush(stderr);
    return global_failures == 0 ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_MPI=ON (and MPI + OpenFHE installed) to enable this test.");
    return 0;
}
#endif
