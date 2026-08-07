#ifdef TC_WITH_BGV_BATCHED

// ── Full transitive-closure correctness tests (BGV row/column-packed) ───
//
// Compares btc::bounded_transitive_closure under BGVRowColBackend against
// the plaintext Floyd-Warshall reference, integrating the backend into the
// SAME algorithms.hpp recursion used by every other backend (no changes to
// algorithms.hpp) -- see step 10 of the row/column-packed backend spec.
//
// Restricted to power-of-two N (this backend's batch_size == N invariant
// forbids anything else -- see bgv_rowcol_backend.hpp), so the graph set
// here differs from test_bgv_batched_tc.cpp / test_bgv.cpp only in that
// N=3 is replaced by N=4 and N=1 keeps its BOTH self-loop cases.

#include <btc/algorithms.hpp>
#include <btc/bgv_batched.hpp>
#include <btc/bgv_rowcol_backend.hpp>

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <random>

namespace bgv_batched = btc::bgv_batched;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static btc::BoolMatrix from_flat(std::size_t n, std::initializer_list<bool> data) {
    btc::BoolMatrix m(n, n);
    auto it = data.begin();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m.set(i, j, *it++);
    return m;
}

static void run_closure_test(const char* name, const btc::BoolMatrix& A, int T,
                              btc::BGVMatMulBackend variant = btc::BGVMatMulBackend::OpenMPTwoPass) {
    const std::size_t n = A.rows();
    const uint32_t depth = btc::EstimateBGVRowColDepth(n, T);

    bgv_batched::Params params;
    // 786433 (not the Params default of 65537) -- this backend's SumOR
    // re-mask (see bgv_rowcol_backend.hpp) roughly doubles multiplicative
    // depth vs. the other backends, which pushes OpenFHE's automatically
    // chosen ring dimension higher than 65537 stays NTT-compatible with
    // ((t-1) % (2*ring_dim) == 0 fails once ring_dim reaches 65536). 786433
    // is the same batching-friendly prime already used throughout
    // config/bgv_batched_profiles.json, compatible with every power-of-two
    // ring dimension up to 131072 (786432 = 2^18 * 3).
    params.plaintext_modulus = 786433;
    params.multiplicative_depth = depth;
    params.batch_size = static_cast<uint32_t>(n);
    auto ctx = bgv_batched::setup(params);

    btc::BGVRowColBackend backend(ctx, n, variant);
    auto enc_A = backend.Encrypt(A);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = backend.Decrypt(enc_S); // throws if row/col views disagree or any slot isn't {0,1}

    auto expected = btc::floyd_warshall(A);
    check(result == expected, name);
    std::printf("%s: PASS\n", name);
}

// ── N=1 (a power of two: 2^0) ─────────────────────────────────────────────
static void test_n1() {
    btc::BoolMatrix A(1, 1, false);
    run_closure_test("n1 (single node, no self-loop)", A, 1, btc::BGVMatMulBackend::Serial);

    btc::BoolMatrix A_loop(1, 1, true);
    run_closure_test("n1 (single node, self-loop)", A_loop, 1, btc::BGVMatMulBackend::OpenMPOptimized);
}

// ── N=2 cycle, exercised across all three backend variants ──────────────
static void test_n2_cycle_all_variants() {
    auto A = from_flat(2, {false, true, true, false}); // 0->1->0
    run_closure_test("n2_cycle [Serial]", A, 2, btc::BGVMatMulBackend::Serial);
    run_closure_test("n2_cycle [OpenMPTwoPass]", A, 2, btc::BGVMatMulBackend::OpenMPTwoPass);
    run_closure_test("n2_cycle [OpenMPOptimized]", A, 2, btc::BGVMatMulBackend::OpenMPOptimized);
}

static void test_empty_graph() {
    btc::BoolMatrix A(4, 4, false);
    run_closure_test("empty_graph (no edges, N=4)", A, 4);
}

static void test_single_edge() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, false, false,
        false, false, false, false,
        false, false, false, false
    });
    run_closure_test("single_edge (0->1 only)", A, 4);
}

static void test_chain() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, true,  false,
        false, false, false, true,
        false, false, false, false
    });
    run_closure_test("chain (0->1->2->3)", A, 3);
}

static void test_disconnected() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, false, false,
        false, false, false, true,
        false, false, false, false
    });
    run_closure_test("disconnected (0->1, 2->3, no cross edges)", A, 4);
}

static void test_dag_multiple_paths() {
    // 0->1, 0->2, 1->3, 2->3: two length-2 paths converge on node 3 -- the
    // same "two witnesses must collapse to 1" case as
    // test_bgv_rowcol_matmul.cpp, now exercised through full closure.
    auto A = from_flat(4, {
        false, true,  true,  false,
        false, false, false, true,
        false, false, false, true,
        false, false, false, false
    });
    run_closure_test("dag_multiple_paths (0->1->3 and 0->2->3 converge)", A, 3);
}

static void test_already_closed() {
    // Complete DAG on 4 nodes is already its own closure.
    auto A = from_flat(4, {
        false, true, true, true,
        false, false, true, true,
        false, false, false, true,
        false, false, false, false
    });
    auto expected = btc::floyd_warshall(A);
    check(expected == A, "test fixture sanity: graph must already equal its own closure");
    run_closure_test("already_closed (complete DAG on 4 nodes)", A, 4);
}

static void test_random_graph() {
    std::mt19937_64 rng(42);
    std::bernoulli_distribution edge(0.35);

    const std::size_t n = 4;
    btc::BoolMatrix A(n, n, false);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j)
                A.set(i, j, edge(rng));

    run_closure_test("random_graph (N=4, seed=42)", A, static_cast<int>(n));
}

static void test_n8_chain() {
    btc::BoolMatrix A(8, 8, false);
    for (std::size_t i = 0; i + 1 < 8; ++i)
        A.set(i, i + 1, true);
    run_closure_test("n8_chain (0->1->...->7)", A, 8);
}

int main() {
    // See test_bgv_rowcol_matmul.cpp's main() for why this is capped: these
    // are small correctness fixtures (N <= 8), not a thread-scaling
    // benchmark.
    omp_set_num_threads(std::min(4, omp_get_max_threads()));

    test_n1();
    test_n2_cycle_all_variants();
    test_empty_graph();
    test_single_edge();
    test_chain();
    test_disconnected();
    test_dag_multiple_paths();
    test_already_closed();
    test_random_graph();
    test_n8_chain();
    std::puts("\nAll BGV row/column-packed transitive-closure tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
