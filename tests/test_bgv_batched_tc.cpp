#ifdef TC_WITH_BGV_BATCHED

// ── Full transitive-closure correctness tests (BGV batched backend) ─────
//
// Compares btc::bounded_transitive_closure under BGVBatchedBackend against
// the plaintext Floyd-Warshall / bounded_transitive_closure reference,
// across the required graph set (N=1, empty, single edge, chain, cycle,
// disconnected components, DAG with multiple paths, already-closed graph,
// deterministic random small graphs).

#include <btc/algorithms.hpp>
#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/bgv_batched_profile_setup.hpp>

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

// Runs bounded_transitive_closure under BGVBatchedBackend and checks every
// decrypted entry is exactly {0,1} and matches the plaintext Floyd-Warshall
// reference exactly.
//
// Uses setup_from_profile (include/btc/bgv_batched_profile_setup.hpp)
// instead of picking a plaintext modulus by hand: it looks up the cheapest
// validated (depth, ring_dimension, plaintext_modulus) profile from
// config/bgv_batched_profiles.json for the required depth/slots. This is
// what replaced the old ad hoc "pass 786433 for the N=8 case" workaround
// (see docs/plaintext_modulus_issue.md for why that guessing was
// unreliable) -- if this test needs a bigger profile than the checked-in
// database provides, regenerate it with
// tools/generate_bgv_batched_profiles rather than hand-picking a modulus.
static void run_closure_test(const char* name, const btc::BoolMatrix& A, int T) {
    const std::size_t n = A.rows();
    const uint32_t depth = btc::EstimateBGVBatchedDepth(n, T);
    const uint64_t slots = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);

    auto ctx = bgv_batched::setup_from_profile(depth, slots);

    btc::BGVBatchedBackend backend(ctx, n);
    auto enc_A = backend.Encrypt(A);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = backend.Decrypt(enc_S); // throws if any slot isn't exactly {0,1}

    auto expected = btc::floyd_warshall(A);
    check(result == expected, name);
    std::printf("%s: PASS\n", name);
}

// ── Required graph 1: N = 1 ──────────────────────────────────────────────
static void test_n1() {
    btc::BoolMatrix A(1, 1, false);
    run_closure_test("n1 (single node, no self-loop)", A, 1);

    btc::BoolMatrix A_loop(1, 1, true);
    run_closure_test("n1 (single node, self-loop)", A_loop, 1);
}

// ── Required graph 2: empty graph (no edges) ─────────────────────────────
static void test_empty_graph() {
    btc::BoolMatrix A(4, 4, false);
    run_closure_test("empty_graph (no edges, N=4)", A, 4);
}

// ── Required graph 3: single directed edge ───────────────────────────────
static void test_single_edge() {
    auto A = from_flat(3, {
        false, true,  false,
        false, false, false,
        false, false, false
    });
    run_closure_test("single_edge (0->1 only)", A, 3);
}

// ── Required graph 4: directed path/chain ────────────────────────────────
static void test_chain() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, true,  false,
        false, false, false, true,
        false, false, false, false
    });
    run_closure_test("chain (0->1->2->3)", A, 3);
}

// ── Required graph 5: directed cycle ─────────────────────────────────────
static void test_cycle() {
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    run_closure_test("cycle (0->1->2->0)", A, 3);
}

// ── Required graph 6: disconnected components ────────────────────────────
static void test_disconnected() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, false, false,
        false, false, false, true,
        false, false, false, false
    });
    run_closure_test("disconnected (0->1, 2->3, no cross edges)", A, 4);
}

// ── Required graph 7: DAG with multiple paths ────────────────────────────
static void test_dag_multiple_paths() {
    // 0->1, 0->2, 1->3, 2->3: two length-2 paths converge on node 3.
    auto A = from_flat(4, {
        false, true,  true,  false,
        false, false, false, true,
        false, false, false, true,
        false, false, false, false
    });
    run_closure_test("dag_multiple_paths (0->1->3 and 0->2->3 converge)", A, 3);
}

// ── Required graph 8: already transitively closed ────────────────────────
static void test_already_closed() {
    // Complete DAG on 3 nodes (0->1, 0->2, 1->2) is already its own closure.
    auto A = from_flat(3, {
        false, true, true,
        false, false, true,
        false, false, false
    });
    auto expected = btc::floyd_warshall(A);
    check(expected == A, "test fixture sanity: graph must already equal its own closure");
    run_closure_test("already_closed (complete DAG on 3 nodes)", A, 3);
}

// ── Required graph 9: deterministic random small graphs ──────────────────
static std::mt19937_64 make_rng() { return std::mt19937_64(42); }

static void test_random_graphs() {
    auto rng = make_rng();
    std::bernoulli_distribution edge(0.35);

    for (std::size_t n : {2, 3, 4}) {
        btc::BoolMatrix A(n, n, false);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (i != j)
                    A.set(i, j, edge(rng));

        char name[64];
        std::snprintf(name, sizeof(name), "random_graph (N=%zu, seed=42)", n);
        run_closure_test(name, A, static_cast<int>(n));
    }
}

// ── Suggested sizes N=1,2,3,4,8: dedicated N=8 correctness check ─────────
//
// N=8 with T=8 needs a deeper circuit (see EstimateBGVBatchedDepth) than
// the smallest profiles in the database support -- no hardcoded plaintext
// modulus needed here anymore (contrast the old 786433 override this test
// used before setup_from_profile existed): run_closure_test's profile
// lookup picks whichever validated profile is big enough automatically.
static void test_n8_chain() {
    btc::BoolMatrix A(8, 8, false);
    for (std::size_t i = 0; i + 1 < 8; ++i)
        A.set(i, i + 1, true);
    run_closure_test("n8_chain (0->1->...->7)", A, 8);
}

int main() {
    test_n1();
    test_empty_graph();
    test_single_edge();
    test_chain();
    test_cycle();
    test_disconnected();
    test_dag_multiple_paths();
    test_already_closed();
    test_random_graphs();
    test_n8_chain();
    std::puts("\nAll BGV-batched transitive-closure tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
