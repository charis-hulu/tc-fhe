#include <btc/algorithms.hpp>

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <random>

static btc::BoolMatrix from_flat(std::size_t n,
                                  std::initializer_list<bool> data) {
    btc::BoolMatrix m(n, n);
    auto it = data.begin();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m.set(i, j, *it++);
    return m;
}

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

// Upper-triangular random matrix (DAG — no cycles).
static btc::BoolMatrix random_dag(std::size_t n, double density,
                                   std::mt19937& rng) {
    std::bernoulli_distribution edge(density);
    btc::BoolMatrix A(n, n, false);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (edge(rng)) A.set(i, j, true);
    return A;
}

// General random matrix (may contain cycles).
static btc::BoolMatrix random_matrix(std::size_t n, double density,
                                      std::mt19937& rng) {
    std::bernoulli_distribution edge(density);
    btc::BoolMatrix A(n, n, false);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j && edge(rng)) A.set(i, j, true);
    return A;
}

static void test_T1_recovers_A() {
    // BTC with T=1 must return the adjacency matrix itself.
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        false, false, false
    });
    check(btc::bounded_transitive_closure(A, 1) == A, "T=1 recovers A");
    std::puts("T1_recovers_A: PASS");
}

static void test_zero_input() {
    btc::BoolMatrix Z(5, 5, false);
    check(btc::bounded_transitive_closure(Z, 10) == Z, "all-zero stays zero");
    std::puts("zero_input: PASS");
}

static void test_dag_vs_floyd_warshall() {
    // For a DAG on N nodes, any simple path has length <= N-1.
    // There are no cycles, so BTC(A, N-1) == Floyd-Warshall(A).
    std::mt19937 rng(42);
    const std::size_t N = 8;
    const int T = static_cast<int>(N) - 1;

    for (int trial = 0; trial < 20; ++trial) {
        auto A   = random_dag(N, 0.3, rng);
        auto btc = btc::bounded_transitive_closure(A, T);
        auto fw  = btc::floyd_warshall(A);
        check(btc == fw, "DAG: BTC(A, N-1) == Floyd-Warshall");
    }
    std::puts("dag_vs_floyd_warshall (20 trials, N=8, T=N-1): PASS");
}

static void test_general_vs_floyd_warshall() {
    // For a general directed graph on N nodes:
    //   - Any simple path (i->j, i!=j) has length <= N-1
    //   - Any simple cycle has length <= N
    // Therefore BTC(A, N) == Floyd-Warshall(A) for every directed graph.
    std::mt19937 rng(123);
    const std::size_t N = 6;
    const int T = static_cast<int>(N); // T = N covers simple paths and cycles

    for (int trial = 0; trial < 30; ++trial) {
        auto A   = random_matrix(N, 0.25, rng);
        auto btc = btc::bounded_transitive_closure(A, T);
        auto fw  = btc::floyd_warshall(A);
        check(btc == fw, "general: BTC(A, N) == Floyd-Warshall");
    }
    std::puts("general_vs_floyd_warshall (30 trials, N=6, T=N): PASS");
}

static void test_3cycle_known() {
    // 0->1->2->0
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    auto fw   = btc::floyd_warshall(A);
    auto btc3 = btc::bounded_transitive_closure(A, 3); // T = N = 3

    // Full transitive closure is all-ones
    check(btc3 == fw, "3-cycle: BTC(T=3) == Floyd-Warshall");

    // T=2 cannot detect self-reachability through the length-3 cycle
    auto btc2 = btc::bounded_transitive_closure(A, 2);
    check(!btc2.get(0, 0) && !btc2.get(1, 1) && !btc2.get(2, 2),
          "3-cycle: BTC(T=2) cannot see self-loops via 3-cycle");

    std::puts("3-cycle_known: PASS");
}

static void test_chain_known() {
    // 0->1->2->3 (4-node chain)
    // Floyd-Warshall: reachable[i][j] = 1 iff i < j
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, true,  false,
        false, false, false, true,
        false, false, false, false
    });
    auto fw    = btc::floyd_warshall(A);
    auto btc3  = btc::bounded_transitive_closure(A, 3); // N-1 = 3

    check(btc3 == fw, "4-chain: BTC(T=3) == Floyd-Warshall");

    // T=1: only direct edges
    auto btc1 = btc::bounded_transitive_closure(A, 1);
    check(btc1 == A, "4-chain: BTC(T=1) == A");

    // T=2: paths of length <= 2
    auto btc2 = btc::bounded_transitive_closure(A, 2);
    // 0->2 reachable in 2 hops, but 0->3 requires 3 hops
    check(btc2.get(0, 1) && btc2.get(0, 2), "4-chain: 0->1, 0->2 reachable at T=2");
    check(!btc2.get(0, 3), "4-chain: 0->3 NOT reachable at T=2");
    check(!btc2.get(1, 3), "4-chain: 1->3 NOT reachable at T=2");

    std::puts("chain_known: PASS");
}

int main() {
    test_T1_recovers_A();
    test_zero_input();
    test_dag_vs_floyd_warshall();
    test_general_vs_floyd_warshall();
    test_3cycle_known();
    test_chain_known();
    std::puts("\nAll transitive closure tests passed.");
    return 0;
}
