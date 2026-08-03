#ifdef TC_WITH_BGV

#include <btc/algorithms.hpp>
#include <btc/bgv.hpp>
#include <btc/bgv_backend.hpp>

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace bgv = btc::bgv;

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

// Runs bounded_transitive_closure under BGV and checks it matches the
// plaintext Floyd-Warshall / bounded_transitive_closure reference exactly.
static void run_closure_test(const char* name, const btc::BoolMatrix& A, int T) {
    const uint32_t depth = btc::EstimateBGVDepth(A.rows(), T);

    bgv::Params params;
    params.multiplicative_depth = depth;
    auto ctx = bgv::setup(params);

    auto enc_A = btc::BGVBackend::encrypt(A, ctx);
    btc::BGVBackend backend(ctx);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = btc::BGVBackend::decrypt(enc_S, ctx);

    auto expected = btc::bounded_transitive_closure(A, T);
    check(result == expected, name);
    std::printf("%s: PASS\n", name);
}

// ── Test 5: Gate truth tables (encrypted AND / OR over all input combos) ───
static void test_gate_truth_tables() {
    bgv::Params params;
    params.multiplicative_depth = 2; // one level for AND/OR, one for XOR's 2*a*b term
    auto ctx = bgv::setup(params);

    const bool values[2] = {false, true};
    for (bool a : values) {
        for (bool b : values) {
            auto ca = bgv::encrypt_bit(ctx, a);
            auto cb = bgv::encrypt_bit(ctx, b);

            auto c_and = bgv::encrypted_and(ctx, ca, cb);
            check(bgv::decrypt_bit(ctx, c_and) == (a && b), "AND truth table entry");

            auto c_or = bgv::encrypted_or(ctx, ca, cb);
            check(bgv::decrypt_bit(ctx, c_or) == (a || b), "OR truth table entry");

            auto c_xor = bgv::encrypted_xor(ctx, ca, cb);
            check(bgv::decrypt_bit(ctx, c_xor) == (a != b), "XOR truth table entry");
        }
    }
    std::puts("gate_truth_tables (AND/OR/XOR, all 4 input combos): PASS");
}

// ── Encrypt/decrypt round trip for both bits ────────────────────────────────
static void test_encrypt_decrypt_bits() {
    bgv::Params params;
    params.multiplicative_depth = 1;
    auto ctx = bgv::setup(params);

    for (int trial = 0; trial < 5; ++trial) {
        auto ct0 = bgv::encrypt_bit(ctx, false);
        check(bgv::decrypt_bit(ctx, ct0) == false, "encrypt/decrypt 0");

        auto ct1 = bgv::encrypt_bit(ctx, true);
        check(bgv::decrypt_bit(ctx, ct1) == true, "encrypt/decrypt 1");
    }
    std::puts("encrypt_decrypt_bits (5 trials): PASS");
}

// ── Test 1: Simple chain 0->1->2->3 ─────────────────────────────────────────
static void test_simple_chain() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, true,  false,
        false, false, false, true,
        false, false, false, false
    });
    const int T = 3; // N-1

    const uint32_t depth = btc::EstimateBGVDepth(A.rows(), T);
    bgv::Params params;
    params.multiplicative_depth = depth;
    auto ctx = bgv::setup(params);

    auto enc_A = btc::BGVBackend::encrypt(A, ctx);
    btc::BGVBackend backend(ctx);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = btc::BGVBackend::decrypt(enc_S, ctx);

    auto expected = btc::floyd_warshall(A);
    check(result == expected, "simple_chain matches Floyd-Warshall");
    check(result.get(0, 2), "0->2 reachable");
    check(result.get(0, 3), "0->3 reachable");
    check(result.get(1, 3), "1->3 reachable");
    std::puts("simple_chain (0->1->2->3): PASS");
}

// ── Test 2: Two paths converging on the same vertex ─────────────────────────
//
// This is the test that ordinary arithmetic matrix multiplication mod 2
// would get wrong: node 3 is reached via two distinct length-2 paths
// (0->1->3 and 0->2->3), and 1+1 = 0 mod 2 would incorrectly cancel them.
// Boolean OR (a+b-ab) must instead report 1 OR 1 = 1.
static void test_two_paths_same_vertex() {
    auto A = from_flat(4, {
        false, true,  true,  false,
        false, false, false, true,
        false, false, false, true,
        false, false, false, false
    });
    const int T = 3;

    const uint32_t depth = btc::EstimateBGVDepth(A.rows(), T);
    bgv::Params params;
    params.multiplicative_depth = depth;
    auto ctx = bgv::setup(params);

    auto enc_A = btc::BGVBackend::encrypt(A, ctx);
    btc::BGVBackend backend(ctx);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = btc::BGVBackend::decrypt(enc_S, ctx);

    auto expected = btc::floyd_warshall(A);
    check(result == expected, "two_paths_same_vertex matches Floyd-Warshall");
    check(result.get(0, 3), "vertex 3 remains reachable from 0 despite two converging paths");
    std::puts("two_paths_same_vertex (0->1->3 and 0->2->3): PASS");
}

// ── Test 3: Disconnected graph -- no false cross-component reachability ────
static void test_disconnected_graph() {
    // Component A: 0->1. Component B: 2->3. No edges between components.
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, false, false,
        false, false, false, true,
        false, false, false, false
    });
    const int T = 3;
    run_closure_test("disconnected_graph", A, T);

    // Extra explicit check: no path crosses components.
    const uint32_t depth = btc::EstimateBGVDepth(A.rows(), T);
    bgv::Params params;
    params.multiplicative_depth = depth;
    auto ctx = bgv::setup(params);
    auto enc_A = btc::BGVBackend::encrypt(A, ctx);
    btc::BGVBackend backend(ctx);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = btc::BGVBackend::decrypt(enc_S, ctx);
    check(!result.get(0, 2) && !result.get(0, 3) && !result.get(1, 2) && !result.get(1, 3),
          "component A cannot reach component B");
    check(!result.get(2, 0) && !result.get(2, 1) && !result.get(3, 0) && !result.get(3, 1),
          "component B cannot reach component A");
}

// ── Test 4: Cycle ────────────────────────────────────────────────────────────
static void test_cycle() {
    // 3-node cycle: 0->1->2->0.
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    const int T = 3; // N, sufficient to see the full cycle wrap around
    run_closure_test("cycle", A, T);
}

int main() {
    test_encrypt_decrypt_bits();
    test_gate_truth_tables();
    test_simple_chain();
    test_two_paths_same_vertex();
    test_disconnected_graph();
    test_cycle();
    std::puts("\nAll BGV tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV=ON (and OpenFHE installed) to enable BGV tests.");
    return 0;
}
#endif
