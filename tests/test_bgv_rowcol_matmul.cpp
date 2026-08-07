#ifdef TC_WITH_BGV_BATCHED

// ── Row/column-packed Boolean matrix multiplication tests ────────────────
//
// Independently tests btc::BGVRowColBackend::matmul across all three
// BGVMatMulBackend variants (Serial, OpenMPTwoPass, OpenMPOptimized) BEFORE
// it's used inside full transitive closure (see test_bgv_rowcol_tc.cpp).
//
// Per the task's explicit requirement, this includes a case that
// distinguishes Boolean OR from arithmetic summation: two witnesses must
// collapse to 1, never to 2. This is the correctness property the whole
// "rotate-and-OR-reduce instead of rotate-and-sum" design exists to
// guarantee (see bgv_rowcol_backend.hpp's file header).

#include <btc/bgv_batched.hpp>
#include <btc/bgv_rowcol_backend.hpp>
#include <btc/matrix.hpp>

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

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

// Plaintext Boolean matmul oracle: C[i][j] = OR_k(A[i][k] AND B[k][j]).
static btc::BoolMatrix plaintext_bool_matmul(const btc::BoolMatrix& A, const btc::BoolMatrix& B) {
    const std::size_t n = A.rows();
    btc::BoolMatrix C(n, n, false);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            bool acc = false;
            for (std::size_t k = 0; k < n; ++k)
                acc = acc || (A.get(i, k) && B.get(k, j));
            C.set(i, j, acc);
        }
    return C;
}

static const char* variant_name(btc::BGVMatMulBackend v) {
    switch (v) {
        case btc::BGVMatMulBackend::Serial: return "Serial";
        case btc::BGVMatMulBackend::OpenMPTwoPass: return "OpenMPTwoPass";
        case btc::BGVMatMulBackend::OpenMPOptimized: return "OpenMPOptimized";
    }
    return "unknown";
}

static void run_matmul_test(const char* name, const btc::BoolMatrix& A, const btc::BoolMatrix& B,
                             btc::BGVMatMulBackend variant) {
    const std::size_t n = A.rows();

    bgv_batched::Params params;
    params.multiplicative_depth = btc::BGVRowColMatmulDepth(n);
    params.batch_size = static_cast<uint32_t>(n); // required: batch_size == N exactly
    auto ctx = bgv_batched::setup(params);

    btc::BGVRowColBackend backend(ctx, n, variant);
    auto enc_A = backend.Encrypt(A);
    auto enc_B = backend.Encrypt(B);
    auto enc_C = backend.matmul(enc_A, enc_B);
    auto result = backend.Decrypt(enc_C); // also cross-checks row/col agreement

    auto expected = plaintext_bool_matmul(A, B);
    check(result == expected, name);
    std::printf("[%s] %s: PASS\n", variant_name(variant), name);
}

static void run_matmul_test_all_variants(const char* name, const btc::BoolMatrix& A, const btc::BoolMatrix& B) {
    run_matmul_test(name, A, B, btc::BGVMatMulBackend::Serial);
    run_matmul_test(name, A, B, btc::BGVMatMulBackend::OpenMPTwoPass);
    run_matmul_test(name, A, B, btc::BGVMatMulBackend::OpenMPOptimized);
}

// ── IsPowerOfTwo / rotation-index sanity checks ──────────────────────────
static void test_is_power_of_two() {
    check(!btc::IsPowerOfTwo(0), "IsPowerOfTwo(0) must be false");
    check(btc::IsPowerOfTwo(1), "IsPowerOfTwo(1) must be true");
    check(btc::IsPowerOfTwo(2), "IsPowerOfTwo(2) must be true");
    check(!btc::IsPowerOfTwo(3), "IsPowerOfTwo(3) must be false");
    check(btc::IsPowerOfTwo(4), "IsPowerOfTwo(4) must be true");
    check(!btc::IsPowerOfTwo(6), "IsPowerOfTwo(6) must be false");
    check(btc::IsPowerOfTwo(8), "IsPowerOfTwo(8) must be true");
    check(btc::IsPowerOfTwo(16), "IsPowerOfTwo(16) must be true");

    // Each power-of-two shift needs BOTH the forward offset and its
    // wrap-compensating offset (shift - N) -- see RotateModN in
    // bgv_rowcol_backend.hpp for why a single EvalRotate isn't enough at
    // batch_size == N.
    auto idx4 = btc::BuildRequiredRotationIndices(4);
    check((idx4 == std::vector<int32_t>{1, -3, 2, -2}),
          "BuildRequiredRotationIndices(4) must be {1, -3, 2, -2}");
    auto idx8 = btc::BuildRequiredRotationIndices(8);
    check((idx8 == std::vector<int32_t>{1, -7, 2, -6, 4, -4}),
          "BuildRequiredRotationIndices(8) must be {1, -7, 2, -6, 4, -4}");

    std::puts("IsPowerOfTwo / BuildRequiredRotationIndices: PASS");
}

// Rejecting non-power-of-two N is required (no padding implemented).
//
// OpenFHE's own BGV RNS parameter generation independently enforces "batch
// size can only be set to zero (full packing) or a power of two"
// (bgvrns-parametergeneration.cpp), so bgv_batched::setup(params) itself
// throws (lbcrypto::config_error, which derives from std::runtime_error)
// before BGVRowColBackend's constructor ever runs for this batch_size=3
// case. Either layer rejecting it is an acceptable outcome here -- this
// test only asserts that a non-power-of-two N can never produce a live
// backend instance, not which layer catches it first.
static void test_rejects_non_power_of_two() {
    bool threw = false;
    try {
        bgv_batched::Params params;
        params.multiplicative_depth = 1;
        params.batch_size = 3; // deliberately NOT a power of two
        auto ctx = bgv_batched::setup(params);
        btc::BGVRowColBackend backend(ctx, 3);
        (void)backend;
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "constructing a N=3 (not a power of two) row/col backend must throw somewhere");
    std::puts("rejects_non_power_of_two: PASS");
}

// ── No witness: A*B should be the all-zero matrix ───────────────────────
static void test_no_witness() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, false, false,
        false, false, false, false,
        false, false, false, false
    });
    auto B = from_flat(4, {
        false, false, false, false,
        false, false, false, false,
        true,  false, false, false,
        false, false, false, false
    });
    run_matmul_test_all_variants("no_witness (A*B is all-zero)", A, B);

    btc::BoolMatrix Z(4, 4, false);
    run_matmul_test_all_variants("no_witness (all-zero inputs)", Z, Z);
}

// ── Exactly one witness ──────────────────────────────────────────────────
static void test_one_witness() {
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, false, false,
        false, false, false, false,
        false, false, false, false
    });
    auto B = from_flat(4, {
        false, false, false, false,
        false, false, true,  false,
        false, false, false, false,
        false, false, false, false
    });
    run_matmul_test_all_variants("one_witness (single k contributes)", A, B);
}

// ── The core OR-vs-sum distinguishing case ───────────────────────────────
//
// A row    = [1,1,0,0]
// B column = [1,1,0,0]
// Boolean dot product must be 1 (OR of two true AND-terms), NOT 2 (their
// arithmetic sum). This is exactly the failure mode an arithmetic
// RotateAndSum would produce -- see bgv_rowcol_backend.hpp's file header.
static void test_or_not_sum_two_witnesses() {
    auto A = from_flat(4, {
        true,  true,  false, false,
        false, false, false, false,
        false, false, false, false,
        false, false, false, false
    });
    auto B = from_flat(4, {
        true,  false, false, false,
        true,  false, false, false,
        false, false, false, false,
        false, false, false, false
    });
    // C[0][0] = OR(A[0][0]&&B[0][0], A[0][1]&&B[1][0], ...) = OR(1&&1, 1&&1, 0, 0) = 1, not 2.
    run_matmul_test_all_variants("or_not_sum_two_witnesses (two paths converge, must collapse to 1)", A, B);

    // Denser case: every row of A and every column of B is all-ones, so
    // every output entry has N=8 witnesses -- maximal witness-count stress
    // test for SumOR's rotate-and-OR-reduce tree.
    btc::BoolMatrix Ones(8, 8, true);
    run_matmul_test_all_variants("or_not_sum_dense (all-ones, N=8 witnesses per entry)", Ones, Ones);
}

// ── Self-product of an identity-like matrix (sanity check) ──────────────
static void test_identity_self_product() {
    auto I = btc::BoolMatrix::identity(4);
    run_matmul_test_all_variants("identity_self_product (I*I == I)", I, I);
}

// ── N=2 minimal case ──────────────────────────────────────────────────────
static void test_n2() {
    auto A = from_flat(2, {true, true, false, true});
    auto B = from_flat(2, {true, false, true, true});
    run_matmul_test_all_variants("n2 (minimal power-of-two size)", A, B);
}

int main() {
    // These are tiny (N <= 8) correctness fixtures, not a scalability
    // benchmark -- cap the OpenMP thread count so OpenMPTwoPass/OpenMPOptimized
    // don't try to spawn omp_get_max_threads() (e.g. hundreds, on a
    // many-core machine) OS threads just to cover a handful of loop
    // iterations. Real thread-count scaling is what benchmarks/bench_bgv_rowcol's
    // --threads flag is for (see the required experimental plan).
    omp_set_num_threads(std::min(4, omp_get_max_threads()));

    test_is_power_of_two();
    test_rejects_non_power_of_two();
    test_n2();
    test_no_witness();
    test_one_witness();
    test_or_not_sum_two_witnesses();
    test_identity_self_product();
    std::puts("\nAll BGV row/column-packed matmul tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
