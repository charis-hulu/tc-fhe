#ifdef TC_WITH_BGV_BATCHED

// ── Packed Boolean matrix multiplication tests ───────────────────────────
//
// Independently tests btc::BGVBatchedBackend::matmul (which dispatches to
// BooleanMatrixMultiplyPacked) BEFORE it's used inside full transitive
// closure (see test_bgv_batched_tc.cpp). Per the task's explicit
// requirement, this includes cases where an output entry has no witness,
// exactly one witness, and multiple witnesses -- the multiple-witness case
// verifies the balanced OR-tree reduction returns Boolean 1 rather than a
// path/witness count (which plain summation would incorrectly produce).

#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/matrix.hpp>

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

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

static void run_matmul_test(const char* name, const btc::BoolMatrix& A, const btc::BoolMatrix& B) {
    const std::size_t n = A.rows();

    bgv_batched::Params params;
    params.multiplicative_depth = btc::BGVBatchedMatmulDepth(n);
    params.batch_size = 64;
    auto ctx = bgv_batched::setup(params);

    btc::BGVBatchedBackend backend(ctx, n);
    auto enc_A = backend.Encrypt(A);
    auto enc_B = backend.Encrypt(B);
    auto enc_C = backend.matmul(enc_A, enc_B);
    auto result = backend.Decrypt(enc_C);

    auto expected = plaintext_bool_matmul(A, B);
    check(result == expected, name);
    std::printf("%s: PASS\n", name);
}

// ── No witness: A*B should be the all-zero matrix ───────────────────────
static void test_no_witness() {
    // A: only edge 0->1. B: only edge 2->0. No k with A[i,k] and B[k,j] both
    // 1 for any (i,j) -- e.g. C[0][0] needs A[0,k] and B[k,0]; A[0,1]=1 but
    // B[1,0]=0, so no witness anywhere.
    auto A = from_flat(3, {
        false, true,  false,
        false, false, false,
        false, false, false
    });
    auto B = from_flat(3, {
        false, false, false,
        false, false, false,
        true,  false, false
    });
    run_matmul_test("no_witness (A*B is all-zero)", A, B);

    // Explicit all-zero-input case too: 0 * 0 matrices.
    btc::BoolMatrix Z(3, 3, false);
    run_matmul_test("no_witness (all-zero inputs)", Z, Z);
}

// ── Exactly one witness ──────────────────────────────────────────────────
static void test_one_witness() {
    // A: 0->1. B: 1->2. C[0][2] = A[0][1] AND B[1][2] = 1, exactly one
    // witness (k=1); every other output entry has zero witnesses.
    auto A = from_flat(3, {
        false, true,  false,
        false, false, false,
        false, false, false
    });
    auto B = from_flat(3, {
        false, false, false,
        false, false, true,
        false, false, false
    });
    run_matmul_test("one_witness (single k contributes)", A, B);
}

// ── Multiple witnesses: OR-tree must collapse to 1, not a witness count ─
static void test_multiple_witnesses() {
    // A: node 0 connects to both 1 and 2. B: both 1 and 2 connect to node 3.
    // C[0][3] = OR_k(A[0][k] AND B[k][3]) has TWO witnesses (k=1 and k=2).
    // Ordinary summation would compute 1+1=2 here; Boolean OR must give 1.
    auto A = from_flat(4, {
        false, true,  true,  false,
        false, false, false, false,
        false, false, false, false,
        false, false, false, false
    });
    auto B = from_flat(4, {
        false, false, false, false,
        false, false, false, true,
        false, false, false, true,
        false, false, false, false
    });
    run_matmul_test("multiple_witnesses (two paths converge, OR-tree collapses to 1)", A, B);

    // Denser case: every row of A and every column of B is all-ones, so
    // every output entry has N witnesses (N=4) -- maximal witness-count
    // stress test for the OR-tree.
    btc::BoolMatrix Ones(4, 4, true);
    run_matmul_test("multiple_witnesses (dense all-ones, N witnesses per entry)", Ones, Ones);
}

// ── Self-product of an identity-like matrix (sanity check) ──────────────
static void test_identity_self_product() {
    auto I = btc::BoolMatrix::identity(3);
    run_matmul_test("identity_self_product (I*I == I)", I, I);
}

int main() {
    test_no_witness();
    test_one_witness();
    test_multiple_witnesses();
    test_identity_self_product();
    std::puts("\nAll BGV-batched packed matmul tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
