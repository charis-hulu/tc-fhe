#ifdef TC_WITH_BGV_BATCHED

// ── Layout tests: packing, masks, broadcasts ─────────────────────────────
//
// These are NOT Boolean transitive-closure tests. They exist specifically
// to catch incorrect rotation direction, row-crossing, or slot-placement
// bugs in the packing primitives BEFORE they're used inside a Boolean
// matmul (see tests/test_bgv_batched_matmul.cpp) or full TC (see
// tests/test_bgv_batched_tc.cpp), per the task's explicit requirement that
// column masking, horizontal broadcast, row masking, and vertical broadcast
// each be independently verified.
//
// A 3x3 matrix of distinct small values is used throughout, matching the
// task's suggested layout-test convention.

#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/matrix.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bgv_batched = btc::bgv_batched;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

namespace {

// 3x3 matrix with distinct values 0..8 (not Boolean -- this file tests raw
// slot placement, not Boolean semantics), row-major: M[i][j] = i*3 + j.
//
//   [0 1 2]
//   [3 4 5]
//   [6 7 8]
std::vector<int64_t> distinct_3x3() {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

} // namespace

// ── Full-matrix row-major packing round trip ────────────────────────────
static void test_pack_unpack_roundtrip() {
    bgv_batched::Params params;
    params.multiplicative_depth = 1;
    params.batch_size = 32;
    auto ctx = bgv_batched::setup(params);

    const std::size_t n = 3;
    auto values = distinct_3x3();
    auto ct = bgv_batched::encrypt_packed(ctx, values);
    auto decoded = bgv_batched::decrypt_packed_raw(ctx, ct, n * n);

    for (std::size_t idx = 0; idx < n * n; ++idx)
        check(decoded[idx] == values[idx], "pack/unpack round trip preserves slot(i,j) = i*N+j layout");

    std::puts("pack_unpack_roundtrip (3x3, slot(i,j)=i*N+j): PASS");
}

// ── Column masking: M^col_k should zero everything except column k ──────
static void test_column_mask() {
    bgv_batched::Params params;
    params.multiplicative_depth = 1;
    params.batch_size = 32;
    auto ctx = bgv_batched::setup(params);

    const std::size_t n = 3;
    auto values = distinct_3x3();
    auto ct = bgv_batched::encrypt_packed(ctx, values);
    btc::BGVBatchedBackend backend(ctx, n);

    for (std::size_t k = 0; k < n; ++k) {
        auto masked = bgv_batched::mask_mult(ctx, ct, backend.ColumnMask(k));
        auto decoded = bgv_batched::decrypt_packed_raw(ctx, masked, n * n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                int64_t expected = (j == k) ? values[i * n + j] : 0;
                check(decoded[i * n + j] == expected, "column mask leaves only column k nonzero");
            }
        }
    }
    std::puts("column_mask (3x3, all k): PASS");
}

// ── Horizontal column broadcast: L_k[i,j] should equal A[i,k] everywhere ─
static void test_horizontal_broadcast() {
    bgv_batched::Params params;
    // Depth 2: BroadcastColumnHorizontally does two ciphertext-plaintext
    // multiplications (the initial column mask, then the final valid-region
    // mask reapplied after the rotation-and-sum) -- see bgv_batched_backend.hpp.
    params.multiplicative_depth = 2;
    params.batch_size = 32;
    auto ctx = bgv_batched::setup(params);

    const std::size_t n = 3;
    auto values = distinct_3x3();
    auto ct = bgv_batched::encrypt_packed(ctx, values);
    btc::BGVBatchedBackend backend(ctx, n);

    for (std::size_t k = 0; k < n; ++k) {
        auto broadcast = backend.BroadcastColumnHorizontally(ct, k);
        auto decoded = bgv_batched::decrypt_packed_raw(ctx, broadcast, n * n);

        for (std::size_t i = 0; i < n; ++i) {
            int64_t expected = values[i * n + k]; // A[i,k]
            for (std::size_t j = 0; j < n; ++j)
                check(decoded[i * n + j] == expected,
                      "horizontal broadcast: every slot in row i must equal A[i,k]");
        }
    }
    std::puts("horizontal_broadcast (3x3, all k): PASS");
}

// ── Row masking: M^row_k should zero everything except row k ────────────
static void test_row_mask() {
    bgv_batched::Params params;
    params.multiplicative_depth = 1;
    params.batch_size = 32;
    auto ctx = bgv_batched::setup(params);

    const std::size_t n = 3;
    auto values = distinct_3x3();
    auto ct = bgv_batched::encrypt_packed(ctx, values);
    btc::BGVBatchedBackend backend(ctx, n);

    for (std::size_t k = 0; k < n; ++k) {
        auto masked = bgv_batched::mask_mult(ctx, ct, backend.RowMask(k));
        auto decoded = bgv_batched::decrypt_packed_raw(ctx, masked, n * n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                int64_t expected = (i == k) ? values[i * n + j] : 0;
                check(decoded[i * n + j] == expected, "row mask leaves only row k nonzero");
            }
        }
    }
    std::puts("row_mask (3x3, all k): PASS");
}

// ── Vertical row broadcast: Q_k[i,j] should equal B[k,j] everywhere ──────
static void test_vertical_broadcast() {
    bgv_batched::Params params;
    // Depth 2: same reasoning as test_horizontal_broadcast above.
    params.multiplicative_depth = 2;
    params.batch_size = 32;
    auto ctx = bgv_batched::setup(params);

    const std::size_t n = 3;
    auto values = distinct_3x3();
    auto ct = bgv_batched::encrypt_packed(ctx, values);
    btc::BGVBatchedBackend backend(ctx, n);

    for (std::size_t k = 0; k < n; ++k) {
        auto broadcast = backend.BroadcastRowVertically(ct, k);
        auto decoded = bgv_batched::decrypt_packed_raw(ctx, broadcast, n * n);

        for (std::size_t j = 0; j < n; ++j) {
            int64_t expected = values[k * n + j]; // B[k,j]
            for (std::size_t i = 0; i < n; ++i)
                check(decoded[i * n + j] == expected,
                      "vertical broadcast: every slot in column j must equal B[k,j]");
        }
    }
    std::puts("vertical_broadcast (3x3, all k): PASS");
}

// ── Capacity check: N*N > available slots must fail with a clear error ──
static void test_capacity_check_rejects_oversized_matrix() {
    bgv_batched::Params params;
    params.multiplicative_depth = 1;
    params.batch_size = 8; // deliberately too small for a 3x3 = 9-slot matrix
    auto ctx = bgv_batched::setup(params);

    bool threw = false;
    try {
        btc::BGVBatchedBackend backend(ctx, 3);
        (void)backend;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "constructing a backend for N*N > available slots must throw, not silently truncate");
    std::puts("capacity_check_rejects_oversized_matrix: PASS");
}

int main() {
    test_pack_unpack_roundtrip();
    test_column_mask();
    test_horizontal_broadcast();
    test_row_mask();
    test_vertical_broadcast();
    test_capacity_check_rejects_oversized_matrix();
    std::puts("\nAll BGV-batched packing/layout tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
