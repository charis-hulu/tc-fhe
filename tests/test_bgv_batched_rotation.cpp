#ifdef TC_WITH_BGV_BATCHED

// ── Rotation direction convention ────────────────────────────────────────
//
// This test exists to independently VERIFY (not assume) which way
// EvalRotate moves packed slot contents in the OpenFHE version this project
// links, per the task requirement "do not assume the sign convention of
// EvalRotate from memory." Every broadcast formula in
// include/btc/bgv_batched_backend.hpp is written in terms of the convention
// pinned down here via btc::bgv_batched::rotateToOffset.

#include <btc/bgv_batched.hpp>

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

int main() {
    bgv_batched::Params params;
    params.multiplicative_depth = 1;
    params.batch_size = 16; // small, fixed batch size so slot indices below are exact
    auto ctx = bgv_batched::setup(params);

    // Distinct values [0,1,2,3,4,5,6,7] in slots 0..7, zero elsewhere.
    std::vector<int64_t> values = {0, 1, 2, 3, 4, 5, 6, 7};
    auto ct = bgv_batched::encrypt_packed(ctx, values);

    bgv_batched::generate_rotation_keys(ctx, {1, -1, 3, -3});

    // Positive rotation by 1: per bgv_batched::rotateToOffset's documented
    // convention, result[i] = v[(i + offset) mod slots] -- i.e. a LEFT
    // cyclic shift. So result[0] should be the original slot 1's value (1),
    // result[6] should be original slot 7's value (7), and the wrap-around
    // brings an original low-index value into the high slots.
    auto rotated_pos1 = bgv_batched::rotateToOffset(ctx, ct, 1);
    auto decoded_pos1 = bgv_batched::decrypt_packed_raw(ctx, rotated_pos1, 8);
    check(decoded_pos1[0] == 1, "rotate(+1): slot 0 should hold original slot 1's value");
    check(decoded_pos1[6] == 7, "rotate(+1): slot 6 should hold original slot 7's value");
    std::printf("rotate(+1) of [0..7] (first 8 slots): ");
    for (auto v : decoded_pos1) std::printf("%lld ", static_cast<long long>(v));
    std::printf("\n");

    // Negative rotation by 1: the inverse -- result[i] = v[(i - 1) mod slots].
    // result[1] should be original slot 0's value (0), result[7] should be
    // original slot 6's value (6).
    auto rotated_neg1 = bgv_batched::rotateToOffset(ctx, ct, -1);
    auto decoded_neg1 = bgv_batched::decrypt_packed_raw(ctx, rotated_neg1, 8);
    check(decoded_neg1[1] == 0, "rotate(-1): slot 1 should hold original slot 0's value");
    check(decoded_neg1[7] == 6, "rotate(-1): slot 7 should hold original slot 6's value");
    std::printf("rotate(-1) of [0..7] (first 8 slots): ");
    for (auto v : decoded_neg1) std::printf("%lld ", static_cast<long long>(v));
    std::printf("\n");

    // Rotating by +1 then by -1 must return the original vector exactly --
    // confirms rotateToOffset(+d) and rotateToOffset(-d) are true inverses,
    // which the broadcast sum formulas in bgv_batched_backend.hpp rely on.
    auto roundtrip = bgv_batched::rotateToOffset(ctx, rotated_pos1, -1);
    auto decoded_roundtrip = bgv_batched::decrypt_packed_raw(ctx, roundtrip, 8);
    for (std::size_t i = 0; i < values.size(); ++i)
        check(decoded_roundtrip[i] == values[i], "rotate(+1) then rotate(-1) must be identity");

    // Larger offset (3) sanity check, same convention.
    auto rotated_pos3 = bgv_batched::rotateToOffset(ctx, ct, 3);
    auto decoded_pos3 = bgv_batched::decrypt_packed_raw(ctx, rotated_pos3, 8);
    check(decoded_pos3[0] == 3, "rotate(+3): slot 0 should hold original slot 3's value");

    auto rotated_neg3 = bgv_batched::rotateToOffset(ctx, ct, -3);
    auto decoded_neg3 = bgv_batched::decrypt_packed_raw(ctx, rotated_neg3, 8);
    check(decoded_neg3[3] == 0, "rotate(-3): slot 3 should hold original slot 0's value");

    // offset == 0 must be a true no-op with no rotation call (see
    // rotateToOffset's implementation) -- verify decrypted content is
    // unchanged.
    auto rotated_zero = bgv_batched::rotateToOffset(ctx, ct, 0);
    auto decoded_zero = bgv_batched::decrypt_packed_raw(ctx, rotated_zero, 8);
    for (std::size_t i = 0; i < values.size(); ++i)
        check(decoded_zero[i] == values[i], "rotate(0) must be identity");

    std::puts("\nDocumented convention: rotateToOffset(ctx, ct, offset) with offset > 0 is a "
              "LEFT cyclic shift: result[i] = original[(i + offset) mod slots]. "
              "offset < 0 shifts right (the inverse). This matches the formulas in "
              "bgv_batched_backend.hpp's BroadcastColumnHorizontallyImpl / "
              "BroadcastRowVerticallyImpl.");
    std::puts("\nAll BGV-batched rotation-direction tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
