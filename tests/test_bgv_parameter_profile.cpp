#ifdef TC_WITH_BGV_BATCHED

// ── Profile database and runtime lookup tests ────────────────────────────
//
// Two groups of tests:
//   1. Profile generation/validity (against the checked-in
//      config/bgv_batched_profiles.json, produced by
//      tools/generate_bgv_batched_profiles -- see docs/bgv_parameter_profiles.md).
//   2. Runtime lookup logic (bgv_parameter_profile::SelectProfile/RequireProfile),
//      exercised against small in-memory Database fixtures so these cases
//      run instantly and don't depend on the checked-in file's exact
//      contents -- exact-match selection, selecting a larger profile when
//      no exact match exists, preferring the smallest compatible ring
//      dimension, rejecting insufficient depth/slots, and rejecting
//      malformed/unvalidated profiles.

#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/bgv_batched_profile_setup.hpp>
#include <btc/bgv_parameter_profile.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace profile_db = btc::bgv_parameter_profile;
namespace bgv_batched = btc::bgv_batched;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

namespace {

profile_db::Profile MakeProfile(uint32_t depth, uint32_t ring_dim, uint64_t t, bool validated = true) {
    profile_db::Profile p;
    p.depth_capacity = depth;
    p.ring_dimension = ring_dim;
    p.plaintext_modulus = t;
    p.validated = validated;
    return p;
}

// Fixture matching the shape (not exact values) of config/bgv_batched_profiles.json:
// a small profile, a medium one, and a large one, each strictly bigger than
// the last in both depth and ring dimension.
profile_db::Database MakeFixtureDatabase() {
    profile_db::Database db;
    db.profiles = {
        MakeProfile(12, 32768, 786433),
        MakeProfile(21, 65536, 786433),
        MakeProfile(32, 131072, 786433),
    };
    return db;
}

} // namespace

// ── Runtime lookup: exact match ──────────────────────────────────────────
static void test_select_exact_match() {
    auto db = MakeFixtureDatabase();
    const auto* p = profile_db::SelectProfile(db, /*required_depth=*/12, /*required_slots=*/16);
    check(p != nullptr, "exact match: a profile must be found");
    check(p->depth_capacity == 12 && p->ring_dimension == 32768, "exact match: smallest profile selected");
    std::puts("select_exact_match: PASS");
}

// ── Runtime lookup: select a larger profile when no exact match exists ──
static void test_select_larger_when_no_exact_match() {
    auto db = MakeFixtureDatabase();
    // required_depth=15 falls strictly between the 12 and 21 capacity
    // profiles -- no profile has depth_capacity exactly 15, so the next
    // one up (21/65536) must be selected.
    const auto* p = profile_db::SelectProfile(db, /*required_depth=*/15, /*required_slots=*/16);
    check(p != nullptr, "larger match: a profile must be found");
    check(p->depth_capacity == 21 && p->ring_dimension == 65536,
          "larger match: smallest profile that still satisfies the requirement is selected");
    std::puts("select_larger_when_no_exact_match: PASS");
}

// ── Runtime lookup: prefer the smallest compatible ring dimension ───────
static void test_select_prefers_smallest_ring_dimension() {
    profile_db::Database db;
    // Two profiles both satisfy depth>=10, slots>=100, but with different
    // ring dimensions -- the smaller one must win (cheapest = smallest
    // ring_dimension first, per ProfileLessThan's ordering).
    db.profiles = {
        MakeProfile(20, 8192, 786433),
        MakeProfile(10, 4096, 786433),
    };
    const auto* p = profile_db::SelectProfile(db, /*required_depth=*/10, /*required_slots=*/100);
    check(p != nullptr, "prefers smallest: a profile must be found");
    check(p->ring_dimension == 4096, "prefers smallest: smaller ring dimension wins even though listed second");
    std::puts("select_prefers_smallest_ring_dimension: PASS");
}

// ── Runtime lookup: reject insufficient depth capacity ───────────────────
static void test_reject_insufficient_depth() {
    auto db = MakeFixtureDatabase();
    // Largest profile has depth_capacity=32; ask for more than any profile provides.
    const auto* p = profile_db::SelectProfile(db, /*required_depth=*/100, /*required_slots=*/16);
    check(p == nullptr, "reject insufficient depth: no profile should satisfy this request");

    bool threw = false;
    try {
        profile_db::RequireProfile(db, 100, 16);
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find("100") != std::string::npos, "error message should mention the required depth");
    }
    check(threw, "RequireProfile must throw when no profile satisfies the depth requirement");
    std::puts("reject_insufficient_depth: PASS");
}

// ── Runtime lookup: reject insufficient slot capacity ─────────────────────
static void test_reject_insufficient_slots() {
    auto db = MakeFixtureDatabase();
    // Largest profile has ring_dimension=131072 (== slot capacity); ask for more.
    const auto* p = profile_db::SelectProfile(db, /*required_depth=*/12, /*required_slots=*/999999999);
    check(p == nullptr, "reject insufficient slots: no profile should satisfy this request");

    bool threw = false;
    try {
        profile_db::RequireProfile(db, 12, 999999999);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "RequireProfile must throw when no profile satisfies the slot requirement");
    std::puts("reject_insufficient_slots: PASS");
}

// ── Runtime lookup: reject malformed/unvalidated profiles ────────────────
static void test_reject_unvalidated_profile() {
    profile_db::Database db;
    // A profile that would otherwise satisfy the request, but is marked
    // validated=false (e.g. a hand-edited or partially-generated entry) --
    // must never be selected.
    db.profiles = {MakeProfile(50, 999999, 786433, /*validated=*/false)};

    const auto* p = profile_db::SelectProfile(db, /*required_depth=*/10, /*required_slots=*/16);
    check(p == nullptr, "unvalidated profile must never be selected, even if its capacity would fit");
    std::puts("reject_unvalidated_profile: PASS");
}

// Malformed JSON / missing file must fail loudly with a descriptive error,
// not a raw nlohmann::json exception or a segfault.
static void test_load_database_missing_file_fails_loudly() {
    bool threw = false;
    try {
        profile_db::LoadDatabase("this/path/does/not/exist.json");
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find("cannot open") != std::string::npos, "error should explain the file could not be opened");
    }
    check(threw, "LoadDatabase must throw std::runtime_error for a missing file");
    std::puts("load_database_missing_file_fails_loudly: PASS");
}

// ── Profile generation/validity: against the checked-in database ────────
//
// Requires config/bgv_batched_profiles.json to exist (generated via
// tools/generate_bgv_batched_profiles --nodes=4,8,16,32 -- see
// docs/bgv_parameter_profiles.md). If it's missing, this test fails with a
// clear message rather than silently skipping, since a missing database
// means setup_from_profile is broken for every caller, not just this test.
static void test_checked_in_database_has_n4_and_n8_profiles() {
    profile_db::Database db;
    try {
        db = profile_db::LoadDatabase(bgv_batched::DefaultProfileDatabasePath());
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "FAIL: could not load %s (%s) -- generate it first with "
            "tools/generate_bgv_batched_profiles --nodes=4,8,16,32\n",
            bgv_batched::DefaultProfileDatabasePath(), e.what());
        std::exit(1);
    }

    // N=4, T=4: EstimateBGVBatchedDepth(4,4) = 12, slots = 16.
    const auto* p4 = profile_db::SelectProfile(db, btc::EstimateBGVBatchedDepth(4, 4), 4ull * 4ull);
    check(p4 != nullptr, "checked-in database must have a profile covering N=4, T=4");
    check((p4->plaintext_modulus - 1) % (2ull * p4->ring_dimension) == 0,
          "N=4 profile must be NTT-compatible: (t-1) % (2*ring_dim) == 0");

    // N=8, T=8: EstimateBGVBatchedDepth(8,8) = 21, slots = 64.
    const auto* p8 = profile_db::SelectProfile(db, btc::EstimateBGVBatchedDepth(8, 8), 8ull * 8ull);
    check(p8 != nullptr, "checked-in database must have a profile covering N=8, T=8");
    check((p8->plaintext_modulus - 1) % (2ull * p8->ring_dimension) == 0,
          "N=8 profile must be NTT-compatible: (t-1) % (2*ring_dim) == 0");

    std::puts("checked_in_database_has_n4_and_n8_profiles: PASS");
}

// End-to-end: setup_from_profile against the real checked-in database must
// actually build a working BGV context (not just select the right numbers
// on paper) -- encrypt/decrypt round trip through the resulting context.
static void test_setup_from_profile_end_to_end() {
    const std::size_t n = 4;
    const int T = 4;
    const uint32_t depth = btc::EstimateBGVBatchedDepth(n, T);
    const uint64_t slots = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);

    bgv_batched::Context ctx;
    try {
        ctx = bgv_batched::setup_from_profile(depth, slots);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "FAIL: setup_from_profile threw for N=4 (%s) -- is config/bgv_batched_profiles.json present?\n",
            e.what());
        std::exit(1);
    }

    auto ct = bgv_batched::encrypt_packed(ctx, {1, 0, 1, 1});
    auto decoded = bgv_batched::decrypt_packed_bits(ctx, ct, 4);
    check(decoded[0] == true && decoded[1] == false && decoded[2] == true && decoded[3] == true,
          "setup_from_profile: encrypt/decrypt round trip through a profile-selected context");
    std::puts("setup_from_profile_end_to_end: PASS");
}

int main() {
    test_select_exact_match();
    test_select_larger_when_no_exact_match();
    test_select_prefers_smallest_ring_dimension();
    test_reject_insufficient_depth();
    test_reject_insufficient_slots();
    test_reject_unvalidated_profile();
    test_load_database_missing_file_fails_loudly();
    test_checked_in_database_has_n4_and_n8_profiles();
    test_setup_from_profile_end_to_end();
    std::puts("\nAll BGV parameter profile tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this test.");
    return 0;
}
#endif
