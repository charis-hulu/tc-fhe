#ifdef TC_WITH_BGV_BATCHED

// ── Offline BGV-batched parameter profile generator ──────────────────────
//
// Solves the plaintext-modulus / ring-dimension incompatibility documented
// in docs/plaintext_modulus_issue.md and include/btc/bgv_parameter_profile.hpp:
// rather than guessing a single "safe" plaintext modulus at runtime, this
// tool searches OFFLINE for validated (depth, ring_dimension,
// plaintext_modulus) triples that OpenFHE actually accepts together under
// HEStd_128_classic, end-to-end tests each candidate, and saves only the
// ones that pass to config/bgv_batched_profiles.json. The runtime backend
// then looks up a profile instead of guessing (see Part 3 of the task this
// implements, not yet wired up in this commit).
//
// Usage:
//   ./generate_bgv_batched_profiles --nodes=4,8,16,32 \
//       --output=config/bgv_batched_profiles.json
//
// Algorithm per requested node count N (see docs comment at the bottom of
// this file for the condensed version):
//   1. required_depth  = btc::EstimateBGVBatchedDepth(N, T=N)   -- reuses
//      the existing, UNMODIFIED depth estimator; this tool does not
//      change TC-round count or matmul depth accounting.
//   2. required_slots  = N * N                                  (overflow-checked)
//   3. candidate ring dimension n = next_power_of_two(required_slots),
//      doubling on failure.
//   4. m = 2 * n.
//   5. t = FirstPrime<NativeInteger>(nBits, m), bumped via NextPrime if
//      below minimum_plaintext_modulus.
//   6. Attempt GenCryptoContext(depth=required_depth, t, n, HEStd_128_classic).
//      If OpenFHE rejects n as insufficient for the security table
//      (OPENFHE_THROW from computeRingDimension's "Case 3" check -- see
//      docs/plaintext_modulus_issue.md), double n and retry from step 4.
//      Any OTHER exception is reported and NOT silently retried.
//   7. On success, validate actual ring dimension, slot capacity, and NTT
//      compatibility.
//   8. End-to-end validate: pack/encrypt/decrypt round trip, one
//      ciphertext-ciphertext multiplication, one rotation.
//   9. Save the profile only if every validation step passed.
//
// This tool does NOT change btc::EstimateBGVBatchedDepth, does NOT change
// the transitive-closure algorithm or Boolean matmul, does NOT touch
// security level (always HEStd_128_classic), and does NOT hardcode one
// "universal" cyclotomic order -- each profile's m is derived from the
// smallest ring dimension that actually works for that profile's depth.

#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/bgv_parameter_profile.hpp>

#include "openfhe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace bgv_batched = btc::bgv_batched;
namespace profile_db = btc::bgv_parameter_profile;

namespace {

// ── CLI args ──────────────────────────────────────────────────────────────

struct Args {
    std::vector<std::size_t> nodes;
    std::string output = "config/bgv_batched_profiles.json";
    uint64_t minimum_plaintext_modulus = 65537; // preserves this repo's existing policy default
};

std::vector<std::size_t> ParseNodeList(const std::string& csv) {
    std::vector<std::size_t> result;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty())
            continue;
        long value = std::stol(item);
        if (value < 1)
            throw std::invalid_argument("generate_bgv_batched_profiles: --nodes values must be >= 1 (got " + item + ")");
        result.push_back(static_cast<std::size_t>(value));
    }
    if (result.empty())
        throw std::invalid_argument("generate_bgv_batched_profiles: --nodes must list at least one value");
    return result;
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    bool nodes_given = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
        if (arg.rfind("--nodes=", 0) == 0) {
            args.nodes = ParseNodeList(value_after("--nodes="));
            nodes_given = true;
        } else if (arg.rfind("--output=", 0) == 0) {
            args.output = value_after("--output=");
        } else if (arg.rfind("--minimum-plaintext-modulus=", 0) == 0) {
            args.minimum_plaintext_modulus = std::stoull(value_after("--minimum-plaintext-modulus="));
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: generate_bgv_batched_profiles --nodes=<n1,n2,...> "
                "[--output=<path>] [--minimum-plaintext-modulus=<n>]\n");
            std::exit(1);
        }
    }
    if (!nodes_given) {
        std::fprintf(stderr, "Missing required --nodes=<n1,n2,...>\n");
        std::exit(1);
    }
    return args;
}

// ── Requirement computation ──────────────────────────────────────────────

struct Requirement {
    std::size_t n = 0;
    uint32_t required_depth = 0;
    uint64_t required_slots = 0;
};

// required_slots = N*N, computed with overflow-safe arithmetic (per task
// spec) since N is caller-supplied and this tool has no upper bound on it.
Requirement ComputeRequirement(std::size_t n) {
    if (n > 0 && n > std::numeric_limits<uint64_t>::max() / n)
        throw std::overflow_error("generate_bgv_batched_profiles: N*N overflows for N=" + std::to_string(n));

    Requirement req;
    req.n = n;
    // T = N: the same conservative bound used throughout this codebase
    // (see examples/bgv_batched_closure.cpp) -- the server cannot assume a
    // graph is acyclic, so it must budget depth for the worst case.
    req.required_depth = btc::EstimateBGVBatchedDepth(n, static_cast<int>(n));
    req.required_slots = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
    return req;
}

std::size_t NextPowerOfTwo(uint64_t value) {
    std::size_t p = 1;
    while (p < value)
        p <<= 1;
    return p;
}

// nBits for FirstPrime: must be large enough that the resulting plaintext
// modulus is >= minimum_plaintext_modulus (the repo's existing policy
// floor -- see Params::plaintext_modulus's documented rationale in
// bgv_batched.hpp) and large enough to exactly represent this circuit's
// transient Boolean values (never exceeding roughly N before the OR gate's
// a+b-ab identity reduces them back to {0,1} -- N fits comfortably in a
// handful of bits for any N this tool would realistically be asked about).
uint32_t RequiredPlaintextBits(uint64_t minimum_plaintext_modulus) {
    uint32_t bits = 0;
    uint64_t v = minimum_plaintext_modulus;
    while (v > 0) {
        ++bits;
        v >>= 1;
    }
    return bits; // e.g. 65537 -> 17 bits, matches FirstPrime's "nBits" semantics
}

// ── Candidate search ──────────────────────────────────────────────────────

struct Candidate {
    uint32_t ring_dimension = 0;
    uint64_t plaintext_modulus = 0;
};

// Finds t = FirstPrime(nBits, m), bumped forward via NextPrime if it comes
// back below minimum_plaintext_modulus (FirstPrime only guarantees "at
// least nBits+1 bits", not "at least this exact value").
uint64_t FindCompatiblePlaintextModulus(uint32_t ring_dimension, uint64_t minimum_plaintext_modulus) {
    const uint64_t m = 2ull * ring_dimension;
    const uint32_t nBits = RequiredPlaintextBits(minimum_plaintext_modulus);

    lbcrypto::NativeInteger t = lbcrypto::FirstPrime<lbcrypto::NativeInteger>(nBits, m);
    while (t.ConvertToInt<uint64_t>() < minimum_plaintext_modulus) {
        t = lbcrypto::NextPrime<lbcrypto::NativeInteger>(t, m);
    }
    return t.ConvertToInt<uint64_t>();
}

// Attempts GenCryptoContext for one (ring_dimension, plaintext_modulus,
// depth) triple. Returns the built context on success. Distinguishes
// "this ring dimension is insufficient for the security table" (caller
// should retry with a larger ring dimension) from genuinely unexpected
// failures (caller should NOT silently retry -- see task spec).
enum class AttemptOutcome { Success, RingDimensionTooSmall, UnexpectedFailure };

struct AttemptResult {
    AttemptOutcome outcome = AttemptOutcome::UnexpectedFailure;
    bgv_batched::Context context;
    std::string error_message;
};

AttemptResult TryBuildContext(uint32_t ring_dimension, uint64_t plaintext_modulus, uint32_t depth) {
    AttemptResult result;
    bgv_batched::Params params;
    params.plaintext_modulus = plaintext_modulus;
    params.multiplicative_depth = depth;
    params.security_level = lbcrypto::HEStd_128_classic; // never HEStd_NotSet, per task constraint
    params.ring_dimension = ring_dimension;               // manual pin -- OpenFHE still validates
                                                           // it against the security table for this
                                                           // depth (see docs/plaintext_modulus_issue.md)
    params.batch_size = 0; // automatic/full batching, per task constraint

    try {
        result.context = bgv_batched::setup(params);
        result.outcome = AttemptOutcome::Success;
    } catch (const lbcrypto::OpenFHEException& e) {
        std::string msg = e.what();
        // OpenFHE's computeRingDimension "Case 3" throws exactly this
        // message shape when the manually-pinned ring dimension is below
        // what the security table requires for this depth -- see
        // bgvrns-parametergeneration.cpp, confirmed against this project's
        // linked OpenFHE build. Only THIS specific failure is treated as
        // "try a bigger ring dimension"; anything else is surfaced as-is.
        if (msg.find("does not comply with HE standards recommendation") != std::string::npos ||
            msg.find("does not meet security requirements") != std::string::npos) {
            result.outcome = AttemptOutcome::RingDimensionTooSmall;
        } else {
            result.outcome = AttemptOutcome::UnexpectedFailure;
        }
        result.error_message = msg;
    }
    return result;
}

// Searches ascending powers of two for the smallest ring dimension that:
//   (a) is >= required_slots,
//   (b) has a compatible plaintext modulus >= minimum_plaintext_modulus,
//   (c) is accepted by OpenFHE's HEStd_128_classic security check for the
//       given depth.
// Doubles the candidate on RingDimensionTooSmall; stops and reports on any
// UnexpectedFailure rather than looping past it.
struct SearchResult {
    bool found = false;
    Candidate candidate;
    bgv_batched::Context context;
};

SearchResult SearchCompatibleParameters(const Requirement& req, uint64_t minimum_plaintext_modulus) {
    SearchResult search;
    std::size_t ring_dimension = NextPowerOfTwo(req.required_slots);

    // Generous but finite upper bound so a genuine bug (e.g. an unexpected
    // exception class being misclassified as "too small") cannot spin
    // forever; 2^28 is far beyond anything this backend could use in
    // practice (memory alone would be prohibitive well before this).
    const std::size_t kMaxRingDimension = 1ull << 28;

    while (ring_dimension <= kMaxRingDimension) {
        uint64_t t = FindCompatiblePlaintextModulus(static_cast<uint32_t>(ring_dimension), minimum_plaintext_modulus);

        std::printf("  trying ring_dimension=%zu plaintext_modulus=%llu ... ",
                     ring_dimension, static_cast<unsigned long long>(t));
        std::fflush(stdout);

        AttemptResult attempt = TryBuildContext(static_cast<uint32_t>(ring_dimension), t, req.required_depth);
        if (attempt.outcome == AttemptOutcome::Success) {
            std::printf("OK\n");
            search.found = true;
            search.candidate.ring_dimension = static_cast<uint32_t>(ring_dimension);
            search.candidate.plaintext_modulus = t;
            search.context = std::move(attempt.context);
            return search;
        }
        if (attempt.outcome == AttemptOutcome::RingDimensionTooSmall) {
            std::printf("rejected (ring dimension too small for security level): %s\n",
                        attempt.error_message.c_str());
            ring_dimension <<= 1;
            continue;
        }
        // UnexpectedFailure: report clearly and stop searching for this N.
        std::printf("UNEXPECTED FAILURE: %s\n", attempt.error_message.c_str());
        throw std::runtime_error(
            "generate_bgv_batched_profiles: unexpected OpenFHE failure while building context "
            "(ring_dimension=" + std::to_string(ring_dimension) + ", t=" + std::to_string(t) +
            ", depth=" + std::to_string(req.required_depth) + "): " + attempt.error_message);
    }

    return search; // found == false
}

// ── End-to-end validation ─────────────────────────────────────────────────
//
// Per task spec: pack a plaintext, generate keys (already done by setup),
// encrypt, decrypt and compare, one ciphertext-ciphertext multiplication,
// one representative rotation key + rotation, decrypt and verify.
bool ValidateEndToEnd(bgv_batched::Context& ctx, uint32_t ring_dimension) {
    // Small distinct-value packed vector, matching this project's rotation
    // test convention (tests/test_bgv_batched_rotation.cpp) -- reused here
    // rather than duplicated logic, since both need "encrypt some slots,
    // decrypt, compare."
    std::vector<int64_t> values = {1, 0, 1, 1, 0, 0, 1, 0};

    auto ct = bgv_batched::encrypt_packed(ctx, values);
    auto decoded = bgv_batched::decrypt_packed_raw(ctx, ct, values.size());
    if (decoded != values) {
        std::printf("    validation FAILED: encrypt/decrypt round trip mismatch\n");
        return false;
    }

    // One representative ciphertext-ciphertext multiplication (AND: a*b).
    auto ct_squared = bgv_batched::encrypted_and(ctx, ct, ct);
    auto decoded_squared = bgv_batched::decrypt_packed_raw(ctx, ct_squared, values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        int64_t expected = values[i] * values[i]; // Boolean AND(x,x) = x*x = x for x in {0,1}
        if (decoded_squared[i] != expected) {
            std::printf("    validation FAILED: ciphertext-ciphertext multiplication mismatch at slot %zu\n", i);
            return false;
        }
    }

    // One representative rotation: rotate by 1 and check the documented
    // left-shift convention (see bgv_batched.hpp::rotateToOffset) holds.
    bgv_batched::generate_rotation_keys(ctx, {1});
    auto rotated = bgv_batched::rotateToOffset(ctx, ct, 1);
    auto decoded_rotated = bgv_batched::decrypt_packed_raw(ctx, rotated, values.size());
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        if (decoded_rotated[i] != values[i + 1]) {
            std::printf("    validation FAILED: rotation result mismatch at slot %zu\n", i);
            return false;
        }
    }

    (void)ring_dimension;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = ParseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    profile_db::Database db;
    db.minimum_plaintext_modulus = args.minimum_plaintext_modulus;
    // GetOPENFHEVersion() (global namespace, declared by openfhe.h) only
    // returns a real version string when BASE_OPENFHE_VERSION was defined
    // at OpenFHE's OWN build time and propagated to consumers -- this
    // project's CMakeLists.txt does not (and should not) redefine that
    // macro itself, so on some installs this resolves to the literal
    // macro name rather than a version number. Recorded best-effort for
    // provenance; not required for profile correctness.
    db.openfhe_version = GetOPENFHEVersion();

    // If --output already exists, load it first so re-running the
    // generator for new node counts ADDS profiles instead of discarding
    // previously validated ones (dedup+sort in SaveDatabase keeps the file
    // stable either way).
    {
        std::ifstream probe(args.output);
        if (probe.good()) {
            try {
                profile_db::Database existing = profile_db::LoadDatabase(args.output);
                db.profiles = existing.profiles;
                std::printf("Loaded %zu existing profile(s) from %s\n", db.profiles.size(), args.output.c_str());
            } catch (const std::exception& e) {
                std::printf("WARNING: could not load existing %s (%s) -- starting a fresh database.\n",
                            args.output.c_str(), e.what());
            }
        }
    }

    bool any_failed = false;

    for (std::size_t n : args.nodes) {
        std::printf("\n=== N=%zu ===\n", n);

        Requirement req;
        try {
            req = ComputeRequirement(n);
        } catch (const std::exception& e) {
            std::printf("  FAILED to compute requirements: %s\n", e.what());
            any_failed = true;
            continue;
        }
        std::printf("  required_depth=%u required_slots=%llu\n",
                    req.required_depth, static_cast<unsigned long long>(req.required_slots));

        SearchResult search;
        try {
            search = SearchCompatibleParameters(req, args.minimum_plaintext_modulus);
        } catch (const std::exception& e) {
            std::printf("  FAILED: %s\n", e.what());
            any_failed = true;
            continue;
        }

        if (!search.found) {
            std::printf("  FAILED: no compatible ring dimension found up to the search ceiling.\n");
            any_failed = true;
            continue;
        }

        // Post-generation validation (task spec step 8).
        const uint32_t actual_ring_dim = bgv_batched::ring_dimension(search.context);
        const uint64_t t = search.candidate.plaintext_modulus;
        bool consistent = true;
        if (actual_ring_dim != search.candidate.ring_dimension) {
            std::printf("  FAILED: actual ring dimension (%u) != requested (%u)\n",
                        actual_ring_dim, search.candidate.ring_dimension);
            consistent = false;
        }
        if (actual_ring_dim < req.required_slots) {
            std::printf("  FAILED: ring dimension (%u) < required slots (%llu)\n",
                        actual_ring_dim, static_cast<unsigned long long>(req.required_slots));
            consistent = false;
        }
        if ((t - 1) % (2ull * actual_ring_dim) != 0) {
            std::printf("  FAILED: (t-1) %% (2*ring_dim) != 0 (t=%llu, ring_dim=%u)\n",
                        static_cast<unsigned long long>(t), actual_ring_dim);
            consistent = false;
        }
        if (!consistent) {
            any_failed = true;
            continue;
        }

        std::printf("  running end-to-end validation (encrypt/decrypt, multiply, rotate)...\n");
        bool ok = false;
        try {
            ok = ValidateEndToEnd(search.context, actual_ring_dim);
        } catch (const std::exception& e) {
            std::printf("  FAILED: end-to-end validation threw: %s\n", e.what());
            any_failed = true;
            continue;
        }
        if (!ok) {
            any_failed = true;
            continue;
        }

        profile_db::Profile profile;
        profile.depth_capacity = req.required_depth;
        profile.ring_dimension = actual_ring_dim;
        profile.plaintext_modulus = t;
        profile.validated = true;
        db.profiles.push_back(profile);

        std::printf("  PASSED: depth_capacity=%u ring_dimension=%u plaintext_modulus=%llu\n",
                    profile.depth_capacity, profile.ring_dimension,
                    static_cast<unsigned long long>(profile.plaintext_modulus));
    }

    profile_db::DeduplicateAndSort(db.profiles);

    try {
        profile_db::SaveDatabase(args.output, db);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to save %s: %s\n", args.output.c_str(), e.what());
        return 1;
    }

    std::printf("\nSaved %zu profile(s) to %s\n", db.profiles.size(), args.output.c_str());
    return any_failed ? 1 : 0;
}

// ── Condensed algorithm summary (see also docs/bgv_parameter_profiles.md) ─
//
// offline generation:
//   requirements (depth, N*N slots) -> search ring dimension n (powers of
//   two) -> generate compatible plaintext modulus t via FirstPrime/NextPrime
//   -> attempt GenCryptoContext(depth, t, n, HEStd_128_classic) -> on
//   "ring dimension too small for security level," double n and retry; on
//   any other failure, stop and report -> on success, validate NTT
//   compatibility and run end-to-end encrypt/decrypt/multiply/rotate ->
//   store only validated profiles, deduplicated and sorted.

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this tool.");
    return 0;
}
#endif
