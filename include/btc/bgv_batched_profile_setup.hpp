#pragma once

#ifdef TC_WITH_BGV_BATCHED

// ── Runtime BGV-batched context setup via the profile database ──────────
//
// Bridges bgv_batched.hpp (crypto context setup) and
// bgv_parameter_profile.hpp (the offline-validated profile database) so
// callers no longer have to guess a plaintext modulus / ring dimension for
// a given circuit size -- see docs/plaintext_modulus_issue.md for why that
// guessing was unreliable, and tools/generate_bgv_batched_profiles.cpp for
// how profiles are generated and validated offline ahead of time.
//
// Kept as a separate header (rather than folded into bgv_batched.hpp or
// bgv_parameter_profile.hpp) so:
//   - bgv_parameter_profile.hpp stays free of any OpenFHE dependency (it
//     only needs nlohmann::json), usable by tooling that just reads/writes
//     the profile format;
//   - bgv_batched.hpp stays a pure crypto-context-setup module unaware of
//     the profile file format or filesystem I/O.
//
// ─────────────────────────────────────────────────────────────────────────

#include "bgv_batched.hpp"
#include "bgv_parameter_profile.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace btc {
namespace bgv_batched {

// Default location of the generated profile database, relative to the
// working directory this project's binaries are conventionally run from
// (matches the "data/<name>.txt" convention used for graph files in
// examples/bgv_batched_closure.cpp and benchmarks/bench_bgv_batched.cpp).
inline const char* DefaultProfileDatabasePath() {
    return "config/bgv_batched_profiles.json";
}

// Builds a BGV-RNS crypto context for a circuit needing `required_depth`
// multiplicative levels and `required_slots` packed SIMD slots (N*N for a
// full-matrix-packed N x N Boolean matrix), by looking up the cheapest
// validated profile in the database at `profile_database_path` instead of
// guessing a plaintext modulus (see bgv_parameter_profile::RequireProfile).
//
// Unlike setup(Params), this function does NOT let the caller override
// plaintext_modulus/ring_dimension/batch_size -- the whole point is that
// those three values must be chosen TOGETHER as a validated triple, and
// the profile database is the only source of validated triples. Security
// level is always HEStd_128_classic (same as every profile was generated
// under) and batch size is always left automatic, matching the profile's
// own `batch_size: 0` (full-batching) assumption.
//
// Throws std::runtime_error (via RequireProfile) with an actionable
// message naming the missing (depth, slots) requirement and the generator
// command to fix it if no validated profile is large enough -- this
// function does NOT fall back to guessing or start an online parameter
// search, per the task's explicit "no silent iterative search" constraint.
//
// After building the context, re-validates the actual ring dimension,
// slot capacity, and NTT compatibility against the profile's recorded
// values, and throws if the OpenFHE build in use disagrees with what the
// profile was validated against (e.g. stale profile file from a different
// OpenFHE version) -- "fail loudly on inconsistent profiles" per the task
// spec, rather than silently trusting a profile that no longer holds.
inline Context setup_from_profile(uint32_t required_depth, uint64_t required_slots,
                                   const std::string& profile_database_path = DefaultProfileDatabasePath()) {
    bgv_parameter_profile::Database db = bgv_parameter_profile::LoadDatabase(profile_database_path);
    const bgv_parameter_profile::Profile& profile =
        bgv_parameter_profile::RequireProfile(db, required_depth, required_slots);

    std::printf("BGV parameter profile selected\n");
    std::printf("  required depth: %u\n", required_depth);
    std::printf("  profile depth capacity: %u\n", profile.depth_capacity);
    std::printf("  required slots: %llu\n", static_cast<unsigned long long>(required_slots));
    std::printf("  ring dimension / slot capacity: %u\n", profile.ring_dimension);
    std::printf("  plaintext modulus: %llu\n", static_cast<unsigned long long>(profile.plaintext_modulus));
    std::printf("  cyclotomic order: %llu\n", 2ull * profile.ring_dimension);
    std::printf("  security level: HEStd_128_classic\n");

    Params params;
    params.plaintext_modulus = profile.plaintext_modulus;
    params.multiplicative_depth = profile.depth_capacity;
    params.security_level = lbcrypto::HEStd_128_classic; // always -- matches every profile's validation
    params.ring_dimension = profile.ring_dimension;
    params.batch_size = 0; // always automatic -- matches the profile's batch_size:0 (full batching) assumption

    Context ctx = setup(params);

    // Re-validate: the profile file is a serialized claim about what a
    // PAST run of GenCryptoContext produced. If it's stale (edited by
    // hand, generated against a different OpenFHE version/build) this
    // catches the mismatch here rather than letting a silently-wrong
    // ring dimension propagate into BGVBatchedBackend's slot-capacity math.
    const uint32_t actual_ring_dim = ring_dimension(ctx);
    if (actual_ring_dim != profile.ring_dimension) {
        throw std::runtime_error(
            "bgv_batched::setup_from_profile: profile from " + profile_database_path +
            " claims ring_dimension=" + std::to_string(profile.ring_dimension) +
            " but GenCryptoContext actually produced " + std::to_string(actual_ring_dim) +
            " -- the profile database is inconsistent with the current OpenFHE build. "
            "Regenerate it with tools/generate_bgv_batched_profiles.");
    }
    if ((profile.plaintext_modulus - 1) % (2ull * actual_ring_dim) != 0) {
        throw std::runtime_error(
            "bgv_batched::setup_from_profile: profile from " + profile_database_path +
            " has plaintext_modulus=" + std::to_string(profile.plaintext_modulus) +
            " which is not NTT-compatible with ring_dimension=" + std::to_string(actual_ring_dim) +
            " -- (t-1) % (2*ring_dim) != 0. The profile database is corrupt or was hand-edited; "
            "regenerate it with tools/generate_bgv_batched_profiles.");
    }
    if (actual_ring_dim < required_slots) {
        throw std::runtime_error(
            "bgv_batched::setup_from_profile: selected profile's ring dimension (" +
            std::to_string(actual_ring_dim) + ") is below the required slot count (" +
            std::to_string(required_slots) + ") -- this indicates a bug in profile selection, "
            "not a missing profile (RequireProfile should have rejected this candidate).");
    }
    if (profile.depth_capacity < required_depth) {
        throw std::runtime_error(
            "bgv_batched::setup_from_profile: selected profile's depth capacity (" +
            std::to_string(profile.depth_capacity) + ") is below the required depth (" +
            std::to_string(required_depth) + ") -- this indicates a bug in profile selection, "
            "not a missing profile (RequireProfile should have rejected this candidate).");
    }

    return ctx;
}

} // namespace bgv_batched
} // namespace btc

#endif // TC_WITH_BGV_BATCHED
