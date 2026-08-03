#pragma once

#ifdef TC_WITH_BGV_BATCHED

// ── BGV batched parameter profile database ───────────────────────────────
//
// Solves the plaintext-modulus / ring-dimension "chicken-and-egg" problem
// documented in docs/plaintext_modulus_issue.md: BGV requires an exact
// plaintext modulus `t` that must be NTT-compatible with whatever ring
// dimension OpenFHE ends up selecting for a given (multiplicative depth,
// security level) pair, but that ring dimension is only known AFTER
// GenCryptoContext runs -- and OpenFHE has no built-in mechanism to jointly
// co-select both (confirmed against OpenFHE's own source and an OpenFHE
// maintainer's forum reply; see docs/plaintext_modulus_issue.md).
//
// Rather than guessing a single "safe" cyclotomic order or plaintext
// modulus for every possible circuit size (fragile -- see the ad hoc
// 786433 override this project used before this file existed), this file
// defines a small, serializable "profile" record: a (depth_capacity,
// ring_dimension, plaintext_modulus) triple that has been END-TO-END
// VALIDATED (see tools/generate_bgv_batched_profiles.cpp) to actually work
// together under HEStd_128_classic. Profiles are generated OFFLINE (once,
// ahead of time, not on every TC run) and stored in
// config/bgv_batched_profiles.json; the runtime backend looks up the
// smallest compatible profile instead of searching or guessing at runtime.
//
// This header is intentionally header-only (matching bgv_batched.hpp /
// bgv_batched_backend.hpp's style) and depends only on nlohmann::json
// (scoped to TC_WITH_BGV_BATCHED in CMakeLists.txt) plus the C++ standard
// library -- it does NOT include openfhe.h, so it can be used by tooling
// that only needs to read/write the profile format without linking OpenFHE
// (though in practice the generator does need OpenFHE to validate profiles
// before saving them).
//
// ─────────────────────────────────────────────────────────────────────────

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace btc {
namespace bgv_parameter_profile {

// One validated (depth, ring dimension, plaintext modulus) combination that
// has been confirmed (see tools/generate_bgv_batched_profiles.cpp) to:
//   - pass OpenFHE's GenCryptoContext under HEStd_128_classic,
//   - satisfy (plaintext_modulus - 1) % (2 * ring_dimension) == 0,
//   - round-trip encrypt/decrypt, one ciphertext-ciphertext multiplication,
//     and one rotation correctly.
//
// `depth_capacity` and `ring_dimension` are deliberately NOT tied to a
// specific N -- a profile with a given depth/ring-dimension capacity can
// serve any (N, T) combination whose required depth and required slots
// (N*N) both fit within it. Storing "max_nodes" would be redundant and
// misleading (support for a graph size depends on BOTH depth and slots
// jointly, not node count directly) -- see docs comment in the task spec
// this file implements.
struct Profile {
    uint32_t depth_capacity = 0;
    uint32_t ring_dimension = 0;
    uint64_t plaintext_modulus = 0;
    bool validated = false;

    // Available packed slots under full batching (batch_size left
    // automatic) equal ring_dimension for BGV's default power-of-two
    // packing -- so slot_capacity is intentionally NOT a separate stored
    // field (see task spec: "slot_capacity is also unnecessary if full BGV
    // batching is used, because it is equal to ring_dimension").
    uint32_t slot_capacity() const { return ring_dimension; }

    bool operator==(const Profile& other) const {
        return depth_capacity == other.depth_capacity &&
               ring_dimension == other.ring_dimension &&
               plaintext_modulus == other.plaintext_modulus &&
               validated == other.validated;
    }
};

// Deterministic ordering: ring_dimension, then depth_capacity, then
// plaintext_modulus (matches the task spec's required sort order, which
// keeps generated files reproducible and produces stable diffs).
inline bool ProfileLessThan(const Profile& a, const Profile& b) {
    if (a.ring_dimension != b.ring_dimension) return a.ring_dimension < b.ring_dimension;
    if (a.depth_capacity != b.depth_capacity) return a.depth_capacity < b.depth_capacity;
    return a.plaintext_modulus < b.plaintext_modulus;
}

// Global configuration shared by every profile in the database, plus the
// list of profiles themselves. Mirrors the JSON shape from the task spec.
struct Database {
    std::string scheme = "BGVRNS";
    std::string security_level = "HEStd_128_classic";
    uint32_t batch_size = 0; // 0 == automatic/full batching, see Params::batch_size
    uint64_t minimum_plaintext_modulus = 65537;
    std::string openfhe_version; // detected build version, filled by the generator
    std::vector<Profile> profiles;
};

// ── JSON (de)serialization ────────────────────────────────────────────────

inline void to_json(nlohmann::json& j, const Profile& p) {
    j = nlohmann::json{
        {"depth_capacity", p.depth_capacity},
        {"ring_dimension", p.ring_dimension},
        {"plaintext_modulus", p.plaintext_modulus},
        {"validated", p.validated},
    };
}

inline void from_json(const nlohmann::json& j, Profile& p) {
    j.at("depth_capacity").get_to(p.depth_capacity);
    j.at("ring_dimension").get_to(p.ring_dimension);
    j.at("plaintext_modulus").get_to(p.plaintext_modulus);
    j.at("validated").get_to(p.validated);
}

inline void to_json(nlohmann::json& j, const Database& db) {
    j = nlohmann::json{
        {"scheme", db.scheme},
        {"security_level", db.security_level},
        {"batch_size", db.batch_size},
        {"minimum_plaintext_modulus", db.minimum_plaintext_modulus},
        {"openfhe_version", db.openfhe_version},
        {"profiles", db.profiles},
    };
}

inline void from_json(const nlohmann::json& j, Database& db) {
    j.at("scheme").get_to(db.scheme);
    j.at("security_level").get_to(db.security_level);
    j.at("batch_size").get_to(db.batch_size);
    j.at("minimum_plaintext_modulus").get_to(db.minimum_plaintext_modulus);
    if (j.contains("openfhe_version"))
        j.at("openfhe_version").get_to(db.openfhe_version);
    j.at("profiles").get_to(db.profiles);
}

// Removes exact duplicate profiles (same depth_capacity, ring_dimension,
// plaintext_modulus, validated) and sorts deterministically per
// ProfileLessThan. Called by the generator before saving, and safe to call
// again at load time in case a hand-edited file has duplicates.
inline void DeduplicateAndSort(std::vector<Profile>& profiles) {
    std::sort(profiles.begin(), profiles.end(), ProfileLessThan);
    profiles.erase(std::unique(profiles.begin(), profiles.end()), profiles.end());
}

// Writes `db` to `path` as pretty-printed JSON (2-space indent), after
// deduplicating and sorting its profiles so the file stays reproducible.
inline void SaveDatabase(const std::string& path, Database db) {
    DeduplicateAndSort(db.profiles);
    nlohmann::json j = db;
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("bgv_parameter_profile::SaveDatabase: cannot open " + path + " for writing");
    out << j.dump(2) << "\n";
}

// Loads `path` and returns its Database. Throws std::runtime_error with a
// descriptive message (not a raw nlohmann::json parse exception) if the
// file is missing or malformed, per the "fail loudly on inconsistent
// profiles" requirement in the task spec.
inline Database LoadDatabase(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("bgv_parameter_profile::LoadDatabase: cannot open " + path +
                                  " -- generate it first with tools/generate_bgv_batched_profiles");

    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("bgv_parameter_profile::LoadDatabase: malformed JSON in " + path +
                                  ": " + e.what());
    }

    Database db;
    try {
        db = j.get<Database>();
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("bgv_parameter_profile::LoadDatabase: " + path +
                                  " does not match the expected profile-database schema: " + e.what());
    }
    return db;
}

// ── Runtime lookup ────────────────────────────────────────────────────────
//
// Selects the cheapest validated profile in `db` that can serve a circuit
// needing `required_depth` multiplicative levels and `required_slots`
// packed SIMD slots (N*N for a full-matrix-packed N x N Boolean matrix).
// "Cheapest" is deterministic: smallest ring_dimension, then smallest
// depth_capacity, then smallest plaintext_modulus (same ordering as
// ProfileLessThan/the stored sort order) -- this matches the task spec's
// selection rule and keeps behavior reproducible across runs.
//
// Returns nullptr (not an exception) when no profile satisfies the
// requirement -- callers are expected to turn that into an actionable
// error message naming the missing (depth, slots) pair and the generator
// command to fix it (see RequireProfile below), per the task spec's
// "return an actionable error ... do not silently choose the largest
// profile and do not start an online iterative search" requirement.
inline const Profile* SelectProfile(const Database& db, uint32_t required_depth, uint64_t required_slots) {
    const Profile* best = nullptr;
    for (const Profile& p : db.profiles) {
        if (!p.validated)
            continue;
        if (p.depth_capacity < required_depth)
            continue;
        if (static_cast<uint64_t>(p.ring_dimension) < required_slots) // slot_capacity() == ring_dimension
            continue;
        if (best == nullptr || ProfileLessThan(p, *best))
            best = &p;
    }
    return best;
}

// Same as SelectProfile, but throws a descriptive std::runtime_error naming
// the generator invocation that would produce a suitable profile, instead
// of returning nullptr. This is the entry point setup code should call --
// see bgv_batched::setup_from_profile in bgv_batched.hpp.
inline const Profile& RequireProfile(const Database& db, uint32_t required_depth, uint64_t required_slots) {
    const Profile* found = SelectProfile(db, required_depth, required_slots);
    if (found != nullptr)
        return *found;

    // Smallest N such that N*N >= required_slots, purely for the
    // suggested --nodes value in the error message below (the generator
    // itself recomputes required_depth/required_slots from N, so this is
    // just a convenience hint, not a value this function trusts).
    uint64_t suggested_n = 1;
    while (suggested_n * suggested_n < required_slots)
        ++suggested_n;

    throw std::runtime_error(
        "bgv_parameter_profile::RequireProfile: No validated BGV parameter profile supports this request.\n\n"
        "  Required multiplicative depth: " + std::to_string(required_depth) + "\n"
        "  Required SIMD slots: " + std::to_string(required_slots) + "\n\n"
        "Generate additional profiles with:\n"
        "  ./generate_bgv_batched_profiles --nodes=" + std::to_string(suggested_n));
}

} // namespace bgv_parameter_profile
} // namespace btc

#endif // TC_WITH_BGV_BATCHED
