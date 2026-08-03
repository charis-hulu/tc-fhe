#ifdef TC_WITH_BGV_BATCHED

#include <btc/algorithms.hpp>
#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/bgv_batched_profile_setup.hpp>
#include <btc/graph_io.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace bgv_batched = btc::bgv_batched;
using clock_type = std::chrono::high_resolution_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// Default graph if --graph is not given: same 4-node chain used by
// tfhe_closure/dghv_closure/bgv_closure, so all backends are directly
// comparable on identical input.
static btc::BoolMatrix default_graph() {
    btc::BoolMatrix A(4, 4, false);
    A.set(0, 1, true);
    A.set(1, 2, true);
    A.set(2, 3, true);
    return A;
}

namespace {

// Minimal manual argument parser, matching bgv_closure/dghv_closure's style.
//
//   bgv_batched_closure [--backend=bgv-batched] [--graph=<path>] [--T=<n>]
//                       [--profile-db=<path>]
//
// --backend exists to satisfy this project's explicit "--backend
// bgv-batched" selection requirement (see docs/bgv_batched.md); this binary
// only implements the bgv-batched backend, so any value other than
// "bgv-batched" is rejected with a clear error rather than silently
// ignored -- selecting a different backend means running tfhe_closure,
// dghv_closure, or bgv_closure instead (this repository picks backends at
// build/binary granularity, not via a shared runtime dispatcher -- see
// docs/bgv_batched.md for the rationale).
//
// Unlike earlier versions of this CLI, plaintext modulus / ring dimension /
// batch size are no longer caller-settable flags: bgv_batched::setup_from_profile
// derives them TOGETHER from the offline-validated profile database (see
// include/btc/bgv_batched_profile_setup.hpp and
// docs/bgv_parameter_profiles.md) instead of the caller guessing a single
// "safe" plaintext modulus per graph size (the ad hoc approach this project
// used before the profile database existed -- see
// docs/plaintext_modulus_issue.md for why that was unreliable).
struct Args {
    std::string backend = "bgv-batched";
    std::string graph_path;
    int T = 0;
    std::string profile_db = btc::bgv_batched::DefaultProfileDatabasePath();
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg.rfind("--backend=", 0) == 0) args.backend = value_after("--backend=");
        else if (arg.rfind("--graph=", 0) == 0) args.graph_path = value_after("--graph=");
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else if (arg.rfind("--profile-db=", 0) == 0) args.profile_db = value_after("--profile-db=");
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: bgv_batched_closure [--backend=bgv-batched] [--graph=<path>] [--T=<n>] "
                "[--profile-db=<path>]\n");
            std::exit(1);
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.backend != "bgv-batched") {
        std::fprintf(stderr,
            "Unknown or unsupported --backend '%s' for this binary. "
            "bgv_batched_closure only implements 'bgv-batched'. "
            "Use tfhe_closure/dghv_closure/bgv_closure for other backends "
            "(this repository selects backends at build/binary granularity, "
            "see docs/bgv_batched.md).\n",
            args.backend.c_str());
        return 1;
    }

    btc::BoolMatrix plain;
    if (!args.graph_path.empty()) {
        try {
            plain = btc::load_graph(args.graph_path);
        } catch (const std::exception& e) {
            std::printf("Failed to load %s: %s\n", args.graph_path.c_str(), e.what());
            return 1;
        }
    } else {
        plain = default_graph();
        std::puts("No --graph given, using built-in 4-node chain (0->1->2->3).");
    }

    const std::size_t N = plain.rows();
    // Default T = N (not N-1): the server operates on *encrypted* adjacency
    // data and cannot see whether the graph contains a cycle, so it cannot
    // safely assume the acyclic-only bound N-1 is sufficient (see the
    // correctness note in algorithms.hpp).
    const int T = args.T > 0 ? args.T : static_cast<int>(N);

    const uint32_t required_depth = btc::EstimateBGVBatchedDepth(N, T);
    const uint64_t required_slots = static_cast<uint64_t>(N) * static_cast<uint64_t>(N);

    std::printf("Backend: bgv-batched (BGV/OpenFHE, full-matrix SIMD packing)\n");
    std::printf("Number of nodes (N): %zu\n", N);
    std::printf("N*N (packed slots required): %llu\n", static_cast<unsigned long long>(required_slots));
    std::printf("Required multiplicative depth: %u\n", required_depth);

    std::puts("Looking up BGV parameter profile and generating crypto context/keys...");
    auto t_start = clock_type::now();
    auto t0 = clock_type::now();
    bgv_batched::Context ctx;
    try {
        ctx = bgv_batched::setup_from_profile(required_depth, required_slots, args.profile_db);
    } catch (const std::exception& e) {
        std::printf("BGV batched setup failed: %s\n", e.what());
        return 1;
    }
    double keygen_ms = ms_since(t0);

    const uint32_t available_slots = bgv_batched::available_slots(ctx);
    std::printf("Ring dimension: %u\n", bgv_batched::ring_dimension(ctx));
    std::printf("Available packed slots: %u\n", available_slots);
    std::printf("Security level: HEStd_128_classic\n");

    if (N * N > available_slots) {
        std::printf(
            "ERROR: N*N (%zu) exceeds available packed slots (%u). Full-matrix "
            "row-major packing requires the entire N x N matrix to fit in one "
            "ciphertext; tiled/multi-ciphertext packing is not implemented in "
            "this backend. This should not happen if setup_from_profile selected "
            "a valid profile -- generate a larger one with "
            "tools/generate_bgv_batched_profiles --nodes=%zu.\n",
            N * N, available_slots, N);
        return 1;
    }

    std::puts("Building masks and generating rotation keys...");
    t0 = clock_type::now();
    btc::BGVBatchedBackend backend(ctx, N);
    double setup_ms = ms_since(t0);
    std::printf("Rotation keys generated: %zu\n", ctx.generated_rotation_offsets.size());

    const int rounds = [&] {
        int r = 0, remaining = T;
        while (remaining > 1) {
            remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
            ++r;
        }
        return r;
    }();
    std::printf("Number of closure rounds: %d\n", rounds);

    std::puts("Encrypting (one packed ciphertext for the whole matrix)...");
    t0 = clock_type::now();
    auto enc_A = backend.Encrypt(plain);
    double encrypt_ms = ms_since(t0);

    std::printf("Running BTC (N=%zu, T=%d) under bgv-batched...\n", N, T);
    t0 = clock_type::now();
    btc::BGVBatchedMatrix enc_S;
    try {
        enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    } catch (const std::exception& e) {
        std::printf("bgv-batched evaluation failed: %s\n", e.what());
        return 1;
    }
    double btc_ms = ms_since(t0);

    std::puts("Decrypting...");
    t0 = clock_type::now();
    btc::BoolMatrix result;
    try {
        result = backend.Decrypt(enc_S);
    } catch (const std::exception& e) {
        std::printf("bgv-batched decryption failed: %s\n", e.what());
        return 1;
    }
    double decrypt_ms = ms_since(t0);

    double total_ms = ms_since(t_start);

    std::puts("\nReachability matrix (decrypted):");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::printf("%d ", result.get(i, j) ? 1 : 0);
        std::puts("");
    }

    auto plain_S = btc::bounded_transitive_closure(plain, T);
    bool ok = (result == plain_S);
    std::printf("\nMatches plaintext result: %s\n", ok ? "YES" : "NO");

    std::puts("\nComputation time:");
    std::printf("  setup (keygen):          %.3f ms\n", keygen_ms);
    std::printf("  masks + rotation keygen: %.3f ms\n", setup_ms);
    std::printf("  encrypt:                 %.3f ms\n", encrypt_ms);
    std::printf("  BTC eval:                %.3f ms\n", btc_ms);
    std::printf("  decrypt:                 %.3f ms\n", decrypt_ms);
    std::printf("  total:                   %.3f ms (%.3f s)\n", total_ms, total_ms / 1000.0);

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this example.");
    return 0;
}
#endif
