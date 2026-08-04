#ifdef TC_WITH_BGV_BATCHED

#include <btc/algorithms.hpp>
#include <btc/bgv_batched.hpp>
#include <btc/bgv_batched_backend.hpp>
#include <btc/bgv_batched_profile_setup.hpp>
#include <btc/graph_io.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

namespace bgv_batched = btc::bgv_batched;
using clock_type = std::chrono::high_resolution_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

namespace {

std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// Usage: bench_bgv_batched [--graph=<name>] [--T=<n>] [--profile-db=<path>]
//
// Plaintext modulus / ring dimension / batch size are no longer
// caller-settable: bgv_batched::setup_from_profile derives them TOGETHER
// from the offline-validated profile database instead of guessing a single
// "safe" plaintext modulus per graph size -- see
// include/btc/bgv_batched_profile_setup.hpp and docs/plaintext_modulus_issue.md.
struct Args {
    std::string graph_name = "graph_N4";
    int T = 0;
    std::string profile_db = btc::bgv_batched::DefaultProfileDatabasePath();
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
        if (arg.rfind("--graph=", 0) == 0) args.graph_name = value_after("--graph=");
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else if (arg.rfind("--profile-db=", 0) == 0) args.profile_db = value_after("--profile-db=");
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: bench_bgv_batched [--graph=<name>] [--T=<n>] [--profile-db=<path>]\n");
            std::exit(1);
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::string path = "data/" + args.graph_name + ".txt";
    btc::BoolMatrix A;
    try {
        A = btc::load_graph(path);
    } catch (const std::exception& e) {
        std::printf("Failed to load %s: %s\n", path.c_str(), e.what());
        return 1;
    }

    const std::size_t N = A.rows();
    const int T = args.T > 0 ? args.T : static_cast<int>(N); // default: full closure (T=N)

    const uint32_t required_depth = btc::EstimateBGVBatchedDepth(N, T);
    const uint64_t required_slots = static_cast<uint64_t>(N) * static_cast<uint64_t>(N);

    const int rounds = [&] {
        int r = 0, remaining = T;
        while (remaining > 1) {
            remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
            ++r;
        }
        return r;
    }();

    std::printf("=== bgv-batched (BGV/OpenFHE, packed) benchmark: %s (N=%zu, T=%d) ===\n",
                args.graph_name.c_str(), N, T);
    std::printf("  backend=bgv-batched N=%zu N*N=%llu\n", N, static_cast<unsigned long long>(required_slots));
    std::printf("  required_depth=%u rounds=%d\n", required_depth, rounds);

    // ── Phase 1: crypto-context creation + key generation ────────────────
    // (includes the profile-database lookup -- see setup_from_profile.)
    auto t0 = clock_type::now();
    bgv_batched::Context ctx;
    try {
        ctx = bgv_batched::setup_from_profile(required_depth, required_slots, args.profile_db);
    } catch (const std::exception& e) {
        std::printf("bgv-batched setup failed: %s\n", e.what());
        return 1;
    }
    double keygen_ms = ms_since(t0);

    const uint32_t ring_dim = bgv_batched::ring_dimension(ctx);
    const uint32_t available_slots = bgv_batched::available_slots(ctx);
    std::printf("  ring_dimension=%u available_slots=%u\n", ring_dim, available_slots);

    if (N * N > available_slots) {
        std::printf("ERROR: N*N (%zu) exceeds available packed slots (%u); "
                     "tiled packing is not implemented. Skipping.\n", N * N, available_slots);
        return 1;
    }

    // ── Phase 2: mask/plaintext preparation + rotation-key generation ────
    // (BGVBatchedBackend's constructor does both; timed together since
    // they're both one-time setup that must happen before any homomorphic
    // op, distinct from keygen above and from encryption below.)
    t0 = clock_type::now();
    btc::BGVBatchedBackend backend(ctx, N);
    double mask_setup_ms = ms_since(t0);
    const std::size_t num_rotation_keys = ctx.generated_rotation_offsets.size();

    // ── Phase 3: encryption ───────────────────────────────────────────────
    t0 = clock_type::now();
    auto enc_A = backend.Encrypt(A);
    double encrypt_ms = ms_since(t0);

    // ── Phase 4: homomorphic TC evaluation ───────────────────────────────
    t0 = clock_type::now();
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    double btc_ms = ms_since(t0);

    // ── Phase 5: decryption and decoding ─────────────────────────────────
    t0 = clock_type::now();
    auto result = backend.Decrypt(enc_S);
    double decrypt_ms = ms_since(t0);

    auto plain_S = btc::bounded_transitive_closure(A, T);
    bool ok = (result == plain_S);

    // Operation-count estimates, matching the task's "expected operation
    // structure" accounting: N ciphertext-ciphertext AND mults + (N-1)
    // OR-tree mults per matmul round, ~2N(N-1) rotations per matmul round in
    // the straightforward implementation, times `rounds` matmul-bearing
    // recursion levels (sum_powers_recursive does one matmul(P,P) and one
    // matmul(P,S) per level, i.e. 2 matmuls/level, plus 1 OR/level).
    const std::size_t matmuls_per_round = 2;
    const std::size_t estimated_cc_mults =
        static_cast<std::size_t>(rounds) * matmuls_per_round * (2 * N - 1 > 0 ? (2 * N - 1) : 0);
    const std::size_t estimated_rotations =
        static_cast<std::size_t>(rounds) * matmuls_per_round *
        (N > 1 ? 2 * N * (N - 1) : 0);

    std::printf("\n  --- timings (phases reported separately, not combined) ---\n");
    std::printf("  crypto-context + keygen:       %.3f ms\n", keygen_ms);
    std::printf("  mask prep + rotation keygen:   %.3f ms\n", mask_setup_ms);
    std::printf("  encrypt (1 packed matrix ct):  %.3f ms\n", encrypt_ms);
    std::printf("  homomorphic BTC evaluation:    %.3f ms (%.3f s)\n", btc_ms, btc_ms / 1000.0);
    std::printf("  decrypt + decode:              %.3f ms\n", decrypt_ms);
    std::printf("  total (sum of the above):      %.3f ms\n",
                keygen_ms + mask_setup_ms + encrypt_ms + btc_ms + decrypt_ms);

    std::printf("\n  --- operation counts ---\n");
    std::printf("  number of TC rounds:                 %d\n", rounds);
    std::printf("  number of generated rotation keys:    %zu\n", num_rotation_keys);
    std::printf("  estimated rotations:                  %zu\n", estimated_rotations);
    std::printf("  estimated ciphertext-ciphertext mults:%zu\n", estimated_cc_mults);

    std::printf("\n  correctness result: %s\n", ok ? "PASS (matches plaintext)" : "FAIL");
    std::puts("\n  Comparisons across legacy/bgv/bgv-batched are only meaningful when all");
    std::puts("  parameter sets target comparable security -- see docs/bgv_batched.md.");
    std::puts("  Use benchmarks/bench_closure, bench_bgv, and bench_bgv_batched on the");
    std::puts("  same --graph to compare backends without modifying any source file.");

    const std::string csv_path = "data/bench_bgv_batched_results.csv";
    bool csv_exists = std::ifstream(csv_path).good();
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists) {
        csv << "timestamp,graph,N,T,ring_dimension,keygen_ms,mask_setup_ms,encrypt_ms,btc_ms,"
               "decrypt_ms,total_ms,matches_plaintext\n";
    }
    csv << timestamp_now() << ',' << args.graph_name << ',' << N << ',' << T << ','
        << ring_dim << ',' << keygen_ms << ',' << mask_setup_ms << ',' << encrypt_ms << ','
        << btc_ms << ',' << decrypt_ms << ','
        << (keygen_ms + mask_setup_ms + encrypt_ms + btc_ms + decrypt_ms) << ','
        << (ok ? "YES" : "NO") << '\n';

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this benchmark.");
    return 0;
}
#endif
