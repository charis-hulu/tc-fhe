#ifdef TC_WITH_BGV

#include <btc/algorithms.hpp>
#include <btc/bgv.hpp>
#include <btc/bgv_backend.hpp>
#include <btc/graph_io.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace bgv = btc::bgv;
using clock_type = std::chrono::high_resolution_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// Default graph if --graph is not given: same 4-node chain used by
// dghv_closure and tfhe_closure, so the three backends are directly
// comparable on identical input.
static btc::BoolMatrix default_graph() {
    // 4-node chain: 0->1->2->3
    btc::BoolMatrix A(4, 4, false);
    A.set(0, 1, true);
    A.set(1, 2, true);
    A.set(2, 3, true);
    return A;
}

namespace {

// Minimal manual argument parser, matching dghv_closure's style:
//
//   bgv_closure [--graph=<path>] [--nodes=<n>] [--T=<n>]
//               [--bgv-plaintext-modulus=<n>] [--bgv-depth=<n>]
//               [--bgv-ring-dim=<n>] [--bgv-batch-size=<n>]
//
// --nodes is only used to size the built-in default graph when --graph is
// not given (it must match the default graph's 4 nodes unless a custom
// --graph is supplied); it does not resize a loaded graph.
struct Args {
    std::string graph_path;
    int T = 0;
    uint64_t plaintext_modulus = 65537;
    uint32_t depth_override = 0; // 0 means "use the estimated depth"
    uint32_t ring_dimension = 0; // 0 means "automatic"
    uint32_t batch_size = 0;     // 0 means "automatic"
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg.rfind("--graph=", 0) == 0) args.graph_path = value_after("--graph=");
        else if (arg.rfind("--nodes=", 0) == 0) { /* accepted for CLI-shape parity; default graph is fixed-size */ }
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else if (arg.rfind("--bgv-plaintext-modulus=", 0) == 0)
            args.plaintext_modulus = std::stoull(value_after("--bgv-plaintext-modulus="));
        else if (arg.rfind("--bgv-depth=", 0) == 0)
            args.depth_override = static_cast<uint32_t>(std::stoul(value_after("--bgv-depth=")));
        else if (arg.rfind("--bgv-ring-dim=", 0) == 0)
            args.ring_dimension = static_cast<uint32_t>(std::stoul(value_after("--bgv-ring-dim=")));
        else if (arg.rfind("--bgv-batch-size=", 0) == 0)
            args.batch_size = static_cast<uint32_t>(std::stoul(value_after("--bgv-batch-size=")));
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: bgv_closure [--graph=<path>] [--T=<n>] "
                "[--bgv-plaintext-modulus=<n>] [--bgv-depth=<n>] "
                "[--bgv-ring-dim=<n>] [--bgv-batch-size=<n>]\n");
            std::exit(1);
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

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
    // Default T = N, not N-1: the server evaluating this circuit works on
    // *encrypted* adjacency data and cannot see whether the graph contains
    // a cycle, so it cannot safely assume T=N-1 (only valid for acyclic
    // graphs) is sufficient. T=N is the bound that is correct for every
    // directed graph (see the correctness note in algorithms.hpp).
    const int T = args.T > 0 ? args.T : static_cast<int>(N);

    const uint32_t estimated_depth = btc::EstimateBGVDepth(N, T);
    const uint32_t configured_depth = args.depth_override > 0 ? args.depth_override : estimated_depth;

    bgv::Params params;
    params.plaintext_modulus = args.plaintext_modulus;
    params.multiplicative_depth = configured_depth;
    params.ring_dimension = args.ring_dimension;
    params.batch_size = args.batch_size;

    std::printf("Backend: BGV/OpenFHE\n");
    std::printf("Number of nodes: %zu\n", N);
    std::printf("Plaintext modulus: %llu\n", static_cast<unsigned long long>(params.plaintext_modulus));
    std::printf("Estimated multiplicative depth: %u\n", estimated_depth);
    std::printf("Configured multiplicative depth: %u\n", configured_depth);

    if (configured_depth < estimated_depth) {
        std::printf(
            "WARNING: configured multiplicative depth (%u) is lower than the "
            "estimated requirement (%u). Ciphertexts will likely exhaust their "
            "level budget mid-computation and decryption may fail or produce "
            "wrong bits. Increase --bgv-depth to at least %u.\n",
            configured_depth, estimated_depth, estimated_depth);
    }

    std::puts("Generating BGV crypto context and keys...");
    auto t_start = clock_type::now();
    auto t0 = clock_type::now();
    bgv::Context ctx;
    try {
        ctx = bgv::setup(params);
    } catch (const std::exception& e) {
        std::printf("BGV setup failed: %s\n", e.what());
        return 1;
    }
    double keygen_ms = ms_since(t0);

    std::printf("Ring dimension: %u\n", bgv::ring_dimension(ctx));
    std::printf("Security level: HEStd_128_classic\n");

    const int rounds = [&] {
        int r = 0, remaining = T;
        while (remaining > 1) {
            remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
            ++r;
        }
        return r;
    }();
    std::printf("Number of closure rounds: %d\n", rounds);

    std::puts("Encrypting...");
    t0 = clock_type::now();
    auto enc_A = btc::BGVBackend::encrypt(plain, ctx);
    double encrypt_ms = ms_since(t0);

    std::printf("Running BTC (N=%zu, T=%d) under BGV...\n", N, T);
    btc::BGVBackend backend(ctx);
    t0 = clock_type::now();
    btc::BGVMatrix enc_S(0);
    try {
        enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    } catch (const std::exception& e) {
        std::printf("BGV evaluation failed: %s\n", e.what());
        return 1;
    }
    double btc_ms = ms_since(t0);

    std::puts("Decrypting...");
    t0 = clock_type::now();
    btc::BoolMatrix result;
    try {
        result = btc::BGVBackend::decrypt(enc_S, ctx);
    } catch (const std::exception& e) {
        std::printf("BGV decryption failed: %s\n", e.what());
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
    std::printf("  setup (keygen):  %.3f ms\n", keygen_ms);
    std::printf("  encrypt:         %.3f ms\n", encrypt_ms);
    std::printf("  BTC eval:        %.3f ms\n", btc_ms);
    std::printf("  decrypt:         %.3f ms\n", decrypt_ms);
    std::printf("  total:           %.3f ms (%.3f s)\n", total_ms, total_ms / 1000.0);

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV=ON (and OpenFHE installed) to enable this example.");
    return 0;
}
#endif
