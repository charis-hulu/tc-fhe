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

namespace {

// Usage: bench_bgv [--graph=<name>] [--T=<n>]
//                   [--bgv-plaintext-modulus=<n>] [--bgv-depth=<n>]
//                   [--bgv-ring-dim=<n>] [--bgv-batch-size=<n>]
//
// --bgv-depth overrides the estimated depth (btc::EstimateBGVDepth); it is
// the caller's job to pick it big enough for the resulting circuit -- this
// benchmark only checks and reports, it does not adjust anything.
struct Args {
    std::string graph_name = "graph_N4";
    int T = 0;
    uint64_t plaintext_modulus = 65537;
    uint32_t depth_override = 0;
    uint32_t ring_dimension = 0;
    uint32_t batch_size = 0;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
        if (arg.rfind("--graph=", 0) == 0) args.graph_name = value_after("--graph=");
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
                "Usage: bench_bgv [--graph=<name>] [--T=<n>] "
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

    const uint32_t estimated_depth = btc::EstimateBGVDepth(N, T);
    const uint32_t configured_depth = args.depth_override > 0 ? args.depth_override : estimated_depth;

    bgv::Params params;
    params.plaintext_modulus = args.plaintext_modulus;
    params.multiplicative_depth = configured_depth;
    params.ring_dimension = args.ring_dimension;
    params.batch_size = args.batch_size;

    std::printf("=== BGV/OpenFHE benchmark: %s (N=%zu, T=%d) ===\n",
                args.graph_name.c_str(), N, T);
    std::printf("  plaintext_modulus=%llu estimated_depth=%u configured_depth=%u\n",
                static_cast<unsigned long long>(params.plaintext_modulus),
                estimated_depth, configured_depth);

    if (configured_depth < estimated_depth) {
        std::printf("WARNING: configured depth (%u) is below the estimated requirement (%u). "
                     "Proceeding anyway -- decryption may fail or produce wrong bits.\n",
                     configured_depth, estimated_depth);
    }

    auto t0 = clock_type::now();
    bgv::Context ctx;
    try {
        ctx = bgv::setup(params);
    } catch (const std::exception& e) {
        std::printf("BGV setup failed: %s\n", e.what());
        return 1;
    }
    double keygen_ms = ms_since(t0);

    std::printf("  ring_dimension=%u\n", bgv::ring_dimension(ctx));

    t0 = clock_type::now();
    auto ct_a = bgv::encrypt_bit(ctx, true);
    double encrypt_ms = ms_since(t0);

    auto ct_b = bgv::encrypt_bit(ctx, false);

    t0 = clock_type::now();
    auto ct_and = bgv::encrypted_and(ctx, ct_a, ct_b);
    double and_ms = ms_since(t0);

    t0 = clock_type::now();
    auto ct_or = bgv::encrypted_or(ctx, ct_a, ct_b);
    double or_ms = ms_since(t0);
    (void)ct_and; (void)ct_or;

    t0 = clock_type::now();
    auto enc_A = btc::BGVBackend::encrypt(A, ctx);
    double encrypt_matrix_ms = ms_since(t0);

    btc::BGVBackend backend(ctx);
    t0 = clock_type::now();
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    double btc_ms = ms_since(t0);

    t0 = clock_type::now();
    auto result = btc::BGVBackend::decrypt(enc_S, ctx);
    double decrypt_ms = ms_since(t0);

    auto plain_S = btc::bounded_transitive_closure(A, T);
    bool ok = (result == plain_S);

    std::printf("  keygen (context+keys):%.3f ms\n", keygen_ms);
    std::printf("  encrypt (1 bit):      %.3f ms\n", encrypt_ms);
    std::printf("  encrypted AND:        %.3f ms\n", and_ms);
    std::printf("  encrypted OR:         %.3f ms\n", or_ms);
    std::printf("  encrypt (N*N matrix): %.3f ms\n", encrypt_matrix_ms);
    std::printf("  full BTC:             %.3f ms (%.3f s)\n", btc_ms, btc_ms / 1000.0);
    std::printf("  decrypt (N*N matrix): %.3f ms\n", decrypt_ms);
    std::printf("  matches plaintext:    %s\n", ok ? "YES" : "NO");
    std::puts("\n  Comparisons across TFHE/DGHV/BGV are only meaningful when all");
    std::puts("  parameter sets target comparable security -- see docs/bgv.md.");

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV=ON (and OpenFHE installed) to enable BGV benchmark.");
    return 0;
}
#endif
