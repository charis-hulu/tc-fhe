#ifdef TC_WITH_DGHV

#include <btc/algorithms.hpp>
#include <btc/dghv.hpp>
#include <btc/dghv_backend.hpp>
#include <btc/graph_io.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace dghv = btc::dghv;
using clock_type = std::chrono::high_resolution_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// Default graph if --graph is not given: small on purpose. DGHV (unlike
// TFHE) has no bootstrapping, so noise -- and therefore how large eta_bits
// must be, and how big the GMP integers get -- grows roughly as
// rho_bits * 2^depth (see dghv.hpp's noise_bits_after_depth). A larger
// graph or larger T needs a much bigger --eta-bits and will run much
// slower; see docs/dghv.md section 10.
static btc::BoolMatrix default_graph() {
    // 4-node chain: 0->1->2->3
    btc::BoolMatrix A(4, 4, false);
    A.set(0, 1, true);
    A.set(1, 2, true);
    A.set(2, 3, true);
    return A;
}

namespace {

// Minimal manual argument parser: every DGHV parameter is supplied by the
// user, not derived automatically. Usage:
//
//   dghv_closure [--graph=<path>] [--T=<n>]
//                [--eta-bits=<n>] [--rho-bits=<n>] [--q-bits=<n>]
//
// gamma_bits = eta_bits + q_bits is derived automatically (see dghv.hpp);
// there is no separate --gamma-bits flag, so it can never disagree with
// --q-bits.
struct Args {
    std::string graph_path;
    int T = 0;
    int eta_bits = -1;
    int rho_bits = -1;
    int q_bits = -1;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg.rfind("--graph=", 0) == 0) args.graph_path = value_after("--graph=");
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else if (arg.rfind("--eta-bits=", 0) == 0) args.eta_bits = std::stoi(value_after("--eta-bits="));
        else if (arg.rfind("--rho-bits=", 0) == 0) args.rho_bits = std::stoi(value_after("--rho-bits="));
        else if (arg.rfind("--q-bits=", 0) == 0) args.q_bits = std::stoi(value_after("--q-bits="));
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: dghv_closure [--graph=<path>] [--T=<n>] "
                "[--eta-bits=<n>] [--rho-bits=<n>] [--q-bits=<n>]\n");
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

    // Fallback values for any of eta_bits/rho_bits/q_bits the user did not
    // override on the command line -- fixed numbers, NOT computed from the
    // graph. These happen to be large enough for the built-in 4-node/T=3
    // default graph (which needs depth 8); a bigger --graph or --T will
    // likely need a bigger --eta-bits too -- rerun and check the printed
    // "max depth supported".
    const int fallback_eta_bits = 9000;
    const int fallback_rho_bits = 32;
    const int fallback_q_bits = 8192;
    dghv::Params params = dghv::manual_params(
        args.eta_bits > 0 ? args.eta_bits : fallback_eta_bits,
        args.rho_bits > 0 ? args.rho_bits : fallback_rho_bits,
        args.q_bits   > 0 ? args.q_bits   : fallback_q_bits,
        /*max_depth=*/0);
    dghv::validate_params(params);

    // Multiplicative depth is still computed from the circuit purely for
    // reporting/validation -- it does NOT influence eta_bits/rho_bits/q_bits,
    // which come entirely from the arguments (or their fallback) above.
    const int depth = btc::transitive_closure_depth(N, T);
    std::printf("Graph: N=%zu, T=%d -> required multiplicative depth = %d\n", N, T, depth);
    std::printf("Using parameters: eta_bits=%d rho_bits=%d q_bits=%d (gamma_bits=%d, "
                "max depth supported = %d)\n",
                params.eta_bits, params.rho_bits, params.q_bits, params.gamma_bits,
                dghv::max_supported_depth(params));

    // Does NOT abort: only warns, so the user can deliberately try
    // under-sized parameters and see decryption fail, if they want to.
    try {
        dghv::require_depth(params, depth);
    } catch (const std::exception& e) {
        std::printf("WARNING: %s\n", e.what());
        std::puts("Proceeding anyway -- decryption may produce wrong bits.");
    }

    std::mt19937_64 rng(42);
    auto t_start = clock_type::now();

    std::puts("Generating key...");
    auto t0 = clock_type::now();
    auto key = dghv::keygen(params, rng);
    double keygen_ms = ms_since(t0);

    std::puts("Encrypting...");
    t0 = clock_type::now();
    auto enc_A = btc::DGHVBackend::encrypt(plain, key, params, rng);
    double encrypt_ms = ms_since(t0);

    std::printf("Running BTC (N=%zu, T=%d) under DGHV...\n", N, T);
    btc::DGHVBackend backend;
    t0 = clock_type::now();
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    double btc_ms = ms_since(t0);

    std::puts("Decrypting...");
    t0 = clock_type::now();
    auto result = btc::DGHVBackend::decrypt(enc_S, key);
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
    std::printf("  keygen:   %.3f ms\n", keygen_ms);
    std::printf("  encrypt:  %.3f ms\n", encrypt_ms);
    std::printf("  BTC eval: %.3f ms\n", btc_ms);
    std::printf("  decrypt:  %.3f ms\n", decrypt_ms);
    std::printf("  total:    %.3f ms (%.3f s)\n", total_ms, total_ms / 1000.0);

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_DGHV=ON to enable this example.");
    return 0;
}
#endif
