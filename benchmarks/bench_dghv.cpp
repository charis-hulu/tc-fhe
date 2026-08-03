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

namespace {

// Usage: bench_dghv [--graph=<name>] [--tier=toy|experiment|secure-target]
//                    [--eta-bits=<n>] [--rho-bits=<n>] [--q-bits=<n>] [--T=<n>]
//
// --eta-bits/--rho-bits/--q-bits override the chosen --tier's fixed values
// field by field; none of them are derived from the graph or from --T.
// gamma_bits = eta_bits + q_bits is always derived, never a separate flag.
// It is the caller's job to pick eta_bits big enough for the resulting
// circuit depth -- require_depth() below only checks and reports, it does
// not adjust anything.
struct Args {
    std::string graph_name = "graph_N4";
    std::string tier = "experiment";
    int eta_bits = -1;
    int rho_bits = -1;
    int q_bits = -1;
    int T = 0;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
        if (arg.rfind("--graph=", 0) == 0) args.graph_name = value_after("--graph=");
        else if (arg.rfind("--tier=", 0) == 0) args.tier = value_after("--tier=");
        else if (arg.rfind("--eta-bits=", 0) == 0) args.eta_bits = std::stoi(value_after("--eta-bits="));
        else if (arg.rfind("--rho-bits=", 0) == 0) args.rho_bits = std::stoi(value_after("--rho-bits="));
        else if (arg.rfind("--q-bits=", 0) == 0) args.q_bits = std::stoi(value_after("--q-bits="));
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: bench_dghv [--graph=<name>] [--tier=toy|experiment|secure-target] "
                "[--eta-bits=<n>] [--rho-bits=<n>] [--q-bits=<n>] [--T=<n>]\n");
            std::exit(1);
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    // graph_N4 by default: DGHV has no bootstrapping, so eta_bits (and GMP
    // integer size) must be large enough for the circuit's multiplicative
    // depth -- noise grows roughly as rho_bits * 2^depth (see dghv.hpp's
    // noise_bits_after_depth). graph_N8 at T=N=8 needs depth 15, which needs
    // a key hundreds of thousands of bits long unless you raise --eta-bits
    // far beyond the tier defaults; graph_N4 needs only depth 8. Pass a
    // larger --graph (and a correspondingly larger --eta-bits) explicitly to
    // see how fast that cost grows.
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

    dghv::Params base;
    if (args.tier == "toy") base = dghv::toy_params(/*max_depth=*/0);
    else if (args.tier == "experiment") base = dghv::experiment_params(/*max_depth=*/0);
    else if (args.tier == "secure-target") base = dghv::secure_target_params(/*max_depth=*/0);
    else {
        std::printf("Unknown tier '%s' (expected toy|experiment|secure-target)\n", args.tier.c_str());
        return 1;
    }
    // --eta-bits/--rho-bits/--q-bits override the tier's fixed fields one at
    // a time; anything not overridden keeps the chosen tier's value
    // verbatim. gamma_bits is always re-derived from the resulting
    // eta_bits + q_bits by manual_params, never copied from `base`.
    dghv::Params params = dghv::manual_params(
        args.eta_bits > 0 ? args.eta_bits : base.eta_bits,
        args.rho_bits > 0 ? args.rho_bits : base.rho_bits,
        args.q_bits   > 0 ? args.q_bits   : base.q_bits,
        /*max_depth=*/0);
    params.tier = base.tier;
    params.note = base.note;

    // Multiplicative depth is computed from the circuit purely for
    // reporting/validation; it does not feed back into eta_bits/rho_bits/q_bits.
    const int depth = btc::transitive_closure_depth(N, T);

    std::printf("=== DGHV benchmark: %s (N=%zu, T=%d, depth=%d, params=%s) ===\n",
                args.graph_name.c_str(), N, T, depth, params.tier.c_str());
    std::printf("  eta_bits=%d rho_bits=%d q_bits=%d (gamma_bits=%d, max depth supported = %d)\n",
                params.eta_bits, params.rho_bits, params.q_bits, params.gamma_bits,
                dghv::max_supported_depth(params));
    std::printf("NOTE: '%s' parameters -- %s\n", params.tier.c_str(), params.note.c_str());

    try {
        dghv::require_depth(params, depth);
    } catch (const std::exception& e) {
        std::printf("WARNING: %s\n", e.what());
        std::puts("Proceeding anyway -- decryption may produce wrong bits.");
    }

    std::mt19937_64 rng(42);

    auto t0 = clock_type::now();
    auto key = dghv::keygen(params, rng);
    double keygen_ms = ms_since(t0);

    t0 = clock_type::now();
    auto ct_a = dghv::encrypt_bit(key, params, true, rng);
    double encrypt_ms = ms_since(t0);

    auto ct_b = dghv::encrypt_bit(key, params, false, rng);

    t0 = clock_type::now();
    auto ct_and = dghv::encrypted_and(ct_a, ct_b);
    double and_ms = ms_since(t0);

    t0 = clock_type::now();
    auto ct_or = dghv::encrypted_or(ct_a, ct_b);
    double or_ms = ms_since(t0);
    (void)ct_and; (void)ct_or;

    t0 = clock_type::now();
    auto enc_A = btc::DGHVBackend::encrypt(A, key, params, rng);
    double encrypt_matrix_ms = ms_since(t0);

    btc::DGHVBackend backend;
    t0 = clock_type::now();
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    double btc_ms = ms_since(t0);

    t0 = clock_type::now();
    auto result = btc::DGHVBackend::decrypt(enc_S, key);
    double decrypt_ms = ms_since(t0);

    auto plain_S = btc::bounded_transitive_closure(A, T);
    bool ok = (result == plain_S);

    std::printf("  keygen:              %.3f ms\n", keygen_ms);
    std::printf("  encrypt (1 bit):     %.3f ms\n", encrypt_ms);
    std::printf("  encrypted AND:       %.3f ms\n", and_ms);
    std::printf("  encrypted OR:        %.3f ms\n", or_ms);
    std::printf("  encrypt (N*N matrix):%.3f ms\n", encrypt_matrix_ms);
    std::printf("  full BTC:            %.3f ms (%.3f s)\n", btc_ms, btc_ms / 1000.0);
    std::printf("  decrypt (N*N matrix):%.3f ms\n", decrypt_ms);
    std::printf("  matches plaintext:   %s\n", ok ? "YES" : "NO");
    std::puts("\n  Comparisons across TFHE and DGHV are only meaningful when both");
    std::puts("  parameter sets are documented and target comparable security --");
    std::puts("  see docs/dghv.md and data/tfhe_params/README.md.");

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_DGHV=ON to enable DGHV benchmark.");
    return 0;
}
#endif
