#ifdef TC_WITH_TFHE

#include <btc/algorithms.hpp>
#include <btc/graph_io.hpp>
#include <btc/tfhe_backend.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

// Default graph if --graph is not given: same 4-node chain used by
// dghv_closure and bgv_closure, so the three backends are directly
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

// Minimal manual argument parser, matching dghv_closure/bgv_closure's style:
//
//   tfhe_closure [--graph=<path>] [--T=<n>]
struct Args {
    std::string graph_path;
    int T = 0;
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
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr, "Usage: tfhe_closure [--graph=<path>] [--T=<n>]\n");
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
    // directed graph (see the correctness note in algorithms.hpp). Unlike
    // DGHV/BGV, TFHE bootstraps after every gate, so a larger T costs more
    // wall-clock time but never risks a depth-budget failure.
    const int T = args.T > 0 ? args.T : static_cast<int>(N);

    std::puts("Generating keys...");
    BooleanClientKey* ckey = nullptr;
    BooleanServerKey* skey = nullptr;
    if (boolean_gen_keys_with_default_parameters(&ckey, &skey) != 0) {
        std::fputs("Key generation failed\n", stderr);
        return 1;
    }

    std::puts("Encrypting...");
    auto enc_A = btc::TFHEBackend::encrypt(plain, ckey);

    std::printf("Running BTC (N=%zu, T=%d) under FHE...\n", N, T);
    btc::TFHEBackend backend(skey);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);

    std::puts("Decrypting...");
    auto result = btc::TFHEBackend::decrypt(enc_S, ckey);

    std::puts("\nReachability matrix (decrypted):");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::printf("%d ", result.get(i, j) ? 1 : 0);
        std::puts("");
    }

    auto plain_S = btc::bounded_transitive_closure(plain, T);
    bool ok = (result == plain_S);
    std::printf("\nMatches plaintext result: %s\n", ok ? "YES" : "NO");

    boolean_destroy_client_key(ckey);
    boolean_destroy_server_key(skey);
    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_TFHE=ON to enable this example.");
    return 0;
}
#endif
