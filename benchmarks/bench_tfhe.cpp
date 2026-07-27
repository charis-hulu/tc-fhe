#ifdef TC_WITH_TFHE

#include <btc/algorithms.hpp>
#include <btc/graph_io.hpp>
#include <btc/tfhe_backend.hpp>
#include <btc/tfhe_params_io.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

namespace {

void print_matrix(const btc::BoolMatrix& M) {
    for (std::size_t i = 0; i < M.rows(); ++i) {
        for (std::size_t j = 0; j < M.cols(); ++j)
            std::printf("%d ", M.get(i, j) ? 1 : 0);
        std::puts("");
    }
}

int log2_ceil(std::size_t n) {
    int t = 0;
    std::size_t v = 1;
    while (v < n) { v <<= 1; ++t; }
    return t < 1 ? 1 : t; // T must be >= 1
}

std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> names = {"graph_N4", "graph_N8", "graph_N16"};
    std::string params_path;
    std::string params_label = "default";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const std::string prefix = "--params=";
        if (arg.rfind(prefix, 0) == 0) {
            params_path = arg.substr(prefix.size());
        } else {
            names = {arg};
        }
    }

    BooleanClientKey* ckey = nullptr;
    BooleanServerKey* skey = nullptr;
    if (!params_path.empty()) {
        std::printf("Generating keys from %s (shared across all graphs)...\n", params_path.c_str());
        BooleanParameters params = btc::load_boolean_parameters(params_path);
        if (boolean_gen_keys_with_parameters(params, &ckey, &skey) != 0) {
            std::fputs("Key generation failed\n", stderr);
            return 1;
        }
        params_label = params_path;
    } else {
        std::puts("Generating keys with default parameters (shared across all graphs)...");
        if (boolean_gen_keys_with_default_parameters(&ckey, &skey) != 0) {
            std::fputs("Key generation failed\n", stderr);
            return 1;
        }
    }
    btc::TFHEBackend backend(skey);

    const std::string csv_path = "data/bench_tfhe_results.csv";
    bool csv_exists = std::ifstream(csv_path).good();
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists) {
        csv << "timestamp,graph,N,T,fhe_ms,matches_plaintext,params\n";
    }

    int passed = 0;
    int failed = 0;

    for (const auto& name : names) {
        std::string path = "data/" + name + ".txt";
        btc::BoolMatrix A;
        try {
            A = btc::load_graph(path);
        } catch (const std::exception& e) {
            std::printf("Skipping %s: %s\n", path.c_str(), e.what());
            continue;
        }

        const std::size_t N = A.rows();
        const int T = log2_ceil(N);

        std::printf("\n=== Graph: %s (N=%zu, T=ceil(log2(N))=%d) ===\n",
                     name.c_str(), N, T);

        auto plain_S = btc::bounded_transitive_closure(A, T);

        auto enc_A = btc::TFHEBackend::encrypt(A, ckey);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto result = btc::TFHEBackend::decrypt(enc_S, ckey);
        bool ok = (result == plain_S);

        std::printf("  FHE computation: %.1f ms (%.3f s)\n", ms, ms / 1000.0);
        std::puts("  Decrypted reachability matrix:");
        print_matrix(result);
        std::printf("  Matches plaintext validator: %s\n", ok ? "YES" : "NO");

        csv << timestamp_now() << ',' << name << ',' << N << ',' << T << ','
            << ms << ',' << (ok ? "YES" : "NO") << ',' << params_label << '\n';

        if (ok) {
            ++passed;
        } else {
            ++failed;
            std::puts("  Expected:");
            print_matrix(plain_S);
        }
    }

    std::printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);

    boolean_destroy_client_key(ckey);
    boolean_destroy_server_key(skey);
    return failed == 0 ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_TFHE=ON to enable TFHE benchmark.");
    return 0;
}
#endif
