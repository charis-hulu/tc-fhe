#ifdef TC_WITH_TFHE_GPU

#include <btc/algorithms.hpp>
#include <btc/graph_io.hpp>
#include <btc/tfhe_gpu_backend.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

namespace {

void print_matrix(const btc::BoolMatrix& M) {
    for (std::size_t i = 0; i < M.rows(); ++i) {
        for (std::size_t j = 0; j < M.cols(); ++j)
            std::printf("%d ", M.get(i, j) ? 1 : 0);
        std::puts("");
    }
}

// Full transitive closure needs T = N hops (simple paths are <= N-1 edges,
// simple cycles are <= N) -- see the correctness note in algorithms.hpp.
int full_closure_T(std::size_t n) {
    return n >= 1 ? static_cast<int>(n) : 1; // T must be >= 1
}

std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// GPU decompressor requires MessageModulus 2 or 4 (1 or 2 bits per block).
// MESSAGE_2_CARRY_2 (MessageModulus=4) is GPU-compatible; unlike the CPU
// boolean backend, this parameter set is fixed (no --params= override).
const char* kParamsLabel = "shortint_message_2_carry_2_ks_pbs_tuniform_2m128";

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("Usage: bench_tfhe_gpu <graph_name>\n", stderr);
        return 1;
    }
    std::string graph_name = argv[1];
    std::string path = "data/" + graph_name + ".txt";

    btc::BoolMatrix plain;
    try {
        plain = btc::load_graph(path);
    } catch (const std::exception& e) {
        std::printf("Failed to load %s: %s\n", path.c_str(), e.what());
        return 1;
    }

    const std::size_t N = plain.rows();
    const int T = full_closure_T(N);

    std::printf("=== Graph: %s (N=%zu, T=N=%d) ===\n", graph_name.c_str(), N, T);

    std::puts("Generating keys...");
    ConfigBuilder* cfg_builder = nullptr;
    Config*        cfg         = nullptr;
    ClientKey*     ckey        = nullptr;
    ServerKey*     skey        = nullptr;

    config_builder_default(&cfg_builder);
    config_builder_use_custom_parameters(&cfg_builder,
        SHORTINT_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128);
    // config_builder_build consumes the builder (Rust ownership), do NOT destroy after
    config_builder_build(cfg_builder, &cfg);

    if (generate_keys(cfg, &ckey, &skey) != 0) {
        std::fputs("Key generation failed\n", stderr);
        return 1;
    }
    // generate_keys consumes the Config (Rust ownership), do NOT destroy after
    std::puts("Building GPU server key...");
    auto gpu_key = btc::make_cuda_server_key(ckey);
    set_cuda_server_key(gpu_key.get());

    auto plain_S = btc::bounded_transitive_closure(plain, T);
    auto ground_truth = btc::floyd_warshall(plain);

    std::puts("Encrypting matrix...");
    auto enc_A = btc::TFHEGPUBackend::encrypt(plain, ckey);

    std::printf("Running BTC (N=%zu T=%d) on GPU...\n", N, T);
    btc::TFHEGPUBackend backend;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  FHE computation: %.1f ms (%.3f s)\n", ms, ms / 1000.0);

    std::puts("Decrypting result...");
    auto result = btc::TFHEGPUBackend::decrypt(enc_S, ckey);

    std::puts("  Decrypted reachability matrix:");
    print_matrix(result);

    bool matches_plaintext = (result == plain_S);
    bool matches_ground_truth = (result == ground_truth);
    bool ok = matches_plaintext && matches_ground_truth;
    std::printf("  Matches plaintext validator: %s\n", matches_plaintext ? "YES" : "NO");
    std::printf("  Matches ground truth (Floyd-Warshall): %s\n", matches_ground_truth ? "YES" : "NO");

    if (!matches_plaintext) {
        std::puts("  Expected (plaintext BTC):");
        print_matrix(plain_S);
    }
    if (!matches_ground_truth) {
        std::puts("  Expected (Floyd-Warshall ground truth):");
        print_matrix(ground_truth);
    }

    const std::string csv_path = "data/bench_tfhe_gpu_results.csv";
    bool csv_exists = std::ifstream(csv_path).good();
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists) {
        csv << "timestamp,graph,N,T,fhe_ms,matches_plaintext,matches_ground_truth,params\n";
    }
    csv << timestamp_now() << ',' << graph_name << ',' << N << ',' << T << ','
        << ms << ',' << (matches_plaintext ? "YES" : "NO") << ','
        << (matches_ground_truth ? "YES" : "NO") << ',' << kParamsLabel << '\n';

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_TFHE_GPU=ON to enable GPU benchmark.");
    return 0;
}
#endif
