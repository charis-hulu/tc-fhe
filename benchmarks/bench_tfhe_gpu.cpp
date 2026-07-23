#ifdef TC_WITH_TFHE_GPU

#include <btc/algorithms.hpp>
#include <btc/tfhe_gpu_backend.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static const int ADJ[8][8] = {
    {0, 1, 0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0},
};

int main() {
    const std::size_t N = 8;
    const int T = 2;

    btc::BoolMatrix plain(N, N, false);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            plain.set(i, j, ADJ[i][j] == 1);

    std::puts("Generating keys...");
    ConfigBuilder* cfg_builder = nullptr;
    Config*        cfg         = nullptr;
    ClientKey*     ckey        = nullptr;
    ServerKey*     skey        = nullptr;

    // GPU decompressor requires MessageModulus 2 or 4 (1 or 2 bits per block).
    // Use MESSAGE_2_CARRY_2 (MessageModulus=4) which is GPU-compatible.
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

    std::puts("Encrypting matrix...");
    auto enc_A = btc::TFHEGPUBackend::encrypt(plain, ckey);

    std::printf("Running BTC (N=%zu T=%d) on GPU...\n", N, T);
    auto t0 = std::chrono::high_resolution_clock::now();
    btc::TFHEGPUBackend backend;
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("GPU FHE computation: %.1f ms\n\n", ms);

    std::puts("Decrypting result...");
    auto result = btc::TFHEGPUBackend::decrypt(enc_S, ckey);

    std::puts("Reachability matrix (decrypted):");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::printf("%d ", result.get(i, j) ? 1 : 0);
        std::puts("");
    }

    auto plain_S = btc::bounded_transitive_closure(plain, T);
    bool ok = (result == plain_S);
    std::printf("\nMatches plaintext: %s\n", ok ? "YES" : "NO");

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_TFHE_GPU=ON to enable GPU benchmark.");
    return 0;
}
#endif
