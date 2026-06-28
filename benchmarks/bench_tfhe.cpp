#ifdef TC_WITH_TFHE

#include <btc/algorithms.hpp>
#include <btc/tfhe_backend.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

// 3-node chain: 0->1->2
// Expected BTC(T=2): [[0,1,1],[0,0,1],[0,0,0]]
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
    BooleanClientKey* ckey = nullptr;
    BooleanServerKey* skey = nullptr;
    if (boolean_gen_keys_with_default_parameters(&ckey, &skey) != 0) {
        std::fputs("Key generation failed\n", stderr);
        return 1;
    }

    std::puts("Encrypting matrix...");
    auto enc_A = btc::TFHEBackend::encrypt(plain, ckey);

    std::printf("Running BTC (N=%zu T=%d) under FHE...\n", N, T);
    auto t0 = std::chrono::high_resolution_clock::now();
    btc::TFHEBackend backend(skey);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("FHE computation: %.1f ms\n\n", ms);

    std::puts("Decrypting result...");
    auto result = btc::TFHEBackend::decrypt(enc_S, ckey);

    std::puts("Decrypted reachability matrix:");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::printf("%d ", result.get(i, j) ? 1 : 0);
        std::puts("");
    }

    auto plain_S = btc::bounded_transitive_closure(plain, T);
    bool ok = (result == plain_S);
    std::printf("\nMatches plaintext: %s\n", ok ? "YES" : "NO");

    boolean_destroy_client_key(ckey);
    boolean_destroy_server_key(skey);
    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_TFHE=ON to enable TFHE benchmark.");
    return 0;
}
#endif
