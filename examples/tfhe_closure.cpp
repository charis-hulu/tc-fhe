#ifdef TC_WITH_TFHE

#include <btc/algorithms.hpp>
#include <btc/tfhe_backend.hpp>

#include <cstdio>
#include <cstdlib>

// Same 8-node graph as the plaintext benchmark
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
    const int T = 5;

    // Build plaintext matrix
    btc::BoolMatrix plain(N, N, false);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            plain.set(i, j, ADJ[i][j] == 1);

    // Generate keys
    std::puts("Generating keys...");
    BooleanClientKey* ckey = nullptr;
    BooleanServerKey* skey = nullptr;
    if (boolean_gen_keys_with_default_parameters(&ckey, &skey) != 0) {
        std::fputs("Key generation failed\n", stderr);
        return 1;
    }

    // Encrypt
    std::puts("Encrypting...");
    auto enc_A = btc::TFHEBackend::encrypt(plain, ckey);

    // Run bounded transitive closure under encryption
    std::printf("Running BTC (N=%zu, T=%d) under FHE...\n", N, T);
    btc::TFHEBackend backend(skey);
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);

    // Decrypt and print
    std::puts("Decrypting...");
    auto result = btc::TFHEBackend::decrypt(enc_S, ckey);

    std::puts("\nReachability matrix (decrypted):");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::printf("%d ", result.get(i, j) ? 1 : 0);
        std::puts("");
    }

    // Verify against plaintext
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
