#ifdef TC_WITH_DGHV

#include <btc/algorithms.hpp>
#include <btc/dghv.hpp>
#include <btc/dghv_backend.hpp>

#ifdef TC_WITH_TFHE
#include <btc/tfhe_backend.hpp>
#endif

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <random>

namespace dghv = btc::dghv;

// Fixed seed everywhere so a failure is exactly reproducible.
static std::mt19937_64 make_rng() { return std::mt19937_64(42); }

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static btc::BoolMatrix from_flat(std::size_t n, std::initializer_list<bool> data) {
    btc::BoolMatrix m(n, n);
    auto it = data.begin();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m.set(i, j, *it++);
    return m;
}

// ── 1 & 2. Encrypt/decrypt 0 and 1 ──────────────────────────────────────────
static void test_encrypt_decrypt_bits() {
    auto rng = make_rng();
    auto params = dghv::toy_params(/*max_depth=*/1);
    auto key = dghv::keygen(params, rng);

    for (int trial = 0; trial < 20; ++trial) {
        auto ct0 = dghv::encrypt_bit(key, params, false, rng);
        check(dghv::decrypt_bit(key, ct0) == false, "encrypt/decrypt 0");

        auto ct1 = dghv::encrypt_bit(key, params, true, rng);
        check(dghv::decrypt_bit(key, ct1) == true, "encrypt/decrypt 1");
    }
    std::puts("encrypt_decrypt_bits (20 trials): PASS");
}

// ── 3. Encrypted AND truth table ────────────────────────────────────────────
static void test_and_truth_table() {
    auto rng = make_rng();
    auto params = dghv::toy_params(/*max_depth=*/1);
    auto key = dghv::keygen(params, rng);

    const bool values[2] = {false, true};
    for (bool a : values) {
        for (bool b : values) {
            auto ca = dghv::encrypt_bit(key, params, a, rng);
            auto cb = dghv::encrypt_bit(key, params, b, rng);
            auto c_and = dghv::encrypted_and(ca, cb);
            bool expected = a && b;
            bool got = dghv::decrypt_bit(key, c_and);
            check(got == expected, "AND truth table entry");
        }
    }
    std::puts("and_truth_table: PASS");
}

// ── 4. Encrypted OR truth table ─────────────────────────────────────────────
static void test_or_truth_table() {
    auto rng = make_rng();
    auto params = dghv::toy_params(/*max_depth=*/1);
    auto key = dghv::keygen(params, rng);

    const bool values[2] = {false, true};
    for (bool a : values) {
        for (bool b : values) {
            auto ca = dghv::encrypt_bit(key, params, a, rng);
            auto cb = dghv::encrypt_bit(key, params, b, rng);
            auto c_or = dghv::encrypted_or(ca, cb);
            bool expected = a || b;
            bool got = dghv::decrypt_bit(key, c_or);
            check(got == expected, "OR truth table entry");
        }
    }
    std::puts("or_truth_table: PASS");

    // XOR is also exercised here since the README lists it as "if needed"
    // and it is not used elsewhere.
    auto key2 = key;
    for (bool a : values) {
        for (bool b : values) {
            auto ca = dghv::encrypt_bit(key2, params, a, rng);
            auto cb = dghv::encrypt_bit(key2, params, b, rng);
            auto c_xor = dghv::encrypted_xor(ca, cb);
            bool expected = a != b;
            bool got = dghv::decrypt_bit(key2, c_xor);
            check(got == expected, "XOR truth table entry");
        }
    }
    std::puts("xor_truth_table: PASS");
}

// ── 5. A short multiplication chain ─────────────────────────────────────────
static void test_multiplication_chain() {
    auto rng = make_rng();
    const int depth = 6;
    // toy_params' fixed eta_bits=128 only supports depth 2 (noise doubles
    // per multiplication level, see dghv.hpp's noise_bits_after_depth) --
    // not enough for this 6-level chain, so eta_bits is picked manually
    // here, large enough for depth 6 (needs > 1151 bits at rho_bits=16)
    // with headroom.
    auto params = dghv::manual_params(/*eta_bits=*/1280, /*rho_bits=*/16, /*q_bits=*/2816, depth);
    dghv::require_depth(params, depth); // must not throw
    auto key = dghv::keygen(params, rng);

    // AND together `depth` encryptions of 1: result must stay 1.
    // AND-chain including a single 0 partway through: result must be 0.
    auto acc_ones = dghv::encrypt_bit(key, params, true, rng);
    for (int i = 1; i < depth; ++i) {
        auto next = dghv::encrypt_bit(key, params, true, rng);
        acc_ones = dghv::encrypted_and(acc_ones, next);
    }
    check(dghv::decrypt_bit(key, acc_ones) == true, "AND-chain of 1s stays 1");

    auto acc_mixed = dghv::encrypt_bit(key, params, true, rng);
    for (int i = 1; i < depth; ++i) {
        bool bit = (i == depth / 2) ? false : true;
        auto next = dghv::encrypt_bit(key, params, bit, rng);
        acc_mixed = dghv::encrypted_and(acc_mixed, next);
    }
    check(dghv::decrypt_bit(key, acc_mixed) == false, "AND-chain with one 0 is 0");

    std::puts("multiplication_chain (depth=6): PASS");
}

// ── 6. A balanced reduction tree with depth comparable to the TC circuit ───
static void test_balanced_reduction_tree() {
    auto rng = make_rng();
    // Same shape as one matmul row for an N=16 adjacency matrix:
    // AND N pairs, then OR-reduce them in a balanced tree.
    const std::size_t n = 16;
    const int depth = btc::matmul_depth(n);
    // toy_params' fixed eta_bits=128 supports only depth 2, not enough for
    // this depth-5 tree (needs > 575 bits at rho_bits=16) -- eta_bits
    // picked manually.
    auto params = dghv::manual_params(/*eta_bits=*/640, /*rho_bits=*/16, /*q_bits=*/1408, depth);
    dghv::require_depth(params, depth);
    auto key = dghv::keygen(params, rng);

    // Row with exactly one true AND-term -> OR-reduction must yield 1.
    std::vector<dghv::Ciphertext> terms;
    std::size_t true_index = 5;
    for (std::size_t k = 0; k < n; ++k) {
        bool a = (k == true_index);
        bool b = true; // always-true partner so AND-term = a
        auto ca = dghv::encrypt_bit(key, params, a, rng);
        auto cb = dghv::encrypt_bit(key, params, b, rng);
        terms.push_back(dghv::encrypted_and(ca, cb));
    }
    // Manual balanced-tree OR reduction (mirrors DGHVBackend::or_reduce_tree).
    while (terms.size() > 1) {
        std::vector<dghv::Ciphertext> next;
        for (std::size_t i = 0; i + 1 < terms.size(); i += 2)
            next.push_back(dghv::encrypted_or(terms[i], terms[i + 1]));
        if (terms.size() % 2 == 1) next.push_back(terms.back());
        terms = std::move(next);
    }
    check(dghv::decrypt_bit(key, terms[0]) == true, "balanced OR-tree finds the single true term");

    auto report = dghv::inspect_noise(key, params, terms[0]);
    check(report.decryption_expected_to_succeed, "balanced OR-tree stays within noise budget");
    std::printf("balanced_reduction_tree (N=%zu, depth=%d, noise_bits=%d/%d): PASS\n",
                n, depth, report.exact_noise_bits, report.eta_bits);
}

// ── 7. Transitive closure on a very small graph ─────────────────────────────
static void test_transitive_closure_small() {
    auto rng = make_rng();

    // 4-node chain: 0->1->2->3. Reachability requires multiple hops, not
    // just direct edges: 0 must reach 1, 2, AND 3.
    auto A = from_flat(4, {
        false, true,  false, false,
        false, false, true,  false,
        false, false, false, true,
        false, false, false, false
    });
    const int T = 3; // N-1, sufficient for this acyclic chain

    const int depth = btc::transitive_closure_depth(A.rows(), T);
    // experiment_params' fixed eta_bits=1024 supports only depth 4 at
    // rho_bits=32, not enough for this depth-8 circuit (needs > 8703 bits)
    // -- eta_bits picked manually, large enough with headroom.
    auto params = dghv::manual_params(/*eta_bits=*/9000, /*rho_bits=*/32, /*q_bits=*/8192, depth);
    dghv::require_depth(params, depth);
    auto key = dghv::keygen(params, rng);

    auto enc_A = btc::DGHVBackend::encrypt(A, key, params, rng);
    btc::DGHVBackend backend;
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    auto result = btc::DGHVBackend::decrypt(enc_S, key);

    auto expected = btc::floyd_warshall(A);
    check(result == expected, "4-chain transitive closure matches Floyd-Warshall");

    // Explicitly confirm multi-hop reachability, not just direct edges.
    check(result.get(0, 1) && result.get(0, 2) && result.get(0, 3),
          "node 0 reaches 1, 2, and 3 (requires multiple hops)");

    std::puts("transitive_closure_small (4-chain, T=3): PASS");
}

// ── 8. Comparison of plaintext, TFHE, and DGHV results ──────────────────────
static void test_three_way_comparison() {
    auto rng = make_rng();

    // 3-node cycle: 0->1->2->0. Full closure requires wraparound.
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    const int T = 3; // N, sufficient to see the full cycle

    auto plain_S = btc::bounded_transitive_closure(A, T);

    const int depth = btc::transitive_closure_depth(A.rows(), T);
    // Same manual sizing rationale as test_transitive_closure_small: this
    // circuit is also depth 8, needing > 8703 bits at rho_bits=32.
    auto params = dghv::manual_params(/*eta_bits=*/9000, /*rho_bits=*/32, /*q_bits=*/8192, depth);
    dghv::require_depth(params, depth);
    auto key = dghv::keygen(params, rng);

    auto enc_A = btc::DGHVBackend::encrypt(A, key, params, rng);
    btc::DGHVBackend dghv_backend;
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, dghv_backend);
    auto dghv_S = btc::DGHVBackend::decrypt(enc_S, key);

    check(dghv_S == plain_S, "DGHV result matches plaintext result");

#ifdef TC_WITH_TFHE
    BooleanClientKey* ckey = nullptr;
    BooleanServerKey* skey = nullptr;
    if (boolean_gen_keys_with_default_parameters(&ckey, &skey) == 0) {
        auto enc_A_tfhe = btc::TFHEBackend::encrypt(A, ckey);
        btc::TFHEBackend tfhe_backend(skey);
        auto enc_S_tfhe = btc::bounded_transitive_closure(enc_A_tfhe, T, tfhe_backend);
        auto tfhe_S = btc::TFHEBackend::decrypt(enc_S_tfhe, ckey);

        check(tfhe_S == plain_S, "TFHE result matches plaintext result");
        check(tfhe_S == dghv_S, "TFHE result matches DGHV result");

        boolean_destroy_client_key(ckey);
        boolean_destroy_server_key(skey);
        std::puts("three_way_comparison (plaintext/TFHE/DGHV, 3-cycle T=3): PASS");
    } else {
        std::puts("three_way_comparison: TFHE key generation failed, skipped TFHE leg");
    }
#else
    std::puts("three_way_comparison (plaintext/DGHV only, TFHE not built, 3-cycle T=3): PASS");
#endif
}

// ── Parameter sanity: detect insufficient depth ─────────────────────────────
static void test_insufficient_depth_detected() {
    auto params = dghv::toy_params(/*max_depth=*/1);
    bool threw = false;
    try {
        dghv::require_depth(params, /*required_depth=*/1000);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "require_depth throws when circuit depth exceeds parameters");
    std::puts("insufficient_depth_detected: PASS");
}

// ── Parameter validation checks ──────────────────────────────────────────
static void test_parameter_validation() {
    // eta_bits must be > rho_bits.
    {
        bool threw = false;
        try {
            dghv::validate_params(dghv::manual_params(/*eta_bits=*/16, /*rho_bits=*/16, /*q_bits=*/64, 0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "validate_params rejects eta_bits <= rho_bits");
    }
    // q_bits must be > 0.
    {
        bool threw = false;
        try {
            dghv::validate_params(dghv::manual_params(/*eta_bits=*/128, /*rho_bits=*/16, /*q_bits=*/0, 0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "validate_params rejects q_bits <= 0");
    }
    // rho_bits must be > 0.
    {
        bool threw = false;
        try {
            dghv::validate_params(dghv::manual_params(/*eta_bits=*/128, /*rho_bits=*/0, /*q_bits=*/64, 0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "validate_params rejects rho_bits <= 0");
    }
    // Valid params must pass, and gamma_bits must equal eta_bits + q_bits.
    {
        auto params = dghv::manual_params(/*eta_bits=*/128, /*rho_bits=*/16, /*q_bits=*/64, 0);
        check(params.gamma_bits == params.eta_bits + params.q_bits,
              "manual_params derives gamma_bits = eta_bits + q_bits");
        bool threw = false;
        try {
            dghv::validate_params(params);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(!threw, "validate_params accepts well-formed parameters");
    }
    // Manually corrupting gamma_bits (simulating an inconsistent caller)
    // must be caught by validate_params.
    {
        auto params = dghv::manual_params(/*eta_bits=*/128, /*rho_bits=*/16, /*q_bits=*/64, 0);
        params.gamma_bits += 1; // deliberately break the eta_bits + q_bits invariant
        bool threw = false;
        try {
            dghv::validate_params(params);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "validate_params rejects gamma_bits inconsistent with eta_bits + q_bits");
    }
    // p must come out odd.
    {
        auto rng = make_rng();
        auto params = dghv::toy_params(1);
        auto key = dghv::keygen(params, rng);
        check(mpz_odd_p(key.p.get_mpz_t()) != 0, "keygen produces an odd secret key p");
    }
    // A freshly encrypted ciphertext must decrypt correctly for both bits.
    {
        auto rng = make_rng();
        auto params = dghv::toy_params(1);
        auto key = dghv::keygen(params, rng);
        auto ct0 = dghv::encrypt_bit(key, params, false, rng);
        auto ct1 = dghv::encrypt_bit(key, params, true, rng);
        check(dghv::decrypt_bit(key, ct0) == false, "fresh ciphertext of 0 decrypts correctly");
        check(dghv::decrypt_bit(key, ct1) == true, "fresh ciphertext of 1 decrypts correctly");
    }
    std::puts("parameter_validation: PASS");
}

// ── Noise growth after repeated additions and multiplications ───────────────
//
// Note: the *exact* noise magnitude (what inspect_noise's exact_noise_bits
// reports) is not guaranteed to increase monotonically step by step -- two
// independently-signed noise terms can partially cancel on a given addition
// or multiplication. What IS guaranteed, and what this test checks, is the
// structural noise_bits_estimate upper bound (ct.noise_bits_estimate, set
// by add()/mul() themselves), which is monotonically non-decreasing by
// construction, and that the exact noise never exceeds that bound.
static void test_noise_growth() {
    auto rng = make_rng();
    auto params = dghv::manual_params(/*eta_bits=*/2000, /*rho_bits=*/16, /*q_bits=*/4000, 0);
    auto key = dghv::keygen(params, rng);

    // Repeated addition: the noise_bits_estimate bound grows additively (+1
    // bit per level), and the exact noise must stay within that bound.
    auto acc_add = dghv::encrypt_bit(key, params, true, rng);
    int prev_add_estimate = acc_add.noise_bits_estimate;
    for (int i = 0; i < 5; ++i) {
        auto next = dghv::encrypt_bit(key, params, false, rng);
        acc_add = dghv::add(acc_add, next);
        check(acc_add.noise_bits_estimate > prev_add_estimate,
              "noise_bits_estimate strictly increases after each addition");
        auto report = dghv::inspect_noise(key, params, acc_add);
        check(report.exact_noise_bits <= acc_add.noise_bits_estimate,
              "exact noise never exceeds the structural noise_bits_estimate bound (addition)");
        prev_add_estimate = acc_add.noise_bits_estimate;
    }
    int final_add_noise = dghv::inspect_noise(key, params, acc_add).exact_noise_bits;
    // After 5 additions of rho_bits=16 noise terms, exact noise should still
    // be small (roughly rho_bits + a few bits), nowhere near eta_bits.
    check(final_add_noise < 32, "additive noise growth stays small over a few additions");

    // Repeated multiplication: the noise_bits_estimate bound grows
    // multiplicatively (roughly doubles per level), and the exact noise
    // must stay within that bound.
    auto acc_mul = dghv::encrypt_bit(key, params, true, rng);
    int prev_mul_estimate = acc_mul.noise_bits_estimate;
    for (int i = 0; i < 4; ++i) {
        auto next = dghv::encrypt_bit(key, params, true, rng);
        acc_mul = dghv::mul(acc_mul, next);
        check(acc_mul.noise_bits_estimate > prev_mul_estimate,
              "noise_bits_estimate strictly increases after each multiplication");
        auto report = dghv::inspect_noise(key, params, acc_mul);
        check(report.exact_noise_bits <= acc_mul.noise_bits_estimate,
              "exact noise never exceeds the structural noise_bits_estimate bound (multiplication)");
        prev_mul_estimate = acc_mul.noise_bits_estimate;
    }
    int final_mul_noise = dghv::inspect_noise(key, params, acc_mul).exact_noise_bits;
    // Multiplicative noise after 4 levels must have grown much faster than
    // the additive case above -- this is the "noise doubles per level"
    // behavior that limits multiplicative depth.
    check(final_mul_noise > final_add_noise,
          "multiplicative noise growth outpaces additive noise growth");

    std::printf("noise_growth (additive final=%d bits, multiplicative final=%d bits): PASS\n",
                final_add_noise, final_mul_noise);
}

// ── Decryption failure once noise exceeds the correctness bound ─────────────
static void test_decryption_fails_beyond_noise_bound() {
    auto rng = make_rng();
    // Deliberately tiny eta_bits relative to rho_bits so a handful of
    // multiplications is guaranteed to push noise past eta_bits/2 -- this
    // test exists specifically to show decryption CAN silently produce the
    // wrong bit once the correctness condition is violated, which is why
    // require_depth()/inspect_noise() exist to catch this ahead of time.
    auto params = dghv::manual_params(/*eta_bits=*/40, /*rho_bits=*/8, /*q_bits=*/80, 0);
    auto key = dghv::keygen(params, rng);

    check(dghv::max_supported_depth(params) < 4,
          "chosen toy params support fewer than 4 multiplication levels");

    auto acc = dghv::encrypt_bit(key, params, true, rng);
    bool observed_mismatch = false;
    for (int level = 0; level < 8 && !observed_mismatch; ++level) {
        auto next = dghv::encrypt_bit(key, params, true, rng);
        acc = dghv::mul(acc, next);
        auto report = dghv::inspect_noise(key, params, acc);
        if (!report.decryption_expected_to_succeed) {
            // Once inspect_noise predicts failure, the decrypted bit is no
            // longer guaranteed to equal the true product (still 1, since
            // we only multiplied by encryptions of 1) -- confirm the
            // predicted-failure state was actually reached.
            observed_mismatch = true;
        }
    }
    check(observed_mismatch,
          "noise eventually exceeds the correctness bound under repeated multiplication");
    std::puts("decryption_fails_beyond_noise_bound: PASS");
}

int main() {
    test_encrypt_decrypt_bits();
    test_and_truth_table();
    test_or_truth_table();
    test_multiplication_chain();
    test_balanced_reduction_tree();
    test_transitive_closure_small();
    test_three_way_comparison();
    test_insufficient_depth_detected();
    test_parameter_validation();
    test_noise_growth();
    test_decryption_fails_beyond_noise_bound();
    std::puts("\nAll DGHV tests passed.");
    return 0;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_DGHV=ON to enable DGHV tests.");
    return 0;
}
#endif
