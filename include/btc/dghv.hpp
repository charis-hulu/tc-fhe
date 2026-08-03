#pragma once

#ifdef TC_WITH_DGHV

// ── DGHV: a leveled, secret-key, integer-based FHE scheme ───────────────────
//
// This is a SIMPLIFIED, SECRET-KEY (symmetric) educational variant of DGHV
// (van Dijk, Gentry, Halevi, Vaikuntanathan, "Fully Homomorphic Encryption
// over the Integers", EUROCRYPT 2010). It is NOT the full public-key DGHV
// construction (there is no public key made of many encryptions of zero, no
// "somewhat homomorphic" public key sampling, and no bootstrapping). See the
// "What this is / is not" section in docs/dghv.md for the precise scope.
//
// ── Ciphertext form ───────────────────────────────────────────────────────
//
//   For secret odd integer p (the secret key) and plaintext bit m in {0,1}:
//
//       c = p*q + 2*r + m                                             (1)
//
//   where q is a large random integer and r is a small random "noise" term
//   with |2r| << p. This is the same ciphertext shape as full DGHV; the
//   simplification is only in how the *encryptor* picks q and r (directly,
//   knowing p) rather than via a public key of encryptions of zero.
//
// ── Decryption rule ───────────────────────────────────────────────────────
//
//       m = [c]_p mod 2                                               (2)
//
//   where [c]_p denotes c reduced into the CENTERED range (-p/2, p/2]
//   (as opposed to the usual [0, p) range). This recovers (2r + m) exactly
//   as long as |2r + m| < p/2 -- the correctness condition tracked by
//   inspect_noise() below.
//
// ── Homomorphic operations ───────────────────────────────────────────────
//
//   Addition:       c_add = c1 + c2            decrypts to m1 XOR-like sum
//                    (noise term becomes r1+r2, doubles in magnitude)
//   Multiplication:  c_mul = c1 * c2            decrypts to m1 * m2
//                    (noise term becomes ~ p*(noise), grows multiplicatively)
//
//   These are the two operations the whole scheme is built from; every
//   encrypted bit-logic gate below (AND/OR/XOR) is defined in terms of them.
//
// ── Why no bootstrapping ──────────────────────────────────────────────────
//
//   Full DGHV supports unbounded computation via bootstrapping: homomorphically
//   evaluating the decryption circuit to "refresh" a noisy ciphertext into a
//   low-noise one. That requires a squashed decryption circuit and a large
//   public key, which would dominate this codebase and obscure the core
//   scheme. Instead we choose parameters up front for a bounded multiplicative
//   depth D (computed from the transitive-closure circuit in dghv_backend.hpp
//   / algorithms.hpp), and prove correctness noise-theoretically: as long as
//   the deepest ciphertext's noise stays below p/2 at decryption time, the
//   result is correct. This is exactly the "leveled" FHE approach requested.
//
// ─────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <gmpxx.h>
#include <random>
#include <stdexcept>
#include <string>

namespace btc {
namespace dghv {

// ── Parameters ────────────────────────────────────────────────────────────
//
// Standard DGHV notation (van Dijk et al., EUROCRYPT 2010, Section 2):
//   eta   ("eta")   -- bit length of the secret key p.
//   rho   ("rho")   -- bit length (magnitude bound) of the fresh noise r.
//   gamma ("gamma") -- bit length of a full ciphertext / public sample
//                       x = p*q + 2*r, i.e. gamma ~ eta + (bit length of q).
//
// gamma is a property of the *ciphertext size*, not of q directly. This
// implementation exposes eta_bits, rho_bits, and q_bits (the bit length of
// the random multiplier q, sometimes written "gamma - eta" in the paper) as
// the independent knobs for manual experimentation, and DERIVES gamma_bits
// from them:
//
//       gamma_bits = eta_bits + q_bits                                (0)
//
// gamma_bits is exposed on the struct (and reported by presets/CLI tools)
// purely for reference against the paper's notation -- it is never an
// independently settable field, so there is no way for gamma_bits and
// q_bits to disagree with each other.
struct Params {
    // Bit-length of the secret key p (an odd integer). This is the
    // "ciphertext modulus" scale; noise must stay far below p/2.
    int eta_bits = 0;

    // Bit-length of the initial noise bound: freshly encrypted ciphertexts
    // use a signed r with |r| < 2^rho_bits.
    int rho_bits = 0;

    // Bit-length of the random multiplier q in c = p*q + 2r + m. This is
    // what the DGHV paper calls "gamma - eta": q is sampled independently
    // of p, so a ciphertext c = p*q + ... has bit length approximately
    // eta_bits + q_bits (see gamma_bits below).
    int q_bits = 0;

    // Derived, NOT independently settable: gamma_bits = eta_bits + q_bits,
    // the paper's gamma (bit length of a full ciphertext). Kept on the
    // struct only as a convenience for printing/reporting; every function
    // in this file samples q using q_bits directly, never gamma_bits.
    int gamma_bits = 0;

    // Maximum multiplicative depth this parameter set is designed to
    // support. Noise roughly squares (doubles in bit-length) per
    // multiplication level, so eta_bits must be chosen so that
    // depth-many squarings of the initial noise still stay below
    // eta_bits - 1 bits (see noise_bits_after_depth / max_supported_depth
    // below).
    int max_multiplicative_depth = 0;

    // Human-readable label: "toy", "experiment", "secure-target", or
    // "manual". Purely descriptive -- never used for arithmetic -- so that
    // no parameter set can silently be mistaken for a security claim.
    std::string tier = "toy";

    // A short note on what this tier means / does not guarantee.
    std::string note;
};

// Builds a Params from the caller-facing knobs (eta_bits, rho_bits, q_bits)
// and derives gamma_bits = eta_bits + q_bits (equation 0 above). Every
// preset function below funnels through this constructor so gamma_bits can
// never be set independently of q_bits.
inline Params make_params(int eta_bits, int rho_bits, int q_bits,
                           int max_depth, std::string tier, std::string note) {
    Params p;
    p.eta_bits = eta_bits;
    p.rho_bits = rho_bits;
    p.q_bits = q_bits;
    p.gamma_bits = eta_bits + q_bits;
    p.max_multiplicative_depth = max_depth;
    p.tier = std::move(tier);
    p.note = std::move(note);
    return p;
}

// ── Parameter validation ────────────────────────────────────────────────
//
// Checks the structural preconditions the scheme relies on. Does NOT check
// multiplicative-depth sufficiency -- see require_depth() further below,
// which is a separate, circuit-dependent check.
inline void validate_params(const Params& params) {
    if (params.eta_bits <= params.rho_bits)
        throw std::invalid_argument(
            "DGHV params: eta_bits (" + std::to_string(params.eta_bits) +
            ") must be > rho_bits (" + std::to_string(params.rho_bits) +
            ") -- the secret key must be larger than the noise it is meant to hide.");
    if (params.q_bits <= 0)
        throw std::invalid_argument(
            "DGHV params: q_bits must be > 0, got " + std::to_string(params.q_bits));
    if (params.rho_bits <= 0)
        throw std::invalid_argument(
            "DGHV params: rho_bits must be > 0, got " + std::to_string(params.rho_bits));
    if (params.gamma_bits != params.eta_bits + params.q_bits)
        throw std::invalid_argument(
            "DGHV params: gamma_bits (" + std::to_string(params.gamma_bits) +
            ") is inconsistent with eta_bits + q_bits (" +
            std::to_string(params.eta_bits + params.q_bits) +
            ") -- gamma_bits must always equal eta_bits + q_bits.");
}

// ── Depth-to-noise relationship (the correctness condition) ────────────────
//
// A fresh ciphertext has noise magnitude |2r| < 2^(rho_bits+1).
// Addition of two ciphertexts at the same level: noise adds, so it at most
// doubles (+1 bit) per addition -- negligible next to multiplication.
// Multiplication of two ciphertexts with noise bounds N1, N2 produces a
// ciphertext whose noise is bounded by roughly N1*N2 (the noise terms
// multiply along with the plaintext/quotient cross terms); we track this
// as a bit-length bound that adds N1_bits + N2_bits + O(1) per level.
//
// After `depth` sequential multiplications starting from rho_bits, the
// noise bit-length grows approximately as:
//
//     noise_bits(depth) = rho_bits * 2^depth + O(depth)                (3)
//
// (each level roughly squares the noise magnitude, i.e. doubles its bit
// length, since both multiplicands carry independently-grown noise). This
// exponential blowup is the fundamental reason a *leveled* (non-bootstrapped)
// scheme like this one needs eta_bits to grow with the circuit's depth --
// see docs/dghv.md section 10 for what that costs in practice.
//
// Decryption is correct as long as noise_bits(depth) < eta_bits - 1 (one
// bit of slack for the mod-2 rounding in the decryption formula (2) above).
inline int noise_bits_after_depth(const Params& params, int depth) {
    if (depth < 0)
        throw std::invalid_argument("noise_bits_after_depth: depth must be >= 0");
    // +1 bit per level for the additions (OR's "+a+b") layered on top of
    // multiplications; this is a deliberately loose (safe) over-count.
    long long bits = params.rho_bits;
    for (int level = 0; level < depth; ++level)
        bits = 2 * bits + 2;
    return static_cast<int>(bits);
}

// Largest depth this parameter set can support while keeping
// noise_bits_after_depth(depth) < eta_bits - 1 (the correctness margin).
inline int max_supported_depth(const Params& params) {
    int depth = 0;
    while (noise_bits_after_depth(params, depth + 1) < params.eta_bits - 1)
        ++depth;
    return depth;
}

// Throws if `params` cannot support `required_depth` multiplications while
// remaining decryptable. Call this before running any circuit so parameter
// mistakes are caught immediately rather than surfacing as silent wrong
// decryptions.
inline void require_depth(const Params& params, int required_depth) {
    int supported = max_supported_depth(params);
    if (required_depth > supported) {
        throw std::runtime_error(
            "DGHV parameters insufficient: circuit needs multiplicative depth " +
            std::to_string(required_depth) + " but params '" + params.tier +
            "' (eta_bits=" + std::to_string(params.eta_bits) + ") only support depth " +
            std::to_string(supported) +
            ". Increase eta_bits or reduce circuit depth.");
    }
}

// Toy parameters: fast enough to encrypt/decrypt/AND/OR thousands of times
// in a unit test, but NOT secure. eta_bits=128 is far below any real
// security target; chosen purely so the tests run in milliseconds.
// `max_depth` is stored on the struct for reference only -- it does NOT
// change eta_bits here. Call require_depth() with your circuit's actual
// depth to check whether this fixed eta_bits is actually big enough; if
// not, either pass a larger --eta-bits on the command line or use
// experiment_params / secure_target_params.
inline Params toy_params(int max_depth) {
    return make_params(/*eta_bits=*/128, /*rho_bits=*/16, /*q_bits=*/384, max_depth,
        "toy",
        "INSECURE. For fast unit tests only. eta_bits=128 gives no real "
        "cryptographic security; a real attacker factors this instantly. "
        "Do not use for anything but correctness testing.");
}

// "Experiment" parameters: large enough to be a meaningful correctness/noise
// experiment (bigger integers, more realistic noise growth curve) but still
// far short of a real security target. Good for benchmarking arithmetic cost
// without waiting on truly cryptographic-size GMP integers. eta_bits is
// fixed, not derived from `max_depth` -- check with require_depth() whether
// it is enough for your circuit.
inline Params experiment_params(int max_depth) {
    return make_params(/*eta_bits=*/1024, /*rho_bits=*/32, /*q_bits=*/3072, max_depth,
        "experiment",
        "NOT a security claim. Larger than toy_params so noise growth and "
        "integer sizes are representative, but eta_bits=1024 is still well "
        "short of the >=2^lambda hardness margin the DGHV approximate-GCD "
        "assumption needs at lambda=128 security (original paper suggests "
        "eta on the order of the security parameter squared or worse, i.e. "
        "eta_bits in the tens of thousands for lambda=128).");
}

// Parameters intended to approximate a real security target (lambda=128-bit
// security against the sparse/approximate-GCD attacks analyzed in the DGHV
// paper and its successors). This is a documented ESTIMATE, not a vetted
// security proof -- picking provably-secure DGHV parameters is a research
// question on its own, and this project does not attempt to resolve it.
// Included so the gap between "toy" and "real" is explicit and visible,
// rather than silently ignored.
inline Params secure_target_params(int max_depth) {
    const int eta_bits = 2 * 128 * 128; // O(lambda^2), the paper's "somewhat homomorphic" regime
    const int q_bits = eta_bits * 9;    // gamma_bits = eta_bits + q_bits ~= 10*eta_bits >> eta_bits, to hide p among q's randomness
    return make_params(eta_bits, /*rho_bits=*/128, q_bits, max_depth,
        "secure-target",
        "ESTIMATED parameters aiming at ~128-bit security following the "
        "scaling rules in the original DGHV paper (eta = O(lambda^2), "
        "rho = lambda). This has NOT been independently verified against "
        "modern lattice/approximate-GCD cryptanalysis and should not be "
        "treated as a vetted secure parameter set. Expect very slow GMP "
        "arithmetic at this size (eta_bits ~ 32768).");
}

// Fully manual parameters: eta_bits, rho_bits, and q_bits supplied directly
// by the caller (e.g. from command-line arguments); gamma_bits is always
// derived as eta_bits + q_bits (equation 0), never taken as a separate
// argument, so it cannot disagree with q_bits. tier/note are informational
// strings only.
inline Params manual_params(int eta_bits, int rho_bits, int q_bits, int max_depth) {
    return make_params(eta_bits, rho_bits, q_bits, max_depth,
        "manual",
        "User-supplied parameters. No built-in security or correctness "
        "claim -- verify with require_depth() and, if security matters, "
        "compare eta_bits/rho_bits/q_bits against a vetted DGHV parameter "
        "selection before trusting this for anything beyond correctness "
        "testing.");
}

// ── Centered modulo ─────────────────────────────────────────────────────
//
// Returns x reduced into the centered range (-m/2, m/2], i.e. [x]_m in the
// notation of equation (2). GMP's mpz_class operator% follows C++ truncating
// division semantics (result takes the sign of the dividend, so it can be
// negative), which is NOT what decryption needs -- this function makes the
// centering explicit rather than relying on ad hoc sign-fixup at each call
// site.
inline mpz_class centered_mod(const mpz_class& x, const mpz_class& m) {
    mpz_class r = x % m;
    if (r < 0)
        r += m; // normalize into [0, m)
    if (2 * r > m)
        r -= m; // recentre into (-m/2, m/2]
    return r;
}

// ── Secret key ────────────────────────────────────────────────────────────

struct SecretKey {
    mpz_class p; // large odd integer, kept private
};

// Generates p as a random odd integer with exactly params.eta_bits bits
// (top bit set, so the bit length is exact, not just an upper bound).
inline SecretKey keygen(const Params& params, std::mt19937_64& rng) {
    validate_params(params);

    gmp_randstate_t state;
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, rng());

    mpz_class candidate;
    mpz_urandomb(candidate.get_mpz_t(), state, params.eta_bits);
    mpz_setbit(candidate.get_mpz_t(), params.eta_bits - 1); // fix top bit: exactly eta_bits long
    mpz_setbit(candidate.get_mpz_t(), 0);                   // force odd, as required by DGHV

    gmp_randclear(state);

    SecretKey key;
    key.p = candidate;
    return key;
}

// ── Ciphertext ────────────────────────────────────────────────────────────
//
// A ciphertext is just the integer c from equation (1). We additionally
// carry a *noise-bit estimate* alongside it purely for debugging/tests
// (see inspect_noise below); it is not used by encryption/decryption/
// homomorphic evaluation themselves, so it can never influence correctness
// -- only observability.
struct Ciphertext {
    mpz_class c;
    int noise_bits_estimate = 0; // debugging aid only, see inspect_noise
};

// ── Encryption ────────────────────────────────────────────────────────────
//
//   c = p*q + 2*r + m                                                  (1)
//
// q is drawn uniformly from [0, 2^q_bits) -- this is the paper's "gamma -
// eta"-bit random multiplier, sampled directly by its own bit length rather
// than via the equivalent floor(2^gamma_bits / p) bound. r is drawn from
// the signed range (-2^rho_bits, 2^rho_bits]. This is the "secret-key"
// simplification: real DGHV encryption instead adds together a public-key
// subset sum of encryptions of zero so that q's distribution hides p; here
// we sample p*q directly because we already hold p.
inline Ciphertext encrypt_bit(const SecretKey& key, const Params& params,
                               bool m, std::mt19937_64& rng) {
    gmp_randstate_t state;
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, rng());

    mpz_class q;
    mpz_urandomb(q.get_mpz_t(), state, params.q_bits);

    // r uniform in [0, 2^(rho_bits+1)) then recentred to (-2^rho_bits, 2^rho_bits]
    mpz_class r;
    mpz_urandomb(r.get_mpz_t(), state, params.rho_bits + 1);
    r -= (mpz_class(1) << params.rho_bits);

    gmp_randclear(state);

    Ciphertext ct;
    ct.c = key.p * q + 2 * r + static_cast<int>(m);
    ct.noise_bits_estimate = params.rho_bits;
    return ct;
}

// ── Decryption ────────────────────────────────────────────────────────────
//
//   m = [c]_p mod 2                                                    (2)
//
// [c]_p (centered_mod(c, p)) equals 2*r+m exactly when |2*r+m| < p/2 (the
// correctness condition). Reducing that further mod 2 strips off the (even)
// 2*r term and leaves m.
inline bool decrypt_bit(const SecretKey& key, const Ciphertext& ct) {
    mpz_class centered = centered_mod(ct.c, key.p);

    mpz_class m = centered % 2;
    if (m < 0)
        m += 2; // C++ % can be negative for a negative centered value; bit is always in {0,1}
    return m != 0;
}

// ── Homomorphic addition ─────────────────────────────────────────────────
//
//   Enc(m1) + Enc(m2) = p*(q1+q2) + 2*(r1+r2) + (m1+m2)
//
// Decrypting gives (m1+m2) mod 2, i.e. plaintext XOR -- matching how
// addition behaves in the underlying ring Z_2 once you reduce mod 2.
inline Ciphertext add(const Ciphertext& a, const Ciphertext& b) {
    Ciphertext out;
    out.c = a.c + b.c;
    out.noise_bits_estimate =
        std::max(a.noise_bits_estimate, b.noise_bits_estimate) + 1;
    return out;
}

// Homomorphic negation is not needed by AND/OR, but subtraction is used to
// build OR below without re-deriving it inline each time.
inline Ciphertext sub(const Ciphertext& a, const Ciphertext& b) {
    Ciphertext out;
    out.c = a.c - b.c;
    out.noise_bits_estimate =
        std::max(a.noise_bits_estimate, b.noise_bits_estimate) + 1;
    return out;
}

// ── Homomorphic multiplication ───────────────────────────────────────────
//
//   Enc(m1) * Enc(m2) = p*q1 * p*q2 + cross terms
//                      = p*(p*q1*q2 + 2*r1*q2 + 2*r2*q1) + 2*(2*r1*r2) + m1*m2
//
// Decrypting gives (m1*m2) mod 2, i.e. plaintext AND -- this is the
// multiplicative structure that makes DGHV a *ring* homomorphism over
// Z_2, and exactly why AND(a,b) = a*b (see below).
//
// This is the operation that consumes multiplicative depth: the noise
// term (2*r1*r2, folded into the new "2r" via the q-multiple of p above)
// is roughly the *product* of the input noises, not their sum.
inline Ciphertext mul(const Ciphertext& a, const Ciphertext& b) {
    Ciphertext out;
    out.c = a.c * b.c;
    out.noise_bits_estimate = a.noise_bits_estimate + b.noise_bits_estimate + 2;
    return out;
}

// ── Encrypted bit logic ───────────────────────────────────────────────────
//
// These are the operations dghv_backend.hpp actually calls. Each is defined
// directly in terms of add/sub/mul above so the ciphertext arithmetic is
// never accidentally short-circuited by ordinary C++ boolean operators.

// AND(a,b) = a*b                    (one multiplication; costs one depth level)
inline Ciphertext encrypted_and(const Ciphertext& a, const Ciphertext& b) {
    return mul(a, b);
}

// OR(a,b) = a + b - a*b              (one multiplication; costs one depth level)
inline Ciphertext encrypted_or(const Ciphertext& a, const Ciphertext& b) {
    Ciphertext sum = add(a, b);
    Ciphertext prod = mul(a, b);
    return sub(sum, prod);
}

// XOR(a,b) = a + b - 2*a*b           (one multiplication; costs one depth level)
// Not used by the AND/OR-based transitive-closure circuit, but included
// since the README lists it as a component to implement clearly.
inline Ciphertext encrypted_xor(const Ciphertext& a, const Ciphertext& b) {
    Ciphertext sum = add(a, b);
    Ciphertext prod = mul(a, b);
    Ciphertext two_prod = add(prod, prod);
    return sub(sum, two_prod);
}

// ── Noise / correctness inspection (debug-only path) ────────────────────
//
// Separate from the normal encrypt/eval/decrypt path above: nothing here
// is called during a normal homomorphic computation. Use it in tests or
// interactively to check how much correctness margin remains.
struct NoiseReport {
    int noise_bits_estimate;  // ct.noise_bits_estimate (structural estimate)
    int exact_noise_bits;     // exact bit-length of [c]_p, needs secret key
    int eta_bits;
    bool decryption_expected_to_succeed;
};

// Requires the secret key, so this is strictly a debugging/testing utility
// -- a real server evaluating homomorphically never has access to it.
inline NoiseReport inspect_noise(const SecretKey& key, const Params& params,
                                  const Ciphertext& ct) {
    mpz_class centered = centered_mod(ct.c, key.p);
    mpz_class abs_centered = centered < 0 ? -centered : centered;

    NoiseReport report;
    report.noise_bits_estimate = ct.noise_bits_estimate;
    report.exact_noise_bits = abs_centered == 0
                                   ? 0
                                   : static_cast<int>(mpz_sizeinbase(abs_centered.get_mpz_t(), 2));
    report.eta_bits = params.eta_bits;
    // Correctness needs |2r+m| < p/2, i.e. exact_noise_bits < eta_bits - 1.
    report.decryption_expected_to_succeed = report.exact_noise_bits < params.eta_bits - 1;
    return report;
}

} // namespace dghv
} // namespace btc

#endif // TC_WITH_DGHV
