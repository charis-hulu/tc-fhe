#pragma once

#ifdef TC_WITH_BGV

// ── BGV: leveled FHE over encrypted integers, via OpenFHE's BGVrns scheme ───
//
// This wraps OpenFHE's BGV-RNS implementation (Brakerski-Gentry-Vaikuntanathan,
// "Fully Homomorphic Encryption without Bootstrapping", ITCS 2012) for exact
// integer arithmetic mod a plaintext prime. Unlike the hand-rolled DGHV scheme
// in dghv.hpp, the cryptographic primitives (key generation, encryption,
// homomorphic add/multiply, RNS modulus chain, relinearization) are all
// implemented by OpenFHE; this file only picks parameters and exposes the
// bit-oriented AND/OR/XOR gate API the rest of the codebase (dghv_backend.hpp
// style) expects.
//
// ── Boolean representation ───────────────────────────────────────────────
//
//   A Boolean value is a BGV plaintext integer in {0, 1} mod the plaintext
//   modulus t. Gates are the same {0,1}-ring identities used by the DGHV
//   backend (see dghv.hpp):
//
//       AND(a,b) = a*b
//       OR(a,b)  = a + b - a*b
//       XOR(a,b) = a + b - 2*a*b
//
//   As long as t > 2 (true for the default t = 65537), these never wrap
//   around, so the decrypted result of any AND/OR/XOR of {0,1} inputs stays
//   exactly {0,1}. This is exact modular arithmetic, not CKKS-style
//   approximate arithmetic -- BGV decryption recovers the plaintext exactly.
//
// ── Leveled, not bootstrapped ────────────────────────────────────────────
//
//   Like the DGHV backend, this is a LEVELED scheme: the multiplicative
//   depth of the whole circuit is estimated up front (see
//   EstimateBGVDepth in bgv_backend.hpp) and baked into the crypto context's
//   SetMultiplicativeDepth before any keys are generated. There is no
//   bootstrapping call anywhere in this file; once a ciphertext's level
//   budget is exhausted, further multiplications produce garbage (OpenFHE
//   may throw, or decryption may simply be wrong) -- exactly the same
//   contract as DGHV's noise budget, just tracked by OpenFHE's RNS modulus
//   chain instead of by hand.
//
// ─────────────────────────────────────────────────────────────────────────

#include "openfhe.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace btc {
namespace bgv {

using Ciphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using PublicKeyT = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

// ── Parameters ────────────────────────────────────────────────────────────
//
// Deliberately small surface: only the knobs that matter for this
// experiment are exposed. Everything else (RNS tower sizing, NTT-friendly
// prime selection, etc.) is left to OpenFHE's automatic parameter
// selection, exactly as the task calls for.
struct Params {
    // Plaintext modulus t. Must be > 2 so {0,1} AND/OR/XOR results never
    // wrap around (see file header). 65537 is a batching-friendly NTT prime
    // commonly used in OpenFHE BGV examples.
    uint64_t plaintext_modulus = 65537;

    // Multiplicative depth the crypto context is built to support. This is
    // a hard ceiling: circuits deeper than this will fail or silently
    // corrupt results. Callers should set this from
    // btc::EstimateBGVDepth(N, T) (bgv_backend.hpp), optionally overridden
    // manually for experiments.
    uint32_t multiplicative_depth = 1;

    // Target security level. HEStd_128_classic is OpenFHE's standard
    // 128-bit-security parameter table lookup.
    lbcrypto::SecurityLevel security_level = lbcrypto::HEStd_128_classic;

    // Ring dimension override. 0 means "let OpenFHE choose automatically"
    // based on plaintext_modulus/multiplicative_depth/security_level.
    uint32_t ring_dimension = 0;

    // Batch size for SIMD slot packing. 0 means "let OpenFHE choose
    // automatically". This backend does not currently pack multiple bits
    // per ciphertext (see the comment in bgv_backend.hpp), so batch size
    // only affects how many plaintext slots exist -- it does not change
    // correctness of the one-bit-per-ciphertext scheme used here.
    uint32_t batch_size = 0;
};

// ── Crypto context / key material ────────────────────────────────────────
//
// Bundles the pieces every homomorphic call needs: the context (public
// evaluation parameters + eval keys) plus the keypair. The secret key is
// only needed for encrypt/decrypt at the client side; homomorphic AND/OR
// only ever touch the crypto context and public evaluation keys.
struct Context {
    CryptoContextT crypto_context;
    PublicKeyT public_key;
    PrivateKeyT secret_key; // kept only on the "client" side of this demo

    // Mirrors Params::multiplicative_depth used to build crypto_context.
    // Kept here (rather than queried back from OpenFHE's internal
    // CryptoParametersBase, which does not expose it on the common base
    // type) purely so remaining_levels() below can report against it.
    uint32_t configured_depth = 0;
};

// Builds a BGV-RNS crypto context from `params`, generates a keypair, and
// generates the relinearization (multiplication) key. PKE, KEYSWITCH, and
// LEVELEDSHE are the only features enabled -- no bootstrapping, no
// advanced-SHE features this circuit does not need.
inline Context setup(const Params& params) {
    if (params.plaintext_modulus <= 2)
        throw std::invalid_argument(
            "bgv::setup: plaintext_modulus must be > 2 so Boolean AND/OR/XOR "
            "results (which can transiently reach values > 1 before the "
            "final gate identity reduces them back to {0,1}) never wrap "
            "around mod t.");
    if (params.multiplicative_depth < 1)
        throw std::invalid_argument("bgv::setup: multiplicative_depth must be >= 1");

    lbcrypto::CCParams<lbcrypto::CryptoContextBGVRNS> cc_params;
    cc_params.SetPlaintextModulus(params.plaintext_modulus);
    cc_params.SetMultiplicativeDepth(params.multiplicative_depth);
    cc_params.SetSecurityLevel(params.security_level);
    if (params.ring_dimension > 0)
        cc_params.SetRingDim(params.ring_dimension);
    if (params.batch_size > 0)
        cc_params.SetBatchSize(params.batch_size);

    Context ctx;
    ctx.configured_depth = params.multiplicative_depth;
    ctx.crypto_context = lbcrypto::GenCryptoContext(cc_params);
    ctx.crypto_context->Enable(lbcrypto::PKE);
    ctx.crypto_context->Enable(lbcrypto::KEYSWITCH);
    ctx.crypto_context->Enable(lbcrypto::LEVELEDSHE);

    auto keypair = ctx.crypto_context->KeyGen();
    if (!keypair.good())
        throw std::runtime_error("bgv::setup: OpenFHE key generation failed");
    ctx.public_key = keypair.publicKey;
    ctx.secret_key = keypair.secretKey;

    // Relinearization keys, needed after every EvalMult so ciphertext size
    // stays constant across the AND/OR-tree chain in bgv_backend.hpp.
    ctx.crypto_context->EvalMultKeyGen(ctx.secret_key);

    return ctx;
}

// Ring dimension actually chosen by OpenFHE (useful for the debug summary
// even when params.ring_dimension was 0 / automatic).
inline uint32_t ring_dimension(const Context& ctx) {
    return ctx.crypto_context->GetRingDimension();
}

// ── Encryption / decryption ──────────────────────────────────────────────
//
// One Boolean value per ciphertext: encoded as a length-1 packed plaintext
// (slot 0 = the bit, all higher slots implicitly zero). See the "future
// batching" note in bgv_backend.hpp for how this could later pack many
// bits per ciphertext.
inline Ciphertext encrypt_bit(const Context& ctx, bool bit) {
    std::vector<int64_t> slot{static_cast<int64_t>(bit ? 1 : 0)};
    auto plaintext = ctx.crypto_context->MakePackedPlaintext(slot);
    return ctx.crypto_context->Encrypt(ctx.public_key, plaintext);
}

// Decrypts `ct` and asserts the recovered value is exactly 0 or 1. Any
// other value is reported as a correctness error (insufficient
// multiplicative depth, mismatched keys, etc.) rather than being silently
// coerced to a Boolean -- see the "no silent conversion" requirement.
inline bool decrypt_bit(const Context& ctx, const Ciphertext& ct) {
    lbcrypto::Plaintext plaintext;
    lbcrypto::DecryptResult result = ctx.crypto_context->Decrypt(ctx.secret_key, ct, &plaintext);
    if (!result.isValid)
        throw std::runtime_error("bgv::decrypt_bit: OpenFHE decryption failed");

    plaintext->SetLength(1);
    int64_t value = plaintext->GetPackedValue()[0];
    if (value != 0 && value != 1)
        throw std::runtime_error(
            "bgv::decrypt_bit: decrypted value " + std::to_string(value) +
            " is not a valid Boolean (0 or 1) -- this indicates insufficient "
            "multiplicative depth or a parameter/key mismatch, not a value "
            "to silently coerce.");
    return value != 0;
}

// ── Encrypted bit logic ───────────────────────────────────────────────────
//
// Same {0,1}-ring identities as dghv.hpp, now evaluated with OpenFHE's
// EvalAdd/EvalSub/EvalMult. Each gate costs exactly one EvalMult, i.e. one
// level of multiplicative depth -- matching the depth accounting in
// bgv_backend.hpp's EstimateBGVDepth.

// AND(a,b) = a*b
inline Ciphertext encrypted_and(const Context& ctx, const Ciphertext& a, const Ciphertext& b) {
    return ctx.crypto_context->EvalMult(a, b);
}

// OR(a,b) = a + b - a*b
inline Ciphertext encrypted_or(const Context& ctx, const Ciphertext& a, const Ciphertext& b) {
    auto sum = ctx.crypto_context->EvalAdd(a, b);
    auto product = ctx.crypto_context->EvalMult(a, b);
    return ctx.crypto_context->EvalSub(sum, product);
}

// XOR(a,b) = a + b - 2*a*b (not used by the transitive-closure circuit,
// included for parity with dghv.hpp's gate set).
inline Ciphertext encrypted_xor(const Context& ctx, const Ciphertext& a, const Ciphertext& b) {
    auto sum = ctx.crypto_context->EvalAdd(a, b);
    auto product = ctx.crypto_context->EvalMult(a, b);
    auto two_product = ctx.crypto_context->EvalAdd(product, product);
    return ctx.crypto_context->EvalSub(sum, two_product);
}

// Remaining multiplicative levels available to `ct` before the depth
// budget set by Params::multiplicative_depth is exhausted. Purely
// informational (for the optional per-round debug printout); does not
// affect correctness.
inline uint32_t remaining_levels(const Context& ctx, const Ciphertext& ct) {
    uint32_t used = static_cast<uint32_t>(ct->GetLevel());
    return used >= ctx.configured_depth ? 0 : ctx.configured_depth - used;
}

} // namespace bgv
} // namespace btc

#endif // TC_WITH_BGV
