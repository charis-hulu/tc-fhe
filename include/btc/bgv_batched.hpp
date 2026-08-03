#pragma once

#ifdef TC_WITH_BGV_BATCHED

// ── BGV batched: SIMD-packed leveled FHE over OpenFHE's BGVrns scheme ──────
//
// This is a SEPARATE backend from bgv.hpp / bgv_backend.hpp. It does not
// modify, replace, or depend on the unpacked (one-ciphertext-per-bit) BGV
// backend; the two coexist so both can be built and benchmarked
// independently (see include/btc/bgv_batched_backend.hpp, TC_WITH_BGV_BATCHED
// in CMakeLists.txt, and --backend bgv-batched in examples/bgv_batched_closure.cpp).
//
// The cryptographic setup (context, keys, encrypt/decrypt, gate identities)
// mirrors bgv.hpp closely -- same {0,1}-ring AND/OR identities, same leveled
// (non-bootstrapped) depth accounting. What's new here is:
//
//   - a packed encode/decode API that maps a full vector of {0,1} values
//     (up to the ring's batch size) into ONE ciphertext instead of one
//     ciphertext per bit;
//   - rotation-key generation for a caller-supplied set of offsets;
//   - a rotateToOffset() helper that documents and wraps EvalRotate's sign
//     convention (verified empirically -- see tests/test_bgv_batched_rotation.cpp
//     for the dedicated test that pins this down rather than assuming it).
//
// ── Boolean representation ───────────────────────────────────────────────
//
//   Same as bgv.hpp: a Boolean value is a BGV plaintext integer in {0, 1}
//   mod the plaintext modulus t (t > 2 so AND/OR never wrap):
//
//       AND(a,b) = a*b
//       OR(a,b)  = a + b - a*b
//
// ── Leveled, not bootstrapped ────────────────────────────────────────────
//
//   Exactly as in bgv.hpp: multiplicative depth is fixed up front via
//   Params::multiplicative_depth (see EstimateBGVBatchedDepth in
//   bgv_batched_backend.hpp) and baked into the crypto context before key
//   generation. No bootstrapping.
//
// ─────────────────────────────────────────────────────────────────────────

#include "openfhe.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace btc {
namespace bgv_batched {

using Ciphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using Plaintext = lbcrypto::Plaintext;
using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using PublicKeyT = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

// ── Parameters ────────────────────────────────────────────────────────────
struct Params {
    // Plaintext modulus t. Must be > 2 so {0,1} AND/OR results never wrap
    // around (see file header). 65537 is a batching-friendly NTT prime
    // commonly used in OpenFHE BGV examples -- same default as bgv.hpp.
    uint64_t plaintext_modulus = 65537;

    // Multiplicative depth the crypto context is built to support. Callers
    // should set this from btc::EstimateBGVBatchedDepth(N, T)
    // (bgv_batched_backend.hpp).
    uint32_t multiplicative_depth = 1;

    // Target security level.
    lbcrypto::SecurityLevel security_level = lbcrypto::HEStd_128_classic;

    // Ring dimension override. 0 means "let OpenFHE choose automatically".
    uint32_t ring_dimension = 0;

    // SIMD batch size (number of packed plaintext slots per ciphertext).
    // 0 means "let OpenFHE choose automatically" (typically ring_dim/2 for
    // BGV's default packing). This backend DOES rely on batching -- unlike
    // bgv.hpp's Params::batch_size, this value (once resolved by OpenFHE)
    // determines the maximum N^2 that fits in a single packed ciphertext
    // (see BooleanMatrixMultiplyPacked's capacity check).
    uint32_t batch_size = 0;
};

// ── Crypto context / key material ────────────────────────────────────────
struct Context {
    CryptoContextT crypto_context;
    PublicKeyT public_key;
    PrivateKeyT secret_key; // kept only on the "client" side of this demo

    uint32_t configured_depth = 0;

    // Rotation offsets for which EvalRotateKeyGen has been called. Recorded
    // purely for instrumentation (benchmark reporting of "number of
    // generated rotation keys" -- see §14 of the task spec).
    std::vector<int32_t> generated_rotation_offsets;
};

// Builds a BGV-RNS crypto context from `params`, generates a keypair, and
// generates the relinearization (multiplication) key. Mirrors bgv::setup;
// rotation keys are generated separately (see generate_rotation_keys below)
// because the required offsets depend on N, which setup() does not know.
inline Context setup(const Params& params) {
    if (params.plaintext_modulus <= 2)
        throw std::invalid_argument(
            "bgv_batched::setup: plaintext_modulus must be > 2 so Boolean "
            "AND/OR results never wrap around mod t.");
    if (params.multiplicative_depth < 1)
        throw std::invalid_argument("bgv_batched::setup: multiplicative_depth must be >= 1");

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
        throw std::runtime_error("bgv_batched::setup: OpenFHE key generation failed");
    ctx.public_key = keypair.publicKey;
    ctx.secret_key = keypair.secretKey;

    ctx.crypto_context->EvalMultKeyGen(ctx.secret_key);

    return ctx;
}

// Ring dimension actually chosen by OpenFHE.
inline uint32_t ring_dimension(const Context& ctx) {
    return ctx.crypto_context->GetRingDimension();
}

// Number of SIMD plaintext slots actually available per ciphertext. This is
// the capacity bound BooleanMatrixMultiplyPacked checks N*N against.
inline uint32_t available_slots(const Context& ctx) {
    return ctx.crypto_context->GetEncodingParams()->GetBatchSize();
}

// ── Rotation keys ─────────────────────────────────────────────────────────
//
// Precomputes and deduplicates rotation keys for exactly the offsets given.
// Must be called once per Context (not inside a matmul/TC round -- see the
// "do not regenerate keys inside a matmul or TC round" requirement). Callers
// (BGVBatchedBackend) are responsible for working out the full set of
// horizontal (+/-1..N-1) and vertical (+/-N..(N-1)*N) offsets a given matrix
// dimension N requires; this function just deduplicates and installs them.
inline void generate_rotation_keys(Context& ctx, const std::vector<int32_t>& offsets) {
    std::set<int32_t> unique_offsets;
    for (int32_t o : offsets)
        if (o != 0) // rotation by 0 is a no-op; never generate a key for it
            unique_offsets.insert(o);

    if (unique_offsets.empty())
        return;

    std::vector<int32_t> to_generate(unique_offsets.begin(), unique_offsets.end());
    ctx.crypto_context->EvalRotateKeyGen(ctx.secret_key, to_generate);
    ctx.generated_rotation_offsets = std::move(to_generate);
}

// ── Rotation direction convention ────────────────────────────────────────
//
// OpenFHE's EvalRotate(ct, index) convention (verified in
// tests/test_bgv_batched_rotation.cpp against the OpenFHE version this
// project links): a POSITIVE index rotates slot contents so that the value
// originally at slot (i + index) moves to slot i -- i.e. it rotates the
// vector LEFT by `index` positions (cyclically). Equivalently: after
// Rotate(v, index), result[i] = v[(i + index) mod slots].
//
// rotateToOffset(ctx, ct, offset) is a thin, self-documenting wrapper so the
// broadcast formulas in bgv_batched_backend.hpp (written in terms of "shift
// slot content by `offset` positions") don't have to re-derive or
// re-question the sign convention at each call site. offset == 0 is handled
// without a rotation call (a rotation by 0 is a no-op, and OpenFHE only has
// keys for the nonzero offsets generated by generate_rotation_keys).
inline Ciphertext rotateToOffset(const Context& ctx, const Ciphertext& ct, int32_t offset) {
    if (offset == 0)
        return ct;
    return ctx.crypto_context->EvalRotate(ct, offset);
}

// ── Packed encryption / decryption ───────────────────────────────────────
//
// Encodes an arbitrary vector of {0,1} values into ONE packed plaintext /
// ciphertext (slots beyond values.size() are implicitly zero-padded by
// MakePackedPlaintext up to the ring's batch size). This is the primitive
// full-matrix row-major packing (see bgv_batched_backend.hpp::PackMatrix)
// is built on.
inline Plaintext make_packed_plaintext(const Context& ctx, const std::vector<int64_t>& values) {
    return ctx.crypto_context->MakePackedPlaintext(values);
}

inline Ciphertext encrypt_packed(const Context& ctx, const std::vector<int64_t>& values) {
    auto plaintext = make_packed_plaintext(ctx, values);
    return ctx.crypto_context->Encrypt(ctx.public_key, plaintext);
}

// Decrypts `ct` and returns the first `count` packed slot values, without
// asserting Boolean-ness -- callers decrypting intermediate debug state
// (masks, broadcasts) may see non-{0,1} values transiently. Boolean-only
// call sites should use decrypt_packed_bits below.
inline std::vector<int64_t> decrypt_packed_raw(const Context& ctx, const Ciphertext& ct, std::size_t count) {
    Plaintext plaintext;
    lbcrypto::DecryptResult result = ctx.crypto_context->Decrypt(ctx.secret_key, ct, &plaintext);
    if (!result.isValid)
        throw std::runtime_error("bgv_batched::decrypt_packed_raw: OpenFHE decryption failed");
    plaintext->SetLength(count);
    auto packed = plaintext->GetPackedValue();
    packed.resize(count, 0);
    return packed;
}

// Decrypts `ct`, asserts every one of the first `count` slots is exactly 0
// or 1 (see the "no silent conversion" requirement), and returns them as
// bools.
inline std::vector<bool> decrypt_packed_bits(const Context& ctx, const Ciphertext& ct, std::size_t count) {
    auto raw = decrypt_packed_raw(ctx, ct, count);
    std::vector<bool> bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (raw[i] != 0 && raw[i] != 1)
            throw std::runtime_error(
                "bgv_batched::decrypt_packed_bits: decrypted slot " + std::to_string(i) +
                " = " + std::to_string(raw[i]) + " is not a valid Boolean (0 or 1) -- "
                "this indicates insufficient multiplicative depth or a parameter/key "
                "mismatch, not a value to silently coerce.");
        bits[i] = (raw[i] != 0);
    }
    return bits;
}

// ── Encrypted bit logic (slot-wise, applies to every packed slot at once) ─
//
// Same {0,1}-ring identities as bgv.hpp, now operating on whole packed
// vectors simultaneously -- this is the "measurable use of BGV batching"
// requirement: one EvalMult/EvalAdd call performs the operation across all
// N*N slots of a packed matrix in parallel, instead of N*N separate calls.

// AND(a,b) = a*b, slot-wise.
inline Ciphertext encrypted_and(const Context& ctx, const Ciphertext& a, const Ciphertext& b) {
    return ctx.crypto_context->EvalMult(a, b);
}

// OR(a,b) = a + b - a*b, slot-wise.
inline Ciphertext encrypted_or(const Context& ctx, const Ciphertext& a, const Ciphertext& b) {
    auto sum = ctx.crypto_context->EvalAdd(a, b);
    auto product = ctx.crypto_context->EvalMult(a, b);
    return ctx.crypto_context->EvalSub(sum, product);
}

// Ciphertext-plaintext multiplication (used for masking with row/column
// masks -- must NOT be implemented as ciphertext-ciphertext EvalMult, see
// the task spec's explicit requirement).
inline Ciphertext mask_mult(const Context& ctx, const Ciphertext& ct, const Plaintext& mask) {
    return ctx.crypto_context->EvalMult(ct, mask);
}

inline Ciphertext add(const Context& ctx, const Ciphertext& a, const Ciphertext& b) {
    return ctx.crypto_context->EvalAdd(a, b);
}

// Remaining multiplicative levels available to `ct` before the depth budget
// set by Params::multiplicative_depth is exhausted. Purely informational.
inline uint32_t remaining_levels(const Context& ctx, const Ciphertext& ct) {
    uint32_t used = static_cast<uint32_t>(ct->GetLevel());
    return used >= ctx.configured_depth ? 0 : ctx.configured_depth - used;
}

} // namespace bgv_batched
} // namespace btc

#endif // TC_WITH_BGV_BATCHED
