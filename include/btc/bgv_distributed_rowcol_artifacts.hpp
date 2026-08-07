#pragma once

#ifdef TC_WITH_BGV_MPI

// ── Offline key generation and ciphertext serialization for the distributed
//    BGV row/column-packed backend ──────────────────────────────────────────
//
// bgv_distributed_rowcol_backend.hpp's BGVDistributedRowColBackend always
// performs context/key generation and input encryption INLINE, every run --
// fine for correctness testing, but it means every benchmark run pays BGV
// key generation and encryption cost before any distributed compute even
// starts. This file separates that into three independently-runnable
// stages, matching claude_offline_bgv_mpi_artifacts.md:
//
//   prepare  (single process, no MPI): generate context + keys + rotation
//            keys, encrypt the input matrix, serialize everything to disk.
//   compute  (SPMD, under mpirun): every rank independently loads the
//            crypto context + eval-mult/rotation keys (NEVER the secret
//            key) and only the row/column ciphertext files IT owns (via
//            BuildProcessGrid2D/ComputeBlockPartition -- the SAME ownership
//            function bgv_distributed_rowcol_backend.hpp's constructor
//            already uses, not a new one), runs the existing distributed
//            transitive closure UNCHANGED, and serializes the still-
//            encrypted result.
//   decrypt  (single process, no MPI): loads the context + secret key +
//            encrypted result, decrypts, optionally compares to plaintext.
//
// None of bgv_distributed_rowcol_backend.hpp's actual matmul/or_mat/
// matmul_pair/assembly logic is touched -- this file only adds a new
// pre-loaded-Context constructor overload to that class (see its doc
// comment) and a GatherEncryptedResultToRoot wrapper; everything else here
// is pure I/O plumbing built on the existing bgv_mpi_serial.hpp
// (de)serialization primitives.
//
// ── Why every rank loads independently instead of an MPI broadcast ────────
//
// EncryptFromRoot's constructor broadcasts a freshly-generated context
// bundle over MPI because rank 0 is the only rank that JUST built it in
// memory. Here, every rank instead reads the SAME already-serialized files
// from a shared filesystem (the artifact directory) -- no MPI communication
// is needed to distribute them, matching the task's "First implement
// independent key deserialization on every rank" guidance. Ciphertext
// ownership works the same way: rather than root loading everything and
// scattering it, every rank works out (via BuildProcessGrid2D +
// ComputeBlockPartition, called locally, no communication) exactly which
// row/column files it needs and reads only those -- ranks sharing a
// row_comm/col_comm block DO redundantly re-read the same files
// independently (matching or_mat's/GatherToRankWithinGrid's existing
// "redundant-recompute-over-broadcast" tradeoff elsewhere in this backend),
// but no rank ever reads a file it doesn't need.
//
// ─────────────────────────────────────────────────────────────────────────

#include "bgv_batched.hpp"
#include "bgv_distributed_rowcol_backend.hpp"
#include "bgv_mpi_serial.hpp"
#include "bgv_rowcol_backend.hpp"
#include "matrix.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace btc {
namespace bgv_distributed_rowcol_artifacts {

namespace fs = std::filesystem;

// Bumped whenever the on-disk layout or the ciphertext-per-file convention
// below changes incompatibly -- ValidateManifest rejects anything else
// rather than guessing.
constexpr int kFormatVersion = 1;
constexpr const char* kPackingLayout = "rowcol-v1";

// ── Manifest ────────────────────────────────────────────────────────────

struct Manifest {
    int format_version = kFormatVersion;
    std::string openfhe_version;
    std::string scheme = "BGVRNS";
    std::size_t matrix_size = 0;
    int transitive_closure_bound = 0;
    uint64_t plaintext_modulus = 0;
    uint32_t ring_dimension = 0;
    uint32_t batch_size = 0;
    uint32_t multiplicative_depth = 0;
    std::string packing_layout = kPackingLayout;
    std::size_t num_row_ciphertexts = 0;
    std::size_t num_col_ciphertexts = 0;
    std::vector<int32_t> rotation_indices;
    // Identifies the (params, keypair) this manifest's ciphertexts/keys were
    // produced with -- see ComputeKeysetId. A compute-mode result manifest
    // copies the SAME keyset_id verbatim from the input manifest it was
    // computed from (compute ranks never see the secret key, so they cannot
    // recompute it) so decrypt mode can catch an accidental artifact-dir
    // mix-up before touching the secret key.
    std::string keyset_id;
};

inline void to_json(nlohmann::json& j, const Manifest& m) {
    j = nlohmann::json{
        {"format_version", m.format_version},
        {"openfhe_version", m.openfhe_version},
        {"scheme", m.scheme},
        {"matrix_size", m.matrix_size},
        {"transitive_closure_bound", m.transitive_closure_bound},
        {"plaintext_modulus", m.plaintext_modulus},
        {"ring_dimension", m.ring_dimension},
        {"batch_size", m.batch_size},
        {"multiplicative_depth", m.multiplicative_depth},
        {"packing_layout", m.packing_layout},
        {"num_row_ciphertexts", m.num_row_ciphertexts},
        {"num_col_ciphertexts", m.num_col_ciphertexts},
        {"rotation_indices", m.rotation_indices},
        {"keyset_id", m.keyset_id},
    };
}

inline void from_json(const nlohmann::json& j, Manifest& m) {
    j.at("format_version").get_to(m.format_version);
    if (j.contains("openfhe_version")) j.at("openfhe_version").get_to(m.openfhe_version);
    j.at("scheme").get_to(m.scheme);
    j.at("matrix_size").get_to(m.matrix_size);
    j.at("transitive_closure_bound").get_to(m.transitive_closure_bound);
    j.at("plaintext_modulus").get_to(m.plaintext_modulus);
    j.at("ring_dimension").get_to(m.ring_dimension);
    j.at("batch_size").get_to(m.batch_size);
    j.at("multiplicative_depth").get_to(m.multiplicative_depth);
    j.at("packing_layout").get_to(m.packing_layout);
    j.at("num_row_ciphertexts").get_to(m.num_row_ciphertexts);
    j.at("num_col_ciphertexts").get_to(m.num_col_ciphertexts);
    j.at("rotation_indices").get_to(m.rotation_indices);
    j.at("keyset_id").get_to(m.keyset_id);
}

inline void SaveManifest(const fs::path& path, const Manifest& m) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("SaveManifest: cannot open " + path.string() + " for writing");
    nlohmann::json j = m;
    out << j.dump(2) << "\n";
    if (!out) throw std::runtime_error("SaveManifest: write failed: " + path.string());
}

inline Manifest LoadManifest(const fs::path& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("LoadManifest: cannot open " + path.string() +
                                  " -- run --mode prepare first to generate it");
    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("LoadManifest: malformed JSON in " + path.string() + ": " + e.what());
    }
    Manifest m;
    try {
        m = j.get<Manifest>();
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("LoadManifest: " + path.string() +
                                  " does not match the expected manifest schema: " + e.what());
    }
    return m;
}

// Rejects (with a specific, actionable message) anything ValidateManifest's
// caller cannot safely proceed with -- see this file's header comment and
// the task's "Reject the artifacts with a clear error if..." list. Only
// checks what's determinable from the manifest alone; RequireKeysetMatch
// and file-existence checks are separate (see below) because they need a
// SECOND manifest or a filesystem probe respectively.
inline void ValidateManifest(const Manifest& m, std::size_t expected_n) {
    if (m.format_version != kFormatVersion)
        throw std::runtime_error("ValidateManifest: unsupported format_version " +
                                  std::to_string(m.format_version) + " (expected " +
                                  std::to_string(kFormatVersion) + ")");
    if (m.packing_layout != kPackingLayout)
        throw std::runtime_error("ValidateManifest: unsupported packing_layout '" + m.packing_layout +
                                  "' (expected '" + kPackingLayout + "')");
    if (m.matrix_size != expected_n)
        throw std::runtime_error("ValidateManifest: matrix_size mismatch -- manifest says " +
                                  std::to_string(m.matrix_size) + ", runtime expected " +
                                  std::to_string(expected_n));
    if (m.num_row_ciphertexts != m.matrix_size || m.num_col_ciphertexts != m.matrix_size)
        throw std::runtime_error("ValidateManifest: inconsistent ciphertext counts -- matrix_size=" +
                                  std::to_string(m.matrix_size) +
                                  " but num_row_ciphertexts=" + std::to_string(m.num_row_ciphertexts) +
                                  " num_col_ciphertexts=" + std::to_string(m.num_col_ciphertexts));
    if (!IsPowerOfTwo(m.matrix_size))
        throw std::runtime_error("ValidateManifest: matrix_size " + std::to_string(m.matrix_size) +
                                  " is not a power of two");
}

inline void RequireKeysetMatch(const Manifest& input_manifest, const Manifest& result_manifest) {
    if (input_manifest.keyset_id != result_manifest.keyset_id)
        throw std::runtime_error(
            "RequireKeysetMatch: keyset_id mismatch between input manifest ('" + input_manifest.keyset_id +
            "') and result manifest ('" + result_manifest.keyset_id +
            "') -- the encrypted result was not produced from this artifact directory's keys/input "
            "(mixed-up --artifact-dir / --result, or a stale result from a previous prepare run)");
}

// ── Keyset identity ─────────────────────────────────────────────────────
//
// Not a cryptographic commitment -- a simple, deterministic fingerprint
// (FNV-1a 64-bit, dependency-free) over the parameter description and the
// serialized public-key bytes, sufficient to catch "these ciphertexts and
// this key material don't actually belong together" (task's own "a simple
// implementation is sufficient" allowance), not to resist adversarial
// collision construction.
inline std::string ComputeKeysetId(const Manifest& m, const std::string& public_key_bytes) {
    std::ostringstream material;
    material << m.scheme << '|' << m.matrix_size << '|' << m.plaintext_modulus << '|' << m.ring_dimension << '|'
             << m.batch_size << '|' << m.multiplicative_depth << '|' << m.packing_layout;
    for (int32_t idx : m.rotation_indices) material << '|' << idx;
    material << '|' << public_key_bytes;

    const std::string s = material.str();
    uint64_t hash = 1469598103934665603ULL; // FNV-1a 64-bit offset basis
    for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ULL; // FNV-1a 64-bit prime
    }
    std::ostringstream hex;
    hex << std::hex << std::setfill('0') << std::setw(16) << hash;
    return hex.str();
}

// ── Artifact directory layout ────────────────────────────────────────────
//
// artifact_dir/
//   manifest.json
//   cryptocontext.bin  public_key.bin  secret_key.bin
//   eval_mult_key.bin  eval_automorphism_key.bin
//   input/row_0000.bin .. row_{N-1}.bin, col_0000.bin .. col_{N-1}.bin
//   result/result_manifest.json, row_0000.bin .. col_{N-1}.bin
//
// Row/column ciphertexts are saved as the FULL N-row/N-column matrix (not
// pre-split by an assumed rank count) -- compute mode's MPI world size is
// only known at compute time, and may legitimately differ run to run (see
// the task's `-np 16` example), so ownership is re-derived at load time via
// BuildProcessGrid2D/ComputeBlockPartition rather than baked into prepare
// mode's output.
struct ArtifactPaths {
    fs::path root;

    fs::path manifest() const { return root / "manifest.json"; }
    fs::path crypto_context() const { return root / "cryptocontext.bin"; }
    fs::path public_key() const { return root / "public_key.bin"; }
    fs::path secret_key() const { return root / "secret_key.bin"; }
    fs::path eval_mult_key() const { return root / "eval_mult_key.bin"; }
    fs::path eval_automorphism_key() const { return root / "eval_automorphism_key.bin"; }

    fs::path input_dir() const { return root / "input"; }
    fs::path row_file(std::size_t i) const { return input_dir() / RowName(i); }
    fs::path col_file(std::size_t j) const { return input_dir() / ColName(j); }

    fs::path result_dir() const { return root / "result"; }
    fs::path result_manifest() const { return result_dir() / "result_manifest.json"; }
    fs::path result_row_file(std::size_t i) const { return result_dir() / RowName(i); }
    fs::path result_col_file(std::size_t j) const { return result_dir() / ColName(j); }

    static std::string RowName(std::size_t i) {
        char buf[24];
        std::snprintf(buf, sizeof(buf), "row_%04zu.bin", i);
        return buf;
    }
    static std::string ColName(std::size_t j) {
        char buf[24];
        std::snprintf(buf, sizeof(buf), "col_%04zu.bin", j);
        return buf;
    }
};

// ── File I/O primitives ──────────────────────────────────────────────────

inline void WriteStringToFile(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("WriteStringToFile: cannot open " + path.string() + " for writing");
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("WriteStringToFile: write failed: " + path.string());
}

inline std::string ReadStringFromFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ReadStringFromFile: cannot open " + path.string() + " for reading");
    std::ostringstream oss;
    oss << in.rdbuf();
    if (in.bad()) throw std::runtime_error("ReadStringFromFile: read failed: " + path.string());
    return oss.str();
}

inline void RequireFileExists(const fs::path& path, int mpi_rank = -1) {
    if (fs::exists(path)) return;
    std::string msg = "RequireFileExists: missing required artifact file: " + path.string();
    if (mpi_rank >= 0) msg += " (MPI rank " + std::to_string(mpi_rank) + ")";
    throw std::runtime_error(msg);
}

// ── CryptoContext / key (de)serialization ───────────────────────────────
//
// Thin file-based wrappers around bgv_mpi_serial.hpp's existing string-based
// (de)serialization (crypto context, public key) plus OpenFHE's own
// SerializeEvalMultKey/SerializeEvalAutomorphismKey (already used the same
// way by bgv_mpi::SerializeContextBundle) -- no new serialization logic,
// just file <-> string plumbing. Secret-key (de)serialization lives ONLY
// here, not in bgv_mpi_serial.hpp, to keep that file's "secret key never
// touches this" contract (see its header comment) visibly true by
// construction -- this file is the one place in the codebase allowed to
// read/write a secret key to disk, and only decrypt mode ever calls
// LoadSecretKey.

inline void SaveCryptoContext(const fs::path& path, const bgv_batched::Context& ctx) {
    std::ostringstream oss(std::ios::binary);
    lbcrypto::Serial::Serialize(ctx.crypto_context, oss, lbcrypto::SerType::SERBINARY());
    WriteStringToFile(path, oss.str());
}

inline void LoadCryptoContext(const fs::path& path, bgv_batched::Context& ctx) {
    RequireFileExists(path);
    std::istringstream iss(ReadStringFromFile(path), std::ios::binary);
    lbcrypto::Serial::Deserialize(ctx.crypto_context, iss, lbcrypto::SerType::SERBINARY());
}

inline void SavePublicKey(const fs::path& path, const bgv_batched::Context& ctx) {
    std::ostringstream oss(std::ios::binary);
    lbcrypto::Serial::Serialize(ctx.public_key, oss, lbcrypto::SerType::SERBINARY());
    WriteStringToFile(path, oss.str());
}

inline void LoadPublicKey(const fs::path& path, bgv_batched::Context& ctx) {
    RequireFileExists(path);
    std::istringstream iss(ReadStringFromFile(path), std::ios::binary);
    lbcrypto::Serial::Deserialize(ctx.public_key, iss, lbcrypto::SerType::SERBINARY());
}

// Only decrypt mode calls this -- see this file's header comment. Compute
// ranks must never call LoadSecretKey (per the task's secret-key isolation
// requirement); nothing in this header wires LoadSecretKey into the compute
// path, so accidentally doing so requires a caller to explicitly opt in.
inline void SaveSecretKey(const fs::path& path, const bgv_batched::Context& ctx) {
    std::ostringstream oss(std::ios::binary);
    lbcrypto::Serial::Serialize(ctx.secret_key, oss, lbcrypto::SerType::SERBINARY());
    WriteStringToFile(path, oss.str());
}

inline void LoadSecretKey(const fs::path& path, bgv_batched::Context& ctx) {
    RequireFileExists(path);
    std::istringstream iss(ReadStringFromFile(path), std::ios::binary);
    lbcrypto::Serial::Deserialize(ctx.secret_key, iss, lbcrypto::SerType::SERBINARY());
}

inline void SaveEvalMultKey(const fs::path& path, const bgv_batched::Context& ctx) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("SaveEvalMultKey: cannot open " + path.string() + " for writing");
    ctx.crypto_context->SerializeEvalMultKey(out, lbcrypto::SerType::SERBINARY());
}

inline void LoadEvalMultKey(const fs::path& path, bgv_batched::Context& ctx) {
    RequireFileExists(path);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("LoadEvalMultKey: cannot open " + path.string() + " for reading");
    if (!ctx.crypto_context->DeserializeEvalMultKey(in, lbcrypto::SerType::SERBINARY()))
        throw std::runtime_error("LoadEvalMultKey: OpenFHE failed to deserialize " + path.string());
}

inline void SaveEvalAutomorphismKey(const fs::path& path, const bgv_batched::Context& ctx) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("SaveEvalAutomorphismKey: cannot open " + path.string() + " for writing");
    ctx.crypto_context->SerializeEvalAutomorphismKey(out, lbcrypto::SerType::SERBINARY());
}

inline void LoadEvalAutomorphismKey(const fs::path& path, bgv_batched::Context& ctx) {
    RequireFileExists(path);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("LoadEvalAutomorphismKey: cannot open " + path.string() + " for reading");
    if (!ctx.crypto_context->DeserializeEvalAutomorphismKey(in, lbcrypto::SerType::SERBINARY()))
        throw std::runtime_error("LoadEvalAutomorphismKey: OpenFHE failed to deserialize " + path.string());
}

inline void SaveCiphertext(const fs::path& path, const bgv_batched::Ciphertext& ct) {
    WriteStringToFile(path, bgv_mpi::SerializeCiphertext(ct));
}

inline bgv_batched::Ciphertext LoadCiphertext(const fs::path& path, int mpi_rank = -1) {
    RequireFileExists(path, mpi_rank);
    return bgv_mpi::DeserializeCiphertext(ReadStringFromFile(path));
}

// ── Timing ────────────────────────────────────────────────────────────────
//
// Matches the task's required category breakdown. `prepare_context_ms` and
// `prepare_keygen_ms` and `prepare_eval_mult_key_ms` are measured as THREE
// separate calls (GenCryptoContext / KeyGen / EvalMultKeyGen) rather than by
// calling bgv_batched::setup() as one opaque blob -- setup() bundles all
// three with no internal timing hooks, and adding them would mean
// instrumenting a file shared by every BGV-batched-family backend for a
// benchmarking-only concern local to this one. Prepare() below therefore
// inlines setup()'s own three lines directly (identical calls, identical
// parameters -- see its body) instead of calling it, purely to get
// separately-timeable phases.
struct PrepareTimings {
    double context_ms = 0;
    double keygen_ms = 0;
    double eval_mult_key_ms = 0;
    double rotation_key_ms = 0;
    double encryption_ms = 0;
    double serialization_ms = 0;
};

struct ComputeTimings {
    double context_load_ms = 0;
    double eval_key_load_ms = 0;
    double ciphertext_load_ms = 0;
    double compute_ms = 0;
    double result_save_ms = 0;
};

struct DecryptTimings {
    double load_ms = 0;
    double decrypt_ms = 0;
};

namespace detail {
using clock_type = std::chrono::steady_clock;
inline double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}
} // namespace detail

// ── Prepare mode ──────────────────────────────────────────────────────────
//
// Single process, no MPI. Builds the context, generates the keypair,
// eval-mult key, and EXACTLY the rotation indices BuildRequiredRotationIndices
// says this N needs (see bgv_rowcol_backend.hpp -- the only EvalRotate call
// site in the whole algorithm is RotateModN/SumOR, so this is complete, not
// a subset), encrypts `plain`, and serializes everything under `artifact_dir`.
inline Manifest Prepare(const fs::path& artifact_dir, const BoolMatrix& plain, int transitive_closure_bound,
                         uint64_t plaintext_modulus, PrepareTimings& timings) {
    const std::size_t n = plain.rows();
    if (plain.rows() != plain.cols())
        throw std::invalid_argument("Prepare: plain matrix must be square");
    if (!IsPowerOfTwo(n))
        throw std::invalid_argument("Prepare: matrix_size " + std::to_string(n) + " is not a power of two");

    ArtifactPaths paths{artifact_dir};
    fs::create_directories(paths.input_dir());
    fs::create_directories(paths.result_dir());

    const uint32_t depth = EstimateBGVRowColDepth(n, transitive_closure_bound);

    // ── Context + keypair + eval-mult key (see PrepareTimings' doc comment
    // for why this inlines bgv_batched::setup()'s three steps instead of
    // calling it) ──────────────────────────────────────────────────────────
    if (plaintext_modulus <= 2)
        throw std::invalid_argument("Prepare: plaintext_modulus must be > 2");

    lbcrypto::CCParams<lbcrypto::CryptoContextBGVRNS> cc_params;
    cc_params.SetPlaintextModulus(plaintext_modulus);
    cc_params.SetMultiplicativeDepth(depth);
    cc_params.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    cc_params.SetBatchSize(static_cast<uint32_t>(n));

    bgv_batched::Context ctx;
    ctx.configured_depth = depth;

    auto t0 = detail::clock_type::now();
    ctx.crypto_context = lbcrypto::GenCryptoContext(cc_params);
    ctx.crypto_context->Enable(lbcrypto::PKE);
    ctx.crypto_context->Enable(lbcrypto::KEYSWITCH);
    ctx.crypto_context->Enable(lbcrypto::LEVELEDSHE);
    timings.context_ms = detail::ms_since(t0);

    t0 = detail::clock_type::now();
    auto keypair = ctx.crypto_context->KeyGen();
    if (!keypair.good()) throw std::runtime_error("Prepare: OpenFHE key generation failed");
    ctx.public_key = keypair.publicKey;
    ctx.secret_key = keypair.secretKey;
    timings.keygen_ms = detail::ms_since(t0);

    t0 = detail::clock_type::now();
    ctx.crypto_context->EvalMultKeyGen(ctx.secret_key);
    timings.eval_mult_key_ms = detail::ms_since(t0);

    const auto rotation_indices = BuildRequiredRotationIndices(n);
    t0 = detail::clock_type::now();
    bgv_batched::generate_rotation_keys(ctx, rotation_indices);
    timings.rotation_key_ms = detail::ms_since(t0);

    // ── Encryption ──────────────────────────────────────────────────────
    t0 = detail::clock_type::now();
    BGVRowColBackend local_backend(ctx, n, BGVMatMulBackend::OpenMPOptimized,
                                    BGVRowColBackend::RotationKeys::AssumeAlreadyInstalled);
    PackedBooleanMatrix enc = local_backend.Encrypt(plain);
    timings.encryption_ms = detail::ms_since(t0);

    // ── Serialization ────────────────────────────────────────────────────
    t0 = detail::clock_type::now();
    SaveCryptoContext(paths.crypto_context(), ctx);
    SavePublicKey(paths.public_key(), ctx);
    SaveSecretKey(paths.secret_key(), ctx);
    SaveEvalMultKey(paths.eval_mult_key(), ctx);
    SaveEvalAutomorphismKey(paths.eval_automorphism_key(), ctx);
    for (std::size_t i = 0; i < n; ++i) SaveCiphertext(paths.row_file(i), enc.row(i));
    for (std::size_t j = 0; j < n; ++j) SaveCiphertext(paths.col_file(j), enc.col(j));

    Manifest manifest;
    manifest.openfhe_version = GetOPENFHEVersion();
    manifest.matrix_size = n;
    manifest.transitive_closure_bound = transitive_closure_bound;
    manifest.plaintext_modulus = plaintext_modulus;
    manifest.ring_dimension = bgv_batched::ring_dimension(ctx);
    manifest.batch_size = bgv_batched::available_slots(ctx);
    manifest.multiplicative_depth = depth;
    manifest.num_row_ciphertexts = n;
    manifest.num_col_ciphertexts = n;
    manifest.rotation_indices = rotation_indices;
    manifest.keyset_id = ComputeKeysetId(manifest, ReadStringFromFile(paths.public_key()));
    SaveManifest(paths.manifest(), manifest);
    timings.serialization_ms = detail::ms_since(t0);

    return manifest;
}

// ── Compute mode ──────────────────────────────────────────────────────────
//
// Loads crypto context, PUBLIC key, eval-mult key, and rotation keys --
// never the secret key (see the task's secret-key isolation requirement,
// enforced here simply by this function never calling LoadSecretKey).
//
// The public key turned out to be genuinely needed, not just optional
// belt-and-braces: BGVDistributedRowColBackend::MatMulOnGrid calls
// local_backend_->EncryptZeroPublic() (bgv_rowcol_backend.hpp) at the very
// START of every matmul, before any dot product or rotation runs, to build
// the zero row/column accumulator -- and EncryptZero encrypts a fresh
// all-zero plaintext via ctx.crypto_context->Encrypt(ctx.public_key, ...),
// which needs a non-null public key. "Compute mode performs no fresh
// encryption" (the task's own reasoning for making public-key loading
// optional) does not hold for this backend once EncryptZeroPublic is taken
// into account -- it IS a fresh encryption, just an internal implementation
// detail of matmul rather than encrypting the caller's input. Confirmed
// empirically on Polaris: without the public key loaded, compute mode threw
// "Key is nullptr" (OpenFHE's CheckKey, cryptocontext.h) immediately, before
// any distributed communication -- consistent with EncryptZeroPublic being
// the very first crypto-context operation MatMulOnGrid performs.
inline bgv_batched::Context LoadContextForCompute(const ArtifactPaths& paths) {
    bgv_batched::Context ctx;
    LoadCryptoContext(paths.crypto_context(), ctx);
    LoadPublicKey(paths.public_key(), ctx);
    LoadEvalMultKey(paths.eval_mult_key(), ctx);
    LoadEvalAutomorphismKey(paths.eval_automorphism_key(), ctx);
    return ctx;
}

// This rank's row/column block, ready to build a DistributedRowColMatrix
// from -- ownership is re-derived from `row_partition`/`col_partition`
// (ComputeBlockPartition(n, grid.Pr)/(n, grid.Pc), the SAME function
// BGVDistributedRowColBackend's constructor already uses), not stored
// per-rank in the manifest (see ArtifactPaths' doc comment on why).
struct OwnedBlock {
    std::vector<std::size_t> row_indices;
    std::vector<bgv_batched::Ciphertext> rows;
    std::vector<std::size_t> col_indices;
    std::vector<bgv_batched::Ciphertext> cols;
};

inline OwnedBlock LoadOwnedCiphertextsForRank(const ArtifactPaths& paths, const BlockPartition& row_partition,
                                               const BlockPartition& col_partition, int process_row,
                                               int process_col, int mpi_rank) {
    const std::size_t row_offset = row_partition.offsets[static_cast<std::size_t>(process_row)];
    const std::size_t local_rows = row_partition.counts[static_cast<std::size_t>(process_row)];
    const std::size_t col_offset = col_partition.offsets[static_cast<std::size_t>(process_col)];
    const std::size_t local_cols = col_partition.counts[static_cast<std::size_t>(process_col)];

    OwnedBlock block;
    block.row_indices.resize(local_rows);
    block.rows.resize(local_rows);
    for (std::size_t li = 0; li < local_rows; ++li) {
        const std::size_t gi = row_offset + li;
        block.row_indices[li] = gi;
        block.rows[li] = LoadCiphertext(paths.row_file(gi), mpi_rank);
    }
    block.col_indices.resize(local_cols);
    block.cols.resize(local_cols);
    for (std::size_t lj = 0; lj < local_cols; ++lj) {
        const std::size_t gj = col_offset + lj;
        block.col_indices[lj] = gj;
        block.cols[lj] = LoadCiphertext(paths.col_file(gj), mpi_rank);
    }
    return block;
}

// Saves the (still-encrypted) full gathered result -- see
// BGVDistributedRowColBackend::GatherEncryptedResultToRoot -- and a result
// manifest carrying the SAME keyset_id as `input_manifest` (compute ranks
// cannot recompute it themselves, see Manifest's doc comment). Call only on
// the rank `full` is meaningful on (rank 0, matching
// GatherEncryptedResultToRoot's convention).
inline void SaveDistributedResult(const ArtifactPaths& paths, const PackedBooleanMatrix& full,
                                   const Manifest& input_manifest) {
    fs::create_directories(paths.result_dir());
    for (std::size_t i = 0; i < full.dim(); ++i) SaveCiphertext(paths.result_row_file(i), full.row(i));
    for (std::size_t j = 0; j < full.dim(); ++j) SaveCiphertext(paths.result_col_file(j), full.col(j));
    SaveManifest(paths.result_manifest(), input_manifest);
}

inline PackedBooleanMatrix LoadDistributedResult(const ArtifactPaths& paths, std::size_t n) {
    std::vector<bgv_batched::Ciphertext> rows(n), cols(n);
    for (std::size_t i = 0; i < n; ++i) rows[i] = LoadCiphertext(paths.result_row_file(i));
    for (std::size_t j = 0; j < n; ++j) cols[j] = LoadCiphertext(paths.result_col_file(j));
    return PackedBooleanMatrix(n, std::move(rows), std::move(cols));
}

// ── Decrypt mode ──────────────────────────────────────────────────────────
//
// Constructs a throwaway local BGVRowColBackend purely to reuse
// BGVRowColBackend::Decrypt's row/column cross-check (see its doc comment)
// -- Decrypt never calls EvalRotate, so RotationKeys::AssumeAlreadyInstalled
// is safe here even though decrypt mode never loads rotation keys at all
// (they are simply not needed for decryption, only for matmul -- see this
// file's header comment on compute mode being the only stage that needs
// them).
inline BoolMatrix DecryptResult(bgv_batched::Context& ctx, std::size_t n, const PackedBooleanMatrix& full) {
    BGVRowColBackend local_backend(ctx, n, BGVMatMulBackend::OpenMPOptimized,
                                    BGVRowColBackend::RotationKeys::AssumeAlreadyInstalled);
    return local_backend.Decrypt(full);
}

} // namespace bgv_distributed_rowcol_artifacts
} // namespace btc

#endif // TC_WITH_BGV_MPI
