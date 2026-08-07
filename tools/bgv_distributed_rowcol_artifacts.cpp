#ifdef TC_WITH_BGV_MPI

// ── CLI for the distributed BGV row/column-packed backend's offline
//    key-generation / ciphertext-serialization workflow ────────────────────
//
// See include/btc/bgv_distributed_rowcol_artifacts.hpp for the actual
// implementation; this file is just argument parsing and mode dispatch.
//
//   ./bgv_distributed_rowcol_artifacts --mode=prepare \
//       --input=data/graph_N4.txt --artifact-dir=artifacts/graph_n4 --T=4
//
//   mpirun -n 4 ./bgv_distributed_rowcol_artifacts --mode=compute \
//       --artifact-dir=artifacts/graph_n4
//
//   ./bgv_distributed_rowcol_artifacts --mode=decrypt \
//       --artifact-dir=artifacts/graph_n4 --expected-input=data/graph_N4.txt
//
// `prepare`/`decrypt` are single-process and never call MPI_Init -- matches
// the task's own usage examples (no `mpirun` prefix on those). `compute`
// (and `full`, the backward-compatible one-shot convenience) call MPI_Init
// unconditionally since compute mode is the SPMD one.

#include <btc/algorithms.hpp>
#include <btc/bgv_distributed_rowcol_artifacts.hpp>
#include <btc/bgv_distributed_rowcol_backend.hpp>
#include <btc/graph_io.hpp>

#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace artifacts = btc::bgv_distributed_rowcol_artifacts;
namespace bgv_batched = btc::bgv_batched;
using clock_type = std::chrono::steady_clock;

namespace {

double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

struct Args {
    std::string mode;
    std::string input;         // prepare: path to a graph_io.hpp-format graph file
    std::string artifact_dir;  // all modes
    int T = 0;                 // prepare: transitive-closure bound
    uint64_t plaintext_modulus = 786433;
    std::string expected_input; // decrypt/full: optional plaintext comparison
};

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
        if (arg.rfind("--mode=", 0) == 0) args.mode = value_after("--mode=");
        else if (arg.rfind("--input=", 0) == 0) args.input = value_after("--input=");
        else if (arg.rfind("--artifact-dir=", 0) == 0) args.artifact_dir = value_after("--artifact-dir=");
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else if (arg.rfind("--plaintext-modulus=", 0) == 0)
            args.plaintext_modulus = std::stoull(value_after("--plaintext-modulus="));
        else if (arg.rfind("--expected-input=", 0) == 0) args.expected_input = value_after("--expected-input=");
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(
                stderr,
                "Usage: bgv_distributed_rowcol_artifacts --mode=prepare|compute|decrypt|full "
                "--artifact-dir=<dir> [options]\n"
                "  prepare: --input=<graph.txt> --T=<n> [--plaintext-modulus=<u64>]. No mpirun.\n"
                "  compute: run under `mpirun -n K`. Requires a prior prepare in --artifact-dir.\n"
                "  decrypt: [--expected-input=<graph.txt>] to compare against plaintext. No mpirun.\n"
                "  full:    prepare + compute + decrypt in one process, for backward-compatible use;\n"
                "           needs --input=<graph.txt> --T=<n> [--expected-input=<graph.txt>].\n");
            std::exit(1);
        }
    }
    if (args.mode.empty()) {
        std::fprintf(stderr, "Missing required --mode=prepare|compute|decrypt|full\n");
        std::exit(1);
    }
    if (args.artifact_dir.empty()) {
        std::fprintf(stderr, "Missing required --artifact-dir=<dir>\n");
        std::exit(1);
    }
    return args;
}

// ── prepare ────────────────────────────────────────────────────────────

int RunPrepare(const Args& args) {
    if (args.input.empty()) {
        std::fprintf(stderr, "prepare: requires --input=<graph.txt>\n");
        return 1;
    }
    if (args.T <= 0) {
        std::fprintf(stderr, "prepare: requires --T=<n> (>=1)\n");
        return 1;
    }

    btc::BoolMatrix plain;
    try {
        plain = btc::load_graph(args.input);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prepare: failed to load %s: %s\n", args.input.c_str(), e.what());
        return 1;
    }

    try {
        artifacts::PrepareTimings timings;
        auto manifest = artifacts::Prepare(args.artifact_dir, plain, args.T, args.plaintext_modulus, timings);

        std::printf("=== prepare complete: %s ===\n", args.artifact_dir.c_str());
        std::printf("  matrix_size=%zu T=%d plaintext_modulus=%llu ring_dimension=%u multiplicative_depth=%u\n",
                     manifest.matrix_size, manifest.transitive_closure_bound,
                     static_cast<unsigned long long>(manifest.plaintext_modulus), manifest.ring_dimension,
                     manifest.multiplicative_depth);
        std::printf("  keyset_id=%s rotation_indices=%zu\n", manifest.keyset_id.c_str(),
                     manifest.rotation_indices.size());
        std::printf("  prepare_context_time_ms=%.2f\n", timings.context_ms);
        std::printf("  prepare_keygen_time_ms=%.2f\n", timings.keygen_ms);
        std::printf("  prepare_eval_mult_key_time_ms=%.2f\n", timings.eval_mult_key_ms);
        std::printf("  prepare_rotation_key_time_ms=%.2f\n", timings.rotation_key_ms);
        std::printf("  prepare_encryption_time_ms=%.2f\n", timings.encryption_ms);
        std::printf("  prepare_serialization_time_ms=%.2f\n", timings.serialization_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prepare: FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}

// ── compute ────────────────────────────────────────────────────────────
//
// Runs entirely inside one nested scope so BGVDistributedRowColBackend (and
// the temporary ProcessGrid2D used to work out this rank's ciphertext
// files) are destroyed -- freeing their MPI communicators -- before
// MPI_Finalize() runs in main(), same hazard test_bgv_distributed_rowcol.cpp
// documents.
int RunComputeBody(const Args& args, int rank) {
    artifacts::ArtifactPaths paths{args.artifact_dir};
    artifacts::ComputeTimings timings;

    artifacts::Manifest input_manifest = artifacts::LoadManifest(paths.manifest());
    artifacts::ValidateManifest(input_manifest, input_manifest.matrix_size);
    const std::size_t n = input_manifest.matrix_size;
    const int T = input_manifest.transitive_closure_bound;

    auto t0 = clock_type::now();
    bgv_batched::Context ctx;
    artifacts::LoadCryptoContext(paths.crypto_context(), ctx);
    // BGVDistributedRowColBackend::MatMulOnGrid calls EncryptZeroPublic()
    // (a fresh encryption of an all-zero plaintext, to build the zero
    // row/column accumulator) BEFORE any dot product or rotation runs --
    // see LoadContextForCompute's doc comment
    // (bgv_distributed_rowcol_artifacts.hpp) for why compute mode needs the
    // public key after all, despite doing no fresh encryption of the
    // caller's actual input. Timed together with the context load since
    // it's the same "establish this process's crypto context" phase, not a
    // separate disk-read pass.
    artifacts::LoadPublicKey(paths.public_key(), ctx);
    timings.context_load_ms = ms_since(t0);

    t0 = clock_type::now();
    artifacts::LoadEvalMultKey(paths.eval_mult_key(), ctx);
    artifacts::LoadEvalAutomorphismKey(paths.eval_automorphism_key(), ctx);
    timings.eval_key_load_ms = ms_since(t0);

    btc::DistributedRowColMatrix enc_A;
    {
        // Ownership: the SAME BuildProcessGrid2D/ComputeBlockPartition
        // functions BGVDistributedRowColBackend's constructor uses,
        // called here first (before that constructor runs) purely to know
        // which ciphertext files this rank needs -- a temporary, thrown
        // away once loading is done (the backend below builds its own
        // equivalent grid internally, deterministically, from the same
        // MPI_COMM_WORLD).
        btc::ProcessGrid2D grid = btc::BuildProcessGrid2D(MPI_COMM_WORLD, n);
        btc::BlockPartition row_partition = btc::ComputeBlockPartition(n, grid.Pr);
        btc::BlockPartition col_partition = btc::ComputeBlockPartition(n, grid.Pc);

        t0 = clock_type::now();
        auto block = artifacts::LoadOwnedCiphertextsForRank(paths, row_partition, col_partition, grid.process_row,
                                                              grid.process_col, rank);
        timings.ciphertext_load_ms = ms_since(t0);

        enc_A = btc::DistributedRowColMatrix(n, block.row_indices, block.rows, block.col_indices, block.cols);
    }

    t0 = clock_type::now();
    btc::BGVDistributedRowColBackend backend(MPI_COMM_WORLD, n, std::move(ctx));
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    timings.compute_ms = ms_since(t0);

    t0 = clock_type::now();
    auto gathered = backend.GatherEncryptedResultToRoot(enc_S);
    if (rank == 0) artifacts::SaveDistributedResult(paths, gathered, input_manifest);
    timings.result_save_ms = ms_since(t0);

    if (rank == 0) {
        std::printf("=== compute complete: %s (world_size=%d) ===\n", args.artifact_dir.c_str(),
                     backend.world_size());
        std::printf("  runtime_context_load_time_ms=%.2f\n", timings.context_load_ms);
        std::printf("  runtime_eval_key_load_time_ms=%.2f\n", timings.eval_key_load_ms);
        std::printf("  runtime_ciphertext_load_time_ms=%.2f\n", timings.ciphertext_load_ms);
        std::printf("  runtime_compute_time_ms=%.2f  <-- distributed FHE computation time\n", timings.compute_ms);
        std::printf("  runtime_result_save_time_ms=%.2f\n", timings.result_save_ms);
        const double online_end_to_end = timings.context_load_ms + timings.eval_key_load_ms +
                                          timings.ciphertext_load_ms + timings.compute_ms + timings.result_save_ms;
        std::printf("  online_end_to_end_time_ms=%.2f\n", online_end_to_end);
    }
    return 0;
}

int RunCompute(const Args& args) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_status = 0;
    std::string error_message;
    try {
        local_status = RunComputeBody(args, rank);
    } catch (const std::exception& e) {
        local_status = 1;
        error_message = e.what();
    }
    if (local_status != 0)
        std::fprintf(stderr, "compute: rank %d FAILED: %s\n", rank, error_message.c_str());

    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return global_status;
}

// ── decrypt ────────────────────────────────────────────────────────────

int RunDecrypt(const Args& args) {
    artifacts::ArtifactPaths paths{args.artifact_dir};
    try {
        artifacts::Manifest input_manifest = artifacts::LoadManifest(paths.manifest());
        artifacts::ValidateManifest(input_manifest, input_manifest.matrix_size);
        artifacts::Manifest result_manifest = artifacts::LoadManifest(paths.result_manifest());
        artifacts::RequireKeysetMatch(input_manifest, result_manifest);
        const std::size_t n = input_manifest.matrix_size;

        artifacts::DecryptTimings timings;

        auto t0 = clock_type::now();
        bgv_batched::Context ctx;
        artifacts::LoadCryptoContext(paths.crypto_context(), ctx);
        artifacts::LoadSecretKey(paths.secret_key(), ctx);
        auto encrypted_result = artifacts::LoadDistributedResult(paths, n);
        timings.load_ms = ms_since(t0);

        t0 = clock_type::now();
        btc::BoolMatrix decrypted = artifacts::DecryptResult(ctx, n, encrypted_result);
        timings.decrypt_ms = ms_since(t0);

        std::printf("=== decrypt complete: %s ===\n", args.artifact_dir.c_str());
        std::printf("  decrypt_load_time_ms=%.2f\n", timings.load_ms);
        std::printf("  decrypt_time_ms=%.2f\n", timings.decrypt_ms);

        if (!args.expected_input.empty()) {
            btc::BoolMatrix plain = btc::load_graph(args.expected_input);
            btc::BoolMatrix expected = btc::bounded_transitive_closure(plain, input_manifest.transitive_closure_bound);
            if (decrypted == expected) {
                std::printf("  MATCH: decrypted result equals plaintext bounded_transitive_closure\n");
            } else {
                std::fprintf(stderr, "  MISMATCH: decrypted result does NOT equal plaintext "
                                      "bounded_transitive_closure\n");
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decrypt: FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}

// ── full (prepare + compute + decrypt, one process, backward-compatible) ─

int RunFull(const Args& args) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Every rank calls MPI_Bcast unconditionally (even though only rank 0's
    // status is meaningful going in) so a prepare failure on rank 0 can
    // never leave other ranks blocked on a collective rank 0 already
    // returned past -- an earlier version of this function used
    // `if (rank == 0) { ...; return status; } MPI_Barrier(...)`, which
    // deadlocks every non-root rank exactly that way whenever prepare fails.
    int prepare_status = 0;
    if (rank == 0) prepare_status = RunPrepare(args);
    MPI_Bcast(&prepare_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (prepare_status != 0) return prepare_status;

    int compute_status = RunCompute(args); // already collective/synchronized internally
    if (compute_status != 0) return compute_status;

    int decrypt_status = 0;
    if (rank == 0) decrypt_status = RunDecrypt(args);
    MPI_Bcast(&decrypt_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return decrypt_status;
}

} // namespace

int main(int argc, char** argv) {
    Args args = ParseArgs(argc, argv);

    if (args.mode == "prepare") return RunPrepare(args);
    if (args.mode == "decrypt") return RunDecrypt(args);

    if (args.mode == "compute" || args.mode == "full") {
        MPI_Init(&argc, &argv);
        int status = (args.mode == "compute") ? RunCompute(args) : RunFull(args);
        MPI_Finalize();
        return status;
    }

    std::fprintf(stderr, "Unknown --mode=%s (expected prepare|compute|decrypt|full)\n", args.mode.c_str());
    return 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_MPI=ON (and MPI + OpenFHE installed) to enable this tool.");
    return 0;
}
#endif
