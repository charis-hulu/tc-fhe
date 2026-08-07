#ifdef TC_WITH_BGV_MPI

// ── Integration test for the offline-artifact prepare/compute/decrypt
//    workflow (include/btc/bgv_distributed_rowcol_artifacts.hpp,
//    tools/bgv_distributed_rowcol_artifacts.cpp) ─────────────────────────
//
// Unlike test_bgv_distributed_rowcol.cpp (which links the backend directly
// into ONE process and calls it in-process), this test deliberately shells
// out to the built CLI tool via std::system() for EVERY stage -- prepare,
// compute, decrypt each run as a genuinely separate OS process, per the
// task's explicit correctness requirement ("must verify that the
// implementation does not accidentally rely on in-memory OpenFHE state from
// the process that generated the keys"). TOOL_BINARY_PATH (see
// tests/CMakeLists.txt) is the exact built path of that tool, injected at
// compile time so this doesn't have to guess the build directory's name.
//
// Requires `mpirun` to be launchable directly from wherever this test
// itself runs (needed for the compute stage) -- same constraint
// test_bgv_distributed_rowcol.cpp's ctest registration already has; on this
// project's Polaris setup that means running inside a PBS job, not from the
// login node directly (mpirun there fails with "Couldn't execute local
// shepherd" -- unrelated to this backend).
//
// Tests N=4, T=4 at MPI ranks 1 and 4 (the task's required minimum
// coverage), each as a full prepare -> compute -> decrypt cycle across
// three separate processes, comparing the final decrypted result against
// the plaintext bounded_transitive_closure reference.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what.c_str());
    }
}

// Runs `cmd` as a genuinely separate OS process and returns true iff it
// exits with status 0.
bool RunProcess(const std::string& cmd) {
    std::fprintf(stderr, "+ %s\n", cmd.c_str());
    std::fflush(stderr);
    int rc = std::system(cmd.c_str());
    return rc == 0;
}

// One full prepare -> compute (`nranks` MPI ranks) -> decrypt cycle, each
// stage its own process (steps 2-11 of the task's required integration
// test). `check()` on the decrypt stage covers requirement 11 ("compare
// against plaintext transitive closure") because RunDecrypt's
// --expected-input already exits nonzero on a mismatch -- see
// tools/bgv_distributed_rowcol_artifacts.cpp's RunDecrypt.
void RunPipeline(const std::string& tool, const std::string& graph_path, const std::string& artifact_dir, int T,
                  int nranks) {
    std::error_code ec;
    fs::remove_all(artifact_dir, ec); // start clean -- a stale artifact dir
                                       // from a previous failed run must not
                                       // leak into this one

    const std::string prepare_cmd = tool + " --mode=prepare --input=" + graph_path +
                                     " --artifact-dir=" + artifact_dir + " --T=" + std::to_string(T);
    check(RunProcess(prepare_cmd), "prepare (artifact_dir=" + artifact_dir + ")");

    const std::string compute_cmd =
        "mpirun -n " + std::to_string(nranks) + " " + tool + " --mode=compute --artifact-dir=" + artifact_dir;
    check(RunProcess(compute_cmd), "compute nranks=" + std::to_string(nranks));

    const std::string decrypt_cmd =
        tool + " --mode=decrypt --artifact-dir=" + artifact_dir + " --expected-input=" + graph_path;
    check(RunProcess(decrypt_cmd), "decrypt nranks=" + std::to_string(nranks) +
                                        " (includes comparison against plaintext bounded_transitive_closure)");
}

} // namespace

int main() {
#ifndef TOOL_BINARY_PATH
    std::fprintf(stderr, "SKIP: TOOL_BINARY_PATH not defined by the build\n");
    return 0;
#else
    const std::string tool = TOOL_BINARY_PATH;
    if (!fs::exists(tool)) {
        std::fprintf(stderr, "SKIP: tool binary not found at %s\n", tool.c_str());
        return 0;
    }

    const std::string graph_path = "data/graph_N4.txt";
    const int T = 4;

    RunPipeline(tool, graph_path, "artifacts/test_bgv_distributed_rowcol_artifacts_np1", T, 1);
    RunPipeline(tool, graph_path, "artifacts/test_bgv_distributed_rowcol_artifacts_np4", T, 4);

    return g_failures == 0 ? 0 : 1;
#endif
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_MPI=ON (and MPI + OpenFHE installed) to enable this test.");
    return 0;
}
#endif
