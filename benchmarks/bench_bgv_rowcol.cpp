#ifdef TC_WITH_BGV_BATCHED

#include <btc/algorithms.hpp>
#include <btc/bgv_batched.hpp>
#include <btc/bgv_rowcol_backend.hpp>
#include <btc/graph_io.hpp>

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

namespace bgv_batched = btc::bgv_batched;
using clock_type = std::chrono::high_resolution_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

namespace {

std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// Peak resident set size in KB, Linux-specific (/proc/self/status VmHWM).
// Returns 0 if unavailable rather than throwing -- this is a "best effort,
// report if available" metric, not a correctness requirement.
long peak_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            long kb = 0;
            std::sscanf(line.c_str(), "VmHWM: %ld", &kb);
            return kb;
        }
    }
    return 0;
}

btc::BGVMatMulBackend parse_variant(const std::string& s) {
    if (s == "serial") return btc::BGVMatMulBackend::Serial;
    if (s == "openmp-twopass") return btc::BGVMatMulBackend::OpenMPTwoPass;
    if (s == "openmp-optimized") return btc::BGVMatMulBackend::OpenMPOptimized;
    std::fprintf(stderr, "Unknown --variant '%s' (expected serial|openmp-twopass|openmp-optimized)\n", s.c_str());
    std::exit(1);
}

const char* variant_name(btc::BGVMatMulBackend v) {
    switch (v) {
        case btc::BGVMatMulBackend::Serial: return "serial";
        case btc::BGVMatMulBackend::OpenMPTwoPass: return "openmp-twopass";
        case btc::BGVMatMulBackend::OpenMPOptimized: return "openmp-optimized";
    }
    return "unknown";
}

// Usage: bench_bgv_rowcol [--graph=<name>] [--T=<n>] [--variant=<v>] [--threads=<n>]
//
// --graph must name a power-of-two-sized graph (data/graph_N{2,4,8,16,32,64}.txt
// exist in this repo) -- this backend has no padding for other sizes (see
// bgv_rowcol_backend.hpp's IsPowerOfTwo requirement).
//
// --threads=1 with --variant=openmp-twopass/openmp-optimized isolates
// "speedup from the row/column packing layout" from "speedup from OpenMP
// parallelism" -- run the same graph at threads=1,2,4,8 to see the split,
// matching the required experimental plan.
struct Args {
    std::string graph_name = "graph_N4";
    int T = 0;
    std::string variant = "openmp-optimized";
    int threads = 0; // 0 = don't override omp_get_max_threads()'s default
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
        if (arg.rfind("--graph=", 0) == 0) args.graph_name = value_after("--graph=");
        else if (arg.rfind("--T=", 0) == 0) args.T = std::stoi(value_after("--T="));
        else if (arg.rfind("--variant=", 0) == 0) args.variant = value_after("--variant=");
        else if (arg.rfind("--threads=", 0) == 0) args.threads = std::stoi(value_after("--threads="));
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::fprintf(stderr,
                "Usage: bench_bgv_rowcol [--graph=<name>] [--T=<n>] "
                "[--variant=serial|openmp-twopass|openmp-optimized] [--threads=<n>]\n");
            std::exit(1);
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    btc::BGVMatMulBackend variant = parse_variant(args.variant);

    if (args.threads > 0) {
        omp_set_num_threads(args.threads);
    }

    std::string path = "data/" + args.graph_name + ".txt";
    btc::BoolMatrix A;
    try {
        A = btc::load_graph(path);
    } catch (const std::exception& e) {
        std::printf("Failed to load %s: %s\n", path.c_str(), e.what());
        return 1;
    }

    const std::size_t N = A.rows();
    if (!btc::IsPowerOfTwo(N)) {
        std::printf("ERROR: N=%zu is not a power of two -- the row/column-packed backend has no "
                     "padding for non-power-of-two sizes. Use --graph=graph_N4/graph_N8/graph_N16/... \n", N);
        return 1;
    }
    const int T = args.T > 0 ? args.T : static_cast<int>(N);

    const uint32_t required_depth = btc::EstimateBGVRowColDepth(N, T);

    const int rounds = [&] {
        int r = 0, remaining = T;
        while (remaining > 1) {
            remaining = (remaining % 2 == 0) ? remaining / 2 : remaining - 1;
            ++r;
        }
        return r;
    }();

    std::printf("=== bgv-rowcol (BGV/OpenFHE, row/column-packed) benchmark: %s (N=%zu, T=%d) ===\n",
                args.graph_name.c_str(), N, T);
    std::printf("  variant=%s requested_threads=%d omp_get_max_threads()=%d\n",
                variant_name(variant), args.threads, omp_get_max_threads());
    std::printf("  required_depth=%u rounds=%d\n", required_depth, rounds);

    // ── Phase 1: crypto-context creation + key generation ────────────────
    // batch_size = N exactly (not automatic, not the profile database's
    // N*N-oriented lookup) -- see bgv_rowcol_backend.hpp's batch_size == N
    // invariant.
    auto t0 = clock_type::now();
    bgv_batched::Params params;
    // See test_bgv_rowcol_tc.cpp for why 786433 (not the Params default of
    // 65537): this backend's SumOR re-mask roughly doubles multiplicative
    // depth vs. the other backends, pushing the automatically-chosen ring
    // dimension past where 65537 stays NTT-compatible.
    params.plaintext_modulus = 786433;
    params.multiplicative_depth = required_depth;
    params.batch_size = static_cast<uint32_t>(N);
    bgv_batched::Context ctx;
    try {
        ctx = bgv_batched::setup(params);
    } catch (const std::exception& e) {
        std::printf("bgv-rowcol setup failed: %s\n", e.what());
        return 1;
    }
    double keygen_ms = ms_since(t0);

    const uint32_t ring_dim = bgv_batched::ring_dimension(ctx);
    const uint32_t batch_size = bgv_batched::available_slots(ctx);
    std::printf("  ring_dimension=%u batch_size=%u\n", ring_dim, batch_size);

    // ── Phase 2: one-hot masks + rotation-key generation ─────────────────
    t0 = clock_type::now();
    btc::BGVRowColBackend backend(ctx, N, variant);
    double mask_setup_ms = ms_since(t0);

    // ── Phase 3: encryption ───────────────────────────────────────────────
    t0 = clock_type::now();
    auto enc_A = backend.Encrypt(A);
    double encrypt_ms = ms_since(t0);

    // ── Phase 4: homomorphic TC evaluation ───────────────────────────────
    backend.reset_counters();
    t0 = clock_type::now();
    auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
    double btc_ms = ms_since(t0);
    auto counters = backend.counters();

    // ── Phase 5: decryption and decoding ─────────────────────────────────
    t0 = clock_type::now();
    auto result = backend.Decrypt(enc_S);
    double decrypt_ms = ms_since(t0);

    auto plain_S = btc::bounded_transitive_closure(A, T);
    bool ok = (result == plain_S);

    long peak_kb = peak_rss_kb();

    std::printf("\n  --- timings (phases reported separately, not combined) ---\n");
    std::printf("  crypto-context + keygen:       %.3f ms\n", keygen_ms);
    std::printf("  masks + rotation keygen:       %.3f ms\n", mask_setup_ms);
    std::printf("  encrypt (N row + N col cts):   %.3f ms\n", encrypt_ms);
    std::printf("  homomorphic BTC evaluation:    %.3f ms (%.3f s)\n", btc_ms, btc_ms / 1000.0);
    std::printf("  decrypt + decode:              %.3f ms\n", decrypt_ms);
    std::printf("  total (sum of the above):      %.3f ms\n",
                keygen_ms + mask_setup_ms + encrypt_ms + btc_ms + decrypt_ms);
    if (peak_kb > 0)
        std::printf("  peak RSS:                      %ld KB (%.1f MB)\n", peak_kb, peak_kb / 1024.0);

    // matmul/SumOR/repacking are not separately timed by this first-cut
    // implementation -- BooleanDotProduct interleaves the AND and the
    // SumOR-tree inside one call, and repacking (MaskMult + EvalAdd into
    // row/column accumulators) is interleaved with dot-product computation
    // in every matmul variant. The operation counters below are the
    // honest substitute: they report exactly how many of each op the BTC
    // evaluation actually performed, rather than a fabricated timing split.
    std::printf("\n  --- operation counts (BTC evaluation phase only) ---\n");
    std::printf("  number of TC rounds:                 %d\n", rounds);
    std::printf("  BooleanDotProduct calls:              %zu\n", counters.dot_products);
    std::printf("  SumOR calls:                           %zu\n", counters.sumor_calls);
    std::printf("  ciphertext-ciphertext EvalMult:        %zu\n", counters.cc_mult);
    std::printf("  ciphertext-plaintext EvalMult (masks): %zu\n", counters.cp_mult);
    std::printf("  EvalAdd:                                %zu\n", counters.add);
    std::printf("  EvalSub:                                %zu\n", counters.sub);
    std::printf("  EvalRotate:                              %zu\n", counters.rotate);

    std::printf("\n  correctness result: %s\n", ok ? "PASS (matches plaintext)" : "FAIL");
    std::puts("\n  Compare against bench_bgv_batched on the same --graph to see the row/column-packed");
    std::puts("  layout vs. whole-matrix-packed layout tradeoff; run this binary at --threads=1 vs.");
    std::puts("  --threads=N with --variant=openmp-* to separate packing speedup from OpenMP speedup.");

    const std::string csv_path = "data/bench_bgv_rowcol_results.csv";
    bool csv_exists = std::ifstream(csv_path).good();
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists) {
        csv << "timestamp,graph,N,T,variant,requested_threads,omp_max_threads,ring_dimension,"
               "multiplicative_depth,keygen_ms,mask_setup_ms,encrypt_ms,btc_ms,decrypt_ms,total_ms,"
               "peak_rss_kb,dot_products,sumor_calls,cc_mult,cp_mult,add,sub,rotate,matches_plaintext\n";
    }
    csv << timestamp_now() << ',' << args.graph_name << ',' << N << ',' << T << ','
        << variant_name(variant) << ',' << args.threads << ',' << omp_get_max_threads() << ','
        << ring_dim << ',' << required_depth << ',' << keygen_ms << ',' << mask_setup_ms << ','
        << encrypt_ms << ',' << btc_ms << ',' << decrypt_ms << ','
        << (keygen_ms + mask_setup_ms + encrypt_ms + btc_ms + decrypt_ms) << ',' << peak_kb << ','
        << counters.dot_products << ',' << counters.sumor_calls << ',' << counters.cc_mult << ','
        << counters.cp_mult << ',' << counters.add << ',' << counters.sub << ',' << counters.rotate << ','
        << (ok ? "YES" : "NO") << '\n';

    return ok ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::puts("Build with -DTC_WITH_BGV_BATCHED=ON (and OpenFHE installed) to enable this benchmark.");
    return 0;
}
#endif
