# Polaris job scripts

Run the bounded transitive closure FHE benchmark on a single Polaris compute
node — CPU boolean backend or GPU backend (4x A100), sweeping graph size N.

## Files

- `gen_graph.py` — generates `data/graph_N{N}.txt` for an arbitrary N (random
  directed graph, no self-loops). Same format/defaults as `benchmarks/gen_graphs.cpp`
  but not limited to {4, 8, 16}.
- `build_tfhe_rs.sh` — builds tfhe-rs' C API with only the features tc-fhe
  needs (`boolean-c-api,shortint` for CPU). Slow (the `tfhe` crate takes a
  long time to compile) — run once, and only again if tfhe-rs itself changes.
  Run inside `tmux`/`screen`: an SSH disconnect SIGHUPs a foreground build.
- `build_tc_fhe.sh` — builds `bench_tfhe` (CPU) against an already-built
  tfhe-rs C API. Fast — run this for every rebuild after editing tc-fhe's own
  CPU code.
- `build_tc_fhe_gpu.sh` — same, but for `bench_tfhe_gpu` against a GPU-feature
  tfhe-rs build.
- `run_polaris_test.pbs` — CPU smoke test: 1 Polaris node, debug queue, a
  single small N (default 32). Run this first to confirm the build and FHE
  run work end-to-end before submitting the full sweep.
- `run_polaris.pbs` — PBS job: requests 1 Polaris node, generates a graph and
  runs `bench_tfhe` for each N in the sweep.
- `run_polaris_gpu_test.pbs` — GPU smoke test: 1 Polaris node (4x A100), debug
  queue, generates a graph and runs `bench_tfhe_gpu` against it. There's no
  dedicated GPU sweep script yet, but `bench_tfhe_gpu` takes a graph name the
  same way `bench_tfhe` does, so `run_polaris.pbs`'s sweep pattern (loop +
  `qsub -v N_VALUES=...`) applies equally if you build a GPU sweep script from it.

## Build

Build tfhe-rs' C API once (on the login node, inside tmux — see
`build_tfhe_rs.sh` for why):

```bash
tmux new -s tfhe-build
scripts/build_tfhe_rs.sh          # CPU: ~/tfhe-rs by default

# or, for the GPU backend (separate clone + feature set):
git clone https://github.com/zama-ai/tfhe-rs.git ~/tfhe-rs-gpu
source set_env.sh                    # cc must resolve to gcc, not NVIDIA's nvc
module load cudatoolkit-standalone    # load AFTER set_env.sh, for nvcc
TFHE_RS_DIR=~/tfhe-rs-gpu FEATURES=high-level-c-api,gpu scripts/build_tfhe_rs.sh
```

Then build tc-fhe itself (fast, rerun any time after editing source):

```bash
scripts/build_tc_fhe.sh        # CPU -> build/
scripts/build_tc_fhe_gpu.sh    # GPU -> build_gpu/ (uses ~/tfhe-rs-gpu by default)
```

Job scripts below assume these builds already exist — they don't build
anything themselves, so `qsub` doesn't need `-v TFHE_ROOT=...` any more.

GPU build gotchas already handled by the scripts above (see comments in
`build_tfhe_rs.sh` for details) if you ever need to redo this by hand:
tfhe-cuda-backend can't find CUDA via `pkg-config` on Polaris and falls back
to a hardcoded path that doesn't exist here, so `LIBRARY_PATH` is pointed at
the real `libcudart` location; and a failed first attempt (e.g. before
`cudatoolkit-standalone` was loaded) leaves a stale CMake cache that needs
clearing (`rm -rf ~/tfhe-rs-gpu/target/release/build/tfhe-cuda-common-*`)
before CUDA will be re-detected.

## Quick test (debug queue, single job)

```bash
qsub scripts/run_polaris_test.pbs
```

Runs one N=32 graph on the debug queue. Override the size (or density/seed)
via `-v`:

```bash
qsub -v N=64,DENSITY=0.1,SEED=42 scripts/run_polaris_test.pbs
```

Check `tc-fhe-test.log` for output and timing before submitting the full
sweep below.

## Full sweep

```bash
qsub scripts/run_polaris.pbs
```

Override the N sweep, edge density, or seed via `-v`:

```bash
qsub -v N_VALUES="64 128 256",DENSITY=0.1 scripts/run_polaris.pbs
```

Default sweep: `N_VALUES="8 16 32"`, `DENSITY=0.2`, `SEED=2024`.

## Output

- Per-N console output is captured in `tc-fhe-sweep.log` (job stdout/stderr).
- Timing results are appended to `data/bench_tfhe_results.csv` (one row per N):
  `timestamp,graph,N,T,fhe_ms,matches_plaintext,matches_ground_truth,params`.
  `matches_plaintext` checks the FHE result against a plaintext run of the
  same bounded-closure algorithm (validates the FHE arithmetic);
  `matches_ground_truth` checks it against `floyd_warshall` (validates T is
  large enough for correctness — see the note in `include/btc/algorithms.hpp`).
  Plot it with:

  ```bash
  python3 benchmarks/plot_bench_tfhe.py data/bench_tfhe_results.csv sweep.png
  ```

## GPU smoke test

```bash
qsub scripts/run_polaris_gpu_test.pbs
```

Override the test size the same way as the CPU test:

```bash
qsub -v N=32 scripts/run_polaris_gpu_test.pbs
```

Default: N=8, density 0.2, seed 2024. Check `tc-fhe-gpu-test.log` for
`nvidia-smi -L` output and timing; results are appended to
`data/bench_tfhe_gpu_results.csv` with the same schema as the CPU CSV above.
`build_tc_fhe_gpu.sh` defaults `CUDA_ROOT` to
`/soft/compilers/cudatoolkit/cuda-12.2.2` (Polaris' `cudatoolkit-standalone`
module) — override via env var if a different version is loaded.

## Notes

- `run_polaris.pbs` / `run_polaris_test.pbs` target the CPU boolean backend
  (`bench_tfhe`) — Polaris GPUs aren't used. One PBS job = one Polaris node,
  one FHE benchmark process per N, run serially across the requested N values.
- FHE cost grows fast with N (`O(N^3 log N)` encrypted gate evaluations —
  `O(log N)` boolean matmuls, each `O(N^3)`), so `walltime` in
  `run_polaris.pbs` may need raising for larger N values — check
  `tc-fhe-sweep.log` timing and adjust `#PBS -l walltime=` accordingly.
- `T` is always `N` (not `N-1`): any directed graph needs up to `N` hops to
  reach its full transitive closure once cycles are possible (simple paths
  are bounded by `N-1`, simple cycles by `N`) — see `include/btc/algorithms.hpp`.
