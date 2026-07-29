# Polaris job scripts

Run the bounded transitive closure FHE benchmark on a single Polaris compute
node — CPU boolean backend (sweeping graph size N) or GPU backend (4x A100,
fixed demo graph, smoke test only for now).

## Files

- `gen_graph.py` — generates `data/graph_N{N}.txt` for an arbitrary N (random
  directed graph, no self-loops). Same format/defaults as `benchmarks/gen_graphs.cpp`
  but not limited to {4, 8, 16}.
- `run_polaris_test.pbs` — CPU smoke test: 1 Polaris node, debug queue, a
  single small N (default 64), 10 min walltime. Run this first to confirm the
  CPU build and FHE run work end-to-end before submitting the full sweep.
- `run_polaris.pbs` — PBS job: requests 1 Polaris node, builds `bench_tfhe`
  with `TC_WITH_TFHE=ON`, generates a graph and runs the benchmark for each N
  in the sweep.
- `run_polaris_gpu_test.pbs` — GPU smoke test: 1 Polaris node (4x A100), debug
  queue, builds `bench_tfhe_gpu` with `TC_WITH_TFHE_GPU=ON` and runs it against
  the fixed 8x8 demo graph baked into `benchmarks/bench_tfhe_gpu.cpp`. Requires
  a tfhe-rs C API build with the GPU/CUDA feature enabled — a CPU-only
  `libtfhe.a` will fail to link. There is no GPU sweep script yet (the GPU
  benchmark doesn't read `graph_N{N}` files like `bench_tfhe` does).

## Quick test (debug queue, single job)

```bash
qsub -A Dist_relational_alg -v TFHE_ROOT=/path/to/tfhe-rs/target/release scripts/run_polaris_test.pbs
```

Runs one N=64 graph on the debug queue with a 30-minute walltime cap. Override
the size with `-v TFHE_ROOT=...,N=32`. Check `tc-fhe-test.log` for output and
timing before submitting the full sweep below.

## Full sweep

You need a local build of [tfhe-rs](https://github.com/zama-ai/tfhe-rs)'s C API
(`make build_c_api`) on Polaris first — point `TFHE_ROOT` at the directory
containing `libtfhe.a` and `tfhe.h`.

```bash
qsub -A Dist_relational_alg -v TFHE_ROOT=/path/to/tfhe-rs/target/release scripts/run_polaris.pbs
```

Override the N sweep, edge density, or seed via `-v`:

```bash
qsub -A Dist_relational_alg \
     -v TFHE_ROOT=/path/to/tfhe-rs/target/release,N_VALUES="64 128 256",DENSITY=0.1 \
     scripts/run_polaris.pbs
```

Default sweep: `N_VALUES="64 128 256 512 1024 2048"`, `DENSITY=0.2`, `SEED=2024`.

## Output

- Per-N console output is captured in `tc-fhe-sweep.log` (job stdout/stderr).
- Timing results are appended to `data/bench_tfhe_results.csv` (one row per N),
  which `benchmarks/plot_bench_tfhe.py` can plot directly:

  ```bash
  python3 benchmarks/plot_bench_tfhe.py data/bench_tfhe_results.csv sweep.png
  ```

## GPU smoke test

```bash
qsub -A Dist_relational_alg -v TFHE_ROOT=/path/to/tfhe-rs-gpu/target/release scripts/run_polaris_gpu_test.pbs
```

Builds and runs `bench_tfhe_gpu` (fixed 8x8 graph, T=2) on one Polaris node's
A100s. Check `tc-fhe-gpu-test.log` for `nvidia-smi -L` output and timing.
Override `CUDA_ROOT` via `-v` if the default (`/usr/local/cuda-12.6`) doesn't
match the loaded `cudatoolkit-standalone` module.

## Notes

- `run_polaris.pbs` / `run_polaris_test.pbs` target the CPU boolean backend
  (`bench_tfhe`) — Polaris GPUs aren't used. One PBS job = one Polaris node,
  one FHE benchmark process, run serially across the requested N values.
- FHE cost grows fast with N (O(log T) boolean matmuls, each O(N^3) encrypted
  gate evaluations), so `walltime` in `run_polaris.pbs` may need raising for
  the larger N values (1024, 2048) — check `tc-fhe-sweep.log` timing and
  adjust `#PBS -l walltime=` accordingly.
- The GPU path (`run_polaris_gpu_test.pbs`) is a smoke test only: it always
  runs the same fixed 8x8 graph, since `bench_tfhe_gpu.cpp` doesn't yet accept
  a `graph_N{N}` argument the way `bench_tfhe.cpp` does. A GPU sweep script
  would need that support added first.
