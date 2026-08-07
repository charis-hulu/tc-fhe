# BGV Row/Column-Packed Backend (OpenFHE, OpenMP)

This document explains the **bgv-rowcol** backend:
`include/btc/bgv_rowcol_backend.hpp`. It is a **third, separate** BGV
backend alongside `bgv_backend.hpp` (one ciphertext per matrix element) and
`bgv_batched_backend.hpp` (one ciphertext per whole `N x N` matrix) — it
does not modify, depend on, or replace either. All three are independently
buildable, selectable, and benchmarkable under the same `TC_WITH_BGV_BATCHED`
CMake option (this backend reuses `bgv_batched.hpp`'s crypto-context
primitives rather than re-deriving them).

## 1. What this backend adds

Instead of one ciphertext per element (`bgv_backend.hpp`) or one ciphertext
for the whole matrix (`bgv_batched_backend.hpp`), this backend packs **one
ciphertext per matrix row and one ciphertext per matrix column**
(`PackedBooleanMatrix`). Because a row and a column are already
slot-aligned, a dot product `C[i][j] = OR_k(A[i][k] AND B[k][j])` is a
single slot-wise `EvalMult` between `A.row(i)` and `B.col(j)` — no rotation
is needed just to bring `A[i][k]` and `B[k][j]` into the same slot (contrast
`bgv_batched_backend.hpp`'s broadcast-then-mask approach, which needs that
alignment work because the whole matrix shares one ciphertext).

The backend also adds **OpenMP parallelism** over the `N` independent output
rows (and, in one variant, columns), on top of the encrypted computation
itself — see §5.

## 2. Why not arithmetic `RotateAndSum`

The transitive-closure algorithm (`algorithms.hpp`, `sum_powers_recursive`)
needs a **Boolean-semiring** matmul: `C[i][j] = OR_k(A[i][k] AND B[k][j])`.
A tempting first implementation computes the dot product as

```cpp
auto products = cc->EvalMult(row, col);   // A[i][k] * B[k][j], per slot
auto cij = RotateAndSum(products);         // sum_k A[i][k] * B[k][j]
```

This computes a **walk count** (how many `k` satisfy both terms), not a
Boolean existence result. Fed into `sum_powers_recursive`'s repeated
squaring, that walk count stops being idempotent `{0,1}`, so the recursion
no longer computes transitive closure at all, and risks silently exceeding
the plaintext modulus after enough rounds (`t` is only sized here assuming
`{0,1}` semantics). This backend replaces "rotate-and-sum" with
"rotate-and-OR-reduce" (`SumOR`), using the same `{0,1}`-ring OR identity
(`a+b-ab`) already used throughout this project. `tests/test_bgv_rowcol_matmul.cpp`'s
`or_not_sum_two_witnesses` test exists specifically to catch a regression
back to arithmetic summation: two converging witnesses must decrypt to `1`,
never `2`.

## 3. `SumOR` and the `batch_size == N` masking subtlety

`SumOR` reduces all `N` slots to their Boolean OR (replicated into every
slot) via a log-doubling rotate-and-OR schedule (`ceil(log2 N)` steps,
shifts `1, 2, 4, ..., N/2`).

The crypto context's packed batch size is forced to equal `N` exactly
(`Params::batch_size = N`, checked in `BGVRowColBackend`'s constructor). It
is tempting to assume this makes a plain `EvalRotate(ct, shift)` behave as
an exact cyclic rotation modulo `N` "for free" (some SIMD-batching designs
rely on the encoding replicating a sub-batch across the ring). **This
assumption is wrong for this OpenFHE build**, verified empirically while
building this backend: encrypting `N` values at `batch_size == N` leaves
every slot from `N` up to the ring's full `~ring_dim/2` capacity as an
honest **zero** (not a replica of the first `N` slots), and `EvalRotate`
performs a plain physical rotation over that entire (huge) range — content
that "falls off" the end of the logical `[0,N)` window walks into the zero
padding instead of wrapping back to slot `0`.

So a correct "rotate the logical window by `shift`, wrapping within `N`"
(`RotateModN`) combines two physical rotations whose valid ranges are
disjoint and together cover `[0,N)`:

```cpp
lo = EvalRotate(ct, shift)        // correct on slots [0, N-shift)
hi = EvalRotate(ct, shift - N)    // correct on slots [N-shift, N)
combined = EvalAdd(lo, hi)        // correct on all of [0, N)...
```

...but `combined` is **not clean beyond slot `N`**: `hi`'s invalid region
lands close to the window (around slot `N` to `2N-shift`), close enough
that the *next* rotation step reads it right back into the window,
corrupting later steps. This is the same "wrap-around leaking into the
matrix region" hazard `bgv_batched_backend.hpp`'s `valid_region_mask_`
defends against, at whole-matrix scale instead of a single log-doubling
reduction. `RotateModN` therefore re-masks `combined` down to exactly
`[0,N)` with a `[1]*N` plaintext mask after every step.

## 4. Multiplicative depth and repacking cost

Each `SumOR` step costs **two** `EvalMult` calls, not one: one for the
Boolean OR (`a+b-ab`'s `a*b` term) and one for `RotateModN`'s re-mask. In
this OpenFHE build's default scaling technique (FIXEDAUTO/FLEXIBLEAUTO),
ciphertext-plaintext `EvalMult` consumes a multiplicative-depth level
exactly like ciphertext-ciphertext `EvalMult` — it is *not* free the way
`EvalAdd`/`EvalRotate` are. The same is true of the repacking `MaskMult`
(the one-hot mask multiply that isolates `cij` into its target slot before
`EvalAdd`-ing it into a row/column accumulator).

So one full matmul costs:

```
1 (slot-wise AND) + 2*ceil(log2 N) (SumOR: OR + re-mask per step)
+ 1 (repacking MaskMult) = 2 + 2*ceil(log2 N)
```

(`BGVRowColMatmulDepth`). An earlier version of this backend undercounted
both the `SumOR` re-mask and the repacking mask, and produced non-Boolean
decrypted values at runtime (ran out of levels mid-circuit) before being
corrected — the same undercounting pitfall already documented for
`bgv_batched_backend.hpp`'s `mask_mult`.

Because this backend's depth grows roughly twice as fast in `N` as the
other two backends, its automatically-chosen ring dimension can outgrow
OpenFHE's default plaintext modulus's NTT-compatibility window sooner.
Tests and benchmarks here explicitly set `plaintext_modulus = 786433` (the
same batching-friendly prime already used throughout
`config/bgv_batched_profiles.json`, compatible with every power-of-two ring
dimension up to `131072`) rather than relying on `Params`'s default `65537`.

## 5. OpenMP variants and race-avoidance strategy

`BGVMatMulBackend` selects between three matmul implementations, all
computing the identical result:

- **`Serial`** — no OpenMP. Never materializes an `N x N` grid of
  ciphertexts; only `2N` row/column accumulators plus one temporary `cij`
  are live at a time.
- **`OpenMPTwoPass`** — race-free by construction: Stage 1 parallelizes
  over output rows (each iteration writes only `out_rows[i]`), Stage 2
  separately parallelizes over output columns. No two threads ever touch
  the same accumulator, at the cost of computing every `BooleanDotProduct`
  **twice** (once per stage) — a deliberate first-cut correctness-over-
  efficiency tradeoff.
- **`OpenMPOptimized`** — computes every `cij` exactly once: threads
  parallelize over output rows and accumulate column contributions into
  their own **thread-local** column accumulator array (never a shared
  mutable ciphertext), reduced into the final columns after the parallel
  region. Uses `O(threads * N)` temporary accumulators instead of
  two-pass's `O(N)`.

OpenFHE's own `DCRTPolyImpl` (the RNS polynomial type backing every
ciphertext) parallelizes its per-RNS-tower arithmetic internally via
`#pragma omp parallel for`, and its `ParallelControls` constructor leaves
nested-parallelism-disabling calls commented out. Left unchecked, this
backend's own outer `#pragma omp parallel for` would spawn `P` outer
threads each spawning their own inner tower-parallel threads —
oversubscription, not a hypothetical (this is exactly what the OpenMP
row/column feasibility analysis flagged as a risk before this backend was
built). `BGVRowColBackend`'s (and `BGVDistributedRowColBackend`'s)
constructor therefore calls `omp_set_max_active_levels(GetConfiguredMaxActiveLevels())`
once, which defaults to 1 so nested OpenFHE-internal parallel regions
execute serially instead of spawning more threads. Set
`TC_OMP_MAX_ACTIVE_LEVELS=2` in the environment to allow one level of
nesting instead -- this doesn't create extra core budget (oversubscription
is bounded by a node's physical core count, not by how many MPI
ranks/nodes are in the job), so pair it with a correspondingly lower
`OMP_NUM_THREADS`/`--depth` per rank.

## 6. Memory tradeoffs

`Serial` and `OpenMPTwoPass` both hold `O(N)` live ciphertexts at a time.
`OpenMPOptimized` holds `O(threads * N)` (thread-local column
accumulators) — measure peak RSS (`bench_bgv_rowcol` reports it via
`/proc/self/status VmHWM` on Linux) before defaulting to `OpenMPOptimized`
at large `N` or high thread counts.

## 7. How to build and run

Same CMake option as the other two OpenFHE backends:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV_BATCHED=ON \
      -DCMAKE_PREFIX_PATH=/path/to/openfhe-install
cmake --build build --target bench_bgv_rowcol test_bgv_rowcol_matmul test_bgv_rowcol_tc
```

This also links `OpenMP::OpenMP_CXX` into the `btc` target (added
specifically for this backend — see the root `CMakeLists.txt`'s
`TC_WITH_BGV_BATCHED` block), so every consumer target actually gets the
OpenMP compiler flag for its own translation unit, not just OpenFHE's own
libraries.

```bash
ctest --test-dir build --output-on-failure -R test_bgv_rowcol

./build/benchmarks/bench_bgv_rowcol --graph=graph_N8 --T=8 \
    --variant=openmp-twopass --threads=4
```

`--graph` must name a power-of-two-sized graph (`data/graph_N{2,4,8,16,32,64}.txt`
in this repo) — this backend has no padding for other sizes. Run the same
graph at `--threads=1` vs. `--threads=N` (holding `--variant` fixed) to
separate speedup from the row/column packing layout from speedup from
OpenMP parallelism itself, per the original feasibility analysis's
experimental-plan requirement.

## 8. Known limitations / not implemented here

- No padding to the next power of two — callers must supply power-of-two
  `N` (see `IsPowerOfTwo`).
- No tiled/incremental row processing to bound `OpenMPOptimized`'s
  `O(threads * N)` accumulator memory at large `N` — see §16 of the
  original design spec for the tiling idea, left as future work.
- Per-phase timing (matmul vs. `SumOR` vs. repacking) is not separately
  instrumented; `bench_bgv_rowcol` reports operation counts
  (`cc_mult`, `cp_mult`, `add`, `sub`, `rotate`, `dot_products`,
  `sumor_calls`) instead, since `BooleanDotProduct` interleaves the AND and
  the `SumOR` tree inside one call.
