# BGV Batched Backend (OpenFHE, SIMD-packed)

This document explains the **bgv-batched** backend:
`include/btc/bgv_batched.hpp` (crypto context, packed encode/decode, rotation
keys) and `include/btc/bgv_batched_backend.hpp` (masks, broadcasts, packed
Boolean matmul, the `BGVBatchedBackend` adapter). It is a **separate**
backend from `include/btc/bgv.hpp` / `bgv_backend.hpp` (the unpacked,
one-ciphertext-per-bit BGV backend already in this repository) — neither
file modifies, depends on, or replaces the other. Both are independently
buildable, selectable, and benchmarkable; see §3/§4.

## 1. What this backend adds over the unpacked BGV backend

`BGVBackend` (`bgv_backend.hpp`) encrypts each Boolean matrix entry as its
own ciphertext: an `N x N` matrix costs `N^2` ciphertexts, and one matmul
dot product costs `N` ciphertext-ciphertext multiplications plus a balanced
OR-tree, entirely in scalar (non-SIMD) ciphertexts. This backend instead
packs **one complete `N x N` matrix into a single ciphertext**, using
OpenFHE BGV's SIMD plaintext slots, and evaluates the AND/OR gates of an
entire matmul round across all `N^2` output positions in parallel with a
constant number of `EvalMult`/`EvalAdd`/`EvalRotate` calls per inner-product
index `k` (not per output cell). This is the "measurable use of BGV
batching" this backend exists to provide.

## 2. How to install / locate OpenFHE

Identical to the unpacked BGV backend — see `docs/bgv.md` §2. This backend
uses the same OpenFHE `PKE` module, no additional OpenFHE features.

## 3. How to configure CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV_BATCHED=ON \
      -DCMAKE_PREFIX_PATH=/path/to/openfhe-install
cmake --build build
```

`TC_WITH_BGV_BATCHED` (default `OFF`) is an independent CMake option from
`TC_WITH_BGV`, `TC_WITH_TFHE`, and `TC_WITH_DGHV` — enable any subset. Both
`TC_WITH_BGV` and `TC_WITH_BGV_BATCHED` share the same `find_package(OpenFHE
CONFIG REQUIRED)` include/link setup in the root `CMakeLists.txt` (enabling
either, or both, triggers it once); each option only gates its own
`target_compile_definitions`, so building with just `TC_WITH_BGV_BATCHED=ON`
does not require or enable the unpacked `bgv` backend, and vice versa. When
`TC_WITH_BGV_BATCHED` is `OFF`, `bgv_batched_closure` / `bench_bgv_batched` /
`test_bgv_batched_*` still build (each has a `#else` stub `main()` that
prints a message and exits 0) but contain no OpenFHE code.

## 4. How to run the bgv-batched backend

```bash
# Default graph (built-in 4-node chain 0->1->2->3):
./build/examples/bgv_batched_closure

# Explicit backend selection (this binary only implements bgv-batched; see
# §11 for why backend selection here is build/binary-level, not a single
# shared dispatcher):
./build/examples/bgv_batched_closure --backend=bgv-batched

# Custom graph and T (plaintext modulus / ring dimension are looked up
# automatically from the profile database, not passed by hand -- see
# docs/bgv_parameter_profiles.md):
./build/examples/bgv_batched_closure --graph=data/graph_N8.txt --T=5

./build/benchmarks/bench_bgv_batched --graph=graph_N4
ctest --test-dir build --output-on-failure -R test_bgv_batched
```

## 5. Main parameters

Exposed via `btc::bgv_batched::Params` (`include/btc/bgv_batched.hpp`):

| Field | Meaning | Default |
|---|---|---|
| `plaintext_modulus` | Modulus `t`; must be `> 2`. | `65537` |
| `multiplicative_depth` | Hard ceiling on sequential `EvalMult` calls. See §8. | `1` (caller must override, see `EstimateBGVBatchedDepth`) |
| `security_level` | OpenFHE security-parameter table lookup. | `HEStd_128_classic` |
| `ring_dimension` | `0` = automatic. | `0` |
| `batch_size` | SIMD plaintext-slot count. **Unlike `bgv::Params::batch_size`, this backend's correctness and capacity DO depend on it**: the resolved batch size (`bgv_batched::available_slots`) is the hard ceiling on `N*N` — see §6. | `0` (automatic) |

`Params` is still the low-level knob set used internally by `bgv_batched::setup`,
but callers (the CLI, benchmarks, tests) no longer construct it by hand for
normal use. Instead, `bgv_batched::setup_from_profile(required_depth,
required_slots)` (`include/btc/bgv_batched_profile_setup.hpp`) looks up a
validated `(plaintext_modulus, ring_dimension, multiplicative_depth)` triple
from `config/bgv_batched_profiles.json` and fills in `Params` from that --
see [docs/bgv_parameter_profiles.md](bgv_parameter_profiles.md) for the full
mechanism and why guessing a single "safe" plaintext modulus per graph size
(this backend's original approach) was unreliable.

## 6. Packing layout

Full-matrix row-major packing: one ciphertext holds one complete `N x N`
matrix, using

```
slot(i, j) = i * N + j
```

Example (3x3):

```
A = [ a00 a01 a02 ]        packed slots:
    [ a10 a11 a12 ]   ->   [ a00 a01 a02 a10 a11 a12 a20 a21 a22 0 0 ... 0 ]
    [ a20 a21 a22 ]                ^-------------- N*N = 9 --------------^
```

Only the first `N*N` slots are matrix slots; everything from slot `N*N`
onward is zero-padded (`MakePackedPlaintext` zero-pads automatically, and a
precomputed **valid-region mask** — `[1]*N*N ++ [0]*(slots-N*N)` — is
reapplied after every broadcast in `bgv_batched_backend.hpp` so rotation
wrap-around can never leak nonzero content back into the matrix region from
the padding, or vice versa).

**Capacity check.** `BGVBatchedBackend`'s constructor verifies `N*N <=`
`bgv_batched::available_slots(ctx)` before doing anything else, and throws
`std::invalid_argument` (naming `N`, `N*N`, and the available slot count)
if the matrix does not fit. There is no silent fallback to scalar
encryption or to the legacy/unpacked backend — tiled/multi-ciphertext
packing for matrices that don't fit is explicitly out of scope for this
first version (see §12).

## 7. Rotation direction convention

`tests/test_bgv_batched_rotation.cpp` independently pins down OpenFHE's
`EvalRotate` sign convention (rather than assuming it) using a small
distinct-value vector `[0,1,...,7]`: **a positive rotation index is a LEFT
cyclic shift** — `EvalRotate(ct, offset)` produces `result[i] =
original[(i + offset) mod slots]`. `btc::bgv_batched::rotateToOffset(ctx,
ct, offset)` wraps this (documented at its definition in
`bgv_batched.hpp`), and every broadcast formula below is written directly in
terms of it so the sign convention only has to be derived once.

## 8. Packed Boolean matrix multiplication

`BGVBatchedBackend::matmul` computes, for `C = A ⊙_B B`:

```
C[i,j] = OR_k ( A[i,k] AND B[k,j] )
```

by constructing, for each inner-product index `k` in `[0, N)`, two packed
broadcast ciphertexts covering **every** output position `(i,j)`
simultaneously:

```
L_k[i,j] = A[i,k]      (column k of A, broadcast horizontally across each row)
Q_k[i,j] = B[k,j]      (row k of B, broadcast vertically down each column)
T_k      = L_k AND Q_k  =>  T_k[i,j] = A[i,k] * B[k,j]  for every (i,j) at once
```

then OR-reducing `T_0 .. T_{N-1}` in a **balanced binary tree**
(`BGVBatchedBackend`'s private `OrReduceTree`, mirroring
`BGVBackend::or_reduce_tree`) rather than a linear chain.

### Constructing `L_k` (horizontal column broadcast)

1. Mask: `columnK = EvalMult(A, columnMask[k])` — ciphertext-**plaintext**
   multiplication (`bgv_batched::mask_mult`), never ciphertext-ciphertext.
   `columnMask[k][i,j] = 1` iff `j == k`, so `columnK`'s only nonzero
   content per row `i` sits at flat slot `i*N+k`.
2. Broadcast: for each `j` in `[0, N)`, rotate `columnK` so slot `i*N+k`'s
   value lands at slot `i*N+j`, and sum the results. Using the left-shift
   convention from §7 (`rotated[idx] = orig[idx + offset]`), landing content
   from `i*N+k` at `i*N+j` requires `offset = k - j`:

   ```
   L_k = sum_{j=0}^{N-1} rotateToOffset(columnK, k - j)
   ```

   `j == k` contributes the unrotated `columnK` term directly (no rotation
   call for offset zero, per the task's explicit requirement).

### Constructing `Q_k` (vertical row broadcast)

Symmetric, with row masks and rotations by multiples of `N`:

1. Mask: `rowK = EvalMult(B, rowMask[k])`, `rowMask[k][i,j] = 1` iff
   `i == k`; nonzero content sits at flat slots `k*N+j`.
2. Broadcast: `offset = (k - i) * N` moves content from `k*N+j` to `i*N+j`:

   ```
   Q_k = sum_{i=0}^{N-1} rotateToOffset(rowK, (k - i) * N)
   ```

   `i == k` contributes the unrotated `rowK` term directly.

Both broadcasts reapply the valid-region mask after summation (§6) so
padding slots stay exactly zero even after the rotation-and-sum.

## 9. Why ordinary summation cannot replace Boolean OR

Identical reasoning to the unpacked BGV backend (`docs/bgv.md` §6): if
`C[i][j]` is computed as `sum_k(A[i,k] * B[k,j])` instead of `OR_k(...)`,
two witnesses (`k1 != k2` both contributing `1`) would add to `2`, not
collapse to Boolean `1`. `tests/test_bgv_batched_matmul.cpp`'s
`multiple_witnesses` case verifies the balanced OR-tree correctly collapses
two (and, in a denser case, `N`) converging witnesses to exactly `1`.

## 10. Expected operation structure and multiplicative depth

Per matmul round of two `N x N` matrices, in the straightforward
(non-baby-step-giant-step) implementation used here:

- `N` ciphertext-ciphertext multiplications for the `T_k` AND terms,
- `N - 1` ciphertext-ciphertext multiplications for the balanced OR
  reduction,
- ciphertext-**plaintext** multiplications for the `2N` row/column masks
  (`N` column masks + `N` row masks, one mask-multiply each),
- approximately `2N(N-1)` rotations (each broadcast loops over `N-1`
  nonzero offsets, and there are `2N` broadcasts — `N` column + `N` row —
  per matmul).

So `M_mult_per_round = 2N - 1` (matching the task's formula), and the
multiplicative depth per round is:

```
depth(matmul) = 1 (AND) + ceil(log2 N) (OR-tree)
```

implemented as `btc::BGVBatchedMatmulDepth(n)` — **identical in shape** to
`BGVMatmulDepth` (`bgv_backend.hpp`): packing changes ciphertext count and
rotation count, not circuit depth, since every AND/OR here is still exactly
one `EvalMult` regardless of how many slots it acts on in parallel.

For the full transitive closure (`sum_powers_recursive`'s repeated
squaring, `ceil(log2 T)` recursion levels, each contributing
`depth(matmul) + 1`):

```
depth(TC) = ceil(log2(max(1, T))) * (2 + ceil(log2 N))
```

implemented as `btc::EstimateBGVBatchedDepth(numberOfNodes, T)`. This
estimate is exact for this circuit (not merely an upper bound), for the
same reason as the unpacked backend: no other multiplications occur.
`examples/bgv_batched_closure.cpp` and `benchmarks/bench_bgv_batched.cpp`
compute and print both the estimated and configured depth, warning (not
silently proceeding with wrong output) when the configured depth is too
low.

## 11. Backend selection

This repository selects backends at **build/binary granularity** (a CMake
option per backend, compiled into a separate example/test/benchmark binary
per backend — see the "Build" section of the root `README.md`), not via a
single shared runtime `--backend` dispatcher. `bgv_batched_closure` follows
that existing convention (matching `tfhe_closure`/`dghv_closure`/
`bgv_closure`), and additionally accepts an explicit `--backend=bgv-batched`
flag: since this binary only ever implements the bgv-batched backend, any
other `--backend` value is rejected with a clear error message rather than
silently ignored, satisfying the task's requirement for an explicit,
validated backend-selection flag without inventing a cross-backend runtime
dispatcher this codebase does not otherwise have. The legacy/unpacked
backends (`tfhe_closure`, `dghv_closure`, `bgv_closure`) are unmodified and
remain the default build targets; `bgv-batched` must be explicitly enabled
via `-DTC_WITH_BGV_BATCHED=ON` and explicitly run.

## 12. Current limitations

- **No tiled/multi-ciphertext packing**: matrices with `N*N` larger than the
  ring's available slot count are rejected outright (§6) rather than
  silently falling back to scalar encryption or splitting across multiple
  ciphertexts. Generating a larger profile (`tools/generate_bgv_batched_profiles
  --nodes=<N>`, see `docs/bgv_parameter_profiles.md`) raises the ceiling but
  does not remove it.
- **No baby-step-giant-step optimization**: rotation count is
  `O(N)` per broadcast / `O(N^2)` per matmul round, the "straightforward"
  complexity the task explicitly calls for in this first version, not the
  asymptotically better BSGS matrix-multiplication techniques used in some
  homomorphic ML literature.
- **No bootstrapping**: like the unpacked BGV and DGHV backends, this is a
  leveled scheme — `multiplicative_depth` is a hard ceiling fixed before key
  generation, and must be sized via `EstimateBGVBatchedDepth` (or larger).
- **Performance is not optimized beyond what packing itself buys**: masks
  and rotation keys are precomputed once and reused (never rebuilt inside a
  matmul or TC round), but no further algorithmic optimization (BSGS,
  diagonal encoding, hoisted rotations) has been applied. See
  `bench_bgv_batched.cpp` for actual timings, and compare directly against
  `bench_bgv.cpp` (unpacked) on the same `--graph` to see what packing
  itself buys before further optimization.
- **No independent security audit** beyond OpenFHE's own
  `HEStd_128_classic` table, same caveat as the unpacked BGV backend.
