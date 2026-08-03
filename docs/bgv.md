# BGV Backend (OpenFHE)

This document explains the BGV backend in this repository:
`include/btc/bgv.hpp` (a thin wrapper around OpenFHE's BGV-RNS scheme) and
`include/btc/bgv_backend.hpp` (the adapter that plugs it into
`bounded_transitive_closure`, mirroring `dghv_backend.hpp`).

## 1. What BGV is being used for

BGV (Brakerski-Gentry-Vaikuntanathan, "Fully Homomorphic Encryption without
Bootstrapping", ITCS 2012) is a leveled, exact-integer FHE scheme: plaintexts
are integers mod a plaintext modulus `t`, and homomorphic addition/
multiplication are exact modular ring operations (no approximation, unlike
CKKS). This project uses OpenFHE's RNS variant of BGV (`CryptoContextBGVRNS`)
purely as an alternative encrypted backend for the same bounded
transitive-closure computation the TFHE and DGHV backends already perform:
Boolean values `{0,1}` are encrypted as BGV plaintext integers, and AND/OR
gates are built from the same `{0,1}`-ring identities DGHV uses (`a*b` for
AND, `a+b-a*b` for OR). Unlike DGHV (a hand-rolled, secret-key-only, toy
construction — see `docs/dghv.md`), BGV here is a full library implementation
with proper RNS modulus-chain management, so it is both faster and closer to
a production-grade leveled scheme.

## 2. How to install / locate OpenFHE

This backend needs an installed build of [OpenFHE](https://github.com/openfheorg/openfhe-development)
(v1.x) with the standard `PKE` module. From an OpenFHE source checkout:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /path/to/openfhe-install
```

If OpenFHE is installed to a non-system prefix, point CMake at it via
`CMAKE_PREFIX_PATH` or `OpenFHE_DIR` when configuring this project (see §3).

## 3. How to configure CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV=ON \
      -DCMAKE_PREFIX_PATH=/path/to/openfhe-install
cmake --build build
```

`TC_WITH_BGV` (default `OFF`) is an independent option, exactly like
`TC_WITH_TFHE` and `TC_WITH_DGHV` — enable any subset of the three. When
`TC_WITH_BGV` is `OFF`, `bgv_closure`/`bench_bgv`/`test_bgv` still build
(each has a `#else` stub `main()` that prints a message and exits 0) but
contain no OpenFHE code and link no OpenFHE libraries, so builds without
OpenFHE installed are unaffected. When `TC_WITH_BGV=ON` but OpenFHE cannot be
found, `find_package(OpenFHE REQUIRED)` fails the CMake configure step with a
clear "could not find OpenFHE" error rather than a confusing build-time link
failure.

## 4. How to run the BGV backend

```bash
# Default graph (built-in 4-node chain 0->1->2->3), default parameters:
./build/examples/bgv_closure

# Custom graph, T, and BGV parameters (all optional):
./build/examples/bgv_closure \
    --graph=data/graph_N8.txt --T=5 \
    --bgv-plaintext-modulus=65537 --bgv-depth=20

./build/benchmarks/bench_bgv --graph=graph_N4 --bgv-plaintext-modulus=65537
ctest --test-dir build --output-on-failure -R test_bgv
```

## 5. Main BGV parameters

Exposed via `btc::bgv::Params` (`include/btc/bgv.hpp`):

| Field | Meaning | Default |
|---|---|---|
| `plaintext_modulus` | The modulus `t` plaintext integers live in. Must be `> 2` so `{0,1}` AND/OR/XOR results never wrap. `65537` is a common NTT-friendly (batching-capable) prime used throughout OpenFHE's own BGV examples. | `65537` |
| `multiplicative_depth` | Hard ceiling on sequential `EvalMult` calls the crypto context supports (baked into the RNS modulus chain at `GenCryptoContext` time). Must be `>=` the circuit's actual depth — see §8. | `1` (caller must override, see `EstimateBGVDepth`) |
| `security_level` | OpenFHE's standard security-parameter table lookup. | `HEStd_128_classic` |
| `ring_dimension` | Polynomial ring dimension. `0` means "let OpenFHE choose automatically" from the other parameters. | `0` (automatic) |
| `batch_size` | SIMD plaintext-slot count. `0` means automatic. Not currently used for packing (see §9) — only affects how many plaintext slots exist. | `0` (automatic) |

All other OpenFHE-internal parameters (RNS tower count/sizes, NTT prime
selection, key-switching technique, etc.) are left to OpenFHE's automatic
selection, exactly as the rest of this codebase leaves TFHE's low-level LWE/
GLWE parameters to `boolean_gen_keys_with_default_parameters` unless a
parameter file is explicitly loaded.

## 6. Why ordinary matrix multiplication mod 2 is incorrect

Boolean reachability requires `C[i][j] = OR_k(A[i][k] AND B[k][j])`. If you
instead compute `C[i][j] = sum_k(A[i][k] * B[k][j]) mod 2` (ordinary integer
matrix multiplication reduced mod 2), two *distinct* paths through different
intermediate nodes `k1 != k2` both contributing a `1` term get **added**:
`1 + 1 = 2 = 0 mod 2` — the two paths cancel each other out, incorrectly
reporting the destination as unreachable. Boolean OR must instead compute
`1 OR 1 = 1`. This is precisely what `test_two_paths_same_vertex` in
`tests/test_bgv.cpp` checks: a graph with two length-2 paths converging on
the same vertex, verified to remain reachable under BGV.

## 7. The Boolean OR polynomial

Using the standard `{0,1}`-ring identities (identical to DGHV, see
`docs/dghv.md` §5):

```
AND(a, b) = a * b
OR(a, b)  = a + b - a*b
```

`bgv::encrypted_and` is `EvalMult(a, b)`. `bgv::encrypted_or` computes
`EvalAdd(a, b)`, `EvalMult(a, b)`, and subtracts the two
(`EvalSub(sum, product)`). Both cost exactly one `EvalMult`, i.e. one level
of multiplicative depth — the unit of cost the depth formula in §8 counts.
`bgv::encrypted_xor` (`a + b - 2ab`) is also implemented for parity with
`dghv.hpp`'s gate set, though the transitive-closure circuit itself only
needs AND/OR.

## 8. Expected multiplicative depth

Identical circuit-depth accounting to DGHV (`docs/dghv.md` §7) — the circuit
shape (`sum_powers_recursive`'s repeated squaring, `matmul`'s AND-then-
balanced-OR-tree) is backend-independent, only the per-multiplication
primitive differs:

**One matmul.** For an `N x N` adjacency matrix, `C[i][j] = OR_k(A[i][k] AND
B[k][j])` ANDs `N` pairs (1 level), then OR-reduces the `N` terms in a
**balanced binary tree** (`BGVBackend::or_reduce_tree`) rather than a linear
chain, so depth grows as `O(log N)` instead of `O(N)`:

```
depth(matmul) = 1 (AND) + ceil(log2 N) (OR-tree)
```

Implemented as `btc::BGVMatmulDepth(n)`.

**The full recursion.** `sum_powers_recursive` (`algorithms.hpp`) halves `T`
each level; per level it computes `P_new = matmul(P,P)` and
`S_new = S | matmul(P,S)`, i.e. one matmul followed by one more OR
(`EvalMult`). With `ceil(log2 T)` levels:

```
depth(sum_powers_recursive) = ceil(log2 T) * (depth(matmul) + 1)
                             = ceil(log2 T) * (2 + ceil(log2 N))
```

Implemented as `btc::EstimateBGVDepth(numberOfNodes, T)`. Since
`bounded_transitive_closure` is a thin wrapper around `sum_powers_recursive`,
this is also the depth of the whole encrypted computation, and — because
every AND/OR here costs exactly one `EvalMult` and nothing else multiplies —
the estimate is **exact** for this circuit, not merely a conservative upper
bound.

`examples/bgv_closure.cpp` and `benchmarks/bench_bgv.cpp` compute this
estimate, use it as `Params::multiplicative_depth` unless `--bgv-depth`
overrides it, and print both the estimated and configured depth. If the
configured depth is lower than the estimate, they print a clear warning
before running the (potentially expensive) computation, rather than silently
producing wrong ciphertexts — `bgv::setup` itself only requires
`multiplicative_depth >= 1`; the sufficiency check against a specific
circuit is the caller's responsibility, exactly as `dghv::require_depth` is
opt-in and separate from `dghv::validate_params`.

## 9. Current limitation: one ciphertext per matrix element

Like `DGHVBackend` and `TFHEBackend`, `BGVBackend` encrypts each Boolean
matrix entry as its own ciphertext (`N^2` ciphertexts per `N x N` matrix, `N`
AND-terms per dot product). This keeps the implementation simple, readable,
and directly comparable across all three backends. OpenFHE's BGV supports
SIMD slot packing (`SetBatchSize`, `EvalRotate`) that could pack many bits
into a single ciphertext and evaluate them in parallel — this is
deliberately **not** implemented in this first version, since the codebase
has no existing packing abstraction to build on and correctness/readability
were prioritized (see the "Future batching" comment on `BGVBackend::matmul`
in `bgv_backend.hpp`).

## 10. SIMD packing as future work

A follow-up optimization could pack each row of an adjacency matrix (or each
column of AND-terms in a dot product) into the slots of one ciphertext, and
replace the current per-element `EvalMult`/`EvalAdd` loop with a smaller
number of packed `EvalMult` + `EvalRotate`-based reduction calls, cutting
ciphertext count roughly by the batch size. This is out of scope here — see
`bgv_backend.hpp`'s "Future batching" comment for the specific insertion
point — and should be evaluated as a separate, purely performance-motivated
change once correctness of the unpacked version is established.

## 11. Build and run (full reference)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV=ON \
      -DCMAKE_PREFIX_PATH=/path/to/openfhe-install
cmake --build build
ctest --test-dir build --output-on-failure -R test_bgv

./build/examples/bgv_closure
./build/examples/bgv_closure --graph=data/graph_N8.txt --T=5 --bgv-depth=20
./build/benchmarks/bench_bgv --graph=graph_N4
```

`TC_WITH_BGV` is independent of `TC_WITH_TFHE` and `TC_WITH_DGHV` — enable
any subset.

## 12. Current limitations

- **No bootstrapping**: like DGHV, this backend cannot evaluate circuits
  deeper than `Params::multiplicative_depth` was set to. Doubling the graph
  size or `T` roughly doubles the required depth (§8); `multiplicative_depth`
  is never computed automatically by `bgv::setup` itself — callers must pass
  `btc::EstimateBGVDepth(N, T)` (or override manually) before calling it.
- **One ciphertext per matrix element** (§9): no SIMD batching yet, so
  ciphertext count and runtime scale with `N^2` per matrix, `N` per dot
  product, same as DGHV and TFHE.
- **No independent security audit of parameter choices beyond OpenFHE's own
  `HEStd_128_classic` table**: OpenFHE's security-level lookup is a
  well-established reference, but this project has not independently
  verified the resulting concrete parameters beyond trusting OpenFHE's own
  implementation.
- **Performance is not optimized**: matches this repository's stated
  priorities (correctness, readability, simple structure) over speed — see
  `bench_bgv.cpp` for actual timings.
