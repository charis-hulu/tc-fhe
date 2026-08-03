# tc-fhe

Bounded transitive closure over the Boolean semiring, with a Zama TFHE-rs encrypted backend,
a leveled DGHV (integer-based) encrypted backend, a leveled BGV (OpenFHE) encrypted backend,
and a SIMD-batched BGV (OpenFHE, full-matrix packing) encrypted backend.

## What it does

Computes S(T) = A¹ ∨ A² ∨ … ∨ Aᵀ (reachability up to T hops in a directed graph) using divide-and-conquer Boolean matrix multiplication — both in plaintext and fully homomorphically encrypted.

## Algorithm

| Case      | S(T)                     | P(T) = Aᵀ     |
|-----------|--------------------------|----------------|
| T = 1     | A                        | A              |
| T = 2h    | S(h) ∨ P(h)·S(h)        | P(h)·P(h)      |
| T odd     | S(T−1) ∨ P(T−1)·A       | P(T−1)·A       |

O(log T) Boolean matrix multiplications.

## Structure

```
include/btc/
  matrix.hpp          — BoolMatrix (dense, uint8_t)
  backend.hpp         — PlaintextBackend + Backend concept docs
  algorithms.hpp      — sum_powers_recursive<B>, bounded_transitive_closure<B>
  tfhe_backend.hpp    — TFHEBackend (TFHE-rs boolean CPU)
  dghv.hpp            — DGHV scheme: params, keygen, encrypt/decrypt, homomorphic ops
  dghv_backend.hpp    — DGHVBackend + multiplicative-depth calculation
  bgv.hpp             — BGV (OpenFHE BGVrns) context setup, encrypt/decrypt, homomorphic ops
  bgv_backend.hpp     — BGVBackend + multiplicative-depth estimation
  bgv_batched.hpp         — BGV batched: packed context setup, packed encrypt/decrypt, rotation keys
  bgv_batched_backend.hpp — BGVBatchedBackend: masks, broadcasts, packed Boolean matmul
  bgv_parameter_profile.hpp     — validated (depth, ring_dim, plaintext_modulus) profile database (load/save/select)
  bgv_batched_profile_setup.hpp — setup_from_profile: builds a context by looking up a profile
src/
  btc.cpp             — PlaintextBackend::matmul, floyd_warshall
tools/
  generate_bgv_batched_profiles.cpp — offline: search + validate + save BGV parameter profiles
config/
  bgv_batched_profiles.json — generated profile database (see docs/bgv_parameter_profiles.md)
tests/
  test_matmul.cpp
  test_algorithms.cpp
  test_transitive_closure.cpp
  test_dghv.cpp
  test_bgv.cpp
  test_bgv_batched_rotation.cpp — EvalRotate sign-convention verification
  test_bgv_batched_packing.cpp  — packing/mask/broadcast layout tests
  test_bgv_batched_matmul.cpp   — packed Boolean matmul vs plaintext oracle
  test_bgv_batched_tc.cpp       — full closure vs Floyd-Warshall
  test_bgv_parameter_profile.cpp — profile selection/validity + setup_from_profile end-to-end
benchmarks/
  bench_closure.cpp   — plaintext timing
  bench_tfhe.cpp      — FHE encrypt → BTC → decrypt
  bench_dghv.cpp      — DGHV encrypt → BTC → decrypt
  bench_bgv.cpp       — BGV encrypt → BTC → decrypt
  bench_bgv_batched.cpp — BGV batched (packed) encrypt → BTC → decrypt
docs/
  dghv.md             — DGHV construction, depth derivation, parameters, limitations
  bgv.md              — BGV construction, depth derivation, parameters, limitations
  bgv_batched.md      — BGV batched: packing layout, broadcast construction, limitations
  bgv_parameter_profiles.md — offline profile generation + runtime lookup mechanism
  plaintext_modulus_issue.md — background: why plaintext modulus can't be auto-selected by OpenFHE
```

## Build

### Plaintext only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/benchmarks/bench_closure
```

### With TFHE-rs (CPU boolean backend)

Requires [tfhe-rs](https://github.com/zama-ai/tfhe-rs) built with `make build_c_api`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_TFHE=ON \
      -DTFHE_ROOT=/path/to/tfhe-rs/target/release
cmake --build build
./build/benchmarks/bench_tfhe
```

### With DGHV (leveled, integer-based)

Requires GMP with C++ bindings (`libgmp-dev` on Debian/Ubuntu). See
[docs/dghv.md](docs/dghv.md) for the full construction, depth derivation,
and parameter tiers.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_DGHV=ON
cmake --build build
./build/examples/dghv_closure
./build/benchmarks/bench_dghv
```

### With BGV (OpenFHE)

Requires an installed build of [OpenFHE](https://github.com/openfheorg/openfhe-development) (v1.x, `PKE` module).
See [docs/bgv.md](docs/bgv.md) for the full construction, depth derivation,
parameters, and why ordinary matrix multiplication mod 2 is incorrect for
Boolean reachability.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV=ON \
      -DCMAKE_PREFIX_PATH=/path/to/openfhe-install
cmake --build build
./build/examples/bgv_closure
./build/examples/bgv_closure --graph=data/graph_N8.txt --T=5 --bgv-depth=20
./build/benchmarks/bench_bgv
```

### With BGV batched (OpenFHE, SIMD full-matrix packing)

A separate backend from plain BGV above: packs one entire `N x N` matrix
into a single ciphertext instead of one ciphertext per entry, exploiting BGV
SIMD slots for the Boolean matmul. Does not modify or depend on the
unpacked BGV backend — both remain independently selectable and
benchmarkable. See [docs/bgv_batched.md](docs/bgv_batched.md) for the full
construction, packing layout, rotation-direction verification, and current
limitations (no tiling: `N*N` must fit in one ciphertext's slots).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV_BATCHED=ON \
      -DCMAKE_PREFIX_PATH=/path/to/openfhe-install
cmake --build build

./build/examples/bgv_batched_closure --backend=bgv-batched
./build/examples/bgv_batched_closure --graph=data/graph_N8.txt --T=5
./build/benchmarks/bench_bgv_batched
```

Plaintext modulus and ring dimension are **not** passed by hand — they're
looked up automatically from a parameter profile database
(`config/bgv_batched_profiles.json`) for the required circuit depth/slots,
instead of guessing a single "safe" value per graph size (which is
unreliable — see [docs/plaintext_modulus_issue.md](docs/plaintext_modulus_issue.md)
for why). Full mechanism: [docs/bgv_parameter_profiles.md](docs/bgv_parameter_profiles.md).

#### Generating the parameter profile database

`config/bgv_batched_profiles.json` is checked in with profiles already
covering `N = 4, 8, 16, 32`. Running `bgv_batched_closure` or
`bench_bgv_batched` on a graph within that range needs no extra setup.

If you need a graph size the checked-in database doesn't cover, generate
(or extend) it with `tools/generate_bgv_batched_profiles`:

```bash
./build/tools/generate_bgv_batched_profiles \
    --nodes=4,8,16,32,64 \
    --output=config/bgv_batched_profiles.json
```

- `--nodes=<n1,n2,...>` — comma-separated node counts to generate profiles
  for (assumes `T = N`, the conservative bound this project uses
  throughout). This is an **offline, one-time** step: for each `N` it
  searches for a compatible ring dimension and plaintext modulus, builds a
  real crypto context, and runs an end-to-end encrypt/decrypt/multiply/
  rotate check before saving — so it can take a while (minutes) for larger
  `N`, but only ever needs to run once per size.
- `--output=<path>` — defaults to `config/bgv_batched_profiles.json`.
  Re-running against an existing file **adds** new profiles rather than
  overwriting it (profiles are deduplicated and sorted, so the file stays
  stable across regenerations).
- `--minimum-plaintext-modulus=<n>` — floor on the generated plaintext
  modulus (default `65537`, this project's original policy value).

If you run `bgv_batched_closure`/`bench_bgv_batched` for an `N` the
database doesn't cover, they fail with an actionable error naming the
missing depth/slot requirement and the exact `--nodes` value to generate,
rather than guessing or silently falling back to something incorrect.

## Roadmap

- [x] Plaintext BTC with PlaintextBackend
- [x] TFHE-rs boolean CPU backend (TFHEBackend)
- [x] DGHV leveled integer backend (DGHVBackend)
- [x] BGV leveled backend via OpenFHE (BGVBackend)
- [x] BGV batched (SIMD full-matrix packing) via OpenFHE (BGVBatchedBackend)
- [ ] TFHE-rs GPU backend (integer API, CUDA)
