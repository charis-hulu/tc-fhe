# tc-fhe

Bounded transitive closure over the Boolean semiring, with a Zama TFHE-rs encrypted backend.

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
src/
  btc.cpp             — PlaintextBackend::matmul, floyd_warshall
tests/
  test_matmul.cpp
  test_algorithms.cpp
  test_transitive_closure.cpp
benchmarks/
  bench_closure.cpp   — plaintext timing
  bench_tfhe.cpp      — FHE encrypt → BTC → decrypt
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

## Roadmap

- [x] Plaintext BTC with PlaintextBackend
- [x] TFHE-rs boolean CPU backend (TFHEBackend)
- [ ] TFHE-rs GPU backend (integer API, CUDA)
