# DGHV Backend

This document explains the DGHV ("Fully Homomorphic Encryption over the
Integers", van Dijk/Gentry/Halevi/Vaikuntanathan, EUROCRYPT 2010) backend in
this repository: `include/btc/dghv.hpp` (the scheme) and
`include/btc/dghv_backend.hpp` (the adapter that plugs it into
`bounded_transitive_closure`).

## What this is / is not

**This is a simplified, secret-key (symmetric) educational variant of DGHV,
not the full public-key DGHV scheme, and it does not implement
bootstrapping.**

- Full DGHV encrypts by adding together a random subset of many
  public-key encryptions of zero, so that anyone (not just the key
  holder) can encrypt without knowing the secret key `p`. This backend
  instead samples `c = p*q + 2*r + m` directly, because the encryptor here
  always has `p` (see `dghv::encrypt_bit` in `dghv.hpp`). The ciphertext
  *form* is identical to full DGHV; only how it gets constructed differs.
- Full DGHV supports unbounded-depth computation via bootstrapping
  (homomorphically evaluating a "squashed" decryption circuit to refresh
  noise). This backend is **leveled**: it computes the multiplicative depth
  a circuit needs ahead of time and picks parameters large enough to survive
  that many multiplications without ever refreshing noise mid-computation.

Both simplifications are exactly what the task asked for: a minimal,
readable prototype that demonstrates correct homomorphic evaluation, with
the option to add the full public-key construction later if needed (not
done here, since it was not required for correctness/readability goals).

## 1. Parameter names (standard DGHV notation)

The DGHV paper (van Dijk et al., EUROCRYPT 2010, Section 2) names three
bit-length parameters:

- **eta** — bit length of the secret key `p`.
- **rho** — bit length (magnitude bound) of the fresh noise `r`.
- **gamma** — bit length of a full ciphertext / public sample
  `x = p*q + 2*r`. Note that **gamma is the bit length of the ciphertext,
  not of `q` directly** — since `c ≈ p*q` for large `q`, this means
  `gamma ≈ eta + (bit length of q)`.

This codebase exposes the three *independent* knobs as `eta_bits`,
`rho_bits`, and `q_bits` (the bit length of the random multiplier `q`,
i.e. the paper's "gamma − eta"), and **derives**:

```
gamma_bits = eta_bits + q_bits                      (0)
```

`gamma_bits` is stored on `Params` for reference/printing (so you can
compare directly against the paper's `gamma`), but it is never an
independent input — every function that samples `q` uses `q_bits`
directly, so `gamma_bits` can never disagree with `q_bits`. See
`dghv::make_params` in `dghv.hpp`, which every preset and `manual_params`
funnels through.

## 2. Ciphertext construction

For a secret odd integer `p` (the secret key, `eta_bits` bits) and a
plaintext bit `m in {0,1}`:

```
c = p*q + 2*r + m                                   (1)
```

- `q` is a large random integer, drawn uniformly from `[0, 2^q_bits)` —
  this is what a real attacker must fail to use to recover `p` from many
  ciphertexts (the "approximate GCD" hardness assumption DGHV relies on).
- `r` is small random noise, drawn from the signed range
  `(-2^rho_bits, 2^rho_bits]` — this is what gets "used up" by homomorphic
  operations and must not grow past `p/2` before decryption.

Implemented by `dghv::encrypt_bit` in `dghv.hpp`.

## 3. Decryption

```
m = [c]_p mod 2                                     (2)
```

where `[c]_p` denotes `c` reduced into the **centered** range `(-p/2, p/2]`
(as opposed to the usual `[0, p)` range) — implemented explicitly as
`dghv::centered_mod(c, p)`, since GMP's `mpz_class operator%` follows C++
truncating-division semantics and can return a negative remainder, which is
not what centered reduction means. Because `c = p*q + (2*r+m)`, reducing
into `[c]_p` cancels the `p*q` term and leaves exactly `2*r+m` *as long as*
`|2*r+m| < p/2`. Reducing that further mod 2 strips off the (even) `2*r`
term and leaves `m`. This is the correctness condition tracked throughout
the codebase: **decryption is correct iff the ciphertext's noise magnitude
stays below `p/2`, i.e. under `eta_bits - 1` bits.**

Implemented by `dghv::decrypt_bit` (and the shared `dghv::centered_mod`
helper it and `dghv::inspect_noise` both call).

## 4. Homomorphic addition and multiplication

Addition:

```
Enc(m1) + Enc(m2) = p*(q1+q2) + 2*(r1+r2) + (m1+m2)
```

decrypts to `(m1+m2) mod 2` — plaintext XOR. Noise adds (at most doubles in
magnitude, i.e. +1 bit): cheap.

Multiplication:

```
Enc(m1) * Enc(m2) = p*(p*q1*q2 + 2*r1*q2 + 2*r2*q1) + 2*(2*r1*r2) + m1*m2
```

decrypts to `(m1*m2) mod 2` — plaintext AND. This is the operation that
consumes multiplicative depth: the new noise term is roughly the *product*
of the input noise magnitudes, so noise bit-length roughly **doubles** per
multiplication rather than growing by a fixed amount.

Both are implemented in `dghv.hpp` as `dghv::add` / `dghv::mul`, operating
directly on `mpz_class` integers so that intermediate values are never
accidentally reduced with ordinary C++ boolean operators — every
homomorphic op is genuine big-integer arithmetic on the ciphertext.

## 5. How AND and OR are represented

Using the standard Boolean-ring identities for `Z_2`:

```
AND(a, b) = a * b
OR(a, b)  = a + b - a*b
```

`dghv::encrypted_and` is exactly `mul(a, b)`. `dghv::encrypted_or` computes
`add(a,b)` and `mul(a,b)` and subtracts. **Both cost exactly one
multiplication**, i.e. one level of multiplicative depth — this is the unit
of cost the depth calculations below count.

`dghv::encrypted_xor` (`a + b - 2ab`) is also implemented, since the README
lists it as a component, even though the transitive-closure circuit itself
only needs AND/OR.

## 6. Why bootstrapping is not used

Bootstrapping requires homomorphically evaluating decryption equation (2)
using a "squashed" version of the secret key spread across a large public
key — a construction that would roughly double the size of this codebase
and obscure the arithmetic this document is trying to explain clearly.
Instead, since the transitive-closure circuit's multiplicative depth is
known in advance (see below), we pick DGHV parameters large enough that
noise never approaches `p/2` for that specific circuit. This is a standard
"leveled FHE" trade-off: correct for a bounded, known depth, at the cost of
needing bigger keys/ciphertexts as depth grows, rather than paying a fixed
(large) bootstrapping cost per gate to support unbounded depth.

## 7. How multiplicative depth is determined

`include/btc/dghv_backend.hpp` derives the depth directly from the circuit
`bounded_transitive_closure` actually evaluates:

**One matmul.** For an `N x N` adjacency matrix, `C[i][j] = OR_k(A[i][k] AND
B[k][j])` ANDs `N` pairs (1 level) then OR-reduces the `N` terms. `DGHVBackend`
does this OR-reduction as a **balanced binary tree**
(`DGHVBackend::or_reduce_tree`), not the linear/serial accumulation
`TFHEBackend` uses — TFHE bootstraps after every gate so its "depth" is not a
resource, but DGHV noise compounds with every sequential multiplication, so a
serial chain of `N-1` ORs would need parameters scaling with `N` instead of
`log N`. A balanced tree keeps this to `ceil(log2 N)` levels:

```
depth(matmul) = 1 (AND) + ceil(log2 N) (OR-tree)
```

Implemented as `btc::matmul_depth(n)`.

**The full recursion.** `sum_powers_recursive` (`algorithms.hpp`) halves `T`
each level; per level it computes `P_new = matmul(P,P)` and
`S_new = S | matmul(P,S)`, i.e. one matmul followed by one more OR. With
`ceil(log2 T)` levels:

```
depth(sum_powers_recursive) = ceil(log2 T) * (depth(matmul) + 1)
                             = ceil(log2 T) * (2 + ceil(log2 N))
```

Implemented as `btc::transitive_closure_depth(n, T)`. Since
`bounded_transitive_closure` is a thin wrapper around
`sum_powers_recursive`, this is also the depth of the whole encrypted
computation. For the common choice `T = N` (full closure, any directed
graph — see the correctness note in `algorithms.hpp`), this is
`O(log N * log N) = O(log^2 N)`, confirming the README's assumption of
"approximately logarithmic" depth in the graph size.

## 8. Connection to the transitive-closure algorithm

`DGHVBackend` in `dghv_backend.hpp` satisfies the same `Backend` concept
`PlaintextBackend` and `TFHEBackend` implement (see `backend.hpp`):
`matmul`, `or_mat`, `copy`, `dim`, plus static `encrypt`/`decrypt` helpers.
No changes to `algorithms.hpp` were needed — `bounded_transitive_closure<Backend>`
and `sum_powers_recursive<Backend>` are called identically for plaintext,
TFHE, and DGHV:

```cpp
// eta_bits/rho_bits/q_bits are supplied by the caller -- e.g. from your own
// constants or from command-line arguments (see examples/dghv_closure.cpp)
// -- not computed from N or T. gamma_bits = eta_bits + q_bits is derived
// automatically. transitive_closure_depth is only used to check whether
// your chosen eta_bits is actually big enough.
auto params = dghv::manual_params(/*eta_bits=*/9000, /*rho_bits=*/32, /*q_bits=*/8192, /*max_depth=*/0);
dghv::validate_params(params); // structural checks: eta_bits > rho_bits, q_bits > 0, ...
dghv::require_depth(params, btc::transitive_closure_depth(N, T)); // throws if insufficient
auto key = dghv::keygen(params, rng);

btc::DGHVBackend backend;
auto enc_A = btc::DGHVBackend::encrypt(plain_A, key, params, rng);
auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
auto plain_S = btc::DGHVBackend::decrypt(enc_S, key);
```

## 9. Build and run

Requires GMP with C++ bindings (`gmp.h`, `gmpxx.h`, `libgmp`, `libgmpxx`;
on Debian/Ubuntu: `apt install libgmp-dev`).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_WITH_DGHV=ON
cmake --build build
ctest --test-dir build --output-on-failure -R test_dghv

# Default graph (built-in 4-node chain, depth 8) with defaults sized for it:
./build/examples/dghv_closure

# Your own graph and parameters, all supplied manually. Larger --graph/--T
# need a larger --eta-bits -- run once, read the printed "required
# multiplicative depth" and "max depth supported", and raise --eta-bits
# until the WARNING disappears and "Matches plaintext result: YES".
./build/examples/dghv_closure --graph=data/graph_N8.txt --T=5 \
    --eta-bits=20000 --rho-bits=64 --q-bits=40000

# bench_dghv's --tier picks a fixed eta_bits/rho_bits/q_bits triple (see
# table below); it is NOT resized for --graph, so check the printed
# depth/eta_bits and override with --eta-bits/--rho-bits/--q-bits if the
# tier's fixed eta_bits is not big enough:
./build/benchmarks/bench_dghv --graph=graph_N4 --tier=experiment --eta-bits=9000
./build/benchmarks/bench_dghv --graph=graph_N8 --tier=toy --eta-bits=200000 --rho-bits=32
```

`TC_WITH_DGHV` is independent of `TC_WITH_TFHE` — enable either, both, or
neither. `tests/test_dghv.cpp` runs a plaintext/TFHE/DGHV three-way
comparison only when both are enabled; otherwise it compares plaintext and
DGHV only.

## 10. Which parameters are toy parameters

`dghv::Params` has a `tier` field that is purely descriptive (never used in
arithmetic) so no parameter set can be silently mistaken for a security
claim. **`eta_bits`, `rho_bits`, and `q_bits` are always plain numbers
supplied by the caller — nothing in this codebase derives `eta_bits` from
the graph size, `T`, or the circuit depth. `gamma_bits` is always derived
as `eta_bits + q_bits` (equation 0) and is never an independent input.**
Four ways to get a `Params`, all in `dghv.hpp`:

| Function | `rho_bits` | `eta_bits` | `q_bits` | `gamma_bits` (derived) | Meaning |
|---|---|---|---|---|---|
| `toy_params(max_depth)` | 16 | 128 (fixed) | 384 (fixed) | 512 | **INSECURE.** Fast (milliseconds per op), but eta_bits=128 gives no real cryptographic security. `max_depth` is stored for reference only — it does **not** change eta_bits. |
| `experiment_params(max_depth)` | 32 | 1024 (fixed) | 3072 (fixed) | 4096 | **NOT a security claim.** Larger integers than toy_params for a more representative noise-growth curve, but still far short of real hardness margins. `max_depth` again does not affect eta_bits. |
| `secure_target_params(max_depth)` | 128 | 32768 (fixed) | 294912 (fixed) | 327680 | An **estimate** aiming at ~128-bit security using the DGHV paper's scaling rule (`eta = O(lambda^2)`, `rho = lambda`). Not independently vetted — a documented starting point, not a certified-secure parameter set. |
| `manual_params(eta_bits, rho_bits, q_bits, max_depth)` | caller | caller | caller | derived | Every independent field supplied directly — use this (or override individual fields of a preset, as `examples/dghv_closure.cpp` and `benchmarks/bench_dghv.cpp` do via `--eta-bits`/`--rho-bits`/`--q-bits`) when you want full manual control. |

Because noise grows **exponentially** with multiplicative depth
(`noise_bits(depth) ≈ rho_bits * 2^depth`, §7), a fixed `eta_bits` that is
enough for a shallow circuit will not be enough for a deeper one —
increasing the graph size or `T` can require a much larger `eta_bits`.
This library does not guess that for you: **you must pick `eta_bits`
yourself**, and check it with `require_depth`/`max_supported_depth` (below)
before trusting the result. `examples/dghv_closure.cpp` and
`benchmarks/bench_dghv.cpp` both print the computed circuit depth and the
max depth your chosen `eta_bits` supports, and warn (without aborting) if
`eta_bits` is too small, so you can see the effect of under-sizing it
deliberately.

Every preset function documents its own tier inline (`Params::note`) so the
caveat travels with the parameters, not just this file.

`dghv::validate_params(params)` checks structural preconditions --
`eta_bits > rho_bits`, `q_bits > 0`, `rho_bits > 0`, and
`gamma_bits == eta_bits + q_bits` -- and throws `std::invalid_argument` if
any is violated. `dghv::keygen` calls it automatically; call it yourself
right after constructing a `Params` (especially a hand-built one) to catch
mistakes before spending time on key generation.

`dghv::require_depth(params, required_depth)` throws if the circuit's
multiplicative depth exceeds what `params.eta_bits` can support
(`max_supported_depth`) — call it after choosing `eta_bits` yourself so a
too-small choice fails loudly at setup time rather than surfacing as a
silently wrong decryption later.

## 11. Current limitations

- **No bootstrapping**: this backend cannot evaluate circuits deeper than
  the parameters were sized for. Doubling the graph size or the number of
  hops `T` roughly doubles the required depth (see §7), and since noise
  grows as `~rho_bits * 2^depth` (`dghv::noise_bits_after_depth`), a fixed
  `eta_bits` chosen for a shallow circuit will not survive a deeper one.
  **`eta_bits` is never computed automatically** — you must pick it
  yourself, larger for deeper circuits, and verify with `require_depth`.
- **No public-key encryption**: only the party holding the secret key can
  encrypt (see "What this is / is not" above). A full public-key extension
  is future work, not implemented here.
- **Ciphertexts are large and grow with depth**: each `mul` roughly doubles
  the bit-length of the operands (they are plain `mpz_class` integers with
  no modular reduction step, unlike lattice-based schemes), so ciphertexts
  at the top of a deep circuit are substantially larger than fresh ones.
  This is a correctness/readability trade-off, not a bug: adding a modular
  reduction step would complicate the ciphertext invariant this document
  relies on.
- **No independent security audit**: `secure_target_params` is a
  documented estimate following the original paper's asymptotic scaling,
  not a peer-reviewed or independently cryptanalyzed parameter set. Real
  DGHV deployments require careful parameter selection against the current
  approximate-GCD literature, which is out of scope for this prototype.
- **Performance is not optimized**: matches the rest of this repository's
  stated priorities (correctness, readability, simple structure) over
  speed — see `bench_dghv.cpp` for actual timings per tier.
