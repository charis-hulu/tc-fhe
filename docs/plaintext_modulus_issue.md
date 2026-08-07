# Problem: BGV plaintext modulus vs. ring dimension incompatibility

## Context

This repository implements a bounded transitive-closure (reachability)
algorithm using OpenFHE's BGV scheme (`CryptoContextBGVRNS`), with a
SIMD-batched backend (`include/btc/bgv_batched.hpp` /
`include/btc/bgv_batched_backend.hpp`) that packs an entire N x N Boolean
adjacency matrix into one ciphertext.

## The problem

Running the batched backend on an 8-node graph fails:

```
./build-bgv-batched/examples/bgv_batched_closure --graph=data/graph_N8.txt
...
Estimated multiplicative depth: 21
Generating BGV crypto context and keys...
Ring dimension: 65536
...
terminate called after throwing an instance of 'lbcrypto::OpenFHEException'
  what():  .../nbtheory-impl.h:l.191:RootOfUnity<...>(): Please provide a
  primeModulus(q) and a cyclotomic number(m) satisfying the condition:
  (q-1)/m is an integer. The values of primeModulus = 65537 and m = 131072
  do not satisfy this condition
```

The same code works fine for a 4-node graph (ring dimension 32768).

### Root cause

Three distinct parameters are involved, and it's easy to conflate them:

| Symbol | Name | Role |
|---|---|---|
| `n` | ring dimension | degree of the polynomial ring; determines SIMD slot count |
| `q` | ciphertext modulus (RNS) | modulus bounding ciphertext coefficients; auto-derived by OpenFHE from depth + security level (not manually set anywhere in this codebase) |
| `t` | **plaintext modulus** | modulus for plaintext integers; **must be chosen manually** for BGV/BFV |

The default plaintext modulus in this codebase is `t = 65537 = 2^16 + 1` (a
Fermat prime). For BGV's packed (SIMD) encoding to work, OpenFHE requires an
NTT root of unity, which needs:

```
(t - 1) mod m == 0        where m = 2 * n  (for power-of-two cyclotomics)
```

Since `t - 1 = 65536 = 2^16`, this constant only has valid divisors `m` up
to `2^16` -- i.e. it only supports ring dimensions up to `n = 32768`.

Separately, **ring dimension `n` is auto-selected by OpenFHE** from
`multiplicative_depth` + `security_level`, with no awareness of what
plaintext modulus the caller intends to use. For this backend:

```
depth(matmul)      = 3 + ceil(log2 N)                      -- see docs/bgv_batched.md §10
depth(TC)          = ceil(log2 T) * (depth(matmul) + 1)
```

For N=4, T=4: depth=12 -> OpenFHE picks n=32768 (fits under 65537's limit).
For N=8, T=8: depth=21 -> OpenFHE picks n=65536 (exceeds 65537's limit) ->
crash at context-generation time, not a graceful fallback.

This is a genuine "chicken-and-egg" dependency: ring dimension depends on
depth (known in advance) and security level (auto), but plaintext-modulus
compatibility depends on whatever ring dimension gets chosen -- and OpenFHE
does not solve this jointly. Confirmed against OpenFHE's own source
(`src/pke/lib/scheme/bgvrns/bgvrns-parametergeneration.cpp`,
`gen-cryptocontext-params-defaults.h`: default `plaintextModulus = 0`,
which is rejected with `OPENFHE_THROW("plaintextModulus cannot be zero.")`)
and confirmed by an OpenFHE maintainer (yspolyakov) on the official forum,
in response to a user asking this exact question ("Automaticly find a good
Plaintext Modulus", openfhe.discourse.group/t/automaticly-find-a-good-plaintext-modulus/312):

> "No, this is the recommended approach. The plaintext modulus depends on
> the application."

There is no bit-size-based or automatic plaintext-modulus selection mode in
`GenCryptoContext`/`ParamsGenBGVRNS`. Every official OpenFHE BGV example
hardcodes a specific NTT-friendly prime (e.g. `65537`, `536903681`).

### Current workaround in this repo

`tests/test_bgv_batched_tc.cpp`'s N=8 test case manually overrides the
plaintext modulus to `786433 = 3*2^18 + 1` (an NTT-friendly prime with a
larger power-of-two factor in `t-1`), chosen by trial to be "big enough."
This works, but is ad hoc: a different N/T combination could again exceed
what `786433` supports, and the user has to manually pick yet another
prime, with no principled way to know in advance what's "big enough."

## Proposed solution

OpenFHE exposes a general-purpose NTT-friendly-prime search utility,
`FirstPrime<IntType>(nBits, m)` (`src/core/include/math/nbtheory.h`), which
finds the smallest prime `t` such that:

1. `t` has at least `nBits` bits, and
2. `(t - 1) mod m == 0`.

The same OpenFHE maintainer's recommended pattern for this exact problem is
to call `FirstPrime(nBits, m)` with `m` set deliberately large (e.g.
`131072 = 2^17`) -- larger than `2 * n` for any ring dimension OpenFHE would
realistically select for this circuit -- so the resulting prime is
guaranteed NTT-compatible with whatever ring dimension gets auto-selected
afterward, regardless of N.

Concretely, proposed change in `btc::bgv_batched::setup` (or a helper
called before it):

```cpp
// Only when the caller hasn't explicitly overridden plaintext_modulus.
// m is chosen deliberately larger than 2 * (any realistic ring dimension
// for this circuit), so the resulting prime is NTT-compatible regardless
// of which ring dimension OpenFHE ends up selecting from depth+security.
constexpr uint64_t kSafeCyclotomicOrder = 131072; // 2^17
uint32_t nBits = /* enough bits for this circuit's transient values, e.g. ~20 */;
PlaintextModulus t = lbcrypto::FirstPrime<lbcrypto::NativeInteger>(nBits, kSafeCyclotomicOrder)
                         .ConvertToInt();
params.SetPlaintextModulus(t);
```

`nBits` itself would be derived from the circuit's maximum transient value
(for this Boolean circuit, values never exceed roughly `N` before being
reduced by the OR gate's `a+b-ab` identity), not from anything related to
ring dimension.

### Why this is believed to be correct

- It removes the need for the CLI user (or test author) to manually guess
  and hardcode a plaintext modulus per graph size.
- It is the officially recommended workaround from an OpenFHE core
  maintainer for precisely this failure mode, not a novel technique.
- It keeps `SetRingDim` untouched (still automatic, still driven by
  depth + `HEStd_128_classic`), so the security-level guarantee OpenFHE
  provides is preserved -- unlike the alternative of manually pinning
  `SetRingDim` and switching to `HEStd_NotSet` (which the `nexus.cpp`
  example in the `openfhe-statistics` repo does, but only to match another
  paper's fixed parameters, not as a general solution).

### Known open questions / possible weaknesses (please critique)

1. **Choice of `m = 131072`.** Is this large enough for all N this project
   might reasonably want to support, or should it be computed dynamically
   from an upper bound on ring dimension for the given depth (e.g. via
   OpenFHE's own security-level tables), rather than a hardcoded constant?
   Is there a risk `FirstPrime` becomes slow or returns a much larger prime
   than necessary at this `m`?
2. **Does forcing `t` to satisfy `(t-1) mod 131072 == 0` over-constrain
   `t`** in a way that produces a needlessly large prime compared to what
   would be minimally required for the ring dimension OpenFHE will actually
   pick, causing unnecessary noise budget/ciphertext size cost?
3. **Interaction with `batch_size`.** This backend currently leaves
   `batch_size` at 0 (automatic) precisely to avoid a different variant of
   this same incompatibility (see `docs/bgv_batched.md` and this project's
   own test comments) -- does auto-picking `t` via `FirstPrime` change that
   reasoning at all?
4. **Correctness of `nBits`.** Is deriving `nBits` from "max transient value
   ~ N" sufcient, or does the OR-tree / repeated squaring in
   `sum_powers_recursive` allow larger transient intermediate values that
   would require a larger `t` regardless of NTT compatibility?
5. Is there a simpler/more idiomatic OpenFHE-native way to achieve the same
   result that this analysis is missing?

---

## Second occurrence: the bgv-rowcol backend (row/column-packed, OpenMP)

The identical failure mode recurred while building a third backend,
`include/btc/bgv_rowcol_backend.hpp` (see `docs/bgv_rowcol.md`) — same root
cause as above, different trigger size. There, `N=4, T=4` (not `N=8, T=8`)
was enough to break the default `t=65537`:

```
BGVRowColMatmulDepth(4) = 2 + 2*ceil(log2 4) = 6
EstimateBGVRowColDepth(4, 4) = 2 * (6 + 1) = 14   -- OpenFHE picks n=65536
65536 is NOT <= 32768 (65537's compatibility limit) -> same crash
```

This backend's depth grows **roughly twice as fast in N** as the packed
whole-matrix backend's (`bgv_batched_backend.hpp`), because its `SumOR`
reduction needs an extra ciphertext-plaintext re-mask per rotation step on
top of the OR itself (see `docs/bgv_rowcol.md` §3-4) — so it runs into this
same wall at smaller N/T than the original backend did.

**Fix applied (not the principled one below):** reused `t = 786433`, the
same prime the original N=8 workaround above already validated, rather than
deriving a new one. This isn't a coincidence-free choice — `786433 = 3*2^18
+ 1`, so `t - 1 = 786432 = 2^18 * 3` is divisible by `2*n` for *any*
power-of-two `n` up to `131072`, which happens to cover every ring
dimension this backend needed for `N <= 8`. It was picked because it was
already sitting in this codebase, proven to work, not because it was
independently derived as correct or minimal for this circuit.

**Directly answering "is 786433 the only choice, can't we use something
smaller?": no.** Any prime `t` with `(t-1) mod (2*n) == 0` works for a given
ring dimension `n`; there are infinitely many, and smaller ones exist. The
catch is that a smaller `t` is only NTT-compatible up to a smaller ceiling
on `n` (e.g. the default `65537 = 2^16+1` only reaches `n <= 32768`, which
is exactly why it broke here at `n=65536`). `786433` isn't special beyond
having a conveniently large power-of-two factor in `t-1` (`2^18`), which
buys compatibility across a wide range of `n` in one shot -- avoiding
having to pick a *different*, smaller-but-narrower-range prime for every
distinct `(N, T)` combination tested. It is a reused shortcut, not a proof
of sufficiency.

**Where this will break again:** for `N` large enough that this backend's
faster-growing depth pushes the auto-selected ring dimension past `131072`
(plausibly around `N=16` or `N=32`, untested), `786433` fails the exact same
way `65537` did at `N=4`. The proposed `FirstPrime`-based fix below applies
equally to this backend and was not implemented here either -- this backend
has no `setup_from_profile`-equivalent (its `batch_size == N` requirement is
incompatible with the existing profile database's `batch_size == 0`
convention, see `docs/bgv_rowcol.md` §7), so it currently has *no* principled
parameter-selection path at all, only this one hardcoded, empirically-tested-
for-N<=8 prime.

## Instructions for ChatGPT

Please:

1. Independently analyze the problem described above (the BGV plaintext
   modulus vs. ring dimension NTT-compatibility issue, including its second
   occurrence in the bgv-rowcol backend) -- confirm or correct the
   root-cause explanation.
2. Propose your own solution (it is fine if it ends up being the same as
   the one proposed above).
3. Critique the proposed `FirstPrime`-based solution above specifically,
   including the five open questions listed at the end. Be concrete about
   any correctness, security, or performance concerns.
