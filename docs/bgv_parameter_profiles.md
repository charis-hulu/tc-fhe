# BGV Batched Parameter Profile Database

This document explains how the `bgv-batched` backend picks a BGV plaintext
modulus and ring dimension for a given circuit, replacing the earlier
ad hoc "guess a plaintext modulus, hope it's NTT-compatible" approach.
Background on *why* this is needed at all is in
[docs/plaintext_modulus_issue.md](plaintext_modulus_issue.md); this
document covers the actual mechanism.

## 1. The problem, briefly

BGV requires a plaintext modulus `t` that is NTT-compatible with whatever
ring dimension OpenFHE ends up selecting for a given multiplicative depth
and security level -- but that ring dimension is only known *after*
`GenCryptoContext` runs, and OpenFHE has no built-in way to co-select both
at once. Guessing a single "safe" `t` (this project previously hardcoded
`786433` for deeper circuits) is fragile: a large enough N/T can still
exceed what any hardcoded value supports.

## 2. The two-stage design

```
offline generation:
  requirements (depth, N*N slots)
    -> search ring dimension n (ascending powers of two)
    -> generate compatible plaintext modulus t via FirstPrime/NextPrime
    -> attempt GenCryptoContext(depth, t, n, HEStd_128_classic)
       - "ring dimension too small for security level" -> double n, retry
       - any other failure -> stop, report (do not silently retry)
    -> on success: validate NTT compatibility + end-to-end
       (encrypt/decrypt, one multiplication, one rotation)
    -> store only validated profiles, deduplicated and sorted

runtime:
  requirements (depth, N*N slots)
    -> load config/bgv_batched_profiles.json
    -> select the smallest validated profile that satisfies both
    -> build the crypto context with that profile's EXACT values
    -> re-validate against what OpenFHE actually produced
    -> run TC
```

The offline half runs once, ahead of time, and is the only place that ever
searches for parameters. The runtime half is a pure lookup -- no search, no
guessing, no `GenCryptoContext` retry loop on the hot path.

## 3. Offline generation

`tools/generate_bgv_batched_profiles.cpp`:

```bash
./build/tools/generate_bgv_batched_profiles \
    --nodes=4,8,16,32 \
    --output=config/bgv_batched_profiles.json
```

For each requested node count `N` (assuming `T = N`, the same conservative
bound used throughout this codebase -- see `algorithms.hpp`'s correctness
note):

1. `required_depth = btc::EstimateBGVBatchedDepth(N, T)` -- the existing,
   **unmodified** depth estimator (see `docs/bgv_batched.md` §10).
2. `required_slots = N * N` (overflow-checked).
3. Candidate ring dimension starts at `next_power_of_two(required_slots)`
   and doubles each time OpenFHE's security-level check rejects it.
4. `m = 2 * ring_dimension`; `t = FirstPrime<NativeInteger>(nBits, m)`,
   bumped via `NextPrime` if below `minimum_plaintext_modulus` (default
   `65537`, preserving this project's original policy floor).
5. `GenCryptoContext` is attempted with `SetMultiplicativeDepth`,
   `SetPlaintextModulus`, `SetRingDim`, and `SetSecurityLevel(HEStd_128_classic)`
   all set explicitly (never `HEStd_NotSet` -- OpenFHE still validates the
   pinned ring dimension against its own security tables and throws if it's
   insufficient; see `docs/plaintext_modulus_issue.md` for how this was
   confirmed against OpenFHE's source).
6. On success: re-validate `actual_ring_dimension == n`,
   `n >= required_slots`, `(t-1) % (2n) == 0`, then run an end-to-end check
   (pack, encrypt, decrypt, one ciphertext-ciphertext multiplication, one
   rotation-key-generation + rotation, decrypt and verify).
7. Only profiles that pass every check are saved.

Re-running the generator with new `--nodes` values loads the existing file
first and adds to it (profiles are deduplicated and sorted, so the file
stays stable and reproducible across regenerations).

## 4. Profile file format

`config/bgv_batched_profiles.json`:

```json
{
  "scheme": "BGVRNS",
  "security_level": "HEStd_128_classic",
  "batch_size": 0,
  "minimum_plaintext_modulus": 65537,
  "openfhe_version": "...",
  "profiles": [
    { "depth_capacity": 12, "ring_dimension": 32768,  "plaintext_modulus": 786433, "validated": true },
    { "depth_capacity": 21, "ring_dimension": 65536,  "plaintext_modulus": 786433, "validated": true },
    { "depth_capacity": 32, "ring_dimension": 131072, "plaintext_modulus": 786433, "validated": true },
    { "depth_capacity": 45, "ring_dimension": 131072, "plaintext_modulus": 786433, "validated": true }
  ]
}
```

Each profile stores exactly `depth_capacity`, `ring_dimension`,
`plaintext_modulus`, `validated` -- no `max_nodes` (redundant: support for a
graph size depends on both depth and slots jointly) and no separate
`slot_capacity` (equal to `ring_dimension` under full batching, which every
profile assumes). Profiles are sorted by `ring_dimension`, then
`depth_capacity`, then `plaintext_modulus`, and de-duplicated -- this keeps
the file's diffs stable across regenerations.

A profile with a given `(depth_capacity, ring_dimension)` capacity serves
**any** `(N, T)` combination whose required depth and required slots
(`N*N`) both fit within it, not just the specific N it was generated for --
e.g. the N=4 profile above (depth 12, 32768 slots) also covers smaller
circuits like N=2 or N=3 at lower T.

## 5. Runtime lookup

`include/btc/bgv_parameter_profile.hpp` (`SelectProfile`/`RequireProfile`)
and `include/btc/bgv_batched_profile_setup.hpp` (`setup_from_profile`):

```cpp
uint32_t depth = btc::EstimateBGVBatchedDepth(N, T);
uint64_t slots = static_cast<uint64_t>(N) * N;

btc::bgv_batched::Context ctx = btc::bgv_batched::setup_from_profile(depth, slots);
```

Selection picks the **cheapest** profile satisfying
`profile.validated && profile.depth_capacity >= depth && profile.ring_dimension >= slots`,
where "cheapest" is deterministic: smallest `ring_dimension`, then smallest
`depth_capacity`, then smallest `plaintext_modulus` -- the same ordering
profiles are stored in, so selection is a linear scan, not a search.

The selected profile's exact `depth_capacity`, `plaintext_modulus`, and
`ring_dimension` are passed to `bgv_batched::setup` (security level is
always `HEStd_128_classic`, batch size always automatic, matching every
profile's validated assumptions) -- the caller never overrides these
individually, because the whole point of a profile is that the three values
were validated **together**.

After the context is built, `setup_from_profile` re-checks the actual ring
dimension, NTT compatibility, and capacity against the profile's recorded
values, and throws if they disagree -- this catches a stale profile file
(hand-edited, or generated against a different OpenFHE build) rather than
silently trusting a claim that may no longer hold.

### No match found

If no validated profile is large enough, `RequireProfile` throws with an
actionable message instead of silently picking the largest available
profile or starting an online search:

```
No validated BGV parameter profile supports this request.

  Required multiplicative depth: 60
  Required SIMD slots: 4096

Generate additional profiles with:
  ./generate_bgv_batched_profiles --nodes=64
```

## 6. CLI usage

`examples/bgv_batched_closure` and `benchmarks/bench_bgv_batched` both use
`setup_from_profile` automatically -- there is no longer a
`--bgv-plaintext-modulus` / `--bgv-ring-dim` / `--bgv-batch-size` flag to
pass by hand:

```bash
./build/examples/bgv_batched_closure --graph=data/graph_N8.txt
./build/benchmarks/bench_bgv_batched --graph=graph_N16
```

`--profile-db=<path>` overrides the database location (default
`config/bgv_batched_profiles.json`) if you're testing against a
differently-generated profile file.

## 7. Regenerating profiles

If you need a graph size the checked-in database doesn't cover:

```bash
./build/tools/generate_bgv_batched_profiles --nodes=64 \
    --output=config/bgv_batched_profiles.json
```

This adds to (not replaces) the existing profiles.

## 8. What this does NOT do

- Does not change the transitive-closure algorithm, round count, Boolean
  matmul, or the multiplicative-depth estimator (`EstimateBGVBatchedDepth`)
  -- all unchanged from `docs/bgv_batched.md`.
- Does not disable OpenFHE's security checks or use `HEStd_NotSet` --
  every profile is validated under `HEStd_128_classic`.
- Does not hardcode one "universal" cyclotomic order for every possible
  configuration -- each profile's ring dimension/modulus pair is whatever
  the offline search actually found and validated for that specific depth.
- Does not regenerate profiles automatically during normal TC execution --
  generation is a separate, explicit, offline step.
- Currently does not support caller-supplied manual plaintext-modulus/
  ring-dimension overrides at the CLI level (the earlier `--bgv-*` flags
  were removed) -- every run goes through the profile database. This may
  be revisited later if a use case needs it.
