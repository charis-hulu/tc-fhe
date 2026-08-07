# Distributed BGV row/column-packed backend (MPI + OpenMP): status

Tracks progress against `claude_distributed_bgv_rowcol_mpi.md`'s three-stage
plan for a distributed version of `include/btc/bgv_rowcol_backend.hpp`.

## What this backend is

A fourth BGV backend, alongside `bgv_backend.hpp` (per-element),
`bgv_batched_backend.hpp` (whole-matrix packing), and `bgv_rowcol_backend.hpp`
(single-process row/column packing, OpenMP only). It exploits the same
observation `OpenMPOptimized` already established locally -- the N^2
encrypted dot products `C[i][j] = SumOR(A.row(i) (*) B.col(j))` are mutually
independent -- but spreads that independence across MPI ranks (and compute
nodes) via a 2D process grid, instead of just OpenMP threads on one process.

## Done

### Stage 1 -- correct distributed matrix multiplication

- **2D MPI process grid** (`ProcessGrid2D`, `BuildProcessGrid2D` in
  `include/btc/bgv_distributed_rowcol_backend.hpp`): balanced `Pr x Pc` grid
  via `MPI_Dims_create` + `MPI_Cart_create`, with `row_comm`/`col_comm`
  subcommunicators via `MPI_Cart_sub`.
- **Non-divisible block partitioning** (`ComputeBlockPartition`): splits `N`
  into `Pr` (or `Pc`) contiguous blocks without assuming divisibility.
- **`DistributedRowColMatrix`**: holds only the row block `I_r` and column
  block `J_c` this rank owns, tagged with global indices -- not the whole
  matrix.
- **Local dot-product phase**: each rank computes `C[i][j]` for `i` in its
  row block and `j` in its column block **exactly once**
  (`BooleanDotProductPublic`, reused unmodified from `BGVRowColBackend`),
  then repacks the single result into both a row-piece and a column-piece
  (`MaskMultPublic`, also reused unmodified) -- mirrors
  `OpenMPOptimized`'s single-computation-reused-twice pattern, with MPI
  ranks standing in for OpenMP threads as the unit of "owns a disjoint
  (i,j) block." Parallelized further within a rank via
  `#pragma omp parallel for collapse(2)`.
- **Row/column assembly** (`AssembleRows`/`AssembleCols`): owner-distributed
  combine within `row_comm`/`col_comm` (`rowOwner(i) = i % Pc`,
  `colOwner(j) = j % Pr`, not funneled through rank 0), using
  `EvalAdd`-after-deserialize (never raw MPI byte reduction) followed by a
  broadcast back to every rank sharing that row/column block.
- **Ownership invariant preserved after every matmul**: rank `(r,c)` ends
  every `matmul`/`or_mat` call holding complete rows `I_r` and complete
  columns `J_c` again, ready for the next recursion step with no
  whole-matrix gather in between.

### Stage 2 -- distributed bounded transitive closure

- **`BGVDistributedRowColBackend` satisfies the `Backend` concept**
  (`matrix_type`, `matmul`, `or_mat`, `copy`, `dim`) exactly like the other
  three backends, so `sum_powers_recursive`/`bounded_transitive_closure`
  (`algorithms.hpp`) work against it **unmodified** -- `P` and `S` stay
  `DistributedRowColMatrix` objects across every recursion level.
- **`or_mat`**: no communication needed (rows/cols already co-located
  identically for `A` and `B`); every rank computes `EvalBooleanOR` for its
  own resident rows/columns.
- **Secret-key discipline**: only rank 0 calls `bgv_batched::setup()` (which
  generates the secret key) and the rotation-key-generating constructor path.
  Every other rank receives a serialized bundle (crypto context, public key,
  eval-mult key, eval-automorphism/rotation keys -- **no secret key**) and
  constructs its local `BGVRowColBackend` with a new
  `RotationKeys::AssumeAlreadyInstalled` mode added to that class
  specifically for this. Only rank 0 can decrypt (`DecryptOnRoot`).
- **Serialization utilities** (`include/btc/bgv_mpi_serial.hpp`):
  `SerializeCiphertext`/`DeserializeCiphertext`,
  `SerializeContextBundle`/`DeserializeContextBundle`, MPI byte-buffer
  transport (`SendBytes`/`RecvBytes`/`BroadcastBytes` and ciphertext-typed
  wrappers). Point-to-point send/recv and broadcast-within-subcommunicator;
  see Stage 3 below for the batched (`*CiphertextBatch`) and non-blocking
  (`Isend*`/`PendingSend`/`WaitAllSends`) additions built on top of these.
- **Root-driven encrypt/decrypt** (`EncryptFromRoot`/`DecryptOnRoot`):
  root encrypts the full matrix once, every rank extracts the row/column
  block it owns via broadcast-and-slice; inverse gather-to-root for final
  decryption.

### Stage 3 -- communication batching, `or_mat` dedup, concurrent branches

Three of Stage 3's items are now implemented (nonblocking communication and
tiled/streaming repacking are not -- see "Not done yet" below):

- **Communication batching.** `AssembleRows`/`AssembleCols` (via the shared
  `AssembleOwnedBucket`/`GatherOwnedToAll` helpers), `or_mat`'s
  `BroadcastOwnedBatch`, and the generalized `GatherToRankWithinGrid`/
  `ScatterFromRankToGrid` (see below) all exchange **one batched message per
  (source, owner) pair or per broadcast root**, instead of one message per
  ciphertext -- `bgv_mpi_serial.hpp`'s `SerializeCiphertextBatch`/
  `SendCiphertextBatch`/`RecvCiphertextBatch`/`BroadcastCiphertextBatch` pack
  many `(global-index, ciphertext)` pairs into a single byte blob. Still not
  a true `MPI_Alltoallv`/`Gatherv` collective, but this cuts message COUNT
  from `O(local block size)` down to `O(comm_size)`, the actual bottleneck
  at N=4/N=8.
- **`or_mat` owner-compute dedup.** A single deterministic owner per
  row/column (`rowOwner(i) = i % row_comm_size`, matching
  `AssembleRows`/`AssembleCols`'s scheme) computes `EvalBooleanOR` once and
  broadcasts the batch, instead of every rank sharing a replicated
  row/column block recomputing the same OR redundantly.
- **Concurrent recursion branches (`matmul_pair`).** Splits `grid_`'s world
  communicator into two disjoint halves (`MPI_Comm_split`), each building
  its OWN full 2D process grid over all N rows/columns, so
  `sum_powers_recursive`'s even-`T` step's two mutually-independent matmul
  calls (`P(2h) = matmul(P_h,P_h)` and the intermediate `matmul(P_h,S_h)`)
  run **concurrently on disjoint MPI ranks**. `algorithms.hpp` is
  deliberately left untouched (it's shared by every backend); instead
  `DistributedSumPowersRecursive`/`DistributedBoundedTransitiveClosure`
  (bottom of `bgv_distributed_rowcol_backend.hpp`) mirror its exact
  recurrence but call `matmul_pair` in place of the two sequential calls.
  Because a sub-grid built over half the ranks generally has a DIFFERENT
  `Pr'/Pc'` than `grid_` (a different rank count factors differently), both
  the matmul OPERANDS and its RESULT need reshaping between `grid_`'s
  partition and the sub-grid's -- `matmul_pair` does this via
  `GatherToRankWithinGrid`/`ScatterFromRankToGrid` (generalized versions of
  `GatherToRoot`/`ScatterFullMatrixToBlocks`, parameterized over an
  arbitrary grid + root instead of hardcoded to `grid_`/rank 0), each
  moving `O(N)` ciphertexts (not `O(N^2)`) -- the same order as
  `EncryptFromRoot`/`DecryptOnRoot`'s existing boundary cost.

### Correctness verification (test_bgv_distributed_rowcol.cpp)

Ran on Polaris at **1, 2, and 4 MPI ranks**, N=4, T=4, all passing:

- Distributed decrypted result equals the plaintext reference.
- Distributed result equals the single-node `OpenMPOptimized` result
  (independent crypto context).
- `BGVRowColBackend::Decrypt`'s row/column cross-check (reused via
  `DecryptOnRoot`) confirms row and column representations agree after
  every distributed matmul.
- Total `dot_products` summed across all ranks equals exactly `N^2` per
  matmul call (4 matmul calls x 16 = 64), confirming no `C[i][j]` is ever
  computed twice globally -- at 1, 2, *and* 4 ranks (including the 2x2 grid
  case, the one most likely to hide a double-count bug).
- `DistributedBoundedTransitiveClosure` (the `matmul_pair`/concurrent-branch
  path) produces the EXACT SAME decrypted result as the sequential
  distributed path above, with the same total `dot_products` (64) -- at 1,
  2, and 4 ranks -- confirming the world-split-and-reconcile machinery
  changes only WHICH ranks do the work, not WHAT gets computed.

### Bugs found and fixed during this work

1. **Missing Cereal registration headers.** `openfhe.h` does not pull in
   `ciphertext-ser.h`/`cryptocontext-ser.h`/`key/key-ser.h` -- without them,
   `Serial::Serialize` throws `cereal::Exception: Trying to save an
   unregistered polymorphic type` at runtime, not a compile error. Fixed by
   including them explicitly in `bgv_mpi_serial.hpp`.
2. **`ProcessGrid2D` move semantics.** Deleting the copy constructor alone
   left `return grid;` and `grid_ = BuildProcessGrid2D(...)` broken --
   needed an explicit move constructor/assignment that nulls out the
   moved-from communicator handles (so `MPI_Comm_free` in the destructor
   only ever runs once per real communicator).
3. **MPI-finalize-ordering crash.** `BGVDistributedRowColBackend` owns MPI
   communicators (via `ProcessGrid2D`); if the backend outlives
   `MPI_Finalize()` (e.g. declared in the same scope as the `MPI_Finalize()`
   call in `main`), its destructor calls `MPI_Comm_free` after MPI has
   already been finalized and crashes. Fixed by scoping the backend inside
   a nested block that closes before `MPI_Finalize()` runs.
4. **The "hang" that turned out to be an `OMP_NUM_THREADS` bug.** Polaris
   compute nodes apparently inherit `OMP_NUM_THREADS=64` from the PBS job
   environment. The test script used `OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}`,
   which only applies a default when the variable is *unset* -- since it was
   already set to 64, the default silently never took effect, and 64 OpenMP
   threads were spawned per `#pragma omp parallel for collapse(2)` region
   for N=4-sized work (at most 16 loop iterations). This didn't deadlock,
   it was just extremely slow due to thread-pool churn, and looked
   indistinguishable from a hang until isolated with a hard `timeout`
   wrapper and a debug run that forced `OMP_NUM_THREADS=2` unconditionally.
   Fixed in `scripts/run_polaris_bgv_distributed_rowcol_test.pbs` by forcing
   the override unconditionally instead of using a `:-` default.
5. **Rendezvous-protocol deadlock in batched `AssembleRows`/`AssembleCols`.**
   The first version of the Stage 3 batching change bucketed each rank's
   partial rows/columns by owner and posted ALL sends (blocking
   `MPI_Send`-based `SendCiphertextBatch`) before posting any receives. That
   is safe for small messages (MPI's eager protocol buffers them and the
   call returns immediately) but not in general: once a batched message is
   large enough to cross MPI's eager/rendezvous threshold, `MPI_Send`
   blocks until the destination posts a matching `MPI_Recv` -- and if TWO
   ranks each need to send to each other and both finish their whole send
   loop before starting their receive loop, both block forever waiting for
   the other. Reproduced empirically: the 1-rank test (whose `row_comm`/
   `col_comm` are trivial, size 1, no real exchange) passed, but the first
   real 2-rank exchange hung until PBS's 45-minute walltime killed it with
   `rank 0 died from signal 15`. Fixed by making the send side non-blocking
   (`MPI_Isend` via `bgv_mpi_serial.hpp`'s new `IsendBytes`/
   `IsendCiphertextBatch`/`PendingSend`/`WaitAllSends`, keeping each
   serialized buffer alive in a `std::deque` until `MPI_Wait`), so a rank can
   always reach its own receive loop regardless of whether its sends have
   been serviced yet. `GatherToRankWithinGrid`'s star-topology gather (root
   services every sender in a fixed loop order, no sender waits on another
   sender) was already deadlock-safe and was left blocking.
6. **`matmul_pair` operand/result shape mismatch.** The first version of
   `matmul_pair` passed `A1`/`B1`/`A2`/`B2` -- shaped according to `grid_`'s
   (the FULL grid's) partition -- directly into `MatMulOnGrid` running over
   a temporary SUB-grid, whose `Pr'/Pc'` is generally different (a smaller
   rank count factors differently via `MPI_Dims_create`). `MatMulOnGrid`
   indexes its operands by the sub-grid's local row/column counts, so this
   crashed with `vector::_M_range_check` (`DistributedRowColMatrix::row_at`/
   `col_at`'s `.at()` bounds check) as soon as a sub-grid's local block was
   larger than what the grid_-shaped operand actually held for that rank.
   Fixed by reshaping operands from `grid_`'s partition into each
   sub-grid's partition BEFORE computing (gather to a fixed rank, re-scatter
   into the sub-grid), symmetric to the result reshape already used to
   bring the OUTPUT back into `grid_`'s shape afterward. A related latent
   bug surfaced by the same fix: `GatherToRankWithinGrid`/
   `ScatterFromRankToGrid` originally communicated over `grid.world`, but
   `matmul_pair` frees its temporary `sub_comm` (which sub-grids receive as
   `grid.world`) immediately after building the sub-grid's Cartesian
   topology -- any later use of `grid.world` on that sub-grid would be a
   use-after-free. Fixed by having both functions communicate over
   `grid.cart` instead (guaranteed identical rank numbering to the input
   communicator via `MPI_Cart_create`'s `reorder=0`, and owned/freed by
   `ProcessGrid2D` itself, so always valid for as long as the grid object
   is alive) -- this also happens to be a strictly better default for
   `grid_` itself, not just sub-grids.

## Not done yet (Stage 3 remainder)

- **Nonblocking communication for the star-topology gather/scatter.**
  `GatherToRankWithinGrid`/`ScatterFromRankToGrid` (used by
  `EncryptFromRoot`/`DecryptOnRoot` and `matmul_pair`'s reshaping) are still
  blocking -- provably deadlock-safe as described above, but a slow sender
  still serializes behind the root's fixed service order rather than
  overlapping with useful work.
- **Tiled/streaming repacking.** The local dot-product phase materializes
  every locally-owned `C[i][j]` (as row/column pieces) before assembly,
  rather than streaming/freeing them incrementally to reduce peak memory.
- **Benchmarking.** No timing/speedup/efficiency measurements have been
  taken yet -- no comparison of single-node `OpenMPOptimized` vs. MPI at
  1/2/4 ranks vs. the full-matrix-packed baseline, no N=8/N=16 runs, no
  communication-volume measurements, no per-phase breakdown (local compute
  vs. repacking vs. serialization vs. MPI communication vs. row/column
  assembly). The task's "Benchmark Plan" and "Required Instrumentation"
  sections are entirely outstanding.
- **N=8 / larger-N correctness testing.** Only N=4, T=4 has been verified
  end-to-end. N=8 is mentioned as "when available" in the task's
  correctness requirements but has not been attempted.
