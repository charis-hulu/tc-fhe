# Time Complexity of Bounded Transitive Closure over FHE

## Problem statement

Given the Boolean adjacency matrix $A \in \{0,1\}^{N \times N}$ of a directed
graph on $N$ nodes, we want the transitive closure

$$S = \bigvee_{k=1}^{T} A^k,$$

i.e. $S[i][j] = 1$ iff there is a directed walk from $i$ to $j$ of length
between $1$ and $T$, where Boolean matrix multiplication uses AND for the
scalar product and OR for the sum. The whole computation is carried out
homomorphically: $A$ is encrypted bit-by-bit, and every AND/OR is evaluated
as a bootstrapped FHE gate on ciphertexts, so the server never sees a
plaintext bit.

**Choice of $T$.** Any simple path in an $N$-node graph has length $\le N-1$,
and any simple cycle has length $\le N$. Consequently $S_N = \text{Floyd–Warshall}(A)$
for *every* directed graph, whereas $T = N-1$ is only sufficient when $A$ is
acyclic. Since the benchmark graphs are general random directed graphs (not
guaranteed acyclic), we always take $T = N$.

## Algorithm

The closure is computed by divide-and-conquer matrix exponentiation
(`sum_powers_recursive`), tracking the pair
$(S_h, P_h) = \left(\bigvee_{k=1}^{h} A^k,\ A^h\right)$:

- $h = 1$: &nbsp; $S_1 = A$, &nbsp; $P_1 = A$
- $T = 2h$: &nbsp; $S_{2h} = S_h \vee (P_h \cdot S_h)$, &nbsp; $P_{2h} = P_h \cdot P_h$
- $T$ odd: &nbsp; $P_T = P_{T-1} \cdot A$, &nbsp; $S_T = S_{T-1} \vee P_T$

Each level of the recursion halves (or decrements) $T$, so the recursion
depth is $O(\log T)$, and each level performs at most two Boolean matrix
multiplications and one elementwise OR.

## Step-by-step cost derivation

### 1. Cost of one homomorphic gate

Every AND/OR is a bootstrapped FHE gate with a fixed latency $t_{\text{gate}}$,
independent of $N$ (it depends only on the FHE security parameters). This is
the fundamental unit of cost — unlike plaintext bit ops, it cannot be
vectorized away by the CPU.

### 2. Cost of Boolean matrix multiplication

$C = A \cdot B$ for $N \times N$ matrices requires, for each of the $N^2$
output cells $C[i][j]$, an AND with each of the $N$ terms and an
$(N-1)$-fold OR-reduction to combine them:

$$\text{gates(matmul)} = N^2 \cdot \big(N_{\text{AND}} + (N-1)_{\text{OR}}\big) = O(N^3).$$

### 3. Cost of elementwise OR

$$\text{gates(or\_mat)} = N^2 = O(N^2).$$

### 4. Cost per recursion level

Each level does $\le 2$ matmuls ($O(N^3)$ each) and $1$ OR ($O(N^2)$), so the
per-level cost is dominated by the matmuls:

$$\text{gates(level)} = O(N^3).$$

### 5. Number of levels

Halving $T$ down to $1$ takes $\lceil \log_2 T \rceil$ levels. With $T = N$:

$$\text{levels} = O(\log N).$$

### 6. Total gate count

$$\text{gates}(N) = O\!\left(N^3 \log N\right)$$

### 7. Wall-clock time

Since every gate costs a fixed $t_{\text{gate}}$ (bootstrapping latency, not
reducible by better plaintext code), the predicted wall-clock time is

$$\text{time}(N) = t_{\text{gate}} \cdot O\!\left(N^3 \log N\right).$$

## Comparison to Floyd–Warshall

Plaintext Floyd–Warshall is $O(N^3)$ but **inherently sequential**: the
$k$-th outer iteration depends on the result of iteration $k-1$, so the
$N^3$ work cannot be parallelized across the $k$ dimension. Our
divide-and-conquer formulation trades a $\log N$ factor in *total* gate
count for a computation that is $O(\log N)$ *sequential* matmul calls, each
of which is $O(N^3)$ work that is fully data-parallel across the $N^2$
independent output cells. This is the right trade-off under FHE, where each
gate is expensive and parallel hardware (multi-core CPU, GPU) is what
actually buys speed — not a smaller total gate count.

Plaintext Boolean matmul also gets a sparsity discount the FHE version
cannot: `PlaintextBackend::matmul` skips a row entirely when $A[i][k] = 0$,
since it can inspect the bit. The FHE backend can never do this — the whole
point of encryption is that the server cannot tell which bits are zero — so
it always performs the full dense $O(N^3)$ gate count regardless of how
sparse the underlying graph is.

## Empirical validation

Measured on the CPU boolean backend (TFHE-rs, default parameters), with
$T = N$ per the correctness requirement above:

| $N$ | $T$ | Gate count (measured) | fhe_ms (measured) |
|---:|---:|---:|---:|
| 8  | 8  | ≈ 5,952  | 61,442.8 |
| 16 | 16 | ≈ 64,512 | 654,422 |

Both points give $t_{\text{gate}} \approx 10.2$ ms, consistent with a fixed
bootstrapping cost independent of $N$. Fitting $\text{time}(N) = k \cdot N^3 \log_2 N$
to the same two points gives $k \approx 39.97$ (in ms, using $\log_2$), with
less than 0.2% disagreement between the two fits — strong evidence for the
$O(N^3 \log N)$ model.

## Theoretical projection

Extrapolating $\text{time}(N) = k \cdot N^3 \log_2 N$ with $k = 39.97$ ms:

| $N$ | $N^3 \log_2 N$ | Predicted time | |
|---:|---:|---:|---|
| 8    | 1,536              | 61.4 s      | (measured) |
| 16   | 16,384             | 654.4 s     | (measured) |
| 32   | 163,840            | ≈ 6,549 s   | ≈ 1.82 h |
| 64   | 1,572,864          | ≈ 62,871 s  | ≈ 17.46 h |
| 128  | 14,680,064         | ≈ 586,796 s | ≈ 6.79 d |
| 256  | 134,217,728        | ≈ 5,364,991 s | ≈ 62.1 d |
| 512  | 1,207,959,552      | ≈ 4.83 × 10⁷ s | ≈ 1.53 yr |
| 1024 | 10,737,418,240     | ≈ 4.29 × 10⁸ s | ≈ 13.6 yr |

## Practical implications

- This is an extrapolation, not a guarantee: it assumes $t_{\text{gate}}$
  stays constant as $N$ grows. In practice, memory pressure from the
  $O(N^2)$ live ciphertexts (each of fixed size, set by the FHE parameters)
  can degrade cache/NUMA behavior well before $N$ reaches the several-hundred
  range, making real measurements worse than the model past that point.
- Polaris' debug queue enforces walltime limits far shorter than the
  projected times for $N \ge 128$, so the sweep script (`run_polaris.pbs`)
  only goes up to $N = 256$ by default — beyond that, a single job is not a
  realistic unit of work.
- The $\log N$ factor is the price of making the computation parallelizable.
  Two levers reduce wall-clock time without changing the $O(N^3 \log N)$
  gate count:
  1. **CPU parallelism**: the $N^2$ output cells of a matmul are independent
     and can be computed concurrently (currently unexploited —
     `TFHEBackend::matmul` is a serial triple loop).
  2. **GPU batching**: CUDA-based bootstrapping amortizes fixed overhead
     across thousands of concurrent gates, but only pays off once a single
     matmul "wave" ($N^2$ independent gates) is large enough to saturate the
     device — small $N$ underutilizes the GPU and can be slower than CPU.
