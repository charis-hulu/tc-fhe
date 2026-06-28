#pragma once

#include "backend.hpp"

#include <stdexcept>
#include <utility>

namespace btc {

// ── sum_powers_recursive ─────────────────────────────────────────────────────
//
// Computes (S_T, P_T) where:
//   S_T = A^1 | A^2 | ... | A^T   (Boolean reachability up to T hops)
//   P_T = A^T                       (exactly-T-hop Boolean adjacency)
//
// Divide-and-conquer recurrence:
//   T == 1        : S = A,  P = A
//   T even (T=2h) : S(2h) = S(h) | P(h)·S(h);  P(2h) = P(h)·P(h)
//   T odd         : P(T)  = P(T-1)·A;            S(T)  = S(T-1) | P(T)
//
// Cost: O(log T) Boolean matrix multiplications.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Backend>
std::pair<typename Backend::matrix_type, typename Backend::matrix_type>
sum_powers_recursive(const typename Backend::matrix_type& A, int T,
                     const Backend& backend) {
    using M = typename Backend::matrix_type;

    if (T < 1)
        throw std::invalid_argument("sum_powers_recursive: T must be >= 1");

    if (T == 1)
        return {backend.copy(A), backend.copy(A)};

    if (T % 2 == 0) {
        const int h = T / 2;
        auto [S_h, P_h] = sum_powers_recursive(A, h, backend);

        // S(2h) = S(h) | P(h)·S(h)
        M second_half = backend.matmul(P_h, S_h);
        M S_T = backend.or_mat(S_h, second_half);
        // P(2h) = P(h)·P(h)
        M P_T = backend.matmul(P_h, P_h);
        return {std::move(S_T), std::move(P_T)};
    } else {
        auto [S_prev, P_prev] = sum_powers_recursive(A, T - 1, backend);

        // P(T) = P(T-1)·A
        M P_T = backend.matmul(P_prev, A);
        // S(T) = S(T-1) | P(T)
        M S_T = backend.or_mat(S_prev, P_T);
        return {std::move(S_T), std::move(P_T)};
    }
}

// ── bounded_transitive_closure ───────────────────────────────────────────────
//
// Returns S_T = OR_{k=1}^{T} A^k.
//
// Correctness note vs Floyd-Warshall:
//   For an N-node graph, any simple path has length <= N-1 and any simple
//   cycle has length <= N. Therefore BTC(A, N) == floyd_warshall(A) for
//   every directed graph; for acyclic graphs, T = N-1 already suffices.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Backend>
typename Backend::matrix_type
bounded_transitive_closure(const typename Backend::matrix_type& A, int T,
                           const Backend& backend) {
    auto [S_T, P_T] = sum_powers_recursive(A, T, backend);
    (void)P_T;
    return std::move(S_T);
}

// Floyd-Warshall transitive closure (plaintext reference).
// R[i][j] = 1 iff there is a directed walk from i to j of any length >= 1.
// R[i][i] = 1 only when node i lies on a directed cycle.
BoolMatrix floyd_warshall(const BoolMatrix& A);

// ── Plaintext convenience wrappers ───────────────────────────────────────────

inline std::pair<BoolMatrix, BoolMatrix>
sum_powers(const BoolMatrix& A, int T) {
    return sum_powers_recursive(A, T, PlaintextBackend{});
}

inline BoolMatrix
bounded_transitive_closure(const BoolMatrix& A, int T) {
    return bounded_transitive_closure(A, T, PlaintextBackend{});
}

} // namespace btc
