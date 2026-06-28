#include <btc/algorithms.hpp>

#include <chrono>
#include <cstdio>

// 8-node directed graph, row-major adjacency matrix (0 = no edge, 1 = edge)
static const int ADJ[8][8] = {
    {0, 1, 0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0},
};

int main() {
    const std::size_t N = 8;
    const int T = 5; // N-1: full transitive closure for this DAG

    btc::BoolMatrix A(N, N, false);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            A.set(i, j, ADJ[i][j] == 1);

    const int REPS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; ++r)
        btc::bounded_transitive_closure(A, T);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;
    std::printf("N=%zu T=%d avg=%.4f ms\n\n", N, T, ms);

    auto result = btc::bounded_transitive_closure(A, T);
    std::printf("Reachability matrix (S_T):\n");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::printf("%d ", result.get(i, j) ? 1 : 0);
        std::printf("\n");
    }
    return 0;
}
