#include <btc/graph_io.hpp>
#include <btc/matrix.hpp>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

btc::BoolMatrix random_matrix(std::size_t n, double density, std::mt19937& rng) {
    std::bernoulli_distribution edge(density);
    btc::BoolMatrix A(n, n, false);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j && edge(rng)) A.set(i, j, true);
    return A;
}

} // namespace

int main() {
    const std::string dir = "data/";
    const std::vector<std::size_t> sizes = {4, 8, 16};
    const double density = 0.2;

    std::mt19937 rng(2024);

    for (std::size_t n : sizes) {
        auto A = random_matrix(n, density, rng);
        std::string path = dir + "graph_N" + std::to_string(n) + ".txt";
        btc::save_graph(path, A);
        std::printf("Wrote %s (N=%zu, density=%.2f)\n", path.c_str(), n, density);
    }

    return 0;
}
