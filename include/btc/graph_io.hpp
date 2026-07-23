#pragma once

#include "matrix.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace btc {

// Text adjacency-matrix format:
//   line 1: N
//   lines 2..N+1: N space-separated 0/1 values (row-major)
inline void save_graph(const std::string& path, const BoolMatrix& A) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("save_graph: cannot open " + path);

    const std::size_t n = A.rows();
    out << n << "\n";
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out << (A.get(i, j) ? 1 : 0);
            if (j + 1 < n) out << ' ';
        }
        out << "\n";
    }
}

inline BoolMatrix load_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_graph: cannot open " + path);

    std::size_t n = 0;
    in >> n;
    BoolMatrix A(n, n, false);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            int v = 0;
            in >> v;
            A.set(i, j, v != 0);
        }
    }
    return A;
}

} // namespace btc
