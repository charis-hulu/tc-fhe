#include <btc/backend.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static btc::BoolMatrix from_flat(std::size_t n,
                                  std::initializer_list<bool> data) {
    btc::BoolMatrix m(n, n);
    auto it = data.begin();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m.set(i, j, *it++);
    return m;
}

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void test_identity_mul() {
    btc::PlaintextBackend B;
    // 3-node chain adjacency matrix
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        false, false, false
    });
    auto I = btc::BoolMatrix::identity(3);
    check(B.matmul(I, A) == A, "I*A == A");
    check(B.matmul(A, I) == A, "A*I == A");
    std::puts("identity_mul: PASS");
}

static void test_zero_mul() {
    btc::PlaintextBackend B;
    btc::BoolMatrix Z(3, 3, false);
    auto A = btc::BoolMatrix::identity(3);
    check(B.matmul(Z, A) == Z, "0*A == 0");
    check(B.matmul(A, Z) == Z, "A*0 == 0");
    std::puts("zero_mul: PASS");
}

static void test_2x2_manual() {
    // A = [[1,1],[0,1]]  B = [[1,0],[1,1]]
    // C[0][0] = (1&1)|(1&1) = 1
    // C[0][1] = (1&0)|(1&1) = 1
    // C[1][0] = (0&1)|(1&1) = 1
    // C[1][1] = (0&0)|(1&1) = 1
    btc::PlaintextBackend B;
    btc::BoolMatrix A(2, 2), M(2, 2);
    A.set(0, 0, true);  A.set(0, 1, true);
    A.set(1, 0, false); A.set(1, 1, true);
    M.set(0, 0, true);  M.set(0, 1, false);
    M.set(1, 0, true);  M.set(1, 1, true);

    auto C = B.matmul(A, M);
    check(C.get(0, 0) && C.get(0, 1) && C.get(1, 0) && C.get(1, 1),
          "2x2 all-ones result");
    std::puts("2x2_manual: PASS");
}

static void test_chain_square() {
    // 0->1->2 chain: A^2 should have only A^2[0][2]=1
    btc::PlaintextBackend B;
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        false, false, false
    });
    auto A2 = B.matmul(A, A);
    auto expected = from_flat(3, {
        false, false, true,
        false, false, false,
        false, false, false
    });
    check(A2 == expected, "chain A^2");
    std::puts("chain_square: PASS");
}

static void test_chain_cube() {
    // A^3 on 3-node chain should be all-zero (path exhausted)
    btc::PlaintextBackend B;
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        false, false, false
    });
    auto A2 = B.matmul(A, A);
    auto A3 = B.matmul(A2, A);
    btc::BoolMatrix Z(3, 3, false);
    check(A3 == Z, "chain A^3 == 0");
    std::puts("chain_cube: PASS");
}

static void test_cycle_square() {
    // 0->1->2->0 (3-cycle)
    // A^2: A^2[i][j] = A[(i+1)%3][j]
    // A^2 = [[0,0,1],[1,0,0],[0,1,0]]
    btc::PlaintextBackend B;
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    auto A2 = B.matmul(A, A);
    auto expected = from_flat(3, {
        false, false, true,
        true,  false, false,
        false, true,  false
    });
    check(A2 == expected, "3-cycle A^2");
    std::puts("cycle_square: PASS");
}

static void test_cycle_cube() {
    // A^3 on 3-cycle = identity
    btc::PlaintextBackend B;
    auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    auto A2 = B.matmul(A, A);
    auto A3 = B.matmul(A2, A);
    auto I  = btc::BoolMatrix::identity(3);
    check(A3 == I, "3-cycle A^3 == I");
    std::puts("cycle_cube: PASS");
}

int main() {
    test_identity_mul();
    test_zero_mul();
    test_2x2_manual();
    test_chain_square();
    test_chain_cube();
    test_cycle_square();
    test_cycle_cube();
    std::puts("\nAll matmul tests passed.");
    return 0;
}
