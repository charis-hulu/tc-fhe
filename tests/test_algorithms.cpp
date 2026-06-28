#include <btc/algorithms.hpp>

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

// 3-node chain: 0->1->2
// A^1 = chain, A^2[0][2]=1 only, A^k = 0 for k >= 3.
static btc::BoolMatrix chain3() {
    return from_flat(3, {
        false, true,  false,
        false, false, true,
        false, false, false
    });
}

static void test_T1() {
    auto A = chain3();
    auto [S, P] = btc::sum_powers(A, 1);
    check(S == A, "T=1: S == A");
    check(P == A, "T=1: P == A");
    std::puts("T=1: PASS");
}

static void test_T2() {
    // T=2 (even, h=1):
    //   S(2) = S(1) | P(1)*S(1) = A | A*A = A | A^2
    //   P(2) = A*A = A^2
    auto A = chain3();
    const auto A2 = from_flat(3, {
        false, false, true,
        false, false, false,
        false, false, false
    });
    const auto S2 = from_flat(3, {
        false, true,  true,
        false, false, true,
        false, false, false
    });
    auto [S, P] = btc::sum_powers(A, 2);
    check(P == A2, "T=2: P == A^2");
    check(S == S2, "T=2: S == A | A^2");
    std::puts("T=2: PASS");
}

static void test_T3() {
    // T=3 (odd):
    //   P(3) = P(2)*A = A^2*A = A^3 = 0 (chain exhausted)
    //   S(3) = S(2) | P(3) = S(2)
    auto A = chain3();
    const auto S2 = from_flat(3, {
        false, true,  true,
        false, false, true,
        false, false, false
    });
    const auto Z = btc::BoolMatrix(3, 3, false);

    auto [S, P] = btc::sum_powers(A, 3);
    check(P == Z, "T=3: P == 0");
    check(S == S2, "T=3: S == S_2 (A^3 adds nothing)");
    std::puts("T=3: PASS");
}

static void test_T4() {
    // T=4 (even, h=2):
    //   S(4) = S(2) | P(2)*S(2) = S2 | A^2*S2
    //   A^2*S2: A^2[0]=[0,0,1] so row 0 picks up S2[2]=[0,0,0] => 0
    //   => P(2)*S(2) = 0 => S(4) = S(2)
    //   P(4) = P(2)*P(2) = A^4 = 0
    auto A = chain3();
    const auto S2 = from_flat(3, {
        false, true,  true,
        false, false, true,
        false, false, false
    });
    const auto Z = btc::BoolMatrix(3, 3, false);

    auto [S, P] = btc::sum_powers(A, 4);
    check(P == Z, "T=4: P == 0");
    check(S == S2, "T=4: S == S_2");
    std::puts("T=4: PASS");
}

static void test_cycle_T3() {
    // 3-cycle: 0->1->2->0
    // A^1 = cycle, A^2 = rotation, A^3 = I
    // S(3) = A | A^2 | I = all-ones
    const auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    const auto I = btc::BoolMatrix::identity(3);
    const auto all_ones = from_flat(3, {
        true, true, true,
        true, true, true,
        true, true, true
    });

    auto [S, P] = btc::sum_powers(A, 3);
    check(P == I, "3-cycle T=3: P == I");
    check(S == all_ones, "3-cycle T=3: S == all-ones");
    std::puts("3-cycle T=3: PASS");
}

static void test_cycle_T6() {
    // A^6 on 3-cycle = I again (period 3).
    // S(6) = A | A^2 | I | A^4 | A^5 | I = all-ones (same as S(3)).
    const auto A = from_flat(3, {
        false, true,  false,
        false, false, true,
        true,  false, false
    });
    const auto all_ones = from_flat(3, {
        true, true, true,
        true, true, true,
        true, true, true
    });

    auto [S, P] = btc::sum_powers(A, 6);
    check(S == all_ones, "3-cycle T=6: S == all-ones");
    std::puts("3-cycle T=6: PASS");
}

static void test_T1_error() {
    btc::BoolMatrix A(2, 2, false);
    bool threw = false;
    try {
        btc::sum_powers(A, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "T=0 must throw");
    std::puts("T=0_throws: PASS");
}

int main() {
    test_T1();
    test_T2();
    test_T3();
    test_T4();
    test_cycle_T3();
    test_cycle_T6();
    test_T1_error();
    std::puts("\nAll algorithm tests passed.");
    return 0;
}
