#include <cassert>
#include "../include/mat.hpp"

int main() {
    // 1. Constructor sanity
    {
        SquareMatrix<int, 4> A = SquareMatrix<int, 4>::make_random(0, 5);
        SquareMatrix<int, 4> B = SquareMatrix<int, 4>::make_random(0, 5);
        assert(true && "construction OK");
    }

    // 2. Naive vs SIMD equals small test
    {
        SquareMatrix<int, 4> A = SquareMatrix<int, 4>::make_random(0, 5);
        SquareMatrix<int, 4> B = SquareMatrix<int, 4>::make_random(0, 5);

        auto C1 = A.mul_naive(B);
        auto C2 = A.mul_simd(B);

        assert(C1 == C2 && "SIMD must match naive for 4x4");
    }

    // 3. Random multiple tests
    {
        for (int iter = 0; iter < 20; iter++) {
            SquareMatrix<int, 8> A = SquareMatrix<int, 8>::make_random(0, 9);
            SquareMatrix<int, 8> B = SquareMatrix<int, 8>::make_random(0, 9);
            assert((A.mul_naive(B) == A.mul_simd(B)) && "8x8 check failed");
        }
    }

    return 0;
}
