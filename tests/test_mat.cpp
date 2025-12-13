#include <cassert>
#include "../include/mat.hpp"

int main() {
    // fixed constructor
    {
        SquareMatrix<int, 4> A{
        //   0   1   2   3 
            00, 10, 20, 30, // 0
            01, 11, 21, 31, // 1
            02, 12, 22, 32, // 2
            03, 13, 23, 33  // 3
        };
        assert(
            A.get(0,0) == 00 &&
            A.get(1,0) == 10 && 
            A.get(3,1) == 31 && 
            A.get(1,2) == 12 && 
            A.get(3,3) == 33 && 
            "manual construction failed"
        );
    }

    // make random constructor sanity
    {
        SquareMatrix<int, 4> A = SquareMatrix<int, 4>::make_random(0, 5);
        SquareMatrix<int, 4> B = SquareMatrix<int, 4>::make_random(0, 5);
        assert(true && "construction OK");
    }

    // naive vs simd equals small test
    {
        SquareMatrix<int, 4> A = SquareMatrix<int, 4>::make_random(0, 5);
        SquareMatrix<int, 4> B = SquareMatrix<int, 4>::make_random(0, 5);

        auto C1 = A.multiply(B, Impl::NAIVE);
        auto C2 = A.multiply(B, Impl::TILING);

        assert(C1 == C2 && "SIMD must match naive for 4x4");
    }

    // random multiple tests
    {
        for (int iter = 0; iter < 20; iter++) {
            SquareMatrix<int, 8> A = SquareMatrix<int, 8>::make_random(0, 9);
            SquareMatrix<int, 8> B = SquareMatrix<int, 8>::make_random(0, 9);
            assert(
                (A.multiply(B, Impl::NAIVE) == A.multiply(B, Impl::TILING)) 
                && "8x8 check failed"
            );
        }
    }

    return 0;
}
