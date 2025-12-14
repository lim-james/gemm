#include <cassert>
#include "../include/mat.hpp"

int main() {
    // fixed constructor - constexpr 
    {
        // constexpr SquareMatrix<int, 4> A{
        // //   0   1   2   3 
        //     00, 10, 20, 30, // 0
        //     01, 11, 21, 31, // 1
        //     02, 12, 22, 32, // 2
        //     03, 13, 23, 33  // 3
        // };
        // static_assert(
        //     A.get(0,0) == 00 &&
        //     A.get(1,0) == 10 && 
        //     A.get(3,1) == 31 && 
        //     A.get(1,2) == 12 && 
        //     A.get(3,3) == 33 
        // );
    }

    return 0;
}
