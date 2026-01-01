#pragma once

#include "aligned_allocator.hpp"
#include "huge_page_allocator.hpp"

#include <array>
#include <experimental/bits/simd.h>
#include <vector>
#include <random>
#include <print>
#include <type_traits>

#include <experimental/simd>

namespace stdx = std::experimental::parallelism_v2;

enum class Impl: char { 
    NAIVE, 
    TRANSPOSED, TRANSPOSED_SIMD, 
    TILED,      TILED_SIMD,      TILED_PREFETCH,    TILED_REGISTERS
};

template<typename T, std::size_t N> requires (N%4==0)
class SquareMatrix {
private:
#if defined(__AVX2__)
    static constexpr std::size_t SIMD_SIZE = 8;
#elif defined(__SSE2__)
    static constexpr std::size_t SIMD_SIZE = 4;
#else
    static constexpr std::size_t SIMD_SIZE = 1; // Scalar fallback
#endif    

    static constexpr std::size_t MAT_WIDTH = std::ceil(double(N) / 48.0) * 48ULL;
    static constexpr std::size_t MAT_SIZE  = MAT_WIDTH * MAT_WIDTH;

    using simd_t = stdx::fixed_size_simd<T, SIMD_SIZE>;

    // static constexpr std::size_t ALIGN = stdx::memory_alignment_v<simd_t>;
    // using aligned_vector = std::vector<T, aligned_allocator<T, ALIGN>>;

    using aligned_vector = std::vector<T, huge_page_allocator<T>>;

    aligned_vector matrix_;
    aligned_vector transposed_;

    constexpr static inline std::size_t getIndex(std::size_t x, std::size_t y) {
        return y * MAT_WIDTH + x;
    }

public:

    static SquareMatrix make_random(T lower_bound, T upper_bound) {
        thread_local std::random_device rd; 
        thread_local std::mt19937 gen(rd()); 
        std::uniform_int_distribution<> distrib(lower_bound, upper_bound); 

        SquareMatrix random_matrix{};
        for (std::size_t x = 0; x < N; ++x) 
            for (std::size_t y = 0; y < N; ++y) 
                random_matrix.matrix_[getIndex(x,y)] = distrib(gen);

        random_matrix.compute_transpose();

        return random_matrix;
    }

    constexpr SquareMatrix()
        : matrix_(MAT_SIZE)
        , transposed_(MAT_SIZE) {}

    template<typename... Args>
        requires(sizeof...(Args) == N*N && 
                 std::conjunction_v<std::is_nothrow_convertible<Args, T>...>) 
    constexpr SquareMatrix(Args&&... args) 
        : matrix_{static_cast<T>(args)...}
        , transposed_(MAT_SIZE) {
        matrix_.resize(MAT_SIZE, 0);
        compute_transpose();
    }

    constexpr const T& get(std::size_t x, std::size_t y) const {
        return matrix_[getIndex(x, y)];
    }

    const T* data() const {
        return matrix_.data();
    }

    const T* data_transposed() const {
        return transposed_.data();
    }

    void print() const {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x < N; ++x) {
                std::print("{:5} ", matrix_[getIndex(x,y)]); 
            }
            std::println();
        }
    }

    constexpr void multiply(
        const SquareMatrix& other, 
        SquareMatrix& out, 
        Impl implementation = Impl::TILED_SIMD
    ) const {
        switch (implementation) {
        case Impl::NAIVE:           multiply_naive(other, out); return;
        case Impl::TRANSPOSED:      multiply_transposed(other, out); return;
        case Impl::TRANSPOSED_SIMD: multiply_simd(other, out); return;
        case Impl::TILED:           multiply_tiled(other, out); return;
        case Impl::TILED_SIMD:      multiply_tiled_simd(other, out); return;
        case Impl::TILED_PREFETCH:  multiply_tiled_prefetch(other, out); return;
        case Impl::TILED_REGISTERS: multiply_tiled_registers(other, out); return;
        default: return;
        }
    }

    constexpr bool operator==(const SquareMatrix& other) const {
        for (std::size_t x = 0; x < N; ++x)
            for (std::size_t y = 0; y < N; ++y)
                if (matrix_[getIndex(x,y)] != other.matrix_[getIndex(x,y)])
                    return false;
        return true;
    }

private:

    // =================================================================
    // SECTION: NAIVE
    // =================================================================

    constexpr void multiply_naive(const SquareMatrix& other, SquareMatrix& out) const {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x < N; ++x) {
                out.matrix_[getIndex(x,y)] = 0;
                for (std::size_t k = 0; k < N; ++k) {
                    out.matrix_[getIndex(x,y)] += matrix_[getIndex(k,y)] * other.matrix_[getIndex(x,k)];
                }
            }
        }
    }

    // =================================================================
    // SECTION: TRANSPOSED
    // =================================================================

    constexpr void compute_transpose() {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x <= y; ++x) {
                transposed_[getIndex(x,y)] = matrix_[getIndex(y,x)];
                transposed_[getIndex(y,x)] = matrix_[getIndex(x,y)];
            }
        }
    }

    void multiply_transposed(const SquareMatrix& other, SquareMatrix& out) const {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x < N; ++x) {
                out.matrix_[getIndex(x,y)] = 0;
                for (std::size_t k = 0; k < N; ++k) {
                    out.matrix_[getIndex(x,y)] += matrix_[getIndex(k,y)] * other.transposed_[getIndex(k,x)];
                }
            }
        }
    }

    // =================================================================
    // SECTION: TRANSPOSED + SIMD
    // =================================================================

    void multiply_simd(const SquareMatrix& other, SquareMatrix& out) const {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x < N; ++x) {
                out.matrix_[getIndex(x,y)] = 0;
                auto a_row = matrix_.data() + y * N;
                auto b_col = other.transposed_.data() + x * N;

                simd_t vsum{};
                for (std::size_t k{}; k < N; k += simd_t::size()) {
                    simd_t va;
                    simd_t vb;

                    va.copy_from(a_row + k, stdx::vector_aligned);
                    vb.copy_from(b_col + k, stdx::vector_aligned);
                    vsum += va * vb;
                }

                out.matrix_[getIndex(x,y)] = stdx::reduce(vsum);
            }
        }
    }

    // =================================================================
    // SECTION: TILED
    // =================================================================

    template<std::size_t TILE_SIZE>
    void pack_tile_linearly(
        const T* mat,
        std::size_t row_offset,
        std::size_t col_offset,
        std::size_t row_limit,
        std::size_t col_limit,
        std::array<T, TILE_SIZE * TILE_SIZE>& pack
    ) const {
        pack.fill(0);
        for (std::size_t row{}; row < row_limit; ++row) {
            for (std::size_t col{}; col < col_limit; ++col) {
                const std::size_t mat_idx  = getIndex(col + col_offset, row + row_offset);
                const std::size_t pack_idx = row * TILE_SIZE + col;
                pack[pack_idx] = mat[mat_idx];
            }
        }
    }

    template<std::size_t TILE_SIZE>
    void microkernel(
        const std::array<T, TILE_SIZE * TILE_SIZE>& a_pack,
        const std::array<T, TILE_SIZE * TILE_SIZE>& bt_pack,
        T* C,
        std::size_t row_offset,
        std::size_t col_offset,
        std::size_t row_limit,
        std::size_t col_limit,
        std::size_t k_blk
    ) const {
        for (std::size_t row{}; row < row_limit; ++row) {
            for (std::size_t col{}; col < col_limit; ++col) {
                const std::size_t c_idx = getIndex(col + col_offset, row + row_offset);
                for (std::size_t k{}; k < k_blk; ++k) {
                    C[c_idx] += a_pack[row * TILE_SIZE + k] * bt_pack[col * TILE_SIZE + k];
                }
            }
        }
    }

    void multiply_tiled(const SquareMatrix& other, SquareMatrix& out) const {
        static constexpr std::size_t TILE_SIZE = 32;

        const T * a_ptr = matrix_.data();
        const T * bt_ptr = other.transposed_.data();
        T * c_ptr = out.matrix_.data();
        
        std::array<T, TILE_SIZE * TILE_SIZE> a_pack;
        std::array<T, TILE_SIZE * TILE_SIZE> bt_pack;

        for (std::size_t i{}; i < N; i += TILE_SIZE) {
            const std::size_t i_blk = std::min(N - i, TILE_SIZE);
            for (std::size_t k{}; k < N; k += TILE_SIZE) {
                const std::size_t k_blk = std::min(N - k, TILE_SIZE);
                pack_tile_linearly<TILE_SIZE>(a_ptr, i, k, i_blk, k_blk, a_pack);

                for (std::size_t j{}; j < N; j += TILE_SIZE) {
                    const std::size_t j_blk = std::min(N - j, TILE_SIZE);
                    pack_tile_linearly<TILE_SIZE>(bt_ptr, j, k, j_blk, k_blk, bt_pack);
                    microkernel<TILE_SIZE>(a_pack, bt_pack, c_ptr, i, j, i_blk, j_blk, k_blk);
                }
            }
        }
    }

    // =================================================================
    // SECTION: TILED + SIMD
    // =================================================================

    template<std::size_t COUNT, std::size_t STRIDE=1, std::size_t I=0>
    constexpr void unroll(auto&& fn) const {
        if constexpr (I < COUNT) {
            fn.template operator()<I>();
            unroll<COUNT, STRIDE, I + STRIDE>(fn);
        }
    }

    template<std::size_t TILE_SIZE>
    void microkernel_simd(
        const std::array<T, TILE_SIZE * TILE_SIZE>& a_pack,
        const std::array<T, TILE_SIZE * TILE_SIZE>& b_pack,
        T* C,
        std::size_t row_offset,
        std::size_t col_offset,
        std::size_t row_limit,
        std::size_t col_limit,
        std::size_t k_limit
    ) const {
        std::array<simd_t, SIMD_SIZE> C_rows; 

        for (std::size_t row{}; row < row_limit; row += SIMD_SIZE) {
            for (std::size_t col{}; col < col_limit; col += SIMD_SIZE) {
                unroll<SIMD_SIZE>([&]<std::size_t i> {
                    C_rows[i].copy_from(
                        C + getIndex(col + col_offset, row + row_offset + i), 
                        stdx::vector_aligned
                    );
                 });

                for (std::size_t k{}; k < k_limit; ++k) {
                    simd_t b;
                    b.copy_from(&b_pack[k * TILE_SIZE + col], stdx::vector_aligned);

                    unroll<SIMD_SIZE>([&]<std::size_t r> {
                        const std::size_t pack_idx = (row + r) * TILE_SIZE + k;
                        C_rows[r] += simd_t(a_pack[pack_idx]) * b;
                    });
                }

                unroll<SIMD_SIZE>([&]<std::size_t i> {
                    C_rows[i].copy_to(
                        C + getIndex(col + col_offset, row + row_offset + i), 
                        stdx::vector_aligned
                    );
                 });
            }
        }
    }

    void multiply_tiled_simd(const SquareMatrix& other, SquareMatrix& out) const {
        static constexpr std::size_t TILE_SIZE = 32;

        const T * a_ptr = matrix_.data();
        const T * b_ptr = other.matrix_.data();
        T * c_ptr = out.matrix_.data();
        
        alignas(64) std::array<T, TILE_SIZE * TILE_SIZE> a_pack;
        alignas(64) std::array<T, TILE_SIZE * TILE_SIZE> b_pack;

        for (std::size_t i{}; i < N; i += TILE_SIZE) {
            const std::size_t i_blk = std::min(N - i, TILE_SIZE);
            for (std::size_t k{}; k < N; k += TILE_SIZE) {
                const std::size_t k_blk = std::min(N - k, TILE_SIZE);
                pack_tile_linearly<TILE_SIZE>(a_ptr, i, k, i_blk, k_blk, a_pack);

                for (std::size_t j{}; j < N; j += TILE_SIZE) {
                    const std::size_t j_blk = std::min(N - j, TILE_SIZE);
                    pack_tile_linearly<TILE_SIZE>(b_ptr, k, j, k_blk, j_blk, b_pack);
                    microkernel_simd<TILE_SIZE>(a_pack, b_pack, c_ptr, i, j, i_blk, j_blk, k_blk);
                }
            }
        }
    }

    // =================================================================
    // SECTION: TILED + SIMD + PREFETCHER
    // =================================================================

    template<std::size_t TILE_SIZE>
    void pack_tile_linearly_prefetched(
        const T* mat,
        std::size_t row_offset,
        std::size_t col_offset,
        std::size_t row_limit,
        std::size_t col_limit,
        std::array<T, TILE_SIZE * TILE_SIZE>& pack
    ) const {
        pack.fill(0);
        for (std::size_t row{}; row < row_limit; ++row) {
            for (std::size_t col{}; col < col_limit; ++col) {
                const std::size_t next_row_index  = getIndex(col + col_offset, row + row_offset + 1);
                _mm_prefetch((const char*)&mat[next_row_index], _MM_HINT_T0);

                const std::size_t mat_idx  = getIndex(col + col_offset, row + row_offset);
                const std::size_t pack_idx = row * TILE_SIZE + col;
                pack[pack_idx] = mat[mat_idx];
            }
        }
    }

    void multiply_tiled_prefetch(const SquareMatrix& other, SquareMatrix& out) const {
        static constexpr std::size_t TILE_SIZE = 32;

        const T * a_ptr = matrix_.data();
        const T * b_ptr = other.matrix_.data();
        T * c_ptr = out.matrix_.data();
        
        alignas(64) std::array<T, TILE_SIZE * TILE_SIZE> a_pack;
        alignas(64) std::array<T, TILE_SIZE * TILE_SIZE> b_pack;

        for (std::size_t i{}; i < N; i += TILE_SIZE) {
            const std::size_t i_blk = std::min(N - i, TILE_SIZE);
            for (std::size_t k{}; k < N; k += TILE_SIZE) {
                const std::size_t k_blk = std::min(N - k, TILE_SIZE);
                pack_tile_linearly_prefetched<TILE_SIZE>(a_ptr, i, k, i_blk, k_blk, a_pack);

                for (std::size_t j{}; j < N; j += TILE_SIZE) {
                    const std::size_t next_tile_index = getIndex(k, j + TILE_SIZE);
                    _mm_prefetch((const char*)&b_ptr[next_tile_index], _MM_HINT_T1);

                    const std::size_t j_blk = std::min(N - j, TILE_SIZE);
                    pack_tile_linearly_prefetched<TILE_SIZE>(b_ptr, k, j, k_blk, j_blk, b_pack);
                    microkernel_simd<TILE_SIZE>(a_pack, b_pack, c_ptr, i, j, i_blk, j_blk, k_blk);
                }
            }
        }
    }

    // =================================================================
    // SECTION: TILED REGISTERS + SIMD
    // ON AMD x86-64 :: AVX2 :: 16 YMM regs :: 12(C) + 2(B) + 1(A) = 15
    // =================================================================

    template<std::size_t TILE_SIZE>
    void microkernel_6x2(
        const std::array<T, TILE_SIZE * TILE_SIZE>& a_pack,
        const std::array<T, TILE_SIZE * TILE_SIZE>& b_pack,
        T* C,
        std::size_t row_offset,
        std::size_t col_offset
    ) const {
        static constexpr std::size_t N_ROWS = 6;
        static constexpr std::size_t N_COLS = 2;
        static constexpr std::size_t C_REGS = N_ROWS * N_COLS;

        std::array<simd_t, C_REGS> c_regs; 
        std::array<simd_t, N_COLS> b_regs;

        for (std::size_t row{}; row < TILE_SIZE; row += N_ROWS) {
            for (std::size_t col{}; col < TILE_SIZE; col += (N_COLS * SIMD_SIZE)) {
                unroll<N_ROWS>([&]<std::size_t r> {
                    unroll<N_COLS>([&]<std::size_t c> {
                        c_regs[r * N_COLS + c].copy_from(
                            C + getIndex(
                                col + col_offset + (c * SIMD_SIZE), 
                                row + row_offset + r
                            ), 
                            stdx::vector_aligned
                        );
                    });
                });

                for (std::size_t k{}; k < TILE_SIZE; ++k) {
                    b_regs[0].copy_from(&b_pack[k * TILE_SIZE + col], stdx::vector_aligned);
                    b_regs[1].copy_from(&b_pack[k * TILE_SIZE + col + SIMD_SIZE], stdx::vector_aligned);

                    unroll<N_ROWS>([&]<std::size_t i> {
                        const auto a = simd_t(a_pack[(row + i) * TILE_SIZE + k]);
                        c_regs[i * 2 + 0] += a * b_regs[0];
                        c_regs[i * 2 + 1] += a * b_regs[1];
                    });
                }

                unroll<N_ROWS>([&]<std::size_t r> {
                    unroll<N_COLS>([&]<std::size_t c> {
                        c_regs[r * N_COLS + c].copy_to(
                            C + getIndex(
                                col + col_offset + (c * SIMD_SIZE), 
                                row + row_offset + r
                            ), 
                            stdx::vector_aligned
                        );
                    });
                });
            }
        }
    }

    void multiply_tiled_registers(const SquareMatrix& other, SquareMatrix& out) const {
        static constexpr std::size_t TILE_SIZE = 48;

        const T * a_ptr = matrix_.data();
        const T * b_ptr = other.matrix_.data();
        T * c_ptr = out.matrix_.data();
        
        alignas(64) std::array<T, TILE_SIZE * TILE_SIZE> a_pack;
        alignas(64) std::array<T, TILE_SIZE * TILE_SIZE> b_pack;

        for (std::size_t i{}; i < MAT_WIDTH; i += TILE_SIZE) {
            for (std::size_t k{}; k < MAT_WIDTH; k += TILE_SIZE) {
                pack_tile_linearly<TILE_SIZE>(a_ptr, i, k, TILE_SIZE, TILE_SIZE, a_pack);

                for (std::size_t j{}; j < MAT_WIDTH; j += TILE_SIZE) {
                    pack_tile_linearly<TILE_SIZE>(b_ptr, k, j, TILE_SIZE, TILE_SIZE, b_pack);
                    microkernel_6x2<TILE_SIZE>(a_pack, b_pack, c_ptr, i, j);
                }
            }
        }
    }
};

