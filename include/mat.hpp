#pragma once

#include "aligned_allocator.hpp"

#include <experimental/bits/simd.h>
#include <vector>
#include <random>
#include <print>
#include <type_traits>

#include <experimental/simd>

namespace stdx = std::experimental::parallelism_v2;

enum class Impl: char { NAIVE, TRANSPOSED, SIMD, MICROKERNEL, TILING };

template<typename T, std::size_t N> requires (N%4==0)
class SquareMatrix {
private:

    using simd_t = stdx::fixed_size_simd<T, 4>;
    static constexpr std::size_t ALIGN = stdx::memory_alignment_v<simd_t>;
    using aligned_vector = std::vector<T, aligned_allocator<T, ALIGN>>;

    aligned_vector matrix_;
    aligned_vector transposed_;

    constexpr static inline std::size_t getIndex(std::size_t x, std::size_t y) {
        return y * N + x;
    }

public:

    static SquareMatrix make_random(T lower_bound, T upper_bound) {
        thread_local std::random_device rd; 
        thread_local std::mt19937 gen(rd()); 
        std::uniform_int_distribution<> distrib(lower_bound, upper_bound); 

        SquareMatrix random_matrix{};
        for (std::size_t i = 0; i < N*N; ++i) 
            random_matrix.matrix_[i] = distrib(gen);

        random_matrix.compute_transpose();

        return random_matrix;
    }

    constexpr SquareMatrix()
        : matrix_(N*N)
        , transposed_(N*N) {}

    template<typename... Args>
        requires(sizeof...(Args) == N*N && 
                 std::conjunction_v<std::is_nothrow_convertible<Args, T>...>) 
    constexpr SquareMatrix(Args&&... args) 
        : matrix_{static_cast<T>(args)...}
        , transposed_(N*N) {
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
                std::print("{:8} ", matrix_[getIndex(x,y)]); 
            }
            std::println();
        }
    }

    constexpr void multiply(
        const SquareMatrix& other, 
        SquareMatrix& out, 
        Impl implementation = Impl::TILING
    ) const {
        switch (implementation) {
        case Impl::NAIVE:       multiply_naive(other, out); return;
        case Impl::TRANSPOSED:  multiply_transposed(other, out); return;
        case Impl::SIMD:        multiply_simd(other, out); return;
        case Impl::MICROKERNEL: multiply_naive(other, out); return;
        case Impl::TILING:      multiply_tiling(other, out); return;
        default: return;
        }
    }

    constexpr bool operator==(const SquareMatrix& other) const {
        for (std::size_t i = 0; i < N*N; ++i)
            if (matrix_[i] != other.matrix_[i])
                return false;
        return true;
    }

private:

    constexpr void compute_transpose() {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x <= y; ++x) {
                transposed_[getIndex(x,y)] = matrix_[getIndex(y,x)];
                transposed_[getIndex(y,x)] = matrix_[getIndex(x,y)];
            }
        }
    }

    constexpr void pack_tiles(
        const T* R0, const T* R1, const T* R2, const T* R3,
        T* pack, std::size_t K_blk
    ) const {
        for (std::size_t k = 0; k < K_blk; ++k) {
            pack[0*K_blk + k] = R0[k];
            pack[1*K_blk + k] = R1[k];
            pack[2*K_blk + k] = R2[k];
            pack[3*K_blk + k] = R3[k];
        }
    }

    void microkernel_4x4(
        const T* A_pack,
        const T* B_pack,
        T* C0, T* C1, T* C2, T* C3,
        std::size_t K_blk
    ) const {
        simd_t c00(0), c01(0), c02(0), c03(0);
        simd_t c10(0), c11(0), c12(0), c13(0);
        simd_t c20(0), c21(0), c22(0), c23(0);
        simd_t c30(0), c31(0), c32(0), c33(0);

        const std::size_t K_simd = (K_blk / simd_t::size()) * simd_t::size();
        for (std::size_t k = 0; k < K_simd; k += simd_t::size()) {
            simd_t a0, a1, a2, a3;
            a0.copy_from(A_pack + 0*K_blk + k, stdx::vector_aligned);
            a1.copy_from(A_pack + 1*K_blk + k, stdx::vector_aligned);
            a2.copy_from(A_pack + 2*K_blk + k, stdx::vector_aligned);
            a3.copy_from(A_pack + 3*K_blk + k, stdx::vector_aligned);

            simd_t b0, b1, b2, b3;
            b0.copy_from(B_pack + 0*K_blk + k, stdx::vector_aligned);
            b1.copy_from(B_pack + 1*K_blk + k, stdx::vector_aligned);
            b2.copy_from(B_pack + 2*K_blk + k, stdx::vector_aligned);
            b3.copy_from(B_pack + 3*K_blk + k, stdx::vector_aligned);

            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        C0[0] += stdx::reduce(c00);
        C0[1] += stdx::reduce(c01);
        C0[2] += stdx::reduce(c02);
        C0[3] += stdx::reduce(c03);

        C1[0] += stdx::reduce(c10);
        C1[1] += stdx::reduce(c11);
        C1[2] += stdx::reduce(c12);
        C1[3] += stdx::reduce(c13);

        C2[0] += stdx::reduce(c20);
        C2[1] += stdx::reduce(c21);
        C2[2] += stdx::reduce(c22);
        C2[3] += stdx::reduce(c23);

        C3[0] += stdx::reduce(c30);
        C3[1] += stdx::reduce(c31);
        C3[2] += stdx::reduce(c32);
        C3[3] += stdx::reduce(c33);
    }

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

    void multiply_microkernel(const SquareMatrix& other, SquareMatrix& out) const {}

    void multiply_tiling(const SquareMatrix& other, SquareMatrix& out) const {
        constexpr std::size_t TILE_SIZE = 32;
        constexpr std::size_t NR = 4;

        const T* A = matrix_.data();
        const T* BT = other.transposed_.data();
        T* C = out.matrix_.data();

        alignas(64) T A_pack[NR * TILE_SIZE];
        alignas(64) T B_pack[NR * TILE_SIZE];

        for (std::size_t ii = 0; ii < N; ii += TILE_SIZE) {
            for (std::size_t jj = 0; jj < N; jj += TILE_SIZE) {
                for (std::size_t kk = 0; kk < N; kk += TILE_SIZE) {
                    
                    const std::size_t i_end = std::min(ii + TILE_SIZE, N);
                    const std::size_t j_end = std::min(jj + TILE_SIZE, N);
                    const std::size_t k_end = std::min(kk + TILE_SIZE, N);

                    const std::size_t K_blk  = k_end - kk;

                    for (std::size_t i = ii; i < i_end; i += NR) {

                        const T* A0 = A + (i+0)*N + kk;
                        const T* A1 = A + (i+1)*N + kk;
                        const T* A2 = A + (i+2)*N + kk;
                        const T* A3 = A + (i+3)*N + kk;

                        pack_tiles(A0, A1, A2, A3, A_pack, K_blk);

                        for (std::size_t j = jj; j < j_end; j += NR) {
                            const T* B0 = BT + (j+0)*N + kk;
                            const T* B1 = BT + (j+1)*N + kk;
                            const T* B2 = BT + (j+2)*N + kk;
                            const T* B3 = BT + (j+3)*N + kk;

                            pack_tiles(B0, B1, B2, B3, B_pack, K_blk);
                                
                            T* C0 = C + (i+0)*N + j;
                            T* C1 = C + (i+1)*N + j;
                            T* C2 = C + (i+2)*N + j;
                            T* C3 = C + (i+3)*N + j;

                            microkernel_4x4(
                                A_pack, B_pack, 
                                C0, C1, C2, C3,
                                K_blk
                            );

                        }
                    }
                }
            }
        }
    }
};

