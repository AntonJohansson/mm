#if !defined(MM_IMPLEMENTATION)
#pragma once
#endif

typedef unsigned char      u8;
typedef float              f32;
typedef double             f64;
typedef unsigned int       u32;
typedef unsigned long long u64;

//void matmul_v0(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);
//
//void matmul_v1(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);
//
//void matmul_v2(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);
//
//void matmul_v3(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);
//
//void matmul_v4(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);
//
//void matmul_v5(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);
//
//void matmul_v6(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l);

#include <ranges>

struct Matrix {
    u32 rows;
    u32 columns;
    f32 *data;

    auto row(u32 i) {
        return std::views::counted(data, rows*columns);
    }

    auto column(u32 j) {
        return std::views::counted(data+j, rows*columns) | std::views::stride(columns);
    }
};

#if defined(MM_IMPLEMENTATION)

#include <string.h> // for memset

#if defined(MM_INCLUDE_CBLAS)
#include <cblas.h>
#endif

#if defined(__x86_64__) && defined(__SSE3__) && defined(__AVX__)
# include <x86intrin.h>
# define MM_INCLUDE_X86_64_AVX
# if !defined(matmul)
#  define matmul matmul_v6
# endif
#else
# if !defined(matmul)
#  define matmul matmul_v3
# endif
#endif

#if defined(MM_ALLOCATE_ALIGNED)
f32 *MM_ALLOCATE_ALIGNED(u32, u32);
#else
# error Please define MM_ALLOCATE_ALIGNED to a function f32 *(u32 alignment, u32 size), it will only be called in a single threaded context
#endif

//
// https://ppc.cs.aalto.fi/
// https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
//

//void matmul_v0(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            c[i*l + j] = 0;
//            for (u32 k = 0; k < m; ++k) {
//                c[i*l + j] += a[i*m + k] * b[k*l + j];
//            }
//        }
//    }
//}
//
//void matmul_v1(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            c[i*l + j] = 0;
//            for (u32 k = 0; k < m; ++k) {
//                c[i*l + j] += a[i*m + k] * b[k*l + j];
//            }
//        }
//    }
//}
//
//void matmul_v2(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    f32 *tb = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * l * m);
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i)
//        for (u32 j = 0; j < m; ++j)
//            tb[i*m + j] = b[j*l + i];
//
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            c[i*l + j] = 0;
//            for (u32 k = 0; k < m; ++k) {
//                c[i*l + j] += a[i*m + k] * tb[j*m + k];
//            }
//        }
//    }
//}
//
//#define pad_align 8
//
//void matmul_v3(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    const u32 ma = (m + pad_align - 1) / pad_align;
//    const u32 la = (l + pad_align - 1) / pad_align;
//    const u32 pm = ma*pad_align;
//    const u32 pl = la*pad_align;
//
//    f32 *pa  = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * n * pm);
//    f32 *ptb = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * pl * pm);
//    memset(pa,  0, sizeof(f32) * n * pm);
//    memset(ptb, 0, sizeof(f32) * pl * pm);
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            pa[i*pm + j] = a[i*m + j];
//        }
//    }
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            ptb[i*pm + j] = b[j*l + i];
//        }
//    }
//
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            f32 d[pad_align] = {0};
//
//            for (u32 ka = 0; ka < ma; ++ka) {
//                for (u32 kb = 0; kb < pad_align; ++kb) {
//                    d[kb] += pa[i*pm + ka*pad_align + kb] * ptb[j*pm + ka*pad_align + kb];
//                }
//            }
//
//            c[i*l + j] = 0;
//            for (u32 k = 0; k < pad_align; ++k) {
//                c[i*l + j] += d[k];
//            }
//        }
//    }
//}
//
//// For the following two functions see:
////   https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
//
//#if defined(MM_INCLUDE_X86_64_AVX)
//inline f32 hsum_ps_sse3(__m128 v) {
//    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
//    __m128 sums = _mm_add_ps(v, shuf);
//    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
//    sums        = _mm_add_ss(sums, shuf);
//    return        _mm_cvtss_f32(sums);
//}
//
//inline f32 hsum256_ps_avx(__m256 v) {
//    __m128 vlow  = _mm256_castps256_ps128(v);
//    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
//           vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
//    return hsum_ps_sse3(vlow);                  // and inline the sse3 version, which is optimal for AVX
//    // (no wasted instructions, and all of them are the 4B minimum)
//}
//
//void matmul_v4(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    const u32 ma = (m + pad_align - 1) / pad_align;
//    const u32 la = (l + pad_align - 1) / pad_align;
//    const u32 pm = ma*pad_align;
//    const u32 pl = la*pad_align;
//
//    f32 *pa  = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * n * pm);
//    f32 *ptb = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * pl * pm);
//    memset(pa,  0, sizeof(f32) * n * pm);
//    memset(ptb, 0, sizeof(f32) * pl * pm);
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            pa[i*pm + j] = a[i*m + j];
//        }
//    }
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            ptb[i*pm + j] = b[j*l + i];
//        }
//    }
//
//    f32 *d = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32));
//    memset(d, 0, sizeof(f32)*8);
//
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            __m256 vd = _mm256_loadu_ps(&d[0]);
//
//            for (u32 ka = 0; ka < ma; ++ka) {
//                __m256 va = _mm256_loadu_ps( &pa[i*pm + ka*pad_align]);
//                __m256 vb = _mm256_loadu_ps(&ptb[j*pm + ka*pad_align]);
//                vd = _mm256_fmadd_ps(va, vb, vd);
//            }
//
//            c[i*l + j] = 0;
//            for (u32 kb = 0; kb < pad_align; ++kb) {
//                c[i*l + j] += vd[kb];
//            }
//        }
//    }
//}
//
//void matmul_v5(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    const u32 ma = (m + pad_align - 1) / pad_align;
//    const u32 la = (l + pad_align - 1) / pad_align;
//    const u32 pm = ma*pad_align;
//    const u32 pl = la*pad_align;
//
//    f32 *pa  = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * n * pm);
//    f32 *ptb = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * pl * pm);
//    memset(pa,  0, sizeof(f32) * n * pm);
//    memset(ptb, 0, sizeof(f32) * pl * pm);
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            pa[i*pm + j] = a[i*m + j];
//        }
//    }
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            ptb[i*pm + j] = b[j*l + i];
//        }
//    }
//
//    f32 *d = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32));
//    memset(d, 0, sizeof(f32)*8);
//
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            __m256 vd = _mm256_loadu_ps(&d[0]);
//
//            for (u32 ka = 0; ka < ma; ++ka) {
//                __m256 va = _mm256_loadu_ps( &pa[i*pm + ka*pad_align]);
//                __m256 vb = _mm256_loadu_ps(&ptb[j*pm + ka*pad_align]);
//                vd = _mm256_fmadd_ps(va, vb, vd);
//            }
//
//            c[i*l + j] = hsum256_ps_avx(vd);
//        }
//    }
//}
//
//inline f32 hadd(__m256 vd, u32 nb) {
//    f32 sum = 0;
//    for (int kb = 0; kb < nb; ++kb)
//        sum += vd[kb];
//    return sum;
//}
//
//#define NB 8
//#define ND 3
//
//void matmul_v6(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    const u32 ma = (m + NB - 1) / NB;
//    const u32 la = (l + NB - 1) / NB;
//    const u32 pm = ma*NB;
//    const u32 pl = la*NB;
//
//    const u32 nc = (n + ND - 1) / ND;
//    const u32 ncd = nc*ND;
//
//    const u32 mc = (m + ND - 1) / ND;
//    const u32 mcd = mc*ND;
//
//    f32 *pa  = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32)*ncd*ma);
//    f32 *ptb = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32)*mcd*la);
//    memset(pa, 0, sizeof(f32)*8 * ncd * ma);
//    memset(ptb, 0, sizeof(f32)*8 * mcd * la);
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            pa[i*pm + j] = a[i*m + j];
//        }
//    }
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            ptb[i*pm + j] = b[j*l + i];
//        }
//    }
//
//    f32 *d = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32)*ND*ND);
//    memset(d, 0, sizeof(f32)*8*ND*ND);
//
//#pragma omp parallel for
//    for (u32 ic = 0; ic < nc; ++ic) {
//        for (u32 jc = 0; jc < nc; ++jc) {
//            __m256 vd00 = _mm256_loadu_ps(&d[0*ND + 0]);
//            __m256 vd01 = _mm256_loadu_ps(&d[0*ND + 1]);
//            __m256 vd02 = _mm256_loadu_ps(&d[0*ND + 2]);
//            __m256 vd10 = _mm256_loadu_ps(&d[1*ND + 0]);
//            __m256 vd11 = _mm256_loadu_ps(&d[1*ND + 1]);
//            __m256 vd12 = _mm256_loadu_ps(&d[1*ND + 2]);
//            __m256 vd20 = _mm256_loadu_ps(&d[2*ND + 0]);
//            __m256 vd21 = _mm256_loadu_ps(&d[2*ND + 1]);
//            __m256 vd22 = _mm256_loadu_ps(&d[2*ND + 2]);
//
//            for (u32 ka = 0; ka < ma; ++ka) {
//                __m256 va0 = _mm256_loadu_ps( &pa[(ic*ND + 0)*ma*NB + ka*NB]);
//                __m256 va1 = _mm256_loadu_ps( &pa[(ic*ND + 1)*ma*NB + ka*NB]);
//                __m256 va2 = _mm256_loadu_ps( &pa[(ic*ND + 2)*ma*NB + ka*NB]);
//                __m256 vb0 = _mm256_loadu_ps(&ptb[(jc*ND + 0)*ma*NB + ka*NB]);
//                __m256 vb1 = _mm256_loadu_ps(&ptb[(jc*ND + 1)*ma*NB + ka*NB]);
//                __m256 vb2 = _mm256_loadu_ps(&ptb[(jc*ND + 2)*ma*NB + ka*NB]);
//                vd00 = _mm256_fmadd_ps(va0, vb0, vd00);
//                vd01 = _mm256_fmadd_ps(va0, vb1, vd01);
//                vd02 = _mm256_fmadd_ps(va0, vb2, vd02);
//                vd10 = _mm256_fmadd_ps(va1, vb0, vd10);
//                vd11 = _mm256_fmadd_ps(va1, vb1, vd11);
//                vd12 = _mm256_fmadd_ps(va1, vb2, vd12);
//                vd20 = _mm256_fmadd_ps(va2, vb0, vd20);
//                vd21 = _mm256_fmadd_ps(va2, vb1, vd21);
//                vd22 = _mm256_fmadd_ps(va2, vb2, vd22);
//            }
//
//            u32 i, j;
//            i = ic*ND + 0; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd00, NB);
//            i = ic*ND + 0; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd01, NB);
//            i = ic*ND + 0; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd02, NB);
//            i = ic*ND + 1; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd10, NB);
//            i = ic*ND + 1; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd11, NB);
//            i = ic*ND + 1; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd12, NB);
//            i = ic*ND + 2; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd20, NB);
//            i = ic*ND + 2; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd21, NB);
//            i = ic*ND + 2; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd22, NB);
//        }
//    }
//}
//#endif

#if defined(MM_INCLUDE_CBLAS)
//void gemm(f32 * restrict c,
//          const f32 * restrict a,
//          const f32 * restrict b,
//          u32 n, u32 m, u32 l) {
//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n,l,m, 1.0f, a,m, b,l, 0.0f, c,m);
//}
#endif

#endif // MM_IMPLEMENTATION
