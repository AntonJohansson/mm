#include <time.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cblas.h>

#define EPSILON 1e-4

typedef unsigned char u8;
typedef float         f32;
typedef double        f64;
typedef unsigned int  u32;

//
// Time measuring
//

static inline void time_current(struct timespec *t) {
    assert(clock_gettime(CLOCK_MONOTONIC, t) == 0);
}

static inline void time_diff(struct timespec *c, struct timespec *a, struct timespec *b) {
    c->tv_sec = a->tv_sec - b->tv_sec;
    c->tv_nsec = a->tv_nsec - b->tv_nsec;
}

static inline size_t time_nanoseconds(struct timespec *t) {
    return t->tv_sec * 1000000000ull + t->tv_nsec;
}

//
// Memory management
//

typedef struct StackAllocator {
    u8 *memory;
    u32 size;
    u32 top;
} StackAllocator;

StackAllocator stack;

static void *allocate_memory(StackAllocator *stack, u32 size) {
    assert(stack->top + size < stack->size);
    u8 *ptr = stack->memory + stack->top;
    stack->top += size;
    return (void *) ptr;
}

static void *allocate_aligned_memory(StackAllocator *stack, u32 alignment, u32 size) {
    u32 new_top = alignment*(stack->top + alignment - 1)/alignment;
    assert(new_top + size < stack->size);
    u8 *ptr = stack->memory + new_top;
    stack->top = new_top + size;
    return (void *) ptr;
}

static void random_matrix(f32 *mat, u32 n, u32 m) {
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < m; ++j)
            mat[i*m + j] = 1.0f - 2.0f * ((f32) rand() / (f32) RAND_MAX);
}

static void print_matrix(const char *name, f32 *mat, u32 n, u32 m) {
    printf("%s:\n", name);
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            printf("%8.0f ", mat[i*m + j]);
        }
        printf("\n");
    }
}

//
// https://ppc.cs.aalto.fi/
//

__attribute__((noinline))
static void matmul_v0(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < l; ++j) {
            c[i*l + j] = 0;
            for (u32 k = 0; k < m; ++k) {
                c[i*l + j] += a[i*m + k] * b[k*l + j];
            }
        }
    }
}

__attribute__((noinline))
static void matmul_v1(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < l; ++j) {
            c[i*l + j] = 0;
            for (u32 k = 0; k < m; ++k) {
                asm("# here");
                c[i*l + j] += a[i*m + k] * b[k*l + j];
            }
        }
    }
}

__attribute__((noinline))
static void matmul_v2(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
    u32 top = stack.top;
    f32 *tb = allocate_memory(&stack, sizeof(f32) * l * m);
#pragma omp parallel for
    for (u32 i = 0; i < l; ++i)
        for (u32 j = 0; j < m; ++j)
            tb[i*m + j] = b[j*l + i];

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < l; ++j) {
            c[i*l + j] = 0;
            for (u32 k = 0; k < m; ++k) {
                asm("# here");
                c[i*l + j] += a[i*m + k] * tb[j*m + k];
            }
        }
    }
    stack.top = top;
}

#define pad_align 8

__attribute__((noinline))
static void matmul_v3(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
    const u32 ma = (m + pad_align - 1) / pad_align;
    const u32 la = (l + pad_align - 1) / pad_align;
    const u32 pm = ma*pad_align;
    const u32 pl = la*pad_align;

    u32 top = stack.top;
    f32 *pa  = allocate_memory(&stack, sizeof(f32) * n * pm);
    f32 *ptb = allocate_memory(&stack, sizeof(f32) * pl * pm);
    memset(pa,  0, sizeof(f32) * n * pm);
    memset(ptb, 0, sizeof(f32) * pl * pm);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            pa[i*pm + j] = a[i*m + j];
        }
    }
#pragma omp parallel for
    for (u32 i = 0; i < l; ++i) {
        for (u32 j = 0; j < m; ++j) {
            ptb[i*pm + j] = b[j*l + i];
        }
    }

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < l; ++j) {
            f32 d[pad_align] = {0};

            for (u32 ka = 0; ka < ma; ++ka) {
                for (u32 kb = 0; kb < pad_align; ++kb) {
                    asm("# here");
                    d[kb] += pa[i*pm + ka*pad_align + kb] * ptb[j*pm + ka*pad_align + kb];
                }
            }

            c[i*l + j] = 0;
            for (u32 k = 0; k < pad_align; ++k) {
                c[i*l + j] += d[k];
            }
        }
    }
    stack.top = top;
}

#include <x86intrin.h>

static inline f32 *f32_8x_allocate(StackAllocator *stack, u32 n) {
    return allocate_aligned_memory(stack, 8*sizeof(f32), 8*sizeof(f32)*n);
}

// For the following two functions see:
//   https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction

#if defined(__SSE3__)
f32 hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}
#endif

#if defined(__AVX__) && defined(__SSE3__)
f32 hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
           vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);                  // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}
#endif

__attribute__((noinline))
static void matmul_v4(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
    const u32 ma = (m + pad_align - 1) / pad_align;
    const u32 la = (l + pad_align - 1) / pad_align;
    const u32 pm = ma*pad_align;
    const u32 pl = la*pad_align;

    u32 top = stack.top;
    f32 *pa  = allocate_memory(&stack, sizeof(f32) * n * pm);
    f32 *ptb = allocate_memory(&stack, sizeof(f32) * pl * pm);
    memset(pa,  0, sizeof(f32) * n * pm);
    memset(ptb, 0, sizeof(f32) * pl * pm);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            pa[i*pm + j] = a[i*m + j];
        }
    }
#pragma omp parallel for
    for (u32 i = 0; i < l; ++i) {
        for (u32 j = 0; j < m; ++j) {
            ptb[i*pm + j] = b[j*l + i];
        }
    }

    f32 *d = f32_8x_allocate(&stack, 1);
    memset(d, 0, sizeof(f32)*8);

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < l; ++j) {
            __m256 vd = _mm256_loadu_ps(&d[0]);

            for (u32 ka = 0; ka < ma; ++ka) {
                __m256 va = _mm256_loadu_ps( &pa[i*pm + ka*pad_align]);
                __m256 vb = _mm256_loadu_ps(&ptb[j*pm + ka*pad_align]);
                vd = _mm256_fmadd_ps(va, vb, vd);
            }

            c[i*l + j] = 0;
            for (u32 kb = 0; kb < pad_align; ++kb) {
                c[i*l + j] += vd[kb];
            }
        }
    }
    stack.top = top;
}

__attribute__((noinline))
static void matmul_v5(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
    const u32 ma = (m + pad_align - 1) / pad_align;
    const u32 la = (l + pad_align - 1) / pad_align;
    const u32 pm = ma*pad_align;
    const u32 pl = la*pad_align;

    u32 top = stack.top;
    f32 *pa  = allocate_memory(&stack, sizeof(f32) * n * pm);
    f32 *ptb = allocate_memory(&stack, sizeof(f32) * pl * pm);
    memset(pa,  0, sizeof(f32) * n * pm);
    memset(ptb, 0, sizeof(f32) * pl * pm);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            pa[i*pm + j] = a[i*m + j];
        }
    }
#pragma omp parallel for
    for (u32 i = 0; i < l; ++i) {
        for (u32 j = 0; j < m; ++j) {
            ptb[i*pm + j] = b[j*l + i];
        }
    }

    f32 *d = f32_8x_allocate(&stack, 1);
    memset(d, 0, sizeof(f32)*8);

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < l; ++j) {
            __m256 vd = _mm256_loadu_ps(&d[0]);

            for (u32 ka = 0; ka < ma; ++ka) {
                __m256 va = _mm256_loadu_ps( &pa[i*pm + ka*pad_align]);
                __m256 vb = _mm256_loadu_ps(&ptb[j*pm + ka*pad_align]);
                vd = _mm256_fmadd_ps(va, vb, vd);
            }

            c[i*l + j] = hsum256_ps_avx(vd);
        }
    }
    stack.top = top;
}

static inline f32 hadd(__m256 vd, u32 nb) {
    f32 sum = 0;
    for (int kb = 0; kb < nb; ++kb)
        sum += vd[kb];
    return sum;
}

#define NB 8
#define ND 3

__attribute__((noinline))
static void matmul_v6(f32 * restrict c,
                      const f32 * restrict a,
                      const f32 * restrict b,
                      u32 n, u32 m, u32 l) {
    const u32 ma = (m + NB - 1) / NB;
    const u32 la = (l + NB - 1) / NB;
    const u32 pm = ma*NB;
    const u32 pl = la*NB;

    const u32 nc = (n + ND - 1) / ND;
    const u32 ncd = nc*ND;

    const u32 mc = (m + ND - 1) / ND;
    const u32 mcd = mc*ND;

    u32 top = stack.top;
    f32 *pa  = f32_8x_allocate(&stack, ncd * ma);
    f32 *ptb = f32_8x_allocate(&stack, mcd * la);
    memset(pa, 0, sizeof(f32)*8 * ncd * ma);
    memset(ptb, 0, sizeof(f32)*8 * mcd * la);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            pa[i*pm + j] = a[i*m + j];
        }
    }
#pragma omp parallel for
    for (u32 i = 0; i < l; ++i) {
        for (u32 j = 0; j < m; ++j) {
            ptb[i*pm + j] = b[j*l + i];
        }
    }

    f32 *d = f32_8x_allocate(&stack, ND*ND);
    memset(d, 0, sizeof(f32)*8*ND*ND);

#pragma omp parallel for
    for (u32 ic = 0; ic < nc; ++ic) {
        for (u32 jc = 0; jc < nc; ++jc) {
            __m256 vd00 = _mm256_loadu_ps(&d[0*ND + 0]);
            __m256 vd01 = _mm256_loadu_ps(&d[0*ND + 1]);
            __m256 vd02 = _mm256_loadu_ps(&d[0*ND + 2]);
            __m256 vd10 = _mm256_loadu_ps(&d[1*ND + 0]);
            __m256 vd11 = _mm256_loadu_ps(&d[1*ND + 1]);
            __m256 vd12 = _mm256_loadu_ps(&d[1*ND + 2]);
            __m256 vd20 = _mm256_loadu_ps(&d[2*ND + 0]);
            __m256 vd21 = _mm256_loadu_ps(&d[2*ND + 1]);
            __m256 vd22 = _mm256_loadu_ps(&d[2*ND + 2]);

            for (u32 ka = 0; ka < ma; ++ka) {
                __m256 va0 = _mm256_loadu_ps( &pa[(ic*ND + 0)*ma*NB + ka*NB]);
                __m256 va1 = _mm256_loadu_ps( &pa[(ic*ND + 1)*ma*NB + ka*NB]);
                __m256 va2 = _mm256_loadu_ps( &pa[(ic*ND + 2)*ma*NB + ka*NB]);
                __m256 vb0 = _mm256_loadu_ps(&ptb[(jc*ND + 0)*ma*NB + ka*NB]);
                __m256 vb1 = _mm256_loadu_ps(&ptb[(jc*ND + 1)*ma*NB + ka*NB]);
                __m256 vb2 = _mm256_loadu_ps(&ptb[(jc*ND + 2)*ma*NB + ka*NB]);
                vd00 = _mm256_fmadd_ps(va0, vb0, vd00);
                vd01 = _mm256_fmadd_ps(va0, vb1, vd01);
                vd02 = _mm256_fmadd_ps(va0, vb2, vd02);
                vd10 = _mm256_fmadd_ps(va1, vb0, vd10);
                vd11 = _mm256_fmadd_ps(va1, vb1, vd11);
                vd12 = _mm256_fmadd_ps(va1, vb2, vd12);
                vd20 = _mm256_fmadd_ps(va2, vb0, vd20);
                vd21 = _mm256_fmadd_ps(va2, vb1, vd21);
                vd22 = _mm256_fmadd_ps(va2, vb2, vd22);
            }

            u32 i, j;
            i = ic*ND + 0; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd00, NB);
            i = ic*ND + 0; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd01, NB);
            i = ic*ND + 0; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd02, NB);
            i = ic*ND + 1; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd10, NB);
            i = ic*ND + 1; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd11, NB);
            i = ic*ND + 1; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd12, NB);
            i = ic*ND + 2; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd20, NB);
            i = ic*ND + 2; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd21, NB);
            i = ic*ND + 2; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd22, NB);
        }
    }
    stack.top = top;
}

__attribute__((noinline))
static void gemm(f32 * restrict c,
                 const f32 * restrict a,
                 const f32 * restrict b,
                 u32 n, u32 m, u32 l) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n,m,l, 1.0f, a,n, b,m, 0.0f, c,n);
}

static bool matrix_equal(f32 *a, f32 *b, u32 n, u32 m) {
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < m; ++j)
            if (fabs(a[i*m + j] - b[i*m + j]) >= EPSILON)
                return false;
    return true;
}

typedef void matmul_func(f32 * restrict, const f32 * restrict, const f32 * restrict, u32, u32, u32);

static inline void bench_matmul(matmul_func *mm, u32 iterations, f32 *c, f32 *a, f32 *b, u32 n, u32 m, u32 l) {
    struct timespec before = {0};
    struct timespec after  = {0};
    struct timespec diff   = {0};

    time_current(&before);

    for (u32 i = 0; i < iterations; ++i)
        mm(c, a, b, n,m,l);

    time_current(&after);
    time_diff(&diff, &after, &before);
    const size_t ns = time_nanoseconds(&diff) / iterations;
    const f64 useful_instructions = 1.0 * (f64) n * (f64) m * (f64) l;
    //printf("%s: %4llu s (%10llu us), %lf Gflops/s\n", name, ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);
    printf("%10llu", ns/1000ull);
    //printf("%10.2lf", useful_instructions / (f64) ns);
}

int main() {
    //{
    //    // 4x f32 packed vfma
    //    //
    //    // 2x vfma's, port 0 and port 1
    //    // 4 cycles to execute
    //    // => 0.5 vfma's per cycle
    //    //
    //    // num_vfmas = n*m*(l/4)
    //    // vfmas_per_s = 16 * 3.5e9 * 0.5
    //    const u32 n = 800;
    //    const u32 m = 800;
    //    const u32 l = 800;
    //    const f64 cpu_hz_high = 5.5e9;
    //    const f64 cpu_hz_low  = 3.0e9;
    //    const u32 num_threads = 16;
    //    const f64 throughput = 0.5;
    //    const f64 num_vfmas = (f64) n * (f64) m * (f64) l / 4.0;
    //    const f64 vfmas_per_s_mt_high = (num_threads * cpu_hz_high * throughput);
    //    const f64 vfmas_per_s_mt_low  = (num_threads * cpu_hz_low * throughput);
    //    const f64 vfmas_per_s_st_high = (cpu_hz_high * throughput);
    //    const f64 vfmas_per_s_st_low  = (cpu_hz_low * throughput);
    //    printf("Estimated optimal performance AMD Zen 4 (low: %.1f GHz, high %.1f GHz, %u threads)\n", cpu_hz_low/1e9, cpu_hz_high/1e9, num_threads);
    //    printf("  mt high %7.2lf s, %7.2lf Gflops/s\n", num_vfmas/vfmas_per_s_mt_high, vfmas_per_s_mt_high/1e9);
    //    printf("  mt low  %7.2lf s, %7.2lf Gflops/s\n", num_vfmas/vfmas_per_s_mt_low,  vfmas_per_s_mt_low/1e9);
    //    printf("  st high %7.2lf s, %7.2lf Gflops/s\n", num_vfmas/vfmas_per_s_st_high, vfmas_per_s_st_high/1e9);
    //    printf("  st low  %7.1lf s, %7.2lf Gflops/s\n", num_vfmas/vfmas_per_s_st_low,  vfmas_per_s_st_low/1e9);
    //    printf("\n");
    //}

    const u32 memory_size = 1 << 28;
    void *memory = malloc(memory_size);
    assert(memory != NULL);

    stack = (StackAllocator) {
        .memory = memory,
        .size = memory_size,
        .top = 0,
    };

    const u32 iterations = 10;
    const u32 step = 10;
    const u32 max_size = 1000;
    printf("%10s", "n");
    printf("%10s", "v0");
    printf("%10s", "v1");
    printf("%10s", "v2");
    printf("%10s", "v3");
    printf("%10s", "v4");
    printf("%10s", "v5");
    printf("%10s", "v6");
    printf("%10s", "gemm");
    printf("\n");
    for (u32 i = step; i <= max_size; i += step) {
        const u32 n = i;
        const u32 m = i;
        const u32 l = i;

        f32 *a  = allocate_memory(&stack, sizeof(f32) * n * m);
        f32 *b  = allocate_memory(&stack, sizeof(f32) * m * l);
        f32 *c  = allocate_memory(&stack, sizeof(f32) * n * l);
        f32 *d  = allocate_memory(&stack, sizeof(f32) * n * l);

        random_matrix(a, n,m);
        random_matrix(b, m,l);

        printf("%10u", i);

        bench_matmul(matmul_v0, iterations, c,a,b, n,m,l); memcpy(d, c, sizeof(f32)*n*l);
        bench_matmul(matmul_v1, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
        bench_matmul(matmul_v2, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
        bench_matmul(matmul_v3, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
        bench_matmul(matmul_v4, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
        bench_matmul(matmul_v5, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
        bench_matmul(matmul_v6, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
        bench_matmul(gemm,      iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));

        stack.top = 0;

        printf("\n");
    }

    return 0;
}
