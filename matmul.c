#include <time.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef float f32;
typedef double f64;
typedef unsigned int u32;

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

static void random_matrix(f32 *m, u32 n) {
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < n; ++j)
            m[i*n + j] = 1.0f - 2.0f * ((f32) rand() / (f32) RAND_MAX);
}

__attribute__((noinline))
static void matmul_v0(f32 *c, f32 *a, f32 *b, u32 n) {
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            c[i*n + j] = 0;
            for (u32 k = 0; k < n; ++k) {
                asm("# here");
                c[i*n + j] += a[i*n + k] * b[k*n + j];
            }
        }
    }
}

__attribute__((noinline))
static void matmul_v1(f32 *c, f32 *a, f32 *b, u32 n) {
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            c[i*n + j] = 0;
            for (u32 k = 0; k < n; ++k) {
                asm("# here");
                c[i*n + j] += a[i*n + k] * b[k*n + j];
            }
        }
    }
}

__attribute__((noinline))
static void matmul_v2(f32 *c, f32 *a, f32 *b, u32 n) {
    f32 *tb = malloc(sizeof(f32) * n * n);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < n; ++j)
            tb[i*n + j] = b[j*n + i];

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            c[i*n + j] = 0;
            for (u32 k = 0; k < n; ++k) {
                asm("# here");
                c[i*n + j] += a[i*n + k] * tb[j*n + k];
            }
        }
    }
    free(tb);
}

__attribute__((noinline))
static void matmul_v3(f32 *c, f32 *a, f32 *b, u32 n) {
    const u32 nb = 8;
    const u32 na = (n + nb - 1) / nb;
    const u32 nab = na*nb;

    f32 *pa = malloc(sizeof(f32) * n * nab);
    f32 *ptb = malloc(sizeof(f32) * n * nab);
    memset(pa, 0, sizeof(f32) * n * nab);
    memset(ptb, 0, sizeof(f32) * n * nab);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            pa[i*nab + j] = a[i*n + j];
            ptb[i*nab + j] = b[j*n + i];
        }
    }

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            f32 d[nb];
            memset(&d, 0, sizeof(f32)*nb);

            for (u32 ka = 0; ka < na; ++ka) {
                for (u32 kb = 0; kb < nb; ++kb) {
                    asm("# here");
                    d[kb] += pa[i*nab + ka*nb + kb] * ptb[j*nab + ka*nb + kb];
                }
            }

            c[i*n + j] = 0;
            for (u32 k = 0; k < nb; ++k) {
                c[i*n + j] += d[k];
            }
        }
    }
    free(pa);
    free(ptb);
}

#include <x86intrin.h>

static inline float *float_8x_malloc(u32 n) {
    void *ptr = NULL;
    assert(posix_memalign(&ptr, sizeof(float)*8, sizeof(float)*8*n) == 0);
    return (float *) ptr;
}

__attribute__((noinline))
static void matmul_v4(f32 *c, f32 *a, f32 *b, u32 n) {
    const u32 nb = 8;
    const u32 na = (n + nb - 1) / nb;
    const u32 nab = na*nb;

    f32 *pa  = float_8x_malloc(n * na);
    f32 *ptb = float_8x_malloc(n * na);
    memset(pa, 0, sizeof(f32)*8 * n * na);
    memset(ptb, 0, sizeof(f32)*8 * n * na);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            pa[i*nab + j] = a[i*n + j];
            ptb[i*nab + j] = b[j*n + i];
        }
    }

    f32 *d = float_8x_malloc(1);
    memset(d, 0, sizeof(f32)*8);

#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            __m256 vd = _mm256_loadu_ps(&d[0]);

            for (u32 ka = 0; ka < na; ++ka) {
                asm("# here");
                __m256 va = _mm256_loadu_ps(&pa[i*nab + ka*nb]);
                __m256 vb = _mm256_loadu_ps(&ptb[j*nab + ka*nb]);
                vd = _mm256_fmadd_ps(va, vb, vd);
            }

            //c[i*n + j] = hsum256_ps_avx(vd);
            c[i*n + j] = 0;
            for (int kb = 0; kb < nb; ++kb) {
                c[i*n + j] += vd[kb];
            }
        }
    }

    free(d);
    free(pa);
    free(ptb);
}

static inline f32 hadd(__m256 vd, u32 nb) {
    f32 sum = 0;
    for (int kb = 0; kb < nb; ++kb)
        sum += vd[kb];
    return sum;
}

__attribute__((noinline))
static void matmul_v5(f32 *c, f32 *a, f32 *b, u32 n) {
    const u32 nb = 8;
    const u32 na = (n + nb - 1) / nb;
    const u32 nab = na*nb;

    const u32 nd = 3;
    const u32 nc = (n + nd - 1) / nd;
    const u32 ncd = nc*nd;

    f32 *pa  = float_8x_malloc(ncd * na);
    f32 *ptb = float_8x_malloc(ncd * na);
    memset(pa, 0, sizeof(f32)*8 * ncd * na);
    memset(ptb, 0, sizeof(f32)*8 * ncd * na);
#pragma omp parallel for
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            pa[i*nab + j] = a[i*n + j];
            ptb[i*nab + j] = b[j*n + i];
        }
    }

    f32 *d = float_8x_malloc(nd*nd);
    memset(d, 0, sizeof(f32)*8*nd*nd);

#pragma omp parallel for
    for (u32 ic = 0; ic < nc; ++ic) {
        for (u32 jc = 0; jc < nc; ++jc) {
            __m256 vd00 = _mm256_loadu_ps(&d[0*nd + 0]);
            __m256 vd01 = _mm256_loadu_ps(&d[0*nd + 1]);
            __m256 vd02 = _mm256_loadu_ps(&d[0*nd + 2]);
            __m256 vd10 = _mm256_loadu_ps(&d[1*nd + 0]);
            __m256 vd11 = _mm256_loadu_ps(&d[1*nd + 1]);
            __m256 vd12 = _mm256_loadu_ps(&d[1*nd + 2]);
            __m256 vd20 = _mm256_loadu_ps(&d[2*nd + 0]);
            __m256 vd21 = _mm256_loadu_ps(&d[2*nd + 1]);
            __m256 vd22 = _mm256_loadu_ps(&d[2*nd + 2]);

            for (u32 ka = 0; ka < na; ++ka) {
                __m256 va0 = _mm256_loadu_ps( &pa[(ic*nd + 0)*na*nb + ka*nb]);
                __m256 va1 = _mm256_loadu_ps( &pa[(ic*nd + 1)*na*nb + ka*nb]);
                __m256 va2 = _mm256_loadu_ps( &pa[(ic*nd + 2)*na*nb + ka*nb]);
                __m256 vb0 = _mm256_loadu_ps(&ptb[(jc*nd + 0)*na*nb + ka*nb]);
                __m256 vb1 = _mm256_loadu_ps(&ptb[(jc*nd + 1)*na*nb + ka*nb]);
                __m256 vb2 = _mm256_loadu_ps(&ptb[(jc*nd + 2)*na*nb + ka*nb]);
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
            i = ic*nd + 0; j = jc*nd + 0; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd00, nb);
            i = ic*nd + 0; j = jc*nd + 1; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd01, nb);
            i = ic*nd + 0; j = jc*nd + 2; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd02, nb);
            i = ic*nd + 1; j = jc*nd + 0; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd10, nb);
            i = ic*nd + 1; j = jc*nd + 1; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd11, nb);
            i = ic*nd + 1; j = jc*nd + 2; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd12, nb);
            i = ic*nd + 2; j = jc*nd + 0; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd20, nb);
            i = ic*nd + 2; j = jc*nd + 1; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd21, nb);
            i = ic*nd + 2; j = jc*nd + 2; if (i < n && j < n) c[(i)*n + (j)] = hadd(vd22, nb);
        }
    }

    free(d);
    free(pa);
    free(ptb);
}

int main() {
    const u32 n = 4000;
    f32 *a = malloc(sizeof(f32) * n * n);
    f32 *b = malloc(sizeof(f32) * n * n);
    f32 *c = malloc(sizeof(f32) * n * n);
    f32 *d = malloc(sizeof(f32) * n * n);
    random_matrix(a, n);
    random_matrix(b, n);

    {
        // 4x f32 packed vfma
        //
        // 2x vfma's, port 0 and port 1
        // 4 cycles to execute
        // => 0.5 vfma's per cycle
        //
        // num_vfmas = n*n*(n/4)
        // vfmas_per_s = 16 * 3.5e9 * 0.5
        const f64 num_vfmas = (f64) n * (f64) n * (f64) n / 4.0;
        const f64 vfmas_per_s_mt_high = (16 * 4.1e9 * 0.5);
        const f64 vfmas_per_s_mt_low  = (16 * 1.7e9 * 0.5);
        const f64 vfmas_per_s_st_high = (4.1e9 * 0.5);
        const f64 vfmas_per_s_st_low  = (1.7e9 * 0.5);
        printf("Estimated optimal performance Ryzen 7 4750U\n");
        printf("  mt high %lf s, %lf Gops/s\n", num_vfmas/vfmas_per_s_mt_high, vfmas_per_s_mt_high/1e9);
        printf("  mt low  %lf s, %lf Gops/s\n", num_vfmas/vfmas_per_s_mt_low,  vfmas_per_s_mt_low/1e9);
        printf("  st high %lf s, %lf Gops/s\n", num_vfmas/vfmas_per_s_st_high, vfmas_per_s_st_high/1e9);
        printf("  st low  %lf s, %lf Gops/s\n", num_vfmas/vfmas_per_s_st_low,  vfmas_per_s_st_low/1e9);
        printf("\n");
    }


    struct timespec before = {0},
                    after = {0},
                    diff = {0};

    size_t ns = 0;
    f64 useful_instructions = 0;

    time_current(&before);
    matmul_v0(c, a, b, n);
    time_current(&after);
    time_diff(&diff, &after, &before);
    ns = time_nanoseconds(&diff);
    useful_instructions = 1.0 * (f64) n * (f64) n * (f64) n;
    printf("v0: %llu s (%llu us), %lf Gops/s\n", ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);

    time_current(&before);
    matmul_v1(c, a, b, n);
    time_current(&after);
    time_diff(&diff, &after, &before);
    ns = time_nanoseconds(&diff);
    useful_instructions = 1.0 * (f64) n * (f64) n * (f64) n;
    printf("v1: %llu s (%llu us), %lf Gops/s\n", ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);

    time_current(&before);
    matmul_v2(c, a, b, n);
    time_current(&after);
    time_diff(&diff, &after, &before);
    ns = time_nanoseconds(&diff);
    useful_instructions = 1.0 * (f64) n * (f64) n * (f64) n;
    printf("v2: %llu s (%llu us), %lf Gops/s\n", ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);

    time_current(&before);
    matmul_v3(c, a, b, n);
    time_current(&after);
    time_diff(&diff, &after, &before);
    ns = time_nanoseconds(&diff);
    useful_instructions = 1.0 * (f64) n * (f64) n * (f64) n;
    printf("v3: %llu s (%llu us), %lf Gops/s\n", ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);

    //memcpy(d, c, sizeof(f32)*n*n);

    time_current(&before);
    matmul_v4(c, a, b, n);
    time_current(&after);
    time_diff(&diff, &after, &before);
    ns = time_nanoseconds(&diff);
    useful_instructions = 1.0 * (f64) n * (f64) n * (f64) n;
    printf("v4: %llu s (%llu us), %lf Gops/s\n", ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);

    //memcpy(d, c, sizeof(f32)*n*n);

    time_current(&before);
    matmul_v5(c, a, b, n);
    time_current(&after);
    time_diff(&diff, &after, &before);
    ns = time_nanoseconds(&diff);
    useful_instructions = 1.0 * (f64) n * (f64) n * (f64) n;
    printf("v5: %llu s (%llu us), %lf Gops/s\n", ns/1000000000ull, ns/1000ull, useful_instructions/(f64) ns);

    //for (int i = 0; i < n; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //        assert(fabs(d[i*n + j] - c[i*n + j]) < 1e-4);
    //    }
    //}

    free(a);
    free(b);
    free(c);
    free(d);

    return 0;
}
