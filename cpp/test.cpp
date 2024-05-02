#define MM_IMPLEMENTATION
#define MM_ALLOCATE_ALIGNED mm_allocate_aligned
#define MM_INCLUDE_CBLAS
#include "matmul.h"

#include <time.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define EPSILON 1e-4

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
    u8 *unaligned_ptr = stack->memory + stack->top;
    u8 *aligned_ptr = (u8 *) (alignment*(((intptr_t) unaligned_ptr + alignment - 1)/alignment));
    intptr_t new_top = (intptr_t) aligned_ptr - (intptr_t) stack->memory;

    assert(new_top + size < stack->size);
    stack->top = new_top + size;

    assert((intptr_t) aligned_ptr % alignment == 0);

    return (void *) aligned_ptr;
}

f32 *mm_allocate_aligned(u32 alignment, u32 size) {
    return (f32 *) allocate_aligned_memory(&stack, alignment, size);
}

//
// Matrix utility functions
//

static void random_matrix(f32 *mat, u32 n, u32 m) {
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < m; ++j)
            mat[i*m + j] = 1.0f - 2.0f * ((f32) rand() / (f32) RAND_MAX);
}

static bool matrix_equal(f32 *a, f32 *b, u32 n, u32 m) {
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            const f32 abserr = fabs(a[i*m + j] - b[i*m + j]);
            if (abserr > EPSILON) {
                printf("\n%f vs %f\n'", a[i*m+j], b[i*m+j]);
                return false;
            }
        }
    }
    return true;
}

//typedef void matmul_func(f32 * restrict, const f32 * restrict, const f32 * restrict, u32, u32, u32);

static inline u64 rdtscp() {
    u64 tsc;
    __asm__ __volatile__(
        "rdtscp;"
        "shl $32, %%rdx;"
        "or %%rdx, %%rax"
        : "=a"(tsc)
        :
        : "%rcx", "%rdx"
    );
    return tsc;
}

//static inline void bench_matmul(matmul_func *mm, u32 iterations, f32 *c, f32 *a, f32 *b, u32 n, u32 m, u32 l) {
//    u64 before = rdtscp();
//    for (u32 i = 0; i < iterations; ++i)
//        mm(c, a, b, n,m,l);
//    u64 after = rdtscp();
//    //const f64 useful_instructions = 1.0 * (f64) n * (f64) m * (f64) l;
//
//    // Uncomment this for raw time values
//    printf("%16llu", (after-before));
//
//    // Uncomment this GFlops estimate
//    //printf("%10.2lf", useful_instructions / (f64) ns);
//}

int main() {
    const u32 memory_size = 1 << 28;
    void *memory = malloc(memory_size);
    assert(memory != NULL);

    //stack = (StackAllocator) {
    //    .memory = memory,
    //    .size = memory_size,
    //    .top = 0,
    //};

    //const u32 iterations = 1;
    //const u32 step = 10;
    //const u32 max_size = 800;
    //printf("%10s", "n");
    //printf("%16s", "v0");
    //printf("%16s", "v1");
    //printf("%16s", "v2");
    //printf("%16s", "v3");
    //printf("%16s", "v4");
    //printf("%16s", "v5");
    //printf("%16s", "v6");
    //printf("%16s", "gemm");
    //printf("\n");
    //for (u32 i = step; i <= max_size; i += step) {
    //    const u32 n = i;
    //    const u32 m = i;
    //    const u32 l = i;

    //    f32 *a  = allocate_memory(&stack, sizeof(f32) * n * m);
    //    f32 *b  = allocate_memory(&stack, sizeof(f32) * m * l);
    //    f32 *c  = allocate_memory(&stack, sizeof(f32) * n * l);
    //    f32 *d  = allocate_memory(&stack, sizeof(f32) * n * l);

    //    random_matrix(a, n,m);
    //    random_matrix(b, m,l);

    //    printf("%10u", i);

    //    bench_matmul(matmul_v0, iterations, c,a,b, n,m,l); memcpy(d, c, sizeof(f32)*n*l);
    //    bench_matmul(matmul_v1, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v2, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v3, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v4, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v5, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v6, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(gemm,      iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));

    //    stack.top = 0;

    //    printf("\n");
    //}

    return 0;
}
