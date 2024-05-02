#include <stdint.h>
uint64_t rdtscp() {
    uint64_t tsc;
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
