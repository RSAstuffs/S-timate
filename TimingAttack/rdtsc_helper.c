/*
 * RDTSC Helper - Small C extension for high-precision CPU cycle counting
 * 
 * Compile with:
 *   gcc -shared -fPIC -o rdtsc_helper.so rdtsc_helper.c
 * 
 * Or use Python's ctypes to load it directly
 */

#include <stdint.h>

#ifdef __x86_64__
static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

uint64_t get_rdtsc(void) {
    return rdtsc();
}
#else
// Fallback for non-x86_64 architectures
#include <time.h>
uint64_t get_rdtsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif










