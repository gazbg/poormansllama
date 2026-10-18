// Exact greedy top-2 over a logits row, vectorised (AVX2 when the CPU has it), with the SAME result as the scalar loop the runner has always used:
//
//     best = 0; best_val = x[0]; second_val = -INF; second_tok = 0;
//     for (v = 1; v < n; ++v) {
//         if (x[v] > best_val)        { second_val = best_val; second_tok = best; best_val = x[v]; best = v; }
//         else if (x[v] > second_val) { second_val = x[v];     second_tok = v; }
//     }
//
// i.e. best = the FIRST index of the maximum; second = the maximum of all the other elements, taking the earliest index among equals (an element equal to the maximum, found later,
// is the second). NaN anywhere, or a maximum of -INF, falls back to the scalar loop, so the result is identical in every case. Header-only, no dependencies.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

struct ls_top2 {
    int   best;
    float best_val;
    int   second;
    float second_val;
};

static inline ls_top2 ls_top2_scalar(const float * x, int n) {
    ls_top2 r;
    r.best       = 0;
    r.best_val   = x[0];
    r.second_val = -INFINITY;
    r.second     = 0;
    for (int v = 1; v < n; ++v) {
        if (x[v] > r.best_val) {
            r.second_val = r.best_val;
            r.second     = r.best;
            r.best_val   = x[v];
            r.best       = v;
        } else if (x[v] > r.second_val) {
            r.second_val = x[v];
            r.second     = v;
        }
    }
    return r;
}

static inline bool ls_cpu_has_avx2() {
#if defined(_MSC_VER)
    static const bool has = [] {
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) { return false; }
        __cpuid(r, 1);
        const bool osxsave = (r[2] & (1 << 27)) != 0;
        const bool avx     = (r[2] & (1 << 28)) != 0;
        if (!osxsave || !avx) { return false; }
        if ((_xgetbv(0) & 6) != 6) { return false; }   // OS saves XMM + YMM state
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;
    }();
    return has;
#else
    return false;
#endif
}

#if defined(_MSC_VER)
// max over [a, b) and whether any element is NaN
static inline float ls_max_avx2(const float * x, int a, int b, bool * has_nan) {
    __m256 m   = _mm256_set1_ps(-INFINITY);
    __m256 nan = _mm256_setzero_ps();
    int i = a;
    for (; i + 8 <= b; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        m   = _mm256_max_ps(m, v);
        nan = _mm256_or_ps(nan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, m);
    float r = lanes[0];
    for (int k = 1; k < 8; ++k) { r = lanes[k] > r ? lanes[k] : r; }
    bool n_ = _mm256_movemask_ps(nan) != 0;
    for (; i < b; ++i) {
        n_ = n_ || (x[i] != x[i]);
        r  = x[i] > r ? x[i] : r;
    }
    *has_nan = n_;
    return r;
}
#endif

static inline ls_top2 ls_top2_exact(const float * x, int n) {
#if defined(_MSC_VER)
    if (n >= 64 && ls_cpu_has_avx2()) {
        bool nan = false;
        const float M = ls_max_avx2(x, 0, n, &nan);
        if (!nan && M > -INFINITY) {
            // pass 2: the first two indices holding M, and the maximum of everything below M
            int   first = -1, second_eq = -1;
            const __m256 vM = _mm256_set1_ps(M);
            __m256 below = _mm256_set1_ps(-INFINITY);
            int i = 0;
            for (; i + 8 <= n && second_eq < 0; i += 8) {
                const __m256 v  = _mm256_loadu_ps(x + i);
                const int    eq = _mm256_movemask_ps(_mm256_cmp_ps(v, vM, _CMP_EQ_OQ));
                if (eq != 0) {
                    for (int k = 0; k < 8; ++k) {
                        if (eq & (1 << k)) {
                            if (first < 0) { first = i + k; } else if (second_eq < 0) { second_eq = i + k; }
                        }
                    }
                }
                below = _mm256_max_ps(below, _mm256_blendv_ps(_mm256_set1_ps(-INFINITY), v, _mm256_cmp_ps(v, vM, _CMP_LT_OQ)));
            }
            float lanes[8];
            _mm256_storeu_ps(lanes, below);
            float below_max = lanes[0];
            for (int k = 1; k < 8; ++k) { below_max = lanes[k] > below_max ? lanes[k] : below_max; }
            // a second element equal to M ends the search early: it is the runner-up (value M, earliest index after the first), nothing can beat it
            if (second_eq >= 0) {
                return ls_top2{ first, M, second_eq, M };
            }
            for (; i < n; ++i) {
                if (x[i] == M) {
                    if (first < 0) { first = i; } else if (second_eq < 0) { second_eq = i; }
                } else if (x[i] > below_max) {
                    below_max = x[i];
                }
            }
            if (second_eq >= 0) {
                return ls_top2{ first, M, second_eq, M };
            }
            // unique maximum: the runner-up is the largest remaining value, at its earliest index
            int sidx = -1;
            for (int j = 0; j < n; ++j) {
                if (j != first && x[j] == below_max) { sidx = j; break; }
            }
            if (sidx >= 0 && first >= 0) {
                return ls_top2{ first, M, sidx, below_max };
            }
        }
    }
#endif
    return ls_top2_scalar(x, n);
}
