// G27: the sampler chain of the owner's llama-server document over ONE logits row, header-only, no dependencies beyond the standard library.
//
// The document's server printed its chain as `logits -> ... -> top-k -> ... -> top-p -> min-p -> ... -> temp-ext -> dist` with top_k = 20, top_p = 0.95, min_p = 0.05, temp = 1.0
// (docs/llamacpp 8gb vram 262k base.md). This is the same chain with llama.cpp's own semantics (src/llama-sampler.cpp):
//   top-k  : keep the K largest logits, sorted descending                                   (llama_sampler_top_k_impl)
//   top-p  : softmax over the kept candidates; keep the shortest prefix whose cumulative probability reaches p (min_keep 1)   (llama_sampler_top_p_apply)
//   min-p  : keep the prefix whose logit is >= max_logit + ln(min_p), i.e. p_i >= min_p * p_max (min_keep 1)                  (llama_sampler_min_p_apply)
//   temp   : divide the kept logits by the temperature                                       (llama_sampler_temp_impl)
//   dist   : softmax over the survivors and draw one with a seeded engine                    (llama_sampler_dist_apply; the engine here is std::mt19937_64 with a hand-written inverse-CDF
//                                                                                              draw, so a seed is portable across standard libraries but does not reproduce llama.cpp's draw, only its distribution)
// The top-K selection is ONE pass over the row with an insertion into a sorted array of K entries (K <= 64): about 250k comparisons for a 248k-token vocabulary, tens of microseconds
// of scalar work per token once the array is warm, instead of the full partial sort over 248k candidates. NaN logits are skipped.
// Used by the runner's serial TG loop under --tg-sampler doc. Greedy decoding (the default) never calls it.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

struct ls_doc_sampler_params {
    int   top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float temp  = 1.0f;
};

// number of candidates left after top-k, top-p and min-p (the survivors the distribution is drawn from); fills `idx` / `logit` with them, sorted descending
static inline int ls_doc_sampler_candidates(const float * x, int n, const ls_doc_sampler_params & P, int * idx, float * logit) {
    const int K = (P.top_k > 0 && P.top_k < 64) ? (P.top_k < n ? P.top_k : n) : (n < 64 ? n : 64);
    int m = 0;
    for (int v = 0; v < n; ++v) {
        const float f = x[v];
        if (f != f) {
            continue;                                   // NaN
        }
        if (m == K && !(f > logit[K - 1])) {
            continue;                                   // not larger than the K-th: the common case after the first few hundred elements
        }
        int j = (m < K) ? m++ : K - 1;
        while (j > 0 && logit[j - 1] < f) {
            logit[j] = logit[j - 1];
            idx[j]   = idx[j - 1];
            --j;
        }
        logit[j] = f;
        idx[j]   = v;
    }
    int keep = m;
    if (keep > 1 && P.top_p < 1.0f) {
        const float mx = logit[0];
        float e[64];
        float z = 0.0f;
        for (int i = 0; i < keep; ++i) {
            e[i] = std::exp(logit[i] - mx);
            z += e[i];
        }
        float cum = 0.0f;
        for (int i = 0; i < keep; ++i) {
            cum += e[i] / z;
            if (cum >= P.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    if (keep > 1 && P.min_p > 0.0f) {
        const float min_logit = logit[0] + std::log(P.min_p);
        int i = 1;
        for (; i < keep; ++i) {
            if (logit[i] < min_logit) {
                break;
            }
        }
        keep = i;
    }
    return keep;
}

// the token the chain draws from the row `x` (n logits); returns -1 for an empty row
static inline int ls_doc_sample(const float * x, int n, const ls_doc_sampler_params & P, std::mt19937_64 & rng) {
    int   idx[64];
    float logit[64];
    const int keep = ls_doc_sampler_candidates(x, n, P, idx, logit);
    if (keep <= 0) {
        return -1;
    }
    if (keep == 1) {
        return idx[0];
    }
    const float t  = P.temp > 0.0f ? P.temp : 1.0f;
    const float mx = logit[0] / t;
    float p[64];
    for (int i = 0; i < keep; ++i) {
        p[i] = std::exp(logit[i] / t - mx);
    }
    // inverse-CDF draw with a hand-written uniform (std::discrete_distribution's mapping is implementation-defined, so a seed would not be portable across standard libraries)
    float z = 0.0f;
    for (int i = 0; i < keep; ++i) {
        z += p[i];
    }
    const double u = (double) (rng() >> 11) * (1.0 / 9007199254740992.0) * (double) z;
    double acc = 0.0;
    for (int i = 0; i < keep; ++i) {
        acc += (double) p[i];
        if (u < acc) {
            return idx[i];
        }
    }
    return idx[keep - 1];
}
