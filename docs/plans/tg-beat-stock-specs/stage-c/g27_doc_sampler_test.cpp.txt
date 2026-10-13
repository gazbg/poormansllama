// Host-only test of ls_doc_sample / ls_doc_sampler_candidates (G27): exactness of the candidate set against a straightforward sort-everything reference of llama.cpp's chain
// (top-k -> top-p -> min-p), the argmax special case, NaN handling, and the empirical distribution of the draw.
// Build:  cl /EHsc /O2 /std:c++17 doc-sampler-test.cpp     (or: g++ -O2 -std=c++17 doc-sampler-test.cpp)
#include "../doc-sampler.h"
#include "../top2-exact.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

// reference: sort ALL logits descending (stable: lower index first on ties), then the three truncations exactly as src/llama-sampler.cpp states them
static std::vector<int> reference_candidates(const std::vector<float> & x, const ls_doc_sampler_params & P) {
    std::vector<int> order(x.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[a] > x[b]; });
    size_t k = (size_t) P.top_k < order.size() ? (size_t) P.top_k : order.size();
    order.resize(k);
    // top-p
    if (order.size() > 1 && P.top_p < 1.0f) {
        const float mx = x[order[0]];
        double z = 0.0;
        for (int i : order) { z += std::exp((double) x[i] - mx); }
        double cum = 0.0;
        size_t keep = order.size();
        for (size_t i = 0; i < order.size(); ++i) {
            cum += std::exp((double) x[order[i]] - mx) / z;
            if (cum >= (double) P.top_p) { keep = i + 1; break; }
        }
        order.resize(keep);
    }
    // min-p
    if (order.size() > 1 && P.min_p > 0.0f) {
        const double min_logit = (double) x[order[0]] + std::log((double) P.min_p);
        size_t i = 1;
        for (; i < order.size(); ++i) {
            if ((double) x[order[i]] < min_logit) { break; }
        }
        order.resize(i);
    }
    return order;
}

int main() {
    std::mt19937_64 rng(777);
    ls_doc_sampler_params P;
    long long rows = 0, bad = 0, boundary = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        const int n = 1 + (int) (rng() % (trial % 11 == 0 ? 30 : 6000));
        std::vector<float> x((size_t) n);
        const int style = (int) (rng() % 4);
        for (int i = 0; i < n; ++i) {
            if (style == 0)      { x[(size_t) i] = (float) ((int) (rng() % 40) - 20) * 0.5f; }              // heavy ties
            else if (style == 1) { x[(size_t) i] = (float) ((double) (rng() % 100000) / 5000.0 - 10.0); }   // spread
            else if (style == 2) { x[(size_t) i] = -((float) (rng() % 1000) / 100.0f); }
            else                 { x[(size_t) i] = (float) ((int) (rng() % 7)) * 3.0f; }
        }
        if (rng() % 3 == 0 && n > 1) { x[rng() % (uint64_t) n] += 12.0f; }   // a tall peak
        int idx[64]; float lg[64];
        const int keep = ls_doc_sampler_candidates(x.data(), n, P, idx, lg);
        const std::vector<int> ref = reference_candidates(x, P);
        ++rows;
        bool same = (size_t) keep == ref.size();
        for (int i = 0; same && i < keep; ++i) {
            // equal logits may come out in any order, so compare the logit VALUES and the set of ties
            if (lg[i] != x[(size_t) ref[(size_t) i]]) { same = false; }
        }
        if (!same) {
            // a candidate set can legitimately differ only by a cumulative-probability value that sits within float rounding of p, or a min-p threshold within rounding
            ++boundary;
            if (++bad <= 5) { printf("MISMATCH n=%d style=%d keep=%d ref=%zu\n", n, style, keep, ref.size()); }
        }
    }
    printf("candidate sets: %lld rows, %lld mismatches (float vs double rounding at the p / min-p boundaries are counted as mismatches too)\n", rows, bad);

    // argmax special case: top_k = 1 is the greedy pick (lowest index among equal maxima, like ls_top2_scalar)
    long long am_bad = 0;
    ls_doc_sampler_params G; G.top_k = 1; G.top_p = 1.0f; G.min_p = 0.0f;
    for (int trial = 0; trial < 20000; ++trial) {
        const int n = 1 + (int) (rng() % 3000);
        std::vector<float> x((size_t) n);
        for (int i = 0; i < n; ++i) { x[(size_t) i] = (float) ((int) (rng() % 50)) * 0.25f; }
        const ls_top2 t = ls_top2_scalar(x.data(), n);
        const int s = ls_doc_sample(x.data(), n, G, rng);
        if (s != t.best) { if (++am_bad <= 5) { printf("ARGMAX MISMATCH n=%d sampled %d greedy %d\n", n, s, t.best); } }
    }
    printf("top_k = 1: %lld mismatches against the greedy pick\n", am_bad);

    // NaN rows: never selected
    long long nan_bad = 0;
    for (int trial = 0; trial < 5000; ++trial) {
        const int n = 8 + (int) (rng() % 500);
        std::vector<float> x((size_t) n);
        for (int i = 0; i < n; ++i) { x[(size_t) i] = (float) ((double) (rng() % 1000) / 100.0); }
        const int nanpos = (int) (rng() % (uint64_t) n);
        x[(size_t) nanpos] = std::nanf("");
        const int s = ls_doc_sample(x.data(), n, P, rng);
        if (s == nanpos || s < 0) { ++nan_bad; }
    }
    printf("NaN rows: %lld bad draws\n", nan_bad);

    // distribution: a fixed row, 400k draws, frequencies against the probabilities of the surviving candidates
    std::vector<float> x(2000, -30.0f);
    const float top[8] = {5.0f, 4.7f, 4.2f, 3.9f, 3.0f, 2.2f, 1.0f, 0.1f};
    for (int i = 0; i < 8; ++i) { x[(size_t) (100 + 37 * i)] = top[i]; }
    int idx[64]; float lg[64];
    const int keep = ls_doc_sampler_candidates(x.data(), (int) x.size(), P, idx, lg);
    std::vector<double> p((size_t) keep);
    double z = 0.0;
    for (int i = 0; i < keep; ++i) { p[(size_t) i] = std::exp((double) lg[i] - lg[0]); z += p[(size_t) i]; }
    std::vector<long long> cnt((size_t) keep, 0);
    const int N = 400000;
    for (int d = 0; d < N; ++d) {
        const int s = ls_doc_sample(x.data(), (int) x.size(), P, rng);
        for (int i = 0; i < keep; ++i) { if (s == idx[i]) { ++cnt[(size_t) i]; break; } }
    }
    double worst = 0.0;
    long long outside = N;
    for (int i = 0; i < keep; ++i) {
        outside -= cnt[(size_t) i];
        const double f = (double) cnt[(size_t) i] / N, e = p[(size_t) i] / z;
        worst = std::max(worst, std::fabs(f - e));
        printf("  cand %d token %d p %.4f observed %.4f\n", i, idx[i], e, f);
    }
    printf("distribution: %d survivors, worst |observed - expected| = %.4f, draws outside the survivors %lld\n", keep, worst, outside);
    const bool ok = bad == 0 && am_bad == 0 && nan_bad == 0 && worst < 0.004 && outside == 0 && keep >= 2;
    printf("doc-sampler test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
