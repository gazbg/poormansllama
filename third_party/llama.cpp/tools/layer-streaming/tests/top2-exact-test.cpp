// Host-only exactness test + timing of ls_top2_exact against the scalar loop.
#include "../top2-exact.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    std::mt19937_64 rng(12345);
    long long checked = 0, bad = 0;
    // 1) exactness on many small / medium arrays with heavy ties, -inf, and positions of the maximum anywhere
    for (int trial = 0; trial < 300000; ++trial) {
        const int n = 1 + (int) (rng() % (trial % 7 == 0 ? 4000 : 200));
        std::vector<float> x(n);
        const int levels = 1 + (int) (rng() % 12);
        for (int i = 0; i < n; ++i) {
            const int r = (int) (rng() % (levels + 1));
            x[i] = (float) r - 3.0f;
            if (rng() % 97 == 0) { x[i] = -INFINITY; }
        }
        if (rng() % 5 == 0 && n > 2) { x[rng() % n] = 100.0f + (float) (rng() % 3); }   // a unique-ish tall peak somewhere
        const ls_top2 a = ls_top2_scalar(x.data(), n);
        const ls_top2 b = ls_top2_exact(x.data(), n);
        ++checked;
        if (a.best != b.best || a.second != b.second || a.best_val != b.best_val || a.second_val != b.second_val) {
            if (++bad <= 5) {
                printf("MISMATCH n=%d scalar (%d %.1f | %d %.1f) exact (%d %.1f | %d %.1f)\n", n, a.best, a.best_val, a.second, a.second_val, b.best, b.best_val, b.second, b.second_val);
            }
        }
    }
    // 2) a NaN anywhere must fall back and still agree
    for (int trial = 0; trial < 2000; ++trial) {
        const int n = 64 + (int) (rng() % 1000);
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) { x[i] = (float) (rng() % 10); }
        x[rng() % n] = NAN;
        const ls_top2 a = ls_top2_scalar(x.data(), n);
        const ls_top2 b = ls_top2_exact(x.data(), n);
        ++checked;
        if (a.best != b.best || a.second != b.second) { ++bad; }
    }
    // 3) vocabulary-sized rows with realistic (non-tied) logits and a few exact ties at the top
    const int NV = 248320;
    std::vector<float> row(NV);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (int trial = 0; trial < 200; ++trial) {
        for (int i = 0; i < NV; ++i) { row[i] = nd(rng); }
        row[rng() % NV] = 30.0f; if (trial % 3 == 0) { row[rng() % NV] = 30.0f; }
        const ls_top2 a = ls_top2_scalar(row.data(), NV);
        const ls_top2 b = ls_top2_exact(row.data(), NV);
        ++checked;
        if (a.best != b.best || a.second != b.second || a.best_val != b.best_val || a.second_val != b.second_val) { ++bad; }
    }
    printf("checked %lld rows, mismatches %lld, avx2 %d\n", checked, bad, (int) ls_cpu_has_avx2());
    // timing on a vocabulary-sized row (result sink keeps the calls alive)
    for (int i = 0; i < NV; ++i) { row[i] = nd(rng); }
    int sink = 0;
    const int reps = 2000;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { row[r % NV] += 0.001f; sink += ls_top2_scalar(row.data(), NV).second; }
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { row[r % NV] += 0.001f; sink += ls_top2_exact(row.data(), NV).second; }
    auto t2 = std::chrono::steady_clock::now();
    // the old path also copied the row into a fresh std::vector each step
    for (int r = 0; r < reps; ++r) { std::vector<float> c(row.data(), row.data() + NV); sink += ls_top2_scalar(c.data(), NV).second; }
    auto t3 = std::chrono::steady_clock::now();
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count() / 2000.0; };
    printf("per row: scalar %.1f us | exact vectorised %.1f us | copy into a fresh vector + scalar (the old path) %.1f us   (sink %d)\n", us(t0, t1), us(t1, t2), us(t2, t3), sink);
    return bad == 0 ? 0 : 1;
}
