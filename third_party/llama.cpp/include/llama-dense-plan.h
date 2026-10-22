#pragma once

// Dense decode ring planner (private fork). Pure arithmetic: no ggml, no CUDA, so a host test covers it.
//
// The ring of the dense decode (src/llama-dense-stream.h) computes some of the host layers on the device from VRAM slots that a copy stream refills while the CPU computes the other host layers. WHICH host
// layers are ring layers is the plan. The plan walks the timeline of one pass (one decode step): the layers in order, a CPU layer costs cpu_ms[il], a device layer (resident or ring) gpu_ms. When a ring layer
// ends, its slot is free and the copy of the next ring layer for that slot can start, as soon as the copy engine (one for all slots) has finished the copy before it. It takes copy_ms * (1 + margin). The next ring
// layer is the first host layer after the last ring layer whose predicted start is not before the copy is done, so the copy is hidden behind the layers that run in between. That is the owner's rule "let L CPU layers
// run, then the next one is loaded" with L replaced by measured time: it gives L = ceil((copy*(1+margin) - gpu) / cpu).
// The first n_slots ring layers are always the first n_slots host layers (they are loaded before the pass starts, so the copies queued at the end of the last pass stay valid whatever plan the next pass uses).
// A layer becomes a ring layer only when it pays: cpu_ms > gpu_ms + overhead_ms (the copy slows the CPU layers around it by about overhead_ms, measured 3 to 5 ms).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

namespace llama_dense_plan {

// n_resident layers of n_layer spread evenly and end-aligned: layer ((i+1)*n_layer)/n_resident - 1 (8 of 64 -> 7,15,...,63; 2 of 64 -> 31,63; the last layer is always resident, which saves the hop before the output head).
inline std::vector<int> resident_layers(int n_layer, int n_resident) {
    std::vector<int> out;
    if (n_layer <= 0 || n_resident <= 0) {
        return out;
    }
    if (n_resident >= n_layer) {
        for (int il = 0; il < n_layer; ++il) {
            out.push_back(il);
        }
        return out;
    }
    for (int i = 0; i < n_resident; ++i) {
        out.push_back((int) (((int64_t) (i + 1) * n_layer) / n_resident) - 1);
    }
    return out;
}

// the layers that are not in `resident` (ascending)
inline std::vector<int> host_layers(int n_layer, const std::vector<int> & resident) {
    std::vector<char> res((size_t) std::max(0, n_layer), 0);
    for (int il : resident) {
        if (il >= 0 && il < n_layer) {
            res[(size_t) il] = 1;
        }
    }
    std::vector<int> out;
    for (int il = 0; il < n_layer; ++il) {
        if (!res[(size_t) il]) {
            out.push_back(il);
        }
    }
    return out;
}

// percentile pct (0..100) of the samples, linear interpolation between ranks; 0 when there are none
inline double percentile(std::vector<double> v, double pct) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const double rank = std::min(1.0, std::max(0.0, pct / 100.0)) * (double) (v.size() - 1);
    const size_t lo = (size_t) std::floor(rank);
    const size_t hi = std::min(v.size() - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (rank - (double) lo);
}

// the last `cap` samples
struct window {
    std::deque<double> v;
    size_t             cap = 16;
    void   push(double x) { v.push_back(x); while (v.size() > cap) { v.pop_front(); } }
    size_t size() const   { return v.size(); }
    double pct(double p) const { return percentile(std::vector<double>(v.begin(), v.end()), p); }
    void   clear()        { v.clear(); }
};

struct inputs {
    int                 n_layer     = 0;
    std::vector<int>    host;                 // the host layers, ascending
    std::vector<double> cpu_ms;               // size n_layer: the time of the layer when the CPU computes it (any value for a layer that is not a host layer)
    double              gpu_ms      = 1.0;    // a layer on the device (resident or ring)
    double              copy_ms     = 32.0;   // the copy of one ring layer (the percentile estimate)
    double              margin      = 0.25;   // safety on the copy time
    double              overhead_ms = 2.0;    // what the copy costs the CPU layers around it, per ring layer
    int                 n_slots     = 2;
    double              gap_ms      = 3.0;    // host time between two passes (sampling, graph launch), for the copies that wrap around the end of a pass
};

struct outcome {
    std::vector<int> ring;                    // ascending; a multiple of n_slots long, or empty
    double           t_ms     = 0.0;          // predicted time of one pass, copies that arrive late included
    double           stall_ms = 0.0;          // predicted waiting for copies (inside the pass and at the wrap)
    int              late     = 0;            // ring layers whose copy is predicted unfinished when the layer starts
};

// the predicted time of one pass for the given ring (copy_ms is used as it is, without the margin)
inline outcome simulate(const inputs & in, const std::vector<int> & ring) {
    outcome o;
    o.ring = ring;
    std::vector<char> is_host((size_t) std::max(0, in.n_layer), 0);
    for (int il : in.host) {
        if (il >= 0 && il < in.n_layer) {
            is_host[(size_t) il] = 1;
        }
    }
    const int n = std::max(1, in.n_slots);
    const int K = (int) ring.size();
    std::vector<int> ord((size_t) std::max(0, in.n_layer), -1);
    for (int j = 0; j < K; ++j) {
        ord[(size_t) ring[(size_t) j]] = j;
    }
    std::vector<double> ready((size_t) K, 0.0);    // when the copy of ring layer j is done (the first n_slots are in place before the pass)
    std::vector<double> start((size_t) K, 0.0);
    std::vector<double> wrap_ready((size_t) n, 0.0);
    double t = 0.0, copy_free = 0.0, stall = 0.0;
    for (int il = 0; il < in.n_layer; ++il) {
        const int j = ord[(size_t) il];
        if (j >= 0) {
            if (ready[(size_t) j] > t) {
                stall += ready[(size_t) j] - t;
                t = ready[(size_t) j];
                o.late++;
            }
            start[(size_t) j] = t;
            t += in.gpu_ms;
            const double cd = std::max(t, copy_free) + in.copy_ms;
            copy_free = cd;
            if (j + n < K) {
                ready[(size_t) (j + n)] = cd;
            } else if (K >= n) {
                wrap_ready[(size_t) (j + n - K)] = cd;
            }
        } else if (is_host[(size_t) il]) {
            t += in.cpu_ms[(size_t) il];
        } else {
            t += in.gpu_ms;
        }
    }
    double wrap_stall = 0.0;
    for (int i = 0; i < n && i < K; ++i) {
        const double next_start = t + in.gap_ms + start[(size_t) i];
        wrap_stall = std::max(wrap_stall, wrap_ready[(size_t) i] - next_start);
        if (wrap_ready[(size_t) i] > next_start) {
            o.late++;
        }
    }
    o.t_ms     = t + wrap_stall;
    o.stall_ms = stall + wrap_stall;
    return o;
}

inline outcome plan(const inputs & in) {
    const int n = std::max(1, in.n_slots);
    if ((int) in.host.size() < n || in.n_layer <= 0) {
        return simulate(in, {});
    }
    std::vector<char> is_host((size_t) in.n_layer, 0);
    for (int il : in.host) {
        if (il >= 0 && il < in.n_layer) {
            is_host[(size_t) il] = 1;
        }
    }
    auto pays = [&](int il) { return in.cpu_ms[(size_t) il] > in.gpu_ms + in.overhead_ms; };

    std::vector<int> ring(in.host.begin(), in.host.begin() + n);
    for (int il : ring) {
        if (!pays(il)) {
            return simulate(in, {});
        }
    }
    std::vector<char> in_ring((size_t) in.n_layer, 0);
    for (int il : ring) {
        in_ring[(size_t) il] = 1;
    }
    // the start of layer q with the ring layers chosen so far (all of them lie before any candidate)
    auto time_to = [&](int q) {
        double t = 0.0;
        for (int il = 0; il < q; ++il) {
            t += (!in_ring[(size_t) il] && is_host[(size_t) il]) ? in.cpu_ms[(size_t) il] : in.gpu_ms;
        }
        return t;
    };
    const double D = in.copy_ms * (1.0 + in.margin);
    double copy_free = 0.0;
    for (size_t j = 0; j < ring.size(); ++j) {
        const double te = time_to(ring[j]) + in.gpu_ms;
        const double cd = std::max(te, copy_free) + D;
        int next = -1;
        for (int q : in.host) {
            if (q <= ring.back() || in_ring[(size_t) q] || !pays(q)) {
                continue;
            }
            if (time_to(q) >= cd) {
                next = q;
                break;
            }
        }
        if (next < 0) {
            break;
        }
        ring.push_back(next);
        in_ring[(size_t) next] = 1;
        copy_free = cd;
    }
    // the ring has to close at the end of a pass: a multiple of the slots; the cut layers stay CPU layers
    ring.resize(ring.size() / (size_t) n * (size_t) n);
    // the copies of the first ring layers for the next pass start at the end of the last ring layers: when they cannot be done in time, give up the last ring layers (a longer tail)
    while (ring.size() > (size_t) n && simulate(in, ring).stall_ms > 0.5) {
        ring.resize(ring.size() - (size_t) n);
    }
    return simulate(in, ring);
}


// The controller: the statistics of the previous passes and the decision to change the plan. Pure: the stream feeds it what it measured (copy times, GPU times of the ring layers, CPU ms per MiB of the host
// layers between ring layers, the fraction of the waits that found the copy unfinished) and asks it between two passes.
struct controller {
    // settings
    double pct         = 95.0;     // confidence: the copy at the p-pct end, the CPU / GPU layer times at the (100 - pct) end (a fast layer leaves the copy the least lead)
    double margin_base = 0.25;
    double overhead_ms = 2.0;
    double c_prior_ms  = 7.5;      // a CPU layer of the average size, until measured
    double g_prior_ms  = 1.0;
    double d_prior_ms  = 33.0;
    double gap_ms      = 3.0;
    int    replan_every = 2;       // passes between two decisions
    double gain_min     = 0.03;    // a plan that is not predicted faster by this much is not worth a graph rebuild
    int    dwell_min    = 8;       // passes a plan stands before a gain-driven change
    size_t window_passes = 16;
    int    n_slots     = 2;
    // layout
    int                 n_layer = 0;
    std::vector<int>    host;      // the candidate layers, ascending
    std::vector<double> mib;       // size n_layer: the package of a host layer in MiB
    // state
    double margin = 0.25;
    window w_cpu, w_gpu, w_copy, w_late;
    int    since = 0;
    int    dwell = 0;
    int    n_changes = 0;

    void configure() {
        margin = margin_base;
        dwell  = dwell_min;       // the first plan comes from priors: the first gain-driven change does not wait
        w_cpu.cap = w_late.cap = window_passes;
        w_gpu.cap = w_copy.cap = window_passes * 8;
    }
    double mean_mib() const {
        double s = 0.0;
        for (int il : host) { s += mib[(size_t) il]; }
        return host.empty() ? 1.0 : s / (double) host.size();
    }
    inputs make_inputs(double m) const {
        inputs in;
        in.n_layer = n_layer;
        in.host    = host;
        const double mm = std::max(1e-6, mean_mib());
        const double c_lo = w_cpu.size() >= 3 ? w_cpu.pct(100.0 - pct) : c_prior_ms / mm;     // ms per MiB
        in.cpu_ms.assign((size_t) std::max(0, n_layer), c_lo * mm);
        for (int il : host) {
            in.cpu_ms[(size_t) il] = c_lo * mib[(size_t) il];
        }
        in.gpu_ms      = std::max(0.3, w_gpu.size() >= 3 ? w_gpu.pct(100.0 - pct) : g_prior_ms);
        in.copy_ms     = w_copy.size() >= 3 ? w_copy.pct(pct) : d_prior_ms;
        in.margin      = m;
        in.overhead_ms = overhead_ms;
        in.n_slots     = n_slots;
        in.gap_ms      = gap_ms;
        return in;
    }
    void observe_copy(double ms)  { w_copy.push(ms); }
    void observe_gpu(double ms)   { w_gpu.push(ms); }
    void observe_pass(double ms_per_mib, double late_fraction) {
        if (ms_per_mib > 0.0) {
            w_cpu.push(ms_per_mib);
        }
        w_late.push(late_fraction);
    }
    double late_mean() const {
        double s = 0.0;
        for (double v : w_late.v) { s += v; }
        return w_late.size() > 0 ? s / (double) w_late.size() : 0.0;
    }
    // Between two passes: true = adopt `next` (a plan with a different ring that is predicted faster by at least 1%, or the current one is predicted late). The late-copy feedback runs here: copies that
    // were late widen the margin, a clean window narrows it again.
    bool decide(const std::vector<int> & ring, std::vector<int> & next, outcome * nw_out = nullptr, outcome * cur_out = nullptr) {
        if (w_cpu.size() < 3 || w_copy.size() < 3) {
            return false;
        }
        if (++since < replan_every) {
            return false;
        }
        since = 0;
        dwell += replan_every;
        const double late = late_mean();
        if (w_late.size() >= 4 && late > 0.10 && margin < 0.6) {
            margin = std::min(0.6, margin + 0.10);
            w_late.clear();
        } else if (w_late.size() >= window_passes / 2 && late == 0.0 && margin > margin_base) {
            margin = std::max(margin_base, margin - 0.05);
            w_late.clear();
        }
        const inputs  in  = make_inputs(margin);
        const outcome nw  = plan(in);
        const outcome cur = simulate(in, ring);
        if (nw_out)  { *nw_out = nw; }
        if (cur_out) { *cur_out = cur; }
        if (nw.ring.empty() || nw.ring == ring) {
            return false;
        }
        // Leaving the current plan: (a) it is predicted late AND copies were really late in the window (a copy-time spike alone moves the p95 and predicts a stall that never comes: measured, one 36.9 ms copy among
        // 33.5 ms ones changed an 8-layer ring to a 6-layer one and back), at once; or (b) the new one is predicted faster by gain_min, and the plan has stood for dwell_min passes (plans near the flat top differ by 1-2%,
        // which the statistics cannot tell apart: with 1% and no dwell the simulated controller alternated between 4 and 6 ring layers every few passes).
        const bool urgent = cur.stall_ms > 2.0 && late > 0.0;
        const bool better = nw.t_ms < cur.t_ms * (1.0 - gain_min) && dwell >= dwell_min;
        if (!urgent && !better) {
            return false;
        }
        dwell = 0;
        next = nw.ring;
        n_changes++;
        return true;
    }
};

} // namespace llama_dense_plan
