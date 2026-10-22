// Host-only unit test for the dense decode ring planner (include/llama-dense-plan.h): the resident split, the percentile helper and the plan.
// No GPU, no model, no ggml.
#include "../include/llama-dense-plan.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace llama_dense_plan;

static int failures = 0;

static std::string str(const std::vector<int> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

static void expect(bool ok, const char * what, const std::string & detail = "") {
    if (!ok) {
        fprintf(stderr, "FAIL %s %s\n", what, detail.c_str());
        failures++;
    }
}

static void expect_ring(const char * what, const std::vector<int> & got, const std::vector<int> & want) {
    expect(got == want, what, "got [" + str(got) + "] want [" + str(want) + "]");
}

static inputs make(int n_layer, const std::vector<int> & resident, double c, double g, double copy, double margin) {
    inputs in;
    in.n_layer = n_layer;
    in.host    = host_layers(n_layer, resident);
    in.cpu_ms.assign((size_t) n_layer, c);
    in.gpu_ms  = g;
    in.copy_ms = copy;
    in.margin  = margin;
    return in;
}

int main() {
    // --- the resident split (AC-1)
    expect_ring("resident 8 of 64",  resident_layers(64, 8),  {7, 15, 23, 31, 39, 47, 55, 63});
    expect_ring("resident 2 of 64",  resident_layers(64, 2),  {31, 63});
    expect_ring("resident 1 of 64",  resident_layers(64, 1),  {63});
    {
        std::vector<int> want;
        for (int il = 3; il < 64; il += 4) {
            want.push_back(il);
        }
        expect_ring("resident 16 of 64 = the attention layers", resident_layers(64, 16), want);
    }
    expect(resident_layers(64, 64).size() == 64 && resident_layers(64, 100).size() == 64, "resident N >= n_layer is all");
    expect(resident_layers(64, 0).empty(), "resident 0 is none");
    for (int n_res = 1; n_res <= 64; ++n_res) {
        const std::vector<int> r = resident_layers(64, n_res);
        bool ok = (int) r.size() == n_res && r.back() == 63;
        for (size_t i = 1; i < r.size(); ++i) {
            ok = ok && r[i] > r[i - 1];
        }
        expect(ok, "resident split ascending, N long, ends at the last layer", "N=" + std::to_string(n_res));
        const std::vector<int> h = host_layers(64, r);
        expect((int) h.size() == 64 - n_res, "host is the complement", "N=" + std::to_string(n_res));
    }
    // 63 resident layers on an odd model: still distinct and ascending
    {
        const std::vector<int> r = resident_layers(61, 7);
        bool ok = r.size() == 7 && r.back() == 60;
        for (size_t i = 1; i < r.size(); ++i) {
            ok = ok && r[i] > r[i - 1];
        }
        expect(ok, "resident 7 of 61");
    }

    // --- the percentile helper
    {
        std::vector<double> v;
        for (int i = 1; i <= 100; ++i) {
            v.push_back((double) i);
        }
        expect(std::abs(percentile(v, 0) - 1.0) < 1e-9, "p0");
        expect(std::abs(percentile(v, 100) - 100.0) < 1e-9, "p100");
        expect(std::abs(percentile(v, 50) - 50.5) < 1e-9, "p50");
        expect(std::abs(percentile(v, 95) - 95.05) < 1e-9, "p95");
        expect(std::abs(percentile(v, 5) - 5.95) < 1e-9, "p5");
        expect(percentile({}, 95) == 0.0, "empty");
        expect(percentile({7.0}, 95) == 7.0, "one sample");
        window w;
        w.cap = 3;
        for (int i = 1; i <= 5; ++i) {
            w.push((double) i);
        }
        expect(w.size() == 3 && std::abs(w.pct(0) - 3.0) < 1e-9, "window keeps the last cap samples");
    }

    // --- the plan (AC-2)
    // 8 resident (res:8): 7.5 ms per CPU layer in situ, copy 32 ms, margin 0.25. The measured flat top of the static plans is 8 to 10 ring layers (spaced:5..7, 2.46-2.49 tok/s)
    {
        const inputs in = make(64, resident_layers(64, 8), 7.5, 1.0, 32.0, 0.25);
        const outcome o = plan(in);
        expect(o.ring.size() >= 8 && o.ring.size() <= 10, "res:8 ring size in the measured flat top (8-10)", str(o.ring));
        expect(o.ring.size() >= 2 && o.ring[0] == 0 && o.ring[1] == 1, "res:8 starts with the first two host layers", str(o.ring));
        expect(o.stall_ms <= 0.5, "res:8 plan has no predicted stall", std::to_string(o.stall_ms));
    }
    // the best MTP command layout: host 0-37 (26 resident), 8 ms per CPU layer. The measured flat top is 6 to 8 ring layers (spaced:5 on 0-35 / 0-37)
    {
        std::vector<int> resident;
        for (int il = 38; il < 64; ++il) {
            resident.push_back(il);
        }
        const inputs in = make(64, resident, 8.0, 1.0, 32.0, 0.25);
        const outcome o = plan(in);
        expect(o.ring.size() >= 6 && o.ring.size() <= 8, "host 0-37 ring size in the measured flat top (6-8)", str(o.ring));
        expect(o.ring.size() >= 2 && o.ring[0] == 0 && o.ring[1] == 1, "host 0-37 starts with the first two host layers", str(o.ring));
        expect(o.stall_ms <= 0.5, "host 0-37 plan has no predicted stall", std::to_string(o.stall_ms));
    }
    // invariants over a grid
    for (int n_slots = 2; n_slots <= 3; ++n_slots) {
        for (int n_res : {2, 8, 16, 28}) {
            for (double c : {4.0, 7.0, 10.0, 16.0}) {
                for (double d : {10.0, 20.0, 32.0, 45.0, 70.0}) {
                    for (double m : {0.0, 0.25, 0.6}) {
                        inputs in = make(64, resident_layers(64, n_res), c, 1.0, d, m);
                        in.n_slots = n_slots;
                        const outcome o = plan(in);
                        char tag[160];
                        snprintf(tag, sizeof(tag), "slots=%d res=%d c=%.1f d=%.1f m=%.2f", n_slots, n_res, c, d, m);
                        bool asc = true;
                        for (size_t i = 1; i < o.ring.size(); ++i) {
                            asc = asc && o.ring[i] > o.ring[i - 1];
                        }
                        expect(asc, "ring ascending", tag);
                        expect(o.ring.size() % (size_t) n_slots == 0, "ring a multiple of the slots", tag);
                        if (!o.ring.empty()) {
                            for (int i = 0; i < n_slots; ++i) {
                                expect(o.ring[(size_t) i] == in.host[(size_t) i], "first ring layers are the first host layers", tag);
                            }
                            for (int il : o.ring) {
                                expect(std::find(in.host.begin(), in.host.end(), il) != in.host.end(), "ring layers are host layers", tag);
                            }
                            if (o.ring.size() > (size_t) n_slots) {
                                expect(o.stall_ms <= 0.5, "no predicted stall", std::string(tag) + " stall=" + std::to_string(o.stall_ms));
                            }
                        }
                        // a ring that does not pay is not built
                        if (c <= 1.0 + in.overhead_ms) {
                            expect(o.ring.empty(), "no ring when the CPU layer is not slower than the device layer plus the interference", tag);
                        }
                    }
                }
            }
        }
    }
    // a longer copy never gives more ring layers
    {
        size_t prev = 1000;
        for (double d = 5.0; d <= 120.0; d += 5.0) {
            const outcome o = plan(make(64, resident_layers(64, 8), 7.5, 1.0, d, 0.25));
            expect(o.ring.size() <= prev, "ring size does not grow with the copy time", "d=" + std::to_string(d));
            prev = o.ring.size();
        }
    }
    // a slower CPU layer gives at least as many ring layers (more to gain, shorter lead needed)
    {
        size_t prev = 0;
        for (double c = 3.0; c <= 20.0; c += 1.0) {
            const outcome o = plan(make(64, resident_layers(64, 8), c, 1.0, 32.0, 0.25));
            expect(o.ring.size() >= prev, "ring size does not shrink with a slower CPU layer", "c=" + std::to_string(c));
            prev = o.ring.size();
        }
    }
    // fewer host layers than slots: no ring
    expect(plan(make(64, resident_layers(64, 63), 8.0, 1.0, 32.0, 0.25)).ring.empty(), "one host layer, two slots: no ring");
    // simulate: a hand-built late plan reports its stall, the planned one does not
    {
        const inputs in = make(64, resident_layers(64, 8), 7.5, 1.0, 32.0, 0.25);
        const outcome late = simulate(in, {0, 1, 2, 3});
        expect(late.stall_ms > 10.0, "a ring of adjacent layers is predicted late", std::to_string(late.stall_ms));
        const outcome ok = simulate(in, plan(in).ring);
        expect(ok.stall_ms <= 0.5, "the planned ring is predicted on time", std::to_string(ok.stall_ms));
        const outcome none = simulate(in, {});
        expect(ok.t_ms < none.t_ms, "the planned ring is predicted faster than no ring", std::to_string(ok.t_ms) + " vs " + std::to_string(none.t_ms));
    }

    // --- the controller on a synthetic machine (G-DYN, AC-3 stability in simulation): the plan settles, a slower CPU gives more ring layers, a slower copy fewer, late copies are not left alone
    {
        struct world { double c_per_mib; double g; double d; };
        uint64_t seed = 12345;
        auto rnd = [&]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return (double) ((seed >> 33) & 0xFFFFFF) / (double) 0x1000000; };
        auto noisy = [&](double v, double rel) { return v * (1.0 + rel * (2.0 * rnd() - 1.0)); };

        controller ctl;
        ctl.n_layer = 64;
        ctl.host    = host_layers(64, resident_layers(64, 8));
        ctl.mib.assign(64, 347.0);
        for (int il = 3; il < 64; il += 4) {
            ctl.mib[(size_t) il] = 357.0;
        }
        ctl.configure();
        std::vector<int> ring = plan(ctl.make_inputs(ctl.margin)).ring;
        expect(!ring.empty(), "controller: the first plan from the priors has ring layers");

        const double c_a = 7.5 / 347.0;
        const double phases[3][2] = { { 0, 100 }, { 100, 200 }, { 200, 320 } };
        const world W[3] = { { c_a, 0.8, 32.0 }, { 2.0 * c_a, 0.8, 32.0 }, { c_a, 0.8, 48.0 } };
        size_t size_end[3] = { 0, 0, 0 };
        int    changes_in[3] = { 0, 0, 0 };
        double late_last20[3] = { 0, 0, 0 };
        for (int ph = 0; ph < 3; ++ph) {
            std::vector<double> late_hist;
            for (int pass = (int) phases[ph][0]; pass < (int) phases[ph][1]; ++pass) {
                std::vector<int> next;
                if (ctl.decide(ring, next)) {
                    ring = next;
                    changes_in[ph]++;
                }
                // the true machine runs the pass
                inputs real;
                real.n_layer = 64;
                real.host    = ctl.host;
                real.cpu_ms.assign(64, 0.0);
                for (int il : ctl.host) {
                    real.cpu_ms[(size_t) il] = W[ph].c_per_mib * ctl.mib[(size_t) il];
                }
                real.gpu_ms = W[ph].g;
                real.copy_ms = W[ph].d;
                const outcome out = simulate(real, ring);
                const int K = (int) ring.size();
                for (int k = 0; k < K; ++k) {
                    double dd = noisy(W[ph].d, 0.06);
                    if (rnd() < 0.03) {
                        dd *= 1.5;                       // an occasional slow copy
                    }
                    ctl.observe_copy(dd);
                    ctl.observe_gpu(noisy(W[ph].g, 0.15));
                }
                const double late_fraction = K > 0 ? (double) out.late / (double) K : 0.0;
                ctl.observe_pass(noisy(W[ph].c_per_mib, 0.04), late_fraction);
                late_hist.push_back(late_fraction);
            }
            size_end[ph] = ring.size();
            double s = 0.0;
            const size_t n = late_hist.size();
            for (size_t i = n - 20; i < n; ++i) {
                s += late_hist[i];
            }
            late_last20[ph] = s / 20.0;
        }
        printf("controller simulation: ring layers at the end of the phases %zu / %zu / %zu, plan changes %d / %d / %d, late fraction (last 20 passes) %.3f / %.3f / %.3f, final margin %.2f\n",
               size_end[0], size_end[1], size_end[2], changes_in[0], changes_in[1], changes_in[2], late_last20[0], late_last20[1], late_last20[2], ctl.margin);
        expect(changes_in[0] <= 6, "controller: at most 6 plan changes in the first 100 passes", std::to_string(changes_in[0]));
        expect(size_end[0] >= 6 && size_end[0] <= 12, "controller: the settled ring is within +-2 of the measured flat top 8-10 (AC-3)", std::to_string(size_end[0]));
        expect(size_end[1] > size_end[0], "controller: a slower CPU gives more ring layers", std::to_string(size_end[0]) + " -> " + std::to_string(size_end[1]));
        expect(changes_in[1] <= 6, "controller: at most 6 plan changes after the CPU got slower", std::to_string(changes_in[1]));
        expect(changes_in[2] <= 6, "controller: at most 6 plan changes after the copy got slower", std::to_string(changes_in[2]));
        expect(size_end[2] < size_end[1], "controller: a slower copy (and a CPU back to normal) gives fewer ring layers than the slow-CPU phase", std::to_string(size_end[1]) + " -> " + std::to_string(size_end[2]));
        expect(late_last20[0] <= 0.05, "controller: late copies <= 5% once settled (phase 1)", std::to_string(late_last20[0]));
        expect(late_last20[1] <= 0.05, "controller: late copies <= 5% once settled (phase 2)", std::to_string(late_last20[1]));
        expect(late_last20[2] <= 0.05, "controller: late copies <= 5% once settled (phase 3)", std::to_string(late_last20[2]));
    }

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test-dense-ring-plan: all checks passed\n");
    return 0;
}
