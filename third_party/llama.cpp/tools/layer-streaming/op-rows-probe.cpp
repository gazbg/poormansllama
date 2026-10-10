// private research: does a MUL_MAT_ID output row depend on the batch row count?
//
// The layer-residency split computes the same routed experts in two legs (a 6-wide
// head and a 2-wide tail on Qwen3.6-35B-A3B) where the shipped path computes one
// 8-wide op. This probe runs those shapes over ONE quantized expert bank and reports
// the difference for the SAME rows and the SAME activations, so any difference can
// only come from the kernel's response to the row count (tiling, split-K, or a
// different kernel being selected at a different width).
//
// No model load and no llama context: the bank is synthesized at the model's real
// geometry and quantization type.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct op_run {
    int64_t            n_used = 0;
    std::vector<float> out;   // [m, n_used, n_tok]
};

struct cmp_res {
    double  max_abs   = 0.0;
    double  max_rel   = 0.0;
    double  med_rel   = 0.0;
    double  ref_scale = 0.0;
    int64_t n_elem    = 0;
    int64_t n_diff    = 0;
    bool    bit_equal = true;

    void merge(const cmp_res & o) {
        max_abs   = std::max(max_abs, o.max_abs);
        max_rel   = std::max(max_rel, o.max_rel);
        med_rel   = std::max(med_rel, o.med_rel);
        ref_scale = std::max(ref_scale, o.ref_scale);
        n_elem   += o.n_elem;
        n_diff   += o.n_diff;
        bit_equal = bit_equal && o.bit_equal;
    }
};

static void fill_f32(std::vector<float> & v, uint32_t seed, float lo, float hi) {
    uint32_t s = seed ? seed : 1u;
    for (float & x : v) {
        s = s * 1664525u + 1013904223u;
        const float u = (float) ((s >> 8) & 0xFFFFFFu) / (float) 0xFFFFFFu;
        x = lo + u * (hi - lo);
    }
}

// quantize an F32 block-wise into the layout ggml expects for `type`
static std::vector<uint8_t> quantize_all(ggml_type type, const std::vector<float> & src) {
    const int64_t blck = ggml_blck_size(type);
    std::vector<uint8_t> dst(ggml_row_size(type, (int64_t) src.size()));
    for (int64_t b = 0; b < (int64_t) src.size() / blck; ++b) {
        ggml_quantize_chunk(type, src.data(), dst.data(), b * blck, 1, blck, nullptr);
    }
    return dst;
}

static bool run_case(ggml_backend_t backend, ggml_type type_a,
                     int64_t k, int64_t m, int64_t n_mats, int64_t n_tok,
                     const std::vector<uint8_t> & bank,
                     const std::vector<float> & act,       // [k, n_pos, n_tok]
                     const std::vector<int32_t> & ids,     // [n_pos, n_tok]
                     op_run & res,
                     std::string & err) {
    const int64_t n_pos = (int64_t) ids.size() / n_tok;
    const size_t ctx_bytes = 32u * 1024u * 1024u + bank.size() + act.size() * sizeof(float) * 2;

    ggml_init_params ip = { ctx_bytes, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { err = "ggml_init failed"; return false; }

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
    ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_pos, n_tok);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_pos, n_tok);
    ggml_set_name(as, "exps");
    ggml_set_name(idt, "ids");
    ggml_set_name(b, "act");
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, idt);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    bool ok = false;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    do {
        if (buf == nullptr) { err = "backend buffer allocation failed"; break; }
        if (!ggml_backend_supports_op(backend, out)) { err = "backend does not support MUL_MAT_ID"; break; }
        if (ggml_nbytes(as) != bank.size()) {
            err = "bank byte size mismatch: " + std::to_string(ggml_nbytes(as)) + " != " + std::to_string(bank.size());
            break;
        }
        ggml_backend_tensor_set(as, bank.data(), 0, bank.size());
        ggml_backend_tensor_set(idt, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(b, act.data(), 0, act.size() * sizeof(float));
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) { err = std::string("compute failed: ") + ggml_status_to_string(st); break; }
        res.n_used = n_pos;
        res.out.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, res.out.data(), 0, res.out.size() * sizeof(float));
        ok = true;
    } while (false);

    if (buf) { ggml_backend_buffer_free(buf); }
    ggml_free(ctx);
    return ok;
}

// compare position pa of a against position pb of b, over every column and token
static cmp_res cmp_positions(const op_run & a, int64_t pa, const op_run & b, int64_t pb, int64_t m, int64_t n_tok) {
    cmp_res r;
    std::vector<double> rels;
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t c = 0; c < m; ++c) {
            const size_t ia = (size_t) (c + pa * m + t * m * a.n_used);
            const size_t ib = (size_t) (c + pb * m + t * m * b.n_used);
            const double va = (double) a.out[ia];
            const double vb = (double) b.out[ib];
            const double ad = std::fabs(va - vb);
            const double sc = std::max(std::fabs(va), std::fabs(vb));
            r.ref_scale = std::max(r.ref_scale, std::fabs(vb));
            r.max_abs   = std::max(r.max_abs, ad);
            r.n_elem   += 1;
            if (memcmp(&a.out[ia], &b.out[ib], sizeof(float)) != 0) { r.bit_equal = false; r.n_diff += 1; }
            if (sc > 0.0) { rels.push_back(ad / sc); r.max_rel = std::max(r.max_rel, ad / sc); }
        }
    }
    if (!rels.empty()) {
        std::sort(rels.begin(), rels.end());
        r.med_rel = rels[rels.size() / 2];
    }
    return r;
}

static cmp_res cmp_vectors(const std::vector<float> & a, const std::vector<float> & b) {
    cmp_res r;
    std::vector<double> rels;
    for (size_t i = 0; i < a.size(); ++i) {
        const double va = (double) a[i], vb = (double) b[i];
        const double ad = std::fabs(va - vb);
        const double sc = std::max(std::fabs(va), std::fabs(vb));
        r.ref_scale = std::max(r.ref_scale, std::fabs(vb));
        r.max_abs   = std::max(r.max_abs, ad);
        r.n_elem   += 1;
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) { r.bit_equal = false; r.n_diff += 1; }
        if (sc > 0.0) { rels.push_back(ad / sc); r.max_rel = std::max(r.max_rel, ad / sc); }
    }
    if (!rels.empty()) {
        std::sort(rels.begin(), rels.end());
        r.med_rel = rels[rels.size() / 2];
    }
    return r;
}

// weighted sum in the split's shape: legs[0] contributes its own positions, then legs[1]
static std::vector<float> weighted_sum(const std::vector<const op_run *> & legs,
                                       const std::vector<std::vector<int64_t>> & pos,
                                       const std::vector<double> & w,
                                       int64_t m, int64_t n_tok) {
    std::vector<float> acc((size_t) (m * n_tok), 0.0f);
    size_t wi = 0;
    for (size_t L = 0; L < legs.size(); ++L) {
        for (size_t p = 0; p < pos[L].size(); ++p) {
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t c = 0; c < m; ++c) {
                    const size_t src = (size_t) (c + pos[L][p] * m + t * m * legs[L]->n_used);
                    acc[(size_t) (c + t * m)] += (float) (w[wi] * (double) legs[L]->out[src]);
                }
            }
            wi += 1;
        }
    }
    return acc;
}

static const char * band(double max_rel) {
    if (max_rel <= 1e-6) { return "bit-or-fp32-exact"; }
    if (max_rel <= 1e-5) { return "negligible"; }
    if (max_rel <= 1e-4) { return "small"; }
    if (max_rel <= 1e-3) { return "material"; }
    return "large";
}

struct shape_result {
    int64_t     k = 0, m = 0;
    cmp_res     repeat;         // the 8-wide op run twice: the noise floor
    cmp_res     rc_2_vs_8;      // the same rows in a 2-wide batch against the 8-wide batch
    cmp_res     rc_1_vs_8;      // the same rows, one at a time
    cmp_res     rc_6_vs_8;
    cmp_res     split_vs_whole; // weighted 6+2 against weighted 8
    cmp_res     onegraph;       // both legs in ONE graph against the same legs alone
    bool        ok = false;
    std::string err;
};

static std::vector<int32_t> ids_slice(const std::vector<int32_t> & route_ids, int64_t from, int64_t n, int64_t n_tok) {
    std::vector<int32_t> v((size_t) (n * n_tok));
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t p = 0; p < n; ++p) { v[(size_t) (p + t * n)] = route_ids[(size_t) (from + p)]; }
    }
    return v;
}

// the activations for `n` positions of each token (one vector per token, repeated)
static std::vector<float> act_for(const std::vector<float> & one_per_token, int64_t k, int64_t n, int64_t n_tok) {
    std::vector<float> v((size_t) (k * n * n_tok));
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t p = 0; p < n; ++p) {
            for (int64_t c = 0; c < k; ++c) { v[(size_t) (c + p * k + t * k * n)] = one_per_token[(size_t) (c + t * k)]; }
        }
    }
    return v;
}

static void run_shape(ggml_backend_t backend, ggml_type type_a, int64_t k, int64_t m, int64_t n_mats,
                      int64_t n_tok, const std::vector<int32_t> & route_ids, shape_result & sr) {
    const int64_t n_used = (int64_t) route_ids.size();
    sr.k = k;
    sr.m = m;

    std::vector<float> wf((size_t) (k * m * n_mats));
    std::vector<float> act1((size_t) (k * n_tok));      // one activation vector per token
    fill_f32(wf,   0x51ed2701u + (uint32_t) k, -0.05f, 0.05f);
    fill_f32(act1, 0x9e3779b9u + (uint32_t) k, -1.0f,   1.0f);
    const std::vector<uint8_t> bank = quantize_all(type_a, wf);

    op_run whole8, whole8_b, head6, tail2;
    std::string err;
    if (!run_case(backend, type_a, k, m, n_mats, n_tok, bank, act_for(act1, k, n_used, n_tok), ids_slice(route_ids, 0, n_used, n_tok), whole8,   err) ||
        !run_case(backend, type_a, k, m, n_mats, n_tok, bank, act_for(act1, k, n_used, n_tok), ids_slice(route_ids, 0, n_used, n_tok), whole8_b, err) ||
        !run_case(backend, type_a, k, m, n_mats, n_tok, bank, act_for(act1, k, 6, n_tok),      ids_slice(route_ids, 0, 6, n_tok),      head6,    err) ||
        !run_case(backend, type_a, k, m, n_mats, n_tok, bank, act_for(act1, k, 2, n_tok),      ids_slice(route_ids, n_used - 2, 2, n_tok), tail2, err)) {
        sr.err = err;
        return;
    }

    sr.repeat = cmp_positions(whole8_b, 0, whole8, 0, m, n_tok);
    for (int64_t p = 0; p < 2; ++p) {
        sr.rc_2_vs_8.merge(cmp_positions(tail2, p, whole8, n_used - 2 + p, m, n_tok));
    }
    for (int64_t p = 0; p < 6; ++p) {
        sr.rc_6_vs_8.merge(cmp_positions(head6, p, whole8, p, m, n_tok));
    }
    for (int64_t p = 0; p < n_used; ++p) {
        op_run one;
        if (!run_case(backend, type_a, k, m, n_mats, n_tok, bank, act_for(act1, k, 1, n_tok), ids_slice(route_ids, p, 1, n_tok), one, err)) {
            sr.err = err;
            return;
        }
        sr.rc_1_vs_8.merge(cmp_positions(one, 0, whole8, p, m, n_tok));
    }

    std::vector<double> w((size_t) n_used, 0.0);
    {
        double s = 0.0;
        for (int64_t p = 0; p < n_used; ++p) { w[(size_t) p] = 0.5 + 0.05 * (double) p; s += w[(size_t) p]; }
        for (int64_t p = 0; p < n_used; ++p) { w[(size_t) p] /= s; }
    }
    {
        const std::vector<std::vector<int64_t>> pos_w = { { 0, 1, 2, 3, 4, 5, 6, 7 } };
        const std::vector<std::vector<int64_t>> pos_s = { { 0, 1, 2, 3, 4, 5 }, { 0, 1 } };
        const std::vector<float> whole_sum = weighted_sum({ &whole8 }, pos_w, w, m, n_tok);
        const std::vector<float> split_sum = weighted_sum({ &head6, &tail2 }, pos_s, w, m, n_tok);
        sr.split_vs_whole = cmp_vectors(split_sum, whole_sum);
    }

    // both legs in ONE graph, which is the shipped split's shape
    {
        const size_t ctx_bytes = 32u * 1024u * 1024u + bank.size() + 4u * 1024u * 1024u;
        ggml_init_params ip = { ctx_bytes, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_tensor * idh = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, n_tok);
        ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_tok);
        ggml_tensor * bh  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 6, n_tok);
        ggml_tensor * bt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 2, n_tok);
        ggml_tensor * oh  = ggml_mul_mat_id(ctx, as, bh, idh);
        ggml_tensor * ot  = ggml_mul_mat_id(ctx, as, bt, idt);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, oh);
        ggml_build_forward_expand(gf, ot);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            sr.err = "one-graph buffer allocation failed";
        } else {
            const std::vector<int32_t> ih = ids_slice(route_ids, 0, 6, n_tok);
            const std::vector<int32_t> it = ids_slice(route_ids, n_used - 2, 2, n_tok);
            const std::vector<float> ah = act_for(act1, k, 6, n_tok);
            const std::vector<float> at = act_for(act1, k, 2, n_tok);
            ggml_backend_tensor_set(as,  bank.data(), 0, bank.size());
            ggml_backend_tensor_set(idh, ih.data(), 0, ih.size() * sizeof(int32_t));
            ggml_backend_tensor_set(idt, it.data(), 0, it.size() * sizeof(int32_t));
            ggml_backend_tensor_set(bh,  ah.data(), 0, ah.size() * sizeof(float));
            ggml_backend_tensor_set(bt,  at.data(), 0, at.size() * sizeof(float));
            const ggml_status st = ggml_backend_graph_compute(backend, gf);
            if (st != GGML_STATUS_SUCCESS) {
                sr.err = std::string("one-graph compute failed: ") + ggml_status_to_string(st);
            } else {
                op_run oh_r, ot_r;
                oh_r.n_used = 6; oh_r.out.assign((size_t) ggml_nelements(oh), 0.0f);
                ot_r.n_used = 2; ot_r.out.assign((size_t) ggml_nelements(ot), 0.0f);
                ggml_backend_tensor_get(oh, oh_r.out.data(), 0, oh_r.out.size() * sizeof(float));
                ggml_backend_tensor_get(ot, ot_r.out.data(), 0, ot_r.out.size() * sizeof(float));
                for (int64_t p = 0; p < 6; ++p) { sr.onegraph.merge(cmp_positions(oh_r, p, head6, p, m, n_tok)); }
                for (int64_t p = 0; p < 2; ++p) { sr.onegraph.merge(cmp_positions(ot_r, p, tail2, p, m, n_tok)); }
            }
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
    }

    if (sr.err.empty()) { sr.ok = true; }
}

static void emit(FILE * js, bool & first_json, const char * label, const cmp_res & r) {
    if (js == nullptr) { return; }
    fprintf(js, "%s\n    {\"what\": \"%s\", \"max_abs\": %.9g, \"max_rel\": %.9g, \"med_rel\": %.9g, "
                "\"ref_scale\": %.9g, \"bit_equal\": %s, \"n_elem\": %lld, \"n_diff\": %lld}",
            first_json ? "" : ",", label, r.max_abs, r.max_rel, r.med_rel, r.ref_scale,
            r.bit_equal ? "true" : "false", (long long) r.n_elem, (long long) r.n_diff);
    first_json = false;
}

static void show(const char * label, const cmp_res & r) {
    printf("  %-30s max_abs=%.3e max_rel=%.3e med_rel=%.3e scale=%.3e bits_equal=%s n_diff=%lld\n",
           label, r.max_abs, r.max_rel, r.med_rel, r.ref_scale, r.bit_equal ? "yes" : "no",
           (long long) r.n_diff);
}

int main(int argc, char ** argv) {
    std::string device = "CPU";
    std::string out_json;
    std::string shape  = "both";
    ggml_type   type_a = GGML_TYPE_Q4_K;
    int64_t     k        = 2048;   // n_embd
    int64_t     m        = 512;    // n_ff per expert
    int64_t     n_expert = 256;
    int64_t     n_tok    = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "-d" || a == "--device")   { device = next(); }
        else if (a == "-k")                 { k = atoll(next().c_str()); }
        else if (a == "--ff")               { m = atoll(next().c_str()); }
        else if (a == "--experts")          { n_expert = atoll(next().c_str()); }
        else if (a == "--tokens")           { n_tok = atoll(next().c_str()); }
        else if (a == "--shape")            { shape = next(); }
        else if (a == "--json")             { out_json = next(); }
        else if (a == "--type") {
            const std::string t = next();
            if      (t == "Q4_K") { type_a = GGML_TYPE_Q4_K; }
            else if (t == "Q6_K") { type_a = GGML_TYPE_Q6_K; }
            else if (t == "F16")  { type_a = GGML_TYPE_F16;  }
            else { fprintf(stderr, "unknown --type %s\n", t.c_str()); return 2; }
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    // the CPU backend is registered statically, so asking for it first keeps a
    // host-only run from initializing the CUDA driver just to enumerate devices
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
    if (dev == nullptr) {
        ggml_backend_load_all();
        dev = ggml_backend_dev_by_name(device.c_str());
    }
    if (dev == nullptr) {
        fprintf(stderr, "device '%s' not found; available:", device.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
        fprintf(stderr, "\n");
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) { fprintf(stderr, "failed to init backend %s\n", device.c_str()); return 2; }

    const std::vector<int32_t> route_ids = { 3, 17, 42, 88, 129, 200, 233, 250 };

    std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> shapes;
    if (shape == "both" || shape == "gate_up") { shapes.push_back({ "gate_up", { k, m } }); }
    if (shape == "both" || shape == "down")    { shapes.push_back({ "down",    { m, k } }); }
    if (shapes.empty()) { fprintf(stderr, "unknown --shape %s\n", shape.c_str()); return 2; }

    printf("op-rows-probe: device=%s type=%s bank_rows=%lld tokens=%lld\n",
           ggml_backend_name(backend), ggml_type_name(type_a), (long long) n_expert, (long long) n_tok);
    printf("routed positions=%zu (ids:", route_ids.size());
    for (int32_t e : route_ids) { printf(" %d", e); }
    printf(")\n");

    FILE * js = nullptr;
    if (!out_json.empty()) {
        js = fopen(out_json.c_str(), "wb");
        if (js == nullptr) { fprintf(stderr, "cannot write %s\n", out_json.c_str()); return 2; }
        fprintf(js, "{\n  \"probe\": \"op-rows-probe\", \"device\": \"%s\", \"type\": \"%s\", \"n_expert\": %lld, \"tokens\": %lld, \"shapes\": [",
                ggml_backend_name(backend), ggml_type_name(type_a), (long long) n_expert, (long long) n_tok);
    }

    bool any_fail   = false;
    bool first_json = true;
    for (const auto & sh : shapes) {
        shape_result sr;
        run_shape(backend, type_a, sh.second.first, sh.second.second, n_expert, n_tok, route_ids, sr);
        if (!sr.ok) {
            printf("  shape %-8s FAILED: %s\n", sh.first.c_str(), sr.err.c_str());
            any_fail = true;
            if (js) { fprintf(js, "%s\n    {\"name\": \"%s\", \"ok\": false, \"error\": \"%s\"}", first_json ? "" : ",", sh.first.c_str(), sr.err.c_str()); }
            first_json = false;
            continue;
        }
        printf("  shape %s: k=%lld m=%lld\n", sh.first.c_str(), (long long) sr.k, (long long) sr.m);
        if (js) { fprintf(js, "%s\n    {\"name\": \"%s\", \"ok\": true, \"k\": %lld, \"m\": %lld, \"comparisons\": [",
                          first_json ? "" : ",", sh.first.c_str(), (long long) sr.k, (long long) sr.m); }
        bool fj = true;
        show("same-op repeat (noise floor)", sr.repeat);          emit(js, fj, "same-op repeat (noise floor)", sr.repeat);
        show("same rows: 2-wide vs 8-wide",  sr.rc_2_vs_8);       emit(js, fj, "same rows: 2-wide vs 8-wide",  sr.rc_2_vs_8);
        show("same rows: 1-wide vs 8-wide",  sr.rc_1_vs_8);       emit(js, fj, "same rows: 1-wide vs 8-wide",  sr.rc_1_vs_8);
        show("same rows: 6-wide vs 8-wide",  sr.rc_6_vs_8);       emit(js, fj, "same rows: 6-wide vs 8-wide",  sr.rc_6_vs_8);
        show("weighted 6+2 vs weighted 8",   sr.split_vs_whole);  emit(js, fj, "weighted 6+2 vs weighted 8",   sr.split_vs_whole);
        show("legs in 1 graph vs alone",     sr.onegraph);        emit(js, fj, "legs in 1 graph vs alone",     sr.onegraph);
        if (js) { fprintf(js, "\n    ]}"); }
        first_json = false;

        printf("  %s row-count band: 2-wide vs 8-wide = %s (max_rel %.3e)\n",
               sh.first.c_str(), band(sr.rc_2_vs_8.max_rel), sr.rc_2_vs_8.max_rel);
        printf("  %s split band: weighted 6+2 vs 8 = %s (max_rel %.3e, fraction of the 1.3e-3 observation: %.4f)\n",
               sh.first.c_str(), band(sr.split_vs_whole.max_rel), sr.split_vs_whole.max_rel,
               sr.split_vs_whole.max_rel / 1.3e-3);
    }
    printf("op-rows-probe RESULT: %s\n", any_fail ? "FAILED" : "OK");
    if (js) { fprintf(js, "\n  ]\n}\n"); fclose(js); }

    ggml_backend_free(backend);
    return any_fail ? 1 : 0;
}
