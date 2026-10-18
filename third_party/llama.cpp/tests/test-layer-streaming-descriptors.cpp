// T-GPLR-01: host-only unit test for the architecture-neutral package
// descriptor, its validation, the capability probe, and the extracted Qwen3.5
// dense adapter. No CUDA, no model load: fake llama_layer structs are filled
// with tiny no_alloc ggml tensors and checked for exact member order against
// the reference package lists.
#include "../src/llama-layer-streaming.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

static int failures = 0;

static ggml_context * g_ctx = nullptr;

static ggml_tensor * fake(const char * name) {
    ggml_tensor * t = ggml_new_tensor_1d(g_ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, name);
    return t;
}

// reference member order from the pre-extraction implementation (REC_PACKAGE /
// ATT_PACKAGE / MTP_MEMBERS); the adapter must emit exactly this order.
static ggml_tensor * llama_layer::* REC_ORDER[] = {
    &llama_layer::attn_norm,   &llama_layer::attn_post_norm,
    &llama_layer::wqkv,        &llama_layer::wqkv_gate,
    &llama_layer::ssm_conv1d,  &llama_layer::ssm_dt,
    &llama_layer::ssm_a,       &llama_layer::ssm_beta,
    &llama_layer::ssm_alpha,   &llama_layer::ssm_norm,
    &llama_layer::ssm_out,
    &llama_layer::ffn_gate,    &llama_layer::ffn_down,
    &llama_layer::ffn_up,
    &llama_layer::wqkv_s,      &llama_layer::wqkv_gate_s,
    &llama_layer::ssm_out_s,   &llama_layer::ssm_alpha_s,
    &llama_layer::ssm_beta_s,
    &llama_layer::ffn_gate_s,  &llama_layer::ffn_up_s,
    &llama_layer::ffn_down_s,
};

static ggml_tensor * llama_layer::* ATT_ORDER[] = {
    &llama_layer::attn_norm,   &llama_layer::attn_post_norm,
    &llama_layer::wq,          &llama_layer::wk,
    &llama_layer::wv,          &llama_layer::wo,
    &llama_layer::attn_q_norm, &llama_layer::attn_k_norm,
    &llama_layer::ffn_gate,    &llama_layer::ffn_down,
    &llama_layer::ffn_up,
    &llama_layer::wq_s,        &llama_layer::wk_s,
    &llama_layer::wv_s,        &llama_layer::wo_s,
    &llama_layer::ffn_gate_s,  &llama_layer::ffn_up_s,
    &llama_layer::ffn_down_s,
};

// reference MTP order: 18 base members then 8 nextn members
struct mtp_ref {
    ggml_tensor * llama_layer::*       base;
    ggml_tensor * llama_layer_nextn::* nextn;
};
static const mtp_ref MTP_ORDER[] = {
    { &llama_layer::attn_norm },   { &llama_layer::attn_post_norm },
    { &llama_layer::wq },          { &llama_layer::wk },
    { &llama_layer::wv },          { &llama_layer::wo },
    { &llama_layer::attn_q_norm }, { &llama_layer::attn_k_norm },
    { &llama_layer::ffn_gate },    { &llama_layer::ffn_down },
    { &llama_layer::ffn_up },
    { &llama_layer::wq_s },        { &llama_layer::wk_s },
    { &llama_layer::wv_s },        { &llama_layer::wo_s },
    { &llama_layer::ffn_gate_s },  { &llama_layer::ffn_up_s },
    { &llama_layer::ffn_down_s },
    { nullptr, &llama_layer_nextn::eh_proj },          { nullptr, &llama_layer_nextn::eh_proj_s },
    { nullptr, &llama_layer_nextn::embed_tokens },     { nullptr, &llama_layer_nextn::enorm },
    { nullptr, &llama_layer_nextn::hnorm },
    { nullptr, &llama_layer_nextn::shared_head_head }, { nullptr, &llama_layer_nextn::shared_head_head_s },
    { nullptr, &llama_layer_nextn::shared_head_norm },
};

static void check_order(const char * what, const llama_ls_package & pkg,
                        const llama_layer & l,
                        ggml_tensor * llama_layer::* const * order, size_t n) {
    if (pkg.members.size() != n) {
        fprintf(stderr, "FAIL %s: %zu members, expected %zu\n", what, pkg.members.size(), n);
        failures++;
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        if (pkg.members[i].dst_base != order[i] || pkg.members[i].dst_nextn != nullptr) {
            fprintf(stderr, "FAIL %s: member %zu destination differs from reference order\n", what, i);
            failures++;
            return;
        }
        // the source must pair with the destination: a swapped pairing passes
        // an order-only check but copies the wrong tensor
        if (pkg.members[i].src != l.*(order[i])) {
            fprintf(stderr, "FAIL %s: member %zu src/dst pairing mismatch\n", what, i);
            failures++;
            return;
        }
    }
    fprintf(stderr, "ok   %s: %zu members in reference order\n", what, n);
}

int main() {
    ggml_init_params gp = { /*mem_size=*/1u * 1024u * 1024u, /*mem_buffer=*/nullptr, /*no_alloc=*/true };
    g_ctx = ggml_init(gp);
    if (g_ctx == nullptr) {
        fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }

    const llama_ls_adapter * ad = llama_ls_qwen35_adapter();
    if (ad == nullptr || ad->package == nullptr || ad->mtp_package == nullptr) {
        fprintf(stderr, "FAIL: qwen35 adapter missing\n");
        return 1;
    }

    // --- positive: recurrent layer package -----------------------------------
    {
        llama_layer l;
        l.attn_norm = fake("an"); l.attn_post_norm = fake("pn");
        l.wqkv = fake("qkv"); l.wqkv_gate = fake("qkvg");
        l.ssm_conv1d = fake("cv"); l.ssm_dt = fake("dt"); l.ssm_a = fake("a");
        l.ssm_beta = fake("be"); l.ssm_alpha = fake("al"); l.ssm_norm = fake("sn");
        l.ssm_out = fake("so");
        l.ffn_gate = fake("fg"); l.ffn_down = fake("fd"); l.ffn_up = fake("fu");
        l.wqkv_s = fake("qkvs"); l.wqkv_gate_s = fake("qkvg s");
        l.ssm_out_s = fake("sos"); l.ssm_alpha_s = fake("als"); l.ssm_beta_s = fake("bes");
        l.ffn_gate_s = fake("fgs"); l.ffn_up_s = fake("fus"); l.ffn_down_s = fake("fds");
        llama_ls_package pkg;
        std::string err;
        if (!ad->package(nullptr, 0, l, pkg, err)) {
            fprintf(stderr, "FAIL recurrent package: %s\n", err.c_str());
            failures++;
        } else {
            check_order("recurrent order", pkg, l, REC_ORDER, sizeof(REC_ORDER) / sizeof(REC_ORDER[0]));
            std::string verr;
            if (!llama_ls_validate_package(pkg, verr, /*require_src=*/true)) {
                fprintf(stderr, "FAIL recurrent validate: %s\n", verr.c_str());
                failures++;
            }
            if (std::string(pkg.cls) != "recurrent" || std::string(pkg.policy) != "dense-layer") {
                fprintf(stderr, "FAIL recurrent labels: %s/%s\n", pkg.cls, pkg.policy);
                failures++;
            }
        }
    }

    // --- positive: attention layer package (null _s members allowed) ----------
    {
        llama_layer l;
        l.attn_norm = fake("an"); l.attn_post_norm = fake("pn");
        l.wq = fake("q"); l.wk = fake("k"); l.wv = fake("v"); l.wo = fake("o");
        l.attn_q_norm = fake("qn"); l.attn_k_norm = fake("kn");
        l.ffn_gate = fake("fg"); l.ffn_down = fake("fd"); l.ffn_up = fake("fu");
        llama_ls_package pkg;
        std::string err;
        if (!ad->package(nullptr, 3, l, pkg, err)) {
            fprintf(stderr, "FAIL attention package: %s\n", err.c_str());
            failures++;
        } else {
            check_order("attention order", pkg, l, ATT_ORDER, sizeof(ATT_ORDER) / sizeof(ATT_ORDER[0]));
            std::string verr;
            if (!llama_ls_validate_package(pkg, verr, /*require_src=*/false)) {
                fprintf(stderr, "FAIL attention validate: %s\n", verr.c_str());
                failures++;
            }
            if (std::string(pkg.cls) != "attention") {
                fprintf(stderr, "FAIL attention label: %s\n", pkg.cls);
                failures++;
            }
        }
    }

    // --- positive: MTP package ------------------------------------------------
    {
        llama_layer l;
        l.attn_norm = fake("an"); l.wq = fake("q"); l.wk = fake("k"); l.wv = fake("v"); l.wo = fake("o");
        l.attn_q_norm = fake("qn"); l.attn_k_norm = fake("kn");
        l.ffn_gate = fake("fg"); l.ffn_down = fake("fd"); l.ffn_up = fake("fu");
        l.nextn.eh_proj = fake("eh"); l.nextn.embed_tokens = fake("et");
        l.nextn.enorm = fake("en"); l.nextn.hnorm = fake("hn");
        l.nextn.shared_head_head = fake("sh"); l.nextn.shared_head_norm = fake("shn");
        llama_ls_package pkg;
        std::string err;
        if (!ad->mtp_package(nullptr, l, pkg, err)) {
            fprintf(stderr, "FAIL mtp package: %s\n", err.c_str());
            failures++;
        } else if (pkg.members.size() != 26) {
            fprintf(stderr, "FAIL mtp members: %zu != 26\n", pkg.members.size());
            failures++;
        } else if (std::string(pkg.cls) != "mtp") {
            fprintf(stderr, "FAIL mtp label: %s\n", pkg.cls);
            failures++;
        } else {
            bool order_ok = true;
            for (size_t i = 0; i < 26; ++i) {
                const auto & m = pkg.members[i];
                if (m.dst_base != MTP_ORDER[i].base || m.dst_nextn != MTP_ORDER[i].nextn) {
                    fprintf(stderr, "FAIL mtp member %zu destination differs\n", i);
                    order_ok = false;
                    break;
                }
                const ggml_tensor * want = MTP_ORDER[i].base ? l.*(MTP_ORDER[i].base)
                                                             : l.nextn.*(MTP_ORDER[i].nextn);
                if (m.src != want) {
                    fprintf(stderr, "FAIL mtp member %zu src pairing mismatch\n", i);
                    order_ok = false;
                    break;
                }
            }
            if (order_ok) {
                fprintf(stderr, "ok   mtp package: 26 members in reference order\n");
            } else {
                failures++;
            }
        }
    }

    // --- negative: invalid descriptors must be rejected ------------------------
    auto expect_invalid = [&](const char * what, const llama_ls_package & pkg, bool require_src) {
        std::string err;
        if (llama_ls_validate_package(pkg, err, require_src)) {
            fprintf(stderr, "FAIL %s: descriptor accepted but must be rejected\n", what);
            failures++;
        } else {
            fprintf(stderr, "ok   %s rejected: %s\n", what, err.c_str());
        }
    };
    {
        llama_ls_package pkg;
        expect_invalid("empty package", pkg, false);
    }
    {
        llama_ls_package pkg;
        llama_ls_member m;
        m.src = fake("dup"); m.dst_base = &llama_layer::wq;
        pkg.members = { m, m };
        pkg.cls = "attention"; pkg.policy = "dense-layer";
        pkg.excludes_static = true; pkg.excludes_state = true;
        expect_invalid("duplicate destination", pkg, true);
    }
    {
        llama_ls_package pkg;
        llama_ls_member m;
        m.src = fake("mut"); m.dst_base = &llama_layer::wq; m.mutable_state = true;
        pkg.members = { m };
        pkg.cls = "attention"; pkg.policy = "dense-layer";
        pkg.excludes_static = true; pkg.excludes_state = true;
        expect_invalid("mutable member", pkg, true);
    }
    {
        llama_ls_package pkg;
        llama_ls_member m;
        m.src = fake("both"); m.dst_base = &llama_layer::wq; m.dst_nextn = &llama_layer_nextn::enorm;
        pkg.members = { m };
        pkg.cls = "attention"; pkg.policy = "dense-layer";
        pkg.excludes_static = true; pkg.excludes_state = true;
        expect_invalid("two destinations", pkg, true);
    }
    {
        llama_ls_package pkg;
        llama_ls_member m;
        m.src = ggml_new_tensor_1d(g_ctx, GGML_TYPE_F32, 0); // zero bytes
        m.dst_base = &llama_layer::wq;
        pkg.members = { m };
        pkg.cls = "attention"; pkg.policy = "dense-layer";
        pkg.excludes_static = true; pkg.excludes_state = true;
        expect_invalid("zero-byte source", pkg, true);
    }
    {
        llama_ls_package pkg;
        llama_ls_member m;
        m.src = fake("nostate"); m.dst_base = &llama_layer::wq;
        pkg.members = { m };
        pkg.cls = "attention"; pkg.policy = "dense-layer";
        pkg.excludes_static = true; pkg.excludes_state = false;
        expect_invalid("missing state exclusion", pkg, true);
    }
    {
        llama_ls_package pkg;
        llama_ls_member m;
        m.src = fake("nolabel"); m.dst_base = &llama_layer::wq;
        pkg.members = { m };
        pkg.cls = ""; pkg.policy = "";
        pkg.excludes_static = true; pkg.excludes_state = true;
        expect_invalid("missing labels", pkg, true);
    }

    // --- capability probe ------------------------------------------------------
    {
        std::string reason;
        const llama_ls_adapter * a = llama_ls_select_adapter(LLM_ARCH_QWEN35, reason);
        if (a == nullptr || std::string(a->name) != "qwen35") {
            fprintf(stderr, "FAIL probe: qwen35 adapter not selected (%s)\n", reason.c_str());
            failures++;
        }
        a = llama_ls_select_adapter(LLM_ARCH_QWEN35MOE, reason);
        if (a == nullptr || std::string(a->name) != "qwen35moe") {
            fprintf(stderr, "FAIL probe: qwen35moe adapter not selected (reason=%s)\n", reason.c_str());
            failures++;
        } else {
            fprintf(stderr, "ok   probe: qwen35moe adapter selected (policy %s)\n", a->policy);
        }
        a = llama_ls_select_adapter(LLM_ARCH_LLAMA, reason);
        if (a != nullptr || reason.empty()) {
            fprintf(stderr, "FAIL probe: llama must report unsupported with a reason\n");
            failures++;
        }
    }

    // --- T-GPLR-02: MoE expert-bank package ------------------------------------
    {
        const llama_ls_adapter * ad = llama_ls_qwen35moe_adapter();
        if (ad == nullptr || ad->package == nullptr) {
            fprintf(stderr, "FAIL: qwen35moe adapter missing\n");
            failures++;
        } else {
            // complete separate bank: gate+up+down, router/shexp/attn/ssm/norms excluded
            llama_layer l;
            l.ffn_gate_inp = fake("router");        // must be EXCLUDED
            l.ffn_gate_shexp = fake("shexp");        // excluded
            l.ffn_up_shexp = fake("shexpu");         // excluded
            l.attn_norm = fake("an");                // excluded
            l.ssm_a = fake("ssm");                   // excluded
            l.wq = fake("q");                        // excluded
            l.ffn_gate_exps = fake("ge");
            l.ffn_up_exps   = fake("ue");
            l.ffn_down_exps = fake("de");
            llama_ls_package pkg;
            std::string err;
            if (!ad->package(nullptr, 0, l, pkg, err)) {
                fprintf(stderr, "FAIL moe package: %s\n", err.c_str());
                failures++;
            } else if (pkg.members.size() != 3) {
                fprintf(stderr, "FAIL moe members: %zu != 3\n", pkg.members.size());
                failures++;
            } else if (std::string(pkg.cls) != "moe-expert-bank" ||
                       std::string(pkg.policy) != "moe-expert-bank") {
                fprintf(stderr, "FAIL moe labels: %s/%s\n", pkg.cls, pkg.policy);
                failures++;
            } else {
                ggml_tensor * llama_layer::* exps[3] = {
                    &llama_layer::ffn_gate_exps, &llama_layer::ffn_up_exps, &llama_layer::ffn_down_exps };
                bool ok = true;
                for (int i = 0; i < 3; ++i) {
                    if (pkg.members[i].dst_base != exps[i] || pkg.members[i].src != l.*(exps[i])) {
                        fprintf(stderr, "FAIL moe member %zu pairing\n", (size_t) i);
                        ok = false;
                    }
                }
                // no router/shexp/attn/ssm/norm member may appear
                for (const auto & m : pkg.members) {
                    if (m.dst_base == &llama_layer::ffn_gate_inp ||
                        m.dst_base == &llama_layer::ffn_gate_shexp ||
                        m.dst_base == &llama_layer::attn_norm ||
                        m.dst_base == &llama_layer::ssm_a ||
                        m.dst_base == &llama_layer::wq) {
                        fprintf(stderr, "FAIL moe package includes a non-expert member\n");
                        ok = false;
                    }
                }
                std::string verr;
                if (ok && !llama_ls_validate_package(pkg, verr, true)) {
                    fprintf(stderr, "FAIL moe validate: %s\n", verr.c_str());
                    ok = false;
                }
                if (ok) {
                    fprintf(stderr, "ok   moe separate bank: 3 experts, classes excluded\n");
                } else {
                    failures++;
                }
            }

            // fused gate_up + down bank
            llama_layer lf;
            lf.ffn_gate_up_exps = fake("gue");
            lf.ffn_down_exps    = fake("de");
            llama_ls_package pf;
            std::string fe;
            if (!ad->package(nullptr, 0, lf, pf, fe)) {
                fprintf(stderr, "FAIL moe fused package: %s\n", fe.c_str());
                failures++;
            } else if (pf.members.size() != 2 ||
                       pf.members[0].dst_base != &llama_layer::ffn_gate_up_exps ||
                       pf.members[1].dst_base != &llama_layer::ffn_down_exps) {
                fprintf(stderr, "FAIL moe fused members: %zu\n", pf.members.size());
                failures++;
            } else {
                fprintf(stderr, "ok   moe fused bank: 2 members\n");
            }

            // negative: missing one routed-expert tensor (partial bank) -> hard error
            llama_layer lp;
            lp.ffn_gate_exps = fake("ge");
            lp.ffn_up_exps   = fake("ue");
            // ffn_down_exps absent
            llama_ls_package pp;
            std::string pe;
            if (ad->package(nullptr, 5, lp, pp, pe) ||
                pe.find("incomplete routed-expert bank") == std::string::npos) {
                fprintf(stderr, "FAIL moe partial bank not rejected: ok=%d err=%s\n", (int) false, pe.c_str());
                failures++;
            } else if (!pp.members.empty()) {
                fprintf(stderr, "FAIL moe partial bank left members populated\n");
                failures++;
            } else {
                fprintf(stderr, "ok   moe partial bank rejected: %s\n", pe.c_str());
            }

            // negative: zero-byte routed-expert tensor -> hard error
            llama_layer lz;
            lz.ffn_gate_exps = fake("ge");
            lz.ffn_up_exps   = fake("ue");
            lz.ffn_down_exps = ggml_new_tensor_1d(g_ctx, GGML_TYPE_F32, 0); // zero bytes
            llama_ls_package pz;
            std::string ze;
            if (ad->package(nullptr, 7, lz, pz, ze) ||
                ze.find("incomplete routed-expert bank") == std::string::npos) {
                fprintf(stderr, "FAIL moe zero-byte expert not rejected\n");
                failures++;
            } else {
                fprintf(stderr, "ok   moe zero-byte expert rejected: %s\n", ze.c_str());
            }
        }
    }

    ggml_free(g_ctx);

    if (failures) {
        fprintf(stderr, "test-layer-streaming-descriptors: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test-layer-streaming-descriptors: all descriptor checks passed\n");
    return 0;
}
