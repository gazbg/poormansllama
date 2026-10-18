// T-GPLR-01: Qwen3.5 dense adapter for the experimental CUDA layer-residency
// path (private research, gated by LLAMA_LAYER_STREAMING). This is the only
// place that knows which llama_layer members form a pageable Qwen3.5 decoder
// package; the common slot/lifecycle code consumes validated descriptors only.

#ifdef LLAMA_LAYER_STREAMING

#include "../llama-layer-streaming.h"

#include "llama-model.h"

namespace {

using layer_member = ggml_tensor * llama_layer::*;

// member order is the copy order; it is part of the reference behavior and
// must not change without re-validating the Qwen3.8 parity evidence.
const layer_member REC_PACKAGE[] = {
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

const layer_member ATT_PACKAGE[] = {
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

// T-Q38HG-06: the NextN (MTP) layer is a full-attention block plus its
// projection and shared-head tensors. The nextn members live in the nested
// `nextn` struct, so each member carries a base or nextn destination.
struct layer_mtp_member {
    ggml_tensor * llama_layer::*       base  = nullptr;
    ggml_tensor * llama_layer_nextn::* nextn = nullptr;
};

const layer_mtp_member MTP_MEMBERS[] = {
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

bool qwen35_package(const llama_model * /*model*/, int il, const llama_layer & src,
                    llama_ls_package & out, std::string & err) {
    const bool rec = src.ssm_a != nullptr; // Gated DeltaNet layer marker
    const layer_member * pkg = rec ? REC_PACKAGE : ATT_PACKAGE;
    const size_t n = rec ? sizeof(REC_PACKAGE) / sizeof(REC_PACKAGE[0])
                         : sizeof(ATT_PACKAGE) / sizeof(ATT_PACKAGE[0]);
    out.members.clear();
    out.members.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        llama_ls_member m;
        m.src = src.*(pkg[i]);
        m.dst_base = pkg[i];
        out.members.push_back(m);
    }
    out.cls     = rec ? "recurrent" : "attention";
    out.policy  = "dense-layer";
    out.excludes_static = true;
    out.excludes_state  = true;
    if (out.members.empty()) {
        err = "qwen35: empty package for layer " + std::to_string(il);
        return false;
    }
    return true;
}

bool qwen35_mtp_package(const llama_model * /*model*/, const llama_layer & src,
                        llama_ls_package & out, std::string & err) {
    out.members.clear();
    for (const auto & mm : MTP_MEMBERS) {
        llama_ls_member m;
        m.src = mm.base ? src.*(mm.base) : src.nextn.*(mm.nextn);
        m.dst_base   = mm.base;
        m.dst_nextn  = mm.nextn;
        out.members.push_back(m);
    }
    out.cls     = "mtp";
    out.policy  = "dense-layer";
    out.excludes_static = true;
    out.excludes_state  = true;
    if (out.members.empty()) {
        err = "qwen35: empty MTP package";
        return false;
    }
    return true;
}

// T-GPLR-02: Qwen3.5/3.6 MoE adapter. The rolling package is the complete
// routed-expert bank for one MoE layer (gate/up/down, or fused gate_up +
// down). Router, shared experts, attention, recurrent, and norms stay outside
// the window (resident). A layer missing any expected expert tensor is a hard
// error before any copy submission, never a partial package.
bool qwen35moe_package(const llama_model * /*model*/, int il, const llama_layer & src,
                       llama_ls_package & out, std::string & err) {
    out.members.clear();
    auto add = [&](ggml_tensor * llama_layer::* m) {
        llama_ls_member e;
        e.src = src.*m;
        e.dst_base = m;
        out.members.push_back(e);
    };
    const bool fused = src.ffn_gate_up_exps != nullptr;
    if (fused) {
        add(&llama_layer::ffn_gate_up_exps);
        add(&llama_layer::ffn_down_exps);
    } else {
        add(&llama_layer::ffn_gate_exps);
        add(&llama_layer::ffn_up_exps);
        add(&llama_layer::ffn_down_exps);
    }
    // completeness: every routed-expert tensor must be present and non-empty
    size_t present = 0;
    for (const auto & m : out.members) {
        if (m.src != nullptr && ggml_nbytes(m.src) > 0) {
            present++;
        }
    }
    const size_t expected = fused ? 2u : 3u;
    if (present != expected) {
        err = "qwen35moe: incomplete routed-expert bank in layer " + std::to_string(il) +
              " (" + std::to_string(present) + "/" + std::to_string(expected) + " tensors present)";
        out.members.clear();
        return false;
    }
    out.cls     = "moe-expert-bank";
    out.policy  = "moe-expert-bank";
    out.excludes_static = true;
    out.excludes_state  = true;
    return true;
}

// The MoE adapter has no MTP rolling package (MTP execution is out of scope
// for the first prefill acceptance); mtp_package stays null and begin_handoff
// fails loudly if asked for one.
static const llama_ls_adapter adapter_qwen35moe = {
    "qwen35moe",
    "moe-expert-bank",
    /*merge_resident=*/true,
    /*causal_implicit=*/true,
    qwen35moe_package,
    nullptr,
};

} // anonymous namespace

static const llama_ls_adapter adapter_qwen35 = {
    "qwen35",
    "dense-layer",
    /*merge_resident=*/false,
    /*causal_implicit=*/true,
    qwen35_package,
    qwen35_mtp_package,
};

const llama_ls_adapter * llama_ls_qwen35_adapter() {
    return &adapter_qwen35;
}

const llama_ls_adapter * llama_ls_qwen35moe_adapter() {
    return &adapter_qwen35moe;
}

#endif // LLAMA_LAYER_STREAMING
