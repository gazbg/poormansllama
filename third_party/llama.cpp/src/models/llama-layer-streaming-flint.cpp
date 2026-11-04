// Flint (Flint Preview MoE) adapter for the opt-in CUDA layer-major prefill (private research, gated by LLAMA_LAYER_STREAMING).
//
// The rolling package of an MoE layer is the complete routed-expert bank (gate, up, down; or the fused gate_up and down), as for the Qwen3.6 MoE adapter. Router, shared experts, attention (with its
// sinks), norms, embedding, output head, and the KV caches stay outside the window (resident). The first n_layer_dense_lead layers (two in flint-UD-Q5_K_XL) have a plain dense FFN and no experts: their
// "package" is that FFN. They are never paged (the pass treats them as resident, "native"); the descriptor exists because the common code enumerates and validates a package for every trunk layer.
// A MoE layer missing any expected expert tensor is a hard error before any copy submission, never a partial package.

#ifdef LLAMA_LAYER_STREAMING

#include "../llama-layer-streaming.h"

#include "llama-model.h"

namespace {

bool flint_package(const llama_model * /*model*/, int il, const llama_layer & src,
                   llama_ls_package & out, std::string & err) {
    out.members.clear();
    auto add = [&](ggml_tensor * llama_layer::* m) {
        llama_ls_member e;
        e.src      = src.*m;
        e.dst_base = m;
        out.members.push_back(e);
    };

    const bool moe = src.ffn_gate_exps != nullptr || src.ffn_gate_up_exps != nullptr || src.ffn_up_exps != nullptr || src.ffn_down_exps != nullptr;
    size_t expected = 0;
    if (moe) {
        const bool fused = src.ffn_gate_up_exps != nullptr;
        if (fused) {
            add(&llama_layer::ffn_gate_up_exps);
            add(&llama_layer::ffn_down_exps);
            expected = 2;
        } else {
            add(&llama_layer::ffn_gate_exps);
            add(&llama_layer::ffn_up_exps);
            add(&llama_layer::ffn_down_exps);
            expected = 3;
        }
        out.cls = "moe-expert-bank";
    } else {
        add(&llama_layer::ffn_gate);
        add(&llama_layer::ffn_up);
        add(&llama_layer::ffn_down);
        expected = 3;
        out.cls  = "dense-ffn";
    }

    size_t present = 0;
    for (const auto & m : out.members) {
        if (m.src != nullptr && ggml_nbytes(m.src) > 0) {
            present++;
        }
    }
    if (present != expected) {
        err = std::string("flint: incomplete ") + (moe ? "routed-expert bank" : "dense FFN") + " in layer " + std::to_string(il) +
              " (" + std::to_string(present) + "/" + std::to_string(expected) + " tensors present)";
        out.members.clear();
        return false;
    }
    out.policy          = "moe-expert-bank";
    out.excludes_static = true;
    out.excludes_state  = true;
    return true;
}

// merge_resident: a slot bundle is the resident layer with ONLY the package members replaced by device slot tensors.
// causal_implicit: the full-attention layers may use the mask-free causal flash-attention op; the window layers never do (decided per layer in the graph).
const llama_ls_adapter adapter_flint = {
    "flint",
    "moe-expert-bank",
    /*merge_resident=*/true,
    /*causal_implicit=*/true,
    flint_package,
    nullptr,
};

} // anonymous namespace

const llama_ls_adapter * llama_ls_flint_adapter() {
    return &adapter_flint;
}

#endif // LLAMA_LAYER_STREAMING
