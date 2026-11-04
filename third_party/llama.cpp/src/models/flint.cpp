#include "models.h"
#include "llama-moe-hybrid.h"

void llama_model_flint::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    ml.get_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_flint::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp        = hparams.n_ff_exp();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        hparams.n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_flint::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

const llama_layer & llama_model_flint::graph::layer(int il) const {
#ifdef LLAMA_LAYER_STREAMING
    if (lm != nullptr && lm->layer != nullptr) {
        return *lm->layer;
    }
#endif
    return model.layers[il];
}

// The attention half of layer il on the normed input `cur`: q/k/v, their norms, RoPE on the layers that have it, the (sliding-window or full) attention with the layer's sinks, the output projection.
// Shared by the ordinary graph and the layer-major pass's pieces, so the arithmetic is one piece of code.
ggml_tensor * llama_model_flint::graph::build_layer_attn(llm_graph_input_attn_kv_iswa * inp_attn, ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    // compute Q and K and RoPE them
    auto [Qcur, Kcur, Vcur] = build_qkv(layer(il), cur,
            n_embd_head, n_head, n_head_kv, il);

    Qcur = build_norm(Qcur, layer(il).attn_q_norm, nullptr, LLM_NORM_RMS, il);
    Kcur = build_norm(Kcur, layer(il).attn_k_norm, nullptr, LLM_NORM_RMS, il);
    if (hparams.has_rope(il)) {
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return build_attn(inp_attn,
            layer(il).wo, layer(il).wo_b, layer(il).wo_s,
            Qcur, Kcur, Vcur, nullptr, layer(il).attn_sinks, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
}

// The FFN half of layer il on the normed attention output `cur`: a dense FFN on the leading layers, else the routed experts plus the shared experts. The caller adds the residual.
ggml_tensor * llama_model_flint::graph::build_layer_ffn(ggml_tensor * cur, int il) {
    if ((uint32_t) il < hparams.n_layer_dense_lead) {
        cur = build_ffn(cur,
                layer(il).ffn_up, NULL, layer(il).ffn_up_s,
                layer(il).ffn_gate, NULL, layer(il).ffn_gate_s,
                layer(il).ffn_down, NULL, layer(il).ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
        return cur;
    }

    ggml_tensor * moe_pub = nullptr;
    ggml_tensor * moe_out = nullptr;
#ifdef LLAMA_LAYER_STREAMING
    // opt-in hybrid decode (env LLAMA_MOE_HYBRID): the routed experts from the VRAM cache, the misses on the host (qwen35moe.cpp has the same call)
    if (layer_streaming == nullptr && lm == nullptr) {
        moe_out = llama_moe_hybrid_build_ffn(*this, model, il, cur, &moe_pub);
    }
#endif
    if (moe_out == nullptr) {
        moe_out = build_moe_ffn(cur,
            layer(il).ffn_gate_inp,
            layer(il).ffn_up_exps,
            layer(il).ffn_gate_exps,
            layer(il).ffn_down_exps,
            layer(il).ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il, nullptr, nullptr,
            layer(il).ffn_up_exps_s,
            layer(il).ffn_gate_exps_s,
            layer(il).ffn_down_exps_s);
    }
    cb(moe_out, "ffn_moe_out", il);

    {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer(il).ffn_up_shexp, NULL, layer(il).ffn_up_shexp_s,
                layer(il).ffn_gate_shexp, NULL, layer(il).ffn_gate_shexp_s,
                layer(il).ffn_down_shexp, NULL, layer(il).ffn_down_shexp_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    }
#ifdef LLAMA_LAYER_STREAMING
    if (moe_pub != nullptr) {
        // the host engine's sum over the experts that missed the VRAM cache
        cur = llama_moe_hybrid_build_wait(*this, cur, moe_pub);
    }
#endif
    return cur;
}

llama_model_flint::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
#ifdef LLAMA_LAYER_STREAMING
    if (params.lm != nullptr) {
        build_lm_graph();
        return;
    }
#endif
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, layer(il).attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        cur = build_layer_attn(inp_attn, cur, inp_pos, il);
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp, layer(il).ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// G32 for flint: ONE piece of a layer-major prefill pass, built per launch (llama-lm-prefill.cpp), see llm_graph_lm. The pieces hand the hidden state to each other through the pass's slab.
// A piece builds only the inputs its layer reads: RoPE positions for a layer that has RoPE (the window layers), and the attention input of build_attn_inp_kv_iswa_lm() (the base cache's indices for
// a full-attention layer, the window cache's indices and the pass's persistent window mask for a window layer). An input tensor that nothing reads is never allocated, and set_inputs would then
// write to a null buffer.
#ifdef LLAMA_LAYER_STREAMING
void llama_model_flint::graph::build_lm_graph() {
    GGML_ASSERT(lm != nullptr);

    const int64_t n_embd_io = hparams.n_embd_inp();
    const bool    slab_bf16 = (lm->act_type == GGML_TYPE_BF16);

    auto act = [&](void * addr, int64_t rows) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, lm->act_type, n_embd_io, rows);
        ggml_set_name(t, "lm_act");
        ggml_backend_tensor_alloc(lm->act_buf, t, addr);
        return t;
    };

    ggml_tensor * cur;

    if (lm->kind == 0) {
        cur = build_inp_embd(model.tok_embd);
        cb(cur, "model.input_embed", -1);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, act(lm->act_out, n_tokens)));
    } else if (lm->kind == 1) {
        const int il = lm->il;

        GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());
        GGML_ASSERT(hparams.n_embd_head_v() == n_rot);

        auto * inp_attn = build_attn_inp_kv_iswa_lm();

        ggml_tensor * inp_pos = hparams.has_rope(il) ? (lm->inp_pos != nullptr ? build_inp_pos_lm() : build_inp_pos()) : nullptr;

        ggml_tensor * inpSA = act(lm->act_in, n_tokens);
        if (slab_bf16) {
            inpSA = ggml_cast(ctx0, inpSA, GGML_TYPE_F32);
        }

        cur = build_norm(inpSA, layer(il).attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_layer_attn(inp_attn, cur, inp_pos, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer(il).ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, act(lm->act_out, n_tokens)));
    } else {
        cur = act(lm->act_in, 1);
        if (slab_bf16) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        }

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur, model.output_s);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
}
#endif // LLAMA_LAYER_STREAMING
