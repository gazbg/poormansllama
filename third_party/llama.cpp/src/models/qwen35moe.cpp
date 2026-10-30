#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"

#ifdef LLAMA_LAYER_STREAMING
#include "llama-layer-streaming.h"
#include "llama-expert-split-policy.h"
#include "llama-moe-hybrid.h"
#endif

// T-GPLR-82 S2: the host expert sub-op reads each MODEL LOADER tensor's declared
// buffer placement to prove no weight source was rewired.
#include "ggml-backend.h"

const llama_layer & llama_model_qwen35moe::graph::layer(int il) const {
#ifdef LLAMA_LAYER_STREAMING
    if (lm != nullptr && lm->layer != nullptr) {
        return *lm->layer;
    }
    if (ls != nullptr) {
        return ls->slot_layer(il);
    }
#endif
    return model.layers[il];
}

void llama_model_qwen35moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_35B_A3B; break;
        case 48: type = LLM_TYPE_122B_A10B; break;
        case 60: type = LLM_TYPE_397B_A17B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // MTP block looks like a full-attention Qwen3.5 decoder block with MoE FFN.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, mtp_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, mtp_flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, mtp_flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, mtp_flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35moe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
#ifdef LLAMA_LAYER_STREAMING
    ls = const_cast<llama_layer_streaming *>(params.layer_streaming);
    if (ls != nullptr) {
        build_streaming_graphs();
        return;
    }
    if (params.lm != nullptr) {
        build_lm_graph();
        return;
    }
#endif
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer(il).attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, layer(il).attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // post-norm hidden state feeds both the LM head and the MTP seed below
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

#ifdef LLAMA_LAYER_STREAMING
// T-GPLR-02: one cgraph per piece (embed, per logical decoder layer, output)
// for the MoE layer-major prefill. Layer graphs read the persistent
// activation pair and write the next one; KV / Gated DeltaNet state is
// written in-graph exactly like the normal path. Weight tensors come through
// layer(il): the slot bundle for package members (device slot), the resident
// model tensors for everything else (adapter merge_resident).
void llama_model_qwen35moe::graph::build_streaming_graphs() {
    GGML_ASSERT(ls != nullptr);

    const auto * mctx_hybrid = static_cast<const llama_memory_hybrid_context *>(mctx);
    const int32_t rs_z = mctx_hybrid ? mctx_hybrid->get_recr()->get_rs_z() : -1;

    ggml_context * ctx_io = ls->begin_build(n_tokens, rs_z, (int32_t) n_outputs);
    if (ctx_io == nullptr) {
        GGML_ABORT("layer streaming: %s", ls->error_msg.c_str());
    }
    ctx0 = ctx_io;

    const auto * act = ls->cur_set()->act;

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;

    // T-GPLR-12: when activation slabs are BF16, the act tensor objects are
    // BF16 type (half the slab footprint). All layer computation stays in
    // F32; ggml_cpy handles F32→BF16 on write and we cast BF16→F32 on read.
    // Same contract as the dense builder (T-Q38PW-02 in qwen35.cpp) -- the
    // MoE path previously read the BF16 slab directly and crashed in
    // build_norm (GGML_ASSERT(src0->type == GGML_TYPE_F32)).
    const bool slab_bf16 = (ls->act_slab_dtype == GGML_TYPE_BF16);

    // embed graph: tokens -> embeddings -> act[0]
    gf = ls->new_graph(-1);
    cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.input_embed", -1);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, act[0]));

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // T-GPLR-13: row-set TG sets split every layer graph in two parts.
    // Part A (graphs[1+il]): embed-consumers, attention, residual, post-norm
    // and the ROUTER. Its last node copies the FFN input (the post-norm
    // tensor) into act[0] and expands the router's selection probs, so both
    // survive replay for the executor to read. Part B (ff_graphs[il]): the
    // FFN bound to a host-supplied selected_experts INPUT; it reads its
    // input back from act[0] and writes the layer output into act[(il+1)&1]
    // like the single-part graph does. Routing arithmetic is untouched:
    // part B recomputes the identical norm/router from resident weights and
    // its top-k comes from the executor's host derivation of part A's
    // probabilities (same stable tie-break).
    const bool rowset_split = ls->row_set_requested && ls->phase == 1;

    // one graph per logical decoder layer
    for (int il = 0; il < n_layer; ++il) {
        gf = ls->new_graph(il);
        res->t_layer_inp[il] = act[il & 1];

        ggml_tensor * inpSA = act[il & 1];
        if (slab_bf16) {
            inpSA = ggml_cast(ctx0, inpSA, GGML_TYPE_F32);
        }

        cur = build_norm(inpSA, layer(il).attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * attn_post_norm = build_norm(cur, layer(il).attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        if (rowset_split) {
            // part A tail: the router (IDs are derived host-side from its
            // probabilities) and the residual published for part B. The
            // residual crosses the split through a dedicated persistent
            // buffer tensor (ff_residual) so part B stays self-contained --
            // a cross-graph node reference would pull part A's ancestry
            // into part B and silently recompute attention.
            ggml_tensor * logits = build_lora_mm(layer(il).ffn_gate_inp, attn_post_norm);
            cb(logits, "ffn_moe_logits", il);
            ggml_tensor * probs = ggml_soft_max(ctx0, logits);
            cb(probs, "ffn_moe_probs", il);
            // dead-end node (no consumers in part A): flag it as a graph
            // output so the allocator never reuses its block before the
            // executor reads the router's probabilities between the parts.
            ggml_set_output(probs);
            ggml_build_forward_expand(gf, probs);
            // the residual crosses through the set's FREE act slot
            // act[(il+1)&1] (the slot the single-part graph would write this
            // layer's output into): part A copies the pre-FFN residual there
            // and part B reads it as its input, then adds it after the FFN.
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ffn_residual, act[(il + 1) & 1]));

            // part B: FFN on the slot's staged expert rows, bound to a
            // host-supplied top-k input. The post-norm, router, and SHARED
            // EXPERT branch recompute bit-exactly from resident weights (the
            // same build_layer_ffn the single-part graph uses); only the
            // routed top-k comes from sel_in (derived from part A's
            // probabilities with the same stable tie-break as the argsort).
            gf = ls->new_ff_graph(il);
            ggml_tensor * residual_in = act[(il + 1) & 1];
            ggml_tensor * sel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_expert_used, n_tokens);
            ggml_set_input(sel_in);
            ggml_set_name(sel_in, "rowset_selected_experts");
            ls->cur_set()->ff_input.back() = sel_in;

            // T-GPLR-62 S2: the same experts as POSITIONS in the staged window.
            // Same shape, type and lifetime as sel_in; the executor fills both
            // from the same host ids in the same pass. It is created and
            // populated unconditionally so the mapping is exercised and
            // auditable, but it is passed to the FFN ONLY when the slot is
            // actually narrower than the bank -- while the slot is full width a
            // position index would be the wrong address and the graph must keep
            // consuming the router's absolute ids.
            ggml_tensor * sel_pos_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_expert_used, n_tokens);
            ggml_set_input(sel_pos_in);
            ggml_set_name(sel_pos_in, "rowset_selected_experts_pos");
            ls->cur_set()->ff_input_wpos.back() = sel_pos_in;
            const bool window_live = ls->row_set_window_width > 0;

            cur = build_norm(residual_in, layer(il).attn_post_norm, nullptr, LLM_NORM_RMS, il);

            // T-GPLR-82 S2: the routed top-k is SPLIT into a device head leg and
            // a host leg, each constant width, and the two are summed through the
            // existing add chain. Both legs take their weights from ONE
            // derivation over the FULL top_k, which is why neither leg can
            // normalise inside build_moe_ffn: each would normalise over its own
            // subset and the pair would sum to 2.0.
            //
            // The split is POSITIONAL and therefore a fixed graph shape. Which
            // rows land on the host is executor state (ff_input_hsel), so the
            // membership rule can change per token without a rebuild. S2 keeps
            // the rule positional; S3 is what makes it residency-aware.
            const int host_rows = ls->experts_host_n;
            // T-GPLR-82 S3/S4 (2026-10-10): the split shape commits host_rows
            // rows per (layer, token) to the CPU leg -- these widths are fixed
            // HERE and the host leg's ids are the UNCONDITIONAL tail of the row
            // (ff_input_hsel), so building it while the executor cannot admit a
            // row changes the arithmetic for a run whose ledger reports zero host
            // rows. MEASURED (session gplr82-ac1, exe 66b4eb9e, arm A2d, budget
            // at its default 0): host share 0.000000 and
            // sum(experts_device) == sum(experts_total) == 163,840, yet ffn_dev = 2
            // (mixed) on all 20,480 records against 1 for the control, and
            // divergence from the control at TG step 1 with all 512 logits FINITE.
            // So the split is built only when ls_split_in_play -- the conjuncts
            // the executor's own admission predicate applies -- says it can run;
            // otherwise control falls through to the UNSPLIT FFN graph below,
            // exactly as with the flag absent.
            llama_ls_policy::ls_split_enable split_e;
            split_e.host_w    = ls->experts_host_n;
            split_e.budget_us = ls->experts_host_budget_us;
            split_e.recon     = ls->row_set_recon;
            split_e.spec_on   = ls->row_set_spec_on;
            split_e.spec_mode = ls->row_set_spec_mode;
            split_e.spec_arm  = ls->row_set_spec_arm;
            split_e.spec_pool = ls->row_set_spec_pool;
            const bool split_in_play = llama_ls_policy::ls_split_in_play(split_e);

            // T-GPLR-82 FALSIFIER (2026-10-10, diagnostic, default off): build the
            // SAME constant-width partition, but with the tail leg on the backend that
            // owns the slot's staged weights instead of on the CPU. The shipped host leg
            // is CPU-placed by CONSTRUCTION (it binds the loader's host-placed banks), so
            // the split's measured step-0 shift cannot otherwise be separated from the
            // partition itself. Nothing else about the shape changes: same dev_rows head,
            // same host_rows tail, same ONE w_full derivation, same positional suffix.
            const bool tail_device = ls->experts_host_leg_device && host_rows > 0;
            const bool split_build  = split_in_play || tail_device;
            if (tail_device) {
                static bool falsifier_warned = false;
                if (!falsifier_warned) {
                    falsifier_warned = true;
                    LLAMA_LOG_WARN("T-GPLR-82 FALSIFIER: --experts-host-leg device is ON -- the split's "
                                   "tail leg (%d of %d routed rows per layer/token) is built on the DEVICE "
                                   "backend. This is a diagnostic partition, NOT a placement policy; no host "
                                   "leg is built and no row executes on the CPU.",
                                   host_rows, (int) n_expert_used);
                }
            }
            if (host_rows > 0 && !split_in_play && !tail_device) {
                // Loud, not silent. A run with the flag set and the mechanism OFF
                // must be distinguishable from a flag-absent run by the operator,
                // not only by the ledger's zero host share.
                static bool warned_once = false;
                if (!warned_once) {
                    warned_once = true;
                    LLAMA_LOG_WARN("T-GPLR-82: --experts-host %d is NOT in play "
                                   "(budget %.1f us/token, recon %d, spec_on %d, spec_mode %d, "
                                   "spec_arm %d, spec_pool %d) -- building the UNSPLIT FFN graph "
                                   "for every layer of this run, exactly as with the flag absent",
                                   ls->experts_host_n, ls->experts_host_budget_us,
                                   (int) ls->row_set_recon, (int) ls->row_set_spec_on,
                                   ls->row_set_spec_mode, ls->row_set_spec_arm,
                                   ls->row_set_spec_pool);
                }
            }
            if (split_build) {
                // Loud bounds check, not a bare write. The sizing lives in init()
                // and must precede every build; an earlier placement made this an
                // out-of-bounds write, which is how the access violation
                // (0xC0000005) during "building persistent graphs" was produced.
                if ((size_t) il >= ls->row_set_split_built.size()) {
                    GGML_ABORT("T-GPLR-82: split flag vector holds %zu entries, no slot for layer %d",
                               ls->row_set_split_built.size(), il);
                }
                // T-GPLR-82 FALSIFIER: the device-tail mode builds the split SHAPE but
                // no host leg, so the executor must read 0 here -- that is what keeps the
                // hsel fill, the plan reorder and the host per-row scan off, and makes the
                // ledger report 0 host rows and host_finite -1 honestly.
                ls->row_set_split_built[(size_t) il] = (split_in_play && !tail_device) ? 1 : 0;
                if (host_rows >= n_expert_used) {
                    GGML_ABORT("T-GPLR-82: --experts-host %d leaves no device rows in the top_k of %d for layer %d",
                               host_rows, n_expert_used, il);
                }
                const int dev_rows = n_expert_used - host_rows;

                // ONE derivation over the whole top_k, mirroring build_moe_ffn's
                // own gather/norm/scale order exactly so the values are the ones
                // the pre-S2 graph produced.
                ggml_tensor * h_logits = build_lora_mm(layer(il).ffn_gate_inp, cur);
                cb(h_logits, "ffn_moe_logits", il);
                ggml_tensor * h_probs = ggml_soft_max(ctx0, h_logits);
                cb(h_probs, "ffn_moe_probs", il);
                ggml_tensor * w_full = ggml_get_rows(ctx0,
                    ggml_reshape_3d(ctx0, h_probs, 1, n_expert, n_tokens), sel_in);
                w_full = ggml_reshape_2d(ctx0, w_full, n_expert_used, n_tokens);
                w_full = ggml_div(ctx0, w_full, ggml_clamp(ctx0, ggml_sum_rows(ctx0, w_full),
                                                           6.103515625e-5, INFINITY));
                w_full = ggml_reshape_3d(ctx0, w_full, 1, n_expert_used, n_tokens);
                if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
                    w_full = ggml_scale(ctx0, w_full, hparams.expert_weights_scale);
                }
                cb(w_full, "ffn_moe_weights_split", il);

                // kept alive and named so the executor can scan the activation both legs read
                ggml_set_name(cur, "split_ffn_in");
                ggml_set_output(cur);

                // T-GPLR-82 FALSIFIER: WHICH SIDE OF THE ROW EACH LEG TAKES. The shipped
                // split is head-at-0 plus a host tail in the SUFFIX. A suffix view cannot be
                // expressed here: ggml bounds-checks a view as data_size + view_offs <=
                // nbytes(src), and a [host_rows, n_tokens] I32 view with the parent row
                // stride needs (T-1)*nb1 + host_rows*4 bytes PLUS an offset of dev_rows*nb1,
                // which overruns (measured: GGML_ASSERT at ggml.c:1787 while building the
                // persistent layer graphs). The device-tail mode therefore keeps the same
                // 6/2 widths and the same ONE w_full derivation, but puts the TAIL at offset
                // 0 and the HEAD at host_rows*len: the wider head view then ends exactly at
                // the parent last byte, and the two legs cover disjoint complete positions.
                // Which side holds which rows is not a fixed property of the shipped
                // mechanism either -- its membership is plan-chosen per token. The shipped
                // path is untouched: both offsets stay 0, head first.
                const size_t id_off  = tail_device ? (size_t) host_rows * sel_in->nb[0]     : 0;
                const size_t pos_off = tail_device ? (size_t) host_rows * sel_pos_in->nb[0] : 0;
                const size_t w_off   = tail_device ? (size_t) host_rows * w_full->nb[1]     : 0;
                // device head leg: dev_rows positions -- the FIRST ones in the shipped split,
                // the LAST ones in the device-tail diagnostic (which holds the tail at 0)
                ggml_tensor * sel_head = ggml_view_2d(ctx0, sel_in, dev_rows, n_tokens, sel_in->nb[1], id_off);
                ggml_tensor * w_dev = ggml_view_3d(ctx0, w_full, 1, dev_rows, n_tokens,
                                                    w_full->nb[1], w_full->nb[2], w_off);
                ggml_tensor * pos_head = window_live
                    ? ggml_view_2d(ctx0, sel_pos_in, dev_rows, n_tokens, sel_pos_in->nb[1], pos_off)
                    : nullptr;

                ggml_tensor * dev_out = build_layer_ffn(cur, il, sel_head, pos_head, w_dev, dev_rows);
                ggml_set_name(dev_out, "split_dev_out");
                ggml_set_output(dev_out);
                ggml_tensor * host_out;
                if (tail_device) {
                    // T-GPLR-82 FALSIFIER: the SAME positional suffix, weights, and
                    // reduction as the host leg below, computed on the backend that owns
                    // the slot's staged weights. Suffix VIEWS of the device leg's own
                    // inputs, so no new INPUT and no executor-side fill exists: every
                    // demanded row is staged and awaited exactly as in the unsplit path.
                    // Offset 0 with the PARENT strides: the tail holds the first host_rows
                    // positions in this mode, so there is no suffix arithmetic to bound.
                    ggml_tensor * sel_tail = ggml_view_2d(ctx0, sel_in, host_rows, n_tokens, sel_in->nb[1], 0);
                    ggml_tensor * pos_tail = window_live
                        ? ggml_view_2d(ctx0, sel_pos_in, host_rows, n_tokens, sel_pos_in->nb[1], 0)
                        : nullptr;
                    ggml_tensor * w_tail = ggml_view_3d(ctx0, w_full, 1, host_rows, n_tokens,
                                                        w_full->nb[1], w_full->nb[2], 0);
                    host_out = build_moe_tail_leg_device(cur, il, sel_tail, pos_tail, w_tail, host_rows);
                } else {
                    // host leg ids: a separate INPUT, not a view of sel_in. The host
                    // leg runs on the CPU backend and must not inherit a view of a
                    // device tensor, and S3 needs to choose host membership freely.
                    ggml_tensor * hsel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, host_rows, n_tokens);
                    ggml_set_input(hsel_in);
                    ggml_set_name(hsel_in, "rowset_selected_experts_host");
                    ls->cur_set()->ff_input_hsel.back() = hsel_in;
                    host_out = build_moe_host_leg(cur, il, hsel_in,
                        ggml_view_3d(ctx0, w_full, 1, host_rows, n_tokens,
                                     w_full->nb[1], w_full->nb[2], (size_t) dev_rows * w_full->nb[1]),
                        host_rows);
                    // diagnostic: recompute the same suffix rows on the device next to the CPU leg; observed only, never summed
                    static const bool host_shadow = getenv("LLAMA_LS_HOST_SHADOW") != nullptr;
                    if (host_shadow && n_tokens == 1) {
                        ggml_tensor * sel_sfx = ggml_view_2d(ctx0, sel_in, host_rows, n_tokens, sel_in->nb[1], (size_t) dev_rows * sel_in->nb[0]);
                        ggml_tensor * w_sfx = ggml_view_3d(ctx0, w_full, 1, host_rows, n_tokens, w_full->nb[1], w_full->nb[2], (size_t) dev_rows * w_full->nb[1]);
                        ggml_tensor * shadow = build_moe_tail_leg_device(cur, il, sel_sfx, nullptr, w_sfx, host_rows);
                        ggml_set_name(shadow, "split_shadow_out");
                        ggml_set_output(shadow);
                        ggml_build_forward_expand(gf, shadow);
                        ggml_set_name(host_out, "split_host_out");
                        ggml_set_output(host_out);
                        ggml_tensor * ref_full = build_layer_ffn(cur, il, sel_in, nullptr);
                        ggml_set_name(ref_full, "split_ref_out");
                        ggml_set_output(ref_full);
                        ggml_build_forward_expand(gf, ref_full);
                    }
                }
                cur = ggml_add(ctx0, dev_out, host_out);
                cb(cur, "ffn_out", il);
            } else {
                static const bool flow_diag = getenv("LLAMA_LS_HOST_SHADOW") != nullptr;
                if (flow_diag) {
                    ggml_set_name(cur, "split_ffn_in");
                    ggml_set_output(cur);
                }
                cur = build_layer_ffn(cur, il, sel_in, window_live ? sel_pos_in : nullptr);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, residual_in);
            cb(cur, "post_moe", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, act[(il + 1) & 1]));
            continue;
        }

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // T-GPLR-02: keep the router's selection probabilities live so the
        // executor can read them for observational route telemetry. The
        // captured top-k view aliases the ARGSORT node; its src[0] is the
        // selection_probs tensor. Marking it a graph output extends its buffer
        // lifetime to the end of the graph (no allocator reuse). Expanding an
        // already-computed node changes no arithmetic.
        {
            const auto & routes = ls->cur_set()->routes;
            const auto rit = routes.find(il);
            if (rit != routes.end()) {
                ggml_tensor * base = rit->second;
                while (base->op == GGML_OP_VIEW && base->view_src != nullptr) {
                    base = base->view_src; // topk view -> argsort node
                }
                if (base->op == GGML_OP_ARGSORT && base->src[0] != nullptr) {
                    // This is a dead-end diagnostic leaf in the ordinary graph.
                    // Keep its allocation alive until the post-compute AER
                    // readback; row-set part A marks the same tensor as output.
                    ggml_set_output(base->src[0]);
                    ggml_build_forward_expand(gf, base->src[0]); // selection probs
                }
            }
        }

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // write the layer output into the next activation slot
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, act[(il + 1) & 1]));
    }

    // output graph: final norm + LM head
    gf = ls->new_graph(-2);

    cur = act[n_layer & 1];
    if (slab_bf16) {
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (!ls->finish_build()) {
        GGML_ABORT("layer streaming: %s", ls->error_msg.c_str());
    }
}
#endif // LLAMA_LAYER_STREAMING

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35moe::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(layer(il).wqkv, input, layer(il).wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(layer(il).wqkv_gate, input, layer(il).wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35moe::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer(il), cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, layer(il).attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer(il).attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(layer(il).wo, cur, layer(il).wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(layer(il).ssm_beta, cur, layer(il).ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(layer(il).ssm_alpha, cur, layer(il).ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, layer(il).ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, layer(il).ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = layer(il).ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, layer(il).ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(layer(il).ssm_out, final_output, layer(il).ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn(ggml_tensor * cur, const int il, ggml_tensor * selected_experts_in, ggml_tensor * selected_experts_weight_in, ggml_tensor * moe_weights_in, const int routed_rows) {
    // Check if this is an MoE layer
    GGML_ASSERT(layer(il).ffn_gate_inp != nullptr);

    // T-GPLR-82 S2: 0 means the pre-S2 shape -- the full routed top_k, weights
    // derived inside build_moe_ffn. A non-zero value is this leg's share of a
    // split top_k and is only ever set together with moe_weights_in.
    const int64_t n_rows = routed_rows > 0 ? routed_rows : n_expert_used;

    ggml_tensor * moe_out = nullptr;
    ggml_tensor * moe_pub = nullptr;
#ifdef LLAMA_LAYER_STREAMING
    // opt-in hybrid decode (env LLAMA_MOE_HYBRID): GPU slot cache plus a host pool for the misses; nullptr means the stock path
    if (ls == nullptr && lm == nullptr && selected_experts_in == nullptr && moe_weights_in == nullptr && routed_rows == 0) {
        moe_out = llama_moe_hybrid_build_ffn(*this, model, il, cur, &moe_pub);
    }
#endif
    if (moe_out == nullptr) {
    moe_out =
        build_moe_ffn(cur,
            layer(il).ffn_gate_inp,
            layer(il).ffn_up_exps,
            layer(il).ffn_gate_exps,
            layer(il).ffn_down_exps,
            nullptr,
            n_expert, n_rows,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer(il).ffn_gate_up_exps,
            layer(il).ffn_up_exps_s,
            layer(il).ffn_gate_exps_s,
            layer(il).ffn_down_exps_s,
            selected_experts_in,
            nullptr, 1e-5f,   // post_exp_norm, post_exp_norm_eps (upstream, before the fork's row-set parameters)
            selected_experts_weight_in,
            moe_weights_in);
    }
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (layer(il).ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer(il).ffn_up_shexp, NULL, layer(il).ffn_up_shexp_s,
                layer(il).ffn_gate_shexp, NULL, layer(il).ffn_gate_shexp_s,
                layer(il).ffn_down_shexp, NULL, layer(il).ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate = build_lora_mm(layer(il).ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        // Apply the gate to the shared expert output
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

#ifdef LLAMA_LAYER_STREAMING
    if (moe_pub != nullptr) {
        // add the host engine's sum over the experts that missed the VRAM cache
        cur = llama_moe_hybrid_build_wait(*this, cur, moe_pub);
    }
#endif

    return cur;
}

// G32: ONE piece of a layer-major prefill pass, built per launch (llama-lm-prefill.cpp), see llm_graph_lm. The pieces hand the hidden state to each other through the pass's slab; every
// INPUT tensor of the piece is pulled into the graph at the end, because a piece of one layer type never reads the inputs of the other type (an unallocated input would crash set_inputs).
#ifdef LLAMA_LAYER_STREAMING
void llama_model_qwen35moe::graph::build_lm_graph() {
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

        int sections[4];
        std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

        auto * inp = build_inp_mem_hybrid();

        ggml_tensor * inp_pos = build_inp_pos();

        ggml_tensor * inpSA = act(lm->act_in, n_tokens);
        if (slab_bf16) {
            inpSA = ggml_cast(ctx0, inpSA, GGML_TYPE_F32);
        }

        cur = build_norm(inpSA, layer(il).attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * attn_post_norm = build_norm(cur, layer(il).attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, act(lm->act_out, n_tokens)));
    } else {
        cur = act(lm->act_in, 1);
        if (slab_bf16) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        }

        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur, model.output_s);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    for (ggml_tensor * t = ggml_get_first_tensor(ctx0); t != nullptr; t = ggml_get_next_tensor(ctx0, t)) {
        if (t->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_build_forward_expand(gf, t);
        }
    }
}
#endif // LLAMA_LAYER_STREAMING

// T-GPLR-82 S2: the HOST expert sub-op.
//
// WHY IT BINDS model.layers[il] AND NOT layer(il). In this graph
// llama_model_qwen35moe::graph::layer() returns ls->slot_layer(il) -- the SLOT's
// layer, whose expert tensors are the compact CUDA copies load_row_set stages.
// The host leg must read the very bytes the DMA is reading, so it binds the
// model LOADER's tensors: the runner forced those to a CPU buffer type at load
// time (the exps buft override) and the slot never rewires them. That is the
// placement the SDD permits -- the weight is host-placed and stays host-placed,
// so nothing was rewired. The guard below checks that from the tensors'
// declared placement instead of asserting it in a comment.
//
// ALL THREE routed-expert members are bound on purpose. bytes_staged in
// load_row_set counts the gate/up/down slabs of every package member, so a host
// leg that bound only some of them would leave the byte accounting out of step
// with the copy loop that is still running for them.
#ifdef LLAMA_LAYER_STREAMING
ggml_tensor * llama_model_qwen35moe::graph::build_moe_host_leg(
        ggml_tensor * cur, const int il,
        ggml_tensor * host_ids, ggml_tensor * host_weights, const int n_host) const {
    GGML_ASSERT(ls != nullptr && "build_moe_host_leg is only reachable on the row-set path");
    const llama_layer & host_layer = model.layers[il];

    const ggml_tensor * members[3] = {
        host_layer.ffn_gate_exps, host_layer.ffn_up_exps, host_layer.ffn_down_exps,
    };
    if (host_layer.ffn_gate_up_exps != nullptr) {
        // fused gate_up bank: gate and up are ONE tensor, so the bound count is 2
        members[0] = host_layer.ffn_gate_up_exps;
        members[1] = nullptr;
    }

    std::vector<llama_ls_policy::ls_rewire_observation> obs;
    obs.reserve(3);
    for (int i = 0; i < 3; ++i) {
        if (members[i] == nullptr) {
            continue;
        }
        const bool is_host =
            ggml_backend_buft_is_host(ggml_backend_buffer_get_type(members[i]->buffer));
        // The staging fact is read from the buffer's USAGE, which is what
        // separates a weight at its DECLARED placement from one the
        // scheduler's compute_splits path has put in a compute buffer. Both
        // usages are host-placed, so is_host alone cannot tell them apart --
        // this is the field that makes the placement assertion sharp rather
        // than satisfied by accident.
        const bool in_staging =
            ggml_backend_buffer_get_usage(members[i]->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        // min_batch_override_live stays false: the T-GPLR-16 MIN_BATCH=1 route
        // is not in this tree (reverted 2026-09-22), so there is no override
        // to be live. It is passed explicitly rather than defaulted so that
        // re-introducing that route forces a decision HERE, at the only place
        // in the tree where a host leg is built.
        obs.push_back({true, is_host, false, in_staging});
    }
    if (obs.empty()) {
        GGML_ABORT("T-GPLR-82: layer %d has no routed-expert tensors to bind to the host leg", il);
    }

    const llama_ls_policy::ls_rewire_verdict verdict = llama_ls_policy::ls_check_no_rewire(obs);
    if (!verdict.ok) {
        // Loud and terminal on purpose. SDD section 4 forbids repairing this,
        // and the graph builder has no return path the executor inspects, so
        // continuing here would run a graph whose weights are not the ones the
        // manifest names -- the one outcome this whole class must never have.
        GGML_ABORT("T-GPLR-82: %s (layer %d, %u of %u host-leg weights)",
                   verdict.refuse_reason, il, verdict.rewired, verdict.host_leg_weights);
    }
    ls->host_leg_builds         += 1;
    ls->host_leg_weights_bound  += verdict.host_leg_weights;

    ggml_tensor * exp_rows = nullptr;
    ggml_tensor * moe_out = build_moe_ffn(cur,
        host_layer.ffn_gate_inp,
        host_layer.ffn_up_exps,
        host_layer.ffn_gate_exps,
        host_layer.ffn_down_exps,
        nullptr,
        n_expert, n_host,
        LLM_FFN_SILU, true,
        hparams.expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
        nullptr, host_layer.ffn_gate_up_exps,
        nullptr, nullptr, nullptr,
        host_ids,
        nullptr, 1e-5f,   // post_exp_norm, post_exp_norm_eps (upstream, before the fork's row-set parameters)
        // ABSOLUTE ids: this leg's weights are the FULL bank, so there is no
        // compact window to address and no position table to pass. Passing the
        // device leg's window positions here would be the addressing bug the
        // T-GPLR-62 S2 comment warns about, so this stays null deliberately.
        nullptr,
        host_weights,
        &exp_rows);
    if (exp_rows == nullptr) {
        GGML_ABORT("T-GPLR-61: host leg for layer %d did not capture its per-row outputs", il);
    }
    auto * gs = ls->cur_set();
    if (gs == nullptr || gs->ff_host_experts.empty()) {
        GGML_ABORT("T-GPLR-61: ff_host_experts not sized before the host leg for layer %d", il);
    }
    // The executor scans this tensor after part B for non-finite values
    // (SDD moe-row-split failure contract): pre-aggregation rows, so a bad
    // value can be attributed to the exact expert that produced it.
    gs->ff_host_experts.back() = exp_rows;
    return moe_out;
}

// T-GPLR-82 FALSIFIER (2026-10-10, diagnostic). The SAME constant-width tail leg
// as build_moe_host_leg, computed on the backend that owns the SLOT layer's staged
// weights instead of on the CPU. EVERYTHING but the weight source and the resulting
// backend is identical on purpose: same positional suffix, same n_tail width, same
// pre-derived weights from the ONE w_full, the same build_moe_ffn reduction into the
// same add chain. That is what makes a difference measured here attributable: a shift
// that survives with no CPU row in the partition is not the CPU's.
//
// Reachable only under --experts-host-leg device (the graph builder branches on it).
// The shared expert is deliberately not built here: the head leg already carries it,
// and adding it twice would move the layer output far more than any numeric effect
// this diagnostic is trying to see. The rewire guard is not consulted because this
// mode builds no host leg at all -- there is nothing host-placed to rewire.
ggml_tensor * llama_model_qwen35moe::graph::build_moe_tail_leg_device(
        ggml_tensor * cur, const int il,
        ggml_tensor * tail_ids, ggml_tensor * tail_pos, ggml_tensor * tail_weights,
        const int n_tail) const {
    GGML_ASSERT(ls != nullptr && "build_moe_tail_leg_device is only reachable on the row-set path");
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer(il).ffn_gate_inp,
            layer(il).ffn_up_exps,
            layer(il).ffn_gate_exps,
            layer(il).ffn_down_exps,
            nullptr,
            n_expert, n_tail,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer(il).ffn_gate_up_exps,
            layer(il).ffn_up_exps_s,
            layer(il).ffn_gate_exps_s,
            layer(il).ffn_down_exps_s,
            tail_ids,
            nullptr, 1e-5f,   // post_exp_norm, post_exp_norm_eps (upstream, before the fork's row-set parameters)
            tail_pos,
            tail_weights);
    return moe_out;
}
#endif // LLAMA_LAYER_STREAMING

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
llama_model_qwen35moe::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35MOE MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35MOE MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj    && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm      && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm      && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp     && "MTP block missing ffn_gate_inp");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    // MoE FFN — routed experts plus gated shared expert (mirrors qwen35moe).
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "mtp_shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "mtp_ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn= cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    // G30: LLAMA_MTP_DRAFT_VOCAB_N = N > 0 draws the draft only from the token ids [0, N). The first N rows of the head are a contiguous prefix of the weight, so a 2-D view of them costs nothing and the
    // lm head reads N / n_vocab of the bytes; the draft pick only has to be a good guess (the verify keeps the output exact). The logits are padded back to the full row so the logits layout is unchanged.
    static const int64_t g30_draft_vocab_n = []() { const char * e = getenv("LLAMA_MTP_DRAFT_VOCAB_N"); return e ? (int64_t) atoll(e) : (int64_t) 0; }();
    const int64_t g30_head_rows = head_w->ne[1];
    if (g30_draft_vocab_n > 0 && g30_draft_vocab_n < g30_head_rows) {
        ggml_tensor * head_sub = ggml_view_2d(ctx0, head_w, head_w->ne[0], g30_draft_vocab_n, head_w->nb[1], 0);
        cur = build_lora_mm(head_sub, cur, head_s);
        cur = ggml_pad(ctx0, cur, (int) (g30_head_rows - g30_draft_vocab_n), 0, 0, 0);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
