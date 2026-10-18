// T-GPLR-00: host-only unit test for the GGUF layer-residency inventory.
// No CUDA, no model load: writes tiny synthetic GGUF files (metadata + F32
// tensor data) and checks the deterministic JSON byte classes, KV/state
// arithmetic from actual metadata, and malformed-metadata errors.
#include "../src/llama-layer-streaming.h"

#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

static int failures = 0;

// ---------------------------------------------------------------------------
// tiny synthetic GGUF writer

struct fx_tensor {
    std::string  name;
    ggml_type    type;
    int64_t      ne[4];
};

static std::string fx_path(const char * name) {
    char path[512];
#if defined(_WIN32)
    snprintf(path, sizeof(path), "%s\\%s", getenv("TEMP") ? getenv("TEMP") : ".", name);
#else
    snprintf(path, sizeof(path), "/tmp/%s", name);
#endif
    return path;
}

static std::string write_fixture_gguf(const char * fname,
                                      const std::vector<std::pair<std::string, std::string>> & kvs,
                                      const std::vector<fx_tensor> & tensors,
                                      const std::vector<std::pair<std::string, std::vector<uint8_t>>> & bool_arrays = {},
                                      const std::vector<std::string> & as_string = {}) {
    const std::string path = fx_path(fname);
    gguf_context * ctx = gguf_init_empty();

    // architecture keys are U32-valued; general.* keys are strings; as_string
    // forces a key to be written as a string (type-mismatch negative fixtures)
    for (const auto & kv : kvs) {
        const bool str = kv.first.compare(0, 7, "general") == 0 ||
                         std::find(as_string.begin(), as_string.end(), kv.first) != as_string.end();
        if (str) {
            gguf_set_val_str(ctx, kv.first.c_str(), kv.second.c_str());
        } else {
            gguf_set_val_u32(ctx, kv.first.c_str(), (uint32_t) strtoul(kv.second.c_str(), nullptr, 10));
        }
    }
    for (const auto & ba : bool_arrays) {
        gguf_set_arr_data(ctx, ba.first.c_str(), GGUF_TYPE_BOOL, ba.second.data(), (int) ba.second.size());
    }

    // one shared zero buffer: every F32 tensor points into it
    std::vector<int8_t> zeros(64 * 64 * 4, 0);

    ggml_init_params gp = { /*mem_size=*/64 * 1024, /*mem_buffer=*/nullptr, /*no_alloc=*/true };
    ggml_context * gctx = ggml_init(gp);

    for (const auto & t : tensors) {
        ggml_tensor * cur = ggml_new_tensor(gctx, t.type, 4, t.ne);
        ggml_set_name(cur, t.name.c_str());
        gguf_add_tensor(ctx, cur);
        gguf_set_tensor_data(ctx, t.name.c_str(), zeros.data());
    }

    const bool ok = gguf_write_to_file(ctx, path.c_str(), /*only_meta=*/false);
    ggml_free(gctx);
    gguf_free(ctx);
    if (!ok) {
        fprintf(stderr, "FAIL: could not write fixture %s\n", path.c_str());
        failures++;
    }
    return path;
}

// ---------------------------------------------------------------------------
// JSON substring helpers (the report is a flat single-line JSON document)

static bool json_has(const std::string & j, const std::string & needle) {
    return j.find(needle) != std::string::npos;
}

static void expect_json(const char * what, const std::string & j, const std::string & needle) {
    if (!json_has(j, needle)) {
        fprintf(stderr, "FAIL %s: JSON missing '%s'\n  json=%s\n", what, needle.c_str(), j.c_str());
        failures++;
    }
}

// ---------------------------------------------------------------------------
// dense fixture: qwen35, 8 blocks, 1 nextn -> 7 trunk layers

static std::string dense_fixture() {
    std::vector<std::pair<std::string, std::string>> kvs = {
        { "general.architecture", "qwen35" },
        { "general.name", "DenseFixture" },
        { "qwen35.block_count", "8" },
        { "qwen35.embedding_length", "64" },
        { "qwen35.attention.head_count_kv", "2" },
        { "qwen35.attention.key_length", "256" },
        { "qwen35.attention.value_length", "256" },
        { "qwen35.ssm.conv_kernel", "4" },
        { "qwen35.ssm.state_size", "128" },
        { "qwen35.ssm.inner_size", "6144" },
        { "qwen35.ssm.group_count", "16" },
        { "qwen35.full_attention_interval", "4" },
        { "qwen35.nextn_predict_layers", "1" },
    };
    std::vector<fx_tensor> ts;
    auto t = [&](const char * name, int64_t a, int64_t b) {
        fx_tensor x{};
        x.name = name;
        x.type = GGML_TYPE_F32;
        x.ne[0] = a; x.ne[1] = b; x.ne[2] = 1; x.ne[3] = 1;
        ts.push_back(x);
    };
    t("token_embd.weight", 64, 64);   // 16384 B
    t("output_norm.weight", 64, 1);  //   256 B
    t("output.weight", 64, 64);       // 16384 B
    for (int il = 0; il < 7; ++il) {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
        t(nm, 64, 1);
        if (il != 3) { // recurrent layers
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", il);
            t(nm, 64, 64);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", il);
            t(nm, 64, 64);
        } else { // full-attention layer
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", il);
            t(nm, 64, 64);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", il);
            t(nm, 64, 64);
        }
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il);
        t(nm, 64, 64);
    }
    // MTP block 7: one non-nextn tensor + one nextn tensor
    t("blk.7.attn_norm.weight", 64, 1);       // mtp_layer 256 B
    t("blk.7.nextn.eh_proj.weight", 64, 64);  // mtp_nextn 16384 B
    return write_fixture_gguf("q38ls-inspect-dense.gguf", kvs, ts);
}

// ---------------------------------------------------------------------------
// MoE fixture: qwen35moe, 4 blocks, 1 nextn -> 3 trunk layers, 4 experts

static std::string moe_fixture(bool complete = true) {
    std::vector<std::pair<std::string, std::string>> kvs = {
        { "general.architecture", "qwen35moe" },
        { "general.name", "MoeFixture" },
        { "qwen35moe.block_count", "4" },
        { "qwen35moe.embedding_length", "64" },
        { "qwen35moe.attention.head_count_kv", "2" },
        { "qwen35moe.attention.key_length", "256" },
        { "qwen35moe.attention.value_length", "256" },
        { "qwen35moe.ssm.conv_kernel", "4" },
        { "qwen35moe.ssm.state_size", "128" },
        { "qwen35moe.ssm.inner_size", "4096" },
        { "qwen35moe.ssm.group_count", "16" },
        { "qwen35moe.full_attention_interval", "4" },
        { "qwen35moe.nextn_predict_layers", "1" },
        { "qwen35moe.expert_count", "4" },
        { "qwen35moe.expert_used_count", "2" },
    };
    std::vector<fx_tensor> ts;
    auto t = [&](const char * name, int64_t a, int64_t b) {
        fx_tensor x{};
        x.name = name;
        x.type = GGML_TYPE_F32;
        x.ne[0] = a; x.ne[1] = b; x.ne[2] = 1; x.ne[3] = 1;
        ts.push_back(x);
    };
    t("token_embd.weight", 64, 64);
    t("output_norm.weight", 64, 1);
    t("output.weight", 64, 64);
    for (int il = 0; il < 3; ++il) {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
        t(nm, 64, 1);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", il);
        t(nm, 64, 64);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", il);
        t(nm, 64, 64);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
        t(nm, 64, 64);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
        t(nm, 64, 64);
        if (complete || il != 1) {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
            t(nm, 64, 64);
        }
    }
    t("blk.3.attn_norm.weight", 64, 1);
    t("blk.3.nextn.enorm.weight", 64, 1);
    return write_fixture_gguf("q38ls-inspect-moe.gguf", kvs, ts);
}

// ---------------------------------------------------------------------------

static void run(const std::string & path, int64_t ctx, ggml_type kv, uint64_t vram,
                llama_ls_inventory & inv) {
    inv = llama_ls_model_inventory(path.c_str(), ctx, kv, vram, "unit-test");
}

int main() {
    // --- dense happy path ---------------------------------------------------
    {
        const std::string p = dense_fixture();
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (!inv.ok) {
            fprintf(stderr, "FAIL dense: inventory error: %s\n", inv.error.c_str());
            failures++;
        } else {
            const std::string & j = inv.json;
            expect_json("dense arch", j, "\"architecture\":\"qwen35\"");
            expect_json("dense name", j, "\"model_name\":\"DenseFixture\"");
            expect_json("dense trunk", j, "\"trunk_layers\":7,\"total_block_count\":8,\"nextn_predict_layers\":1");
            expect_json("dense experts absent", j, "\"expert_count\":0,\"expert_used_count\":0");
            // static = 16384+256+16384 = 33024
            expect_json("dense static", j, "\"static\":33024");
            // dense per-layer classes: norms 7*256=1792 attn_qkv+ssm_out 6*32768=196608
            // attn q+k (layer 3) 2*16384=32768 ffn 7*16384=114688; split:
            // attention (attn_q,k) = 32768, recurrent (qkv+ssm_out) = 196608,
            // dense_ffn = 114688, layer_norms = 1792
            expect_json("dense per_layer", j, "\"per_layer\":345856");
            expect_json("dense attn class", j, "\"attention\":32768");
            expect_json("dense rec class", j, "\"recurrent\":196608");
            expect_json("dense ffn class", j, "\"dense_ffn\":114688");
            expect_json("dense norms class", j, "\"layer_norms\":1792");
            expect_json("dense router class", j, "\"router\":0");
            expect_json("dense shexp class", j, "\"shared_expert\":0");
            expect_json("dense mtp_layer", j, "\"mtp_layer\":256");
            expect_json("dense mtp_nextn", j, "\"mtp_nextn\":16384");
            // KV: attn layers among 0..6 with (il+1)%4==0 -> only il=3.
            // n_embd_k_gqa = 2*256 = 512; Q8_0 row = 512/32*34 = 544; kv_size=262144
            // kv_bytes = 1 * 2 * 544 * 262144 = 285212672
            expect_json("dense kv", j, "\"kv_bytes\":285212672");
            expect_json("dense kv layers", j, "\"attn_layers\":1,\"recurrent_layers\":6");
            // state: r=(4-1)*(6144+2*16*128)=30720 -> 122880 B; s=128*6144=786432 -> 3145728 B
            // 6 * (122880+3145728) = 19611648
            expect_json("dense state", j, "\"recurrent_state_bytes\":19611648");
            expect_json("dense total", j, "\"total_data_bytes\":395520");
            expect_json("dense bounded1", j, "\"bounded_one_slot\":305203200");
            expect_json("dense vram null", j, "\"physical_vram_bytes\":null");
        }
        remove(p.c_str());
    }

    // --- MoE happy path -------------------------------------------------------
    {
        const std::string p = moe_fixture(true);
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 1024, inv);
        if (!inv.ok) {
            fprintf(stderr, "FAIL moe: inventory error: %s\n", inv.error.c_str());
            failures++;
        } else {
            const std::string & j = inv.json;
            expect_json("moe arch", j, "\"architecture\":\"qwen35moe\"");
            expect_json("moe trunk", j, "\"trunk_layers\":3,\"total_block_count\":4,\"nextn_predict_layers\":1");
            expect_json("moe experts", j, "\"expert_count\":4,\"expert_used_count\":2");
            expect_json("moe static", j, "\"static\":33024");
            // per-layer: 3*256 (norms) + 3*16384 (gate_inp) + 3*16384 (shexp) = 99072
            expect_json("moe per_layer", j, "\"per_layer\":99072");
            // routed experts: 3 layers * 3 tensors * 16384 = 147456
            expect_json("moe experts bytes", j, "\"routed_experts\":147456");
            expect_json("moe package", j, "{\"layer\":0,\"bytes\":49152,\"members\":3}");
            expect_json("moe largest", j, "\"moe_largest_bytes\":49152,\"moe_largest_layer\":0");
            // KV: attn layers among 0..2: (il+1)%4==0 -> none. kv_bytes = 0
            expect_json("moe kv zero", j, "\"kv_bytes\":0");
            expect_json("moe attn layers", j, "\"attn_layers\":0,\"recurrent_layers\":3");
            // state: r=3*(4096+4096)=24576 -> 98304; s=128*4096=524288 -> 2097152; 3*(98304+2097152)=6586368
            expect_json("moe state", j, "\"recurrent_state_bytes\":6586368");
            // total: static 33024 + per-layer 99072 + experts 147456 + mtp 256 + 256 = 280064
            expect_json("moe total", j, "\"total_data_bytes\":280064");
            // capacity vs 1024 B VRAM: full+runtime (279712+0+6586368) > 1024 -> not fit
            expect_json("moe full not fit", j, "\"full_fits_physical\":false");
            expect_json("moe no sub-full budget", j, "\"require_sub_full_budget\":false");
            expect_json("moe bounded not fit", j, "\"bounded_one_fits_physical\":false");
        }
        remove(p.c_str());
    }

    // --- malformed metadata -----------------------------------------------------
    {
        // unknown architecture
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "llama" },
        };
        const std::string p = write_fixture_gguf("q38ls-inspect-badarch.gguf", kvs, {});
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok || !json_has("e" + inv.error, "unsupported architecture")) {
            fprintf(stderr, "FAIL badarch: expected unsupported-architecture error, got ok=%d err=%s\n",
                    (int) inv.ok, inv.error.c_str());
            failures++;
        }
        remove(p.c_str());
    }
    {
        // missing required metadata: expert_count
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "qwen35moe" },
            { "qwen35moe.block_count", "4" },
            { "qwen35moe.embedding_length", "64" },
            { "qwen35moe.attention.head_count_kv", "2" },
            { "qwen35moe.attention.key_length", "256" },
            { "qwen35moe.attention.value_length", "256" },
            { "qwen35moe.ssm.conv_kernel", "4" },
            { "qwen35moe.ssm.state_size", "128" },
            { "qwen35moe.ssm.inner_size", "4096" },
            { "qwen35moe.ssm.group_count", "16" },
        };
        const std::string p = write_fixture_gguf("q38ls-inspect-nokv.gguf", kvs, {});
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok || inv.error.find("expert_count") == std::string::npos) {
            fprintf(stderr, "FAIL missing expert_count: err=%s\n", inv.error.c_str());
            failures++;
        }
        remove(p.c_str());
    }
    {
        // unclassified tensor
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "qwen35" },
            { "qwen35.block_count", "1" },
            { "qwen35.embedding_length", "64" },
            { "qwen35.attention.head_count_kv", "2" },
            { "qwen35.attention.key_length", "256" },
            { "qwen35.attention.value_length", "256" },
            { "qwen35.ssm.conv_kernel", "4" },
            { "qwen35.ssm.state_size", "128" },
            { "qwen35.ssm.inner_size", "6144" },
            { "qwen35.ssm.group_count", "16" },
        };
        std::vector<fx_tensor> ts;
        fx_tensor junk{};
        junk.name = "mystery.weight";
        junk.type = GGML_TYPE_F32;
        junk.ne[0] = 64; junk.ne[1] = 1; junk.ne[2] = 1; junk.ne[3] = 1;
        ts.push_back(junk);
        const std::string p = write_fixture_gguf("q38ls-inspect-junk.gguf", kvs, ts);
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok || inv.error.find("unclassified tensor") == std::string::npos) {
            fprintf(stderr, "FAIL unclassified tensor: err=%s\n", inv.error.c_str());
            failures++;
        }
        remove(p.c_str());
    }
    {
        // incomplete expert bank (layer 1 missing ffn_down_exps)
        const std::string p = moe_fixture(false);
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok || inv.error.find("incomplete routed-expert bank") == std::string::npos) {
            fprintf(stderr, "FAIL incomplete bank: err=%s\n", inv.error.c_str());
            failures++;
        }
        remove(p.c_str());
    }
    {
        // nonexistent file
        llama_ls_inventory inv;
        run("q38ls-inspect-does-not-exist.gguf", 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok) {
            fprintf(stderr, "FAIL missing file: expected error\n");
            failures++;
        }
    }
    {
        // expert_used_count > expert_count is implausible metadata -> error
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "qwen35moe" },
            { "qwen35moe.block_count", "4" },
            { "qwen35moe.embedding_length", "64" },
            { "qwen35moe.attention.head_count_kv", "2" },
            { "qwen35moe.attention.key_length", "256" },
            { "qwen35moe.attention.value_length", "256" },
            { "qwen35moe.ssm.conv_kernel", "4" },
            { "qwen35moe.ssm.state_size", "128" },
            { "qwen35moe.ssm.inner_size", "4096" },
            { "qwen35moe.ssm.group_count", "16" },
            { "qwen35moe.expert_count", "4" },
            { "qwen35moe.expert_used_count", "8" },
        };
        const std::string p = write_fixture_gguf("q38ls-inspect-badexp.gguf", kvs, {});
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok || inv.error.find("expert_used_count") == std::string::npos) {
            fprintf(stderr, "FAIL expert_used>count: err=%s\n", inv.error.c_str());
            failures++;
        }
        remove(p.c_str());
    }
    {
        // type-mismatched metadata (block_count as string) -> explicit error, no abort
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "qwen35" },
            { "qwen35.block_count", "eight" },
            { "qwen35.embedding_length", "64" },
            { "qwen35.attention.head_count_kv", "2" },
            { "qwen35.attention.key_length", "256" },
            { "qwen35.attention.value_length", "256" },
            { "qwen35.ssm.conv_kernel", "4" },
            { "qwen35.ssm.state_size", "128" },
            { "qwen35.ssm.inner_size", "6144" },
            { "qwen35.ssm.group_count", "16" },
        };
        const std::string p = write_fixture_gguf(
            "q38ls-inspect-badtype.gguf", kvs, {}, {}, { "qwen35.block_count" });
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok || inv.error.find("block_count") == std::string::npos) {
            fprintf(stderr, "FAIL type mismatch: err=%s\n", inv.error.c_str());
            failures++;
        }
        remove(p.c_str());
    }
    {
        // fused gate_up expert bank (blk.N.ffn_gate_up_exps + ffn_down_exps)
        // is a complete package supported by this tree
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "qwen35moe" },
            { "qwen35moe.block_count", "2" },
            { "qwen35moe.embedding_length", "64" },
            { "qwen35moe.attention.head_count_kv", "2" },
            { "qwen35moe.attention.key_length", "256" },
            { "qwen35moe.attention.value_length", "256" },
            { "qwen35moe.ssm.conv_kernel", "4" },
            { "qwen35moe.ssm.state_size", "128" },
            { "qwen35moe.ssm.inner_size", "4096" },
            { "qwen35moe.ssm.group_count", "16" },
            { "qwen35moe.nextn_predict_layers", "1" },
            { "qwen35moe.expert_count", "4" },
            { "qwen35moe.expert_used_count", "2" },
        };
        std::vector<fx_tensor> ts;
        auto t = [&](const char * name, int64_t a, int64_t b) {
            fx_tensor x{};
            x.name = name; x.type = GGML_TYPE_F32;
            x.ne[0] = a; x.ne[1] = b; x.ne[2] = 1; x.ne[3] = 1;
            ts.push_back(x);
        };
        t("token_embd.weight", 64, 64);
        t("output_norm.weight", 64, 1);
        t("output.weight", 64, 64);
        t("blk.0.attn_norm.weight", 64, 1);
        t("blk.0.ffn_gate_up_exps.weight", 64, 64); // 16384 B fused
        t("blk.0.ffn_down_exps.weight", 64, 64);   // 16384 B
        t("blk.1.attn_norm.weight", 64, 1);
        t("blk.1.nextn.enorm.weight", 64, 1);
        const std::string p = write_fixture_gguf("q38ls-inspect-fused.gguf", kvs, ts);
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (!inv.ok) {
            fprintf(stderr, "FAIL fused bank: %s\n", inv.error.c_str());
            failures++;
        } else {
            expect_json("fused routed bytes", inv.json, "\"routed_experts\":32768");
            expect_json("fused bank members", inv.json, "{\"layer\":0,\"bytes\":32768,\"members\":2}");
        }
        remove(p.c_str());
    }
    {
        // attention.recurrent_layers array overrides the interval heuristic:
        // blocks 0..2 trunk with [0,0,1] -> layer 2 is full-attention
        std::vector<std::pair<std::string, std::string>> kvs = {
            { "general.architecture", "qwen35moe" },
            { "qwen35moe.block_count", "4" },
            { "qwen35moe.embedding_length", "64" },
            { "qwen35moe.attention.head_count_kv", "2" },
            { "qwen35moe.attention.key_length", "256" },
            { "qwen35moe.attention.value_length", "256" },
            { "qwen35moe.ssm.conv_kernel", "4" },
            { "qwen35moe.ssm.state_size", "128" },
            { "qwen35moe.ssm.inner_size", "4096" },
            { "qwen35moe.ssm.group_count", "16" },
            { "qwen35moe.nextn_predict_layers", "1" },
            { "qwen35moe.expert_count", "4" },
            { "qwen35moe.expert_used_count", "2" },
        };
        std::vector<fx_tensor> ts;
        auto t = [&](const char * name, int64_t a, int64_t b) {
            fx_tensor x{};
            x.name = name; x.type = GGML_TYPE_F32;
            x.ne[0] = a; x.ne[1] = b; x.ne[2] = 1; x.ne[3] = 1;
            ts.push_back(x);
        };
        t("token_embd.weight", 64, 64);
        t("output_norm.weight", 64, 1);
        t("output.weight", 64, 64);
        for (int il = 0; il < 3; ++il) {
            char nm[64];
            snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
            t(nm, 64, 1);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", il);
            t(nm, 64, 64);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
            t(nm, 64, 64);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
            t(nm, 64, 64);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
            t(nm, 64, 64);
            if (il == 2) { // full-attention layer using fused qkv (review #4)
                snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", il);
                t(nm, 64, 64);
            } else {
                snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", il);
                t(nm, 64, 64);
            }
        }
        t("blk.3.attn_norm.weight", 64, 1);
        t("blk.3.nextn.enorm.weight", 64, 1);
        std::vector<std::pair<std::string, std::vector<uint8_t>>> arrs = {
            { "qwen35moe.attention.recurrent_layers", { 1, 1, 0, 0 } },
        };
        const std::string p = write_fixture_gguf("q38ls-inspect-recr.gguf", kvs, ts, arrs);
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (!inv.ok) {
            fprintf(stderr, "FAIL recurrent array: %s\n", inv.error.c_str());
            failures++;
        } else {
            expect_json("recr array layers", inv.json, "\"attn_layers\":1,\"recurrent_layers\":2");
            // layer 2 full-attention fused attn_qkv -> ATTENTION (16384), not recurrent
            expect_json("recr attn class", inv.json, "\"attention\":16384");
            expect_json("recr recurrent class", inv.json, "\"recurrent\":32768");
            // kv for 1 attn layer: 1 * 262144 * (544 + 544) = 285212672
            expect_json("recr kv", inv.json, "\"kv_bytes\":285212672");
        }
        remove(p.c_str());
    }
    {
        // file_bytes on a real fixture must be a plausible exact size, never null/garbage
        const std::string p = dense_fixture();
        llama_ls_inventory inv;
        run(p, 262144, GGML_TYPE_Q8_0, 0, inv);
        if (inv.ok) {
            if (!json_has(inv.json, "\"file_bytes\":")) {
                fprintf(stderr, "FAIL file_bytes missing\n");
                failures++;
            }
            FILE * fp = fopen(p.c_str(), "rb");
            if (fp) {
                fseek(fp, 0, SEEK_END);
                const long long fs = ftell(fp); // fixture is small; fine here
                fclose(fp);
                if (!json_has(inv.json, "\"file_bytes\":" + std::to_string(fs))) {
                    fprintf(stderr, "FAIL file_bytes mismatch (actual %lld): %s\n", fs, inv.json.c_str());
                    failures++;
                }
            }
        }
        remove(p.c_str());
    }

    // ------------------------------------------------------------------------
    // T-GPLR-05: host-only capability probe — deterministic result for every
    // input, supported policies only for adapter-declared architectures.
    {
        fprintf(stderr, "capability probe checks:\n");
        // dense fixture: supported, adapter qwen35, policy dense-layer
        {
            const std::string p = dense_fixture();
            llama_ls_capability cap;
            const std::string rec = llama_ls_probe_capability(p.c_str(), cap);
            if (!cap.supported || cap.arch != "qwen35" || cap.adapter != "qwen35" ||
                cap.policy != "dense-layer" || rec != "dense-layer" || cap.reason.empty()) {
                fprintf(stderr, "FAIL probe dense: supported=%d arch=%s adapter=%s policy=%s rec=%s\n",
                        (int) cap.supported, cap.arch.c_str(), cap.adapter.c_str(), cap.policy.c_str(), rec.c_str());
                failures++;
            }
            remove(p.c_str());
        }
        // MoE fixture: supported, adapter qwen35moe, policy moe-expert-bank
        {
            const std::string p = moe_fixture();
            llama_ls_capability cap;
            const std::string rec = llama_ls_probe_capability(p.c_str(), cap);
            if (!cap.supported || cap.arch != "qwen35moe" || cap.adapter != "qwen35moe" ||
                cap.policy != "moe-expert-bank" || rec != "moe-expert-bank" || cap.reason.empty()) {
                fprintf(stderr, "FAIL probe moe: supported=%d arch=%s adapter=%s policy=%s rec=%s\n",
                        (int) cap.supported, cap.arch.c_str(), cap.adapter.c_str(), cap.policy.c_str(), rec.c_str());
                failures++;
            }
            remove(p.c_str());
        }
        // unsupported architecture: deterministic refusal with a stable reason
        {
            std::vector<std::pair<std::string, std::string>> kvs = {
                { "general.architecture", "llama" },
                { "llama.block_count", "2" },
            };
            const std::string p = write_fixture_gguf("ls_probe_unsupported.gguf", kvs, {}, {}, {});
            llama_ls_capability cap;
            const std::string rec = llama_ls_probe_capability(p.c_str(), cap);
            if (cap.supported || rec != "off" || cap.reason.find("no layer-residency adapter") == std::string::npos) {
                fprintf(stderr, "FAIL probe unsupported: supported=%d rec=%s reason=%s\n",
                        (int) cap.supported, rec.c_str(), cap.reason.c_str());
                failures++;
            }
            remove(p.c_str());
        }
        // missing file: never throws, off with a stable reason
        {
            llama_ls_capability cap;
            const std::string rec = llama_ls_probe_capability("Z:/definitely/missing/model.gguf", cap);
            if (cap.supported || rec != "off" || cap.reason.empty()) {
                fprintf(stderr, "FAIL probe missing: supported=%d rec=%s reason=%s\n",
                        (int) cap.supported, rec.c_str(), cap.reason.c_str());
                failures++;
            }
        }
    }

    if (failures) {
        fprintf(stderr, "test-layer-streaming-model-inspect: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test-layer-streaming-model-inspect: all inventory + capability checks passed\n");
    return 0;
}
