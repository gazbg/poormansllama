// T-GPLR-40 addendum: ownership probe for the 144k pass-2 crash.
//
// Reproduces the --pp-repeat crash shape with STOCK libllama only — no
// layer-streaming, no residency module, no common helpers:
//   1. create a context (ctx 144128, ubatch from argv, KV type from argv)
//   2. decode the prompt in ubatch-sized batches (fill pass 1)
//   3. llama_memory_clear(mem, true)  <- the pass boundary
//   4. decode the same prompt again   (fill pass 2)
// If the crash is in stock q8 KV clear/reuse at long ctx, step 4 dies with an
// illegal memory access exactly like the residency run did. f16 is the control.
//
// Build (see tools/run-gplr40-stock.ps1): cl against the built llama.dll.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static void batch_add(llama_batch & b, llama_token id, llama_pos pos, int seq, bool logits) {
    b.token   [b.n_tokens] = id;
    b.pos     [b.n_tokens] = pos;
    b.n_seq_id[b.n_tokens] = 1;
    b.seq_id  [b.n_tokens][0] = seq;
    b.logits  [b.n_tokens] = logits;
    b.n_tokens++;
}

static ggml_type kv_type_from_str(const char * s) {
    if (strcmp(s, "f16") == 0) return GGML_TYPE_F16;
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8") == 0) return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> [kv=f16|q8_0] [ubatch=1024] [ctx=144128]\n", argv[0]);
        return 2;
    }
    const char * model_path  = argv[1];
    const char * prompt_path = argv[2];
    const char * kv          = argc > 3 ? argv[3] : "q8_0";
    const int    ubatch      = argc > 4 ? atoi(argv[4]) : 1024;
    const int    n_ctx_req   = argc > 5 ? atoi(argv[5]) : 144128;

    FILE * f = fopen(prompt_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", prompt_path); return 2; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz + 1);
    size_t rd = fread(buf.data(), 1, sz, f);
    fclose(f);
    buf[rd] = 0;
    fprintf(stderr, "probe: prompt file %ld bytes\n", (long) rd);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_tok = -llama_tokenize(vocab, buf.data(), (int32_t) rd, nullptr, 0, false, false);
    std::vector<llama_token> tokens((size_t) n_tok);
    if (llama_tokenize(vocab, buf.data(), (int32_t) rd, tokens.data(), (int32_t) tokens.size(), false, false) != n_tok) {
        fprintf(stderr, "tokenize mismatch\n"); return 1;
    }
    fprintf(stderr, "probe: %d tokens\n", n_tok);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = (uint32_t) n_ctx_req;
    cparams.n_batch  = (uint32_t) ubatch;
    cparams.n_ubatch = (uint32_t) ubatch;
    cparams.n_seq_max = 1;
    cparams.type_k   = kv_type_from_str(kv);
    cparams.type_v   = kv_type_from_str(kv);
    if (cparams.type_k == GGML_TYPE_COUNT) { fprintf(stderr, "bad kv type %s\n", kv); return 2; }
    fprintf(stderr, "probe: ctx=%d ubatch=%d kv=%s\n", n_ctx_req, ubatch, kv);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    for (int pass = 1; pass <= 2; ++pass) {
        fprintf(stderr, "probe: === fill pass %d ===\n", pass);
        int n_past = 0;
        for (int i = 0; i < n_tok; i += ubatch) {
            const int n = (i + ubatch <= n_tok) ? ubatch : (n_tok - i);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; ++j) {
                batch_add(b, tokens[(size_t)(i + j)], n_past++, 0, false);
            }
            b.logits[b.n_tokens - 1] = true;
            if (llama_decode(ctx, b) != 0) {
                fprintf(stderr, "probe: DECODE FAILED pass %d at token %d\n", pass, i);
                return 1;
            }
            llama_batch_free(b);
            if ((i / ubatch) % 20 == 0) {
                fprintf(stderr, "probe: pass %d at token %d / %d\n", pass, i, n_tok);
            }
        }
        fprintf(stderr, "probe: pass %d COMPLETE (%d tokens)\n", pass, n_past);
        if (pass == 1) {
            fprintf(stderr, "probe: llama_memory_clear(mem, true) -- the pass boundary\n");
            llama_memory_clear(mem, true);
        }
    }
    fprintf(stderr, "probe: SURVIVED both fills with kv=%s — no crash in stock libllama\n", kv);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
