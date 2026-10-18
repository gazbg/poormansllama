// T-GPLR-05 fixture generator: writes a minimal GGUF with an architecture that
// has no layer-residency adapter (llama). Host-only; no CUDA, no model load.
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <output.gguf>\\n", argv[0]);
        return 2;
    }
    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "general.name", "UnsupportedFixture");
    gguf_set_val_u32(ctx, "llama.block_count", 2);
    gguf_set_val_u32(ctx, "llama.embedding_length", 16);
    ggml_init_params gp = { /*mem_size=*/16 * 1024, /*mem_buffer=*/nullptr, /*no_alloc=*/true };
    ggml_context * gctx = ggml_init(gp);
    int64_t ne[4] = { 16, 16, 1, 1 };
    ggml_tensor * t = ggml_new_tensor(gctx, GGML_TYPE_F32, 4, ne);
    ggml_set_name(t, "token_embd.weight");
    gguf_add_tensor(ctx, t);
    static int8_t zeros[16 * 16 * 4] = {0};
    gguf_set_tensor_data(ctx, "token_embd.weight", zeros);
    const bool ok = gguf_write_to_file(ctx, argv[1], /*only_meta=*/false);
    ggml_free(gctx);
    gguf_free(ctx);
    fprintf(stderr, "wrote %s: %s\\n", argv[1], ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
