// Plan H93, milestone 1 of the two-GPU prefill (sequence split): can the state of a sequence computed on one GPU be handed to a second model instance on the other GPU and continued there
// with the same result? The first s tokens are decoded on GPU 0, the sequence state (attention K/V and the recurrent state of the delta-net layers) is exported and imported into a second
// model + context on GPU 1, the remaining tokens are decoded there, and the logits of the last token are compared with a single decode of the whole prompt on GPU 0.
// This is the invariant the pipelined version rests on; it says nothing about speed (the experts run on the host here, as in the server, and nothing overlaps).
//   g++ -O2 -std=c++17 -I third_party/llama.cpp/include -I third_party/llama.cpp/ggml/include tools/pascal/seqsplit_probe.cpp -L build/llama-cuda/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$PWD/build/llama-cuda/bin -o build/pascal/seqsplit_probe
//   build/pascal/seqsplit_probe MODEL.gguf TEXT.txt [n_tokens=1024] [split=512]
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static llama_model * load(const char * path, const char * dev_name) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
    if (dev == nullptr) {
        std::fprintf(stderr, "no device %s\n", dev_name);
        std::exit(1);
    }
    static ggml_backend_dev_t devs[2];
    devs[0] = dev;
    devs[1] = nullptr;
    llama_model_tensor_buft_override ov[2] = { { "\\.ffn_(up|gate|down)_exps\\.weight", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };
    llama_model_params mp = llama_model_default_params();
    mp.devices               = devs;
    mp.n_gpu_layers          = 99;
    mp.split_mode            = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu              = 0;
    mp.tensor_buft_overrides = ov;
    llama_model * m = llama_model_load_from_file(path, mp);
    if (m == nullptr) {
        std::fprintf(stderr, "cannot load the model on %s\n", dev_name);
        std::exit(1);
    }
    return m;
}

static llama_context * make_ctx(llama_model * m, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = n_ctx;
    cp.n_batch  = 512;
    cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * c = llama_init_from_model(m, cp);
    if (c == nullptr) {
        std::fprintf(stderr, "cannot create a context\n");
        std::exit(1);
    }
    return c;
}

// decode tokens[from, to) of sequence 0 at positions from.., ubatch by ubatch; the logits of the last token stay in the context
static void decode_range(llama_context * c, const std::vector<llama_token> & t, int from, int to) {
    const int ub = 512;
    for (int i = from; i < to; i += ub) {
        const int n = std::min(ub, to - i);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int k = 0; k < n; ++k) {
            b.token[k]     = t[(size_t) (i + k)];
            b.pos[k]       = i + k;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = 0;
            b.logits[k]    = (i + k == to - 1) ? 1 : 0;
        }
        b.n_tokens = n;
        if (llama_decode(c, b) != 0) {
            std::fprintf(stderr, "decode failed at %d\n", i);
            std::exit(1);
        }
        llama_batch_free(b);
    }
}

static std::vector<float> last_logits(llama_context * c, int n_vocab) {
    const float * l = llama_get_logits_ith(c, -1);
    return std::vector<float>(l, l + n_vocab);
}

static void compare(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    double maxd = 0.0, sumd = 0.0;
    size_t ia = 0, ib = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        maxd = std::max(maxd, d);
        sumd += d;
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    // KL(softmax(a) || softmax(b))
    double ma = a[ia], mb = b[ib], za = 0.0, zb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { za += std::exp(a[i] - ma); zb += std::exp(b[i] - mb); }
    double kl = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double pa = std::exp(a[i] - ma) / za, pb = std::exp(b[i] - mb) / zb;
        if (pa > 1e-12) kl += pa * (std::log(pa) - std::log(std::max(pb, 1e-30)));
    }
    std::printf("%-44s max|dlogit| %.5f  mean|dlogit| %.6f  KL %.3e  argmax %s (%zu / %zu)\n", what, maxd, sumd / a.size(), kl, ia == ib ? "same" : "DIFFERENT", ia, ib);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf TEXT.txt [n_tokens=1024] [split=512]\n", argv[0]);
        return 2;
    }
    const int N = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int S = argc > 4 ? std::atoi(argv[4]) : 512;
    ggml_backend_load_all();
    llama_backend_init();

    llama_model * ma = load(argv[1], "CUDA0");
    llama_model * mb = load(argv[1], "CUDA1");
    const llama_vocab * vocab = llama_model_get_vocab(ma);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream f(argv[2]);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str().substr(0, (size_t) N * 8);
    std::vector<llama_token> toks((size_t) N * 4);
    int nt = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (nt < N) {
        std::fprintf(stderr, "text gives only %d tokens\n", nt);
        return 1;
    }
    toks.resize((size_t) N);
    std::printf("model on CUDA0 and CUDA1, %d tokens, split at %d\n", N, S);

    llama_context * ca  = make_ctx(ma, 4096);
    llama_context * ca2 = make_ctx(ma, 4096);
    llama_context * cb  = make_ctx(mb, 4096);
    llama_context * cb2 = make_ctx(mb, 4096);

    // reference: the whole prompt on GPU 0, twice (the run-to-run floor), and on GPU 1 (the cross-GPU floor)
    decode_range(ca, toks, 0, N);
    const std::vector<float> ref = last_logits(ca, n_vocab);
    decode_range(ca2, toks, 0, N);
    compare("GPU 0 whole prompt, run 1 vs run 2", ref, last_logits(ca2, n_vocab));
    decode_range(cb2, toks, 0, N);
    compare("GPU 0 whole prompt vs GPU 1 whole prompt", ref, last_logits(cb2, n_vocab));

    // split: first S tokens on GPU 0 (a fresh context), state to GPU 1, rest there
    llama_context * cs = make_ctx(ma, 4096);
    decode_range(cs, toks, 0, S);
    const size_t sz = llama_state_seq_get_size_ext(cs, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> blob(sz);
    const size_t got = llama_state_seq_get_data_ext(cs, blob.data(), blob.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::printf("sequence state after %d tokens: %zu bytes (%.1f MiB), exported %zu\n", S, sz, sz / 1048576.0, got);
    const size_t put = llama_state_seq_set_data_ext(cb, blob.data(), got, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::printf("imported into the GPU 1 context: %zu bytes\n", put);
    if (put == 0) {
        std::printf("RESULT: FAIL (the import was refused)\n");
        return 1;
    }
    decode_range(cb, toks, S, N);
    compare("GPU 0 whole prompt vs split GPU 0 | GPU 1", ref, last_logits(cb, n_vocab));
    return 0;
}
