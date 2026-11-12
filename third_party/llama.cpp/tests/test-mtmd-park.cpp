// VOD-1 (docs/plans/vision-on-demand-plan.md): park / unpark of the vision projector. GPU test, run under tools/with-gpu-lock.ps1:
//   test-mtmd-park <mmproj.gguf> <text-model.gguf> [cycles=100]
// The text model is loaded VOCAB ONLY (no weights): mtmd needs its vocab and n_embd to tokenize the image markers.
//
// What it checks (plan criteria VOD-2 / VOD-3 at library level):
//   1. a context created with park_on_init holds no device memory (cudaMemGetInfo free memory within 64 MiB of the free memory with no projector at all);
//   2. after `cycles` x (unpark, encode, park) the embeddings of every fixture image are BYTE-IDENTICAL (memcmp) to the ones a resident (never parked) context produces;
//   3. the free device memory after each park is within 64 MiB of the baseline (no leak, the backend pool and the compute buffers are gone);
//   4. unpark of an already unparked context, and park of a parked one, are no-ops that succeed.
// Reports: t_unpark_ms, t_first_encode_ms (includes the compute-buffer allocation of the real batch), t_park_ms, device bytes resident while unparked.
#include "mtmd.h"
#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool dev_free_mib(double & free_mib) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(dev, &f, &t);
            free_mib = f / 1048576.0;
            return true;
        }
    }
    return false;
}

// deterministic test image: a coloured square on a grey field plus a gradient, so that every patch differs
static std::vector<unsigned char> make_image(int w, int h, int seed) {
    std::vector<unsigned char> rgb((size_t) w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char * p = &rgb[((size_t) y * w + x) * 3];
            const bool in = x > w / 3 && x < 2 * w / 3 && y > h / 3 && y < 2 * h / 3;
            p[0] = in ? (unsigned char) (40 + seed * 50) : (unsigned char) ((x * 255) / w);
            p[1] = in ? (unsigned char) (200 - seed * 40) : (unsigned char) ((y * 255) / h);
            p[2] = in ? (unsigned char) (30 + seed * 20) : (unsigned char) (((x + y) * 255) / (w + h));
        }
    }
    return rgb;
}

struct Fix {
    int w, h;
    std::vector<unsigned char> rgb;
    std::vector<float> ref;   // embeddings of the resident context
    size_t n_tokens = 0;
};

// encode the image chunk of `text` + one bitmap with `ctx`, copy the embeddings out
static bool encode_one(mtmd_context * ctx, const Fix & f, std::vector<float> & out, size_t & n_tokens, size_t n_embd) {
    mtmd_bitmap * bm = mtmd_bitmap_init(f.w, f.h, f.rgb.data());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    std::string marker = mtmd_default_marker();
    mtmd_input_text txt = { marker.c_str(), marker.size(), false, true };
    const mtmd_bitmap * bms[1] = { bm };
    int32_t rc = mtmd_tokenize(ctx, chunks, &txt, bms, 1);
    bool ok = rc == 0;
    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; ok && i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            img = c;
        }
    }
    ok = ok && img != nullptr;
    if (ok) {
        mtmd_batch * b = mtmd_batch_init(ctx);
        ok = mtmd_batch_add_chunk(b, img) == 0 && mtmd_batch_encode(b) == 0;
        if (ok) {
            n_tokens = mtmd_input_chunk_get_n_tokens(img);
            const float * e = mtmd_batch_get_output_embd(b, img);
            ok = e != nullptr;
            if (ok) {
                out.assign(e, e + n_tokens * n_embd);
            }
        }
        mtmd_batch_free(b);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bm);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <mmproj.gguf> <text-model.gguf> [cycles=100]\n", argv[0]);
        return 2;
    }
    const char * mmproj = argv[1];
    const char * tmodel = argv[2];
    const int cycles = argc > 3 ? atoi(argv[3]) : 100;
    int fails = 0;
    auto check = [&](bool c, const char * what) {
        printf("%s %s\n", c ? "ok  " : "FAIL", what);
        if (!c) { fails++; }
    };

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(tmodel, mp);
    if (!model) { fprintf(stderr, "cannot load the vocab of %s\n", tmodel); return 2; }
    const size_t n_embd = (size_t) llama_model_n_embd_inp(model);

    std::vector<Fix> fx(3);
    const int dims[3][2] = { { 448, 448 }, { 672, 512 }, { 896, 896 } };
    for (int i = 0; i < 3; i++) {
        fx[i].w = dims[i][0]; fx[i].h = dims[i][1];
        fx[i].rgb = make_image(fx[i].w, fx[i].h, i);
    }

    double free_none = 0;
    if (!dev_free_mib(free_none)) { fprintf(stderr, "no GPU device\n"); return 2; }

    // 1. the reference: a resident context
    mtmd_context_params cp = mtmd_context_params_default();
    cp.print_timings = false;
    mtmd_context * res = mtmd_init_from_file(mmproj, model, cp);
    if (!res) { fprintf(stderr, "resident init failed\n"); return 2; }
    double free_res = 0; dev_free_mib(free_res);
    printf("resident projector: device free %.1f -> %.1f MiB (%.1f MiB used)\n", free_none, free_res, free_none - free_res);
    for (auto & f : fx) {
        if (!encode_one(res, f, f.ref, f.n_tokens, n_embd)) { fprintf(stderr, "resident encode failed\n"); return 2; }
        printf("  fixture %dx%d: %zu tokens\n", f.w, f.h, f.n_tokens);
    }
    mtmd_free(res);
    res = nullptr;
    double free_base = 0; dev_free_mib(free_base);   // the baseline: no projector (the resident one is freed)
    check(std::fabs(free_base - free_none) < 64.0, "resident projector freed completely (free memory back within 64 MiB)");

    // 2. the parked context
    cp.park_on_init = true;
    double t0 = now_ms();
    mtmd_context * pk = mtmd_init_from_file(mmproj, model, cp);
    const double t_init = now_ms() - t0;
    if (!pk) { fprintf(stderr, "parked init failed\n"); return 2; }
    double free_parked = 0; dev_free_mib(free_parked);
    printf("parked init: %.1f ms, device free %.1f MiB (baseline %.1f)\n", t_init, free_parked, free_base);
    check(mtmd_is_parked(pk), "context is parked after park_on_init");
    check(std::fabs(free_parked - free_base) < 64.0, "parked context holds no device memory (within 64 MiB of the baseline)");
    check(mtmd_park(pk), "park of a parked context is a no-op that succeeds");

    double max_leak = 0, sum_unpark = 0, sum_first = 0, sum_park = 0, max_resident = 0;
    int n_identical = 0, n_total = 0;
    for (int c = 0; c < cycles; c++) {
        for (size_t k = 0; k < fx.size(); k++) {
            t0 = now_ms();
            bool ok = mtmd_unpark(pk);
            const double t_unpark = now_ms() - t0;
            if (!ok) { check(false, "unpark succeeded"); return 1; }
            std::vector<float> emb;
            size_t n_tok = 0;
            t0 = now_ms();
            ok = encode_one(pk, fx[k], emb, n_tok, n_embd);
            const double t_enc = now_ms() - t0;
            double free_up = 0; dev_free_mib(free_up);
            max_resident = std::fmax(max_resident, free_base - free_up);
            if (!ok) { check(false, "encode on the unparked context"); return 1; }
            n_total++;
            if (n_tok == fx[k].n_tokens && emb.size() == fx[k].ref.size() && memcmp(emb.data(), fx[k].ref.data(), emb.size() * sizeof(float)) == 0) {
                n_identical++;
            } else if (n_total - n_identical <= 3) {
                size_t nd = 0; double maxd = 0;
                for (size_t i = 0; i < emb.size() && i < fx[k].ref.size(); i++) {
                    if (emb[i] != fx[k].ref[i]) { nd++; maxd = std::fmax(maxd, std::fabs(emb[i] - fx[k].ref[i])); }
                }
                printf("  DIFF cycle %d fixture %zu: %zu of %zu floats differ, max abs %.3g\n", c, k, nd, emb.size(), maxd);
            }
            t0 = now_ms();
            ok = mtmd_park(pk);
            const double t_park = now_ms() - t0;
            if (!ok) { check(false, "park succeeded"); return 1; }
            double free_down = 0; dev_free_mib(free_down);
            max_leak = std::fmax(max_leak, free_base - free_down);
            sum_unpark += t_unpark; sum_first += t_enc; sum_park += t_park;
            if (c == 0) {
                printf("  cycle 0 fixture %zu: unpark %.1f ms, encode %.1f ms (incl. compute buffer alloc), park %.1f ms, resident while unparked %.1f MiB, after park %.1f MiB\n",
                       k, t_unpark, t_enc, t_park, free_base - free_up, free_base - free_down);
            }
        }
    }
    const int n = cycles * (int) fx.size();
    printf("%d unpark/encode/park rounds: mean unpark %.1f ms, mean encode %.1f ms, mean park %.1f ms; max resident while unparked %.1f MiB; max residue after park %.1f MiB\n",
           n, sum_unpark / n, sum_first / n, sum_park / n, max_resident, max_leak);
    check(n_identical == n_total, "embeddings byte-identical to the resident context in every round");
    check(max_leak < 64.0, "no device residue after park (max over all rounds < 64 MiB)");
    check(mtmd_unpark(pk) && mtmd_unpark(pk), "unpark of an unparked context is a no-op that succeeds");
    check(mtmd_park(pk), "final park");

    mtmd_free(pk);
    llama_model_free(model);
    llama_backend_free();
    printf("%s (%d identical of %d)\n", fails == 0 ? "ALL PASS" : "FAILED", n_identical, n_total);
    return fails == 0 ? 0 : 1;
}
