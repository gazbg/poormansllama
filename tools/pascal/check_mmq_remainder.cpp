// Bit-exactness of GGML_CUDA_MMQ_REMAINDER: the same MUL_MAT_ID on the CUDA backend with the mode off and on must give identical bits.
// The routing is uniform or skewed (expert popularity ~ 1/(e+1)^0.8), so the number of columns an expert gets (and the size of its leftover tile) covers every class.
// With the default mode, every case also prints one `HASH name fnv` line: compare them between two library builds (LD_LIBRARY_PATH) to check a kernel change for bit-identity.
//   g++ -O2 -I third_party/llama.cpp/ggml/include tools/pascal/check_mmq_remainder.cpp -L build/llama-cuda/bin -lggml -lggml-base -lggml-cpu -lggml-cuda \
//       -Wl,-rpath,$PWD/build/llama-cuda/bin -o build/pascal/check_mmq_remainder
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

struct problem {
    ggml_type type;
    int       n_mats, n_used, m, n, k;
    bool      skew;
    const char * name;
};

static std::vector<float> run(ggml_backend_t be, const problem & p, const std::vector<char> & w, const std::vector<float> & b, const std::vector<int32_t> & ids, const char * mode) {
    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, p.type, p.k, p.m, p.n_mats);
    ggml_tensor * bt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, p.k, p.n_used, p.n);
    ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, p.n_used, p.n);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, bt, idt);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(as, w.data(), 0, w.size());
    ggml_backend_tensor_set(bt, b.data(), 0, b.size()*sizeof(float));
    ggml_backend_tensor_set(idt, ids.data(), 0, ids.size()*sizeof(int32_t));
    setenv("GGML_CUDA_MMQ_REMAINDER", mode ? mode : "0", 1);   // "off" is the explicit 0: the default is mode 1 on Pascal
    ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    std::vector<float> res((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    unsetenv("GGML_CUDA_MMQ_REMAINDER");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::printf("no CUDA backend\n"); return 2; }
    const problem probs[] = {
        { GGML_TYPE_Q4_K,   256, 8,  512, 2048, 2048, false, "q4_K gate/up uniform 2048 tok" },
        { GGML_TYPE_Q4_K,   256, 8,  512, 2048, 2048, true,  "q4_K gate/up skewed  2048 tok" },
        { GGML_TYPE_Q5_K,   256, 8, 2048, 2048,  512, true,  "q5_K down    skewed  2048 tok" },
        { GGML_TYPE_Q6_K,   256, 8, 2048, 1000,  512, true,  "q6_K down    skewed  1000 tok" },
        { GGML_TYPE_Q4_K,    64, 8,  256,  777, 1024, true,  "q4_K 64 experts, 777 tok" },
        { GGML_TYPE_Q5_K,   128, 6,  384,  333, 1024, true,  "q5_K 128 experts, 333 tok" },
        { GGML_TYPE_Q8_0,    64, 8,  256, 1500, 1024, true,  "q8_0 64 experts, 1500 tok" },
        { GGML_TYPE_Q4_0,    64, 8,  256, 1500, 1024, true,  "q4_0 64 experts, 1500 tok" },
        { GGML_TYPE_IQ4_XS,  64, 8,  256, 1500, 1024, true,  "iq4_xs 64 experts, 1500 tok" },
        { GGML_TYPE_Q4_K,   256, 8,  512,  130, 2048, true,  "q4_K, 130 tok (few columns)" },
    };
    int bad = 0, total = 0;
    for (const problem & p : probs) {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> wf((size_t) p.k*p.m);
        std::vector<char>  w((size_t) p.n_mats*p.m*ggml_row_size(p.type, p.k));
        for (int e = 0; e < p.n_mats; ++e) {
            for (auto & v : wf) { v = nd(rng); }
            if (ggml_quantize_requires_imatrix(p.type)) { std::printf("skip %s (needs an imatrix)\n", p.name); goto next; }
            ggml_quantize_chunk(p.type, wf.data(), w.data() + (size_t) e*p.m*ggml_row_size(p.type, p.k), 0, p.m, p.k, nullptr);
        }
        {
            std::vector<float> b((size_t) p.k*p.n_used*p.n);
            for (auto & v : b) { v = nd(rng); }
            std::vector<double> pop((size_t) p.n_mats);
            for (int e = 0; e < p.n_mats; ++e) { pop[(size_t) e] = p.skew ? 1.0/std::pow(e + 1.0, 0.8) : 1.0; }
            std::discrete_distribution<int> pick(pop.begin(), pop.end());
            std::vector<int32_t> ids((size_t) p.n_used*p.n);
            std::vector<int> count((size_t) p.n_mats, 0);
            for (int t = 0; t < p.n; ++t) {
                std::set<int> s;
                while ((int) s.size() < p.n_used) { s.insert(pick(rng)); }
                int i = 0;
                for (int e : s) { ids[(size_t) t*p.n_used + i++] = e; count[(size_t) e]++; }
            }
            int rem_hist[5] = {};   // experts by leftover class: none, <=16, <=32, <=48, <=63
            for (int e = 0; e < p.n_mats; ++e) { const int r = count[(size_t) e] % 64; rem_hist[r == 0 ? 0 : r <= 16 ? 1 : r <= 32 ? 2 : r <= 48 ? 3 : 4]++; }
            const std::vector<float> ref = run(be, p, w, b, ids, nullptr);
            {   // FNV-1a of the output bits with the default mode: run this binary against two sets of libraries and compare the HASH lines to see whether a kernel change moved any bit
                unsigned long long h = 1469598103934665603ULL;
                for (float v : ref) { unsigned u; memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ULL; }
                std::printf("HASH %-34s %016llx\n", p.name, h);
            }
            std::printf("%-34s leftover classes (0/16/32/48/63): %3d %3d %3d %3d %3d |", p.name, rem_hist[0], rem_hist[1], rem_hist[2], rem_hist[3], rem_hist[4]);
            for (const char * mode : { "1", "2", "3" }) {
                const std::vector<float> got = run(be, p, w, b, ids, mode);
                size_t nd_ = 0;
                for (size_t i = 0; i < ref.size(); ++i) { nd_ += memcmp(&ref[i], &got[i], sizeof(float)) != 0; }
                ++total; bad += nd_ != 0;
                std::printf(" mode %s %s |", mode, nd_ == 0 ? "identical" : "DIFFERS");
                if (nd_ != 0) { std::printf(" (%zu of %zu floats)", nd_, ref.size()); }
            }
            std::printf("\n");
        }
        next:;
    }
    std::printf("%d comparisons, %d differ\n", total, bad);
    ggml_backend_free(be);
    return bad == 0 ? 0 : 1;
}
