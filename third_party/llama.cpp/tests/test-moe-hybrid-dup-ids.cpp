// Regression test for the hybrid MoE decode on a model whose routed-expert banks are NOT quantized (the bf16 layers of the Unsloth Q8_K_XL quants of Qwen3.6-35B-A3B).
//
// The hybrid engine (src/llama-moe-hybrid.cpp) maps every expert that is not in its VRAM cache to ONE zero slot, so the ids of a token routinely repeat a slot.
// CUDA MUL_MAT_ID sends floating-point banks through mmf, whose id helper keeps ONE entry per (token, expert): the other outputs are never written and the model reads
// stale memory (measured 2026-10-19: Q8_K_XL answered "2, 3, 5, 5, 7, 9, ..." for the first primes while Q4_K_S, whose banks are all quantized and go through mmvq, was right).
// The hybrid graph marks its MUL_MAT_ID nodes with GGML_MOE_HYBRID_ID_FLAG_DUP so the dispatch keeps them on the per-output kernels (mmvq / mmvf).
//
// This test builds exactly that situation: expert banks of every type a model may use, [k, n, E] with the last slot all zero, ids with 0..8 distinct hits followed by the
// zero slot repeated, 1..4 tokens (the hybrid widths). The flagged node must match the CPU backend, which handles repeated ids correctly. The UNflagged node is run and
// reported but not asserted: whether the stale memory happens to be right is not a property to pin.
//
// Needs a CUDA device; without one it prints SKIP and returns 0. It initializes CUDA, so run it under tools/with-gpu-lock.ps1.
//
//   test-moe-hybrid-dup-ids
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_checks   = 0;
static int n_failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        ++n_checks;                                          \
        if (!(cond)) {                                       \
            ++n_failures;                                    \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
        }                                                    \
    } while (0)

static const int K      = 256; // contraction length of the bank rows (a multiple of every block size used)
static const int N_ROWS = 64;  // output rows of one expert
static const int N_SLOT = 20;  // slots incl. the zero slot (the last one)
static const int N_USED = 8;

static std::vector<char> quantize_rows(ggml_type type, const std::vector<float> & f, int64_t n_rows, int64_t n_per_row) {
    std::vector<char> out(ggml_row_size(type, n_per_row)*n_rows);
    if (type == GGML_TYPE_F32) {
        std::memcpy(out.data(), f.data(), out.size());
    } else {
        const ggml_type_traits * tt = ggml_get_type_traits(type);
        for (int64_t r = 0; r < n_rows; ++r) {
            tt->from_float_ref(f.data() + r*n_per_row, out.data() + r*ggml_row_size(type, n_per_row), n_per_row);
        }
    }
    return out;
}

// y = mul_mat_id(W, x, ids) on `backend`; returns [N_ROWS * N_USED * T] floats
static std::vector<float> run_mul_mat_id(ggml_backend_t backend, ggml_type type, const std::vector<char> & w, const std::vector<float> & x, const std::vector<int32_t> & ids, int T, bool flag) {
    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * W   = ggml_new_tensor_3d(ctx, type, K, N_ROWS, N_SLOT);
    ggml_tensor * X   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * IDS = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, T);
    ggml_tensor * Y   = ggml_mul_mat_id(ctx, W, X, IDS);
    if (flag) {
        ((int32_t *) Y->op_params)[GGML_MOE_HYBRID_ID_FLAG_IDX] = GGML_MOE_HYBRID_ID_FLAG_DUP;
    }
    ggml_set_output(Y);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(W, w.data(), 0, w.size());
    ggml_backend_tensor_set(X, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(IDS, ids.data(), 0, ids.size()*sizeof(int32_t));
    // poison the output so an output that is never written cannot pass by luck of zero-initialised memory
    std::vector<float> poison((size_t) N_ROWS*N_USED*T, 12345.0f);
    ggml_backend_tensor_set(Y, poison.data(), 0, poison.size()*sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    std::vector<float> y((size_t) N_ROWS*N_USED*T, 0.0f);
    if (st == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(Y, y.data(), 0, y.size()*sizeof(float));
    } else {
        y.assign(y.size(), NAN);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

// largest |a - b| over the RMS of b; NaN-safe (any NaN in a counts as infinitely wrong)
static double rel_err(const std::vector<float> & a, const std::vector<float> & b) {
    double ss = 0.0, mx = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        ss += (double) b[i]*b[i];
    }
    const double rms = std::sqrt(ss/(double) b.size()) + 1e-12;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (!(d == d)) {
            return INFINITY;
        }
        mx = std::max(mx, d);
    }
    return mx/rms;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        std::printf("SKIP: no GPU backend\n");
        return 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (gpu == nullptr || cpu == nullptr) {
        std::printf("SKIP: backend init failed\n");
        return 0;
    }
    std::printf("device: %s\n", ggml_backend_dev_description(dev));

    struct type_case { ggml_type type; double tol; };
    const type_case types[] = {
        { GGML_TYPE_BF16, 3e-2 }, { GGML_TYPE_F16, 3e-2 }, { GGML_TYPE_F32, 1e-3 }, { GGML_TYPE_Q8_0, 4e-2 }, { GGML_TYPE_Q4_K, 1.5e-1 }, // CUDA quantizes the activation to q8_1, the CPU to q8_K: the noise floor of a k=256 dot, not the ids
    };

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const type_case & tc : types) {
        // expert banks: real random experts 0 .. N_SLOT-2, the last slot (the hybrid zero slot) is all zero
        std::vector<float> wf((size_t) K*N_ROWS*N_SLOT, 0.0f);
        for (size_t i = 0; i < (size_t) K*N_ROWS*(N_SLOT - 1); ++i) {
            wf[i] = nd(rng)*0.1f;
        }
        const std::vector<char> w = quantize_rows(tc.type, wf, (int64_t) N_ROWS*N_SLOT, K);

        for (int T = 1; T <= 4; ++T) {
            for (int variant = 0; variant < 4; ++variant) {
                std::vector<float> x((size_t) K*T);
                for (float & v : x) {
                    v = nd(rng);
                }
                // variant 0: no repeat (m = 8 hits); 1: m random hits then the zero slot repeated; 2: all misses; 3: a single hit
                std::vector<int32_t> ids((size_t) N_USED*T);
                bool has_dup = false;
                for (int t = 0; t < T; ++t) {
                    int m = variant == 0 ? N_USED : variant == 1 ? (int) (rng() % (N_USED + 1)) : variant == 2 ? 0 : 1;
                    std::vector<int32_t> pick(N_SLOT - 1);
                    for (int i = 0; i < N_SLOT - 1; ++i) {
                        pick[i] = i;
                    }
                    std::shuffle(pick.begin(), pick.end(), rng);
                    for (int i = 0; i < N_USED; ++i) {
                        ids[(size_t) t*N_USED + i] = i < m ? pick[i] : N_SLOT - 1;
                    }
                    has_dup = has_dup || (N_USED - m) > 1;
                }

                const std::vector<float> ref  = run_mul_mat_id(cpu, tc.type, w, x, ids, T, false);
                const std::vector<float> got  = run_mul_mat_id(gpu, tc.type, w, x, ids, T, true);
                const std::vector<float> bare = run_mul_mat_id(gpu, tc.type, w, x, ids, T, false);
                const double e_flag = rel_err(got, ref);
                const double e_bare = rel_err(bare, ref);
                CHECK(e_flag < tc.tol, "%s T=%d variant=%d: flagged MUL_MAT_ID differs from the CPU reference by %.3g (tolerance %.3g)", ggml_type_name(tc.type), T, variant, e_flag, tc.tol);
                std::printf("  %-5s T=%d variant=%d dup=%d flagged err %.2e%s | unflagged err %.2e%s\n", ggml_type_name(tc.type), T, variant, (int) has_dup, e_flag,
                            e_flag < tc.tol ? " ok" : " FAIL", e_bare, (e_bare >= tc.tol && has_dup) ? "  <- the unflagged dispatch drops repeated ids" : "");
            }
        }
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    std::printf("%d checks, %d failures\n", n_checks, n_failures);
    return n_failures == 0 ? 0 : 1;
}
