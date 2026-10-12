// Standalone benchmark of the decode flash-attention kernels for the Qwen3.6-35B-A3B attention layers:
// head_dim 256, 16 Q heads, 2 KV heads (GQA 8), q8_0 K and V, one query token, KV length N (padded to 256).
// Compiled against the vendored ggml-cuda headers WITHOUT touching the tree. Variant 0 is the vector kernel of the tree
// (the kernel the model run uses), variant 1 is the candidate. Layers are rotated through separate KV buffers so that the
// L2 cache is cold the way it is in the model (10 attention layers read 10 x N x 1,088 B per token).
//
// For each KV length it checks both kernels against a double-precision CPU reference (heads 0 and 9, two softmax scales, four mask tails including
// a fully masked 128-key tile), compares all 16 heads of the candidate with the vector kernel, then times the two (10 layers rotated, cold L2).
// Build with tools\fa-bench\build.bat, run under the GPU mutex (tools\with-gpu-lock.ps1).
//
//   fa-bench.exe [N ...]        default N = 256 768 4096 24576 51200 102400 204800
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec.cuh"
#include "fattn-gqa8.cuh"   // the tree's own header: the benchmark measures exactly the shipped kernel

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// the two ggml symbols the headers' error macros reference (this benchmark links against nothing)
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d: %s\n", file, line, fmt);
    exit(1);
}
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s in %s at %s:%d: %s\n", stmt, func, file, line, msg);
    exit(1);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static constexpr int D       = 256;
static constexpr int NHEAD   = 16;
static constexpr int NKV     = 2;
static constexpr int GQA     = NHEAD / NKV;
static constexpr int NLAYER  = 10;     // attention layers of the model

struct layer_buf {
    char * K = nullptr;
    char * V = nullptr;
};

// host q8_0 quantization of one row of D floats into D/32 blocks (34 bytes each)
static void quant_row_q8_0(const float * x, char * out) {
    for (int b = 0; b < D / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d  = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        block_q8_0 * q = (block_q8_0 *) (out + b * sizeof(block_q8_0));
        q->d = __float2half(d);
        for (int i = 0; i < 32; ++i) q->qs[i] = (int8_t) std::lrintf(x[b * 32 + i] * id);
    }
}

static float dequant_q8_0(const char * row, int i) {
    const block_q8_0 * q = (const block_q8_0 *) (row + (i / 32) * sizeof(block_q8_0));
    return __half2float(q->d) * q->qs[i % 32];
}

// the geometry launch_fattn() picks for the vector kernel with ncols1 = 1, ncols2 = 1 (stream_k off)
static int pick_parallel_blocks(int ntiles_KV, int ntiles_dst, int nsm, int max_blocks_per_sm) {
    int parallel_blocks = std::min(max_blocks_per_sm, ntiles_KV);
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0, efficiency_percent_best = 0;
    for (int pbt = parallel_blocks; pbt <= ntiles_KV; ++pbt) {
        const int nblocks_total = ntiles_dst * pbt;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int eff = 100 * nblocks_total / (nwaves * blocks_per_wave);
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) break;
        if (eff > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = eff;
            parallel_blocks = pbt;
        }
    }
    return parallel_blocks;
}

struct run_ctx {
    int N;
    layer_buf layers[NLAYER];
    float * Q;       // [NHEAD][D]
    half *  mask;    // [N]
    float * dst;     // [NHEAD][D]
    float * dst_tmp; // [pb][NHEAD][D]
    float2 * meta;   // [pb][NHEAD]
    int nsm;
    float scale = 0.0625f;   // softmax scale given to both kernels
};

static void launch_vec_old(const run_ctx & c, int layer, cudaStream_t st, int * pb_out) {
    using kern_t = decltype(&flash_attn_ext_vec<D, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>);
    kern_t kern = flash_attn_ext_vec<D, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>;
    int occ = 1;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, 128, 0));
    const int ntiles_KV = (c.N + D - 1) / D;
    const int pb = pick_parallel_blocks(ntiles_KV, 1 * NHEAD * 1, c.nsm, occ);
    if (pb_out) *pb_out = pb;
    const float scale = c.scale;
    const uint3 ne01 = init_fastdiv_values(1);
    const int nb11 = 544, nb12 = 272, nb21 = 544, nb22 = 272;
    dim3 grid(1, pb, NHEAD), block(32, 4, 1);
    kern<<<grid, block, 0, st>>>(
        (const char *) c.Q, c.layers[layer].K, c.layers[layer].V, (const char *) c.mask, nullptr, nullptr,
        pb > 1 ? c.dst_tmp : c.dst, c.meta,
        scale, 0.0f, 1.0f, 1.0f, 16u, 0.0f,
        D, ne01, NHEAD, 1, D * 4, D * 4, NHEAD * D * 4,
        D, c.N, NKV, 1, nb11, nb12, (int64_t) c.N * 544,
        nb21, nb22, (int64_t) c.N * 544,
        c.N, 1, 1, c.N * 2, c.N * 2, (int64_t) c.N * 2);
    CK(cudaGetLastError());
    if (pb > 1) {
        flash_attn_combine_results<D><<<dim3(1, NHEAD, 1), dim3(D, 1, 1), pb * sizeof(float2), st>>>(c.dst_tmp, c.meta, c.dst, pb);
        CK(cudaGetLastError());
    }
}

static void launch_candidate(const run_ctx & c, int layer, cudaStream_t st, int * pb_out) {
    const int pb = ggml_cuda_fattn_gqa8_parallel_blocks(c.N, NKV, c.nsm, ggml_cuda_fattn_gqa8_occupancy(0));
    if (pb_out) *pb_out = pb;
    ggml_cuda_fattn_gqa8_launch((const char *) c.Q, D * 4, c.layers[layer].K, 544, 272, c.layers[layer].V, 544, 272, c.mask, c.dst, c.dst_tmp, c.meta, c.N, NKV, pb, c.scale, st);
}

int main(int argc, char ** argv) {
    std::vector<int> Ns;
    for (int i = 1; i < argc; ++i) Ns.push_back(atoi(argv[i]));
    if (Ns.empty()) Ns = {256, 768, 4096, 24576, 51200, 102400, 204800};

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int mclk = 0, bus = 0;
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("device %s, %d SMs, L2 %d MiB, mem clock %d kHz, bus %d bit\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, mclk, bus);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int failures = 0;

    for (int N : Ns) {
        run_ctx c;
        c.N = N;
        c.nsm = prop.multiProcessorCount;
        const size_t kv_bytes = (size_t) N * 544;
        // random K/V rows in q8_0, token-major layout with the two KV heads interleaved per token (the llama.cpp cache layout)
        std::vector<char> hK(kv_bytes), hV(kv_bytes);
        std::vector<float> row(D);
        for (int t = 0; t < N; ++t) {
            for (int h = 0; h < NKV; ++h) {
                for (int i = 0; i < D; ++i) row[i] = nd(rng);
                quant_row_q8_0(row.data(), hK.data() + (size_t) t * 544 + h * 272);
                for (int i = 0; i < D; ++i) row[i] = nd(rng);
                quant_row_q8_0(row.data(), hV.data() + (size_t) t * 544 + h * 272);
            }
        }
        for (int l = 0; l < NLAYER; ++l) {
            CK(cudaMalloc(&c.layers[l].K, kv_bytes));
            CK(cudaMalloc(&c.layers[l].V, kv_bytes));
            CK(cudaMemcpy(c.layers[l].K, hK.data(), kv_bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(c.layers[l].V, hV.data(), kv_bytes, cudaMemcpyHostToDevice));
        }
        std::vector<float> hQ(NHEAD * D);
        for (auto & v : hQ) v = nd(rng);
        CK(cudaMalloc(&c.Q, hQ.size() * sizeof(float)));
        CK(cudaMemcpy(c.Q, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&c.mask, N * sizeof(half)));
        const int PBMAX = 512;
        CK(cudaMalloc(&c.dst, NHEAD * D * sizeof(float)));
        CK(cudaMalloc(&c.dst_tmp, (size_t) PBMAX * NHEAD * D * sizeof(float)));
        CK(cudaMalloc(&c.meta, (size_t) PBMAX * NHEAD * sizeof(float2)));

        cudaStream_t st;
        CK(cudaStreamCreate(&st));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));

        // correctness against a double-precision CPU reference for heads 0 and 9 (one per KV head), for several masked tails
        // (a tail longer than 128 keys makes a whole tile -inf)
        const int ref_heads[2] = {0, 9};
        double err_by_variant[2] = {0, 0};
        for (float scl : {0.0625f, 0.2f}) for (int tail : {0, 37, 200, 255}) {
            c.scale = scl;   // the vector kernel and the candidate must both honour the op's scale, not only 1/sqrt(D)
            const int n_valid = N - tail;
            if (n_valid < 1) continue;
            std::vector<half> hmask(N);
            for (int t = 0; t < N; ++t) hmask[t] = __float2half(t < n_valid ? 0.0f : -INFINITY);
            CK(cudaMemcpy(c.mask, hmask.data(), N * sizeof(half), cudaMemcpyHostToDevice));
            std::vector<double> ref[2];
            for (int r = 0; r < 2; ++r) {
                const int h = ref_heads[r], kvh = h / GQA;
                std::vector<double> sc(n_valid);
                double mx = -1e300;
                for (int t = 0; t < n_valid; ++t) {
                    double s = 0;
                    for (int i = 0; i < D; ++i) s += (double) hQ[h * D + i] * dequant_q8_0(hK.data() + (size_t) t * 544 + kvh * 272, i);
                    s *= (double) c.scale;
                    sc[t] = s;
                    mx = std::max(mx, s);
                }
                double sum = 0;
                ref[r].assign(D, 0.0);
                for (int t = 0; t < n_valid; ++t) {
                    const double p = exp(sc[t] - mx);
                    sum += p;
                    for (int i = 0; i < D; ++i) ref[r][i] += p * dequant_q8_0(hV.data() + (size_t) t * 544 + kvh * 272, i);
                }
                for (int i = 0; i < D; ++i) ref[r][i] /= sum;
            }
            for (int variant = 0; variant < 2; ++variant) {
                CK(cudaMemset(c.dst, 0, NHEAD * D * sizeof(float)));
                if (variant == 0) launch_vec_old(c, 0, st, nullptr); else launch_candidate(c, 0, st, nullptr);
                CK(cudaStreamSynchronize(st));
                std::vector<float> out(NHEAD * D);
                CK(cudaMemcpy(out.data(), c.dst, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double maxerr = 0, maxref = 0;
                for (int r = 0; r < 2; ++r) {
                    for (int i = 0; i < D; ++i) {
                        maxerr = std::max(maxerr, std::fabs((double) out[ref_heads[r] * D + i] - ref[r][i]));
                        maxref = std::max(maxref, std::fabs(ref[r][i]));
                    }
                }
                err_by_variant[variant] = std::max(err_by_variant[variant], maxerr / maxref);
                if (!(maxerr / maxref < 5e-2) || std::isnan(maxerr)) {
                    printf("  FAIL N=%d tail=%d variant %d: max abs err %.3e (ref max %.3f)\n", N, tail, variant, maxerr, maxref);
                    ++failures;
                }
            }
        }
        // all heads: the candidate against the tree vector kernel (tail 37, the case of the model)
        {
            c.scale = 0.0625f;
            const int n_valid = N - std::min(37, N - 1);
            std::vector<half> hmask(N);
            for (int t = 0; t < N; ++t) hmask[t] = __float2half(t < n_valid ? 0.0f : -INFINITY);
            CK(cudaMemcpy(c.mask, hmask.data(), N * sizeof(half), cudaMemcpyHostToDevice));
            std::vector<float> o0(NHEAD * D), o1(NHEAD * D);
            launch_vec_old(c, 0, st, nullptr); CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(o0.data(), c.dst, o0.size() * sizeof(float), cudaMemcpyDeviceToHost));
            launch_candidate(c, 0, st, nullptr); CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(o1.data(), c.dst, o1.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double md = 0, mr = 0;
            for (size_t i = 0; i < o0.size(); ++i) { md = std::max(md, (double) std::fabs(o0[i] - o1[i])); mr = std::max(mr, (double) std::fabs(o0[i])); }
            printf("N=%7d correctness: max relative error vs the double reference over 4 mask tails: tree %.2e candidate %.2e | all 16 heads, candidate vs tree: max abs diff %.2e (output max %.3f)\n",
                   N, err_by_variant[0], err_by_variant[1], md, mr);
        }

        // timing: ITER tokens, each token visits the NLAYER layers in order (cold L2), mask with the 37 padded keys of the model
        const int ITER = N >= 51200 ? 20 : (N >= 4096 ? 100 : 400);
        for (int variant = 0; variant < 2; ++variant) {
            int pb = 0;
            if (variant == 0) launch_vec_old(c, 0, st, &pb); else launch_candidate(c, 0, st, &pb);
            for (int w = 0; w < 3; ++w) for (int l = 0; l < NLAYER; ++l) { if (variant == 0) launch_vec_old(c, l, st, nullptr); else launch_candidate(c, l, st, nullptr); }
            CK(cudaStreamSynchronize(st));
            CK(cudaEventRecord(e0, st));
            for (int it = 0; it < ITER; ++it)
                for (int l = 0; l < NLAYER; ++l) { if (variant == 0) launch_vec_old(c, l, st, nullptr); else launch_candidate(c, l, st, nullptr); }
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us_layer = ms * 1000.0 / (ITER * NLAYER);
            const double gbs = (double) N * 1088.0 / (us_layer * 1e-6) / 1e9;
            printf("N=%7d variant %d (%s): parallel_blocks %3d | %8.1f us per layer | %6.1f GB/s of KV\n",
                   N, variant, variant == 0 ? "tree vec kernel" : "candidate      ", pb, us_layer, gbs);
        }

        for (int l = 0; l < NLAYER; ++l) { cudaFree(c.layers[l].K); cudaFree(c.layers[l].V); }
        cudaFree(c.Q); cudaFree(c.mask); cudaFree(c.dst); cudaFree(c.dst_tmp); cudaFree(c.meta);
        cudaStreamDestroy(st);
    }
    printf("%s (%d failures)\n", failures ? "FAILED" : "all correctness checks passed", failures);
    return failures ? 1 : 0;
}
