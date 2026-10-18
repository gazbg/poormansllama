#pragma once

// Opt-in decode attention for one query token (G30: up to four, the rows of an MTP verify step), head_dim 256, q8_0 K and V, 8 query heads per KV head (LLAMA_FATTN_GQA8=1).
// G30: blockIdx.x is the row (token) of the step; every row is its own CTA, the K/V tiles of the rows are read by neighbouring CTAs and hit in L2.
// The vector kernel gives each query head its own CTA, so a KV head is read and dequantized 8 times, and each key costs a warp
// shuffle reduction per head. Here one CTA serves one KV head: QK is one thread per key (dp4a against the q8_1 Q of the 8 heads,
// no cross-lane reduction), PV is one warp per key (lane = 8 dims, the V row is dequantized once for the 8 heads).
// Tiles of 128 keys are dealt round-robin over gridDim.y like in the vector kernel and the partial results use the same
// dst_tmp / meta layout, so flash_attn_combine_results finishes the job.

#include "common.cuh"
#include "fattn-common.cuh"

#include <atomic>

namespace fattn_gqa8 {

constexpr int D    = 256;
constexpr int NH   = 8;         // query heads per KV head
constexpr int NW   = 4;         // warps per CTA
constexpr int TILE = 32 * NW;   // keys per CTA iteration
constexpr int NBLK = D / 32;    // q8_0 blocks per row
static_assert(FATTN_KQ_STRIDE % TILE == 0, "a KV length that is a multiple of FATTN_KQ_STRIDE must hold whole tiles");

static __device__ __forceinline__ int sx8(uint32_t x, int i) {   // signed byte i of x
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return (int) (int8_t) (x >> (8 * i));
#else
    int r;
    asm("bfe.s32 %0, %1, %2, 8;" : "=r"(r) : "r"(x), "r"(8 * i));
    return r;
#endif
}

__launch_bounds__(128, 2)
static __global__ void kernel(
        const char * GGML_CUDA_RESTRICT Qp, const char * GGML_CUDA_RESTRICT Kp, const char * GGML_CUDA_RESTRICT Vp, const half * GGML_CUDA_RESTRICT mask,
        float * GGML_CUDA_RESTRICT dst, float2 * GGML_CUDA_RESTRICT dst_meta,
        const float scale, const int N, const int nb01, const int nb02, const int nb11, const int nb12, const int nb21, const int nb22, const int mask_stride, const int n_head) {
#ifdef FLASH_ATTN_AVAILABLE
    const int row   = blockIdx.x;                    // G30: the token of the step
    mask += (size_t) row * mask_stride;
    const int lane  = threadIdx.x;
    const int warp  = threadIdx.y;
    const int tid   = warp * 32 + lane;
    const int kvh   = blockIdx.z;
    const int head0 = kvh * NH;
    Kp += (size_t) kvh * nb12;
    Vp += (size_t) kvh * nb22;

    __shared__ __align__(16) int    Qq[NH * 64];
    __shared__ __align__(16) float2 Qds[NH * 8];
    __shared__ __align__(16) float Pw[NW][32 * NH];
    __shared__ __align__(16) float red[NW][NH * D];
    __shared__ float mw[NW][NH];
    __shared__ float lw[NW][NH];

    // Q to q8_1, pre-scaled, the same quantization as the vector kernel
    for (int j = warp; j < NH; j += NW) {
        const float * Q_f = (const float *) (Qp + (size_t) row * nb01 + (size_t) (head0 + j) * nb02);
#pragma unroll
        for (int i0 = 0; i0 < D / (int) sizeof(int); i0 += 32) {
            quantize_q8_1_to_shared<float2, 32>(Q_f + i0 * sizeof(int), scale, Qq + j * 64 + i0, Qds + j * 8 + i0 / QI8_1);
        }
    }
    __syncthreads();

    float m[NH], l[NH], acc[NH][8];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        m[h] = -FLT_MAX / 2.0f;
        l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[h][i] = 0.0f;
    }

    const int blk  = lane >> 2;                      // q8_0 block of this lane's 8 dims in the PV stage
    const int voff = 34 * blk + 2 + 8 * (lane & 3);  // byte offset of those 8 int8 values inside a row
    const int vwa  = voff & ~3;
    const int vsh  = (voff & 3) * 8;                 // 16 for even blocks, 0 for odd blocks

    for (int k0 = blockIdx.y * TILE; k0 < N; k0 += gridDim.y * TILE) {
        const int key = k0 + warp * 32 + lane;

        // QK: one thread per key, the row (272 B) is read once as 17 x 16 B
        uint32_t w[68];
        {
            const uint4 * rowp = (const uint4 *) (Kp + (size_t) key * nb11);
#pragma unroll
            for (int g = 0; g < 17; ++g) {
                const uint4 v = rowp[g];
                w[4 * g + 0] = v.x; w[4 * g + 1] = v.y; w[4 * g + 2] = v.z; w[4 * g + 3] = v.w;
            }
        }
        float S[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) S[h] = 0.0f;
#pragma unroll
        for (int b = 0; b < NBLK; ++b) {
            const int mm = b >> 1;   // blocks are 34 B: even blocks start 2 B into a word, odd blocks are word aligned
            int kq[8];
            float dK;
            if ((b & 1) == 0) {
                dK = __half2float(__ushort_as_half((unsigned short) (w[17 * mm] & 0xFFFFu)));
#pragma unroll
                for (int j = 0; j < 8; ++j) kq[j] = (int) __funnelshift_r(w[17 * mm + j], w[17 * mm + j + 1], 16);
            } else {
                dK = __half2float(__ushort_as_half((unsigned short) (w[17 * mm + 8] >> 16)));
#pragma unroll
                for (int j = 0; j < 8; ++j) kq[j] = (int) w[17 * mm + 9 + j];
            }
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                const int4 q0 = *(const int4 *) &Qq[h * 64 + b * 8];
                const int4 q1 = *(const int4 *) &Qq[h * 64 + b * 8 + 4];
                int s = 0;
                s = ggml_cuda_dp4a(kq[0], q0.x, s);
                s = ggml_cuda_dp4a(kq[1], q0.y, s);
                s = ggml_cuda_dp4a(kq[2], q0.z, s);
                s = ggml_cuda_dp4a(kq[3], q0.w, s);
                s = ggml_cuda_dp4a(kq[4], q1.x, s);
                s = ggml_cuda_dp4a(kq[5], q1.y, s);
                s = ggml_cuda_dp4a(kq[6], q1.z, s);
                s = ggml_cuda_dp4a(kq[7], q1.w, s);
                S[h] += (float) s * (dK * Qds[h * 8 + b].x);
            }
        }
        const float mk = __half2float(mask[key]);
#pragma unroll
        for (int h = 0; h < NH; ++h) S[h] += mk;

        // online softmax over the 32 keys of this warp's tile
        float p[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            const float tmax  = warp_reduce_max<WARP_SIZE>(S[h]);
            const float m_new = fmaxf(m[h], tmax);
            const float sc    = expf(m[h] - m_new);
            m[h] = m_new;
            p[h] = expf(S[h] - m_new);
            l[h] = l[h] * sc + p[h];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[h][i] *= sc;
        }
        *(float4 *) &Pw[warp][lane * 8]     = make_float4(p[0], p[1], p[2], p[3]);
        *(float4 *) &Pw[warp][lane * 8 + 4] = make_float4(p[4], p[5], p[6], p[7]);
        __syncwarp();

        // PV: one key at a time per warp, lane = 8 consecutive dims
        const char * Vt = Vp + (size_t) (k0 + warp * 32) * nb21;
#pragma unroll 4
        for (int kk = 0; kk < 32; ++kk) {
            const char * vr = Vt + (size_t) kk * nb21;
            const float4 pa = *(const float4 *) &Pw[warp][kk * 8];
            const float4 pb = *(const float4 *) &Pw[warp][kk * 8 + 4];
            const float  dv = __half2float(*(const half *) (vr + 34 * blk));
            const uint32_t v0 = *(const uint32_t *) (vr + vwa);
            const uint32_t v1 = *(const uint32_t *) (vr + vwa + 4);
            const uint32_t v2 = vsh ? *(const uint32_t *) (vr + vwa + 8) : 0u;
            const uint32_t lo = vsh ? __funnelshift_r(v0, v1, 16) : v0;
            const uint32_t hi = vsh ? __funnelshift_r(v1, v2, 16) : v1;
            float qf[8];
            qf[0] = (float) sx8(lo, 0); qf[1] = (float) sx8(lo, 1); qf[2] = (float) sx8(lo, 2); qf[3] = (float) sx8(lo, 3);
            qf[4] = (float) sx8(hi, 0); qf[5] = (float) sx8(hi, 1); qf[6] = (float) sx8(hi, 2); qf[7] = (float) sx8(hi, 3);
            const float pd[NH] = {pa.x * dv, pa.y * dv, pa.z * dv, pa.w * dv, pb.x * dv, pb.y * dv, pb.z * dv, pb.w * dv};
#pragma unroll
            for (int h = 0; h < NH; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[h][i] = fmaf(pd[h], qf[i], acc[h][i]);
            }
        }
        __syncwarp();
    }

    // merge the 4 warps, write the CTA's partial result (normalized when this is the only CTA of the head)
    float lsum[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) lsum[h] = warp_reduce_sum<WARP_SIZE>(l[h]);
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h) { mw[warp][h] = m[h]; lw[warp][h] = lsum[h]; }
    }
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        *(float4 *) &red[warp][h * D + lane * 8]     = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        *(float4 *) &red[warp][h * D + lane * 8 + 4] = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        float mt = mw[0][h];
#pragma unroll
        for (int ww = 1; ww < NW; ++ww) mt = fmaxf(mt, mw[ww][h]);
        float st = 0.0f, o0 = 0.0f, o1 = 0.0f;
#pragma unroll
        for (int ww = 0; ww < NW; ++ww) {
            const float f = expf(mw[ww][h] - mt);
            st += lw[ww][h] * f;
            o0 += red[ww][h * D + 2 * tid] * f;
            o1 += red[ww][h * D + 2 * tid + 1] * f;
        }
        if (gridDim.y == 1) { o0 /= st; o1 /= st; }
        *(float2 *) &dst[(((size_t) row * n_head + head0 + h) * gridDim.y + blockIdx.y) * D + 2 * tid] = make_float2(o0, o1);
        if (tid == 0) dst_meta[((size_t) row * n_head + head0 + h) * gridDim.y + blockIdx.y] = make_float2(mt, st);
    }
#else
    GGML_UNUSED_VARS(Qp, Kp, Vp, mask, dst, dst_meta, scale, N, nb01, nb02, nb11, nb12, nb21, nb22, mask_stride, n_head);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

} // namespace fattn_gqa8

static bool ggml_cuda_fattn_gqa8_enabled() {
    static const bool on = []() {
        const char * e = getenv("LLAMA_FATTN_GQA8");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// Below this KV length the vector kernel is as fast or faster (measured: equal at 768 keys, 1.5x slower at 256); LLAMA_FATTN_GQA8_MIN_KV overrides.
static int ggml_cuda_fattn_gqa8_min_kv() {
    static const int v = []() {
        const char * e = getenv("LLAMA_FATTN_GQA8_MIN_KV");
        return e != nullptr ? atoi(e) : 2048;
    }();
    return v;
}

// Blocks per SM for the kernel, queried once per device. The first query loads the kernel module, so applies() makes it on the first flash-attention op of any
// size and not in the middle of a decode (a driver call on the launching thread while the hybrid engine's GPU kernels spin is the deadlock of G5).
static int ggml_cuda_fattn_gqa8_occupancy(const int device) {
    static std::atomic<int> occ[GGML_CUDA_MAX_DEVICES];
    int o = occ[device].load(std::memory_order_relaxed);
    if (o == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, fattn_gqa8::kernel, 128, 0));
        o = std::max(o, 1);
        occ[device].store(o, std::memory_order_relaxed);
    }
    return o;
}

// Shapes this kernel serves: one to four tokens (G30), one sequence, head_dim 256, q8_0 K and V, 8 query heads per KV head, an f16 mask, no sinks, no alibi, no softcap.
static bool ggml_cuda_fattn_gqa8_applies(const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_gqa8_enabled()) {
        return false;
    }
    (void) ggml_cuda_fattn_gqa8_occupancy(ggml_cuda_get_device());
    if (dst->src[1]->ne[1] < ggml_cuda_fattn_gqa8_min_kv()) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    return sinks == nullptr && mask != nullptr && max_bias == 0.0f && softcap == 0.0f &&
        Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && mask->type == GGML_TYPE_F16 &&
        Q->ne[0] == fattn_gqa8::D && K->ne[0] == fattn_gqa8::D && V->ne[0] == fattn_gqa8::D &&
        Q->ne[1] >= 1 && Q->ne[1] <= 4 && mask->ne[1] >= Q->ne[1] && Q->nb[1] % sizeof(float) == 0 && mask->nb[1] % sizeof(ggml_fp16_t) == 0 &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        K->ne[2] >= 1 && Q->ne[2] == fattn_gqa8::NH * K->ne[2] && V->ne[2] == K->ne[2] &&
        K->ne[1] > 0 && K->ne[1] % FATTN_KQ_STRIDE == 0 && V->ne[1] == K->ne[1] && mask->ne[0] >= K->ne[1] &&
        Q->nb[0] == sizeof(float) && Q->nb[2] % sizeof(float) == 0 && (uintptr_t) Q->data % sizeof(float) == 0 &&
        K->nb[0] == sizeof(block_q8_0) && V->nb[0] == sizeof(block_q8_0) &&
        mask->nb[0] == sizeof(ggml_fp16_t) && dst->nb[0] == sizeof(float) && dst->nb[1] == fattn_gqa8::D * sizeof(float) &&
        K->nb[1] % 16 == 0 && K->nb[2] % 16 == 0 && V->nb[1] % 16 == 0 && V->nb[2] % 16 == 0 &&
        (uintptr_t) K->data % 16 == 0 && (uintptr_t) V->data % 16 == 0;
}

// Same wave-efficiency search as launch_fattn() uses for the vector kernel, with one tile row per KV head.
static int ggml_cuda_fattn_gqa8_parallel_blocks(const int N, const int nkv, const int nrows, const int nsm, const int occ) {
    const int ntiles_KV = N / fattn_gqa8::TILE;
    const int blocks_per_wave = nsm * occ;
    int pb = std::min(occ, ntiles_KV), nwaves_best = 0, eff_best = 0;
    for (int pbt = pb; pbt <= ntiles_KV; ++pbt) {
        const int nblocks_total = nkv * nrows * pbt;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int eff = 100 * nblocks_total / (nwaves * blocks_per_wave);
        if (eff_best >= 95 && nwaves > nwaves_best) {
            break;
        }
        if (eff > eff_best) {
            nwaves_best = nwaves;
            eff_best = eff;
            pb = pbt;
        }
    }
    return pb;
}

// scale is the op's own softmax scale (op_params[0]), applied to Q before it is quantized, like launch_fattn() does
static void ggml_cuda_fattn_gqa8_launch(const char * Q, const int nb01, const int nb02, const char * K, const int nb11, const int nb12, const char * V, const int nb21, const int nb22,
        const half * mask, const int mask_stride, float * dst, float * dst_tmp, float2 * dst_meta, const int N, const int nkv, const int nrows, const int pb, const float scale, cudaStream_t stream) {
    fattn_gqa8::kernel<<<dim3(nrows, pb, nkv), dim3(32, fattn_gqa8::NW, 1), 0, stream>>>(
        Q, K, V, mask, pb > 1 ? dst_tmp : dst, dst_meta, scale, N, nb01, nb02, nb11, nb12, nb21, nb22, mask_stride, nkv * fattn_gqa8::NH);
    CUDA_CHECK(cudaGetLastError());
    if (pb > 1) {
        flash_attn_combine_results<fattn_gqa8::D><<<dim3(nrows, nkv * fattn_gqa8::NH, 1), dim3(fattn_gqa8::D, 1, 1), pb * sizeof(float2), stream>>>(dst_tmp, dst_meta, dst, pb);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void ggml_cuda_flash_attn_ext_gqa8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int N     = (int) K->ne[1];
    const int nkv   = (int) K->ne[2];
    const int nrows = (int) Q->ne[1];
    const int pb    = ggml_cuda_fattn_gqa8_parallel_blocks(N, nkv, nrows, ggml_cuda_info().devices[ctx.device].nsm, ggml_cuda_fattn_gqa8_occupancy(ctx.device));

    // the pool is a stack: allocate in declaration order so that the destructors free in reverse
    ggml_cuda_pool_alloc<float>  dst_tmp(ctx.pool());
    ggml_cuda_pool_alloc<float2> dst_meta(ctx.pool());
    if (pb > 1) {
        dst_tmp.alloc((size_t) pb * ggml_nelements(dst));
    }
    dst_meta.alloc((size_t) pb * ggml_nrows(dst));
    ggml_cuda_fattn_gqa8_launch((const char *) Q->data, (int) Q->nb[1], (int) Q->nb[2], (const char *) K->data, (int) K->nb[1], (int) K->nb[2],
        (const char *) V->data, (int) V->nb[1], (int) V->nb[2], (const half *) mask->data, (int) (mask->nb[1] / sizeof(ggml_fp16_t)), (float *) dst->data, dst_tmp.ptr, dst_meta.ptr,
        N, nkv, nrows, pb, scale, ctx.stream());
}
