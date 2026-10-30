#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Prefill form for head size 128 and a scalar gate (no snapshots): LPC lanes per state column, each lane keeps 128/LPC rows of its column in registers.
// The generic kernel gives a warp to a column and reduces over its 32 lanes twice per token (two 5-step shuffle chains); here a reduction is log2(LPC) shuffles and a lane does 128/LPC FMAs per pass.
// k and q of T tokens go through shared memory, one copy per block of 128 threads. The sums are formed in another order than in the generic kernel.
template <int LPC, int T>
__global__ void __launch_bounds__(128, 4)
gated_delta_net_quad(const float * q, const float * k, const float * v, const float * g, const float * beta,
                     const float * curr_state, float * dst, float * state,
                     int64_t H, int64_t n_tokens, int64_t n_seqs,
                     int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                     int64_t sb1, int64_t sb2, int64_t sb3,
                     const uint3 neqk1_magic, const uint3 rq3_magic, float scale) {
    constexpr int S_v = 128;
    constexpr int RPL = S_v/LPC;            // rows per lane
    constexpr int CPW = 32/LPC;             // columns per warp
    constexpr int CPB = 4*CPW;              // columns per block
    constexpr int PAD = S_v + LPC;          // floats per staged token row: one pad float per row group so that the groups of a warp read different banks
    __shared__ float ks[T*PAD];
    __shared__ float qs[T*PAD];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid  = threadIdx.x;
    const int      lane = tid & 31;
    const int      warp = tid >> 5;
    const int      rg   = lane & (LPC - 1);              // row group of the lane
    const int      col  = blockIdx.z*CPB + warp*CPW + lane/LPC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence*H*S_v*S_v + h_idx*S_v*S_v;
    const int64_t state_out_offset = (sequence*H + h_idx)*S_v*S_v;
    const float * s_in      = curr_state + state_in_offset + col*S_v + rg*RPL;
    float *       s_out     = state + state_out_offset + col*S_v + rg*RPL;
    float *       attn_data = dst + (sequence*n_tokens*H + h_idx)*S_v;

    ggml_cuda_pdl_sync();
    float S[RPL];
#pragma unroll
    for (int j = 0; j < RPL; ++j) {
        S[j] = s_in[j];
    }

    for (int t0 = 0; t0 < n_tokens; t0 += T) {
        const int nt = (int) min((int64_t) T, n_tokens - t0);
        __syncthreads();   // the previous chunk has been consumed
        for (int idx = tid; idx < nt*S_v; idx += 128) {
            const int     tt = idx >> 7;
            const int     i  = idx & 127;
            const int64_t o  = iq3*sq3 + (t0 + tt)*sq2 + iq1*sq1;
            const int     pi = tt*PAD + i + i/RPL;
            ks[pi] = k[o + i];
            qs[pi] = q[o + i];
        }
        __syncthreads();

        for (int tt = 0; tt < nt; ++tt) {
            const int64_t t          = t0 + tt;
            const int64_t gb_offset  = sequence*sb3 + t*sb2 + h_idx*sb1;
            const float   beta_val   = beta[gb_offset];
            const float   g_val      = expf(g[gb_offset]);
            const float   v_col      = v[sequence*sv3 + t*sv2 + h_idx*sv1 + col];
            const float * kr = ks + tt*PAD + rg*(RPL + 1);
            const float * qr = qs + tt*PAD + rg*(RPL + 1);

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int j = 0; j < RPL; j += 4) {
                a0 += S[j + 0]*kr[j + 0];
                a1 += S[j + 1]*kr[j + 1];
                a2 += S[j + 2]*kr[j + 2];
                a3 += S[j + 3]*kr[j + 3];
            }
            float kv = (a0 + a1) + (a2 + a3);
#pragma unroll
            for (int m = 1; m < LPC; m <<= 1) {
                kv += __shfl_xor_sync(0xffffffff, kv, m);
            }

            const float delta = (v_col - g_val*kv)*beta_val;

            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
            for (int j = 0; j < RPL; j += 4) {
                S[j + 0] = g_val*S[j + 0] + kr[j + 0]*delta;
                S[j + 1] = g_val*S[j + 1] + kr[j + 1]*delta;
                S[j + 2] = g_val*S[j + 2] + kr[j + 2]*delta;
                S[j + 3] = g_val*S[j + 3] + kr[j + 3]*delta;
                b0 += S[j + 0]*qr[j + 0];
                b1 += S[j + 1]*qr[j + 1];
                b2 += S[j + 2]*qr[j + 2];
                b3 += S[j + 3]*qr[j + 3];
            }
            float at = (b0 + b1) + (b2 + b3);
#pragma unroll
            for (int m = 1; m < LPC; m <<= 1) {
                at += __shfl_xor_sync(0xffffffff, at, m);
            }
            if (rg == 0) {
                attn_data[t*S_v*H + col] = at*scale;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < RPL; ++j) {
        s_out[j] = S[j];
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            if constexpr (!KDA && !keep_rs_t) {
                // opt-in (GGML_CUDA_GDN_QUAD=1): the prefill form above, for 16 tokens or more
                static const int quad = [] { const char * e = getenv("GGML_CUDA_GDN_QUAD"); return e != nullptr ? atoi(e) : 0; }();
                static const int quad_t = [] { const char * e = getenv("GGML_CUDA_GDN_QUAD_T"); return e != nullptr ? atoi(e) : 8; }();
                if (quad > 0 && n_tokens >= 16 && warp_size == 32) {
                    const int lpc = quad == 1 ? 4 : quad;   // 1 = the default of 4 lanes per column
                    const int cpb = 4*(32/lpc);
                    const ggml_cuda_kernel_launch_params lp_quad = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 128/cpb), dim3(128, 1, 1), 0, stream);
#define GDN_QUAD_LAUNCH(L, TT) ggml_cuda_kernel_launch(gated_delta_net_quad<L, TT>, lp_quad, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale)
                    if      (lpc == 2 && quad_t == 4)  { GDN_QUAD_LAUNCH(2, 4); }
                    else if (lpc == 2 && quad_t == 8)  { GDN_QUAD_LAUNCH(2, 8); }
                    else if (lpc == 2 && quad_t == 16) { GDN_QUAD_LAUNCH(2, 16); }
                    else if (lpc == 4 && quad_t == 4)  { GDN_QUAD_LAUNCH(4, 4); }
                    else if (lpc == 4 && quad_t == 16) { GDN_QUAD_LAUNCH(4, 16); }
                    else if (lpc == 8 && quad_t == 4)  { GDN_QUAD_LAUNCH(8, 4); }
                    else if (lpc == 8 && quad_t == 8)  { GDN_QUAD_LAUNCH(8, 8); }
                    else if (lpc == 8 && quad_t == 16) { GDN_QUAD_LAUNCH(8, 16); }
                    else                               { GDN_QUAD_LAUNCH(4, 8); }
#undef GDN_QUAD_LAUNCH
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    // H51: GGML_CUDA_GDN_EPS=e scales the output of every gated delta net layer by 1+e (a measurement knob for the noise floor of the first-token KL; default 0)
    static const float gdn_eps = [] { const char * e = getenv("GGML_CUDA_GDN_EPS"); return e != nullptr ? (float) atof(e) : 0.0f; }();
    const float scale = (1.0f / sqrtf((float) S_v)) * (1.0f + gdn_eps);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
