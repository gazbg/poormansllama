// Inner loop of the Q4_K MMQ kernel on Pascal (dp4a) in shared memory only: the arithmetic of vec_dot_q4_K_q8_1_impl_mmq, several thread tiles and hoisting patterns.
// Tile: I = 64 rows of weights (33 ints per row = packed nibbles of 256 values, 4 ints of scales and mins, one half2 d / dmin per row), J = 64 columns
// (4 half2 scales + 32 ints per column), K = 256 per pass (two calls of 32 ints, each two steps of 16). All variants must give the same bits.
//   nvcc -O3 -arch=sm_61 -std=c++17 tools/pascal/mmq_q4k_inner_bench.cu -o build/pascal/mmq_q4k_inner_bench && CUDA_VISIBLE_DEVICES=0 build/pascal/mmq_q4k_inner_bench
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int I = 64, J = 64;
constexpr int XS = 33, SCS = 4 + 0, YS = 36;
constexpr int ITERS = 3000;

struct Smem {
    int    xq[I*XS];
    int    xsc[I*SCS + I/8 + 8];
    half2  xdm[I];
    int    y[J*YS];
};

template <int ABL>
__device__ __forceinline__ float one_pair_real(const int * v, const int * u, const uint8_t * sc, const uint8_t * m, const half2 & dm4, const half2 * ds8) {
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int sumi_d = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { sumi_d = __dp4a(ABL == 2 ? v[j] : ((v[j] >> (4*i)) & 0x0F0F0F0F), u[i*8 + j], sumi_d); }
        if (ABL == 3) { sumf_d += (float) sumi_d; continue; }
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        if (ABL != 1) { sumf_m += ds8f.y *   m[i]; }
    }
    if (ABL == 3) { return sumf_d; }
    const float2 dm4f = __half22float2(dm4);
    return ABL == 1 ? dm4f.x*sumf_d : dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// K0: the real loop, 256 threads, 2 rows x 8 columns
template <int ABL>
__device__ __forceinline__ void dot_k0(const Smem & sm, float * sum, int k00) {
    const int * yq = sm.y + 4; const half2 * yds = (const half2 *) sm.y;
    for (int k01 = 0; k01 < 32; k01 += 16) {
        const int k0 = k00 + k01;
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += 8) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < I; i0 += 32) {
                const int i = i0 + threadIdx.x;
                const uint8_t * sc = (const uint8_t *) &sm.xsc[i*(32/8) + i/8 + k0/32] + 2*(k01/16);
                sum[j0/8*(I/32) + i0/32] += one_pair_real<ABL>(&sm.xq[i*XS + k0/2], &yq[j*YS + k01], sc, sc + 8, sm.xdm[i], &yds[j*YS + k01/8]);
            }
        }
    }
}

// K1: same thread tile, unpacked weights, scales and mins of the rows held in registers across the 8 columns
template <int R, int C, int MAPPING>
__device__ __forceinline__ void dot_k1(const Smem & sm, float * sum, int k00) {
    const int * yq = sm.y + 4; const half2 * yds = (const half2 *) sm.y;
    const int lane = threadIdx.x, warp = threadIdx.y;
    for (int k01 = 0; k01 < 32; k01 += 16) {
        const int k0 = k00 + k01;
        int lo[R][8], hi[R][8]; int scv[R][2], mv[R][2]; half2 dm[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = MAPPING == 0 ? lane + 32*r : (lane % 16) + 16*r;
            const int * v = &sm.xq[i*XS + k0/2];
#pragma unroll
            for (int q = 0; q < 8; ++q) { const int a = v[q]; lo[r][q] = a & 0x0F0F0F0F; hi[r][q] = (a >> 4) & 0x0F0F0F0F; }
            const uint8_t * sc = (const uint8_t *) &sm.xsc[i*(32/8) + i/8 + k0/32] + 2*(k01/16);
            scv[r][0] = sc[0]; scv[r][1] = sc[1]; mv[r][0] = sc[8]; mv[r][1] = sc[9];
            dm[r] = sm.xdm[i];
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int j = MAPPING == 0 ? warp + 8*c : (lane / 16) + 2*warp + (C == 8 ? 8 : 16)*c;
            int u[16];
#pragma unroll
            for (int q = 0; q < 16; ++q) { u[q] = yq[j*YS + k01 + q]; }
            const float2 d0 = __half22float2(yds[j*YS + k01/8]), d1 = __half22float2(yds[j*YS + k01/8 + 1]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int q = 0; q < 8; ++q) { s0 = __dp4a(lo[r][q], u[q], s0); }
#pragma unroll
                for (int q = 0; q < 8; ++q) { s1 = __dp4a(hi[r][q], u[8 + q], s1); }
                float sumf_d = 0.0f, sumf_m = 0.0f;
                sumf_d += d0.x * (scv[r][0] * s0); sumf_m += d0.y * mv[r][0];
                sumf_d += d1.x * (scv[r][1] * s1); sumf_m += d1.y * mv[r][1];
                const float2 dm4f = __half22float2(dm[r]);
                sum[c*R + r] += dm4f.x*sumf_d - dm4f.y*sumf_m;
            }
        }
    }
}

template <int VARIANT, int NACC, int NTHY>
__global__ void __launch_bounds__(32*NTHY, 1)
bench(const int * gx, const int * gy, float * out, int iters) {
    __shared__ Smem sm;
    const int tid = threadIdx.x + 32*threadIdx.y, nt = 32*NTHY;
    for (int n = tid; n < I*XS; n += nt) { sm.xq[n] = gx[n]; }
    for (int n = tid; n < J*YS; n += nt) { sm.y[n] = gy[n]; }
    for (int n = tid; n < I*SCS + I/8 + 8; n += nt) { sm.xsc[n] = 0x01020304 + n % 7 * 0x01010101; }
    for (int n = tid; n < I; n += nt) { sm.xdm[n] = __floats2half2_rn(0.01f + 0.001f * (n % 5), 0.02f); }
    __syncthreads();
    for (int j = tid; j < J; j += nt) { for (int b = 0; b < 4; ++b) { ((half2 *) sm.y)[j*YS + b] = __floats2half2_rn(0.01f * (1 + b), 0.5f * (1 + j % 3)); } }
    __syncthreads();
    float sum[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a) { sum[a] = 0.0f; }
    for (int it = 0; it < iters; ++it) {
        if (VARIANT == 0)      { dot_k0<0>(sm, sum, 0); dot_k0<0>(sm, sum, 32); }
        else if (VARIANT == 11) { dot_k0<1>(sm, sum, 0); dot_k0<1>(sm, sum, 32); }
        else if (VARIANT == 12) { dot_k0<2>(sm, sum, 0); dot_k0<2>(sm, sum, 32); }
        else if (VARIANT == 13) { dot_k0<3>(sm, sum, 0); dot_k0<3>(sm, sum, 32); }
        else if (VARIANT == 1) { dot_k1<2, 8, 0>(sm, sum, 0); dot_k1<2, 8, 0>(sm, sum, 32); }
        else if (VARIANT == 2) { dot_k1<4, 8, 1>(sm, sum, 0); dot_k1<4, 8, 1>(sm, sum, 32); }   // 128 threads: 4 rows x 8 columns
        else                   { dot_k1<4, 4, 1>(sm, sum, 0); dot_k1<4, 4, 1>(sm, sum, 32); }   // 256 threads: 4 rows x 4 columns
        __syncthreads();
    }
    double t = 0.0;
#pragma unroll
    for (int a = 0; a < NACC; ++a) { t += (double) sum[a]; }
    out[blockIdx.x * nt + tid] = (float) t;
}

template <int VARIANT, int NACC, int NTHY>
static void run(const char * name, const int * gx, const int * gy) {
    const int blocks = 15 * 8;
    float * dout; CK(cudaMalloc(&dout, blocks * 32*NTHY * sizeof(float)));
    dim3 th(32, NTHY);
    bench<VARIANT, NACC, NTHY><<<blocks, th>>>(gx, gy, dout, 4);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    bench<VARIANT, NACC, NTHY><<<blocks, th>>>(gx, gy, dout, ITERS);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, bench<VARIANT, NACC, NTHY>);
    int bpsm = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, bench<VARIANT, NACC, NTHY>, 32*NTHY, 0);
    const double tops = (double) blocks * ITERS * I * J * 256 * 2.0 / (ms * 1e-3) / 1e12;
    std::vector<float> h(blocks * 32*NTHY); CK(cudaMemcpy(h.data(), dout, h.size()*4, cudaMemcpyDeviceToHost));
    double cs = 0; for (float v : h) cs += v;
    printf("%-38s %7.2f ms  %6.2f TOPS  regs %3d  (%d blocks/SM)  checksum %.9e\n", name, ms, tops, fa.numRegs, bpsm, cs);
    cudaFree(dout);
}

int main() {
    std::vector<int> hx(I*XS), hy(J*YS);
    srand(5);
    for (auto & v : hx) { v = rand() ^ (rand() << 8); }
    for (auto & v : hy) { v = rand() ^ (rand() << 8); }
    int * gx, * gy;
    CK(cudaMalloc(&gx, hx.size()*4)); CK(cudaMalloc(&gy, hy.size()*4));
    CK(cudaMemcpy(gx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gy, hy.data(), hy.size()*4, cudaMemcpyHostToDevice));
    printf("Q4_K inner loop, tile %d x %d, K = 256 per pass\n", I, J);
    run<0, 16, 8>("K0 real loop (2x8, 256 thr)",           gx, gy);
    run<11, 16, 8>("K0 ablation: no min term",              gx, gy);
    run<12, 16, 8>("K0 ablation: no nibble unpack",         gx, gy);
    run<13, 16, 8>("K0 ablation: sum += (float) sumi only",  gx, gy);
    run<1, 16, 8>("K1 hoisted unpack (2x8, 256 thr)",      gx, gy);
    run<2, 32, 4>("K2 hoisted, 4x8 (128 thr)",             gx, gy);
    run<3, 16, 8>("K3 hoisted, 4x4 (256 thr)",             gx, gy);
    return 0;
}
