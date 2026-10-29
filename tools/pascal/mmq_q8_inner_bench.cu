// Inner loop of the dense Q8_0 MMQ kernel on Pascal (dp4a), in shared memory only: which thread tile / load pattern reaches how much of the dp4a peak?
// The tile is the one of the real kernel: weights I x 256 int8 (64 ints per row, row stride 65 ints), activations J columns x (4 scales + 32 ints), K = 256 per iteration
// (two passes of 32 ints), the scales of a 32-value block applied to the integer sum of 8 dp4a as in vec_dot_q8_0_q8_1_impl. All variants must give the same bits.
//   nvcc -O3 -arch=sm_61 -std=c++17 tools/pascal/mmq_q8_inner_bench.cu -o build/pascal/mmq_q8_inner_bench && CUDA_VISIBLE_DEVICES=0 build/pascal/mmq_q8_inner_bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int I = 128, J = 64;
constexpr int XS = 65;                 // row stride of the weight tile in ints (the real kernel's)
constexpr int XS4 = 68;                // padded to a multiple of 4 for 128-bit loads
constexpr int YS = 36;                 // 4 floats of scales + 32 ints per column
constexpr int ITERS = 3000;

// exact int -> float for |s| < 2^22 with a full-rate integer add and a float add instead of the quarter-rate I2F
template <bool MAGIC> __device__ __forceinline__ float i2f(int s) {
    if (MAGIC) { return __int_as_float(s + 0x4B400000) - 12582912.0f; }
    return (float) s;
}

struct Smem {
    int   xq[I*XS4];
    float xd[I*8 + I/4 + 8];
    int   y[J*YS];
};

// ---------------------------------------------------------------- V0: the real kernel's loop, 256 threads, thread tile 4 rows x 8 columns, nothing hoisted by hand
template <int STRIDE, bool MAGIC>
__device__ __forceinline__ void dot_v0(const int * xq, const float * xd, const int * y, float * sum, int k00) {
    const int * yq = y + 4; const float * yd = (const float *) y;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += 8) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < I; i0 += 32) {
                const int i = i0 + threadIdx.x;
                int s = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) { s = __dp4a(xq[i*STRIDE + k0 + v], yq[j*YS + k0 % 32 + v], s); }
                sum[j0/8*(I/32) + i0/32] += xd[i*8 + i/4 + k0/8] * yd[j*YS + (k0/8) % 4] * i2f<MAGIC>(s);
            }
        }
    }
}

// ---------------------------------------------------------------- V1: 256 threads, 4 rows x 8 columns, the 4 rows' weights held in registers per 32-value block
template <int STRIDE, bool VECY, bool VECX, bool MAGIC>
__device__ __forceinline__ void dot_v1(const int * xq, const float * xd, const int * y, float * sum, int k00) {
    const int * yq = y + 4; const float * yd = (const float *) y;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
        int xr[4][8]; float xdr[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = threadIdx.x + 32*r;
            if (VECX) {
                const int4 a = *(const int4 *) &xq[i*STRIDE + k0], b = *(const int4 *) &xq[i*STRIDE + k0 + 4];
                xr[r][0] = a.x; xr[r][1] = a.y; xr[r][2] = a.z; xr[r][3] = a.w; xr[r][4] = b.x; xr[r][5] = b.y; xr[r][6] = b.z; xr[r][7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { xr[r][v] = xq[i*STRIDE + k0 + v]; }
            }
            xdr[r] = xd[i*8 + i/4 + k0/8];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int j = threadIdx.y + 8*c;
            int yv[8];
            if (VECY) {
                const int4 a = *(const int4 *) &yq[j*YS + k0 % 32], b = *(const int4 *) &yq[j*YS + k0 % 32 + 4];
                yv[0] = a.x; yv[1] = a.y; yv[2] = a.z; yv[3] = a.w; yv[4] = b.x; yv[5] = b.y; yv[6] = b.z; yv[7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { yv[v] = yq[j*YS + k0 % 32 + v]; }
            }
            const float ydv = yd[j*YS + (k0/8) % 4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                int s = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) { s = __dp4a(xr[r][v], yv[v], s); }
                sum[c*4 + r] += xdr[r] * ydv * i2f<MAGIC>(s);
            }
        }
    }
}

template <int STRIDE, bool VECY, bool VECX, bool MAGIC>
__device__ __forceinline__ void dot_v5(const int * xq, const float * xd, const int * y, float * sum, int k00) {
    int isum[32];
#pragma unroll
    for (int q = 0; q < 32; ++q) { isum[q] = 0; }
    const int * yq = y + 4; const float * yd = (const float *) y;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
        int xr[4][8]; float xdr[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = threadIdx.x + 32*r;
            if (VECX) {
                const int4 a = *(const int4 *) &xq[i*STRIDE + k0], b = *(const int4 *) &xq[i*STRIDE + k0 + 4];
                xr[r][0] = a.x; xr[r][1] = a.y; xr[r][2] = a.z; xr[r][3] = a.w; xr[r][4] = b.x; xr[r][5] = b.y; xr[r][6] = b.z; xr[r][7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { xr[r][v] = xq[i*STRIDE + k0 + v]; }
            }
            xdr[r] = xd[i*8 + i/4 + k0/8];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int j = threadIdx.y + 8*c;
            int yv[8];
            if (VECY) {
                const int4 a = *(const int4 *) &yq[j*YS + k0 % 32], b = *(const int4 *) &yq[j*YS + k0 % 32 + 4];
                yv[0] = a.x; yv[1] = a.y; yv[2] = a.z; yv[3] = a.w; yv[4] = b.x; yv[5] = b.y; yv[6] = b.z; yv[7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { yv[v] = yq[j*YS + k0 % 32 + v]; }
            }
            const float ydv = yd[j*YS + (k0/8) % 4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                int s = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) { s = __dp4a(xr[r][v], yv[v], s); }
                isum[c*4 + r] += s;
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 32; ++q) { sum[q] += (float) isum[q]; }
}

template <int STRIDE, bool VECY, bool VECX, bool MAGIC>
__device__ __forceinline__ void dot_v6(const int * xq, const float * xd, const int * y, float * sum, int k00) {
    const int * yq = y + 4; const float * yd = (const float *) y;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
        int xr[4][8]; float xdr[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = threadIdx.x + 32*r;
            if (VECX) {
                const int4 a = *(const int4 *) &xq[i*STRIDE + k0], b = *(const int4 *) &xq[i*STRIDE + k0 + 4];
                xr[r][0] = a.x; xr[r][1] = a.y; xr[r][2] = a.z; xr[r][3] = a.w; xr[r][4] = b.x; xr[r][5] = b.y; xr[r][6] = b.z; xr[r][7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { xr[r][v] = xq[i*STRIDE + k0 + v]; }
            }
            xdr[r] = xd[i*8 + i/4 + k0/8];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int j = threadIdx.y + 8*c;
            int yv[8];
            if (VECY) {
                const int4 a = *(const int4 *) &yq[j*YS + k0 % 32], b = *(const int4 *) &yq[j*YS + k0 % 32 + 4];
                yv[0] = a.x; yv[1] = a.y; yv[2] = a.z; yv[3] = a.w; yv[4] = b.x; yv[5] = b.y; yv[6] = b.z; yv[7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { yv[v] = yq[j*YS + k0 % 32 + v]; }
            }
            const float ydv = yd[j*YS + (k0/8) % 4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                int s = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) { s = __dp4a(xr[r][v], yv[v], s); }
                sum[c*4 + r] += (float) s;
            }
        }
    }
}

// ---------------------------------------------------------------- V3: 256 threads, 8 rows x 4 columns: lane%16 + 16r rows, lane/16 + 2*warp + 16c columns
template <int STRIDE, bool VECY, bool VECX, bool MAGIC>
__device__ __forceinline__ void dot_v3(const int * xq, const float * xd, const int * y, float * sum, int k00) {
    const int * yq = y + 4; const float * yd = (const float *) y;
    const int lane = threadIdx.x, warp = threadIdx.y;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
        int xr[8][8]; float xdr[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i = (lane % 16) + 16*r;
            if (VECX) {
                const int4 a = *(const int4 *) &xq[i*STRIDE + k0], b = *(const int4 *) &xq[i*STRIDE + k0 + 4];
                xr[r][0] = a.x; xr[r][1] = a.y; xr[r][2] = a.z; xr[r][3] = a.w; xr[r][4] = b.x; xr[r][5] = b.y; xr[r][6] = b.z; xr[r][7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { xr[r][v] = xq[i*STRIDE + k0 + v]; }
            }
            xdr[r] = xd[i*8 + i/4 + k0/8];
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int j = (lane / 16) + 2*warp + 16*c;
            int yv[8];
            if (VECY) {
                const int4 a = *(const int4 *) &yq[j*YS + k0 % 32], b = *(const int4 *) &yq[j*YS + k0 % 32 + 4];
                yv[0] = a.x; yv[1] = a.y; yv[2] = a.z; yv[3] = a.w; yv[4] = b.x; yv[5] = b.y; yv[6] = b.z; yv[7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { yv[v] = yq[j*YS + k0 % 32 + v]; }
            }
            const float ydv = yd[j*YS + (k0/8) % 4];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int s = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) { s = __dp4a(xr[r][v], yv[v], s); }
                sum[c*8 + r] += xdr[r] * ydv * i2f<MAGIC>(s);
            }
        }
    }
}

// ---------------------------------------------------------------- V4: 128 threads (4 warps), 8 rows x 8 columns: lane%16 + 16r rows, lane/16 + 2*warp + 8c columns
template <int STRIDE, bool VECY, bool VECX, bool MAGIC>
__device__ __forceinline__ void dot_v4(const int * xq, const float * xd, const int * y, float * sum, int k00) {
    const int * yq = y + 4; const float * yd = (const float *) y;
    const int lane = threadIdx.x, warp = threadIdx.y;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
        int xr[8][8]; float xdr[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i = (lane % 16) + 16*r;
            if (VECX) {
                const int4 a = *(const int4 *) &xq[i*STRIDE + k0], b = *(const int4 *) &xq[i*STRIDE + k0 + 4];
                xr[r][0] = a.x; xr[r][1] = a.y; xr[r][2] = a.z; xr[r][3] = a.w; xr[r][4] = b.x; xr[r][5] = b.y; xr[r][6] = b.z; xr[r][7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { xr[r][v] = xq[i*STRIDE + k0 + v]; }
            }
            xdr[r] = xd[i*8 + i/4 + k0/8];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int j = (lane / 16) + 2*warp + 8*c;
            int yv[8];
            if (VECY) {
                const int4 a = *(const int4 *) &yq[j*YS + k0 % 32], b = *(const int4 *) &yq[j*YS + k0 % 32 + 4];
                yv[0] = a.x; yv[1] = a.y; yv[2] = a.z; yv[3] = a.w; yv[4] = b.x; yv[5] = b.y; yv[6] = b.z; yv[7] = b.w;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) { yv[v] = yq[j*YS + k0 % 32 + v]; }
            }
            const float ydv = yd[j*YS + (k0/8) % 4];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int s = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) { s = __dp4a(xr[r][v], yv[v], s); }
                sum[c*8 + r] += xdr[r] * ydv * i2f<MAGIC>(s);
            }
        }
    }
}

template <int VARIANT, int NACC, int STRIDE, bool VECY, bool VECX, int NTHY, bool MAGIC>
__global__ void __launch_bounds__(32*NTHY, 1)
bench(const int * gx, const float * gd, const int * gy, float * out, int iters) {
    __shared__ Smem sm;
    const int tid = threadIdx.x + 32*threadIdx.y, nt = 32*NTHY;
    for (int n = tid; n < I*STRIDE; n += nt) { sm.xq[n] = gx[n % (I*XS)]; }
    for (int n = tid; n < I*8 + I/4 + 8; n += nt) { sm.xd[n] = gd[n % 1024]; }
    for (int n = tid; n < J*YS; n += nt) { sm.y[n] = gy[n % (J*YS)]; }
    __syncthreads();
    // scales of the activations are floats in the first 4 words of each column
    for (int j = tid; j < J; j += nt) { for (int b = 0; b < 4; ++b) { ((float *) sm.y)[j*YS + b] = 0.01f*(1 + b + j % 3); } }
    __syncthreads();
    float sum[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a) { sum[a] = 0.0f; }
    for (int it = 0; it < iters; ++it) {
        if (VARIANT == 0)      { dot_v0<STRIDE, MAGIC>(sm.xq, sm.xd, sm.y, sum, 0); dot_v0<STRIDE, MAGIC>(sm.xq, sm.xd, sm.y, sum, 32); }
        else if (VARIANT == 1) { dot_v1<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 0); dot_v1<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 32); }
        else if (VARIANT == 5) { dot_v5<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 0); dot_v5<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 32); }
        else if (VARIANT == 6) { dot_v6<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 0); dot_v6<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 32); }
        else if (VARIANT == 3) { dot_v3<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 0); dot_v3<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 32); }
        else                   { dot_v4<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 0); dot_v4<STRIDE, VECY, VECX, MAGIC>(sm.xq, sm.xd, sm.y, sum, 32); }
        __syncthreads();
    }
    double t = 0.0;
#pragma unroll
    for (int a = 0; a < NACC; ++a) { t += (double) sum[a]; }
    out[(blockIdx.x * 32*NTHY) + tid] = (float) t;
}

template <int VARIANT, int NACC, int STRIDE, bool VECY, bool VECX, int NTHY, bool MAGIC>
static void run(const char * name, const int * gx, const float * gd, const int * gy, double * base_tops, float * ref_out, int * n_ref) {
    const int blocks = 15 * 8;
    float * dout; CK(cudaMalloc(&dout, blocks * 32*NTHY * sizeof(float)));
    dim3 th(32, NTHY);
    bench<VARIANT, NACC, STRIDE, VECY, VECX, NTHY, MAGIC><<<blocks, th>>>(gx, gd, gy, dout, 4);   // warm-up
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    bench<VARIANT, NACC, STRIDE, VECY, VECX, NTHY, MAGIC><<<blocks, th>>>(gx, gd, gy, dout, ITERS);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, bench<VARIANT, NACC, STRIDE, VECY, VECX, NTHY, MAGIC>);
    int bpsm = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, bench<VARIANT, NACC, STRIDE, VECY, VECX, NTHY, MAGIC>, 32*NTHY, 0);
    const double macs = (double) blocks * ITERS * I * J * 256;   // per iteration: K = 256
    const double tops = macs * 2.0 / (ms * 1e-3) / 1e12;
    std::vector<float> h(blocks * 32*NTHY); CK(cudaMemcpy(h.data(), dout, h.size()*4, cudaMemcpyDeviceToHost));
    double cs = 0; for (float v : h) cs += v;
    printf("%-34s %7.2f ms  %6.2f TOPS  regs %3d  (%d blocks/SM)  checksum %.6e\n", name, ms, tops, fa.numRegs, bpsm, cs);
    if (*base_tops == 0) { *base_tops = tops; }
    cudaFree(dout);
}

int main() {
    std::vector<int> hx(I*XS), hy(J*YS); std::vector<float> hd(1024);
    srand(3);
    for (auto & v : hx) { v = rand() ^ (rand() << 8); }
    for (auto & v : hy) { v = rand() ^ (rand() << 8); }
    for (auto & v : hd) { v = 0.001f * (1 + rand() % 50); }
    int * gx, * gy; float * gd;
    CK(cudaMalloc(&gx, hx.size()*4)); CK(cudaMalloc(&gy, hy.size()*4)); CK(cudaMalloc(&gd, hd.size()*4));
    CK(cudaMemcpy(gx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gy, hy.data(), hy.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gd, hd.data(), hd.size()*4, cudaMemcpyHostToDevice));
    double base = 0; float ref[1]; int nref = 0;
    printf("dp4a peak about 29 TOPS (1,920 lanes x 4 MAC x 2 x 1,898 MHz); one pass = %d x %d tile, K = 256\n", I, J);
    run<0, 32, XS,  false, false, 8, false>("V0 real loop (4x8, 256 thr)",        gx, gd, gy, &base, ref, &nref);
    run<0, 32, XS,  false, false, 8, true >("V0 + magic int->float",               gx, gd, gy, &base, ref, &nref);
    run<1, 32, XS,  true,  false, 8, false>("V1 4x8 regs, y as int4",              gx, gd, gy, &base, ref, &nref);
    run<1, 32, XS,  true,  false, 8, true >("V1 4x8 regs, y as int4 + magic",      gx, gd, gy, &base, ref, &nref);
    run<6, 32, XS,  true,  false, 8, false>("V1 ablation: no scales (sum += (float) s)", gx, gd, gy, &base, ref, &nref);
    run<5, 32, XS,  true,  false, 8, false>("V1 ablation: int sums only",        gx, gd, gy, &base, ref, &nref);
    return 0;
}
