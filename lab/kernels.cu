// Kernels for the launch-overhead lab. Compiled at run time with NVRTC (see lab/cuda_util.py).
//
// All tensors are FP16 with FP32 accumulation. The GEMV is the operation that dominates
// single-stream decode: every weight is read once per token, so it is memory-bound and
// its floor is (weight bytes) / (memory bandwidth).

#include <cuda_fp16.h>

// ---------------------------------------------------------------- launch cost probe

extern "C" __global__ void empty_kernel() {}

// ---------------------------------------------------------------- helpers

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Sum over the block; the result is valid in thread 0. blockDim.x must be a multiple of 32.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float partial[32];
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.f;
        v = warp_sum(v);
    }
    return v;
}

// ---------------------------------------------------------------- GEMV

// y[M] = W[M, K] . x[K]   (W row-major)
// One warp per output row. Each lane reads 16 bytes (8 halves) at a time, so consecutive
// lanes read consecutive 16-byte chunks of the row: fully coalesced 512-byte warp loads.
// Requires K % 8 == 0 and 16-byte-aligned W and x.
extern "C" __global__ void gemv_f16(const __half* __restrict__ W, const __half* __restrict__ x,
                                    __half* __restrict__ y, int M, int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const uint4* w = reinterpret_cast<const uint4*>(W + (size_t)row * K);
    const uint4* v = reinterpret_cast<const uint4*>(x);
    float acc = 0.f;
    for (int i = lane; i < (K >> 3); i += 32) {
        const uint4 a = w[i];
        const uint4 b = __ldg(&v[i]);
        const __half2* ah = reinterpret_cast<const __half2*>(&a);
        const __half2* bh = reinterpret_cast<const __half2*>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __half22float2(ah[j]);
            const float2 fb = __half22float2(bh[j]);
            acc = fmaf(fa.x, fb.x, acc);
            acc = fmaf(fa.y, fb.y, acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = __float2half(acc);
}

// ---------------------------------------------------------------- bandwidth probe

// Reads n 16-byte words once and writes one value per block. Used to measure the bandwidth
// this card actually delivers for a pure streaming read, as a practical ceiling for the GEMV.
extern "C" __global__ void read_sum(const uint4* __restrict__ p, unsigned long long n,
                                    float* __restrict__ out) {
    float s = 0.f;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 v = p[i];
        s += __uint_as_float((v.x ^ v.y ^ v.z ^ v.w) & 0x3f7fffffu);  // uses every byte, stays finite
    }
    s = block_sum(s);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// ---------------------------------------------------------------- small per-layer kernels

// out = x * rsqrt(mean(x^2) + eps) * w      one block per vector
extern "C" __global__ void rmsnorm_f16(const __half* __restrict__ x, const __half* __restrict__ w,
                                       __half* __restrict__ out, int n, float eps) {
    __shared__ float scale;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(x[i]);
        s = fmaf(v, v, s);
    }
    s = block_sum(s);
    if (threadIdx.x == 0) scale = rsqrtf(s / n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = __float2half(__half2float(x[i]) * scale * __half2float(w[i]));
}

// out = silu(gate) * up
extern "C" __global__ void silu_mul_f16(const __half* __restrict__ gate, const __half* __restrict__ up,
                                        __half* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = __half2float(gate[i]);
        out[i] = __float2half(g / (1.f + __expf(-g)) * __half2float(up[i]));
    }
}

// x += y   (residual connection)
extern "C" __global__ void add_inplace_f16(__half* __restrict__ x, const __half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}
