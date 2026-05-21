// Wave 17 W2c — row-wise softmax kernel for cuBLAS-3-kernel MLA attention.
//
// Identical structure to cublas-attn-gqa/softmax.cu — works on any
// (num_rows, seq) f32 score matrix and writes f16 probs. MLA changes only
// the *number* of rows (B * n_h * S, no GQA broadcast) and seq stays = S.
//
// One CUDA block per row. 128 threads/block reduce via shared memory:
// max -> sum-of-exp -> normalize -> f16 store.
//
// The attention scale (1/sqrt(qk_head_dim_TRUE)) is fused. Per ADR-0005,
// the scale uses the TRUE qk dim (192), NOT the padded qk_eff (256), so
// score magnitudes are bit-identical between native and padded variants.
// Caller passes the correct scale; this kernel is variant-agnostic.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <int BLOCK_THREADS>
__global__ void mla_row_softmax_scale_kernel(
    const float* __restrict__ scores,  // (num_rows, seq)
    __half* __restrict__ probs,        // (num_rows, seq)
    int seq,
    float scale)                       // 1/sqrt(qk_head_dim_TRUE)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_in  = scores + (size_t)row * seq;
    __half*      row_out = probs  + (size_t)row * seq;

    __shared__ float s_red[BLOCK_THREADS];

    // Pass 1: max over the row, with scale folded in
    float local_max = -INFINITY;
    for (int i = tid; i < seq; i += BLOCK_THREADS) {
        float v = row_in[i] * scale;
        if (v > local_max) local_max = v;
    }
    s_red[tid] = local_max;
    __syncthreads();

    for (int off = BLOCK_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float other = s_red[tid + off];
            if (other > s_red[tid]) s_red[tid] = other;
        }
        __syncthreads();
    }
    float row_max = s_red[0];
    __syncthreads();

    // Pass 2: sum of exp(x*scale - max)
    float local_sum = 0.0f;
    for (int i = tid; i < seq; i += BLOCK_THREADS) {
        float v = row_in[i] * scale - row_max;
        local_sum += __expf(v);
    }
    s_red[tid] = local_sum;
    __syncthreads();

    for (int off = BLOCK_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) s_red[tid] += s_red[tid + off];
        __syncthreads();
    }
    float row_sum = s_red[0];
    float inv_sum = 1.0f / row_sum;
    __syncthreads();

    // Pass 3: normalize, write f16
    for (int i = tid; i < seq; i += BLOCK_THREADS) {
        float v = row_in[i] * scale - row_max;
        float e = __expf(v) * inv_sum;
        row_out[i] = __float2half(e);
    }
}

extern "C" void launch_mla_row_softmax_scale(
    const float* d_scores,
    __half*      d_probs,
    int          num_rows,
    int          seq,
    float        scale,
    cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 128;
    dim3 grid(num_rows);
    dim3 block(BLOCK_THREADS);
    mla_row_softmax_scale_kernel<BLOCK_THREADS>
        <<<grid, block, 0, stream>>>(d_scores, d_probs, seq, scale);
}
