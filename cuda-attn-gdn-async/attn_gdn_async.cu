// Wave 22.9 — cuda-attn-gdn-async — single-timestep GDN decode using a
// software pipeline built from cuda::pipeline + cuda::memcpy_async.
//
// Background. Wave 22.8 investigation (docs/research/wave17-w1c-tma-vs-ldg128-
// investigation.md) found that cuTile's 610 GB/s vs nvcc-W1c's 417.7 GB/s
// gap on the GDN-decode bench shape (B=1 H=16 d_k=d_v=256 BLOCK_V=64) is
// NOT a TMA/UTMALDG advantage — neither kernel uses TMA. The gap is a
// warp-specialised producer/consumer software pipeline backed by Blackwell
// async-transaction barriers (SYNCS.PHASECHK.TRANS64.TRYWAIT, ~100 KiB
// static smem, REG=255). nvcc's W1c kernel issues LDG.E.128 directly on
// the SM's LSU inside the math warp with no producer/consumer split, so
// load latency is exposed in the FFMA recurrence.
//
// W22.9 hypothesis. Decoupling the `S_in` row load from the FFMA chain
// using a multi-stage ring buffer fed by cuda::memcpy_async (which lowers
// to cp.async / LDGSTS on sm_120) should let the LSU run ahead of the
// math, hiding load latency the way cuTile's barrier pipeline does.
//
// Design.
//   * TPB=16, BLOCK_V=64  (identical thread-stripe layout to W1c, so the
//     async pipeline is the only changed variable).
//   * N_STAGES=4 ring buffer in smem holding S_in row-tiles (alpha-scaled
//     post-load). Each thread owns 4 contiguous d_v columns => 16 B per
//     row per thread => 256 B per ring slot at TPB=16, BLOCK_V=64.
//     Total ring smem at D_K=256: 4 stages * 64 (BLOCK_V) * 4 B = 1 KiB —
//     small, fits trivially.
//   * cuda::pipeline<cuda::thread_scope_thread> per thread. Pass-1
//     prologue: issue memcpy_async for the first N_STAGES rows. Steady
//     state: for k in [0..D_K), wait_prior<N_STAGES-1> on stage k,
//     consume that stage (alpha-scale, FFMA into u_acc, store to
//     smem_S[k]), then issue memcpy_async for stage k+N_STAGES.
//     Drain at the end.
//   * Pass 2 (S_out write + o accumulate) reads smem_S[k]·VLANES (already
//     populated by pass 1) — same as W1c. We could pipeline the gmem
//     STORE too, but cuda::memcpy_async is gmem→smem only on this arch;
//     STG.E.128 stays.
//
// Per-block shared memory:
//   smem_q[D_K]                                   (f32)
//   smem_k[D_K]                                   (f32)
//   smem_S[D_K * VLANES]                          (float4) — pass-1 cache
//   smem_ring[N_STAGES * VLANES]                  (float4) — async ring buffer
//   For D_K=256, BLOCK_V=64: 256·4 + 256·4 + 256·16·16 + 4·16·16 =
//     1 KiB + 1 KiB + 64 KiB + 1 KiB = 67 KiB. Within sm_120's 100 KiB
//     dynamic-smem budget after opt-in.
//
// Build:
//   make attn_gdn_async    (uses Makefile in this folder)
//
// Acceptance (per W22.9 task):
//   * compiles
//   * correctness vs PyTorch GDN-naive ≤ 1e-3 (output) / 5e-3 (state)
//   * SASS shows LDGSTS (cp.async) > 0 OR mbarrier-class instructions
//
// Numerics. The async path multiplies S_in by alpha BEFORE staging the
// ring slot — so pass 2's smem_S read is byte-identical to W1c. The only
// numerical change vs W1c is FFMA-ordering inside the u accumulation
// (we still iterate k in order, so ordering is identical). Therefore we
// expect bit-identical or near-identical results to W1c.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================================
// .npy loader (NPY1.0 / NPY2.0; little-endian only). Cloned from cuda-attn-gdn.
// ============================================================================
struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;   // "<f2" or "<f4"
    std::vector<uint8_t> data;
    size_t elem_size = 0;
};

static bool read_npy(const std::string& path, Npy& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    uint8_t magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not an NPY file\n", path.c_str()); fclose(f); return false;
    }
    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
    uint32_t header_len;
    if (ver[0] == 1) {
        uint16_t hl; if (fread(&hl, 1, 2, f) != 2) { fclose(f); return false; }
        header_len = hl;
    } else {
        if (fread(&header_len, 1, 4, f) != 4) { fclose(f); return false; }
    }
    std::string hdr(header_len, ' ');
    if (fread(&hdr[0], 1, header_len, f) != header_len) { fclose(f); return false; }

    auto find_after = [&](const std::string& key) -> size_t {
        size_t p = hdr.find(key);
        return p == std::string::npos ? std::string::npos : p + key.size();
    };
    size_t p = find_after("'descr':");
    if (p == std::string::npos) { fprintf(stderr, "no descr\n"); fclose(f); return false; }
    while (p < hdr.size() && hdr[p] != '\'') ++p; ++p;
    size_t q = p;
    while (q < hdr.size() && hdr[q] != '\'') ++q;
    out.dtype = hdr.substr(p, q - p);
    if (out.dtype == "<f2") out.elem_size = 2;
    else if (out.dtype == "<f4") out.elem_size = 4;
    else { fprintf(stderr, "unsupported dtype %s\n", out.dtype.c_str()); fclose(f); return false; }

    p = find_after("'shape':");
    if (p == std::string::npos) { fprintf(stderr, "no shape\n"); fclose(f); return false; }
    while (p < hdr.size() && hdr[p] != '(') ++p; ++p;
    q = p;
    while (q < hdr.size() && hdr[q] != ')') ++q;
    std::string shape_str = hdr.substr(p, q - p);
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && !isdigit(shape_str[pos])) ++pos;
        if (pos >= shape_str.size()) break;
        int64_t v = 0;
        while (pos < shape_str.size() && isdigit(shape_str[pos])) { v = v * 10 + (shape_str[pos] - '0'); ++pos; }
        out.shape.push_back(v);
    }

    size_t nelem = 1;
    for (auto d : out.shape) nelem *= (size_t)d;
    out.data.resize(nelem * out.elem_size);
    if (fread(out.data.data(), 1, out.data.size(), f) != out.data.size()) {
        fprintf(stderr, "%s: short read\n", path.c_str()); fclose(f); return false;
    }
    fclose(f);
    return true;
}

// ============================================================================
// GDN-decode kernel — async-pipelined Pass 1 (S_in row stream).
//
// Template params:
//   D_K       — key dim (state rows). Compile-time so loops fully unroll.
//   BLOCK_V   — d_v-tile width per block (multiple of 4; we vectorize as float4).
//   N_STAGES  — number of in-flight cuda::memcpy_async slots in the ring.
//
// Runtime parallelism (matches W1c):
//   gridDim  = (B * H, D_V / BLOCK_V)
//   blockDim = (BLOCK_V / 4)   each thread owns 4 d_v columns (one float4 stripe)
// ============================================================================
template <int D_K, int BLOCK_V, int N_STAGES>
__global__ void gdn_decode_async_kernel(
    const __half* __restrict__ Q,       // (B*H, D_K)
    const __half* __restrict__ K,       // (B*H, D_K)
    const __half* __restrict__ V,       // (B*H, D_V)
    const __half* __restrict__ Alpha,   // (B*H,)
    const __half* __restrict__ Beta,    // (B*H,)
    const float*  __restrict__ S_in,    // (B*H, D_K, D_V)
    float*        __restrict__ S_out,   // (B*H, D_K, D_V)
    __half*       __restrict__ O,       // (B*H, D_V)
    int B_H, int D_V)
{
    static_assert(BLOCK_V % 4 == 0, "BLOCK_V must be multiple of 4 for float4 lanes");
    static_assert(N_STAGES >= 2,    "N_STAGES must be >= 2 to overlap load with consume");
    constexpr int VLANES = BLOCK_V / 4;     // # float4 stripes per block = active threads
    constexpr int TPB    = VLANES;

    const int bh  = blockIdx.x;             // 0..B*H-1
    const int bv  = blockIdx.y;             // 0..D_V/BLOCK_V-1
    const int tid = threadIdx.x;

    const int col0 = bv * BLOCK_V + tid * 4;  // first d_v column for this thread

    // ── Shared-memory layout (dynamic). ──
    //   smem_q  [D_K]                  — q upcast (float)
    //   smem_k  [D_K]                  — k upcast (float)
    //   smem_S  [D_K  * VLANES]        — alpha-scaled state tile (float4),
    //                                    populated in Pass 1, consumed in Pass 2
    //   smem_ring [N_STAGES * VLANES]  — Pass-1 async ring buffer (float4)
    extern __shared__ unsigned char smem_raw[];
    float*  smem_q    = reinterpret_cast<float*>(smem_raw);                              // [D_K]
    float*  smem_k    = smem_q + D_K;                                                    // [D_K]
    float4* smem_S    = reinterpret_cast<float4*>(smem_k + D_K);                         // [D_K * VLANES]
    float4* smem_ring = smem_S + (size_t)D_K * VLANES;                                   // [N_STAGES * VLANES]

    // ── Cooperative load of q, k (f16 → f32) into smem. D_K small (≤256). ──
    const __half* Qbh = Q + (size_t)bh * D_K;
    const __half* Kbh = K + (size_t)bh * D_K;
    #pragma unroll 4
    for (int kk = tid; kk < D_K; kk += TPB) {
        smem_q[kk] = __half2float(Qbh[kk]);
        smem_k[kk] = __half2float(Kbh[kk]);
    }

    __shared__ float s_alpha;
    __shared__ float s_beta;
    if (tid == 0) {
        s_alpha = __half2float(Alpha[bh]);
        s_beta  = __half2float(Beta[bh]);
    }

    // Load v stripe (4 halves → float4) — small, sync path is fine.
    float4 v_vec;
    {
        const __half* Vbh = V + (size_t)bh * D_V;
        const float2* Vptr = reinterpret_cast<const float2*>(Vbh + col0);
        float2 v_packed = *Vptr;
        const __half* hv = reinterpret_cast<const __half*>(&v_packed);
        v_vec.x = __half2float(hv[0]);
        v_vec.y = __half2float(hv[1]);
        v_vec.z = __half2float(hv[2]);
        v_vec.w = __half2float(hv[3]);
    }

    __syncthreads();
    const float alpha = s_alpha;
    const float beta  = s_beta;

    // ── Pass 1: streamed S_in[k, col0:col0+4] via async pipeline ──
    //
    // Per-thread base for the state row at d_k=k:
    //   Sbh + k*D_V + col0   (float*, 4 floats = 16 bytes)
    const size_t bh_state_off = (size_t)bh * D_K * D_V;
    const float* Sbh = S_in + bh_state_off;

    // cuda::pipeline at thread scope. Each thread runs its OWN producer/
    // consumer software pipeline against the device LSU. memcpy_async
    // lowers to cp.async on sm_80+ (sm_120 inherits) when source is gmem
    // and dest is smem, both 16-byte aligned, copy size in {4,8,16}.
    auto pipe = cuda::make_pipeline();

    constexpr int K_LIMIT = D_K;
    const int prologue = (N_STAGES < K_LIMIT) ? N_STAGES : K_LIMIT;

    // ── Prologue: launch first `prologue` async copies, no waits. ──
    #pragma unroll
    for (int s = 0; s < N_STAGES; ++s) {
        if (s < K_LIMIT) {
            pipe.producer_acquire();
            // Source: gmem float4 at row s, this thread's stripe.
            const float4* gptr =
                reinterpret_cast<const float4*>(Sbh + (size_t)s * D_V + col0);
            // Dest: smem ring slot s, this thread's lane.
            float4* sptr = smem_ring + (size_t)s * VLANES + tid;
            // 16-byte async copy. cuda::aligned_size_t<16> hints the
            // compiler we're 16-byte-aligned so it picks LDGSTS.E.128
            // (the cp.async.ca.shared.global path on sm_120).
            cuda::memcpy_async(sptr, gptr, cuda::aligned_size_t<16>(sizeof(float4)), pipe);
            pipe.producer_commit();
        }
    }

    // ── Steady state: for each k, wait for stage k, consume, prefetch k+N_STAGES. ──
    float4 u_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 4
    for (int k = 0; k < K_LIMIT; ++k) {
        // Wait for stage k to be ready. wait_prior<N-1> = "at most N-1 in
        // flight" = "stage at the head of the queue is done".
        pipe.consumer_wait();

        const int slot = k % N_STAGES;
        // Read alpha-scaled S_in row from ring. Note: we apply alpha here
        // (not in the producer) to keep the producer step a pure copy
        // (memcpy_async cannot do the multiply; we'd need a 2-step path
        // otherwise). The order matches W1c byte-for-byte.
        float4 s = smem_ring[(size_t)slot * VLANES + tid];
        s.x *= alpha;
        s.y *= alpha;
        s.z *= alpha;
        s.w *= alpha;

        // Cache the alpha-scaled tile for Pass 2.
        smem_S[(size_t)k * VLANES + tid] = s;

        // u_acc += k_k · S_scaled[k]
        const float kk = smem_k[k];
        u_acc.x += kk * s.x;
        u_acc.y += kk * s.y;
        u_acc.z += kk * s.z;
        u_acc.w += kk * s.w;

        // Release this slot for the producer to refill.
        pipe.consumer_release();

        // Prefetch k + N_STAGES into the freed slot.
        const int next_k = k + N_STAGES;
        if (next_k < K_LIMIT) {
            pipe.producer_acquire();
            const float4* gptr =
                reinterpret_cast<const float4*>(Sbh + (size_t)next_k * D_V + col0);
            float4* sptr = smem_ring + (size_t)slot * VLANES + tid;
            cuda::memcpy_async(sptr, gptr,
                               cuda::aligned_size_t<16>(sizeof(float4)), pipe);
            pipe.producer_commit();
        }
    }

    // ── Residual r = v - u (per-thread; no cross-thread reduction). ──
    float4 r;
    r.x = v_vec.x - u_acc.x;
    r.y = v_vec.y - u_acc.y;
    r.z = v_vec.z - u_acc.z;
    r.w = v_vec.w - u_acc.w;

    // Pass 1 wrote smem_S[*][tid]; ensure visibility before Pass 2 reads.
    // cuda::pipeline at thread scope is a per-thread mechanism; smem_S is
    // only read by THIS thread in Pass 2 (each lane owns its stripe), so
    // strictly we don't need a __syncthreads() for smem_S. We emit one
    // anyway to keep the kernel safe under future TPB changes — the cost
    // is negligible (one BAR.SYNC).
    __syncthreads();

    // ── Pass 2: S_out = S_scaled + beta · k ⊗ r,  o += q[k] · S_out_row ──
    float* Sout_bh = S_out + bh_state_off;
    float4 o_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        float4 s = smem_S[(size_t)k * VLANES + tid];   // S_scaled[k, my_4_cols]
        const float kk = smem_k[k];
        const float qk = smem_q[k];
        const float bk = beta * kk;

        s.x = s.x + bk * r.x;
        s.y = s.y + bk * r.y;
        s.z = s.z + bk * r.z;
        s.w = s.w + bk * r.w;

        // Vectorized 128-bit store of S_out row.
        float4* sout_ptr =
            reinterpret_cast<float4*>(Sout_bh + (size_t)k * D_V + col0);
        *sout_ptr = s;                               // STG.E.128

        o_acc.x += qk * s.x;
        o_acc.y += qk * s.y;
        o_acc.z += qk * s.z;
        o_acc.w += qk * s.w;
    }

    // ── Store o (f16). 4 halves contiguously → 8-byte store. ──
    __half* Obh = O + (size_t)bh * D_V;
    __half halves[4];
    halves[0] = __float2half(o_acc.x);
    halves[1] = __float2half(o_acc.y);
    halves[2] = __float2half(o_acc.z);
    halves[3] = __float2half(o_acc.w);
    float2 packed;
    memcpy(&packed, halves, sizeof(packed));
    *reinterpret_cast<float2*>(Obh + col0) = packed;
}

// ============================================================================
// Host launcher
// ============================================================================
struct GDNShape {
    const char* name;
    int batch;
    int n_heads;
    int d_k;
    int d_v;
};

template <int D_K, int BLOCK_V, int N_STAGES>
static void launch_gdn_async(const GDNShape& sh,
                             const __half* dQ, const __half* dK, const __half* dV,
                             const __half* dA, const __half* dB,
                             const float*  dS_in, float* dS_out, __half* dO,
                             cudaStream_t stream)
{
    const int B_H = sh.batch * sh.n_heads;
    const int VLANES = BLOCK_V / 4;
    const int TPB = VLANES;
    dim3 grid(B_H, sh.d_v / BLOCK_V);
    dim3 block(TPB);

    // smem: q, k (f32, D_K each) + S_scaled tile (D_K * VLANES * 16 B) +
    //       ring buffer (N_STAGES * VLANES * 16 B).
    size_t smem_bytes =
        2 * D_K * sizeof(float) +
        (size_t)D_K * VLANES * sizeof(float4) +
        (size_t)N_STAGES * VLANES * sizeof(float4);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)gdn_decode_async_kernel<D_K, BLOCK_V, N_STAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
    }

    gdn_decode_async_kernel<D_K, BLOCK_V, N_STAGES>
        <<<grid, block, smem_bytes, stream>>>(
            dQ, dK, dV, dA, dB, dS_in, dS_out, dO, B_H, sh.d_v);
}

// Dispatch (D_K, BLOCK_V) at runtime to the right template instantiation.
// N_STAGES=4 is the default — small enough to fit smem, large enough to
// overlap two LSU latencies on Blackwell.
static void launch_gdn_dispatch(const GDNShape& sh,
                                const __half* dQ, const __half* dK, const __half* dV,
                                const __half* dA, const __half* dB,
                                const float*  dS_in, float* dS_out, __half* dO,
                                cudaStream_t stream)
{
    if (sh.d_k == 64 && sh.d_v == 64) {
        launch_gdn_async<64, 64, 4>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 256 && sh.d_v == 256) {
        launch_gdn_async<256, 64, 4>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else {
        fprintf(stderr, "unsupported (d_k=%d, d_v=%d) — only correctness (64/64) and qwen3_next_decode (256/256) wired\n",
                sh.d_k, sh.d_v);
        exit(2);
    }
}

// ============================================================================
// Correctness driver — same shape as cuda-attn-gdn (W1c). Tolerance ≤ 1e-3.
// ============================================================================
static int run_correctness(const std::string& inputs_dir) {
    GDNShape sh{"correctness", /*B*/2, /*H*/4, /*d_k*/64, /*d_v*/64};
    printf("[gdn-async] === correctness run (B=%d H=%d d_k=%d d_v=%d) ===\n",
           sh.batch, sh.n_heads, sh.d_k, sh.d_v);

    Npy q, k, v, a, b, sin_, oexp, sexp;
    if (!read_npy(inputs_dir + "/gdn_correctness_q_f16.npy",            q))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_k_f16.npy",            k))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_v_f16.npy",            v))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_alpha_f16.npy",        a))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_beta_f16.npy",         b))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_S_in_f32.npy",         sin_)) return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_o_expected_f16.npy",   oexp)) return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_S_out_expected_f32.npy", sexp)) return 1;

    const int B_H = sh.batch * sh.n_heads;
    size_t qkv_elems = (size_t)B_H * sh.d_k;
    size_t v_elems   = (size_t)B_H * sh.d_v;
    size_t s_elems   = (size_t)B_H * sh.d_k * sh.d_v;
    size_t o_elems   = (size_t)B_H * sh.d_v;
    size_t scal      = (size_t)B_H;

    if (q.data.size() != qkv_elems * 2 ||
        k.data.size() != qkv_elems * 2 ||
        v.data.size() != v_elems * 2 ||
        a.data.size() != scal * 2 ||
        b.data.size() != scal * 2 ||
        sin_.data.size() != s_elems * 4 ||
        oexp.data.size() != o_elems * 2 ||
        sexp.data.size() != s_elems * 4) {
        fprintf(stderr, "[gdn-async] correctness: NPY shape/size mismatch\n");
        return 1;
    }

    __half *dQ, *dK, *dV, *dA, *dB, *dO;
    float  *dSin, *dSout;
    CK(cudaMalloc(&dQ,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dK,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dV,    v_elems   * sizeof(__half)));
    CK(cudaMalloc(&dA,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dB,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dSin,  s_elems   * sizeof(float)));
    CK(cudaMalloc(&dSout, s_elems   * sizeof(float)));
    CK(cudaMalloc(&dO,    o_elems   * sizeof(__half)));

    CK(cudaMemcpy(dQ,    q.data.data(),    q.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,    k.data.data(),    k.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,    v.data.data(),    v.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,    a.data.data(),    a.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,    b.data.data(),    b.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSin,  sin_.data.data(), sin_.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dSout, 0,                s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0,                o_elems * sizeof(__half)));

    launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    std::vector<float>  hs(s_elems);
    CK(cudaMemcpy(ho.data(), dO,    o_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), dSout, s_elems * sizeof(float),  cudaMemcpyDeviceToHost));

    const __half* o_exp_h = reinterpret_cast<const __half*>(oexp.data.data());
    const float*  s_exp_f = reinterpret_cast<const float*> (sexp.data.data());

    double max_abs_o = 0.0, max_rel_o = 0.0, exp_o_mag = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double aw = fabs((double)want);
        if (aw > exp_o_mag) exp_o_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel_o) max_rel_o = r;
    }
    double max_abs_s = 0.0, max_rel_s = 0.0, exp_s_mag = 0.0;
    for (size_t i = 0; i < s_elems; ++i) {
        float got = hs[i];
        float want = s_exp_f[i];
        double aw = fabs((double)want);
        if (aw > exp_s_mag) exp_s_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_s) max_abs_s = a;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel_s) max_rel_s = r;
    }

    const double ATOL_O = 1e-3;
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;

    printf("[gdn-async] o    max_abs=%.3e max_rel=%.3e   |want|max=%.3e   %s\n",
           max_abs_o, max_rel_o, exp_o_mag, ok_o ? "OK" : "FAIL");
    printf("[gdn-async] Sout max_abs=%.3e max_rel=%.3e   |want|max=%.3e   %s\n",
           max_abs_s, max_rel_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 2;
}

// ============================================================================
// Bench-shape smoke (NO timed iters — orchestrator runs ./bench separately).
// ============================================================================
static int run_bench_shape_smoke(const std::string& inputs_dir) {
    GDNShape sh{"qwen3_next_decode", 1, 16, 256, 256};
    printf("\n[gdn-async] === bench-shape smoke (B=%d H=%d d_k=%d d_v=%d) ===\n",
           sh.batch, sh.n_heads, sh.d_k, sh.d_v);

    Npy q, k, v, a, b, sin_, oexp, sexp;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_q_f16.npy",            q))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_k_f16.npy",            k))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_v_f16.npy",            v))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_alpha_f16.npy",        a))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_beta_f16.npy",         b))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_S_in_f32.npy",         sin_)) return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_o_expected_f16.npy",   oexp)) return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_S_out_expected_f32.npy", sexp)) return 1;

    const int B_H = sh.batch * sh.n_heads;
    size_t qkv_elems = (size_t)B_H * sh.d_k;
    size_t v_elems   = (size_t)B_H * sh.d_v;
    size_t s_elems   = (size_t)B_H * sh.d_k * sh.d_v;
    size_t o_elems   = (size_t)B_H * sh.d_v;
    size_t scal      = (size_t)B_H;

    __half *dQ, *dK, *dV, *dA, *dB, *dO;
    float  *dSin, *dSout;
    CK(cudaMalloc(&dQ,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dK,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dV,    v_elems   * sizeof(__half)));
    CK(cudaMalloc(&dA,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dB,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dSin,  s_elems   * sizeof(float)));
    CK(cudaMalloc(&dSout, s_elems   * sizeof(float)));
    CK(cudaMalloc(&dO,    o_elems   * sizeof(__half)));

    CK(cudaMemcpy(dQ,    q.data.data(),    q.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,    k.data.data(),    k.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,    v.data.data(),    v.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,    a.data.data(),    a.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,    b.data.data(),    b.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSin,  sin_.data.data(), sin_.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dSout, 0,                s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0,                o_elems * sizeof(__half)));

    launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    std::vector<float>  hs(s_elems);
    CK(cudaMemcpy(ho.data(), dO,    o_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), dSout, s_elems * sizeof(float),  cudaMemcpyDeviceToHost));

    const __half* o_exp_h = reinterpret_cast<const __half*>(oexp.data.data());
    const float*  s_exp_f = reinterpret_cast<const float*> (sexp.data.data());

    double max_abs_o = 0.0, max_abs_s = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
    }
    for (size_t i = 0; i < s_elems; ++i) {
        double a = fabs((double)hs[i] - (double)s_exp_f[i]);
        if (a > max_abs_s) max_abs_s = a;
    }
    bool ok_o = max_abs_o <= 1e-2;
    bool ok_s = max_abs_s <= 1e-2;
    printf("[gdn-async] (qwen3) o    max_abs=%.3e   %s\n", max_abs_o, ok_o ? "OK" : "FAIL");
    printf("[gdn-async] (qwen3) Sout max_abs=%.3e   %s\n", max_abs_s, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 3;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[gdn-async] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[gdn-async] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[gdn-async] correctness FAILED at correctness shape — stopping\n");
        return rc;
    }
    rc = run_bench_shape_smoke(inputs_dir);
    return rc;
}
