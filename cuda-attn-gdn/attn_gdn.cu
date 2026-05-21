// Wave 17 W1c — nvcc CUDA C++ Gated DeltaNet (GDN) single-timestep decode.
//
// Reference: cutile-attn-gdn/main.py — same algorithm in cuTile (610 GB/s on
// RTX 5090 sm_120). This C++ port targets the same shapes and grid layout
// but uses thread-level vectorized loads (float4 → LDG.E.128 / STG.E.128)
// to push past cuTile's scalar f32 LDG ceiling on the state tile traffic.
//
// Per (batch, head, bv-block) block:
//   load q[0:d_k]        f16
//   load k[0:d_k]        f16
//   load v[bv:bv+BV]     f16
//   load alpha, beta     f16  (per-head scalars)
//   load S_in[0:d_k, bv:bv+BV]   f32   <-- the dominant traffic, 64 KB at d_k=256, BV=64
//
// Compute:
//   S_scaled  = alpha * S_in
//   u         = k^T  · S_scaled                     (BV,) f32
//   residual  = v - u                                (BV,) f32
//   S_out     = S_scaled + beta · k ⊗ residual      (d_k, BV) f32
//   o         = q^T · S_out                          (BV,) f32
//
// Store S_out (f32) and o (f16). Single fused kernel — no HBM round-trip
// between the state update and the output read.
//
// Memory-bound: per-iter HBM traffic ≈ 8.2 KiB (state read+write dominates),
// per-iter compute ≈ 6.3 MFLOPS. AI ≈ 0.77 flops/byte; ridge ≈ 25.
// HMMA is NOT applicable (no matmul-shaped op large enough to win on TC
// fragments); all matmul-flavored ops are matvec / outer products at d_k=256.
//
// Grid (matching cuTile):  (B*H, D_V / BLOCK_V)
// Block: TPB threads, where each thread owns 4 contiguous d_v columns.
// TPB = BLOCK_V / 4 (at least 32 to fill a warp; correctness shape pads).
//
// Build:
//   /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ -o attn_gdn attn_gdn.cu

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

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================================
// .npy loader (NPY1.0 / NPY2.0; little-endian only). Same as cuda-attn-gqa.
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
    fread(ver, 1, 2, f);
    uint32_t header_len;
    if (ver[0] == 1) {
        uint16_t hl; fread(&hl, 1, 2, f); header_len = hl;
    } else {
        fread(&header_len, 1, 4, f);
    }
    std::string hdr(header_len, ' ');
    fread(&hdr[0], 1, header_len, f);

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
// GDN-decode kernel.
//
// Template params:
//   D_K     — key dim (state rows). Compile-time so the inner loops fully unroll.
//   BLOCK_V — d_v-tile width per block (must be multiple of 4; we vectorize as float4).
//
// Runtime parallelism:
//   gridDim  = (B * H, D_V / BLOCK_V)
//   blockDim = (BLOCK_V / 4)   each thread owns 4 d_v columns (one float4 stripe)
//
// Memory layout in HBM (matches the harness packing):
//   Q       (B*H, D_K)         f16
//   K       (B*H, D_K)         f16
//   V       (B*H, D_V)         f16
//   Alpha   (B*H,)             f16
//   Beta    (B*H,)             f16
//   S_in    (B*H, D_K, D_V)    f32 row-major over (D_K, D_V)
//   S_out   (B*H, D_K, D_V)    f32 row-major
//   O       (B*H, D_V)         f16
//
// Per-thread state in regs:
//   u_local  (4 floats)         partial result of  k · S_scaled  for this stripe
//   r_local  (4 floats)         residual = v - u
//   o_local  (4 floats)         partial output q · S_out
//   plus per-iteration register float4 holding the current d_k-row of S.
//
// Per-block shared mem:
//   smem_q[D_K]   (f32)   q upcast once
//   smem_k[D_K]   (f32)   k upcast once
//   smem_S_scaled[D_K * (BLOCK_V/4)] (float4)  the alpha-scaled state tile,
//                                      kept in shared between u and S_out passes.
//   For D_K=256, BLOCK_V=64: 256 * 16 * 16B = 64 KB. Within sm_120's 100 KB budget.
//   For D_K=64,  BLOCK_V=64: 64  *  16 * 16B = 16 KB. Tiny.
// ============================================================================
template <int D_K, int BLOCK_V>
__global__ void gdn_decode_kernel(
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
    constexpr int VLANES = BLOCK_V / 4;     // # float4 stripes per block = active threads
    constexpr int TPB    = VLANES;          // one thread per stripe

    const int bh = blockIdx.x;              // 0..B*H-1
    const int bv = blockIdx.y;              // 0..D_V/BLOCK_V-1
    const int tid = threadIdx.x;

    const int col0 = bv * BLOCK_V + tid * 4;  // first d_v column for this thread

    // ── Shared memory layout ──
    extern __shared__ unsigned char smem_raw[];
    float*  smem_q  = reinterpret_cast<float*>(smem_raw);                    // [D_K]
    float*  smem_k  = smem_q + D_K;                                          // [D_K]
    float4* smem_S  = reinterpret_cast<float4*>(smem_k + D_K);                // [D_K * VLANES]

    // ── Load q, k (f16 → f32) cooperatively into shared memory ──
    // Each thread strides through D_K. D_K is small (<= 256) so this is fast.
    const __half* Qbh = Q + (size_t)bh * D_K;
    const __half* Kbh = K + (size_t)bh * D_K;
    for (int k = tid; k < D_K; k += TPB) {
        smem_q[k] = __half2float(Qbh[k]);
        smem_k[k] = __half2float(Kbh[k]);
    }

    // Per-block scalars: alpha, beta. Have thread 0 fetch and broadcast via smem.
    __shared__ float s_alpha;
    __shared__ float s_beta;
    if (tid == 0) {
        s_alpha = __half2float(Alpha[bh]);
        s_beta  = __half2float(Beta[bh]);
    }

    // Per-thread vector slots.
    float4 v_vec;
    {
        // Each thread loads 4 f16 v components and converts → float4.
        // Reads (B*H, D_V), 4 contiguous halves at column [col0, col0+4).
        const __half* Vbh = V + (size_t)bh * D_V;
        // Group as 4 contiguous f16 = 8 bytes. Use a single LDG.E.64 then convert.
        const float2* Vptr = reinterpret_cast<const float2*>(Vbh + col0);
        float2 v_packed = *Vptr;  // 8 bytes = 4 halves
        const __half* hv = reinterpret_cast<const __half*>(&v_packed);
        v_vec.x = __half2float(hv[0]);
        v_vec.y = __half2float(hv[1]);
        v_vec.z = __half2float(hv[2]);
        v_vec.w = __half2float(hv[3]);
    }

    __syncthreads();
    const float alpha = s_alpha;
    const float beta  = s_beta;

    // ── Pass 1: load (D_K, BLOCK_V) state tile, scale by alpha, write to shared,
    //   accumulate u_local = sum_k k[k] * (alpha * S_in[k, col0:col0+4]) per thread. ──
    //
    // Per row k of S_in (D_V floats): each thread loads 4 contiguous floats →
    // float4 → LDG.E.128 (16 B per thread).
    //
    // Pointer: S_in[bh, k, bv*BLOCK_V + tid*4]
    const size_t bh_state_off = (size_t)bh * D_K * D_V;
    const float* Sbh = S_in + bh_state_off;

    float4 u_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        // Vectorized 128-bit load of 4 f32 columns at row k.
        const float4* sptr =
            reinterpret_cast<const float4*>(Sbh + (size_t)k * D_V + col0);
        float4 s = *sptr;                         // <-- LDG.E.128
        s.x *= alpha;
        s.y *= alpha;
        s.z *= alpha;
        s.w *= alpha;

        // Cache S_scaled into shared memory for the second pass.
        smem_S[(size_t)k * VLANES + tid] = s;

        // Accumulate u_local += k_k * s_scaled
        float kk = smem_k[k];
        u_acc.x += kk * s.x;
        u_acc.y += kk * s.y;
        u_acc.z += kk * s.z;
        u_acc.w += kk * s.w;
    }

    // ── Residual r = v - u  (per-thread 4 floats; no cross-thread reduction needed
    //   because each (col)·(d_k) pair lives on the same thread). ──
    float4 r;
    r.x = v_vec.x - u_acc.x;
    r.y = v_vec.y - u_acc.y;
    r.z = v_vec.z - u_acc.z;
    r.w = v_vec.w - u_acc.w;

    // ── Pass 2: S_out = S_scaled + beta · k ⊗ r,  then  o += q[k] * S_out_row ──
    float* Sout_bh = S_out + bh_state_off;
    float4 o_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        float4 s = smem_S[(size_t)k * VLANES + tid];   // S_scaled[k, my_4_cols]
        float kk = smem_k[k];
        float qk = smem_q[k];
        float bk = beta * kk;

        // S_out = S_scaled + (beta · k) · r  (broadcast in BLOCK_V dim — but r is
        // per-thread already, so this is a per-element FMA on r.{xyzw}).
        s.x = s.x + bk * r.x;
        s.y = s.y + bk * r.y;
        s.z = s.z + bk * r.z;
        s.w = s.w + bk * r.w;

        // Vectorized 128-bit store of S_out row.
        float4* sout_ptr =
            reinterpret_cast<float4*>(Sout_bh + (size_t)k * D_V + col0);
        *sout_ptr = s;                               // <-- STG.E.128

        // Accumulate o = q · S_out  along d_k axis.
        o_acc.x += qk * s.x;
        o_acc.y += qk * s.y;
        o_acc.z += qk * s.z;
        o_acc.w += qk * s.w;
    }

    // ── Store o (f16). Each thread writes 4 halves contiguously → 8-byte store. ──
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

template <int D_K, int BLOCK_V>
static void launch_gdn(const GDNShape& sh,
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

    // Shared mem: 2 * D_K floats (q, k) + D_K * VLANES float4 (S_scaled tile).
    size_t smem_bytes = 2 * D_K * sizeof(float) + (size_t)D_K * VLANES * sizeof(float4);

    // For sm_120 the default per-block dynamic-smem max is 48 KB; opt in to
    // larger if we exceed (D_K=256 BV=64 → 16+16+64 KB tile = needs the full 100 KB).
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)gdn_decode_kernel<D_K, BLOCK_V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
    }

    gdn_decode_kernel<D_K, BLOCK_V><<<grid, block, smem_bytes, stream>>>(
        dQ, dK, dV, dA, dB, dS_in, dS_out, dO, B_H, sh.d_v);
}

// Dispatch (D_K, BLOCK_V) at runtime to the right template instantiation.
static void launch_gdn_dispatch(const GDNShape& sh,
                                const __half* dQ, const __half* dK, const __half* dV,
                                const __half* dA, const __half* dB,
                                const float*  dS_in, float* dS_out, __half* dO,
                                cudaStream_t stream)
{
    if (sh.d_k == 64 && sh.d_v == 64) {
        launch_gdn<64, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 256 && sh.d_v == 256) {
        launch_gdn<256, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else {
        fprintf(stderr, "unsupported (d_k=%d, d_v=%d) — only correctness (64/64) and qwen3_next_decode (256/256) wired\n",
                sh.d_k, sh.d_v);
        exit(2);
    }
}

// ============================================================================
// Correctness driver
// ============================================================================
static int run_correctness(const std::string& inputs_dir) {
    GDNShape sh{"correctness", /*B*/2, /*H*/4, /*d_k*/64, /*d_v*/64};
    printf("[gdn] === correctness run (B=%d H=%d d_k=%d d_v=%d) ===\n",
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
    size_t qkv_elems = (size_t)B_H * sh.d_k;     // q, k same shape (B,H,d_k)
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
        fprintf(stderr, "[gdn] correctness: NPY shape/size mismatch\n"
                "  q   bytes=%zu  expect=%zu\n"
                "  k   bytes=%zu  expect=%zu\n"
                "  v   bytes=%zu  expect=%zu\n"
                "  a   bytes=%zu  expect=%zu\n"
                "  b   bytes=%zu  expect=%zu\n"
                "  S   bytes=%zu  expect=%zu\n"
                "  oE  bytes=%zu  expect=%zu\n"
                "  sE  bytes=%zu  expect=%zu\n",
                q.data.size(), qkv_elems*2,
                k.data.size(), qkv_elems*2,
                v.data.size(), v_elems*2,
                a.data.size(), scal*2,
                b.data.size(), scal*2,
                sin_.data.size(), s_elems*4,
                oexp.data.size(), o_elems*2,
                sexp.data.size(), s_elems*4);
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

    // Output check (f16): compare upcast f32 to expected f16 upcast.
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

    // Tolerance: o is f16-cast — atol 5e-3 (W17 plan accept: max_abs_err ≤ 1e-3 vs PyTorch GDN-naive
    // for OUTPUT; state is f32-precision but with f16 input, expect f16-roundoff scale).
    const double ATOL_O = 1e-3;
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;

    printf("[gdn] o    max_abs=%.3e max_rel=%.3e   |want|max=%.3e   %s\n",
           max_abs_o, max_rel_o, exp_o_mag, ok_o ? "OK" : "FAIL");
    printf("[gdn] Sout max_abs=%.3e max_rel=%.3e   |want|max=%.3e   %s\n",
           max_abs_s, max_rel_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 2;
}

// ============================================================================
// Bench-shape launch (NO timed iters per Wave 17 W1c instructions; just a
// single kernel-launch for compile-and-correctness sanity at the larger shape).
// Compares a few output samples to the saved expected NPY to confirm the
// (D_K=256, BLOCK_V=64) instantiation is also numerically sound.
// ============================================================================
static int run_bench_shape_smoke(const std::string& inputs_dir) {
    GDNShape sh{"qwen3_next_decode", 1, 16, 256, 256};
    printf("\n[gdn] === bench-shape smoke (B=%d H=%d d_k=%d d_v=%d) ===\n",
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
    // Bench shape uses larger d_k=256 → more accumulated f16-roundoff in u/o.
    // Output is cast to f16 (eps ~4e-4 per output magnitude); tolerate looser bound.
    bool ok_o = max_abs_o <= 1e-2;
    bool ok_s = max_abs_s <= 1e-2;
    printf("[gdn] (qwen3) o   max_abs=%.3e   %s\n", max_abs_o, ok_o ? "OK" : "FAIL");
    printf("[gdn] (qwen3) Sout max_abs=%.3e %s\n", max_abs_s, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 3;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[gdn] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[gdn] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[gdn] correctness FAILED at correctness shape — stopping\n");
        return rc;
    }
    rc = run_bench_shape_smoke(inputs_dir);
    return rc;
}
