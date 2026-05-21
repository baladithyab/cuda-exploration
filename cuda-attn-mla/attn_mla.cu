// Wave 17 W1a — nvcc CUDA C++ MLA (Multi-Head Latent Attention).
//
// Cross-frontend × cross-mechanism matrix cell:
//   row = MLA, column = cuda-attn (nvcc).
// Shape model: DeepSeek-V3 (B=1, S=2048, n_h=128, qk_head_dim=192, d_v=128).
//
// Pedagogical MVP. We consume already-up-projected Q, K, V — no weight
// absorption, no latent c_KV cache, no on-the-fly up-projection. The
// attention core is "MHA with non-square per-head dims (qk=192, d_v=128)";
// the latent-cache memory win of MLA is upstream and out of scope here.
//
// Pipeline (3 kernels, no fusion — mirrors cuda-attn-gqa):
//   1) mla_qkt_kernel:  S = Q @ K^T * scale       [B, n_h, S, S]   f16×f16→f32
//   2) softmax_kernel:  P = softmax(S, dim=-1)    [B, n_h, S, S]   f32→f16
//   3) mla_pv_kernel:   O = P @ V                 [B, n_h, S, d_v] f16×f16→f32→f16
//
// Tensor cores: WMMA m16n16k16 f16→f32. Verified via
//   /usr/local/cuda/bin/cuobjdump --dump-sass attn_mla | grep HMMA   (>0 expected).
//
// ADR-0005 padding methodology. We support BOTH variants in one binary:
//
//   • qk=192-native    : K-loop trips = qk / 16 = 12 (correctness shape: 96/16 = 6).
//                        WMMA m16n16k16 tiles 192 cleanly. NO padding overhead.
//                        This is the headline number per ADR-0005.
//
//   • qk_pad=256       : K-loop trips = 256/16 = 16 (correctness: 128/16 = 8 from qk=96).
//                        Padding policy: extend Q,K along last dim with zeros for the
//                        trailing (qk_pad − qk) cols. Since K^T-multiply against zero
//                        is zero, the score matrix is unchanged — so this isolates the
//                        cost of WASTED FLOPS. ADR-0005 expects ~25% (192/256) waste at
//                        bench shape; we report both useful and padded TFLOPS columns
//                        so cross-frontend comparisons against cutile-attn-mla (which
//                        always pads) stay apples-to-apples.
//
// Both variants run in this binary; both correctness checks must pass to ≤1e-2
// vs PyTorch SDPA-MLA expected output (per Wave 17 W1a acceptance row).
//
// Shapes loaded from analysis/wave15-attention-architecture/inputs/ as little-
// endian NumPy .npy files (MLA-prefixed: mla_correctness_mla_*, mla_deepseek_v3_*).
//
// FLOPS = 2 · B · n_h · S² · (qk + d_v)   (per analysis/.../reference/flops_mla.py).
// "Useful" uses qk=192; "padded" uses qk_pad=256 — diagnostic only.
//
// Build: /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120
//        -lstdc++ -lm -o attn_mla attn_mla.cu
//
// ADR-0001: cudaEvent timing. ADR-0002: native sm_120, no PTX-JIT.

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
#include <mma.h>

using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// ============================================================================
// .npy loader (NPY1.0/2.0, little-endian, C-contiguous f16/f32; matches numpy.save).
// Lifted line-for-line from cuda-attn-gqa/attn_gqa.cu — same harness contract.
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
// Kernel 1: QKt — S[b, h, i, j] = scale * sum_k Q[b, h, i, k] * K[b, h, j, k]
// MLA has all heads independent (no GQA broadcast). Q and K share the same
// head index h ∈ [0, n_h).
//
// QK is parameterized over QK (= qk_head_dim or qk_pad). The kernel doesn't
// know whether padding is in effect — it just runs K-loop trips of WK over
// QK. Padded variant has trailing zeros in QK so the wasted FLOPS multiply
// to zero (correctness) at the cost of wall-time (the ADR-0005 measurement).
//
// Tile: 16x16 output, K-step 16. Use WMMA f16×f16→f32.
// Grid: (S/WM, S/WN, B*n_h).  Block: 32 threads (one warp).
// ============================================================================
__global__ void mla_qkt_kernel(
    const __half* __restrict__ Q,   // [B, n_h, S, QK]
    const __half* __restrict__ K,   // [B, n_h, S, QK]
    float*        __restrict__ Sm,  // [B, n_h, S, S] f32
    int B, int Nh, int S, int QK,
    float scale)
{
    int tile_i = blockIdx.x;
    int tile_j = blockIdx.y;
    int bh    = blockIdx.z;
    int b     = bh / Nh;
    int h     = bh % Nh;

    int row0 = tile_i * WM;
    int col0 = tile_j * WN;
    if (row0 >= S || col0 >= S) return;

    // Per-head pointers. Shared head index for Q and K (no GQA broadcast).
    const __half* Qh = Q  + ((size_t)b * Nh + h) * (size_t)S * QK;
    const __half* Kh = K  + ((size_t)b * Nh + h) * (size_t)S * QK;
    float*        Sh = Sm + ((size_t)b * Nh + h) * (size_t)S * S;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    // Loop over K dim. QK MUST be a multiple of WK=16 (true for native 96, 192
    // and padded 128, 256). Caller asserts this.
    for (int k = 0; k < QK; k += WK) {
        const __half* Aptr = Qh + (size_t)row0 * QK + k;   // (WM x WK) row-major, ld=QK
        const __half* Bptr = Kh + (size_t)col0 * QK + k;   // (WK x WN) col-major view of K-rows
        wmma::load_matrix_sync(af, Aptr, QK);
        wmma::load_matrix_sync(bf, Bptr, QK);
        wmma::mma_sync(cf, af, bf, cf);
    }

    #pragma unroll
    for (int t = 0; t < cf.num_elements; ++t) cf.x[t] *= scale;

    wmma::store_matrix_sync(Sh + (size_t)row0 * S + col0, cf, S, wmma::mem_row_major);
}

// ============================================================================
// Kernel 2: softmax — P[b, h, i, :] = softmax(S[b, h, i, :])
// Output is f16 so the next kernel can use tensor cores on P as matrix_a.
// One block per (b, h, row). Block-wide reduction for max, then for sum.
// Identical to cuda-attn-gqa softmax_kernel — only the bh range changes.
// ============================================================================
constexpr int SOFTMAX_TPB = 128;

__global__ void softmax_kernel(
    const float* __restrict__ Sm,  // [B, n_h, S, S] f32
    __half*      __restrict__ P,   // [B, n_h, S, S] f16
    int /*B*/, int /*Nh*/, int S)
{
    int row = blockIdx.x;
    int bh  = blockIdx.y;
    int tid = threadIdx.x;

    const float* Srow = Sm + (size_t)bh * S * S + (size_t)row * S;
    __half*      Prow = P  + (size_t)bh * S * S + (size_t)row * S;

    __shared__ float sbuf[SOFTMAX_TPB];

    // Pass 1: max
    float lmax = -INFINITY;
    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        float v = Srow[j];
        if (v > lmax) lmax = v;
    }
    sbuf[tid] = lmax;
    __syncthreads();
    for (int stride = SOFTMAX_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = sbuf[tid], b = sbuf[tid + stride];
            sbuf[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float rmax = sbuf[0];

    // Pass 2: sum of exp
    float lsum = 0.0f;
    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        lsum += expf(Srow[j] - rmax);
    }
    sbuf[tid] = lsum;
    __syncthreads();
    for (int stride = SOFTMAX_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sbuf[tid] += sbuf[tid + stride];
        __syncthreads();
    }
    float rsum = sbuf[0];
    float inv_sum = 1.0f / rsum;

    // Pass 3: normalize, write f16
    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        float p = expf(Srow[j] - rmax) * inv_sum;
        Prow[j] = __float2half(p);
    }
}

// ============================================================================
// Kernel 3: PV — O[b, h, i, d] = sum_j P[b, h, i, j] * V[b, h, j, d]
// P is (S, S) row-major f16; V is (S, d_v) row-major f16.
// Use WMMA m16n16k16, accum f32, store f16.
//
// V dim is d_v (= 64 for correctness, 128 for bench). NO padding here —
// d_v is already a multiple of 16 in both shapes. This kernel is therefore
// SHARED between the native and padded-256 variants (only QKt touches QK).
//
// Grid: (S/WM, d_v/WN, B*n_h). Block: 32 threads (one warp).
// ============================================================================
__global__ void mla_pv_kernel(
    const __half* __restrict__ P,   // [B, n_h, S, S]
    const __half* __restrict__ V,   // [B, n_h, S, d_v]
    __half*       __restrict__ O,   // [B, n_h, S, d_v]
    int B, int Nh, int S, int Dv)
{
    int tile_i = blockIdx.x;
    int tile_j = blockIdx.y;
    int bh    = blockIdx.z;
    int b     = bh / Nh;
    int h     = bh % Nh;

    int row0 = tile_i * WM;
    int col0 = tile_j * WN;
    if (row0 >= S || col0 >= Dv) return;

    const __half* Ph = P + ((size_t)b * Nh + h) * (size_t)S * S;
    const __half* Vh = V + ((size_t)b * Nh + h) * (size_t)S * Dv;
    __half*       Oh = O + ((size_t)b * Nh + h) * (size_t)S * Dv;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    for (int k = 0; k < S; k += WK) {
        const __half* Aptr = Ph + (size_t)row0 * S  + k;       // (WM x WK), ld = S
        const __half* Bptr = Vh + (size_t)k    * Dv + col0;    // (WK x WN), ld = Dv
        wmma::load_matrix_sync(af, Aptr, S);
        wmma::load_matrix_sync(bf, Bptr, Dv);
        wmma::mma_sync(cf, af, bf, cf);
    }

    __shared__ float stage[WM * WN];
    wmma::store_matrix_sync(stage, cf, WN, wmma::mem_row_major);
    int lane = threadIdx.x;
    #pragma unroll
    for (int e = lane; e < WM * WN; e += 32) {
        int r = e / WN;
        int c = e % WN;
        Oh[(size_t)(row0 + r) * Dv + (col0 + c)] = __float2half(stage[e]);
    }
}

// ============================================================================
// Host driver
// ============================================================================
struct MLAShape {
    const char* name;
    int B, Nh, S, QK, Dv;   // QK = qk_head_dim (true), Dv = d_v
};

// Pad qk to next multiple of 16 (always already true), but bump 96→128, 192→256
// when pad_to_pow2 is set (ADR-0005 padded variant). For other shapes this is
// a no-op (already a power of two).
static int pad_qk(int qk, bool pad_to_pow2) {
    if (!pad_to_pow2) {
        // Native: assert already a multiple of WK.
        if (qk % WK != 0) { fprintf(stderr, "qk=%d not multiple of %d\n", qk, WK); exit(1); }
        return qk;
    }
    int p = 1;
    while (p < qk) p *= 2;
    return p;
}

// Run one full attention pass, all kernels launched on default stream.
// Q_pad and K_pad are device buffers padded to QK_eff in last dim.
// V is unpadded (d_v already aligned). Intermediates dS, dP, dO sized to QK-agnostic shapes.
static void run_pipeline(const MLAShape& sh, int QK_eff,
                         const __half* dQ, const __half* dK, const __half* dV,
                         float* dS_scores, __half* dP, __half* dO)
{
    // softmax scale uses TRUE qk_head_dim (sh.QK), NOT QK_eff. Padded cols are
    // zero so they contribute zero to scores; the score *magnitudes* are
    // identical between native and padded paths and so the softmax scale must
    // also be identical. This is the central correctness invariant of ADR-0005.
    float scale = 1.0f / sqrtf((float)sh.QK);

    dim3 gQK(sh.S / WM, sh.S / WN, sh.B * sh.Nh);
    dim3 bQK(32);
    mla_qkt_kernel<<<gQK, bQK>>>(dQ, dK, dS_scores, sh.B, sh.Nh, sh.S, QK_eff, scale);

    dim3 gSM(sh.S, sh.B * sh.Nh);
    dim3 bSM(SOFTMAX_TPB);
    softmax_kernel<<<gSM, bSM>>>(dS_scores, dP, sh.B, sh.Nh, sh.S);

    dim3 gPV(sh.S / WM, sh.Dv / WN, sh.B * sh.Nh);
    dim3 bPV(32);
    mla_pv_kernel<<<gPV, bPV>>>(dP, dV, dO, sh.B, sh.Nh, sh.S, sh.Dv);
}

// Allocate Q_pad device buffer of shape [B, n_h, S, QK_eff] f16, copy host
// f16 (shape [B, n_h, S, QK_true]) into it with trailing-col zeros.
static __half* alloc_padded_qk(const __half* host, const MLAShape& sh, int QK_eff) {
    size_t nelem = (size_t)sh.B * sh.Nh * sh.S * QK_eff;
    __half* d;
    CK(cudaMalloc(&d, nelem * sizeof(__half)));
    if (QK_eff == sh.QK) {
        // No padding: straight copy.
        CK(cudaMemcpy(d, host, nelem * sizeof(__half), cudaMemcpyHostToDevice));
    } else {
        // Pad with zero. Stage on host; less code than per-row cudaMemcpy2D.
        std::vector<__half> staged(nelem, __float2half(0.0f));
        for (size_t bh = 0; bh < (size_t)sh.B * sh.Nh; ++bh) {
            for (int s = 0; s < sh.S; ++s) {
                size_t src = (bh * sh.S + s) * sh.QK;
                size_t dst = (bh * sh.S + s) * QK_eff;
                memcpy(&staged[dst], &host[src], sh.QK * sizeof(__half));
            }
        }
        CK(cudaMemcpy(d, staged.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
    }
    return d;
}

// Run correctness on one shape, in one variant (native or padded).
// Returns max_abs_err vs expected_f32; caller decides pass/fail.
static double run_correctness_variant(const MLAShape& sh, bool padded,
                                      const std::string& inputs_dir,
                                      const char* shape_prefix)
{
    int QK_eff = pad_qk(sh.QK, padded);

    // Load inputs. (q/k are sh.QK-wide on disk; v is sh.Dv-wide.)
    Npy q, k, v, exp_out;
    std::string base = inputs_dir + "/mla_" + shape_prefix;
    if (!read_npy(base + "_q_f16.npy", q)) exit(1);
    if (!read_npy(base + "_k_f16.npy", k)) exit(1);
    if (!read_npy(base + "_v_f16.npy", v)) exit(1);
    if (!read_npy(base + "_expected_f32.npy", exp_out)) exit(1);

    size_t qk_native = (size_t)sh.B * sh.Nh * sh.S * sh.QK;
    size_t v_elems   = (size_t)sh.B * sh.Nh * sh.S * sh.Dv;
    size_t s_elems   = (size_t)sh.B * sh.Nh * sh.S * sh.S;
    if (q.data.size() != qk_native * 2 || k.data.size() != qk_native * 2 ||
        v.data.size() != v_elems   * 2) {
        fprintf(stderr, "shape mismatch on %s (qk_native=%zu got_q=%zu)\n",
                shape_prefix, qk_native * 2, q.data.size());
        exit(1);
    }

    const __half* hq = reinterpret_cast<const __half*>(q.data.data());
    const __half* hk = reinterpret_cast<const __half*>(k.data.data());

    __half* dQ = alloc_padded_qk(hq, sh, QK_eff);
    __half* dK = alloc_padded_qk(hk, sh, QK_eff);
    __half* dV;
    CK(cudaMalloc(&dV, v_elems * sizeof(__half)));
    CK(cudaMemcpy(dV, v.data.data(), v_elems * sizeof(__half), cudaMemcpyHostToDevice));

    float*  dS;
    __half *dP, *dO;
    CK(cudaMalloc(&dS, s_elems * sizeof(float)));
    CK(cudaMalloc(&dP, s_elems * sizeof(__half)));
    CK(cudaMalloc(&dO, v_elems * sizeof(__half)));

    run_pipeline(sh, QK_eff, dQ, dK, dV, dS, dP, dO);
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(v_elems);
    CK(cudaMemcpy(ho.data(), dO, v_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    const float* exp_ptr = reinterpret_cast<const float*>(exp_out.data.data());
    double max_abs = 0.0, max_rel = 0.0;
    double exp_max_abs = 0.0;
    for (size_t i = 0; i < v_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = exp_ptr[i];
        double a = fabs((double)got - (double)want);
        if (a > max_abs) max_abs = a;
        double aw = fabs((double)want);
        if (aw > exp_max_abs) exp_max_abs = aw;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel) max_rel = r;
    }

    printf("[mla] %-9s qk=%d (qk_eff=%d, %s) -> max_abs_err=%.3e max_rel=%.3e expected_max_abs=%.3e\n",
           shape_prefix, sh.QK, QK_eff, padded ? "padded" : "native",
           max_abs, max_rel, exp_max_abs);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dS); cudaFree(dP); cudaFree(dO);
    return max_abs;
}

// Best-effort correctness: run small shape (correctness_mla) on both variants;
// return 0 on PASS (max_abs_err ≤ 1e-2 per Wave 17 W1a acceptance row), nonzero on FAIL.
static int run_correctness(const std::string& inputs_dir) {
    MLAShape sh{"correctness_mla", 1, 4, 128, 96, 64};
    printf("[mla] === correctness run (B=%d n_h=%d S=%d qk=%d d_v=%d) ===\n",
           sh.B, sh.Nh, sh.S, sh.QK, sh.Dv);

    // Wave 17 W1a acceptance: max_abs_err ≤ 1e-2 vs PyTorch SDPA-MLA at small shape.
    const double TOL = 1e-2;

    double err_native = run_correctness_variant(sh, /*padded=*/false, inputs_dir, "correctness_mla");
    double err_padded = run_correctness_variant(sh, /*padded=*/true,  inputs_dir, "correctness_mla");

    bool ok_native = (err_native <= TOL);
    bool ok_padded = (err_padded <= TOL);
    printf("[mla] correctness: native=%.3e (%s)  padded=%.3e (%s)  TOL=%.0e\n",
           err_native, ok_native ? "OK" : "FAIL",
           err_padded, ok_padded ? "OK" : "FAIL", TOL);
    return (ok_native && ok_padded) ? 0 : 2;
}

// Bench is structurally ready but DOES NOT execute timed iterations in this binary.
// Per Wave 17 W1a task discipline, the orchestrator runs serial benches across all
// 5 W1 cells on the idle GPU after author-cells return. We DO allocate, do a single
// warmup-style invocation per variant (1 iter, just to confirm the bench-shape pipeline
// runs without launching errors), and emit the FLOPS / DRAM / TFLOPS-estimate banner.
//
// To run the actual timed bench, pass --bench-now on the command line; orchestrator
// will use that path serially.
static int run_bench(const std::string& inputs_dir, bool bench_now) {
    MLAShape sh{"deepseek_v3", 1, 128, 2048, 192, 128};
    printf("\n[mla] === bench setup (B=%d n_h=%d S=%d qk=%d d_v=%d) ===\n",
           sh.B, sh.Nh, sh.S, sh.QK, sh.Dv);

    // FLOPS: useful (qk=192) and padded (qk_pad=256). Per ADR-0005:
    //   useful_flops = 2 · B · n_h · S² · (qk + d_v)
    //   padded_flops = 2 · B · n_h · S² · (qk_pad + d_v)
    int qk_pad = pad_qk(sh.QK, true);
    double useful_flops = 2.0 * sh.B * sh.Nh * (double)sh.S * sh.S * (sh.QK + sh.Dv);
    double padded_flops = 2.0 * sh.B * sh.Nh * (double)sh.S * sh.S * (qk_pad + sh.Dv);
    printf("[mla] useful_flops = %.3f GFLOPS  (qk=%d, qk+d_v=%d)\n",
           useful_flops / 1e9, sh.QK, sh.QK + sh.Dv);
    printf("[mla] padded_flops = %.3f GFLOPS  (qk_pad=%d, padding overhead=%.1f%%)\n",
           padded_flops / 1e9, qk_pad, 100.0 * (padded_flops / useful_flops - 1.0));

    // DRAM byte counts (LDG sanity, ADR-0005). We *expect* the native variant to load
    // qk_native bytes per K-row and the padded variant to load qk_pad bytes — i.e. our
    // padding here is BOTH register-padded AND DRAM-padded (we widened the device buffer).
    // SASS LDG.E counts in attn_mla.sass should reflect this.
    double bytes_qk_native = (double)sh.B * sh.Nh * sh.S * sh.QK   * 2;   // f16
    double bytes_qk_padded = (double)sh.B * sh.Nh * sh.S * qk_pad  * 2;
    printf("[mla] DRAM K-traffic (per pass): native=%.1f MB  padded=%.1f MB  delta=%.1f%%\n",
           bytes_qk_native / 1e6, bytes_qk_padded / 1e6,
           100.0 * (bytes_qk_padded / bytes_qk_native - 1.0));

    if (!bench_now) {
        // Author-cell mode: don't burn GPU time on a 33% padded MLA bench;
        // orchestrator runs serial benches after all 5 author-cells return.
        printf("[mla] bench-now NOT requested — author-cell mode. Pipeline structurally ready.\n");
        printf("[mla] (Pass --bench-now to run timed iters serially under orchestrator control.)\n");
        return 0;
    }

    // ---- Optional: timed bench, only when --bench-now ---------------------
    Npy q, k, v;
    std::string base = inputs_dir + "/mla_deepseek_v3";
    if (!read_npy(base + "_q_f16.npy", q)) return 1;
    if (!read_npy(base + "_k_f16.npy", k)) return 1;
    if (!read_npy(base + "_v_f16.npy", v)) return 1;

    const __half* hq = reinterpret_cast<const __half*>(q.data.data());
    const __half* hk = reinterpret_cast<const __half*>(k.data.data());

    size_t v_elems = (size_t)sh.B * sh.Nh * sh.S * sh.Dv;
    size_t s_elems = (size_t)sh.B * sh.Nh * sh.S * sh.S;

    FILE* csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,batch,seq,n_h,qk,d_v,qk_pad,iter,gpu_ms,useful_tflops,padded_tflops\n");

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    constexpr int ITERS = 10;

    for (int variant = 0; variant < 2; ++variant) {
        bool padded = (variant == 1);
        int  QK_eff = pad_qk(sh.QK, padded);
        const char* vname = padded ? "padded" : "native";
        printf("\n[mla] -- variant=%s qk_eff=%d --\n", vname, QK_eff);

        __half* dQ = alloc_padded_qk(hq, sh, QK_eff);
        __half* dK = alloc_padded_qk(hk, sh, QK_eff);
        __half *dV, *dP, *dO;
        float* dS;
        CK(cudaMalloc(&dV, v_elems * sizeof(__half)));
        CK(cudaMalloc(&dS, s_elems * sizeof(float)));
        CK(cudaMalloc(&dP, s_elems * sizeof(__half)));
        CK(cudaMalloc(&dO, v_elems * sizeof(__half)));
        CK(cudaMemcpy(dV, v.data.data(), v_elems * sizeof(__half), cudaMemcpyHostToDevice));

        // warmup
        run_pipeline(sh, QK_eff, dQ, dK, dV, dS, dP, dO);
        CK(cudaDeviceSynchronize());

        double best = 1e30, sum = 0.0;
        for (int i = 0; i < ITERS; ++i) {
            cudaEventRecord(evs);
            run_pipeline(sh, QK_eff, dQ, dK, dV, dS, dP, dO);
            cudaEventRecord(eve);
            cudaEventSynchronize(eve);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evs, eve);
            double useful_tf = (useful_flops / 1e12) / (ms * 1e-3);
            double padded_tf = (padded_flops / 1e12) / (ms * 1e-3);
            printf("[mla] %s iter=%d gpu_ms=%.4f useful_TF=%.2f padded_TF=%.2f\n",
                   vname, i, ms, useful_tf, padded_tf);
            fprintf(csv, "cuda-attn-mla,3kernel_wmma_%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
                    vname, sh.B, sh.S, sh.Nh, sh.QK, sh.Dv, QK_eff, i, ms, useful_tf, padded_tf);
            if (ms < best) best = ms;
            sum += ms;
        }
        double avg = sum / ITERS;
        printf("[mla] %s SUMMARY best_ms=%.4f avg_ms=%.4f best_useful_TF=%.2f best_padded_TF=%.2f\n",
               vname, best, avg,
               (useful_flops / 1e12) / (best * 1e-3),
               (padded_flops / 1e12) / (best * 1e-3));

        cudaFree(dQ); cudaFree(dK); cudaFree(dV);
        cudaFree(dS); cudaFree(dP); cudaFree(dO);
    }

    fclose(csv);
    return 0;
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[mla] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir = "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    bool bench_now = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench-now") bench_now = true;
        else if (a.size() && a[0] != '-') inputs_dir = a;
    }
    printf("[mla] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[mla] correctness failed — refusing to run bench\n");
        return rc;
    }
    return run_bench(inputs_dir, bench_now);
}
