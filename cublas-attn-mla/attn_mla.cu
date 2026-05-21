// Wave 17 W2c — cuBLAS-3-kernel MLA (Multi-Head Latent Attention) attention.
//
// Cross-frontend × cross-mechanism matrix cell:
//   row = MLA, column = cublas-attn (cuBLAS GemmEx + custom softmax kernel).
// Shape model: DeepSeek-V3 (B=1, S=2048, n_h=128, qk_head_dim=192, d_v=128).
//
// Pipeline (3 kernels, mirrors cublas-attn-gqa structure but adapted to MLA's
// non-square per-head dims and absence of GQA broadcast):
//
//   Stage 1 (cublasGemmEx):  scores = (Q @ K^T) * scale   [B, n_h, S, S]   f32
//   Stage 2 (custom softmax kernel): probs = softmax(scores)               f16
//   Stage 3 (cublasGemmEx):  out    = probs @ V          [B, n_h, S, d_v]  f16
//
// f16 matmul inputs/outputs, f32 compute accumulator, CUBLAS_GEMM_DEFAULT_TENSOR_OP.
// MATCHES the cublas-attn-gqa harness contract; only differences are:
//   • no GQA broadcast (q-head index = kv-head index, h_q == h_kv == h)
//   • non-square per-head: QK_eff vs Dv (qk_head_dim != d_v)
//   • per-stage timing emitted in printf only (no CSV by default — author cell);
//     bench-now timed iters disabled per Wave 17 W2c task discipline.
//
// ADR-0005 padding methodology. We support BOTH variants in one binary:
//
//   • qk=192-native    : K-loop trips inside cublasGemmEx are over QK=192.
//                        cuBLAS heuristics will pick a TC algo that handles the
//                        non-power-of-two K cleanly (192 = 12·16). NO padding overhead.
//                        This is the headline number per ADR-0005.
//
//   • qk_pad=256       : K-loop trips over QK=256. Padding policy: extend Q,K along
//                        last dim with zeros for the trailing (qk_pad − qk) cols.
//                        Score matrix is bit-identical to native (zeros·anything = 0).
//                        Diagnostic: isolates wasted-FLOPS cost.
//
// Per ADR-0005, useful_flops uses qk=192, padded_flops uses qk_pad=256, and
// the bit-identical correctness invariant applies between the two variants.
//
// FLOPS = 2 · B · n_h · S² · (qk + d_v)   (per analysis/.../reference/flops_mla.py).
//
// Build: /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120
//        -lstdc++ -lm -lcublas -o attn_mla attn_mla.cu softmax.cu
//
// Author cell only: NO timed bench by default; correctness check + structural
// readiness only. Use --bench-now to opt in (orchestrator runs serial benches).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS status %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================================
// .npy loader — same contract as cublas-attn-gqa/attn_gqa.cu and
// cuda-attn-mla/attn_mla.cu. Supports NPY1.0/2.0 little-endian f16/f32 C-contig.
// ============================================================================

struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;   // "<f2" or "<f4"
    std::vector<uint8_t> data;
    size_t elem_size() const { return dtype == "<f2" ? 2 : 4; }
    size_t num_elems() const {
        size_t n = 1; for (auto d : shape) n *= (size_t)d; return n;
    }
};

static Npy load_npy(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    uint8_t magic[6];
    if (fread(magic, 1, 6, f) != 6) { fprintf(stderr, "short read on %s\n", path); exit(1); }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path); exit(1);
    }
    uint8_t ver_major = 0, ver_minor = 0;
    if (fread(&ver_major, 1, 1, f) != 1) exit(1);
    if (fread(&ver_minor, 1, 1, f) != 1) exit(1);
    uint32_t header_len = 0;
    if (ver_major == 1) {
        uint16_t hl16 = 0;
        if (fread(&hl16, 2, 1, f) != 1) exit(1);
        header_len = hl16;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) exit(1);
    }
    std::string header(header_len, ' ');
    if (fread(header.data(), 1, header_len, f) != header_len) exit(1);

    Npy out;
    auto dp = header.find("'descr':");
    if (dp == std::string::npos) dp = header.find("\"descr\":");
    auto sq1 = header.find("'", dp + 8);
    auto sq2 = header.find("'", sq1 + 1);
    out.dtype = header.substr(sq1 + 1, sq2 - sq1 - 1);

    if (header.find("'fortran_order': True") != std::string::npos) {
        fprintf(stderr, "%s: fortran_order not supported\n", path); exit(1);
    }

    auto sp = header.find("'shape':");
    auto lp = header.find("(", sp);
    auto rp = header.find(")", lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    int64_t cur = 0; bool have = false;
    for (char c : shape_str) {
        if (c >= '0' && c <= '9') { cur = cur * 10 + (c - '0'); have = true; }
        else { if (have) { out.shape.push_back(cur); cur = 0; have = false; } }
    }
    if (have) out.shape.push_back(cur);

    size_t n_bytes = out.num_elems() * out.elem_size();
    out.data.resize(n_bytes);
    if (fread(out.data.data(), 1, n_bytes, f) != n_bytes) {
        fprintf(stderr, "%s: short read on data\n", path); exit(1);
    }
    fclose(f);
    return out;
}

// ============================================================================
// softmax launcher (defined in softmax.cu)
// ============================================================================

extern "C" void launch_mla_row_softmax_scale(
    const float* d_scores,
    __half*      d_probs,
    int          num_rows,
    int          seq,
    float        scale,
    cudaStream_t stream);

// ============================================================================
// MLA shape config
// ============================================================================

struct MLAShape {
    const char* name;
    int B;
    int Nh;     // n_heads (no GQA broadcast in MLA — Q, K, V all share h)
    int S;
    int QK;     // qk_head_dim (true)
    int Dv;     // d_v
};

// Pad qk to next power of two if pad_to_pow2 set.
// Native (pad_to_pow2=false) just returns qk and asserts QK%16==0.
static int pad_qk(int qk, bool pad_to_pow2) {
    if (!pad_to_pow2) {
        if (qk % 16 != 0) {
            fprintf(stderr, "qk=%d not multiple of 16 — cuBLAS TC GEMM requires K%%16==0\n", qk);
            exit(1);
        }
        return qk;
    }
    int p = 1;
    while (p < qk) p *= 2;
    return p;
}

// per-head row-major offset for [B, Nh, S, D] tensors
static inline size_t head_offset(int b, int h, int S, int D, int Nh) {
    return ((size_t)b * Nh + h) * (size_t)S * D;
}

// ============================================================================
// run_attention_once — one full forward pass over (B, n_h) heads.
// QK_eff is the on-device width of Q/K (192 native or 256 padded).
// stage_ms[0..2] receive QKt / softmax / PV elapsed times.
// ============================================================================
static void run_attention_once(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const MLAShape& sh,
    int QK_eff,
    const __half* d_Q,   // [B, Nh, S, QK_eff] f16
    const __half* d_K,   // [B, Nh, S, QK_eff] f16
    const __half* d_V,   // [B, Nh, S, Dv]     f16
    float*        d_scores,   // [B, Nh, S, S] f32
    __half*       d_probs,    // [B, Nh, S, S] f16
    __half*       d_out,      // [B, Nh, S, Dv] f16
    float stage_ms[3],
    cudaEvent_t ev_a, cudaEvent_t ev_b, cudaEvent_t ev_c, cudaEvent_t ev_d)
{
    const int B  = sh.B;
    const int Nh = sh.Nh;
    const int S  = sh.S;
    const int Dv = sh.Dv;
    const float alpha = 1.0f, beta = 0.0f;

    // -----------------------------------------------------------------------
    // Stage 1: scores = Q @ K^T (f32 accum, f16 inputs).
    //
    // For each (b, h):
    //   row-major Q_head: (S, QK_eff)  -> col-major view (QK_eff, S), lda = QK_eff
    //   row-major K_head: (S, QK_eff)  -> col-major view (QK_eff, S), ldb = QK_eff
    //   row-major C_head: (S, S)       -> col-major view (S, S),     ldc = S
    //
    // Want row-major C[i,j] = sum_k Q[i,k] * K[j,k]
    //                       = sum_k Q_col[k,i] * K_col[k,j]
    //                       = (Q_col^T * K_col)[i,j]   (col-major)
    //   => gemm(op_A=T, op_B=N, M=S, N=S, K=QK_eff,
    //           A = K_col (QK_eff,S) lda=QK_eff   <- swap A/B in cuBLAS-col order
    //           B = Q_col (QK_eff,S) ldb=QK_eff
    //           C = scores_col (S,S) ldc=S)
    // (cuBLAS view: C_col = K_col^T * Q_col evaluates to scores[j,i] in col-major,
    //  which equals scores[i,j] in row-major after transpose. So we ask op_A=T on
    //  the FIRST cuBLAS arg, with K as A and Q as B — same trick as cublas-attn-gqa.)
    //
    // Scale (1/sqrt(QK_TRUE)) is applied INSIDE the softmax kernel, so we leave
    // alpha=1 here. (cuBLAS could fold scale via alpha; we keep it inside softmax
    // to match the cublas-attn-gqa harness contract exactly.)
    // -----------------------------------------------------------------------

    CK(cudaEventRecord(ev_a, stream));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Nh; ++h) {
            const __half* q_head = d_Q + head_offset(b, h, S, QK_eff, Nh);
            const __half* k_head = d_K + head_offset(b, h, S, QK_eff, Nh);
            float*        s_head = d_scores + head_offset(b, h, S, S, Nh);
            CB(cublasGemmEx(
                cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                S, S, QK_eff,
                &alpha,
                k_head, CUDA_R_16F, QK_eff,    // A = K_col (QK_eff, S)
                q_head, CUDA_R_16F, QK_eff,    // B = Q_col (QK_eff, S)
                &beta,
                s_head, CUDA_R_32F, S,          // C_col (S, S) lda=S
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CK(cudaEventRecord(ev_b, stream));

    // -----------------------------------------------------------------------
    // Stage 2: softmax(scores * scale_TRUE) -> probs (f16).
    // Per ADR-0005: scale uses TRUE qk (sh.QK), NOT QK_eff. Padded cols are
    // zero so they contribute zero to scores; magnitudes are bit-identical
    // between native and padded variants and so the softmax scale is identical.
    // -----------------------------------------------------------------------
    int num_rows = B * Nh * S;
    float scale = 1.0f / sqrtf((float)sh.QK);
    launch_mla_row_softmax_scale(d_scores, d_probs, num_rows, S, scale, stream);
    CK(cudaEventRecord(ev_c, stream));

    // -----------------------------------------------------------------------
    // Stage 3: out = probs @ V (f16 in/out, f32 accum).
    //   row-major probs_head: (S, S),  col-major view (S, S),  lda = S
    //   row-major V_head:     (S, Dv), col-major view (Dv, S), lda = Dv
    //   row-major O_head:     (S, Dv), col-major view (Dv, S), lda = Dv
    //
    // Want O[i,j]_row = sum_k probs[i,k] * V[k,j]
    //                 = sum_k probs_col[k,i] * V_col[j,k]
    //   O_col[j,i] = (V_col * probs_col)[j,i]
    //   => gemm(op_A=N, op_B=N, M=Dv, N=S, K=S,
    //           A = V_col (Dv, S) lda=Dv,
    //           B = probs_col (S, S) ldb=S,
    //           C = O_col (Dv, S) ldc=Dv)
    // -----------------------------------------------------------------------
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Nh; ++h) {
            const __half* v_head = d_V + head_offset(b, h, S, Dv, Nh);
            const __half* p_head = d_probs + head_offset(b, h, S, S, Nh);
            __half*       o_head = d_out + head_offset(b, h, S, Dv, Nh);
            CB(cublasGemmEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                Dv, S, S,
                &alpha,
                v_head, CUDA_R_16F, Dv,
                p_head, CUDA_R_16F, S,
                &beta,
                o_head, CUDA_R_16F, Dv,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CK(cudaEventRecord(ev_d, stream));

    CK(cudaEventSynchronize(ev_d));
    float ms_ab = 0.0f, ms_bc = 0.0f, ms_cd = 0.0f;
    CK(cudaEventElapsedTime(&ms_ab, ev_a, ev_b));
    CK(cudaEventElapsedTime(&ms_bc, ev_b, ev_c));
    CK(cudaEventElapsedTime(&ms_cd, ev_c, ev_d));
    stage_ms[0] = ms_ab;
    stage_ms[1] = ms_bc;
    stage_ms[2] = ms_cd;
}

// ============================================================================
// FLOPS helpers
// ============================================================================
static double mla_flops(const MLAShape& s, int qk_eff) {
    // 2 * B * n_h * S² * (qk_eff + d_v)
    return 2.0 * s.B * s.Nh * (double)s.S * s.S * (qk_eff + s.Dv);
}

// ============================================================================
// alloc_padded_qk — copy host f16 of shape [B, Nh, S, QK_true] into device
// buffer [B, Nh, S, QK_eff] with trailing zeros if QK_eff > QK_true.
// Same contract as cuda-attn-mla/attn_mla.cu.
// ============================================================================
static __half* alloc_padded_qk(const __half* host, const MLAShape& sh, int QK_eff) {
    size_t nelem = (size_t)sh.B * sh.Nh * sh.S * QK_eff;
    __half* d;
    CK(cudaMalloc(&d, nelem * sizeof(__half)));
    if (QK_eff == sh.QK) {
        CK(cudaMemcpy(d, host, nelem * sizeof(__half), cudaMemcpyHostToDevice));
    } else {
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

// ============================================================================
// Run correctness on one shape, in one variant (native or padded).
// Returns max_abs_err vs PyTorch SDPA-MLA expected output.
// ============================================================================
static double run_correctness_variant(
    cublasHandle_t cublas, cudaStream_t stream,
    const MLAShape& sh, bool padded,
    const std::string& inputs_dir,
    const char* shape_prefix)
{
    int QK_eff = pad_qk(sh.QK, padded);

    // Load inputs.
    Npy q = load_npy((inputs_dir + "/mla_" + shape_prefix + "_q_f16.npy").c_str());
    Npy k = load_npy((inputs_dir + "/mla_" + shape_prefix + "_k_f16.npy").c_str());
    Npy v = load_npy((inputs_dir + "/mla_" + shape_prefix + "_v_f16.npy").c_str());
    Npy e = load_npy((inputs_dir + "/mla_" + shape_prefix + "_expected_f32.npy").c_str());

    size_t qk_native_elems = (size_t)sh.B * sh.Nh * sh.S * sh.QK;
    size_t v_elems         = (size_t)sh.B * sh.Nh * sh.S * sh.Dv;
    size_t s_elems         = (size_t)sh.B * sh.Nh * sh.S * sh.S;

    if (q.data.size() != qk_native_elems * 2 || k.data.size() != qk_native_elems * 2 ||
        v.data.size() != v_elems * 2) {
        fprintf(stderr, "shape mismatch on %s (qk_native=%zu got_q=%zu got_k=%zu got_v=%zu)\n",
                shape_prefix, qk_native_elems * 2, q.data.size(), k.data.size(), v.data.size());
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

    cudaEvent_t ev_a, ev_b, ev_c, ev_d;
    CK(cudaEventCreate(&ev_a));
    CK(cudaEventCreate(&ev_b));
    CK(cudaEventCreate(&ev_c));
    CK(cudaEventCreate(&ev_d));

    float ms_stage[3];
    run_attention_once(cublas, stream, sh, QK_eff, dQ, dK, dV, dS, dP, dO, ms_stage,
                       ev_a, ev_b, ev_c, ev_d);

    std::vector<__half> ho(v_elems);
    CK(cudaMemcpy(ho.data(), dO, v_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    const float* exp_ptr = reinterpret_cast<const float*>(e.data.data());
    double max_abs = 0.0, max_rel = 0.0;
    double exp_max_abs = 0.0;
    for (size_t i = 0; i < v_elems; ++i) {
        float got  = __half2float(ho[i]);
        float want = exp_ptr[i];
        double a = fabs((double)got - (double)want);
        if (a > max_abs) max_abs = a;
        double aw = fabs((double)want);
        if (aw > exp_max_abs) exp_max_abs = aw;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel) max_rel = r;
    }

    printf("[cublas-mla] %-15s qk=%d (qk_eff=%d, %s) -> max_abs_err=%.3e max_rel=%.3e expected_max_abs=%.3e\n",
           shape_prefix, sh.QK, QK_eff, padded ? "padded" : "native",
           max_abs, max_rel, exp_max_abs);
    printf("[cublas-mla]   stages (warmup-1iter ms): QKt=%.3f softmax=%.3f PV=%.3f total=%.3f\n",
           ms_stage[0], ms_stage[1], ms_stage[2], ms_stage[0] + ms_stage[1] + ms_stage[2]);

    cudaEventDestroy(ev_a); cudaEventDestroy(ev_b);
    cudaEventDestroy(ev_c); cudaEventDestroy(ev_d);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dS); cudaFree(dP); cudaFree(dO);
    return max_abs;
}

// ============================================================================
// run_correctness — small shape, both variants. Wave 17 W2c acceptance:
// max_abs_err ≤ 1e-2 vs PyTorch SDPA-MLA expected, BIT-IDENTICAL between
// native and padded (per ADR-0005 invariant).
// ============================================================================
static int run_correctness(cublasHandle_t cublas, cudaStream_t stream,
                           const std::string& inputs_dir)
{
    MLAShape sh{"correctness_mla", 1, 4, 128, 96, 64};
    printf("\n[cublas-mla] === correctness run (B=%d n_h=%d S=%d qk=%d d_v=%d) ===\n",
           sh.B, sh.Nh, sh.S, sh.QK, sh.Dv);

    const double TOL = 1e-2;

    double err_native = run_correctness_variant(
        cublas, stream, sh, /*padded=*/false, inputs_dir, "correctness_mla");
    double err_padded = run_correctness_variant(
        cublas, stream, sh, /*padded=*/true,  inputs_dir, "correctness_mla");

    bool ok_native = (err_native <= TOL);
    bool ok_padded = (err_padded <= TOL);
    printf("[cublas-mla] correctness: native=%.3e (%s)  padded=%.3e (%s)  TOL=%.0e\n",
           err_native, ok_native ? "OK" : "FAIL",
           err_padded, ok_padded ? "OK" : "FAIL", TOL);
    return (ok_native && ok_padded) ? 0 : 2;
}

// ============================================================================
// Bench setup banner — same author-cell discipline as cuda-attn-mla.
// Allocates device buffers, runs ONE warmup invocation per variant to confirm
// the bench-shape pipeline launches cleanly. Timed iters only when --bench-now.
// ============================================================================
static int run_bench(cublasHandle_t cublas, cudaStream_t stream,
                     const std::string& inputs_dir, bool bench_now)
{
    MLAShape sh{"deepseek_v3", 1, 128, 2048, 192, 128};
    printf("\n[cublas-mla] === bench setup (B=%d n_h=%d S=%d qk=%d d_v=%d) ===\n",
           sh.B, sh.Nh, sh.S, sh.QK, sh.Dv);

    int qk_pad = pad_qk(sh.QK, true);
    double useful_flops = mla_flops(sh, sh.QK);
    double padded_flops = mla_flops(sh, qk_pad);
    printf("[cublas-mla] useful_flops = %.3f GFLOPS  (qk=%d, qk+d_v=%d)\n",
           useful_flops / 1e9, sh.QK, sh.QK + sh.Dv);
    printf("[cublas-mla] padded_flops = %.3f GFLOPS  (qk_pad=%d, padding overhead=%.1f%%)\n",
           padded_flops / 1e9, qk_pad, 100.0 * (padded_flops / useful_flops - 1.0));

    double bytes_qk_native = (double)sh.B * sh.Nh * sh.S * sh.QK   * 2;
    double bytes_qk_padded = (double)sh.B * sh.Nh * sh.S * qk_pad  * 2;
    printf("[cublas-mla] DRAM K-traffic (per pass): native=%.1f MB  padded=%.1f MB  delta=%.1f%%\n",
           bytes_qk_native / 1e6, bytes_qk_padded / 1e6,
           100.0 * (bytes_qk_padded / bytes_qk_native - 1.0));

    if (!bench_now) {
        printf("[cublas-mla] bench-now NOT requested — author-cell mode.\n");
        printf("[cublas-mla] (Pass --bench-now to run timed iters serially under orchestrator control.)\n");
        return 0;
    }

    // ---- Optional: timed bench, only when --bench-now ---------------------
    Npy q = load_npy((inputs_dir + "/mla_deepseek_v3_q_f16.npy").c_str());
    Npy k = load_npy((inputs_dir + "/mla_deepseek_v3_k_f16.npy").c_str());
    Npy v = load_npy((inputs_dir + "/mla_deepseek_v3_v_f16.npy").c_str());

    const __half* hq = reinterpret_cast<const __half*>(q.data.data());
    const __half* hk = reinterpret_cast<const __half*>(k.data.data());

    size_t v_elems = (size_t)sh.B * sh.Nh * sh.S * sh.Dv;
    size_t s_elems = (size_t)sh.B * sh.Nh * sh.S * sh.S;

    FILE* csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,batch,seq,n_h,qk,d_v,qk_eff,iter,gpu_ms,useful_tflops,padded_tflops\n");

    cudaEvent_t ev_a, ev_b, ev_c, ev_d;
    CK(cudaEventCreate(&ev_a));
    CK(cudaEventCreate(&ev_b));
    CK(cudaEventCreate(&ev_c));
    CK(cudaEventCreate(&ev_d));

    constexpr int ITERS = 10;

    for (int variant = 0; variant < 2; ++variant) {
        bool padded = (variant == 1);
        int  QK_eff = pad_qk(sh.QK, padded);
        const char* vname = padded ? "padded" : "native";
        printf("\n[cublas-mla] -- variant=%s qk_eff=%d --\n", vname, QK_eff);

        __half* dQ = alloc_padded_qk(hq, sh, QK_eff);
        __half* dK = alloc_padded_qk(hk, sh, QK_eff);
        __half *dV, *dP, *dO;
        float*  dS;
        CK(cudaMalloc(&dV, v_elems * sizeof(__half)));
        CK(cudaMalloc(&dS, s_elems * sizeof(float)));
        CK(cudaMalloc(&dP, s_elems * sizeof(__half)));
        CK(cudaMalloc(&dO, v_elems * sizeof(__half)));
        CK(cudaMemcpy(dV, v.data.data(), v_elems * sizeof(__half), cudaMemcpyHostToDevice));

        float ms_stage[3];
        // warmup
        run_attention_once(cublas, stream, sh, QK_eff, dQ, dK, dV, dS, dP, dO, ms_stage,
                           ev_a, ev_b, ev_c, ev_d);

        double best = 1e30, sum = 0.0;
        for (int i = 0; i < ITERS; ++i) {
            run_attention_once(cublas, stream, sh, QK_eff, dQ, dK, dV, dS, dP, dO, ms_stage,
                               ev_a, ev_b, ev_c, ev_d);
            double total = ms_stage[0] + ms_stage[1] + ms_stage[2];
            double useful_tf = (useful_flops / 1e12) / (total * 1e-3);
            double padded_tf = (padded_flops / 1e12) / (total * 1e-3);
            printf("[cublas-mla] %s iter=%d total=%.4fms QKt=%.3f softmax=%.3f PV=%.3f useful_TF=%.2f padded_TF=%.2f\n",
                   vname, i, total, ms_stage[0], ms_stage[1], ms_stage[2], useful_tf, padded_tf);
            fprintf(csv, "cublas-attn-mla,3kernel_cublas_%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
                    vname, sh.B, sh.S, sh.Nh, sh.QK, sh.Dv, QK_eff, i, total, useful_tf, padded_tf);
            if (total < best) best = total;
            sum += total;
        }
        double avg = sum / ITERS;
        printf("[cublas-mla] %s SUMMARY best_ms=%.4f avg_ms=%.4f best_useful_TF=%.2f best_padded_TF=%.2f\n",
               vname, best, avg,
               (useful_flops / 1e12) / (best * 1e-3),
               (padded_flops / 1e12) / (best * 1e-3));

        cudaFree(dQ); cudaFree(dK); cudaFree(dV);
        cudaFree(dS); cudaFree(dP); cudaFree(dO);
    }

    cudaEventDestroy(ev_a); cudaEventDestroy(ev_b);
    cudaEventDestroy(ev_c); cudaEventDestroy(ev_d);
    fclose(csv);
    return 0;
}

// ============================================================================
int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cublas-mla] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    cublasHandle_t cublas; CB(cublasCreate(&cublas));
    int ver = 0; CB(cublasGetVersion(cublas, &ver));
    printf("[cublas-mla] cuBLAS version: %d.%d.%d\n",
           ver / 10000, (ver / 100) % 100, ver % 100);
    // CUBLAS_DEFAULT_MATH; tensor cores enabled per-call via CUBLAS_GEMM_DEFAULT_TENSOR_OP.
    // Matches cublas-attn-gqa harness contract.
    CB(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    CB(cublasSetStream(cublas, stream));

    std::string inputs_dir = "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    bool bench_now = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench-now") bench_now = true;
        else if (!a.empty() && a[0] != '-') inputs_dir = a;
    }
    printf("[cublas-mla] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(cublas, stream, inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[cublas-mla] correctness failed — refusing to run bench\n");
        cudaStreamDestroy(stream);
        cublasDestroy(cublas);
        return rc;
    }

    rc = run_bench(cublas, stream, inputs_dir, bench_now);

    cudaStreamDestroy(stream);
    cublasDestroy(cublas);
    return rc;
}
