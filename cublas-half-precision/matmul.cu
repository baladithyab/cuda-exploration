// Wave 14.1: cuBLAS half-precision baselines (hgemm, bgemm, sgemm-tf32).
// Apples-to-apples reference for cuTile mixed-precision Wave 13.1 numbers:
//   cuTile f16  @ N=4096: 172.5 TFLOPS
//   cuTile bf16 @ N=4096: 159.8 TFLOPS
//   cuTile tf32 @ N=4096:  84.0 TFLOPS
//
// Size sweep N in {1024, 2048, 4096}, 1 warmup + 10 timed iters.
// Mirrors cublas-matmul/matmul.cu structure exactly; only the kernels change.
// ADR-0001 (cudaEvent timing), ADR-0002 (sm_120 native).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define MAXN 4096
#define WARMUPS 1
#define ITERS 10

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS status %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

// Host reference for spot-check (double accumulator, row-major, stride = n).
static float ref_elem(const float* hA, const float* hB, int n, int row, int col) {
    double acc = 0.0;
    for (int k = 0; k < n; ++k) {
        acc += (double)hA[row * n + k] * (double)hB[k * n + col];
    }
    return (float)acc;
}

static void sort_times(double* t, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (t[j] < t[i]) { double x = t[i]; t[i] = t[j]; t[j] = x; }
}

struct BenchStat {
    double best_ms[3], median_ms[3], best_tf[3], median_tf[3];
    int ok_count;
};

// --- Bench kernel variants ---------------------------------------------------
// All use row-major A,B,C with the (B,A) column-major trick: cuBLAS thinks it
// is computing C_col = B_col * A_col; we interpret C as row-major C = A*B.
// ld = n for all three.

enum Variant { V_HGEMM, V_BGEMM, V_SGEMM_TF32 };
static const char* variant_name(Variant v) {
    switch (v) {
        case V_HGEMM:      return "hgemm";
        case V_BGEMM:      return "bgemm";
        case V_SGEMM_TF32: return "sgemm_tf32";
    }
    return "?";
}
static const char* variant_desc(Variant v) {
    switch (v) {
        case V_HGEMM:      return "f16 in/out, f32 accum (cublasGemmEx TENSOR_OP)";
        case V_BGEMM:      return "bf16 in/out, f32 accum (cublasGemmEx TENSOR_OP)";
        case V_SGEMM_TF32: return "f32 in/out, TF32 internal (cublasSgemm + TF32_TENSOR_OP_MATH)";
    }
    return "?";
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cublas-hp] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    cublasHandle_t h; CB(cublasCreate(&h));
    int ver = 0; CB(cublasGetVersion(h, &ver));
    printf("[cublas-hp] version: %d.%d.%d\n", ver / 10000, (ver / 100) % 100, ver % 100);

    const int NS[3] = {1024, 2048, 4096};
    const size_t max_elems = (size_t)MAXN * MAXN;

    // Host arrays (f32 reference, used to fill + for spot-check ground truth).
    float *hA = (float*)malloc(max_elems * sizeof(float));
    float *hB = (float*)malloc(max_elems * sizeof(float));
    float *hC = (float*)malloc(max_elems * sizeof(float));

    // Typed staging buffers (host-side).
    __half *hA_h = (__half*)malloc(max_elems * sizeof(__half));
    __half *hB_h = (__half*)malloc(max_elems * sizeof(__half));
    __half *hC_h = (__half*)malloc(max_elems * sizeof(__half));
    __nv_bfloat16 *hA_b = (__nv_bfloat16*)malloc(max_elems * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hB_b = (__nv_bfloat16*)malloc(max_elems * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hC_b = (__nv_bfloat16*)malloc(max_elems * sizeof(__nv_bfloat16));

    // Device buffers: one pair per dtype class (alloc max size once).
    __half *dA_h, *dB_h, *dC_h;
    __nv_bfloat16 *dA_b, *dB_b, *dC_b;
    float *dA_f, *dB_f, *dC_f;
    CK(cudaMalloc(&dA_h, max_elems * sizeof(__half)));
    CK(cudaMalloc(&dB_h, max_elems * sizeof(__half)));
    CK(cudaMalloc(&dC_h, max_elems * sizeof(__half)));
    CK(cudaMalloc(&dA_b, max_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dB_b, max_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dC_b, max_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dA_f, max_elems * sizeof(float)));
    CK(cudaMalloc(&dB_f, max_elems * sizeof(float)));
    CK(cudaMalloc(&dC_f, max_elems * sizeof(float)));

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    FILE* csv = fopen("cublas-half-precision/results.csv", "w");
    if (!csv) csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,N,iter,gpu_ms,tflops\n");

    BenchStat stats[3];
    for (int v = 0; v < 3; ++v) stats[v].ok_count = 0;

    const float alpha_f = 1.0f, beta_f = 0.0f;

    for (int vi = 0; vi < 3; ++vi) {
        Variant var = (Variant)vi;
        printf("\n[cublas-hp] ===== variant: %s (%s) =====\n",
               variant_name(var), variant_desc(var));

        // Set math mode appropriate to variant.
        if (var == V_SGEMM_TF32) {
            CB(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
            printf("[cublas-hp] math mode: CUBLAS_TF32_TENSOR_OP_MATH\n");
        } else {
            // DEFAULT_MATH on modern cuBLAS permits tensor-core use for f16/bf16.
            // Also explicitly set TENSOR_OP_MATH (deprecated name mapped to default).
            CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
            printf("[cublas-hp] math mode: CUBLAS_DEFAULT_MATH (tensor cores enabled via GemmEx algo)\n");
        }

        for (int ni = 0; ni < 3; ++ni) {
            int n = NS[ni];
            size_t nelem = (size_t)n * n;
            double flops = 2.0 * (double)n * n * n;

            // Fill host f32 reference arrays with same pattern as cublas-matmul.
            for (int i = 0; i < n * n; ++i) {
                hA[i] = (i % 7) * 0.01f;
                hB[i] = (i % 11) * 0.01f;
            }

            // Convert + copy to device per variant.
            if (var == V_HGEMM) {
                for (size_t i = 0; i < nelem; ++i) {
                    hA_h[i] = __float2half(hA[i]);
                    hB_h[i] = __float2half(hB[i]);
                }
                CK(cudaMemcpy(dA_h, hA_h, nelem * sizeof(__half), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dB_h, hB_h, nelem * sizeof(__half), cudaMemcpyHostToDevice));
                CK(cudaMemset(dC_h, 0, nelem * sizeof(__half)));
            } else if (var == V_BGEMM) {
                for (size_t i = 0; i < nelem; ++i) {
                    hA_b[i] = __float2bfloat16(hA[i]);
                    hB_b[i] = __float2bfloat16(hB[i]);
                }
                CK(cudaMemcpy(dA_b, hA_b, nelem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dB_b, hB_b, nelem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
                CK(cudaMemset(dC_b, 0, nelem * sizeof(__nv_bfloat16)));
            } else { // V_SGEMM_TF32
                CK(cudaMemcpy(dA_f, hA, nelem * sizeof(float), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dB_f, hB, nelem * sizeof(float), cudaMemcpyHostToDevice));
                CK(cudaMemset(dC_f, 0, nelem * sizeof(float)));
            }

            // Kernel launcher. Uses (B,A) swap so output is row-major C = A*B.
            auto launch = [&]() -> cublasStatus_t {
                if (var == V_HGEMM) {
                    return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n, n,
                        &alpha_f,
                        dB_h, CUDA_R_16F, n,
                        dA_h, CUDA_R_16F, n,
                        &beta_f,
                        dC_h, CUDA_R_16F, n,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                } else if (var == V_BGEMM) {
                    return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n, n,
                        &alpha_f,
                        dB_b, CUDA_R_16BF, n,
                        dA_b, CUDA_R_16BF, n,
                        &beta_f,
                        dC_b, CUDA_R_16BF, n,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                } else { // V_SGEMM_TF32
                    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n, n, &alpha_f,
                        dB_f, n,
                        dA_f, n,
                        &beta_f,
                        dC_f, n);
                }
            };

            printf("[cublas-hp] N=%d %s, %.2f GFLOP/iter\n",
                   n, variant_name(var), flops / 1e9);

            // Warmup.
            for (int w = 0; w < WARMUPS; ++w) { CB(launch()); }
            CK(cudaDeviceSynchronize());

            double times[ITERS];
            float ms;
            for (int i = 0; i < ITERS; ++i) {
                cudaEventRecord(evs);
                CB(launch());
                cudaEventRecord(eve);
                cudaEventSynchronize(eve);
                cudaEventElapsedTime(&ms, evs, eve);
                times[i] = ms;
                double tflops = (flops / 1e12) / (ms / 1000.0);
                printf("[cublas-hp] N=%d %s iter=%d gpu_ms=%.3f tflops=%.3f\n",
                       n, variant_name(var), i, ms, tflops);
                fprintf(csv, "cublas-half-precision,%s,%d,%d,%.6f,%.6f\n",
                        variant_name(var), n, i, ms, tflops);
            }

            // Correctness spot-check at (0,0), (n/2, n/2), (n-1, n-1).
            int pts[3][2] = { {0, 0}, {n / 2, n / 2}, {n - 1, n - 1} };
            int ok = 0;
            // Tolerance per dtype class.
            float tol = (var == V_SGEMM_TF32) ? 1e-3f : ((var == V_BGEMM) ? 5e-2f : 1e-2f);

            if (var == V_HGEMM) {
                CK(cudaMemcpy(hC_h, dC_h, nelem * sizeof(__half), cudaMemcpyDeviceToHost));
                for (int pi = 0; pi < 3; ++pi) {
                    int r = pts[pi][0], c = pts[pi][1];
                    float got = __half2float(hC_h[r * n + c]);
                    float want = ref_elem(hA, hB, n, r, c);
                    float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
                    const char* tag = (rel < tol) ? "OK" : "FAIL";
                    if (rel < tol) { ok++; stats[vi].ok_count++; }
                    printf("[cublas-hp] N=%d %s check (%d,%d): got=%.4f want=%.4f rel=%.3e %s\n",
                           n, variant_name(var), r, c, got, want, rel, tag);
                }
            } else if (var == V_BGEMM) {
                CK(cudaMemcpy(hC_b, dC_b, nelem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                for (int pi = 0; pi < 3; ++pi) {
                    int r = pts[pi][0], c = pts[pi][1];
                    float got = __bfloat162float(hC_b[r * n + c]);
                    float want = ref_elem(hA, hB, n, r, c);
                    float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
                    const char* tag = (rel < tol) ? "OK" : "FAIL";
                    if (rel < tol) { ok++; stats[vi].ok_count++; }
                    printf("[cublas-hp] N=%d %s check (%d,%d): got=%.4f want=%.4f rel=%.3e %s\n",
                           n, variant_name(var), r, c, got, want, rel, tag);
                }
            } else {
                CK(cudaMemcpy(hC, dC_f, nelem * sizeof(float), cudaMemcpyDeviceToHost));
                for (int pi = 0; pi < 3; ++pi) {
                    int r = pts[pi][0], c = pts[pi][1];
                    float got = hC[r * n + c];
                    float want = ref_elem(hA, hB, n, r, c);
                    float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
                    const char* tag = (rel < tol) ? "OK" : "FAIL";
                    if (rel < tol) { ok++; stats[vi].ok_count++; }
                    printf("[cublas-hp] N=%d %s check (%d,%d): got=%.4f want=%.4f rel=%.3e %s\n",
                           n, variant_name(var), r, c, got, want, rel, tag);
                }
            }
            printf("[cublas-hp] N=%d %s correctness: %d/3 OK (tol=%.0e)\n",
                   n, variant_name(var), ok, tol);

            // Median / best.
            double sorted[ITERS];
            for (int i = 0; i < ITERS; ++i) sorted[i] = times[i];
            sort_times(sorted, ITERS);
            double bst = sorted[0];
            double med = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]);
            stats[vi].best_ms[ni] = bst;
            stats[vi].median_ms[ni] = med;
            stats[vi].best_tf[ni] = (flops / 1e12) / (bst / 1000.0);
            stats[vi].median_tf[ni] = (flops / 1e12) / (med / 1000.0);
        }
    }

    // --- Summary ---
    printf("\n[cublas-hp] ===== SUMMARY =====\n");
    for (int vi = 0; vi < 3; ++vi) {
        Variant var = (Variant)vi;
        printf("[cublas-hp] --- %s (%s) ---\n", variant_name(var), variant_desc(var));
        printf("[cublas-hp] %6s  %10s  %10s  %10s  %10s\n",
               "N", "best_ms", "median_ms", "best_TF", "median_TF");
        for (int ni = 0; ni < 3; ++ni) {
            int n = NS[ni];
            printf("[cublas-hp] %6d  %10.3f  %10.3f  %10.3f  %10.3f\n",
                   n, stats[vi].best_ms[ni], stats[vi].median_ms[ni],
                   stats[vi].best_tf[ni], stats[vi].median_tf[ni]);
        }
        printf("[cublas-hp] %s correctness: %d/9 OK\n", variant_name(var), stats[vi].ok_count);
    }

    // Headline apples-to-apples table (best TFLOPS @ N=4096).
    printf("\n[cublas-hp] ===== N=4096 apples-to-apples (best TFLOPS) =====\n");
    printf("[cublas-hp] cublas hgemm       : %8.2f TF\n", stats[V_HGEMM].best_tf[2]);
    printf("[cublas-hp] cublas bgemm       : %8.2f TF\n", stats[V_BGEMM].best_tf[2]);
    printf("[cublas-hp] cublas sgemm-tf32  : %8.2f TF\n", stats[V_SGEMM_TF32].best_tf[2]);
    printf("[cublas-hp] cuTile f16 (W13.1) :   172.50 TF\n");
    printf("[cublas-hp] cuTile bf16 (W13.1):   159.80 TF\n");
    printf("[cublas-hp] cuTile tf32 (W13.1):    84.00 TF\n");
    printf("[cublas-hp] cuTile/cublas hgemm: %6.1f%%\n",
           100.0 * 172.5 / stats[V_HGEMM].best_tf[2]);
    printf("[cublas-hp] cuTile/cublas bgemm: %6.1f%%\n",
           100.0 * 159.8 / stats[V_BGEMM].best_tf[2]);
    printf("[cublas-hp] cuTile/cublas tf32 : %6.1f%%\n",
           100.0 * 84.0 / stats[V_SGEMM_TF32].best_tf[2]);

    fclose(csv);
    cudaFree(dA_h); cudaFree(dB_h); cudaFree(dC_h);
    cudaFree(dA_b); cudaFree(dB_b); cudaFree(dC_b);
    cudaFree(dA_f); cudaFree(dB_f); cudaFree(dC_f);
    free(hA); free(hB); free(hC);
    free(hA_h); free(hB_h); free(hC_h);
    free(hA_b); free(hB_b); free(hC_b);
    cublasDestroy(h);
    return 0;
}
