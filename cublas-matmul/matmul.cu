// cuBLAS sgemm reference baseline: strict f32 (CUBLAS_PEDANTIC_MATH, no TF32)
// for apples-to-apples comparison against naive CUDA / wgpu / cuda-oxide matmuls.
// Wave 2 W2A: size sweep N in {1024, 2048, 4096}, 1 warmup + 10 timed iters.
// Per ADR-0001 (cudaEvent timing), ADR-0002 (sm_120 native), ADR-0003 (PEDANTIC).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAXN 4096
#define WARMUPS 1
#define ITERS 10

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS status %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

// Host reference for spot-check (row-major, stride = n).
static float ref_elem(const float* hA, const float* hB, int n, int row, int col) {
    double acc = 0.0;
    for (int k = 0; k < n; ++k) {
        acc += (double)hA[row * n + k] * (double)hB[k * n + col];
    }
    return (float)acc;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cublas] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    cublasHandle_t h; CB(cublasCreate(&h));
    int ver = 0; CB(cublasGetVersion(h, &ver));
    printf("[cublas] version: %d.%d.%d\n", ver / 10000, (ver / 100) % 100, ver % 100);
    CB(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
    printf("[cublas] math mode: CUBLAS_PEDANTIC_MATH (strict f32, no TF32)\n");

    const int NS[3] = {1024, 2048, 4096};
    const size_t max_bytes = (size_t)MAXN * MAXN * sizeof(float);

    float *hA = (float*)malloc(max_bytes);
    float *hB = (float*)malloc(max_bytes);
    float *hC = (float*)malloc(max_bytes);

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, max_bytes));
    CK(cudaMalloc(&dB, max_bytes));
    CK(cudaMalloc(&dC, max_bytes));

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    FILE* csv = fopen("cublas-matmul/results.csv", "w");
    if (!csv) csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,N,iter,gpu_ms,tflops\n");

    double best_ms[3], median_ms[3], best_tf[3], median_tf[3];
    int total_ok = 0;

    const float alpha = 1.0f, beta = 0.0f;

    for (int ni = 0; ni < 3; ++ni) {
        int n = NS[ni];
        size_t nbytes = (size_t)n * n * sizeof(float);
        double flops = 2.0 * (double)n * n * n;

        // Fill host arrays with stride == n (same pattern as cuda-matmul).
        for (int i = 0; i < n * n; ++i) {
            hA[i] = (i % 7) * 0.01f;
            hB[i] = (i % 11) * 0.01f;
        }
        CK(cudaMemcpy(dA, hA, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(dC, 0, nbytes));

        // Row-major trick: C = A*B (row-major) == (B^T * A^T)^T in column-major.
        // Pass dB first (as "A" to cuBLAS) then dA (as "B"); result in dC is our
        // row-major C. ld = n for all three.
        auto sgemm = [&]() {
            return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                               n, n, n, &alpha,
                               dB, n,
                               dA, n,
                               &beta,
                               dC, n);
        };

        printf("[cublas] N=%d sgemm f32, %.2f GFLOP/iter\n", n, flops / 1e9);

        // Warmup (absorbs autotune / cuBLAS first-call setup).
        for (int w = 0; w < WARMUPS; ++w) {
            CB(sgemm());
        }
        CK(cudaDeviceSynchronize());

        double times[ITERS];
        float ms;
        for (int i = 0; i < ITERS; ++i) {
            cudaEventRecord(evs);
            CB(sgemm());
            cudaEventRecord(eve);
            cudaEventSynchronize(eve);
            cudaEventElapsedTime(&ms, evs, eve);
            times[i] = ms;
            double tflops = (flops / 1e12) / (ms / 1000.0);
            printf("[cublas] N=%d iter=%d gpu_ms=%.3f tflops=%.3f\n", n, i, ms, tflops);
            fprintf(csv, "cublas-matmul,sgemm,%d,%d,%.6f,%.6f\n", n, i, ms, tflops);
        }

        // Correctness spot-check at (0,0), (n/2, n/2), (n-1, n-1).
        CK(cudaMemcpy(hC, dC, nbytes, cudaMemcpyDeviceToHost));
        int pts[3][2] = { {0, 0}, {n / 2, n / 2}, {n - 1, n - 1} };
        int ok = 0;
        for (int pi = 0; pi < 3; ++pi) {
            int r = pts[pi][0], c = pts[pi][1];
            float got = hC[r * n + c];
            float want = ref_elem(hA, hB, n, r, c);
            float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
            const char* tag = (rel < 1e-3f) ? "OK" : "FAIL";
            if (rel < 1e-3f) { ok++; total_ok++; }
            printf("[cublas] N=%d check (%d,%d): got=%.4f want=%.4f rel=%.3e %s\n",
                   n, r, c, got, want, rel, tag);
        }
        printf("[cublas] N=%d correctness: %d/3 OK\n", n, ok);

        // Sort times for median.
        double sorted[ITERS];
        for (int i = 0; i < ITERS; ++i) sorted[i] = times[i];
        for (int i = 0; i < ITERS; ++i)
            for (int j = i + 1; j < ITERS; ++j)
                if (sorted[j] < sorted[i]) {
                    double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                }
        double bst = sorted[0];
        double med = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]);
        best_ms[ni] = bst;
        median_ms[ni] = med;
        best_tf[ni] = (flops / 1e12) / (bst / 1000.0);
        median_tf[ni] = (flops / 1e12) / (med / 1000.0);
    }

    printf("\n[cublas] ===== SUMMARY =====\n");
    printf("[cublas] %6s  %10s  %10s  %10s  %10s\n", "N", "best_ms", "median_ms", "best_TF", "median_TF");
    for (int ni = 0; ni < 3; ++ni) {
        int n = NS[ni];
        printf("[cublas] %6d  %10.3f  %10.3f  %10.3f  %10.3f\n",
               n, best_ms[ni], median_ms[ni], best_tf[ni], median_tf[ni]);
    }
    printf("[cublas] correctness: %d/9 OK\n", total_ok);

    fclose(csv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cublasDestroy(h);
    return 0;
}
