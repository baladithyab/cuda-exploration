// Wave 4 W4B: memory-bandwidth reference kernel.
// c[i] = a[i] + b[i] — 3-buffer streaming (read A, read B, write C).
// Block 256 threads; cudaEvent timing; 1 warmup + 10 timed iters per N.
// Sweeps N ∈ {1M, 16M, 64M, 256M} f32 to find the bandwidth-bound regime.
// Per-iter GB/s = (3 * N * 4) / (gpu_ms / 1000) / 1e9.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define BS 256
#define WARMUPS 1
#define ITERS 10

__global__ void vecadd(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    const size_t NS[4] = {
        1ull * 1024 * 1024,
        16ull * 1024 * 1024,
        64ull * 1024 * 1024,
        256ull * 1024 * 1024
    };
    const int NN = 4;
    const size_t NMAX = NS[NN - 1];
    const size_t max_bytes = NMAX * sizeof(float);

    float *hA = (float*)malloc(max_bytes);
    float *hB = (float*)malloc(max_bytes);
    float *hC = (float*)malloc(max_bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }

    // Fill host arrays once at max size; kernel only reads the first n elements.
    for (size_t i = 0; i < NMAX; ++i) {
        hA[i] = (i % 7) * 0.01f;
        hB[i] = (i % 11) * 0.01f;
    }

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, max_bytes));
    CK(cudaMalloc(&dB, max_bytes));
    CK(cudaMalloc(&dC, max_bytes));
    CK(cudaMemcpy(dA, hA, max_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, max_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    FILE* csv = fopen("cuda-vecadd-bench/results.csv", "w");
    if (!csv) csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,N_elems,iter,gpu_ms,GB_per_s\n");

    double best_ms[4], median_ms[4], best_gb[4], median_gb[4];

    for (int ni = 0; ni < NN; ++ni) {
        size_t n = NS[ni];
        double traffic_bytes = 3.0 * (double)n * 4.0;
        size_t nbytes = n * sizeof(float);

        CK(cudaMemset(dC, 0, nbytes));

        int grid = (int)((n + BS - 1) / BS);
        printf("[cuda] N=%zu vecadd f32, traffic=%.2f GB/iter\n", n, traffic_bytes / 1e9);

        for (int w = 0; w < WARMUPS; ++w) {
            vecadd<<<grid, BS>>>(dA, dB, dC, (int)n);
        }
        CK(cudaDeviceSynchronize());

        double times[ITERS];
        float ms;
        for (int i = 0; i < ITERS; ++i) {
            cudaEventRecord(evs);
            vecadd<<<grid, BS>>>(dA, dB, dC, (int)n);
            cudaEventRecord(eve);
            cudaEventSynchronize(eve);
            cudaEventElapsedTime(&ms, evs, eve);
            times[i] = ms;
            double gbps = (traffic_bytes / 1e9) / (ms / 1000.0);
            printf("[cuda] N=%zu iter=%d gpu_ms=%.4f GB/s=%.2f\n", n, i, ms, gbps);
            fprintf(csv, "cuda,vecadd,%zu,%d,%.6f,%.6f\n", n, i, ms, gbps);
        }

        // Correctness at 0, n/2, n-1
        size_t idxs[3] = {0, n / 2, n - 1};
        int ok = 0;
        for (int pi = 0; pi < 3; ++pi) {
            size_t idx = idxs[pi];
            float got;
            CK(cudaMemcpy(&got, dC + idx, sizeof(float), cudaMemcpyDeviceToHost));
            float want = hA[idx] + hB[idx];
            float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
            const char* tag = (rel < 1e-3f) ? "OK" : "FAIL";
            if (rel < 1e-3f) ok++;
            printf("[cuda] N=%zu check idx=%zu got=%.4f want=%.4f rel=%.3e %s\n",
                   n, idx, got, want, rel, tag);
        }
        printf("[cuda] N=%zu correctness: %d/3 OK\n", n, ok);

        double sorted[ITERS];
        memcpy(sorted, times, sizeof(sorted));
        for (int i = 0; i < ITERS; ++i)
            for (int j = i + 1; j < ITERS; ++j)
                if (sorted[j] < sorted[i]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
        double bst = sorted[0];
        double med = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]);
        best_ms[ni] = bst;
        median_ms[ni] = med;
        best_gb[ni] = (traffic_bytes / 1e9) / (bst / 1000.0);
        median_gb[ni] = (traffic_bytes / 1e9) / (med / 1000.0);
    }

    printf("\n[cuda] ===== SUMMARY =====\n");
    printf("[cuda] %10s  %10s  %10s  %10s  %10s\n", "N", "best_ms", "med_ms", "best_GBs", "med_GBs");
    for (int ni = 0; ni < NN; ++ni) {
        printf("[cuda] %10zu  %10.4f  %10.4f  %10.2f  %10.2f\n",
               NS[ni], best_ms[ni], median_ms[ni], best_gb[ni], median_gb[ni]);
    }

    fclose(csv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
