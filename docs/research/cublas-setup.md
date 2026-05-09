# cuBLAS sgemm reference baseline — research

Toolkit on this box: **CUDA 13.2** (`nvcc V13.2.78`), cuBLAS **v13.4.0.1**.
(Context said 12.0 but `/usr/local/cuda` is 13.2; code below is API-compatible
back to CUDA 11.)

## Linking

cuBLAS ships inside the CUDA toolkit — no extra install needed. Verified:

```bash
$ ls /usr/local/cuda/lib64/libcublas*
/usr/local/cuda/lib64/libcublas.so
/usr/local/cuda/lib64/libcublas.so.13
/usr/local/cuda/lib64/libcublas.so.13.4.0.1
/usr/local/cuda/lib64/libcublasLt.so
/usr/local/cuda/lib64/libcublasLt.so.13
/usr/local/cuda/lib64/libcublasLt.so.13.4.0.1
/usr/local/cuda/lib64/libcublasLt_static.a
/usr/local/cuda/lib64/libcublas_static.a

$ ls /usr/local/cuda/include/cublas*
/usr/local/cuda/include/cublas.h          # legacy API (avoid)
/usr/local/cuda/include/cublasLt.h        # cuBLASLt (lightweight, new API)
/usr/local/cuda/include/cublasXt.h        # multi-GPU
/usr/local/cuda/include/cublas_api.h      # internal, pulled in by cublas_v2.h
/usr/local/cuda/include/cublas_v2.h       # modern handle-based API  <-- we use this
```

Libraries: `libcublas` holds the classic BLAS (sgemm etc.) — this is what we
need. `libcublasLt` is a lower-level API used internally; don't call it
directly. `libcublas_static.a` exists for static builds but dynamic is fine.

Invocation:

```bash
nvcc -O3 -arch=sm_120 -std=c++17 matmul.cu -lcublas -o matmul
```

Notes:
- `-lcublas` is enough. nvcc automatically adds `-L/usr/local/cuda/lib64` and
  `-I/usr/local/cuda/include`, so no explicit `-L` / `-I` needed. When linking
  with plain `g++` instead, use `-L/usr/local/cuda/lib64 -lcublas -lcudart`.
- `sm_120` targets RTX 5090 (Blackwell GB202). Fall back to `-arch=sm_90` if
  the toolchain complains.
- Runtime: the loader needs `libcublas.so.13`. `ldconfig` covers
  `/usr/local/cuda/lib64` on standard installs; otherwise
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`.

**If cuBLAS were missing** (it isn't here): Ubuntu
`sudo apt install libcublas-dev-13-2`, RHEL `sudo dnf install libcublas-devel-cuda-13-x`.
The toolkit meta-package (`cuda-toolkit-13-2`) already pulls it in.

## API call shape

Include the v2 header and use the handle-based API:

```c
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,  // OP_N / OP_T / OP_C
    int m, int n, int k,        // op(A): m×k, op(B): k×n, C: m×n
    const float *alpha,         // pointer (host by default)
    const float *A, int lda,    // column-major, leading dim lda
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);
```

Math: `C = alpha * op(A) * op(B) + beta * C`, all column-major. Returns
`cublasStatus_t`; wrap every call in an error check.

## Row-major workaround

Our inputs are **row-major N×N float arrays** (matching `cuda-matmul/matmul.cu`).
cuBLAS is **column-major**. Physically transposing would cost a full matmul's
worth of bandwidth. Use the identity

```
(A · B)^T = B^T · A^T
```

A row-major matrix `M` with leading dim `ld` has the exact same byte layout as
its transpose viewed column-major with the same `ld`. So: passing row-major
`A`, `B` to cuBLAS with `OP_N, OP_N` makes cuBLAS see `A^T`, `B^T`. Asking it
to compute `B^T · A^T = (A·B)^T` yields a column-major buffer whose bytes
equal `A·B` in row-major. Read it back as row-major: correct.

Concretely, for `C = A · B` with all three matrices row-major, `m = n = k = N`:

```c
const float alpha = 1.0f, beta = 0.0f;
cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N /*m = cols of C, cols of B*/,
            N /*n = rows of C, rows of A*/,
            N /*k*/,
            &alpha,
            dB, N,    // "A" in cuBLAS's view is our B
            dA, N,    // "B" in cuBLAS's view is our A
            &beta,
            dC, N);
```

Pattern generalizes: for row-major `A` (M×K) · `B` (K×N) → row-major `C` (M×N),
call `cublasSgemm(h, OP_N, OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N)`.
Leading dims = number of columns of each row-major matrix.

## Initialization sequence

Minimal working skeleton (error handling elided):

```c
cublasHandle_t h;
cublasCreate(&h);                          // 1. create handle

float *dA, *dB, *dC;                       // 2. allocate device buffers
cudaMalloc(&dA, bytes);
cudaMalloc(&dB, bytes);
cudaMalloc(&dC, bytes);

cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);   // 3. H2D copies
cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

const float alpha = 1.0f, beta = 0.0f;               // 4. sgemm
cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);

cudaDeviceSynchronize();                             // 5. sync

cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);   // 6. D2H

cudaFree(dA); cudaFree(dB); cudaFree(dC);            // 7. cleanup
cublasDestroy(h);
```

The `cublasCreate` call is expensive (workspace alloc, device probe, JIT). Do
it once, outside the timed region. Pointer-mode defaults to
`CUBLAS_POINTER_MODE_HOST` so stack-allocated `&alpha` / `&beta` work.

## Timing

cuBLAS calls are **asynchronous** — `cublasSgemm` returns as soon as the
kernel is launched. Use CUDA events around each call and sync on the end
event before reading elapsed time.

```c
cudaEvent_t evs, eve;
cudaEventCreate(&evs); cudaEventCreate(&eve);

// warmup (not timed, but do sync once so JIT/autotune is out of the way)
cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, dB, N, dA, N, &beta, dC, N);
cudaDeviceSynchronize();

float ms;
for (int i = 0; i < 5; ++i) {
    cudaEventRecord(evs);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, dB, N, dA, N, &beta, dC, N);
    cudaEventRecord(eve);
    cudaEventSynchronize(eve);
    cudaEventElapsedTime(&ms, evs, eve);
    // record ms
}
```

Same 1 warmup + 5 timed iters as `cuda-matmul`. cuBLAS may autotune on the
first call for a given shape; warmup absorbs that.

## Expected performance

Target GPU: RTX 5090 (Blackwell, GB202).

- **FP32 SoL (no Tensor Cores):** ~104 TFLOPS.
- **cuBLAS sgemm realization:** 60–90 % of SoL at 4096² → ~60–95 TFLOPS.
- **cuda-matmul naive kernel:** a few TFLOPS (reference lower bound).
- **TF32 Tensor Core path on Blackwell:** ~200–400 TFLOPS. If our "sgemm"
  run exceeds ~150 TFLOPS, TF32 leaked in — see *Math mode*.

~70 TFLOPS on 4096² sgemm is in-band. Report best + median over the 5 iters.

## Version detection

Log the cuBLAS version at startup so benchmark dumps are self-describing.

```c
int v = 0;
cublasGetVersion(h, &v);   // handle-qualified variant; works in v2 API
// v is encoded as MAJOR*10000 + MINOR*100 + PATCH
printf("[cublas] version: %d.%d.%d\n", v/10000, (v/100)%100, v%100);
```

On this box it prints `13.4.0` (shared object is `libcublas.so.13.4.0.1`). A
standalone `cublasGetProperty(libraryPropertyType, int*)` with `MAJOR_VERSION`
/ `MINOR_VERSION` / `PATCH_LEVEL` also works; we use `cublasGetVersion`
because it's a single call.

## Math mode (TF32 vs strict f32)

On Ampere+ (and very much on Blackwell), cuBLAS may downcast f32 inputs to
**TF32** (mantissa 23→10) and run on Tensor Cores for a large speedup. This
is controlled per-handle by `cublasSetMathMode`:

- `CUBLAS_DEFAULT_MATH` — library may use TF32 / Tensor Cores for f32.
- `CUBLAS_TF32_TENSOR_OP_MATH` — explicitly allow TF32 Tensor Core path.
- `CUBLAS_PEDANTIC_MATH` — no downcasting; IEEE-754 f32 multiply + accumulate.
- `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` — flag: forbid
  reduced-precision accumulate.

For this benchmark we want the **strict f32 path** so that the number we
report compares apples-to-apples with our wgpu and cuda-oxide naive kernels
(both of which do plain f32 FMA). We therefore set

```c
cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
```

Rationale: `CUBLAS_DEFAULT_MATH` policy has shifted between CUDA versions and
on Blackwell will likely pick TF32 silently for f32 GEMMs, giving a 2–4×
number that isn't apples-to-apples. `CUBLAS_PEDANTIC_MATH` guarantees no
precision downgrade. We log the mode so the result is self-describing.

(To later benchmark the Tensor Core path on purpose, flip this to
`CUBLAS_TF32_TENSOR_OP_MATH` and label the row "cublas-sgemm-tf32".)

## Complete sample matmul.cu

Drop-in `cublas-matmul/matmul.cu` — mirrors `cuda-matmul/matmul.cu` (same
input pattern, 1 warmup + 5 timed iters, best/median report) plus the
requested spot-checks.

```cpp
// cuBLAS reference: strict f32 sgemm (TF32 disabled) for apples-to-apples
// comparison against naive CUDA / wgpu / cuda-oxide matmuls. 4096x4096 f32.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 4096

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS status %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cublas] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    cublasHandle_t h; CB(cublasCreate(&h));
    int ver = 0; CB(cublasGetVersion(h, &ver));
    printf("[cublas] version: %d.%d.%d\n", ver/10000, (ver/100)%100, ver%100);
    CB(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
    printf("[cublas] math mode: CUBLAS_PEDANTIC_MATH (strict f32, no TF32)\n");

    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N*N; ++i) { hA[i] = (i % 7) * 0.01f; hB[i] = (i % 11) * 0.01f; }

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
    CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    // Row-major trick: compute C = A*B (row-major) by asking cuBLAS for
    // (A*B)^T = B^T * A^T in column-major, with A,B,C passed as-is.
    auto sgemm = [&]() {
        return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, N, N, &alpha,
                           dB, N,
                           dA, N,
                           &beta,
                           dC, N);
    };

    cudaEvent_t evs, eve; CK(cudaEventCreate(&evs)); CK(cudaEventCreate(&eve));
    double total_flops = 2.0 * (double)N * N * N;
    printf("[cublas] sgemm %dx%d f32, %.2f GFLOP/iter\n", N, N, total_flops/1e9);

    // warmup (absorbs autotune / JIT)
    CB(sgemm()); CK(cudaDeviceSynchronize());
    float ms;
    cudaEventRecord(evs); CB(sgemm()); cudaEventRecord(eve);
    cudaEventSynchronize(eve); cudaEventElapsedTime(&ms, evs, eve);
    printf("[cublas] warmup: %.2f ms (%.3f TFLOPS)\n", ms, (total_flops/1e12)/(ms/1000.0));

    double best = 1e30, times[5];
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(evs);
        CB(sgemm());
        cudaEventRecord(eve); cudaEventSynchronize(eve);
        cudaEventElapsedTime(&ms, evs, eve);
        times[i] = ms;
        if (ms < best) best = ms;
        printf("[cublas] iter %d: %.2f ms (%.3f TFLOPS)\n", i, ms, (total_flops/1e12)/(ms/1000.0));
    }
    for (int i = 0; i < 5; ++i) for (int j = i+1; j < 5; ++j)
        if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    double median = times[2];
    printf("\n[cublas] BEST   %.2f ms  %.3f TFLOPS\n", best, (total_flops/1e12)/(best/1000.0));
    printf("[cublas] MEDIAN %.2f ms  %.3f TFLOPS\n", median, (total_flops/1e12)/(median/1000.0));

    // Correctness spot-check at (0,0), (100,100), (4095,4095) — row-major index row*N + col.
    CK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    auto ref = [&](int r, int c) {
        double acc = 0.0;
        for (int k = 0; k < N; ++k) acc += (double)hA[r*N + k] * (double)hB[k*N + c];
        return acc;
    };
    int rs[3] = {0, 100, 4095}, cs[3] = {0, 100, 4095};
    for (int i = 0; i < 3; ++i) {
        int r = rs[i], c = cs[i];
        double expect = ref(r, c);
        double got = hC[r*N + c];
        double rel = fabs(got - expect) / fmax(1e-9, fabs(expect));
        printf("[cublas] spot (%4d,%4d): got=%.4f expect=%.4f relerr=%.2e %s\n",
               r, c, got, expect, rel, rel < 1e-3 ? "OK" : "FAIL");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cublasDestroy(h);
    return 0;
}
```

Build & run:

```bash
nvcc -O3 -arch=sm_120 -std=c++17 cublas-matmul/matmul.cu -lcublas -o cublas-matmul/matmul
./cublas-matmul/matmul
```

Expected log (illustrative numbers):

```
[cublas] device: NVIDIA GeForce RTX 5090 (sm_120)
[cublas] version: 13.4.0
[cublas] math mode: CUBLAS_PEDANTIC_MATH (strict f32, no TF32)
[cublas] sgemm 4096x4096 f32, 137.44 GFLOP/iter
[cublas] warmup: ~2 ms (~65-90 TFLOPS)
...
[cublas] BEST   ~1.8 ms   ~75 TFLOPS
[cublas] spot (   0,   0): ... OK
[cublas] spot ( 100, 100): ... OK
[cublas] spot (4095,4095): ... OK
```

If TFLOPS > ~150, TF32 leaked in — re-check that `cublasSetMathMode(h,
CUBLAS_PEDANTIC_MATH)` runs before the first sgemm.
