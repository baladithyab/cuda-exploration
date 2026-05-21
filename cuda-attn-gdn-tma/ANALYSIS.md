# Wave 22.10 — cuda-attn-gdn-tma (TMA via cp.async.bulk.tensor)

**Scope.** Author + correctness only. NO timed benches in this cell — the
orchestrator runs `./bench` separately on idle GPU.

## Hypothesis under test

W22.8 SASS analysis (`docs/research/wave17-w1c-tma-vs-ldg128-investigation.md`)
showed both W1c (nvcc) and cuTile emit zero `UTMALDG`. cuTile reaches 610 GB/s
without TMA, by warp-specialising scalar `LDG.E` × 128 through a smem-resident
async-barrier pipeline. W22.9 cuda-attn-gdn-async tried `cuda::pipeline` /
`cp.async` and regressed to 245 GB/s vs W1c's 417.7 GB/s, falsifying the
"warp-spec via cuda::pipeline alone closes the gap" hypothesis.

The remaining unexplored axis is **TMA (Tensor Memory Accelerator)** —
specifically `cp.async.bulk.tensor`, which produces SASS `UTMALDG`. This kernel
is the first nvcc-authored data point on this hardware/algorithm with
**`UTMALDG > 0`**.

## Approach

Replace W1c's per-thread float4 load of S_in[k][bv:bv+BV] (16 × `LDG.E.128`
per thread) with **one** TMA bulk-tensor load that brings the entire
`(D_K, BLOCK_V)` slab of S_in into shared memory:

1. Host-side: `cuTensorMapEncodeTiled` builds a 2D tensor descriptor over
   S_in viewed as `(D_V, B_H * D_K)` column-major (TMA convention,
   innermost-first), with box `(BLOCK_V, D_K)`.
2. Descriptor passed by value as a `__grid_constant__` kernel parameter.
3. Device-side: thread 0 issues
   `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes`
   guarded by an `mbarrier` initialised with `arrive_count=1`.
4. All threads `mbarrier.try_wait.parity` for completion.
5. After TMA lands, Pass 1 reads the smem tile via `LDS.128`, scales by alpha
   (in-place into smem), accumulates `u`. Pass 2 is unchanged from W1c.

Pass 2 (S_out store + o read) is intentionally left as W1c-style direct gmem
`STG.E.128` — TMA is applied only to the gmem-load side of the dominant
traffic, so we isolate the load-path hypothesis.

## Build

```
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -ccbin clang-14 -std=c++17 \
    -lineinfo -lstdc++ -lm -o attn_gdn_tma attn_gdn_tma.cu -lcuda
```

`-lcuda` is required for the driver-API symbol `cuTensorMapEncodeTiled`.

## Correctness

```
[gdn-tma] device: NVIDIA GeForce RTX 5090 (sm_120)
[gdn-tma] === correctness run (B=2 H=4 d_k=64 d_v=64) ===
[gdn-tma] o    max_abs=3.052e-05   |want|max=2.954e-01   OK
[gdn-tma] Sout max_abs=2.980e-08   |want|max=4.597e-01   OK

[gdn-tma] === bench-shape smoke (B=1 H=16 d_k=256 d_v=256) ===
[gdn-tma] (qwen3) o   max_abs=6.104e-05   OK
[gdn-tma] (qwen3) Sout max_abs=2.980e-08 OK
```

Both shapes PASS at W1c-equivalent tolerance (`max_abs ≤ 1e-3` for o,
`≤ 5e-3` for S_out). Output errors are `~3e-5` — at the level of f16
round-trip noise — and S_out errors are at f32 epsilon. Numerically the
TMA-loaded tile is bit-identical to the LDG-loaded tile; the residual
errors come solely from the f16 input casts and the f16 output cast.

## SASS instruction-mix table

`/usr/local/cuda/bin/cuobjdump --dump-sass attn_gdn_tma`

| instruction          | W1c (sm_120) | **W22.10 (sm_120)** | notes |
|---|---:|---:|---|
| `UTMALDG`            | **0**        | **2**               | **target signal hit** — one per kernel template instance (D_K=64 + D_K=256) |
| `UTMALDG.2D`         | 0            | 2                   | full opcode form |
| `UTMASTG`            | 0            | 0                   | TMA store path NOT used (Pass 2 still does direct gmem STG) |
| `LDG.E.128`          | 16           | **0**               | the 16 vectorised gmem loads from W1c are gone |
| `LDG.E.64`           | 2            | 2                   | unchanged (q/k halves) |
| `LDG.E` (any)        | ~28          | 46                  | small scalars (alpha/beta + h-loads) |
| `STG.E.128`          | 16           | 16                  | unchanged — Pass 2 state writes |
| `LDS.128`            | 24           | 40                  | more smem reads (TMA tile lives in smem; 2 passes read it) |
| `STS.128`            | —            | 16                  | NEW — alpha-scaled tile written back to smem in Pass 1 |
| `LDGSTS` (cp.async)  | 0            | 0                   | confirms we use bulk.tensor, not legacy cp.async |
| `FFMA`               | 192          | 192                 | unchanged compute |
| `BAR.SYNC`           | 2            | 2                   | the explicit `__syncthreads()` calls |
| `SYNCS`              | 0            | 8                   | NEW — barrier infra around mbarrier / TMA |

The `UTMALDG.2D` instructions appear once per kernel as expected:

```
/*0a20*/  @!UP0 UTMALDG.2D [UR12], [UR4] ;          (D_K=64, D_V=64 instantiation)
/*05a0*/  @!UP0 UTMALDG.2D [UR12], [UR4] ;          (D_K=256, D_V=256 instantiation)
```

The accompanying `BAR.SYNC.DEFER_BLOCKING` and `SYNCS` activity (8 total
SYNCS instructions, up from 0 in W1c) confirms the mbarrier-wait pattern
ptxas lowered correctly.

## Pitfalls hit during authoring

1. **TMA descriptor convention is column-major, not row-major.** S_in is
   stored row-major as `(B_H, D_K, D_V)`. cuTensorMapEncodeTiled expects
   `globalDim[0]` to be the **innermost** (fastest-changing) dimension. So
   `globalDim = {D_V, B_H*D_K}` and `boxDim = {BLOCK_V, D_K}`, with
   coordinates `{coord_x = bv*BLOCK_V, coord_y = bh*D_K}`. Getting this
   backwards silently corrupts data because the load still succeeds at the
   API level.

2. **Innermost box dim in bytes must be a multiple of 16** for
   `CU_TENSOR_MAP_SWIZZLE_NONE`. At BLOCK_V=64 that's 256 B — fine. Smaller
   tiles (e.g. BLOCK_V=4) would also pass, but BLOCK_V=2 (8 B inner) would
   fail at the driver.

3. **TMA destination shared-memory address must be 128-byte aligned.**
   We place `smem_S_raw` at offset 0 of the dynamic-smem allocation and
   declare `extern __shared__ __align__(16) unsigned char smem_raw[]`.
   16B is the C-level alignment; the TMA engine itself enforces the 128B
   gate at runtime. Mis-alignment would give an `Unspecified launch
   failure` after `cudaDeviceSynchronize`.

4. **`mbarrier.arrive.expect_tx` must specify the EXACT byte count the
   TMA will deposit.** For our 2D tile that's `D_K * BLOCK_V * 4`. The
   TMA hardware decrements the tx-count as bytes land; if the expected
   count is wrong the wait spins forever (live-lock).

5. **Pass kernel-parameter `CUtensorMap` by value with
   `__grid_constant__`.** NVIDIA explicitly recommends this in the TMA
   docs. The descriptor is 128 bytes; passing by reference would put it
   in gmem and the kernel would then chase a global pointer per launch,
   defeating the purpose. `__grid_constant__` puts it in constant-memory.

6. **`__CUDA_ARCH__ >= 900` gates the basic 2D variant.** sm_120 (=1200)
   qualifies. The `.cta_group::N` variants are sm_100a/100f/103a/103f
   /110a/110f only (NOT sm_120) — but those are MNNVL/cluster-launch
   features we don't need for single-CTA TMA loads. The plain
   `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::...` form
   on sm_120 is what we want, and the CCCL header
   (`cccl/cuda/__ptx/instructions/generated/cp_async_bulk_tensor.h:253`)
   confirms it gates only on `__CUDA_ARCH__ >= 900`.

7. **`-lcuda` link.** `cuTensorMapEncodeTiled` is a driver-API entry
   (declared in `<cuda.h>`, NOT the runtime API). Forgetting `-lcuda`
   gives a link error at the host driver call.

8. **Inline PTX over `<cuda/ptx>` namespace.** I used hand-written inline
   PTX rather than `cuda::ptx::cp_async_bulk_tensor(...)` so the kernel
   has no CCCL surface dependency and is easy to grep at SASS level.
   The PTX strings are copied verbatim from the CCCL headers, so any
   future PTX-ISA bump is a drop-in upgrade.

## What this enables for downstream waves

This kernel proves **nvcc can emit `UTMALDG` on sm_120 RTX 5090** for the
GDN-decode shape. Combined with W22.8's finding that cuTile reaches
610 GB/s WITHOUT UTMALDG, the next-wave question is binary:

- If `./bench` shows W22.10 ≥ W1c (≥ 417.7 GB/s), TMA-load is at least
  competitive with vectorised LDG on this hardware/algorithm — but
  cuTile's smem-pipeline win remains the larger lever.
- If W22.10 < W1c, that's a strong signal that for THIS algorithm
  (memory-bound, no L2 reuse, single-issue per d_k row), the TMA
  setup overhead dominates the saved instruction count, and the
  cuTile gap is genuinely about the warp-specialised producer/consumer
  smem pipeline (W22.11 already partially explored that axis).

Either way, this gives the W22 series its first nvcc data point on the
TMA load path, completing the matrix:

| wave        | impl      | TMA load? | smem pipeline? | result |
|---|---|---|---|---|
| W1c         | nvcc      | no        | no             | 417.7 GB/s |
| W22.9       | nvcc      | no        | yes (cuda::pipeline) | 245.3 GB/s |
| W22.10      | nvcc      | **yes**   | no             | (bench TBD) |
| cuTile v1.3 | mlir      | no        | yes (warp-spec) | 610.6 GB/s |

## Files

- `attn_gdn_tma.cu` — kernel + correctness driver.
- `bench.cu` — bench harness (compiles, NOT executed in author cell).
- `Makefile` — builds with `-lcuda`; `make sass` dumps instruction mix.
- `run.sh` — clean, build, correctness, SASS dump (no timed bench).
- `attn_gdn_tma.sass` — disassembly evidence (post-`run.sh`).
- `.gitignore` — binaries, sass, logs, CSVs.

## Reproduce

```
cd /home/codeseys/cuda-exploration/cuda-attn-gdn-tma
./run.sh
```
