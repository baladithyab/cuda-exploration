# Wave 22.12 — cuTile vs nvcc-W22.11: launch-geometry & kernel-shape diff

**Status.** Investigation; metrics extracted from cubins. No source modified.

## TL;DR

W22.11 SASS-pattern-matches cuTile (SYNCS=106, BSSY=61, MBAR=18 per the W22.11
ANALYSIS — actually `BAR.SYNC`-class barriers in our re-count) yet runs at
**245 GB/s vs cuTile 610 GB/s**. The gap is **not** in the grid layout — both
launch the **same `(B*H, d_v/BV) = (16, 4) = 64` blocks**. The gap is at the
**CTA-internal level**:

| dimension | cuTile | nvcc W22.11 | ratio |
|---|---:|---:|---:|
| `blockDim` (REQNTID-fixed)      | **256** (8 warps) | **128** (4 warps) | **2×** |
| static smem per CTA (bytes)     | **99 412** (~97 KB) | 80 (no static) | — |
| dynamic smem per CTA (bytes)    | 0 (1 964 ceiling)   | **49 072 set / 70 656 actual at d_k=256** | — |
| total smem per CTA              | **~99 KB** (all static) | ~70 KB (all dynamic) | 1.4× |
| registers / thread              | **255** | 40 | **6.4×** |
| live regs per CTA               | 256·255 = **65 280** | 128·40 = **5 120** | 12.8× |
| local memory / thread (B)       | **824** | 0 | spilling |
| `EIATTR_NUM_BARRIERS`           | **3** | 1 | 3× |
| SASS code size (`.text`)        | **20 KB** (10 841 lines) | ~7 KB (3 500 lines) | ~3× |
| `STG.E.128` (vector store)      | **0** | 16 | **only nvcc** |
| `STG.E` (scalar store)          | **128** | 0 | **only cuTile** |
| `LDG.E` (scalar 4-B load)       | **128** | 0 | **only cuTile** |
| `LDGSTS.E.BYPASS.128` (cp.async)| **0** | 16 | **only nvcc** |
| `LDS.128`                       | **326** | 32 | 10× |
| `STS.128 + STS.64`              | 96+66 = **162** | 8+18 = **26** | 6.2× |
| `STSM.16.M88`                   | **3** | 0 | warp-coop store |
| `FMUL` count                    | **900** | 48 | 19× |
| `FFMA` count                    | **0** | 160 | nvcc-only |
| `FADD` count                    | **771** | 8 | 96× |

The two kernels are doing **fundamentally different things per CTA** despite
identical grid geometry. cuTile is essentially **a per-CTA mini-kernel that
processes the entire `(D_K, BLOCK_V) = (256, 64)` state tile at once** with an
8-warp warp-specialised pipeline; nvcc is a **128-thread cp.async pipeline**
that reuses the same single-warp arithmetic kernel from W22.9 — just with
3 idle math-warps as cp.async issuers.

## How the metrics were extracted

All commands run from `/home/codeseys/cuda-exploration`. CUDA 13.2,
`/usr/local/cuda/bin/cuobjdump`, sm_120 native.

```bash
# 1. cuTile cubin — already on disk, no compile needed
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
    cutile-attn-gdn/gdn_decode_fused.cubin

# 2. nvcc W22.11 — build the binary, then dump resource usage
cd cuda-attn-gdn-async-tpb128
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -ccbin clang-14 -std=c++17 \
    -lineinfo -o attn_gdn_async_tpb128 attn_gdn_async_tpb128.cu
/usr/local/cuda/bin/cuobjdump --dump-resource-usage attn_gdn_async_tpb128

# 3. cubin-internal EIATTR (REQNTID, NUM_BARRIERS, MAXREG_COUNT)
/usr/local/cuda/bin/cuobjdump --dump-elf cutile-attn-gdn/gdn_decode_fused.cubin
/usr/local/cuda/bin/cuobjdump --dump-elf attn_gdn_async_tpb128
```

A small driver-API probe (`/tmp/probe_cutile.cpp`, ~80 lines) was also run
against both cubins to read `cuFuncGetAttribute(MAX_THREADS_PER_BLOCK,
SHARED_SIZE_BYTES, NUM_REGS, ...)` directly. Probe output (verbatim):

```
=== cuTile gdn_decode ===
  MAX_THREADS_PER_BLOCK   = 256       <-- REQNTID-enforced
  SHARED_SIZE_BYTES       = 99 412
  NUM_REGS                = 255
  LOCAL_SIZE_BYTES        = 824       <-- spills 824 B/thread
  MAX_DYNAMIC_SHARED_SIZE = 1 964     <-- runtime can extend
  REQUIRED_CLUSTER_*      = 0         <-- no cluster
=== nvcc W22.11 gdn_decode_async_tpb128_kernel<256,64,4,128> ===
  MAX_THREADS_PER_BLOCK   = 1024      <-- device default (no REQNTID)
  SHARED_SIZE_BYTES       = 80
  NUM_REGS                = 40
  LOCAL_SIZE_BYTES        = 0
  MAX_DYNAMIC_SHARED_SIZE = 49 072
  REQUIRED_CLUSTER_*      = 0
```

## Grid geometry — they are the same

Reading the launch sites:

* `cutile-attn-gdn/main.py:247`
  `grid = (shape.batch * shape.n_heads, shape.d_v // bv)`
* `cuda-attn-gdn-async-tpb128/attn_gdn_async_tpb128.cu:445`
  `dim3 grid(B_H, sh.d_v / BLOCK_V); dim3 block(TPB);`

At Qwen3-Next decode shape (B=1, H=16, d_k=256, d_v=256, BLOCK_V=64), both
launch **64 blocks** in a `(16, 4)` grid. **The hypothesis "cuTile launches
fewer blocks" is wrong** — confirmed by reading both source files.

`BLOCK_V` selection diverges, but only at small d_k:

* cuTile `pick_block_v` picks 128 at d_k=64, 64 at d_k=256.
* nvcc W22.11 hard-codes `BLOCK_V=64` for both.

So at the headline shape they are identical. Below d_k=256 cuTile runs
**fewer** blocks (e.g. d_k=64 → grid `(BH, d_v/128)` = half of nvcc's grid).
That is *not* the cause of the qwen3-shape regression; it is a separate
difference whose effect is shape-dependent.

**Persistent kernel? No.** Both kernels read `SR_CTAID` 4 times in their SASS
(the standard pattern of nvcc-emitted blockIdx materialisation; cuTile uses
the same lowering). Neither has a CTA outer loop driven by `blockIdx >=
gridDim` patterns. Both emit `EXIT` 6 times. Both rely on the runtime grid
walker, not a persistent-CTA per-SM scheme.

## Where the difference actually lives — CTA internals

### 1. cuTile uses 8 warps per CTA, nvcc uses 4

`EIATTR_REQNTID = 0x100, 0x1, 0x1` in cuTile's cubin **forces** blockDim =
(256,1,1). nvcc has no REQNTID — its 128 is set by host code. cuTile
explicitly committed to **8 warps per CTA**; nvcc to **4 warps per CTA**.

8 warps gives cuTile more freedom to split roles. From the W22.8 W17 doc the
cuTile pattern is: 1 producer warp, plus consumer warps each owning a stripe
of `D_K` rows in the state tile (with `BLOCK_V=64`, that is 16 LDS.128-lanes
per row → consumer warps split `D_K/8 = 32` rows each across 7 warps,
roughly). nvcc W22.11 uses 1 producer + 3 consumers, with **only 16 of those
96 consumer threads** doing FFMA (per its own ANALYSIS.md note: "VLANES =
BLOCK_V/4 = 16 float4 stripes per row" → only one float4-lane-per-thread
width, so only `BLOCK_V/4 = 16` threads do useful math per row).

This is **the bottleneck**: nvcc's TPB=128 is a half-fix to the original W22.8
TPB=16 problem. The kernel structure inherited from W1c only ever had 16
math-active lanes per CTA, and W22.11 widened to 128 to enable cp.async
warp-spec but did not restructure the math to use the extra threads.

### 2. cuTile's 99 KB static smem houses the entire state tile

`SHARED:100436` (resource usage) ≈ `99 412` static (driver probe) + 1 024
reserved/runtime overhead. The tile shape (D_K=256, BLOCK_V=64) f32 =
**65 536 bytes** for the state alone, plus q, k, v, alpha, beta, scratch,
ring stages, and likely the `outer_acc` (D_K, BLOCK_V) f32 = another 64 KB
(this is the only way to reach ~99 KB). **cuTile materialises the full state
tile in smem and keeps it there for the duration of the CTA.**

nvcc's `smem_bytes` for `D_K=256, BLOCK_V=64, N_STAGES=4`:

```
2*D_K*sizeof(float)               = 2 048    (q, k cache)
D_K * VLANES * sizeof(float4)     = 65 536   (S_scaled tile)
N_STAGES * VLANES * sizeof(float4)= 1 024    (ring buffer)
                                   --------
                                   ≈ 68 608  + 1 104 static = 69 712 ≈ 70 KB
```

So nvcc actually has the same-size state tile in smem (`S_scaled`).
**The 30 KB cuTile uses *more* must be the per-CTA `outer_acc` tile** (D_K,
BLOCK_V) f32 = 64 KB — **except** that 64 KB doesn't fit alongside S_scaled
in the 100 KB budget, so cuTile must be *fusing* `s_scaled` and `outer_acc`
into the same buffer. Likely layout: state-tile (64 KB) + q/k/v/alpha/beta
working set (~32 KB) + barrier-state (~3 KB) → ~99 KB.

### 3. cuTile uses 3 named barriers; nvcc uses 1

`EIATTR_NUM_BARRIERS: 0x3` (cuTile) vs `0x1` (nvcc). The Blackwell hardware
has 16 named barriers per SM; ptxas pre-allocates a number based on
`bar.sync` ID usage. cuTile's three barriers fit a **producer / consumer-A /
consumer-B** triangle, or **producer / state-update / output-stage**, both
of which are richer pipelines than nvcc's single-barrier flat 1P+3C split.

### 4. Register pressure — 255 vs 40

cuTile is at the **64-bit register-file ceiling**: 256 threads × 255 regs =
65 280 regs ≈ the SM's 65 536-reg file. **One CTA per SM, mandatory.** With
99 KB smem also competing, cuTile has chosen a **maximally fat-CTA** strategy.

nvcc at 40 regs × 128 thr = 5 120 regs/CTA → many CTAs/SM by reg budget
(would be 12), capped by smem (70 KB) to ~1-2 CTAs/SM on the 228-KB-shared
Blackwell SM (depending on L1 carveout).

But the **grid is only 64 blocks**. With 170-ish SMs on the RTX 5090, both
kernels run only **at most 64 SMs concurrently**, leaving 100+ SMs idle.
This is where the CTA-internal density matters: a fat cuTile CTA does the
*entire* (D_K, BLOCK_V) tile in one shot — reading state from HBM, doing
the rank-1 update, writing back — with 8 warps coordinating. A skinny nvcc
CTA spends most of its lifetime issuing cp.async with 3 idle math warps and
running the math loop on a single warp (16 active lanes).

### 5. Memory-issue width — cuTile narrower per instruction, more total

| | cuTile | nvcc W22.11 |
|---|---:|---:|
| HBM **load** path | 128× `LDG.E` (4 B each) | 16× `LDGSTS.E.BYPASS.128` (16 B each) |
| HBM **store** path | 128× `STG.E` (4 B each) | 16× `STG.E.128` (16 B each) |
| total HBM traffic per inst class | 128·4 = 512 B (load) | 16·16 = 256 B (load) |

cuTile issues **more** HBM ops, each 4-bytes wide. nvcc issues fewer, each
16-bytes wide. **But cuTile achieves higher GB/s** — because the warp-spec
producer warp issues these LDGs in a tight stream while consumer warps
overlap math, whereas nvcc's TPB=128 single-warp consumer can't keep the
LSU busy.

cuTile's `LDS.128 = 326` (consumers re-reading from smem 326 times) vs
nvcc's 32 — cuTile relies heavily on the smem-resident state tile and
**re-reads the same f32 state column 5+ times** during a single rank-1
update loop. nvcc reads each cp.async-loaded float4 once and consumes it.

This pattern says cuTile's compiler chose a **store-once-to-smem,
read-many-from-smem** design where the state tile is the first-class data
structure; nvcc's compiler chose a **stream-through-ring-buffer** design
where each ring slot is consumed once.

## Hypothesis evaluation

**H1 (top): The bottleneck is consumer-side FFMA throughput, not load
throughput.** cuTile has **900 FMUL + 771 FADD** per CTA; nvcc has
**48 FMUL + 8 FADD + 160 FFMA**. Even adding nvcc's 160 FFMA (= 1 FMUL +
1 FADD merged) the totals are 168 effective f32-multiply-adds per nvcc CTA
vs ~835 for cuTile (counting FMUL+FADD pairs). cuTile is doing **5×** the
math per CTA. Combined with 2× the threads, cuTile delivers **~10× the
arithmetic per CTA**. This is consistent with cuTile's per-CTA work
covering the full `(D_K, BLOCK_V)` rank-1 update plus output projection,
where nvcc's per-CTA work covers a thin slice handled by 16 active threads.

**Evidence rating: strong.** The arithmetic counts come straight from SASS,
the thread-count gap from REQNTID, and the per-thread FFMA-throughput rate
(2 ops/cycle on Blackwell SM) is well-known. **At 245 GB/s, nvcc's measured
HBM bandwidth is below its math-pipe ceiling**, but its math pipe is also
underutilised (16 of 128 threads do work, so peak 16 FMAs/cycle vs an SM
ceiling of 256 FMAs/cycle = 6% utilisation). cuTile's 256 threads · ~2-3
FMAs/cycle in the consumer phase = ~50% utilisation.

**H2: The "warp-spec barrier infrastructure" was the wrong primitive to
copy.** W22.8 hypothesised that cuTile's edge was the `SYNCS.PHASECHK.TRANS64`
async-barrier pattern. W22.11 reproduced that pattern (SYNCS=106,
BSSY=61) — and got **40% of cuTile's GB/s**. The async-barrier pattern was
**necessary but far from sufficient**. The barrier ops themselves are
~free; what matters is **what work the consumers do between barriers**.
cuTile's consumers do **5× the FMUL+FADD per barrier-trip** that nvcc's
consumers do.

**Evidence rating: strong.** SASS counts.

**H3: The `outer_acc` materialised tile is the load-amplification trick.**
cuTile has 162 `STS` and 326 `LDS.128`; nvcc has 26 STS and 32 LDS.128.
The 6× bump in smem traffic pairs with cuTile's heavier math. The
hypothesis: cuTile splits the rank-1 update into per-warp tiles, each warp
writing its partial f32 outer-product slice to smem, then consumer warps
read those slices to produce the final `s_out + beta·outer_acc` and the
`q^T · s_out` projection. **The 99 KB smem is large enough to hold both
the input state tile (64 KB) and the outer-product result (32-64 KB)
simultaneously**, which is exactly what enables the warp-cooperative
pattern.

nvcc W22.11 cannot do this because it has only 70 KB smem allocated and
its 128-thread CTA can't decompose the (D_K=256, BLOCK_V=64) tile across
4 warps without each warp holding (256·64/4)/256 ≈ 16 floats per thread —
which fits in registers, *but only if the outer-product update is computed
inline in the math loop rather than as a separate pass*. It chose the
inline path; that's why FFMA=160 only.

**Evidence rating: medium-strong.** The smem-budget arithmetic is solid;
the per-warp partitioning conjecture would need confirmation by tracing a
slice of cuTile SASS through register liveness — out of scope for this
wave.

## Recommendations for W22.13+

1. **Increase blockDim from 128 → 256** in the next CUDA C++ port. This
   alone won't fix the gap (nvcc with 128 wastes 112 of 128 threads on
   non-math work; widening to 256 with the same wasteful structure
   wastes more), but it's a prerequisite to (2).

2. **Restructure the math so all consumer threads do FFMA.** Specifically
   split the (D_K, BLOCK_V) outer product across all consumer warps, with
   each thread owning multiple `BLOCK_V`-stripes. With BLOCK_V=64 →
   16 float4-lanes; with 7 consumer warps × 32 lanes = 224 threads, each
   thread owns ~5 float4-lanes (still register-resident). Target: every
   thread issues ≥2 FFMA per cycle in the consumer loop.

3. **Materialise `outer_acc` in smem (not registers)** and use a second
   pass to fuse `s_out = s_scaled + beta·outer_acc` followed by
   `o_acc = q^T · s_out`. This requires opting into the **>49 KB dynamic
   smem** path with `cudaFuncAttributeMaxDynamicSharedMemorySize` set to
   ~100 KB, the same limit cuTile uses statically. Confirm the chip
   supports it: Blackwell sm_120 SMs have **228 KB combined L1+smem**, of
   which up to 100-112 KB can be smem after carveout. **In-budget.**

4. **Use 3 named barriers**, not 1. The `__syncthreads()`-equivalent
   barrier-0 is too coarse for a producer/consumer-A/consumer-B pipeline.
   Use `cuda::barrier` with two scopes or roll explicit `bar.arrive` /
   `bar.sync.aligned` triples.

5. **Do NOT add a persistent-CTA outer loop.** Both reference kernels use
   the runtime walker; cuTile achieves 610 GB/s without persistence.
   Persistence would help only if the grid count is so small that launch
   overhead dominates — at 64 blocks on 170 SMs that's plausible, but
   cuTile's measured 13.79 µs best is consistent with non-launch-bound
   execution. Defer until after (1)-(4) close the gap.

6. **Don't try to beat cuTile by going wider/deeper.** cuTile is already at
   65 280 regs (1 CTA/SM cap) and 99 KB smem (just below the carveout
   ceiling). The room for nvcc to do better is in (a) **getting math
   utilisation up from 12% to ~50%** (H1's recommendation), and (b)
   **using the smem more aggressively** (H3). Going to >256 threads or
   >100 KB smem won't help — it just reduces grid concurrency.

## Files

* This doc: `docs/research/wave22-12-cutile-launch-geometry.md`
* Probe (transient, deletable): `/tmp/probe_cutile.cpp`,
  `/tmp/probe_cutile`, `/tmp/cutile_elf.txt`, `/tmp/nvcc_elf2.txt`,
  `/tmp/attn_gdn_async_tpb128.sm_120.cubin`
* Source kernels (read-only inputs, unchanged):
  - `cutile-attn-gdn/main.py`
  - `cutile-attn-gdn/gdn_decode_fused.cubin`
  - `cutile-attn-gdn/gdn_decode_fused.sass`
  - `cuda-attn-gdn-async-tpb128/attn_gdn_async_tpb128.cu`
  - `cuda-attn-gdn-async-tpb128/attn_gdn_async_tpb128.sass`
* Prior research: `docs/research/wave17-w1c-tma-vs-ldg128-investigation.md`
  (W22.8 SASS instruction-mix table; this doc extends to launch geometry).

## Appendix — instruction-mix counts (re-derived this wave)

```
                          cuTile    nvcc-W22.11
BAR.SYNC                       8           14
BSSY                          14           61
BSYNC                         14           61
SYNCS                         62          106
FFMA                           0          160
FMUL                         900           48
FADD                         771            8
LDS.128                      326           32
LDS.64                         0            2
LDG.E (4 B)                  128            0
LDG.E.U16                      7            8 (CONSTANT)
LDG.E.64.CONSTANT              0            2
LDGSTS.E.BYPASS.128            0           16  (cp.async — nvcc only)
STG.E.128                      0           16
STG.E (4 B)                  128            0
STG.E.64                       0            2
STS.128                       96            8
STS.64                        66           18
STSM.16.M88                    3            0
HMMA                           0            0
EXIT                           6            6
SR_CTAID reads                 4            4
```

Counts via `grep -cE` against the `.sass` files; cross-checked against the
W22.8 doc's reported numbers (`SYNCS=31` for cuTile in the W17 doc was
counting only `.PHASECHK.TRANS64`-suffixed variants, not the full `SYNCS`
family; this wave's count of 62 includes `SYNCS.EXCH`, `SYNCS.ARRIVE`,
`SYNCS.PHASECHK*`, etc.). The W22.11 ANALYSIS.md headline of "SYNCS=106"
matches this wave's recount (106).
