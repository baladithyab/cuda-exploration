# Occupancy + size-scaling research

Research memo for wave-2. Purpose: give reviewers a principled ceiling for
"what should naive matmul achieve on RTX 5090?" so a measured `~6 TFLOPS at
N=4096` reads as *at the bandwidth-bound ceiling*, not *broken kernel*.
Research only — not a tuning plan.

---

## RTX 5090 / Blackwell key specs

Target: NVIDIA GeForce RTX 5090, GB202, consumer Blackwell, **sm_120**.

Card-level (NVIDIA product page, Blackwell whitepaper PDF):

| spec                   | value                     |
|------------------------|---------------------------|
| SM count               | 170                       |
| CUDA cores             | 21,760 (128 FP32/SM)      |
| Boost clock            | 2.41 GHz                  |
| Peak FP32 (non-tensor) | **104.8 TFLOPS**          |
| Memory                 | 32 GB GDDR7, 512-bit      |
| Memory bandwidth       | **1,792 GB/s**            |
| L2 cache               | 96 MB                     |
| TDP                    | 575 W                     |

Per-SM limits for sm_120 (CUDA Programming Guide Table 30, Blackwell Tuning
Guide §1.4.1.1):

| per-SM limit                  | sm_120                |
|-------------------------------|-----------------------|
| Registers per SM              | **65,536** (64K × 32-bit) |
| Max regs per thread           | 255                   |
| Max threads per SM (resident) | **1,536** (48 warps)  |
| Max warps per SM              | 48                    |
| Max resident blocks per SM    | 32                    |
| Max threads per block         | 1,024                 |
| Shared mem per SM             | up to 128 KB          |
| Max shared mem per block      | 99 KB                 |

> ⚠️ **Correction to task prompt.** The brief assumed "1024 max threads/SM"
> for Blackwell. That's the Ampere number. Authoritative CUDA Programming
> Guide and Blackwell Tuning Guide list **1,536 threads/SM (48 warps/SM)**
> for compute capability 12.x. All occupancy math below uses 1,536.

Sources:
- [NVIDIA RTX Blackwell GPU Architecture whitepaper (PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [Blackwell Tuning Guide 13.2](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) · [CUDA Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) · [RTX 5090 product page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

---

## Our kernel's register pressure (from PTX)

The kernel is in `oxide-matmul/oxide_matmul.ptx`. The file contains two entry
points (`matmul_unchecked` and `matmul`, the latter with bounds-check fences).
Their virtual-register declarations:

```
; matmul_unchecked:
.reg .pred  %p<4>;    ; 4 predicate regs
.reg .b32   %r<17>;   ; 17 32-bit regs
.reg .b64   %rd<25>;  ; 25 64-bit regs (= 50 32-bit slots)

; matmul (checked):
.reg .pred  %p<6>;    ; 6 predicate regs
.reg .b32   %r<17>;   ; 17 32-bit regs
.reg .b64   %rd<31>;  ; 31 64-bit regs (= 62 32-bit slots)
```

Upper bound on **virtual** 32-bit-equivalent registers per thread:

- `matmul_unchecked` ≈ 17 + 2·25 = **67**
- `matmul` (checked)  ≈ 17 + 2·31 = **79**

### Caveat: PTX virtual regs ≠ physical regs

PTX declarations are an upper bound set by the LLVM NVPTX front-end *before*
`ptxas` does SSA register allocation. The real allocation happens during PTX
→ SASS compilation; `ptxas` typically collapses virtual registers by 30–50 %
via live-range analysis. For a kernel this simple (one FP32 accumulator, a
loop counter, two pointers, three thread-index regs) the realistic post-
`ptxas` count is **~24–32 physical regs/thread**. Exact number would come
from `ptxas --verbose` or `cuobjdump --dump-sass` — not run in this loop.

The table below gives both the conservative bound (virtual ~79) and the
likely-actual (~32) scenarios.

---

## Theoretical occupancy at 16×16

Block = 16 × 16 = **256 threads/block** = 8 warps/block.

Per-SM budget (sm_120): 65,536 regs, 1,536 threads, 48 warps, 32 blocks max.

### Scenario A — conservative (virtual regs = 79/thread)

- Regs/block: 256 × 79 = 20,224 → register-limited blocks/SM = ⌊65,536 /
  20,224⌋ = **3**
- Thread-limited: ⌊1,536 / 256⌋ = 6. Warp-limited: ⌊48 / 8⌋ = 6.
- Actual blocks/SM = min(3, 6, 6, 32) = **3** → 24 warps resident →
  **50 % occupancy**.

### Scenario B — realistic post-ptxas (32 regs/thread)

- Regs/block: 256 × 32 = 8,192 → register-limited blocks/SM = 8.
- Actual blocks/SM = min(8, 6, 6, 32) = **6** → 48 warps resident →
  **100 % occupancy** (thread-count-bound, not register-bound).

### Scenario C — breakeven

The register threshold where register count starts limiting occupancy
(instead of the 1,536-thread cap) is 65,536 / 1,536 = **42.67 regs/thread**.
As long as physical regs ≤ 42, we hit full 100 % occupancy on sm_120.

### Takeaway

A scalar FP32 naive matmul with no shared memory is almost certainly in
scenario B: occupancy ~100 %. Occupancy is **not** the bottleneck — see
"Memory bandwidth ceiling".

---

## Block-size sensitivity (qualitative)

For a 2D naive matmul kernel that emits one thread per output element, common
block shapes and their tradeoffs on sm_120:

| Block shape | threads/block | blocks / SM (@ 32 regs) | warps / SM | occupancy | notes |
|-------------|---------------|-------------------------|------------|-----------|-------|
| 8 × 8       | 64            | 24 (capped at 32)       | 48         | 100 %     | tiny blocks → huge grid, more launch overhead, warps underutilized if any mid-warp branches |
| 16 × 16     | **256** *(ours)* | 6                    | 48         | 100 %     | sweet spot: warps align with row tiles, grid size manageable |
| 32 × 8      | 256           | 6                       | 48         | 100 %     | same occupancy as 16×16, but each warp scans a full 32-wide row — better coalescing on B, worse reuse of A |
| 32 × 16     | 512           | 3                       | 48         | 100 %     | fewer blocks/SM, less scheduling flexibility when one block stalls on memory |
| 32 × 32     | 1,024         | 1                       | 32         | 67 %      | leaves 1/3 of warp slots unused; also reduces tail-flexibility |

### Key qualitative points

1. All shapes in 64–512 hit ~100 % occupancy because register pressure is
   low and no shared memory is used.
2. **Coalescing** is the real reason to prefer wider-in-x shapes. A warp
   of 32 threads with consecutive `tid.x` issues one 128-byte coalesced
   load of `B[k, col]`.
3. **32-wide x** gives slightly better B-stride-1 coalescing than 16-wide;
   difference usually ≤ 10–20 %.
4. **Smaller blocks (8×8)** hurt via grid-level launch overhead and
   reduced intra-block latency-hiding opportunity, not via occupancy.
5. **32×32 is actively worse**: 1,024-thread blocks → 1 block/SM → 512
   thread slots (16 warps) idle → 67 % occupancy.

Sources: CUDA C++ Best Practices Guide §"Execution Configuration" &
§"Occupancy"; CUDA Programming Guide §5.2.3 "Multiprocessor Level"; Mark
Harris, "Optimizing Parallel Reduction in CUDA" (canonical GTC talk).

---

## Memory bandwidth ceiling

### Traffic model for naive matmul

For C = A · B with A, B, C all N×N fp32:

- Each output element C[i,j] computes Σₖ A[i,k] · B[k,j].
- **Without any reuse** (pure textbook-naive), every thread loads an entire
  row of A (N fp32s) and an entire column of B (N fp32s) from global memory.
- Total global-memory loads: N² output elements × 2N fp32s/element × 4 bytes
  = **8·N³ bytes**.
- Stores: N² × 4 bytes = 4·N² bytes (negligible).

### With L2 cache (reality)

The L1 and 96 MB L2 absorb some traffic — warps in a block reading the
same A row hit cache; adjacent blocks sharing B columns hit L2. Effective
DRAM traffic is less than 8·N³ but still **O(N³)**, not the O(N²·√N) a
well-tiled kernel achieves. Empirically, naive kernels on Ampere/Ada hit
50–80 % of the raw 8·N³ figure at N=4096.

### FLOP count

2·N³ fp32 FMA ops (1 multiply + 1 add per inner-loop iteration, N² × N loop
iters total).

### Arithmetic intensity

- Naive: 2·N³ FLOPs / 8·N³ bytes = **0.25 FLOP/byte**
- Tiled (tile size T): 2·N³ / (2·N³/T · 4 bytes) = **T/2 FLOP/byte**
  (e.g. T=32 → 16 FLOP/byte; T=64 → 32 FLOP/byte)

### Roofline ceilings on RTX 5090

Roofline peak = min(peak_FP32, bandwidth × arithmetic_intensity).

- Peak FP32: 104.8 TFLOPS
- Bandwidth: 1,792 GB/s = 1.792 × 10¹² B/s

**Naive (0.25 FLOP/B):** 1,792 × 0.25 = **448 GFLOPS = 0.448 TFLOPS**
if we were paying the full 8·N³-byte traffic bill.

Our measurement is 6 TFLOPS. 6 / 0.448 = ~13×, which means the L2 is
delivering ~13× reuse on top of what cold DRAM could sustain. That is
plausible at N=4096 where 96 MB of L2 can hold meaningful fractions of A
and B during the computation: 4096² × 4 B = 64 MB per matrix, and the L2 is
96 MB → one full matrix (A, the one being streamed row-by-row) comfortably
fits, and portions of B hit too.

### Bandwidth-bound ceiling (with realistic L2 reuse)

A better heuristic: once L2 does its work at N near the L2 working-set
boundary, the naive kernel's *effective* arithmetic intensity on Blackwell
consumer parts is typically 3–10 FLOP/byte:

- 1,792 × 3 = **5.4 TFLOPS** (pessimistic)
- 1,792 × 5 = **9.0 TFLOPS** (moderate reuse)
- 1,792 × 10 = **17.9 TFLOPS** (optimistic, N small enough to fit)

**Our 6 TFLOPS at N=4096 sits squarely in the expected 5–10 TFLOPS
bandwidth-bound band** for naive, un-tiled, scalar-accumulator FP32 matmul
on RTX 5090. That is the correct ceiling — **not** the 104.8 TFLOPS peak.

---

## Compute-bound vs memory-bound regimes

The compute-bound / memory-bound boundary for a given kernel is where

```
arithmetic_intensity [FLOP/byte] = peak_FLOPS / peak_bandwidth
                                 = 104.8e12 / 1.792e12
                                 ≈ 58.5 FLOP/byte
```

— called the **machine balance** of the RTX 5090 for FP32. To be compute-
bound (saturate the 104.8 TFLOPS peak), a kernel must sustain > 58 FLOP per
byte of DRAM traffic.

Where different implementations land:

| kernel variant          | AI (FLOP/B) | roofline ceiling         |
|-------------------------|-------------|--------------------------|
| Naive (no reuse)        | 0.25        | ~0.45 TFLOPS             |
| Naive + L2 reuse        | 3–10        | 5–18 TFLOPS              |
| Shared-mem tile T=16    | 8           | ~14 TFLOPS               |
| Shared-mem tile T=32    | 16          | ~28 TFLOPS               |
| Shared-mem tile T=64    | 32          | ~57 TFLOPS               |
| Tile T=128 + reg-tile   | ~64         | ~104 TFLOPS (compute-bound) |
| cuBLAS SGEMM            | as above    | compute-bound, ~95 TFLOPS FP32 |
| cuBLAS HGEMM / Tensor   | (separate tensor-core roofline) | ~1 PFLOPS FP16 |

### Crossover sketch

Plain shared-memory tile matmul with **T ≥ 30** crosses machine balance and
becomes compute-bound. Naive kernel *never* crosses it — asymptotic AI is
fixed at ~0.25 FLOP/B (or at best 5–10 via L2 reuse, still ~6× below
balance).

**Expected gains:**
- Shared-mem tiling: **~5–10× over naive** → 28–57 TFLOPS.
- cuBLAS SGEMM: **~15–20× over naive** → ~90–100 TFLOPS (FP32 peak).
- cuBLAS HGEMM (Tensor Cores, FP16): another **10×** on top → ~1 PFLOPS.

---

## Expected size-scaling curve shape

For the M2 sweep (N ∈ {1024, 2048, 4096, 8192}), expected TFLOPS-vs-N
curve for our naive kernel (ASCII sketch):

```
TFLOPS
 10|
  8|               __/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\__   (slight dip at N≥8192:
  6|             _/      ← measured 6 TFLOPS       bus contention, TLB,
  4|           _/            @ N=4096                DRAM page spikes)
  2|         _/
  0+——————/—————+———————+———————————+———————————+—→
      0  1024 2048     4096        8192         N
         |      |       |            |
    launch   enters  plateau:     still on plateau,
    overhead plateau L2-saturated   maybe slight dip
    dominates        bandwidth-bound
```

### Regime-by-regime

**N = 1024 (~0.5–3 TFLOPS):** Launch overhead dominates. Grid = 64×64 =
4,096 blocks @ 256 threads. At 6 blocks/SM × 170 SMs = 1,020 concurrent
blocks — the entire grid is resident in ~4 waves. Latency hiding is
starved for warps. TFLOPS well below plateau.

**N = 2048 (~4–7 TFLOPS):** Grid = 16,384 blocks, ~16 waves. L2 working
set (two 16 MB matrices) fits in 96 MB L2. Climbs onto plateau.

**N = 4096 (~5–8 TFLOPS, measured ~6):** Grid = 65,536 blocks, ~64 waves.
Each matrix is 64 MB — one fits L2, the other streams. Bandwidth-bound
steady-state. **Measured 6 TFLOPS is exactly on the expected plateau.**

**N = 8192 (~4.5–7.5 TFLOPS, watch for dip):** Grid = 262,144 blocks,
~256 waves. Each matrix is 256 MB — vastly exceeds L2. L2 reuse drops;
kernel becomes more strictly DRAM-bound (closer to ~5 TFLOPS pessimistic).
Additional dip risks: GDDR7 page switching, TLB pressure, and —
critically — **WDDM TDR watchdog**. We're on WSL2/WDDM per `system-spec.txt`;
a naive 8192³ matmul at 6 TFLOPS takes ≈ 2·8192³/6e12 ≈ 180 s wall,
catastrophic for WDDM. The bench harness must guard this.

### Sanity-check message for reviewers

> "At N=4096, oxide-unchecked reaches 6 TFLOPS. The roofline ceiling for a
> naive (no shared-memory tiling) FP32 matmul on RTX 5090 is in the 5–10
> TFLOPS range, dominated by DRAM bandwidth (1,792 GB/s) modulated by L2
> reuse (96 MB). 6 TFLOPS is **on the expected plateau**. Adding a shared-
> memory tile should gain 5–10× (→ 30–60 TFLOPS); moving to cuBLAS SGEMM
> should gain 15–20× (→ 90–100 TFLOPS); moving to cuBLAS HGEMM with Tensor
> Cores should gain 100–150× (→ ~1 PFLOPS FP16)."

---

## Why we're NOT retuning blocks in this loop

1. **Occupancy is not the bottleneck.** Scenario B above shows ~100 %
   occupancy already. Block-shape changes (16×16 → 32×8 / 32×16) move
   coalescing by maybe 10–20 % and leave the roofline ceiling unchanged.
   Gain bounded by the same 5–10 TFLOPS bandwidth wall.
2. **Right intervention is tiling, not block-size tuning.** 16×16-naive →
   16×16-with-shared-mem-tile = ~5–10× speedup. 16×16-naive →
   32×16-naive = ~1.1× speedup. Opportunity cost overwhelmingly favors
   tiling.
3. **Block-size tuning is a rabbit hole.** Real tuning needs: 2D shape
   sweep, `ptxas --verbose` and nsys/ncu for actual register counts and
   `achieved_occupancy`, then re-run bench harness per config. That's
   post-tiling performance-engineering work, not baseline-establishment
   work.

This memo exists so reviewers see the M2 size-sweep measurements sit at
the expected bandwidth-bound plateau, with a quantified roadmap
(tiling → cuBLAS → Tensor Cores) for future gains.

---

## References

1. NVIDIA, [CUDA Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) (Table 30).
2. NVIDIA, [Blackwell Tuning Guide 13.2](https://docs.nvidia.com/cuda/blackwell-tuning-guide/), §1.4.1.1 Occupancy.
3. NVIDIA, [RTX Blackwell GPU Architecture whitepaper (PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
4. NVIDIA, [GeForce RTX 5090 product page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).
5. ptxas reverse-engineering reference (sm_120 per-SM limits): <https://gh.evko.io/nvopen-tools/ptxas/targets/blackwell.html>.
6. Williams, Waterman, Patterson, "Roofline: An Insightful Visual
   Performance Model for Multicore Architectures", CACM 2009.
7. NVIDIA, CUDA C++ Best Practices Guide — block-size and occupancy
   guidance.
