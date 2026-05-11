# Wave 13.2 — SASS diff: reduction (cuTile vs cuda-oxide vs nvcc)

> **Headline**: cuTile wins f32 sum-reduction at 256M (1696 GB/s vs ~1520 GB/s
> for oxide/nvcc, +11%) not because its ALU work is leaner — it is actually
> *heavier* per block — but because its **memory-load strategy is different in
> kind**. cuTile issues each tile as a single `UTMALDG.1D` bulk TMA copy
> directly into shared memory; oxide and nvcc load element-by-element
> through the `LDG` / `LD.E` per-thread path.

All cubins built for `sm_120` (RTX 5090, Blackwell). Disassembled with
`cuobjdump --dump-sass` (CUDA 13.2).

## 1. Kernel-level instruction counts

Counts are for the **whole cubin** (kernel symbol only; cuTile reduction
cubin has exactly one function, so the numbers are directly comparable
up to block-tile size). `impl / kernel_fn` is the implementation label.

| metric                     | cutile reduce_sum | oxide reduce_sum | cuda reduce_sum |
| -------------------------- | -----------------: | ----------------: | ---------------: |
| total SASS lines           | 489                | 217               | 169              |
| **FADD**                   | 18                 | 12                | 9                |
| **SHFL** (warp shuffle)    | 7                  | 8                 | 8                |
| **BAR.SYNC**               | 8                  | 1                 | 1                |
| **ATOM** (atomic reduce)   | 1                  | 3                 | 0                |
| `UTMALDG.1D` (TMA bulk)    | **7**              | 0                 | 0                |
| `LDG.E` (per-thread glbl)  | 0                  | 0                 | 1                |
| `LD.E` (untyped glbl)      | 0                  | 3                 | 0                |
| `LDG.E.CONSTANT`           | 0                  | 0                 | 1                |
| `LDS` / `STS`              | 4 / 3              | 2 / 1             | 1 / 1            |
| `SYNCS.*` (async mbar)     | 10                 | 0                 | 0                |
| `BRA`                      | 12                 | 7                 | 3                |

(Full table in `instruction_counts.csv`.)

## 2. Hot-loop structure

### nvcc (`reduce_sum_kernel`) — the textbook 2-stage
```
LOOP:   LDG.E.CONSTANT R11, desc[UR6][R8.64]   // load one f32 per thread
        IADD.64 R4, R4, R8                     // stride step
        FADD R0, R11, R0                       // acc += v
        @!P0 BRA LOOP                          // grid-stride loop
        ...
        SHFL.BFLY R3, R0, 0x10, 0x1f           // warp reduce
        FADD R3, R3, R0
        SHFL.BFLY R4, R3, 0x8, 0x1f
        ...
```
Classic single-line grid-stride + warp-shuffle reduction. 1 FADD + 1 LDG
per element in the hot path.

### oxide — same shape, plus `LD.E` (untyped) and 3 ATOMs
Oxide's LLVM NVPTX backend emits `LD.E` (space-generic) instead of the
typed `LDG.E` nvcc uses. Functionally identical on Blackwell, but it
sits on a less-optimized scheduling lane in some micro-benchmarks. More
revealingly, the oxide kernel has **3 ATOMs** (final store path) vs
nvcc's 0 — the oxide version does one per block at kernel exit; nvcc
folds the final store differently.

### cuTile — TMA bulk copy + warp-shuffle tree
```
// Prologue: issue 4 TMA loads in parallel, each pulling TILE_SIZE=1024
// elements from global straight into shared memory. Tail predicated.
@UP3 UTMALDG.1D [UR16], [UR10], desc[URZ]
@UP1 UTMALDG.1D [UR12], [UR10], desc[URZ]
@UP2 UTMALDG.1D [UR4],  [UR10], desc[URZ]
@UP0 UTMALDG.1D [UR12], [UR10], desc[URZ]
...
BAR.SYNC.DEFER_BLOCKING 0x0                        // wait for TMA arrival
FADD R13, R4, R13                                  // partial accumulators
FADD R4,  R6, R15                                  // (unrolled 4-way)
FADD R9,  R5, R9
FADD R6,  R7, R11
FADD R4, R13, R4                                   // tree-reduce within block
FADD R9,  R9, R6
BAR.SYNC.DEFER_BLOCKING 0x0
FADD R4, R4, R9                                    // single-reg result
SHFL.BFLY R5, R4, 0x10, 0x1f                       // warp reduce
FADD R5, R4, R5
SHFL.BFLY R2, R5, 0x8, 0x1f
...
```

## 3. Why cuTile wins +11%

The instruction counts look *worse* for cuTile (489 vs 169 SASS lines,
3× more `BAR.SYNC` sites). So why is it faster? Three concrete SASS-level
differences:

1. **TMA bulk loads (`UTMALDG.1D`)**: a single instruction issues a
   large (1024-element) tile copy from global straight into shared
   memory, consuming `MaxWarp` bandwidth without tying up register
   file or the LSU for per-thread LDG issue. At 256M elements the
   reduction is bandwidth-bound; reducing the *number* of memory
   transactions in flight (1 per tile vs hundreds per tile) improves
   effective DRAM → L2 → SM throughput. **This is the dominant
   mechanism.**

2. **`SYNCS.*` async-mbarrier / `BAR.SYNC.DEFER_BLOCKING`**: cuTile
   uses Blackwell's new deferred barrier variant, which lets in-flight
   instructions overlap the barrier wait. The nvcc/oxide `BAR.SYNC` is
   the classic blocking form. Count is higher in cuTile (8 vs 1), but
   each is cheaper per-cycle and overlaps with subsequent work.

3. **Block tile size = 1024, grid = 4096**: cuTile processes 4M
   elements per grid pass, same as the oxide layout. The fundamental
   algorithm is identical (grid-stride loop → warp-shuffle reduce →
   atomic_add to out[0]). The +11% is NOT algorithmic; it's the TMA
   path vs the LSU path.

What does **not** explain the gap:
- No vectorized loads anywhere (`LDG.E.128` count = 0 across all three).
- No tensor-core use (irrelevant for reduction but confirms).
- Atomic count is HIGHER in oxide than cuTile (3 vs 1). If anything
  oxide should be slower for that reason; cuTile does 1 atomic_add per
  block.

## 4. Inner-loop latency math (rough)

For 256M elements, 4096 blocks, 1024 threads/block, cuTile does
`256M / (4096 * 1024) = 64` tile-loads per block. Each tile =
`1024 * 4 = 4 KiB`. One `UTMALDG.1D` per 4 KiB = 256 KiB of TMA bulk
load per block, vs nvcc's 256 KiB of individual `LDG` (≥ 32 LDGs per
warp × 32 warps × 64 iters = 65536 LDG issues per block). The
**instruction issue rate on the LSU** is the distinguishing factor.

## 5. Confidence and caveats

- sm_120 native cubins for both nvcc & oxide (oxide PTX is `.target
  sm_89` but we re-`ptxas`'d to sm_120 here; verified identical
  FFMA/LDG counts as what the driver would JIT).
- cuTile cubin extracted via monkeypatched `compile_tile`; confirmed
  valid `ELF 64-bit LSB, NVIDIA CUDA, sm_120` by `file` and by
  `cuobjdump` accepting it.
- No perf experiment was re-run here — we're explaining the +11%
  measured in Wave 12 (1696 vs 1522 GB/s). A future experiment that
  rewrites the nvcc reduction with `cp.async.bulk` + tile loads would
  test whether the TMA path is really the causal factor.
- CSV: `instruction_counts.csv` (same folder).
