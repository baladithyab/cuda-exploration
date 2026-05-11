# Wave 13.2 — SASS diff: tiled matmul (cuTile vs cuda-oxide vs nvcc)

> **Headline**: at f32 on RTX 5090 (sm_120), **none** of the four matmul
> kernels compiled here emits HMMA/BMMA/IMMA/TCGEN05 — tensor cores are
> never engaged for f32 matmul on Blackwell in this build of cuTile v1.3.0.
> Oxide and nvcc emit tight `FFMA` inner loops (192 and 256 FFMAs
> respectively, with `.reuse` operand hints). **cuTile emits NO `FFMA` at
> all** — its compiler lowers `ct.mma` at f32 as separate `FMUL + FADD`
> pairs (2049 FMUL + 2176 FADD in the big `matmul_tiled` variant). That is
> the 5× slowdown: no FMA contraction, and no tensor-core use.

All cubins built for sm_120 native. Disassembled with `cuobjdump
--dump-sass` (CUDA 13.2).

## 1. Kernel-level instruction counts

| metric                    | cutile matmul_tiled (BM=BN=128, BK=16) | cutile matmul_tiled_simple (16×16) | oxide microtile (64×64, 4×4 reg) | cuda tiled (BLOCK=16) |
| ------------------------- | -------------------------------------: | ---------------------------------: | -------------------------------: | --------------------: |
| total SASS lines          | **12169**                              | 1097                               | 1278                             | 1049                  |
| **FFMA**                  | **0**                                  | **0**                              | 192                              | 256                   |
| **HMMA**                  | 0                                      | 0                                  | 0                                | 0                     |
| **BMMA / IMMA / TCGEN05** | 0                                      | 0                                  | 0                                | 0                     |
| **FMUL**                  | **2049**                               | 64                                 | 0                                | 0                     |
| **FADD**                  | **2176**                               | 68                                 | 0                                | 0                     |
| LDS / LDS.128             | 96 / 96                                | 48 / 16                            | 24 / 24                          | 32 / 32               |
| STS / STS.128             | 32 / 32                                | 2 / 0                              | 16 / 0                           | 16 / 0                |
| LDG.E                     | 0                                      | 0                                  | 16                               | 16                    |
| LDG.E.CONSTANT            | 0                                      | 0                                  | 0                                | **16**                |
| **UTMALDG.2D** (TMA 2D)   | **9**                                  | 2                                  | 0                                | 0                     |
| **UTMASTG.2D** (TMA 2D)   | **4**                                  | 0                                  | 0                                | 0                     |
| STL (local stores!)       | **233**                                | 0                                  | 0                                | 0                     |
| LDL (local loads!)        | **265**                                | 0                                  | 0                                | 0                     |
| SYNCS.* (async mbar)      | 33                                     | 53                                 | 0                                | 0                     |
| BAR.SYNC                  | 1                                      | 1                                  | 4                                | 2                     |
| BRA                       | 39                                     | 35                                 | 8                                | 3                     |

(See `instruction_counts.csv`.)

## 2. Hot-loop structure — the 5× difference is right here

### nvcc tiled (BLOCK=16, `matmul_tiled` C++)
```
FFMA R73, R20.reuse, R16, R73        // acc += a.row * b.col  (fused)
FFMA R70, R20.reuse, R17, R70        // `.reuse` = register-file cache hit
FFMA R71, R20.reuse, R18, R71        // compiler vectorized K-loop over 16 lanes
FFMA R20, R20,       R19, R14        // 4-way inner tile
FFMA R76, R24.reuse, R16, R39
FFMA R72, R24.reuse, R17, R72
FFMA R43, R24.reuse, R18, R43
FFMA R42, R24,       R19, R42
... (256 FFMA total, fully unrolled K inner loop)
```
Classic tiled-GEMM SASS: one FFMA per inner-iter-element with register
reuse hints, LDG.E.CONSTANT on the constant-side operand (B matrix is
hoisted to constant-load path), LDS.128 for bulk shared-tile reads.

### oxide microtile (64×64 block, 4×4 register-tile)
```
FFMA R41, R24.reuse, R36, R41       // 4×4 register microtile, 192 FFMAs total
FFMA R62, R24.reuse, R37, R62       // same .reuse pattern as nvcc
FFMA R63, R24.reuse, R38, R63
FFMA R24, R24,       R39, R30
...
```
192 FFMAs = (4×4) × 12 iterations of a K-chunked loop. `.reuse` hints
preserved. The only substantive difference from nvcc: oxide uses `LDG.E`
(space-generic-typed) instead of `LDG.E.CONSTANT`. Otherwise the cores
of the kernels are structurally the same — and the bench reflects that:
45.05 TF (oxide) vs 38.41 TF (nvcc), with oxide in front because of
bigger block tile & register microtile.

### cuTile `matmul_tiled` (BM=128, BN=128, BK=16 via `ct.mma`)
Inner "matmul" portion:
```
FMUL R36, R9.reuse, R36   ;            // multiply only
FMUL R6,  R9,       R32   ;
FMUL R4,  R11.reuse, R37  ;
FMUL R38, R11,       R32  ;
...
FADD R36, R36, R4         ;            // SEPARATE add, NOT fused
FADD R6,  R6,  R38        ;            // round-trip: acc += FMUL result
...
```
and surrounding async TMA + warp-group synchronization:
```
UTMALDG.2D [smem], [desc], UR4         // 2D bulk TMA load
SYNCS.PHASECHK.TRANS64.TRYWAIT P0, ... // wait for arrival
LDS.128 R36, [R29]                     // spill from smem to reg
FMUL ... ; FMUL ... ; FADD ... ; FADD ... ; (no fusion)
SYNCS.ARRIVE.TRANS64.A1T0 ...          // signal next stage
```

Plus **233 STL + 265 LDL**. The kernel is so register-pressured that
it spills heavily to local memory (= L1/L2-backed thread-local DRAM).
That is a catastrophic pattern for any matmul.

## 3. Why cuTile is 5× slower at f32 matmul

Three concrete causes, from most to least impactful:

### (a) NO FMA contraction — FMUL/FADD emitted separately
4225 separate floating-point ops (2049 FMUL + 2176 FADD) where nvcc
emits 256 fused `FFMA`. At one instruction per cycle per warp, that's
16.5× the inner-loop issue count. Even if 3/4 of those FADDs reduce
across the block, the base cost is at least ~4× nvcc on the FP-issue
lane alone. This is identical to the cuda-oxide Wave-1 finding (FMA
contraction disabled) — cuTile 1.3.0 appears to emit f32 matmul with
IEEE-strict rounding and does not contract.

### (b) NO tensor cores at f32 on sm_120 in cuTile 1.3.0
Zero `HMMA` / `BMMA` / `IMMA` / `TCGEN05` instructions. `ct.mma` lowers
to a scalar FMUL+FADD implementation at f32. (It likely uses HMMA at
f16/bf16, but that's untested here.) This matters because on Blackwell
RTX 5090 there IS an f32 tensor-core path (`mma.sync.aligned.m16n8k8.f32`);
cublas uses it. cuTile does not. This alone would explain ~3-4× gap vs
a tensor-core-enabled baseline, but *both* oxide and nvcc are ALSO
running scalar FFMA here — so TC-omission doesn't explain the oxide/
cuTile delta directly; it explains why none of the three matches cuBLAS
(7.57 / 45.05 / 38.41 TF vs ~200 TF cuBLAS f32→TF32 at N=4096).

### (c) Local-memory spilling (233 STL + 265 LDL)
cuTile allocates a 128×128 accumulator per block (16 KiB of registers
per block if held in-reg). Blackwell max registers/thread = 255; at
256 threads × 255 regs = 65535 regs = 256 KiB per block, so *in principle*
a 128×128 f32 accumulator (16 KiB) fits. But the kernel's intermediate
state from the unrolled FMUL+FADD expansion overflows, and the
register allocator spills to local memory. 498 local-memory accesses
in the hot path are a complete performance disaster — each is a cacheable
L1 access in the best case, a DRAM round-trip in the worst.

## 4. What the simple 16×16 cuTile variant shows

`matmul_tiled_simple` (16×16 block tile, mirrors oxide-matmul-tiled
and cuda-matmul-tiled block sizes) still emits 64 FMUL + 68 FADD
(no FFMA) but does **not** spill to local (0 STL / 0 LDL). That
confirms cause (c) is tied to the big 128×128 variant; causes (a)
and (b) apply to both cuTile variants and are the fundamental
ceiling.

## 5. Confidence & caveats

- All four cubins verified `ELF 64-bit LSB, NVIDIA CUDA, sm_120`.
  cuTile cubins extracted by monkeypatching `cuda.tile._compile.
  compile_tile` to dump `result.cubin` during the existing smoke test.
- cuTile version: `cuda.tile 1.3.0` (packaged in
  `cutile-vecadd-bench/.venv`).
- The FMUL+FADD-not-fused finding is striking enough to warrant
  double-checking with another cuTile release; if `ct.mma` is
  intended to lower to a TC path on sm_120, the 2049 FMULs in the
  `matmul_tiled` cubin suggest the lowering pass chose the scalar
  path. Possible root causes:
  - cuTile's accum dtype default may differ from what `ct.zeros(
    (tm, tn), ct.float32)` promises.
  - `ct.mma` may gate on shape constraints that `(128, 128) × (128,
    16)` accumulating into `(128, 128)` doesn't meet for the f32 TC
    path on Blackwell.
  - The `cuda-tile` MLIR lowering may not have a Blackwell f32 TC
    target in 1.3.0 yet.
- A definitive check would be to re-run at dtype=float16 with
  `ct.mma` and inspect the cubin for HMMA — but that's out of scope
  for this SASS-diff task.

## 6. 5-line summary

1. **cuTile matmul SASS contains zero tensor-core instructions** at f32.
2. **cuTile matmul SASS contains zero FFMAs** — math is done via separate
   FMUL + FADD (4225 ops vs nvcc's 256 FFMAs). No FMA contraction.
3. The big `matmul_tiled` (128×128) spills heavily to local memory (233 STL
   + 265 LDL) — register pressure from the huge accumulator.
4. nvcc and oxide both use fully-unrolled FFMA inner loops with `.reuse`
   register-file hints; oxide's 4×4 microtile is measurably more efficient
   than nvcc's 16×16 tile (192 vs 256 FFMAs per block for the same output
   work — 25% fewer instructions per output).
5. cuTile reduction wins +11% because it uses TMA bulk `UTMALDG.1D` loads
   direct-to-shared; cuTile matmul loses 5× because that advantage is
   swamped by the scalar-FMUL+FADD arithmetic.
