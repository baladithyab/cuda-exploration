# Wave 22.2 — mojo-matmul-f16 (author + correctness only)

**Status:** ✅ Authoring + correctness PASS at M=N=K=64 (Wave 22.2 subagent, 2026-05-21)
**Orchestrator:** writes the final analysis after the serial timed bench run at M=N=K=4096.

## What ships in this cell

- `matmul_f16.mojo` — hand-rolled f16-in/f32-acc tiled matmul. Cloned from
  `mojo-matmul-bf16/matmul_bf16.mojo` with three substantive changes (see
  "Diff vs bf16 cell" below).
- `matmul_f16.sass` — captured SASS at M=N=K=64 (395 lines, extracted from `_dump_sass=True`).
- `run.sh`, `.gitignore` — same scaffold as the bf16 cell.
- `run.log` — full stdout (gitignored).

## Correctness gate — PASSED at M=N=K=64

| metric | value | tolerance | result |
|---|---|---|---|
| max_abs_err | **5.96e-7** | atol=1.0 + rtol=1e-2·\|ref\| | ✅ PASS |
| max_rel_err | 5.02e-7 | (loose, see below) | ✅ |
| target | sm_120a | (consumer Blackwell native) | ✅ |
| HMMA.16816.F32 count | **32** | > 0 expected | ✅ |
| LDGSTS count (cp.async) | 64 | mirrors bf16 cell | ✅ |
| UTMALDG count (TMA) | 0 | (Wave 22.1 candidate) | — |

Tolerance per task spec was `atol=1.0 + rtol=1e-2·|ref|`, looser than bf16's
`atol=1e-2 + rtol=1e-3·|ref|` — accommodates f16's narrower range (5-bit
exponent vs bf16's 8-bit). At M=N=K=64 with input magnitudes in [0, 0.255],
expected output magnitude tops out at ~4.16, so the band is ~1.04. Observed
error of 5.96e-7 is **f32-accumulator-noise level** — well below even
bf16's tighter tolerance, because at K=64 the differential summation order
between kernel and CPU ref accumulates very little.

The HMMA.16816 SASS instructions show 2× Wave 21's count (32 vs 16) because
the compiler fully unrolled the outer K-loop at the small problem size
(K/BK=2 iterations × 16 inner-unroll MMAs/iter = 32). Per-iteration MMA
position count matches Wave 21 exactly.

## Diff vs the bf16 cell (mojo-matmul-bf16)

Three changes from `matmul_bf16.mojo`:

1. **`a_type = DType.float16`** everywhere (was `DType.bfloat16`). All
   buffers, LayoutTensors, SIMD frags, and TensorCore loaders flip dtype.
   The host-init pattern `(Float32(...) * 0.001).cast[a_type]()` carries
   over unchanged (pitfall #12 — implicit `Float32 → bf16` cast
   disallowed — applies identically to `Float32 → f16`).

2. **Custom `mma_m16n8k16_f16_f32` helper** replacing the call to
   `from std.gpu.compute.mma import mma`. The Mojo 1.0.0b1 stdlib
   dispatcher `_mma_nvidia` has a lane for bf16 m16n8k16 with f32 acc
   but **no lane for f16 m16n8k16 with f32 acc**. Confirmed by reading
   `mma_nvidia.mojo` upstream:

   > `elif _has_type[(DType.float16, DType.float16, DType.float32, DType.float32)]
   >     and _has_shape[(4, 2, 4, 4)]:` → m16n8**k8** only

   Calling the std `mma()` with `(SIMD[f16, 8], SIMD[f16, 4], SIMD[f32, 4])`
   triggers `_unsupported_mma_op` at compile time:

   ```
   constraint failed: no valid implementation of mma for
   a=8xfloat16, b=4xfloat16, c=4xfloat32, and d=4xfloat32
   ```

   The underlying LLVM NVPTX intrinsic
   `llvm.nvvm.mma.m16n8k16.row.col.f32.f32` **does exist** (per
   `llvm/include/llvm/IR/IntrinsicsNVVM.td`: `m16n8k16:a:f16` is registered
   as `4 × v2f16`, `c/d:f32` as `4 × float`). The cell calls this
   intrinsic directly via `llvm_intrinsic[...]`, bypassing the dispatcher
   gap. Fragment packing matches the m16n8k16 16-bit per-lane PTX 9.7.13.4.6
   distribution: A is 8 f16/lane → 4 v2f16; B is 4 f16/lane → 2 v2f16.

3. **Tolerance loosened** to `atol=1.0 + rtol=1e-2·|ref|` per task spec
   (vs bf16's `atol=1e-2 + rtol=1e-3·|ref|`). Authoring-only at small N;
   orchestrator will set the M=4096 tolerance from observed behavior in
   the timed run.

Everything else (TensorCore-load_a/load_b hybrid, hand-rolled epilogue,
tile shape, smem layout, async-copy thread layout) is byte-identical to
the bf16 cell. The Wave 21 hybrid pattern transfers cleanly.

## Pitfalls discovered

**P1 (NEW — Mojo stdlib gap).** `std.gpu.compute.mma._mma_nvidia` in
Mojo 1.0.0b1 has the bf16 m16n8k16 f32-acc dispatch lane but **not** the
f16 m16n8k16 f32-acc lane. Only m16n8k**8** (4,2,4,4) is dispatched for
f16-in/f32-acc. The fix: call
`llvm_intrinsic["llvm.nvvm.mma.m16n8k16.row.col.f32.f32", ...]` directly,
splitting A (8 f16) into 4×v2f16 and B (4 f16) into 2×v2f16. The
underlying LLVM intrinsic exists upstream (`IntrinsicsNVVM.td` WMMA_REGS
fragment table). This is a candidate for an upstream Mojo PR — copying
the bf16 lane and swapping the type predicate to `DType.float16` would
add the lane in ~20 lines.

**P2 (NEW — SASS suffix convention).** The expected
`HMMA.16816.F32.F16` literal does **not** appear in SASS. The
disassembler emits `HMMA.16816.F32` with **no `.F16` suffix** for
f16-in/f32-acc, because **f16 is the implicit default for HMMA**.
The `.BF16` suffix appears only when inputs are bf16. So the correct
verification predicate is
`grep 'HMMA.16816.F32' | grep -v '.BF16'`, NOT
`grep 'HMMA.16816.F32.F16'`. Confirmed via the PTXAS reverse-engineering
reference (gh.evko.io/nvopen-tools/ptxas) which lists the SASS counter
names as `hmma16816` (f32 acc, f16 in — default) and `hmma16816f16`
(f16 acc, f16 in). The bare `HMMA.16816.F32` IS the f16-in/f32-acc
instruction we want.

**P3 (CARRIES — pitfall #12 from skill).** `Float32 → BFloat16` implicit
cast disallowed in host-buffer init applies identically to `Float32 →
Float16`. The bf16 cell already wraps in `.cast[a_type]()`; no new fix
needed — the same code works as-is when `a_type = DType.float16`.

## What the orchestrator does next

- Run the timed bench at M=N=K=4096 serially with the same kernel, scaling
  problem size by editing `comptime M = N = K = 4096` (currently 64) and
  re-adding the timed `execution_time` loop from the bf16 cell. The
  authoring subagent kept the kernel + harness simple (single launch, full
  CPU correctness check) since the orchestrator runs the bench.
- Compare against Wave 21 bf16 (79.3 TF) — expectation per Wave 21
  W22.2 candidate notes: "should land ~80 TF (parity with bf16, both
  engage TC at the same shape)."
- Update `references/mojo-mma-shapes.md` (rust-gpu-compute skill) to add
  P1 (the f16 m16n8k16 dispatcher gap) and P2 (the SASS suffix
  convention) to the pitfalls list.
