# Wave 22.3 — Padded-SMEM variant analysis

## Status
✅ Compile + correctness PASSED at M=N=K=64
✅ TC engagement preserved (HMMA.16816.F32.BF16 present in SASS)
⚠️ Same static LDS instruction mix as W21 baseline — but SMEM bank
   alignment is meaningfully different (see below).

## Files
- `matmul_bf16_padded.mojo` — variant kernel (BK_PAD=BK+8=40, BN_PAD=BN+8=72)
- `matmul_bf16_padded.sass` — captured SASS (390 lines)
- `run_padded.stdout` / `run_padded.stderr` — full run log
- `matmul_bf16_padded.mojo` allocates padded SMEM via
  `Layout.row_major(BM, BK_PAD)` / `Layout.row_major(BK, BN_PAD)`,
  then exposes a `(BM, BK)` / `(BK, BN)` non-padded *view* with
  explicit stride `(BK_PAD, 1)` / `(BN_PAD, 1)` via
  `Layout(IntTuple(...), IntTuple(...))` — this lets `copy_dram_to_sram_async`,
  `.tile[WM, BK]`, `.tile[MMA_M, MMA_K]` all see clean divisibility while the
  underlying memory honors the padded row stride.

## Numerical result
| metric | M=N=K=64 |
|---|---|
| max_abs_err | 2.38e-7 |
| max_rel_err | 3.01e-7 |
| status | PASSED |

(Identical to W21 baseline at M=64: 2.4e-7 / 3.0e-7. Bit-for-bit equivalent
within the comparator's display precision — confirms padded variant is
mathematically equivalent and the padding bytes are correctly skipped.)

## SASS evidence — instruction mix per K-tile pass

| instruction | W21 baseline | W22.3 padded | delta |
|---|---:|---:|---:|
| HMMA.16816.F32.BF16 | 16 | 16 | 0 |
| LDS.U16 (16-bit smem load) | 32 | 32 | 0 |
| LDS (32-bit smem load) | 19 | 19 | 0 |
| LDGSTS.E.64 (cp.async DRAM→SMEM) | 32 | 32 | 0 |
| LDSM (ldmatrix.x4) | 0 | 0 | 0 |

(Padded run measured 64/32/38/64 because at M=N=K=64 the K-loop is
unrolled to 2 iterations vs W21 at M=N=K=4096 where K/BK=128 → loop is
not unrolled.  Per-pass numbers are identical.)

**Reading:** padding does NOT change which Mojo instruction class the kernel
emits. The static count is identical. The optimization, if it materializes,
is purely a runtime *bank-conflict-replay* effect on the LDS.U16 / LDS loads.

## Bank-conflict alignment shift (the actual point of padding)

The B_smem row stride moves from 128 bytes (W21) to 144 bytes (W22.3).
Confirmed at SASS level: B_smem LDS.U16 offsets in W21 step `0x0 → 0x80`
(128 B) between adjacent rows, in W22.3 they step `0x0 → 0x90` (144 B).

- W21:    row stride = 128 B = 32 banks → row_i mod 32 banks = 0 for all i.
  When 32 threads in a warp stride across rows of B_smem, they all hit
  bank 0 → 32-way conflict serializes into 32 separate transactions.
- W22.3:  row stride = 144 B = 36 banks → row_i mod 32 = (i*4) mod 32.
  Row 0=bank 0, row 1=bank 4, row 2=bank 8, … row 8 wraps to bank 0 again.
  An 8-row stripe of warp-parallel reads spans 8 distinct banks → much less
  serialization.

A_smem moves 64 B → 80 B (BK*2 → BK_PAD*2). 80 mod 32 = 16, so adjacent rows
land on banks 16 apart → 2-row-stripe no-conflict.

## R2's "ldmatrix.x4 → 128-bit LDS" hypothesis: refuted at the SASS level

R2 in Wave 21 hypothesized that padded SMEM would let `ldmatrix.x4`
issue 128-bit loads instead of serializing into 32-bit reads. Both this
variant and the W21 baseline emit **zero LDSM (ldmatrix) instructions**.
Mojo's `TensorCore.load_a/load_b` lowers to scalar `LDS.U16` (16-bit) +
`LDS` (32-bit) loads, not `ldmatrix`. So the hypothesized 128-bit-LDS
upgrade can't materialize — the missing primitive is `ldmatrix` itself,
which is upstream of any padding decision.

The padding still has bank-conflict-replay value on the existing scalar
`LDS.U16` loads, but it cannot be the 5-10% R2 estimated; that estimate
implicitly assumed `ldmatrix` was already in play.

## Pitfalls discovered

1. **`alias` is deprecated for compile-time bindings as of the Mojo we
   tested**. Using `alias X = Layout(...)` inside a function emits a
   warning recommending `comptime`. Both still compile correctly. Used
   `alias` here for backward compatibility with the W21 file's style;
   should migrate when we touch the W21 file again.
2. **`stack_allocation()` requires the source LayoutTensor to have an
   all-known compile-time layout**. The padded `Layout.row_major(BM, BK_PAD)`
   meets this requirement, but a custom `Layout(IntTuple, IntTuple)` for
   the *view* would not — must construct the view via the LayoutTensor
   non-allocating constructor `LayoutTensor[..., view_layout, ...](ptr)`.
   Got this right on the first try by explicitly using the underlying
   pointer from the padded allocation.
3. **`.tile[BM, BK]` with mismatched divisibility silently miscompiles or
   fails**. Padded(BM, BK+8).tile[BM, BK] would try to interpret the
   tensor as a `(1, 1.25)` tile grid. Avoiding this required using a
   non-padded `(BM, BK)` *view* with explicit stride for all subsequent
   slicing — never call `.tile` on the padded layout directly.
4. **`tee | head -200` ate the stdout's correctness print** because the
   SASS dump itself is ~400 lines long. Must use `> stdout 2> stderr` and
   then `tail` separately — this is a generic gotcha for Mojo runs that
   use `_dump_sass=True` at small problem sizes.
5. **Padding bytes are read by cp.async-write logic only?** No — bytes at
   columns [BK..BK_PAD) of A_smem are written by **nothing** (the copy
   only writes the first BK columns of each row), and read by **nothing**
   (the (BM,BK) view does not address them). They contain whatever was in
   SMEM at allocation time. Correctness-safe but worth flagging if any
   future variant tries to re-use those bytes.

## What this wave does NOT prove

- **No timed bench was run** (per task scope). We do not know the
  actual perf delta. The W21 reviewer's 5-10% estimate assumed
  `ldmatrix.x4` lift; that lift is unreachable through padding alone, so
  the realistic delta is whatever bank-conflict-replay savings the
  scalar `LDS.U16` path picks up — possibly smaller than 5%, possibly
  zero, depending on whether the compiler already scheduled the
  W21-baseline LDS.U16 burst around bank conflicts.

## Wave 22.4 / future candidates

- Pair this padded layout with a **manual `ld_matrix.sync.aligned.x4` call**
  (raw PTX inline) instead of `TensorCore.load_a/load_b`. *That* combination
  would actualize R2's full 5-10% hypothesis. The padded layout is a
  prerequisite — `ldmatrix.x4` from a row-major(64,32) source has the
  classic 8-way bank conflict that padding eliminates.
- Repeat with **BK_PAD = BK + 4** (smaller pad) and compare bank
  alignment. Pad+4 = 36 elements = 72 bytes = bank stride 8 mod 32 →
  4-way wraparound — likely worse than +8's 8-way. Pad+8 chosen here
  is the cleanest first try.
- **Sweep at M=N=K=4096** to get the actual TFLOPS delta vs W21's 79.3,
  once we are willing to spend a bench slot.
