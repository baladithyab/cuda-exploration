# Wave 17 — cuda-oxide MLA implementation design

**Scope.** Scope a pedagogical cuda-oxide MLA attention cell (DeepSeek-V3
shape: B=1, S=2048, n_h=128, qk=d_h+d_rope=192, d_v=128) as a minimal diff
off `oxide-attn-gqa` (Wave 16.1, 24.15 TF at GQA bench shape). The cuTile
reference (`cutile-attn-mla`, 112.4 TF) pads 192→256 for tensor-core lane
alignment, eating a documented 25% flop waste. The oxide cell has no TC
engagement at all, so the padding argument does not transfer — this doc
recommends the 192-native path and sizes the delta.

All line numbers cite `oxide-attn-gqa/src/main.rs@862` and
`oxide-matmul-tiled-microtile/src/main.rs@375`.

## 1. Tile geometry: 192 native, BM=BN=64, BK=16 (unchanged)

**Recommendation: 192 native. Do not pad to 256.**

The only reason cuTile pads is that `ct.mma` on sm_120 wants power-of-two
inner dims for HMMA lane packing (Wave 13.1 finding, cited in
`cutile-attn-mla/ANALYSIS.md:55-77`). Oxide emits scalar FFMA through
`core::intrinsics::fmuladdf32` (see qkt inner loop at
`oxide-attn-gqa/src/main.rs:151-182`); scalar FFMA has no alignment
requirement on the K-loop trip count.

Both candidates divide cleanly into the existing K-loop tile `BK=16`:

| Inner dim | num_tiles = qk/16 | wasted FFMA/element | register accum |
|---|---:|---:|---:|
| **192** | **12** | **0** | 16 (unchanged) |
| 256 (padded) | 16 | 25% | 16 (unchanged) |

Register pressure is insensitive to inner dim — the 4×4 microtile holds 16
f32 output accumulators per thread regardless (`oxide-attn-gqa/src/main.rs:102-105`).
Shared-memory pressure is likewise unchanged: `TILE_Q` and `TILE_K` are
sized per K-tile slab (64×16 = 1024 f32 each,
`oxide-attn-gqa/src/main.rs:73-74`), not per full K-dim. A 192-wide inner
dim just runs the outer K-loop 12 times instead of 8 (GQA d=128) or 16
(padded 256). No kernel structure changes.

**Keep BM=BN=64, BK=16.** These are proven at the Wave-7 matmul ceiling
(45 TF f32, `oxide-matmul-tiled-microtile/src/main.rs:29-34`) and the
Wave 16.1 GQA cell (24.15 TF) — the QKt shape only differs in inner dim,
which is orthogonal to the output-tile geometry. Changing BM/BN would
require re-deriving the cooperative-load factors (currently 256 threads ×
4 elems = 1024 = BM×BK,
`oxide-attn-gqa/src/main.rs:115-129`). Not worth it for a shape-only
variation.

## 2. Expected TFLOPS at MLA bench shape

**Range: 20–24 TF (roughly in-line with GQA's 24.15 TF, possibly 5–15%
below).**

Reasoning (arithmetic, not measurement):

- **QKt stage.** GQA QKt is 2.846 ms total × 42% (wave16-summary.md:43) ≈
  1.19 ms for 4·1·32·2048²·128 ≈ 6.87e10 FLOP → ~58 TF kernel-local.
  MLA QKt does 4·1·128·2048²·192 ≈ 4.12e11 FLOP, i.e. **6× more work**
  (4× from n_h, 1.5× from qk). At the same kernel-local throughput this
  takes ≈ 7.2 ms.
- **Softmax stage.** Seq is unchanged (2048) but the grid is
  B·n_h·seq = 4× larger (128 rows per batch vs 32). Softmax is HBM-bound
  on the scores matrix, which is 4× the size → ~0.75 × 4 = 3.0 ms.
- **PV stage.** Seq and d_v=128 identical to GQA's PV shape; 4× more
  heads → ≈ 1.12 × 4 = 4.5 ms.
- **Total ≈ 14.7 ms.** FLOP count = 4·1·128·2048²·(192+128) ≈ 6.87e11 →
  **≈ 46.7 TF** … but this is the upper bound assuming stages scale
  perfectly linearly in head count.

Two reasons the real number will be lower:

1. **Occupancy already saturated.** GQA's QKt launches
   `(32, 32, 32) = 32768 blocks`; MLA launches `(32, 32, 128) = 131072`.
   RTX 5090 has ~170 SMs with multi-block/SM resident; more work
   per-wave won't raise throughput, and wave-tail overhead grows.
2. **HBM pressure on softmax.** Scores buffer is B·n_h·S² =
   128·2048²·4 B = **2.0 GB** (vs GQA's 512 MB). Still fits in 32 GB
   HBM but the round-trip cost per element is the same; softmax stage
   is fully HBM-bound, no speedup.

**Estimate: 20–24 TF headline.** Lower end if softmax HBM-dominates
(plausible: MLA scores = 4× GQA scores, softmax was already 23% of GQA
runtime). vs cuTile's 112.4 TF this is 18–21% — the expected gap given
the (HMMA=384 vs HMMA=0) disparity, plus the 3-kernel vs fused HBM
round-trip that Wave 16 measured at 6.8× for GQA.

(One wildcard: at 192-native inner dim we avoid 25% of cuTile's wasted
flop on QKt, so **the gap might shrink slightly on QKt alone** — 192/256
= 0.75× flops for cuTile-MLA on zero-zero products. But that's 25% of
one stage for cuTile; doesn't close the overall 5× gap.)

## 3. LOC estimate for `src/main.rs`

**Expected: ~890 LOC (+28 vs GQA's 862).**

Budget breakdown from diff against `oxide-attn-gqa/src/main.rs`:

- **Kernel bodies (3):** ≈0 net lines. `gqa_qkt_kernel` becomes
  `mla_qkt_kernel`, `gqa_pv_kernel` becomes `mla_pv_kernel`, softmax
  kernel unchanged byte-for-byte.
- **Shape struct + consts:** +8 lines. Add `qk_head_dim` field
  (`oxide-attn-gqa/src/main.rs:446-453`), update `SHAPE_BENCH` to
  DeepSeek-V3 params, add a `SHAPE_CORRECTNESS` for MLA-small.
- **`run_attention` signature:** +4 lines. Separate `qk` and `d_v`
  parameters, pass both to QKt and PV kernels respectively.
- **Launch configs:** +6 lines. PV grid gridX = d_v/64 = 2 (same as
  GQA); QKt grid unchanged but scale = 1/sqrt(qk) not 1/sqrt(d).
- **Flop count helper:** +2 lines. `mla_flops()` uses
  `4·B·n_h·S²·(qk+d_v)/2` not `4·B·n_q·S²·d`. Actually it's
  `2·B·n_h·S²·qk + 2·B·n_h·S²·d_v`.
- **Input paths:** +4 lines. Replace `gqa_*` paths with `mla_*`; expect
  generator script at `analysis/wave15-*/inputs/mla_*_{q,k,v,expected}_f32.npy`.
- **Banner strings:** +4 lines trivial (wave/impl names).

## 4. Concrete deltas vs `oxide-attn-gqa/src/main.rs`

Line-by-line diff summary (everything else: verbatim copy):

1. **Drop GQA broadcasting.** MLA has n_kv = n_h (one K/V per Q head), no
   groups:
   - Line 85: `let groups = (n_q / n_kv) as usize;` → delete.
   - Line 90: `let h_kv = h_q / groups;` → `let h_kv = h_q;` (or just
     inline as `h_q`).
   - Line 93, 334: `k_base` / `v_base` use `h_q` directly, no divide.
   - Signature: drop `n_kv: u32` param from both matmul kernels (or keep
     for API symmetry and assert `n_kv == n_q` on launch).
2. **Generalize `d_head` into two params.** MLA has asymmetric
   Q/K head dim (qk=192) vs V head dim (d_v=128):
   - `gqa_qkt_kernel(..., d_head: u32, ...)` → `mla_qkt_kernel(..., qk_head_dim: u32, ...)`.
     Inner `num_tiles = qk_head_dim / 16` = 12 at bench shape.
   - `gqa_pv_kernel(..., d_head: u32, ...)` → `mla_pv_kernel(..., d_v: u32, ...)`.
     Inner `num_tiles = seq / 16` = 128 (unchanged — PV's K-dim is seq).
   - `Shape` struct gains `qk: usize` (was `d_head`); `d_head` renames
     to `d_v`.
3. **Scale factor.** Line 610: `1.0 / (d_head as f32).sqrt()` →
   `1.0 / (qk as f32).sqrt()`. Uses the **true** qk=192 not any
   padded value (trivially correct since we don't pad).
4. **PV launch grid X.** Line 630:
   `(shape.d_head / 64) as u32` → `(shape.d_v / 64) as u32` = 2.
   Unchanged value, renamed.
5. **`gqa_flops` → `mla_flops`.** Line 704-707:
   `4.0 * B * n_q * seq² * d_head` →
   `2.0 * B * n_h * seq² * qk + 2.0 * B * n_h * seq² * d_v`
   (QKt + PV FLOPs; softmax is ignored as in GQA).
6. **Host buffer sizes.** Lines 741-749:
   - `q_elems = B · n_h · S · qk` (note `qk` not `d_head`).
   - `k_elems = q_elems` (no KV sharing, so K is same size as Q).
   - `v_elems = B · n_h · S · d_v`.
   - `out_elems = v_elems` (output is Q-shape with d_v width).
7. **Input file paths.** Lines 718-722: `gqa_{name}_*.npy` →
   `mla_{name}_*.npy`. A separate generator
   (`analysis/.../pytorch_reference_mla.py`) must exist — Wave 16.3
   cuTile-MLA used one; reuse or regenerate.

**No changes** required to: softmax kernel (byte-for-byte copy), cooperative
load math in the matmul kernels (TILE_Q/TILE_K = 64×16 slab is
inner-dim-agnostic), FFMA microtile inner loop, event timing harness, NPY
IO.

## 5. Oxide-specific pitfalls

1. **K-loop trip count 12 is fine.** The while loop at
   `oxide-attn-gqa/src/main.rs:110-185` is runtime-bounded; libNVVM
   does not unroll it. 12 vs 8 iterations costs ~1.5× QKt runtime and
   is reflected in the §2 estimate. There is no correctness or codegen
   risk from qk=192 not being a power of two — BK=16 divides it.
2. **Don't forget the FMA intrinsic.** All 16 `fmuladdf32` calls
   (`oxide-attn-gqa/src/main.rs:164-179`) must survive copy-paste. If any
   gets rewritten as `a * b + c`, Wave 3/7's finding applies: libNVVM
   won't contract it to hardware FMA and runtime drops ~30%. Grep for
   `fmuladdf32` on post-copy; expect exactly 32 occurrences (16 qkt + 16
   pv).
3. **Scores buffer is 2 GB.** `scores_elems = B·n_h·S² = 128·2048²·4 B`
   = 2.0 GiB for scores + 2.0 GiB for probs = **4 GiB device memory**
   just for the attention matrix. RTX 5090 has 32 GB HBM, fine, but
   the `from_host` upload path will take ~2s per buffer at PCIe4 speeds
   — add a note about load time in the banner.
4. **Correctness shape.** For MLA correctness, pick something divisible
   by 64 (BM=BN) for seq, divisible by 16 for qk and d_v.
   Suggest: `batch=1, seq=128, n_h=4, qk=48, d_v=32`
   (qk=48 gives 3 K-tiles — validates the qk/16 divisibility path with
   a non-power-of-two value). GQA's correctness shape
   (`oxide-attn-gqa/src/main.rs:455-462`) used qk=d_head=64; we want
   something that exercises the non-power-of-two case at least once.
5. **fmuladdf32 and 192.** No interaction. The intrinsic is per-element;
   the inner-dim sweep count is opaque to it.

## 6. Summary table

| Dimension | GQA (Wave 16.1, shipped) | MLA (proposed) |
|---|---|---|
| Inner dim Q/K | 128 (power-of-two) | **192 (native, no pad)** |
| Inner dim V | 128 | 128 |
| Heads Q | 32 | 128 |
| KV heads | 8 (groups=4) | 128 (groups=1, strip) |
| QKt K-loop trips | 8 | 12 |
| BM × BN × BK | 64 × 64 × 16 | **64 × 64 × 16 (unchanged)** |
| Microtile | 4 × 4 | 4 × 4 (unchanged) |
| LOC | 862 | **~890 (+28)** |
| Expected TF at bench | 24.15 (measured) | **20–24 (estimated)** |
| vs cuTile-MLA 112.4 TF | — | **18–21%** |
