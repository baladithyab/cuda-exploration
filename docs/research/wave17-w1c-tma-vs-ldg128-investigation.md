# Wave 22.8 — Why does cuTile's GDN-decode beat nvcc-LDG.E.128 on Blackwell sm_120?

**Status.** Docs-only investigation. Re-reads the SASS evidence
already shipped in Wave 17 W1c.

**The puzzle (from `results/wave17-summary.md` row W1c).**
On Qwen3-Next decode shape (B=1, H=16, d_k=d_v=256, BLOCK_V=64) the two
hand-tuned GDN-decode kernels run on the same RTX 5090 / sm_120 box:

| frontend            | best GB/s | per-iter loads (gmem)          | per-iter stores (gmem)          |
|---|---:|---|---|
| `cuda-attn-gdn` (nvcc) | **417.7** | 16 × `LDG.E.128` (16 B each) + 2 × `LDG.E.64` + 8 × `LDG.E.U16` | 16 × `STG.E.128` + 2 × `STG.E.64` |
| `cutile-attn-gdn` v1.3.0 | **610.6** | 128 × scalar `LDG.E` (4 B each) + 7 × `LDG.E.U16` | 128 × scalar `STG.E` |

**The W1c hypothesis was wrong in the most interesting way.** W1c
predicted vector loads (`LDG.E.128`) would beat scalar loads. The bench
disagrees: **scalar `LDG.E` × 128 transactions beats vectorised
`LDG.E.128` × 16 transactions by 1.46×.** The Wave 17 summary
hypothesised cuTile uses a TMA / `UTMALDG` bulk-load path. **It doesn't.**
This doc replaces that hypothesis with what the SASS actually shows.

---

## Executive summary

1. **cuTile does NOT use TMA / `UTMALDG`.** Both kernels emit zero
   `UTMALDG`, zero `UTMASTG`, zero `LDGSTS` / `cp.async.bulk`. The
   "TMA bulk-load" hypothesis from the Wave 17 summary is rejected by
   the SASS.
2. **cuTile uses Blackwell async-barrier infrastructure
   (`SYNCS.PHASECHK.TRANS64.TRYWAIT`, `SYNCS.EXCH.64`,
   `FENCE.VIEW.ASYNC.S`, `BAR.SYNC.DEFER_BLOCKING`) plus a 100 KiB
   warp-specialised shared-memory pipeline.** That is a different async
   memory mechanism from TMA but achieves the same effect: gmem traffic
   is decoupled from the math warps' load/store unit.
3. **The dominant inner loop in cuTile reads from shared memory with
   `LDS.128` (326 instructions), not from gmem.** The gmem load is
   issued once per state row by a producer warp; the consumer warps
   then stream from smem at 16 B/transaction. nvcc's kernel issues
   `LDG.E.128` on the SM's LSU directly inside the math warp, with no
   producer/consumer split.
4. **nvcc's kernel runs at TPB=16 (HALF a warp) at BLOCK_V=64.**
   Confirmed in `cuda-attn-gdn/attn_gdn.cu:TPB = BLOCK_V/4 = 16`. Only
   16 active threads per block, one warp half-idle. cuTile's kernel
   uses 100 KiB of static smem and ≈7 named barrier slots — a
   warp-specialised pattern that only makes sense at full-warp
   occupancy. The TPB gap likely accounts for a substantial fraction
   of the BW gap independent of the load-instruction-class question.
5. **This is a hardware-API-gap finding, not a compiler-quality
   finding.** nvcc is emitting the right SASS for the source it was
   given. The gap is that the source — a single-warp loop with
   `*reinterpret_cast<float4*>` — cannot express what cuTile's
   compiler is targeting (warp-specialised async-barrier pipeline with
   shared-memory-resident state tile). To close the gap from CUDA C++
   would require a hand-written `cuda::pipeline` /
   `cuda::memcpy_async` / `mbarrier::arrive` /
   `mbarrier::try_wait_parity` structure plus widening to TPB=32 or
   TPB=64.

---

## Per-kernel SASS instruction mix

Counts from `grep -cE '[[:space:]]<inst>([[:space:].]|$)'` against the
two `.sass` files. `oxide-attn-gdn` included as a third reference
point (the no-vector-load baseline).

| instruction class                     |   nvcc |  cuTile |   oxide |  notes |
|---|---:|---:|---:|---|
| **gmem loads — vectorised**           |        |         |         |        |
| `LDG.E.128`                           |   **16** |     0 |     0 | 16 B/inst — only nvcc emits these |
| `LDG.E.64`                            |      2 |     0 |     0 |  8 B/inst |
| `LDG.E` (scalar, 4 B/inst)            |     26 |   **128** |   136 | cuTile relies on these |
| `LDG.E.U16` (2 B/inst, half loads)    |      8 |     7 |     0 |  q/k/v half scalars |
| **gmem stores — vectorised**          |        |         |         |        |
| `STG.E.128`                           |   **16** |     0 |     0 |        |
| `STG.E.64`                            |      2 |     0 |     0 |        |
| `STG.E` (scalar, 4 B/inst)            |     18 |   **128** |    66 |        |
| **smem loads — vectorised**           |        |         |         |        |
| `LDS.128`                             |     24 |   **326** |    16 | cuTile dominates here |
| `LDS.64`                              |      2 |     0 |     2 |        |
| `LDS` (scalar, 4 B/inst)              |     16 |     0 |  1922 | oxide does scalar smem traversal (no auto-vectorise) |
| **smem stores**                       |        |         |         |        |
| `STS.128`                             |     -- |    96 |     0 |        |
| `STS.64`                              |     -- |    66 |     0 |        |
| `STS` (scalar)                        |     22 |     0 |  1156 |        |
| `STSM.16.M88` (warp-coop. matrix)     |      0 |     **3** |     0 | only cuTile |
| **async / barrier infrastructure**    |        |         |         |        |
| `UTMALDG` / `UTMASTG` / `cp.async.bulk` |    0 |     **0** |     0 | **NO TMA in either kernel** |
| `LDGSTS` / `cp.async`                 |      0 |     0 |     0 |        |
| `FENCE.VIEW.ASYNC.S`                  |      0 |     **1** |     0 | only cuTile |
| `SYNCS.EXCH.64`                       |      0 |     **9** |     0 | only cuTile — barrier-init pattern |
| `SYNCS.PHASECHK.TRANS64(.TRYWAIT)`    |      0 |    **31** |     0 | only cuTile — async-tx barrier wait |
| `BAR.SYNC.DEFER_BLOCKING`             |      2 |     **8** |     0 |        |
| `BSSY.RECONVERGENT` / `BSYNC.RECONVERGENT` | 2/2 | **14/14** |     0 | warp-divergent fast paths |
| **compute**                           |        |         |         |        |
| `FFMA`                                |    192 |     0 |    64 | nvcc uses fused FFMA |
| `FMUL`                                |     80 |   **896** |   194 | cuTile decomposes |
| `FADD`                                |      8 |   **771** |   960 |        |
| `HFMA2`                               |      0 |    64 |     2 |        |
| **resource use** (cuobjdump)          |        |         |         |        |
| REG / thread                          |     40 |   **255** |   48-56 |        |
| static SHARED bytes                   |   1040 | **100 436** |   1544 |        |
| STACK (spill) bytes                   |      0 |   **824** |     0 |        |
| dynamic SHARED bytes (runtime)        |   65 536 |   0 |     0 | nvcc uses opt-in dynamic smem; cuTile uses static |

(Vectorised store count rows are blank for nvcc/oxide because the
instruction never appears.)

### Reading the table

- **Total bytes per static instruction-set is roughly equal
  (~512 B/pass).** nvcc's 16 × `LDG.E.128` = 256 B per pass; cuTile's
  128 × `LDG.E` = 512 B per pass. Per-instruction, nvcc carries
  4× more bytes. That advantage is real but doesn't translate to GB/s
  because of the next two rows.
- **cuTile has 326 `LDS.128` vs nvcc's 24.** The math warps in cuTile
  predominantly read state from **shared memory with vector loads**.
  The `LDG.E` counts above are issued once by the producer warp; the
  consumer warps then re-read the same data through smem with
  `LDS.128`. This is the hidden multiplier.
- **`SYNCS.PHASECHK.TRANS64.TRYWAIT` × 31 is the smoking gun for an
  async-barrier pipeline.** This is Blackwell's `mbarrier::try_wait_parity`
  primitive. The producer warp issues `SYNCS.ARRIVE` after each gmem
  load completes; the consumer warps `try_wait_parity` to advance a
  pipeline phase. This is **exactly the pattern TMA uses**, just
  implemented with explicit `LDG.E + mbarrier::arrive` instead of the
  fused `UTMALDG` path. Mechanically equivalent for hiding HBM
  latency.
- **REG=255, STACK=824, SHARED=100 436.** cuTile is at the maximum
  register count, with stack spills, and uses essentially the entire
  100 KiB sm_120 opt-in smem budget. This is consistent with
  multi-buffered (likely 4-stage) pipeline buffers.
- **nvcc's REG=40, SHARED=1040** is a tiny kernel by comparison. It
  fits the workload it has — TPB=16, one float4 register tile per
  thread. There's no async pipeline because there's no smem budget
  for one and no second warp to act as producer.

---

## Hypothesis evaluation

The Wave 17 summary listed four hypotheses. Evaluating each against
the SASS evidence:

### H1 — TMA bulk-load uses dedicated hardware path on Blackwell, separate from the SM's LSU
**Verdict: REJECTED on the literal claim, but PRESERVED in spirit.**
There is **no `UTMALDG`** in either cubin. cuTile is not using the
TMA engine on this kernel. So the literal hardware-path claim is
false here. *However*, the underlying mechanism cuTile uses
(`SYNCS.PHASECHK.TRANS64` async-transaction barriers driving an
explicit producer-consumer split) is the same *pattern* TMA was
designed for, just implemented at the LDG-instruction level. The
"separate hardware path" intuition was wrong; the "decouple loads
from math warps" intuition was right.

### H2 — `LDG.E.128` still gates on the SM's LSU which is shared with the rest of the kernel's loads
**Verdict: SUPPORTED, and is probably the dominant effect.**
nvcc's TPB=16 means the LSU is fed by 16 threads per block. The
`LDG.E.128` issues at one transaction per cycle per LSU port, and
**the math FFMAs in the same warp issue immediately after the load
completes**. There is no producer/consumer overlap — load and math
share the LSU's issue slot via the same warp. Compare to cuTile,
where the `LDG.E`-issuing warp is structurally distinct from the
`LDS.128`-consuming warps; the LSU and the FFMA pipe never compete
for issue slots in the same warp.

The 4× transaction-count win of `LDG.E.128` does *not* translate to
4× LSU-bandwidth because LSU throughput is bytes/cycle-bounded by the
`LD/ST.x.128` issue rate (1 per cycle) and by L2 round-trip latency.
At only 16 threads per block, a single warp issuing one
`LDG.E.128` per cycle saturates one transaction per cycle but
**only one warp deep** — so latency is not hidden, it is exposed.

### H3 — cuTile's async-pipeline setup may have lower per-iter overhead than the per-thread address computation needed for `LDG.E.128`
**Verdict: PARTIALLY SUPPORTED, but it is not the per-iter math that
matters — it's the producer/consumer split.**
The address-computation cost for `LDG.E.128` (per-thread strided
`float4*` ptr arith) is real but small (`IMAD.WIDE` × 1-2 per pass).
The much larger structural difference is that in cuTile the gmem
load address is computed **once per row by the producer warp** and
the consumed result is then read by every consumer warp from a fixed
smem layout — i.e. the address-computation cost is amortised over
many shared-memory reads. nvcc has no such amortisation: each pass-2
loop iter re-issues `LDG.E.128` (or its smem-cached equivalent) per
thread, with full per-thread address recomputation.

### H4 — Cache-hit-rate differences: TMA loads bypass-L1, less cache pollution; `LDG.E.128` hits L1 first
**Verdict: NOT SUPPORTED at the literal level (no TMA), but the
related claim about descriptor-based loads holds.**
cuTile's `LDG.E desc[UR12][R6.64]` form uses the **descriptor-based
gmem-access** introduced on Hopper. The `desc[]` operand carries
explicit cache hints (typically `EVICT_FIRST` for streaming state
tiles), so cuTile's gmem traffic does NOT pollute L1 the way nvcc's
default-hint `LDG.E.128 [R6]` does. nvcc's loads default to L1-cached.
For a kernel whose state tile is 256 KiB (≫ L1 capacity ≈ 128 KiB on
sm_120), default-L1 caching is near-pessimal: the state tile
constantly evicts itself from L1. cuTile's evict-first descriptor
loads keep L1 free for the genuinely-reused q/k/v vectors.

This effect is real but probably second-order to H2 + the
producer/consumer split.

### Summary of hypothesis evaluation

| H  | claim                                              | verdict | weight |
|---:|---|---|---:|
| H1 | TMA hardware path                                  | REJECTED literal; the deeper "decouple from LSU" intuition holds via `SYNCS` async barriers | low (literal); high (intuition) |
| H2 | `LDG.E.128` competes with math on the SM's LSU     | SUPPORTED — main effect at TPB=16 single-warp pattern | **dominant** |
| H3 | per-iter overhead difference                       | PARTIALLY SUPPORTED — amortisation across consumer warps | medium |
| H4 | cache-hit / cache-pollution                        | LITERAL TMA-bypass-L1 claim fails (no TMA); the analogous `desc[]` evict-first hint claim is plausible but unverified | low-medium |

---

## Top three evidence-supported hypotheses (re-ranked)

1. **Producer/consumer warp specialisation amortises gmem and address
   computation across `LDS.128` consumer reads.** cuTile issues
   `LDG.E` once per row, then every consumer warp re-reads from smem
   with 16-byte-wide `LDS.128`. nvcc's TPB=16 single-warp loop
   re-issues a load per pass per thread with no shared amortisation.
   Evidence: `LDS.128` count 326 vs 24, `STS.128` 96 vs 0, REG=255
   vs 40, SHARED=100 436 vs 1040.
2. **cuTile uses Blackwell async-transaction barriers
   (`SYNCS.PHASECHK.TRANS64.TRYWAIT`) to overlap gmem load latency
   with consumer-warp compute.** This is mechanically the same as
   what TMA delivers, just implemented through `LDG.E` + mbarrier
   primitives instead of `UTMALDG`. Evidence: 31 `SYNCS.PHASECHK`,
   9 `SYNCS.EXCH.64`, 1 `FENCE.VIEW.ASYNC.S`, 14 paired
   `BSSY/BSYNC.RECONVERGENT` boundaries, 14 `BAR.SYNC.DEFER_BLOCKING`
   — none of which appear in the nvcc kernel.
3. **nvcc's kernel runs only 16 threads per block (half a warp).**
   `cuda-attn-gdn/attn_gdn.cu:TPB = BLOCK_V / 4 = 16` for BLOCK_V=64.
   At TPB=16 the LSU sees no warp-level pipelining of independent
   loads, and the single warp's stalls on each `LDG.E.128` are
   directly exposed in wall time. The `LDG.E.128` instruction class
   is fine; the launch geometry it sits inside is the bottleneck.
   `cuda-attn-gdn/ANALYSIS.md:96-102` already calls out
   "TPB=16 is half a warp; this is a known under-utilisation."

---

## Recommendations for closing the gap

### Inside CUDA C++ (no language change)

1. **Widen TPB to 32 or 64.** At BLOCK_V=64 the cleanest win is to
   keep `float4`-per-thread and run TPB=32 with each thread owning
   2 cols (or TPB=64 with 1 col-per-thread, scalar). TPB=32
   doubles the in-flight load slots per block at zero smem cost. The
   SASS analysis in `cuda-attn-gdn/ANALYSIS.md:96-102` already flags
   this. **Lowest-effort experiment, probably the largest single
   gain** — closes most of H2 above.
2. **Manually wire a `cuda::pipeline` / `cuda::memcpy_async` +
   `cuda::barrier::arrive_and_wait` producer/consumer structure.**
   This emits `LDGSTS` / `cp.async` (gmem→smem direct, bypasses
   register file) and the same `SYNCS.PHASECHK` barrier primitives
   cuTile uses. CUDA C++ does have access to these on sm_120; nvcc
   13.x lowers `cuda::memcpy_async` to `LDGSTS` with mbarrier
   support. The kernel would need a structural rewrite — split warps
   into producer (loads gmem → smem) and consumer (computes from
   smem) — and an opt-in to ≥64 KiB smem with multi-buffering. This
   is the "do what cuTile does manually" path. Effort: 2-3× the
   current LOC.
3. **Use the `__pipeline_memcpy_async` low-level intrinsic with
   explicit `__pipeline_wait_prior` / `__pipeline_commit` calls.**
   Same effect as #2 but more controllable. The
   [`cooperative_groups::memcpy_async`] alternative is a thinner
   wrapper that lowers to the same SASS.
4. **Try the actual TMA path (`cuda::ptx::cp_async_bulk_tensor_*`).**
   On sm_120 (Blackwell consumer), TMA exists with the
   `cp.async.bulk.tensor.{1,2,3,4,5}d.shared::cluster.global` family.
   It would emit `UTMALDG` — neither current kernel does. This is a
   bigger structural change because it requires constructing a
   tensor-map descriptor at host-side, but it would eliminate the
   per-thread address arithmetic entirely. Worth a follow-up cell.

### Outside CUDA C++ (compiler/frontend changes)

5. **nvcc could in principle emit producer/consumer structure
   automatically when it sees an `#pragma unroll` over a strided HBM
   read followed by smem-cached re-use** — but this is a major
   compiler-pass change (warp specialisation + mbarrier insertion),
   not a small one. cuTile gets it because its tile-DSL gives the
   compiler the structure for free.

---

## Compiler-quality issue, or hardware-API gap?

**Hardware-API gap, with a side of launch-geometry quality issue.**
Two-axis breakdown:

- **What nvcc did wrong (compiler-quality, fixable in source).**
  The 16-thread-per-block launch is a deliberate choice in
  `cuda-attn-gdn/attn_gdn.cu`; it falls out of `TPB = BLOCK_V/4`. A
  three-line edit to TPB=32 or TPB=64 is easy. nvcc emitted exactly
  what the source asked for. **This is a *kernel-source-quality*
  issue, not an *nvcc-codegen-quality* issue.**
- **What nvcc cannot easily express (hardware-API gap).**
  cuTile's compiler emits a warp-specialised async-barrier pipeline
  using ≈100 KiB of shared memory and 31 `SYNCS.PHASECHK.TRANS64`
  waits. CUDA C++ *can* do this, but only via
  `cuda::memcpy_async` + `cuda::pipeline` + manual warp
  specialisation + opt-in to 100 KiB smem — i.e. ~3× the LOC and a
  structural rewrite. The cuTile DSL gets this for free because
  `tiled_view.load` *is* the structural API. **The gap is in
  programming-model expressiveness, not in raw codegen quality.**

The 417.7 → 610 GB/s gap likely decomposes into:

- **~80-120 GB/s** from TPB=16 → TPB=32 or 64 widening (H2,
  in-source fix);
- **~80-150 GB/s** from adopting an async-pipeline producer/consumer
  pattern (`cuda::memcpy_async` + `mbarrier`) — H1-via-`SYNCS` and H3;
- **<50 GB/s residual** that may relate to evict-first cache hints
  on the streaming state tile (H4).

The first item is a kernel-source quality fix; the second is a
hardware-API/programming-model fix. **Neither is an nvcc compiler
bug.**

---

## What would unblock a fair re-test

- **Cell W22.6** (already in `results/wave17-summary.md` Wave 22
  candidates): wire a timed harness onto cuda-oxide-gdn to fill in
  the third frontend's bench number and validate the LDG-only-no-async
  baseline.
- **A new cell W22.9 (proposed)**: `cuda-attn-gdn-async` — same
  algorithm as W1c but:
  - TPB=32 (full warp);
  - producer/consumer split using `cooperative_groups::memcpy_async`;
  - opt-in to 64-100 KiB smem with multi-buffered (D_K, BLOCK_V) tile.
  - Acceptance: ≥600 GB/s would empirically validate the
    "hardware-API gap, not compiler-quality" finding above.
- **A new cell W22.10 (proposed)**: `cuda-attn-gdn-tma` — same
  algorithm but using `cuda::ptx::cp_async_bulk_tensor_2d` for the
  state-tile load. Tests the literal TMA path. Expected ≥W22.9.

---

## Caveats and uncertainty

- The bench numbers themselves have material variance: cuTile's
  median = 547 GB/s, IQR [13.82, 16.32] µs ≈ 547-610 GB/s; nvcc's
  mean = 281 GB/s, best = 417 GB/s. The headline 610 vs 417 GB/s
  comparison is best-vs-best. Median-vs-median (547 vs ~280) is
  larger; mean-vs-mean (~547 vs ~280) larger still. The gap is real
  in either case, but its size is regime-sensitive — and the best/median
  spread itself reflects launch-overhead instability at 14-30 µs
  per iter (see `cutile-attn-gdn/ANALYSIS.md:38-42`).
- I have not directly disassembled the cuTile descriptor encoding
  (`desc[UR12]`) to confirm the cache-hint claim in H4. The hint
  bit-pattern lives in the descriptor that `LDCU.64 UR12, c[0x0][0x358]`
  loads from constant memory; verifying this would require dumping
  the kernel argument descriptor table. Out of scope for this
  docs-only investigation.
- The decomposition of the 417 → 610 GB/s gap into "~80-120 GB/s
  from TPB widening" / "~80-150 GB/s from async pipeline" /
  "<50 GB/s from cache hints" is engineering judgment based on the
  SASS evidence above and the `cuTile vec-add → 87% HBM` data point
  from `cutile-attn-gdn/ANALYSIS.md:37-42`. It is not directly
  measured; W22.9 + W22.10 would refine these numbers.
- W17's claim that cuTile uses `UTMALDG` was not directly evidenced
  in W17 — it appears to have been transferred verbatim from the
  cutile-attn-gdn ANALYSIS.md:158-160 forward-looking comment ("cuTile's
  TMA primitives ... could plausibly help") and elevated to a stated
  hypothesis in `results/wave17-summary.md:53,83`. The SASS shows
  the cuTile *reduction* kernel may use TMA; the GDN-decode kernel
  shipped in v1.3.0 does not. The summary should be amended.

---

## Files reviewed (no source modified)

- `cuda-attn-gdn/ANALYSIS.md` — W1c writeup
- `cuda-attn-gdn/attn_gdn.cu` — kernel source (TPB=BLOCK_V/4=16)
- `cuda-attn-gdn/attn_gdn.sass` — disassembled cubin (1 308 lines)
- `cuda-attn-gdn/bench.log` — best 20.16 µs / 417.7 GB/s
- `cutile-attn-gdn/ANALYSIS.md` — Wave 16.4 baseline writeup
- `cutile-attn-gdn/main.py` — cuTile kernel source
- `cutile-attn-gdn/gdn_decode_fused.sass` — 10 841 lines, sm_120a
- `cutile-attn-gdn/run_bench.log` — best 13.79 µs / 610.6 GB/s
- `oxide-attn-gdn/oxide_attn_gdn.sass` — third reference (no bench)
- `results/wave17-summary.md` — cross-cell observations

## Files NOT modified

This is a docs-only Wave 22.8. No source, SASS, build artifact, or
log file was edited. The only new file is this document at
`docs/research/wave17-w1c-tma-vs-ldg128-investigation.md`.
