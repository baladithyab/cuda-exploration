# Wave 16.3 — cuTile fused MLA (DeepSeek-V3)

Single `@ct.kernel` implementing FlashAttention-2 forward over MLA-shaped
inputs, on the cuTile 1.3.0 Python frontend. This is the pedagogical MVP:
no weight absorption, no latent cache. The kernel consumes already-up-
projected Q, K, V — so the attention core is "MHA with qk_head_dim = 192,
v_head_dim = 128, n_h = 128" for the DeepSeek-V3 config.

## Headline numbers

| Metric | Value |
|---|---|
| Best TFLOPS (bench) | **112.4 TFLOPS** |
| Median TFLOPS (bench) | 110.1 TFLOPS |
| Best gpu_ms | 3.056 ms |
| Grid | (128, 32) = (batch·n_h, seq/BLOCK_M) |
| Tile shapes | BLOCK_M=BLOCK_N=64, QK_PAD=256, D_V=128 |
| HMMA count in SASS | 384 (tensor cores engaged) |
| MUFU.EX2 count | 68 (softmax `exp` in-register) |
| Ratio vs cutile-attn-gqa (165 TF) | **68.1%** |
| Ratio vs cuBLAS hgemm peak (218 TF) | **51.6%** |

Correctness (f16, atol=5e-3, rtol=5e-3): **OK**, max abs err 1.5e-4.

## Kernel structure

Same skeleton as `cutile-attn-gqa/main.py`:

- Arrays flattened to 2D over `(batch, head, seq)` so `tiled_view` has a
  static 2D shape. For MLA there's no GQA-style KV-head grouping — every Q
  head has its own K/V — so `bid0` maps directly to a flattened head and
  both Q and K/V use the same `bid0`-indexed row base.
- Grid: `(batch · n_h, seq / BLOCK_M)`.
- Q tile `(BLOCK_M, QK_PAD=256)` loaded once, K/V tiles looped.
- Online softmax state in registers: `m_i (BLOCK_M, 1)`, `l_i (BLOCK_M, 1)`,
  `o_acc (BLOCK_M, D_V)` — all f32 accumulators.
- Two `ct.mma` calls per K/V block: QK^T at f16 → f32 (BLOCK_M×BLOCK_N), and
  PV at f16 → f32 (BLOCK_M×D_V).
- Final `o_final = o_acc / l_i` then cast to f16 for `store`.
- `ct.transpose(k_tile)` to feed QK^T (K is row-major per-head).

The **only** structural difference vs the GQA cell is:

1. No `kv_tile_row_base = h_kv_flat * SEQ_TILES_N` — for MLA `bid0` indexes
   a single physical head, so `kv_tile_row_base = bid0 * SEQ_TILES_N`.
2. Q and K are padded to `QK_PAD = 256` (next power of two above 192) before
   upload; the kernel sees a 256-wide inner dim. V and O are 128-wide
   natively.
3. Softmax scale uses the **true** `qk_head_dim = 192`, not `QK_PAD = 256` —
   otherwise the kernel output wouldn't match the PyTorch reference. Padded
   columns are zero so they contribute zero to QK^T regardless of scale.

## The 192→256 padding cost

MLA's `qk_head_dim = d_h + d_rope = 128 + 64 = 192` is not a power of two.
cuTile's `ct.mma` prefers power-of-two inner dims (confirmed by Wave 13.1).
For the MVP we pad to 256 in host-side Q/K arrays, so the trailing 64
columns are zero.

**Direct wasted compute.** QK^T does `(256/192 − 1) = 33%` extra flops that
produce zero contributions. That's a hard 25% ceiling hit on observed
TFLOPS relative to what a 192-native kernel would see: out of every 256
multiply-adds in QK^T, 192 are real work and 64 are wasted on zero-zero
products. HBM traffic for Q and K also inflates by 33%.

**Indirect register/shmem cost.** A 256-wide tile uses more staging
shared-memory per Q/K load; on a register-tight kernel this can push tile
spill thresholds. Empirically at BLOCK_M=BLOCK_N=64 the kernel compiled
and ran cleanly; we didn't try 128 given Wave 15.1's finding that 128 falls
off the register cliff even at d=128 — with d=256 the cliff is almost
certainly steeper.

**What the 192-native alternative looks like.** FlashMLA splits QK^T into
two sub-matmuls: the d_h=128 "nope" part and the d_rope=64 "rope" part,
accumulated separately into `s_acc`. That's a Wave 16.5+ refinement for
this cell. It keeps every tensor-core lane busy with real work, at the
cost of ~2× the number of `ct.mma` calls inside the inner loop.

## Ratios, in context

- **vs cutile-attn-gqa (165 TF, Llama-3-8B).** 68% is remarkably high
  considering:
    - GQA has `qk_head_dim = d_head = 128`, already power-of-two.
    - GQA has n_q=32 Q heads, MLA has n_h=128. The MLA grid (128, 32) is
      4× larger; SM occupancy is higher, which partially compensates for
      the padding waste.
    - MLA's D_V = 128 is the same as GQA's — the PV matmul is identical
      in cost per head, and there are 4× more of them.
  So the 32% shortfall is mostly attributable to the 25% flop waste from
  256-padding plus some register pressure from carrying a 256-wide Q tile.

- **vs cuBLAS hgemm peak (218 TF).** 52% is in-line with the "fused attn
  vs raw GEMM" ratio Wave 15.1 saw for GQA (165/218 = 76%). The ~24%
  additional shortfall is explained by the padding waste plus the usual
  fused-attention overhead (online softmax, reciprocal divide, exp calls).

- **vs what weight absorption would change.** Absorbing `W_UK` into `W_UQ`
  and `W_UV` into `W_O` eliminates both up-projection GEMMs from the
  critical path during decode — attention runs on `(n_h, d_c=512)` query
  tiles against `(T, d_c=512)` cached latent tiles. That's still a 512-wide
  matmul (already power-of-two — free!) but only 1/50 the KV HBM traffic
  of the pedagogical MVP. For prefill (the shape we bench here) weight
  absorption doesn't change the compute count — it still does the full
  `seq² · (qk_head_dim + d_v)` work — but it can change the arithmetic
  intensity and register pressure profile. Decode is the dramatic win.

## Pitfalls hit

1. **Array `bid0` indexing.** Copy-pasting from the GQA kernel initially
   gave `kv_tile_row_base = (bid0 // groups) * SEQ_TILES_N`, but MLA has
   no KV-head sharing. Had to strip the groups divide. Caught before first
   run by re-reading MECHANISMS.md §1.
2. **Softmax scale.** First draft used `1/√QK_PAD = 1/√256`. This is
   algebraically equivalent to `1/√192` only when the padded zeros carry
   zero scale info, which they do — but the PyTorch reference divides by
   `√qk_head_dim = √192`. Mismatch shows up immediately at smoke
   (max_abs ~0.2 vs tol 0.005). Fixed to `1/√qk_head_dim_true = 1/√192`.
3. **Zero-padded columns must actually be zero.** `np.zeros` init before
   the slice copy matters; `np.empty` would leak uninitialized data into
   padded positions and break correctness. Forced `np.zeros` in
   `prepare_device`.
4. **Block-size picker needs `shape.seq` divisibility, not shape.qk.**
   `pick_blocks` predicates on `shape.seq` dividing `bm` and `bn` — the
   QK_PAD dimension isn't tiled, only the M and N (seq) dimensions are.
5. **Input array flattening order.** For MLA, Q has shape `(B, n_h, S, qk)`
   and flattens to `(B·n_h·S, qk_pad)`. Getting the order of (n_h, S)
   correct matters: if we did `(B, S, n_h, qk)` the per-head data wouldn't
   be contiguous and `bid0 * SEQ_TILES_M` would index wrong rows. The
   PyTorch reference already saves in `(B, n_h, S, qk)` order so the
   `.reshape` is a no-op — good.
6. **JIT compile cost**: first launch takes ~600ms. Warmup iters (2) hide
   this from the timed block. The (variable) iter-0 vs iter-1 in bench
   runs shows cache warmup too.

## Files

- `/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/reference/shapes_mla.py` — MLAShape dataclass + 2 shape sets
- `/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/reference/flops_mla.py` — MLA attention-core FLOPS model
- `/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/reference/pytorch_reference_mla.py` — SDPA + naive oracles (cross-check OK at 1e-7 at bench shape)
- `/home/codeseys/cuda-exploration/cutile-attn-mla/main.py` — kernel + smoke/bench/export driver
- `/home/codeseys/cuda-exploration/cutile-attn-mla/run.sh` — orchestrator
- `/home/codeseys/cuda-exploration/cutile-attn-mla/results.csv` — 10 iters at bench
- `/home/codeseys/cuda-exploration/cutile-attn-mla/mla_fwd_fused.sass` — SASS disassembly (384 HMMA, 68 MUFU.EX2)

## Follow-up work (Wave 16.5+)

1. **Split-and-combine QK^T.** Separate the 128-wide nope sub-matmul from
   the 64-wide rope sub-matmul, accumulate into the same `s_acc`. Should
   recover the 25% padding loss.
2. **Weight absorption.** Write the "absorbed" MLA path where the kernel
   takes `(Q, c_KV, W_UK_folded, W_UV_folded)` and up-projects on the fly.
   Decode shape (seq=1 query token) is where this shines.
3. **Bench vs FlashMLA CUDA reference.** Once the absorbed path exists,
   compare directly to the DeepSeek FlashMLA Hopper-tuned kernel on a
   Blackwell-portable build.
