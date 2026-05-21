# mojo-attn-gqa — Wave C1.5 (6th frontend port of GQA attention)

**Goal:** close the Mojo column for GQA in the cross-frontend Rosetta Stone matrix.
Five frontends already exist (cuda-attn-gqa, cutile-attn-gqa, oxide-attn-gqa,
wgpu-attn-gqa, cublas-attn-gqa). This cell is the 6th and final.

## Approach

Port the W22.5b mojo-attn-bf16 cell (which already implemented MLA bit-exact
at 26.36 TF using a 3-kernel decomposition with bf16 hand-MMA + Wave 21 tile
patterns) to GQA. Differences vs MLA:

1. **Q has Nq=32 query heads, K/V have Nkv=8 KV heads.** Inside the kernel,
   `bh_kv = bh // GROUPS` where `GROUPS = Nq / Nkv = 4`. We never expand
   K/V in memory — the indexing math does the broadcast.
2. **Different shape:** B=1, Nq=32, Nkv=8, S=2048, qk=d_h=d_v=128 (matches
   cuda-attn-gqa's `SHAPE_BENCH = "llama3_8b"` exactly).
3. **FLOPS formula:** `4 * B * Nq * S^2 * D` (matches cuda-attn-gqa's
   `flops/iter` convention; treats only Q-side count, K/V broadcast saves
   memory traffic but not compute).

## Tile shape (kept identical to W22.5b)

| param | value | rationale |
|---|---|---|
| BM | 64 | S=2048 / 64 = 32 ✓ |
| BN | 64 | S=2048 / 64 = 32 ✓ (qkt); DV=128 / 64 = 2 ✓ (pv) |
| BK | 32 | QK=128 / 32 = 4 ✓ (qkt K-iters); S=2048 / 32 = 64 ✓ (pv K-iters) |
| WM/WN | 32 | 4 warps/block |
| MMA | 16×8×16 | bf16 → f32 (HMMA.16816.F32.BF16) |
| BLOCK_THREADS | 128 | 4 warps × 32 lanes |

## Code changes from W22.5b

**Shape constants** (split into Nq/Nkv/GROUPS):
```mojo
comptime NQ = 32          # was: NH = 128
comptime NKV = 8
comptime GROUPS = NQ // NKV   # = 4
comptime QK = 128         # was: 192
```

**Layouts**: K/V use `BHKV * S` rows, Q/Sm/P/O use `BHQ * S` rows.

**qkt_kernel & pv_kernel** take an extra comptime `GROUPS` param. Both compute:
```mojo
var bh    = Int(block_idx.z)        # Q-head
var bh_kv = bh // GROUPS            # KV-head (GQA broadcast)
```
And use `head_bm_off_kv = bh_kv * (S // BM)` instead of `head_bm_off` for the
K-tile (qkt) or V-tile (pv) DRAM gather. Q/Sm/P/O still use `head_bm_off = bh * ...`.

**CPU reference** mirrors the same indexing: K and V loaded from `bh_kv`, Q
and O accessed at `bh`.

**FLOPS**: changed from `2 * B * NH * S^2 * (qk + d_v)` to `4 * B * Nq * S^2 * D`
(the cuda-attn-gqa convention).

## Results

| variant | TFLOPS | gpu_ms | notes |
|---|---|---|---|
| **mojo-attn-gqa best** | **28.97 TF** | 2.372 ms | min over 10 iters |
| **mojo-attn-gqa median** | **25.65 TF** | 2.679 ms | |
| mojo-attn-gqa worst | 22.79 TF | 3.015 ms | |

**Correctness:** max_abs_err = 0.0 vs CPU SDPA reference (1024 samples,
GQA broadcast-aware). Bit-exact match. Tolerance was atol=1e-2 + rtol=1e-3
(W21 spec, matches W22.5b). Achieved bit-exact, well under threshold.

**SASS evidence:** 32× `HMMA.16816.F32.BF16` instructions (16 per kernel ×
2 matmul kernels) confirm tensor-core engagement. SASS dumped to `attn_gqa.sass`.

## Cross-frontend GQA matrix (llama3_8b shape: B=1 Nq=32 Nkv=8 S=2048 D=128)

| frontend | TFLOPS | kernel decomposition |
|---|---|---|
| cublas-attn-gqa | 218.00 TF | hgemm reference (cuBLAS) |
| **cutile-attn-gqa** | **165.00 TF** | fused FlashAttention-class |
| oxide-attn-gqa | ~24.00 TF | 3-kernel hand-WMMA |
| cuda-attn-gqa | 23.40 TF (median) | 3-kernel WMMA (nvcc) |
| wgpu-attn-gqa | (CPU-only) | — |
| **mojo-attn-gqa** | **25.65 TF (median)** | **3-kernel hand-MMA (this cell)** |

### Ratios

- **vs cuda-attn-gqa** (apples-to-apples 3-kernel-vs-3-kernel):
  25.65 / 23.40 = **1.10× — Mojo is 10% faster than nvcc** at the same
  algorithm class. Mirrors the W22.5b finding for MLA (26.36 vs 24.17 TF,
  same ratio). The gain comes from BK=32 BF16 staging via cp.async/LDGSTS
  in the qkt kernel + Wave 21's hand-rolled m16n8 epilogue, vs cuda-attn-gqa's
  WMMA m16n16k16 fragment API.
- **vs cutile-attn-gqa** (fused-vs-3-kernel, structural gap):
  25.65 / 165.00 = **0.156×** — Mojo is at 15.6% of cuTile's fused
  FlashAttention-class throughput. Same structural gap as Wave 17 W1a/W1b
  on MLA: the 3-kernel HBM round-trip is the bottleneck, not compiler
  quality. Closing this gap requires fusion (online softmax, no S/P
  materialization).

## Pitfalls

1. **BK=32 divisibility**: GQA's qk=128 divides BK=32 cleanly (128/32 = 4
   K-iters). MLA's qk=192 also worked at BK=32 (W22.5b verified) but was
   right at the edge. Future GQA variants with qk=80 or qk=96 (Mistral, Phi)
   would need BK=16 or shape padding.
2. **Layout halving**: K/V layouts have `BHKV*S` rows = `BHQ*S/GROUPS`, NOT
   `BHQ*S`. If you forget and reuse layout_q for K, the K_lt indexing
   silently overflows the bf16 buffer (allocates 4× more than needed and
   reads out-of-bounds Q-shaped strides). Always recompute layout sizes
   per-tensor.
3. **Bit-exact correctness was a positive surprise** (max_abs_err = 0.0).
   The CPU reference casts probs through bf16 between softmax and PV (matches
   the kernel's `P` intermediate dtype) — without that cast, you'd see ~1e-3
   error from the fp32-vs-bf16 P difference, not zero. The W22.5b pattern
   pre-handled this so the GQA port inherited it.
4. **Mojo's `_dump_sass=True`** spills the full SASS into stdout, not a file.
   For a clean `.sass` file, post-process the run.log:
   ```bash
   awk '/^\t.target.*sm_120a/,/^\[mojo-attn-gqa\] shape:/' run.log | \
     head -n -1 > attn_gqa.sass
   ```
5. **Memory budget at this shape is ~824 MiB device** (vs ~3.5 GiB at the
   MLA shape because GQA has Nq=32 heads vs MLA's NH=128 heads). The S/P
   intermediate tensors dominate (768 MiB combined).

## Reproduce

```bash
cd /home/codeseys/cuda-exploration/mojo-attn-gqa
bash run.sh 2>&1 | tee run.log
```

Files:
- `attn_gqa.mojo` — the kernels + driver
- `run.sh` — pixi-managed mojo invocation
- `run.log` — captured stdout (includes SASS dump, timing, correctness)
- `results.csv` — best/median/worst TF
- `attn_gqa.sass` — extracted SASS (32× HMMA.16816.F32.BF16)
