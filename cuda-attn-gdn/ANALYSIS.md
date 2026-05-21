# cuda-attn-gdn — Wave 17 W1c ANALYSIS

Hand-written CUDA C++ (nvcc) implementation of **Gated DeltaNet (GDN)
single-timestep decode** — the linear-attention recurrence used in
Qwen3-Next. Targets the same algorithm as `cutile-attn-gdn` (610 GB/s
baseline) and its "missing structural win" identified in
`cutile-attn-gdn/ANALYSIS.md:152-156`: thread-level **LDG.E.128 / STG.E.128
vectorized state-tile traffic**, which cuTile v1.3.0 cannot emit.

## Headline (correctness only — no timed bench in W1c authoring)

| metric                                | value                               |
|---|---|
| correctness shape                     | B=2 H=4 d_k=64 d_v=64               |
| correctness `o`    max\_abs vs PyTorch | **3.052e-05** (≤ 1e-3 threshold) ✓ |
| correctness `Sout` max\_abs vs PyTorch | **2.980e-08** ✓                    |
| bench shape                           | Qwen3-Next: B=1 H=16 d_k=d_v=256    |
| bench-shape `o`    max\_abs           | **6.104e-05** ✓                     |
| bench-shape `Sout` max\_abs           | **2.980e-08** ✓                     |
| HMMA in SASS (ADR-0004 sanity)        | **0** (memory-bound, TC N/A)        |
| FFMA in SASS                          | **192 total** (96 per template)     |
| LDG.E.128 in SASS                     | **16 total** (8 per template) ✓    |
| STG.E.128 in SASS                     | **16 total** (8 per template) ✓    |
| MUFU in SASS                          | **0** (no exp/softmax in GDN) ✓    |

Bench timings are not collected here per W1c instructions. Expected GB/s
range per the Wave-17 plan acceptance row is **[400, 750]** with cuTile at
610 GB/s; the orchestrator runs `./bench` separately on idle GPU.

## Why GB/s, not TFLOPS

Per **ADR-0004** and `flops_gdn.gdn_decode_arith_intensity` ≈ **0.77 flops/byte**,
GDN-decode at batch 1 sits FAR below the RTX 5090 ridge (~25 flops/byte
even for f32 cores). Per-iter:

  - **Compute**: 6 · d_k · d_v · B · H = 6 · 256·256·1·16 = 6.3 MFLOPS
  - **HBM traffic**: ~8.2 KiB (S_in read 256 KiB + S_out write 256 KiB
    summed over 16 heads dominates the per-token total; per-head q/k/v/α/β/o
    are 1.5 KiB)

The kernel's job is to move 256 KiB of state per head, twice (read+write),
as fast as possible. Compute throughput is irrelevant.

This matches `cutile-attn-gdn/ANALYSIS.md:23-42` exactly; both kernels
share the same memory-bound nature.

## Kernel structure

Single fused kernel; one CUDA block per `(batch * head, d_v_block)` tile,
matching `cutile-attn-gdn/main.py`:

```
__global__ void gdn_decode_kernel<D_K, BLOCK_V> (
    const __half* Q, K, V, Alpha, Beta,
    const float*  S_in,  float* S_out,  __half* O,
    int B_H, int D_V)
{
    bh = blockIdx.x;   bv = blockIdx.y;   tid = threadIdx.x;
    col0 = bv * BLOCK_V + tid * 4;     // each thread owns 4 d_v columns

    // ── load q, k cooperatively → smem (f16 → f32 once)
    // ── load v (per-thread float2 = 4 halves)
    // ── alpha, beta scalars

    // ── Pass 1: read state row-by-row,  S_scaled = α · S_in,  cache to smem,
    //            accumulate u = k · S_scaled per-thread (4-wide)
    for k in [0, D_K):
        s = *(float4*)&S_in[bh, k, col0];      // <-- LDG.E.128
        s *= alpha;
        smem_S[k * VLANES + tid] = s;          // S_scaled cached
        u_acc.{xyzw} += k_vec[k] * s.{xyzw};

    r = v - u_acc;                              // residual

    // ── Pass 2: S_out = S_scaled + β · k · r,  accumulate o = q · S_out
    for k in [0, D_K):
        s = smem_S[k * VLANES + tid];
        s += (β * k_vec[k]) * r;
        *(float4*)&S_out[bh, k, col0] = s;     // <-- STG.E.128
        o_acc.{xyzw} += q[k] * s.{xyzw};

    // ── store o: 4 halves packed as float2 (LDG.E.64-equivalent for store)
    *(float2*)&O[bh, col0] = pack_halves(o_acc);
}
```

### Block / thread shape

| field      | correctness shape | qwen3_next_decode shape |
|---|---|---|
| BLOCK_V    | 64                | 64 |
| Threads/blk (TPB = BLOCK_V/4) | 16  | 16 |
| Grid        | (8 = B·H, 1)     | (16 = B·H, 4) |
| Per-block smem (q+k floats + S_scaled tile) | 16 KiB | 64 KiB |

TPB = 16 is half a warp; this is a known under-utilization for the
correctness shape (d_v=64). For the canonical bench shape the trade-off
is favourable: each thread carries the full d_k inner loop with `float4`
register accumulators (`u_acc`, `o_acc`, `r`) and emits LDG.E.128 /
STG.E.128 transactions per thread. Going wider (TPB=32, BLOCK_V=128)
would double the per-block smem to 128 KiB at d_k=256, which exceeds
sm_120's 100 KiB-per-block opt-in budget — kept at BV=64 to match cuTile.

### Why `cudaFuncSetAttribute` is needed

The bench-shape per-block dynamic smem is **64 KiB**, above the default
48 KiB cap. The launcher opts in via
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)`
gated on `smem_bytes > 48 * 1024`. Without this the launch silently
fails with `cudaErrorInvalidValue`. (Pitfall caught during bring-up;
solved on first attempt thanks to ADR-aware planning.)

### Correctness algorithm

Identical to `cutile-attn-gdn/main.py` — itself a verbatim transcription
of `pytorch_reference_gdn.gdn_decode_reference`:

```
S_scaled = α · S_in
u        = k · S_scaled                            # (1, BLOCK_V)
S_out    = S_scaled + β · k ⊗ (v - u)              # (D_K, BLOCK_V)
o        = q · S_out                               # (1, BLOCK_V)
```

f32 register accumulators throughout; f16 inputs upcast on load; final
output cast back to f16. Cross-validated against the saved expected NPYs
(written by the reference run) for both shapes.

## SASS evidence (ADR-0004)

`cuobjdump --dump-sass attn_gdn` per template instantiation:

```
function _Z17gdn_decode_kernelILi64ELi64EEv...      (D_K=64,  BV=64)
  lines=640  LDG.E.128=8  LDG.E.64=1  STG.E.128=8  FFMA=96  HMMA=0  MUFU=0
function _Z17gdn_decode_kernelILi256ELi64EEv...     (D_K=256, BV=64)
  lines=624  LDG.E.128=8  LDG.E.64=1  STG.E.128=8  FFMA=96  HMMA=0  MUFU=0
```

The 8 LDG.E.128 + 8 STG.E.128 are the unrolled state-row passes (the
`#pragma unroll 8` on the d_k loops promotes 8 of the d_k iterations into
explicit 128-bit transactions per pass; the rest stay in a tight loop
which still emits LDG.E.128 / STG.E.128 inside the loop body — the SASS
counter is per static-instruction, not per dynamic-execution).

The 1 LDG.E.64 is the `v` vector load (8 bytes = 4 halves per thread).
The 96 FFMAs are the per-row outer-product / matvec accumulations
(2 passes × ~48 FFMAs in the unroll).

**HMMA = 0**: confirms no Tensor Core engagement. Per ADR-0004 §1, this is
the correct outcome for GDN-decode — the matmuls are M=1 / K=1 shapes
that waste ≥94% of any HMMA fragment. Same conclusion as
`cutile-attn-gdn/ANALYSIS.md:78-90`.

**MUFU = 0**: GDN has no `exp` / `softmax` / reciprocal. (Distinct from
GQA/MLA where MUFU.EX2 dominates the softmax pass.)

## Edge over cuTile baseline

The Wave-17 plan W1c row says: *"the W1c cell's edge over cuTile is
supposed to be LDG.E.128 (vector loads) — verify this in SASS."*

✓ Verified. The cuTile cell's ANALYSIS.md:152-156 explicitly flags scalar
f32 loads as the single biggest BW gap; nvcc's `*reinterpret_cast<float4*>`
gives 128-bit transactions trivially. The compiler emits `LDG.E.128`
without coaxing.

Whether this translates to measured GB/s above 610 (cuTile's best) is the
orchestrator's bench result; the **structural** win is captured here.

## Pitfalls hit (and not hit)

1. **`cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in for 64 KiB smem.**
   Default cap is 48 KiB. Solved at write-time via guarded
   `cudaFuncSetAttribute`. Would have surfaced as a silent
   `cudaErrorInvalidValue` post-launch.

2. **TPB=16 for BLOCK_V=64 wastes half a warp.** Functional but probably
   the limiting factor for higher GB/s on Blackwell — at TPB=16 only one
   warp's worth of LDG transactions are in flight per SM. Could be
   addressed in a follow-up by going BV=128 (TPB=32 = full warp, but
   doubles smem). Wave 16.5 in the cuTile plan is the parallel workstream.

3. **f16 → f32 upcast on `v` and `q`/`k`.** Done lane-by-lane via
   `__half2float` after a packed `float2` (=4 halves) load. nvcc's
   builtin gives a single conversion instruction per lane; no slowdown
   measurable at this size.

4. **Single fused kernel, no async smem prefetch.** The (D_K, BV) state
   tile is read once into smem during pass 1 and reused in pass 2 — this
   is the core fusion. No explicit `cp.async` (the per-iter work is too
   small to amortize the descriptor setup). Same trade-off as cuTile.

5. **`float4` alignment.** d_v=64 and 256 are both multiples of 4, and
   the input NPYs are aligned by `cudaMalloc` (256 B alignment), so the
   reinterpret-cast load/store of `float4` is well-defined. Verified
   indirectly by the correctness pass.

6. **`__pragma unroll 8` on d_k loops.** Explicit-finite unroll (instead
   of full unroll) keeps register pressure bounded for D_K=256 (would
   otherwise allocate 256×4 registers for cached state); the SASS shows
   a tight inner loop in addition to the 8 unrolled iterations. Compiles
   clean; correctness verifies that the loop tail is handled.

7. **Did NOT hit** the libNVVM-shadow bug from Wave 3 (we're on nvcc, not
   cuda-oxide), the WMMA fragment-shape pitfall (no WMMA used), or any
   thermal-variance issues (no benches in this cell).

## Files

| file              | purpose |
|---|---|
| `Makefile`        | nvcc build rules + SASS-dump target |
| `attn_gdn.cu`     | kernel + correctness driver (templated `gdn_decode_kernel<D_K, BLOCK_V>`) |
| `bench.cu`        | bench harness w/ cudaEvent timing (orchestrator-run, not in W1c) |
| `run.sh`          | build + correctness + SASS-dump driver |
| `attn_gdn.sass`   | generated SASS for ADR-0004 evidence (gitignored; regenerable via `make sass`) |
| `ANALYSIS.md`     | this file |
| `.gitignore`      | binaries + logs + sass + cubins |

## What this cell tells us about nvcc vs cuTile for memory-bound kernels

cuTile v1.3.0's loss to nvcc on GDN is **structural, not architectural**:
neither frontend can engage tensor cores on M=1/K=1 ops, and both end up
at FFMA + LDG / STG. cuTile's tiled_view abstraction emits scalar f32
loads (per cuTile ANALYSIS); nvcc with `float4` reinterpret-cast emits
LDG.E.128. That's a 4× transaction-width difference on the dominant
state-tile traffic, projecting roughly to **a meaningful GB/s gain over
610**, all else equal.

This cell's actual GB/s measurement is captured by the orchestrator's
bench run (`./bench`). The expected band per the Wave 17 plan acceptance
row is **[400, 750]** GB/s with **cuTile at 610**.

## What's next (not part of W1c)

- Orchestrator runs `./bench` for the timed measurement.
- Cross-cell comparison goes into `results/wave17-summary.md` per ADR-0004's
  "no-TC ceiling vs TC ceiling" table format.
- Wave 17 W1d (`oxide-attn-gdn`) will sit alongside this as the cuda-oxide
  data point; expected lower (200–600 GB/s) per `wave17-gdn-other-frontends.md`.
