# oxide-attn-mla — Wave 17 W1b analysis

**Status:** COMPLETE — correctness passes, SASS is no-TC.
**Frontend:** cuda-oxide v0.1.0 (Rust → libNVVM → PTX → SASS)
**Mechanism:** MLA (Multi-Head Latent Attention, DeepSeek-V3 shape)
**Algorithm:** 3-kernel pedagogical attention (QKt → softmax → PV), f32 throughout
**Hardware:** RTX 5090 / sm_120 / WSL2 / CUDA 13.2

This cell is the no-TC ceiling data point for MLA: cuda-oxide v0.1.0 has no
usable tensor-core API on consumer Blackwell sm_120 (Wave 14.4 finding —
zero `mma.sync` in source, `wgmma` is a placeholder, `tcgen05` is sm_100a-only).
SASS confirms: HMMA = 0 across all three kernels, only scalar FFMA + MUFU.

## Acceptance gate (W17 plan W1b row)

| Criterion | Required | Measured | Status |
|---|---|---|---|
| max_abs_err vs PyTorch SDPA-MLA (correctness shape) | ≤ 1e-2 | **1.19e-7** | ✅ |
| max_abs_err vs PyTorch SDPA-MLA (bench shape) | ≤ 1e-2 | **2.83e-7** | ✅ |
| HMMA count in SASS | 0 | **0** | ✅ ADR-0004 |
| FFMA count in SASS | > 0 | **157** | ✅ ADR-0004 |
| MUFU count in SASS | > 0 (softmax exp) | **14** | ✅ ADR-0004 |
| TFLOPS in [10, 30] (warm, single GPU session) | 10–30 | **24.7 best / 24.4 median** | ✅ (orchestrator does final idle bench) |

`combined_fail` count (numpy-style `|a-b| <= atol + rtol*|b|`) is **0/32768**
on correctness and **0/33554432** on bench shape, so every output element
satisfies the combined-tolerance gate, not just the max.

## Padding policy (per ADR-0005)

**192-native, NO pad.** Per `docs/research/wave17-oxide-mla-design.md` §1:
the only reason cuTile pads `qk = 192 → 256` is that `ct.mma` on sm_120
wants power-of-two inner dims for HMMA lane packing. Oxide emits scalar
FFMA (verified — see SASS counts above), which has no alignment
requirement on the K-loop trip count. So 192 runs natively in 12 K-tiles
of width 16; correctness shape (`qk = 96`) runs in 6 K-tiles. Both are
divisible by `BK = 16`.

Headline FLOPS uses `useful_flops = 2·B·S²·(qk + d_v)·n_h` with `qk = 192`,
matching the cross-frontend headline rule. There is no padded-FLOPS
diagnostic column for this cell — `qk_pad == qk` so the two are identical.

DRAM-vs-register padding sanity (per ADR-0005 §3):
- LDG bytes per QKt-kernel inner load: `B·S·n_h·qk·4` for Q (and same for K),
  matching the **unpadded reference** exactly (no zero-fill bytes at the DRAM
  boundary). Padding here, even if it existed, would be register-only.
- LDG/STG ratios per SASS: `mla_qkt_kernel` LDG=8 STG=16, `mla_pv_kernel`
  LDG=8 STG=16. The LDG count is the cooperative load count for the per-tile
  Q/K (or P/V) shared-memory fill; STG count is the per-thread 4×4 microtile
  store. Both are inner-dim-agnostic.

## Headline numbers (warm, single session — orchestrator owns idle-GPU bench)

DeepSeek-V3 bench shape (`B=1, S=2048, n_h=128, qk=192, d_v=128`):

| Stage | avg ms | % of total |
|---|---:|---:|
| QKt (mla_qkt_kernel) | 7.5 | 52.8% |
| softmax (softmax_kernel) | 3.1 | 21.8% |
| PV (mla_pv_kernel) | 3.6 | 25.4% |
| **total** | **14.3** | 100% |

Best total: **13.9 ms** (24.7 TFLOPS useful). Median: 14.1 ms (24.4 TFLOPS).

Comparison to design-doc estimate: predicted 20–24 TF (`docs/research/wave17-
oxide-mla-design.md` §2). Measured 24.4–24.7 TF — at the upper end. The QKt
estimate (~7.2 ms) matched almost exactly; softmax + PV were ~30% better
than the linear-scaling worst case the doc warned about.

## Compute geometry (unchanged from oxide-attn-gqa)

- Block: `(16, 16, 1)` = 256 threads
- Output tile per block: 64 × 64 (each thread computes 4×4 microtile = 16 scalar accumulators)
- K-tile size: BK=16
- Shared memory per matmul kernel: 2 × 1024 f32 = 8 KiB (Q/P tile + K/V tile)
- QKt grid: `(S/64, S/64, B·n_h)` = `(32, 32, 128) = 131,072 blocks` at bench
- PV grid: `(d_v/64, S/64, B·n_h)` = `(2, 32, 128) = 8,192 blocks` at bench
- Softmax grid: `(B·n_h·S, 1, 1)` = `(262,144, 1, 1)` at bench, block 128 = 4 warps

Memory footprint at bench shape: scores + probs = 2 × 2.0 GiB = **4.0 GiB**
device memory just for the attention matrix. Plus Q (200 MiB), K (200 MiB),
V (134 MiB), out (134 MiB). Total ~4.7 GiB out of 32 GiB HBM. Fits.

## Deltas from oxide-attn-gqa (the template)

Per the design doc §4. Verbatim copies: softmax_kernel, NPY IO, event-timing
harness, cooperative-load math in matmul kernels.

| Change | Lines |
|---|---|
| Drop `groups` and `h_kv = h_q / groups` (MLA: n_kv = n_h, no GQA broadcasting) | -3, +0 |
| Generalize `d_head` → asymmetric (`qk` for QKt, `d_v` for PV) | +6 (sig + locals) |
| Scale = `1/sqrt(qk)` (was `1/sqrt(d_head)`) | 1 line |
| FLOPS = `2·B·n_h·S²·(qk + d_v)` (was `4·B·n_q·S²·d`) | helper rewrite |
| Buffer sizes: K is Q-shape, V/out are V-shape | host fn changes |
| Input file paths: `mla_*` not `gqa_*` | 4 strings |
| `Shape` struct: `qk` + `d_v` fields, drop `n_kv` (always == n_h) | struct change |
| Combined-tolerance correctness check (numpy-style) | per-skill recommendation |

LOC: **869** (vs design-doc estimate 890). ~+7 lines from the template
(862 → 869). Smaller than projected — most of the kernel-body deltas were
zero-line (renames only).

## Pitfalls hit / not-hit

- **`fmuladdf32` survival:** all 32 occurrences (16 in qkt, 16 in pv) preserved
  through the GQA-→-MLA copy-paste. SASS confirms 64 FFMA per matmul kernel
  (16 microtile FMAs × 4 unrolled K-tile iters), no scalar `*+` anywhere.
- **K-loop trip count = 12 (qk=192)** for bench, **6 (qk=96)** for correctness.
  Both divisible by `BK=16`. No correctness or codegen risk per the design
  doc — verified empirically (max_abs_err = 2.83e-7 at bench, 1.19e-7 at
  correctness).
- **Scores buffer is 2.0 GB.** Plus probs = 4.0 GB. Fits on RTX 5090 (32 GB)
  with headroom; no allocation issues. The from_host upload of K/V (200 MiB
  each) takes negligible time.
- **Multi-worker workspace coordination:** Wave 17 ran 5 parallel cell-
  authoring subagents in the same `jj` workspace. Discipline followed —
  this cell only touches `oxide-attn-mla/` paths; any cross-cell `jj st`
  noise comes from sibling workers, not from this cell.
- **Correctness check semantics:** the GQA template's `(tight OR rel) AND
  loose` reporting was misleading when `max_rel` spikes on near-zero
  outputs (softmax tails). Replaced with numpy-style `|a-b| <= atol +
  rtol*|b|` per-element count, plus standalone `max_abs/max_rel` for
  diagnostic visibility. Tracked the rust-gpu-compute skill's atol-OR-rtol
  pitfall.

## Files in this cell

- `Cargo.toml` — git deps on NVlabs/cuda-oxide.git (cuda-core, cuda-device, cuda-host)
- `Cargo.lock` — reproducibility anchor (committed per oxide convention)
- `rust-toolchain.toml` — pins nightly-2026-04-03 + rust-src + rustc-dev (copied from GQA)
- `src/main.rs` — three #[kernel] fns + harness (869 LOC)
- `run.sh` — build/run/sass helper; auto-exports `CUDA_HOME` + `LIBNVVM_PATH` + LLVM 21
- `build.log` — output of `cargo oxide build --arch sm_120`
- `run.log` — output of `cargo oxide run` (warm correctness + warm bench, NOT idle-GPU bench)
- `oxide_attn_mla.cubin` — sm_120 cubin (regenerable, gitignored)
- `oxide_attn_mla.ll` — LLVM IR (regenerable, gitignored)
- `oxide_attn_mla.ltoir` — LTO IR container (regenerable, gitignored)
- `oxide_attn_mla.sass` — SASS dump from `/usr/local/cuda/bin/cuobjdump --dump-sass` (gitignored, regenerable from `run.sh sass`)
- `out_correctness_mla.npy` — small-shape kernel output (gitignored)
- `results.csv` — per-iter timing CSV
- `.gitignore` — same pattern as GQA: target/, *.cubin, *.ll, *.ltoir, *.ptx, *.sass, *.npy

Tracked via `jj`: `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `src/main.rs`,
`run.sh`, `build.log`, `ANALYSIS.md`, `.gitignore`.

## Reproducing

```bash
cd /home/codeseys/cuda-exploration/oxide-attn-mla
bash run.sh build   # cargo oxide build --arch sm_120 → build.log
bash run.sh run     # cargo oxide run → run.log + results.csv + cubin/ll/ltoir
bash run.sh sass    # /usr/local/cuda/bin/cuobjdump --dump-sass → oxide_attn_mla.sass
```

Required env (handled by `run.sh`):
```
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin
```

The PyTorch reference inputs at
`/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs/mla_*`
must exist before run; regenerate via
`python3 analysis/wave15-attention-architecture/reference/pytorch_reference_mla.py`.
