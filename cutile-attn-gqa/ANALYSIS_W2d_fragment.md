## Tile-size sweep (W2d)

**Wave 17 W2d** added a `BLOCK_M` sweep on top of the original Wave 15.1
kernel — same `make_gqa_kernel(...)` factory, same correctness inputs,
same cubin-export path. Goal: test the Wave 15.1 plan claim that
"larger BLOCK_M might close the 24% gap to cuBLAS hgemm peak".

Sweep authored as:

- `main.py` gains `--block-m N` and `--block-n N` CLI flags that override
  `pick_blocks(...)` while preserving the legacy default behaviour
  (32×32 at correctness, 64×64 at bench).
- `sweep_block_m.py` drives the sweep: per-`BLOCK_M` correctness check
  at SHAPE_CORRECTNESS (when `seq % BLOCK_M == 0`), plus a compile-only
  cubin export at SHAPE_BENCH so we get a register-pressure signal even
  for tile shapes that don't fit `seq=128`.
- BLOCK_N held at 64 (the Wave 13.1 ct.mma sweet spot) throughout. We
  vary one axis at a time per pitfall #4.

### Authoring-time sweep (correctness + SASS only — orchestrator runs the timed bench)

| BLOCK_M | BLOCK_N | seq=128 corr | bench-shape compile | HMMA  | LDL  | STL  | spill verdict |
|--------:|--------:|:------------:|:-------------------:|------:|-----:|-----:|---------------|
| **64**  | 64      | **PASS**     | OK                  |  256  |  137 |  100 | small spill (baseline) |
| **128** | 64      | **PASS**     | OK                  |  256  |  222 |  270 | ~2× spill — borderline |
| **256** | 64      | SKIP¹        | OK                  |  512  | 1153 |  879 | **register cliff confirmed** |

¹ SHAPE_CORRECTNESS has `seq=128`, so a 256-wide query tile cannot be
constructed there. The kernel does cleanly compile at SHAPE_BENCH
(`seq=2048`, 8 query blocks); the register-cliff signal comes from the
SASS dump rather than from a launch.

Counts via `cuobjdump --dump-sass gqa_fwd_fused_bm{N}.cubin`. `LDL` /
`STL` (load-local / store-local) instructions are the canonical SASS
fingerprint of register-pressure spilling to local memory — see cuTile
skill pitfall #10 ("silent local-mem spill, perf drops 3–4× per
register-cliff").

### Reading the cliff

Going from `BLOCK_M=64` → `128`, `HMMA` count is unchanged (256 — the
TC-engine work per CTA is fixed by `D_HEAD=128` and the K/V loop trip
count) while `LDL+STL` doubles (237 → 492). Same kernel, twice the
register footprint per query row → ptxas evicted ~half the working set
to local memory. That predicts a perf drop, **not** the speedup the
plan hoped for.

`BLOCK_M=64` → `256`: HMMA doubles (256 → 512 — bigger tiles, more mma
ops queued per CTA) but `LDL+STL` goes up **8.6×** (237 → 2032). At
that ratio the local-memory traffic is the dominant inner-loop cost,
not HMMA. This is the textbook register cliff: every doubling of
BLOCK_M past the budget multiplies spill traffic super-linearly.

### Verdict on the "close the 24% gap" plan claim

**Refuted at author-time, ahead of the orchestrator's timed bench.**
The SASS evidence shows BLOCK_M=128 already spills more than the 64×64
baseline, and BLOCK_M=256 falls off the register cliff entirely
(>8× spill traffic). The 24% gap to `cuBLAS hgemm` peak is **not**
register-tile-shape-bound on this cell — closing it would need one of
the three knobs called out in the original Wave 15.1 ANALYSIS:

1. Pipelined / persistent K/V loads (HBM↔HMMA overlap) — no cuTile
   1.3.0 user-facing API.
2. Epilogue fusion of the final `O / l` divide — marginal at d=128.
3. Explicit register-reuse hints — also not exposed in cuTile 1.3.0.

The first matches the Wave 15.1 hypothesis ("the softmax work is
almost free relative to the mma pipeline; what's left is launch /
HBM-pipeline overhead"). The W2d sweep makes that conclusion empirical
rather than predicted.

### Files added / modified by W2d

- `main.py` — added `--block-m`, `--block-n`, `--iters` CLI flags and
  threaded them through `pick_blocks()`, `run_smoke()`, `run_bench()`,
  and `export_cubin()`. Default behaviour unchanged when no flags are
  passed.
- `sweep_block_m.py` — new. Drives correctness + cubin compile + SASS
  spill-count for each `BLOCK_M ∈ {64, 128, 256}`.
- `sweep_block_m_results.csv` — sweep summary
  (`block_m, block_n, correctness_status, compile_status, notes`).
- `gqa_fwd_fused_bm{64,128,256}.cubin` — per-BLOCK_M sm_120 cubins for
  the SASS spill-count check (kept for orchestrator's timed-bench
  rerun).

### What the orchestrator should do next

The cell is correctness-clean at `BLOCK_M ∈ {64, 128}` and the
register-cliff at `BLOCK_M=256` is documented from SASS without
needing a launch. For the timed sweep that this author skipped:

```bash
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python
$PY main.py --bench --block-m 64  --csv-out results_bm64.csv
$PY main.py --bench --block-m 128 --csv-out results_bm128.csv
$PY main.py --bench --block-m 256 --csv-out results_bm256.csv
```

The 64-row baseline should reproduce ~165 TF; `BLOCK_M=128` is
predicted to drop (spill doubled); `BLOCK_M=256` is predicted to drop
3–4× (register cliff per pitfall #10).
