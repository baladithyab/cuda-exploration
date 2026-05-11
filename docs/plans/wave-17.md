# Wave 17 plan — extending the cross-frontend × cross-mechanism matrix

**Status:** ready for execution
**Created:** 2026-05-11
**Predecessors:** Wave 16 (`838d9f3`) shipped 4-frontend × 3-mechanism matrix; Wave 17 fills MLA+GDN+KDA gaps.

## Goal

Fill the missing cells in the Wave-16 cross-frontend × cross-mechanism matrix:

| Mechanism | cuda-attn (nvcc) | cublas-attn | cutile-attn | oxide-attn | wgpu-attn |
|---|---|---|---|---|---|
| GQA | ✅ W15 | ✅ W15 | ✅ W15 (165 TF) | ✅ W16 (24 TF) | ✅ W16 (CPU only) |
| MLA | **W17.W1a** | **W17.W2c** | ✅ W16 (112 TF) | **W17.W1b** | — (deferred) |
| GDN | **W17.W1c** | DEFERRED ADR-0006 | ✅ W16 (610 GB/s) | **W17.W1d** | — |
| KDA | — (W18+) | — | **W17.W1e** | — | — |

Plus W17.W2d: cuTile GQA tile-size sweep (close 24% gap vs cuBLAS hgemm peak).

**Total cells: 6 new directories + 1 sweep within existing dir.**

## Constraints (from ADRs and skill)

- **No-TC cells (oxide-attn-*) MUST report HMMA=0 + FFMA + MUFU counts in SASS.** Per ADR-0004.
- **MLA cells (any frontend)** report `useful_flops/wall_time` headline (qk=192) + DRAM/register padding sanity check via LDG bytes. Per ADR-0005.
- **KDA cell** is a semantic-fenced fork of `cutile-attn-gdn`. NO operator reordering. Per ADR-0006.
- **Subagents author + correctness only.** Orchestrator runs final benches serially on idle GPU. Per cuda-exploration session convention.
- **Track `Cargo.lock` and `*.sass` per oxide convention; gitignore them in cuTile cells.** Per Wave 16 fix.
- **cuobjdump must be `/usr/local/cuda/bin/cuobjdump` (CUDA 13.2)** — system cuobjdump silently produces empty SASS on sm_120.

## Wave 1 — pure parallel kernel-authoring (no dependencies)

Five subagents work in fully disjoint directories. No shared files. Dispatched in one `delegate_task(tasks=[...])` batch.

### File-ownership table (Wave 1)

| Worker | Directory | Reads (shared, RO) | Writes |
|---|---|---|---|
| W1a `cuda-attn-mla` | `cuda-attn-mla/` | `analysis/wave15-attention-architecture/reference/{shapes.py,tolerances.py,flops.py,pytorch_reference.py}`, `analysis/wave15-attention-architecture/reference/_mla.py`, `cuda-attn-gqa/attn_gqa.cu` (template), `cublas-attn-gqa/main.cu` (NPY harness), `docs/research/wave17-oxide-mla-design.md`, `docs/adrs/0005-mla-padding-methodology.md` | `cuda-attn-mla/{Makefile,attn_mla.cu,bench.cu,run.sh,run.log,build.log,ANALYSIS.md,.gitignore}` |
| W1b `oxide-attn-mla` | `oxide-attn-mla/` | `oxide-attn-gqa/src/main.rs`, `oxide-matmul-tiled-microtile/src/main.rs`, `analysis/wave15-attention-architecture/reference/_mla.py`, `docs/research/wave17-oxide-mla-design.md`, `docs/adrs/0005-mla-padding-methodology.md` | `oxide-attn-mla/{Cargo.toml,Cargo.lock,src/main.rs,run.sh,run.log,build.log,ANALYSIS.md,.gitignore}` |
| W1c `cuda-attn-gdn` | `cuda-attn-gdn/` | `cutile-attn-gdn/{main.py,ANALYSIS.md}`, `cuda-attn-gqa/attn_gqa.cu` (NPY harness), `analysis/wave15-attention-architecture/reference/_gdn.py`, `docs/research/wave17-gdn-other-frontends.md`, `docs/adrs/0004-no-tc-ceiling-reporting.md` | `cuda-attn-gdn/{Makefile,attn_gdn.cu,bench.cu,run.sh,run.log,build.log,ANALYSIS.md,.gitignore}` |
| W1d `oxide-attn-gdn` | `oxide-attn-gdn/` | `oxide-attn-gqa/src/main.rs`, `cutile-attn-gdn/main.py` (algorithm reference), `analysis/wave15-attention-architecture/reference/_gdn.py`, `docs/research/wave17-gdn-other-frontends.md` | `oxide-attn-gdn/{Cargo.toml,Cargo.lock,src/main.rs,run.sh,run.log,build.log,ANALYSIS.md,.gitignore}` |
| W1e `cutile-attn-kda` | `cutile-attn-kda/` | `cutile-attn-gdn/{main.py,ANALYSIS.md}`, `docs/research/wave17-kda-spec.md`, `docs/adrs/0006-kda-as-gdn-extension.md` | `cutile-attn-kda/{main.py,bench.py,correctness.py,ANALYSIS.md,run.log,.gitignore}` |

**Confirmed disjoint** — no two workers touch the same path. Shared reads are read-only.

### Per-cell acceptance test (numeric ranges, per ADR-0004 + empirical-bench-loop)

| Cell | Correctness | Bench expected range | Sanity |
|---|---|---|---|
| W1a `cuda-attn-mla` | max_abs_err vs PyTorch SDPA-MLA ≤ 1e-2 | TFLOPS ∈ [40, 130] (above cuBLAS-3-kernel-GQA's 46 TF, below cuTile-MLA's 112) | HMMA > 0; HMMA-sanity-formula in [0.5, 2.0]; MUFU > 0 |
| W1b `oxide-attn-mla` | max_abs_err vs PyTorch SDPA-MLA ≤ 1e-2 | TFLOPS ∈ [10, 30] (oxide GQA was 24 TF; MLA expected lower per research doc) | HMMA = 0; FFMA > 0; MUFU > 0 |
| W1c `cuda-attn-gdn` | max_abs_err vs PyTorch GDN-naive ≤ 1e-3 | GB/s ∈ [400, 750] (cuTile is 610) | HMMA = 0; FFMA > 0; LDG.E.128 > 0 (the 'win over cuTile' point) |
| W1d `oxide-attn-gdn` | max_abs_err vs PyTorch GDN-naive ≤ 1e-3 | GB/s ∈ [200, 600] (lower than nvcc/cuTile expected, oxide LLVM 21 hasn't optimized this regime) | HMMA = 0; FFMA > 0 |
| W1e `cutile-attn-kda` | max_abs_err vs `naive_recurrent_kda` ≤ 1e-3 | GB/s ∈ [400, 700] (state traffic 8× less than GDN per research doc) | Diff vs `cutile-attn-gdn/main.py` matches the 4 categories in ADR-0006-§2 |

Outside the range = kernel bug or wrong shape; subagent must explain in ANALYSIS.md before claiming "complete."

### Subagent-prompt skeleton (per W1 task)

Each subagent receives:
1. **The plan slice** (their row above) — explicit file-ownership, expected range.
2. **Read these first** (3-5 specific files with line-ranges where applicable).
3. **Implementation guide** (which existing cell to fork, what to change).
4. **Acceptance test** (numeric range + correctness oracle).
5. **Per-file commit instruction.** "Commit after each of: Cargo.toml/Makefile, kernel, harness, run, ANALYSIS. ~5 commits expected; do NOT batch at the end."
6. **SASS check command** verbatim (with `/usr/local/cuda/bin/cuobjdump` path).
7. **Stop criteria.** "If the kernel doesn't pass correctness after 3 attempts, stop and write a `BLOCKED.md` describing what failed; don't keep trying."

Per `references/run-discipline-patterns.md` lessons: explicit per-file commit beats end-of-task batching; iteration cap hits batched commit phase if not pre-empted.

### Wave 1 commit policy

Each subagent commits its own cell as `Wave 17 W1<letter>: <cell-name> -- <one-line-summary>`. Orchestrator does NOT collapse multiple cells into one commit; per-cell commits make blame trivial.

### Wave 1 budget

- 5 parallel subagents × ~600s budget each = ~10 minutes wall-clock if dispatched together.
- Token budget: ~50-150k summaries returning to orchestrator total.
- Iteration budget per subagent: ~30 calls (fork-pattern is mostly write_file; W1a/c/d are larger LOC).

## Wave 2 — depends on Wave 1's TC numbers

After Wave 1 lands, two more cells need the W1 results to scope their tile choices:

| Worker | Directory | Depends on Wave 1 result |
|---|---|---|
| W2c `cublas-attn-mla` | `cublas-attn-mla/` | W1a's HMMA-count + qk-pad choice; needs to mirror padding policy for fair comparison |
| W2d `cutile-attn-gqa-bigger-tiles` | `cutile-attn-gqa/` (in-place sweep, NEW BLOCK_M variants in same dir) | None (data point: cuTile-GQA at BLOCK_M=64 = 165 TF, target close 24% gap to cuBLAS hgemm 218 TF) |

Wave 2 dispatches together. Wave 2 cells produce the cross-MLA + cross-tile-size data for the Wave-17 summary.

### Wave 2 acceptance

| Cell | Bench expected | Sanity |
|---|---|---|
| W2c `cublas-attn-mla` (192-native) | TFLOPS ≥ 70% of cuTile-MLA's 112 TF (cuBLAS hgemm @ qk=192 is friendly enough) | HMMA > 0; cuBLAS uses `cublasGemmEx` with TF32 |
| W2c `cublas-attn-mla` (padded 256) | TFLOPS ≥ cuTile-MLA's 112 TF (no padding overhead in cuBLAS path) | Same |
| W2d `cutile-attn-gqa-bigger-tiles` | TFLOPS at BLOCK_M=128 ≥ 175 TF (above current 165, target 200+) OR a documented "hit register limit, can't grow" finding | HMMA count ≥ current 256 |

### Wave 2 file ownership

- W2c writes to `cublas-attn-mla/` (NEW). Disjoint.
- W2d writes to `cutile-attn-gqa/sweep_*.py` + appends a "Tile-size sweep" section to `cutile-attn-gqa/ANALYSIS.md`. Modifies existing ANALYSIS.md — orchestrator does the merge after subagent returns the new section as a literal markdown fragment.

## Reflexion checkpoint (between Wave 1 and Wave 2)

After Wave 1 commits land, orchestrator dispatches a tiny reflexion subagent:

- Read the 5 Wave-1 commit messages + their ANALYSIS.md files
- Emit a 5-bullet "lessons" appendix to `AGENTS.md` (or create one if absent)
- Topics: which forks went over the LOC estimate, any pitfall not in the research docs, any SASS surprise

Reflexion is single-bullet-per-cell, not detailed retrospectives.

## Phase 8 final review (after Wave 2)

Per ADR convention + skill rule "scale review fan-out to commit risk":

This is a **medium-risk batch** — 6 new cells of executable benchmark code, several with novel methodology (KDA as GDN fork, MLA padding policy first application). Use **3-reviewer cross-family scatter** even if routing falls back to single-family — context isolation still produces orthogonal findings (per memory note 2026-05-10).

Phase 8 reviewer mandates:
1. Re-compute 5+ headline numbers from per-cell run.log via `python median()` (no trusting headline tables)
2. Confirm SASS evidence files exist and grep counts match ANALYSIS.md claims
3. Verify ADR-0006 fork-fence: `diff cutile-attn-gdn/main.py cutile-attn-kda/main.py` should ONLY show changes in the 4 allowed categories
4. Verify ADR-0005 padding sanity: LDG.E byte counts in MLA cells

## Confounders to control (per empirical-bench variant)

- **Thermal contention:** cells benched serially on idle GPU (≤45°C, ≤72W idle), NOT in parallel.
- **Cooldown:** sleep 10s between adjacent runs.
- **Clock-locking:** not available on WSL2; document as known noise source.
- **Time of day:** all numbers from one continuous benching session.
- **Background processes:** check `nvidia-smi` for other CUDA contexts before each cell.

## Rollback plan

If a cell fails correctness or hits a kernel-level blocker:
1. Subagent writes `BLOCKED.md` in the cell directory describing the failure.
2. Orchestrator commits the BLOCKED.md (preserves the attempt as evidence).
3. Cell goes to W18 backlog with the BLOCKED.md as starting context.
4. Wave 17 summary explicitly notes "5 of 6 cells shipped; cell X blocked because Y."

This wave does NOT block on any single cell — partial completion is acceptable.

## Estimated total budget

- Wave 1 dispatch + commits: ~25 min wall-clock
- Wave 1 reruns on idle GPU: ~10 min
- Wave 2 dispatch + commits: ~15 min wall-clock
- Wave 2 reruns on idle GPU: ~5 min
- Reflexion: ~3 min
- Phase 8 scatter: ~5 min wall-clock for the 3 reviewers
- Final summary write-up: ~10 min orchestrator-authored

**Total: ~75 min wall-clock** assuming no major blocker. Each parallel batch is bounded by the slowest subagent (~600s = 10 min).
