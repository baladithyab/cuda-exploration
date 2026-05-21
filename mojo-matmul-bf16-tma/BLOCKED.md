# Wave 22.1 — BLOCKED on Mojo 1.0.0b1 API gap

**Status:** BLOCKED until Mojo ships TMA / cp.async.bulk primitives.
**Date investigated:** 2026-05-21.
**Investigator:** orchestrator (subagent timed out at 600s; orchestrator
ran the API exploration inline via 30+ probe compiles).

## Goal

Trigger UTMALDG TMA loads from a Mojo kernel via `std.gpu.sync.cp_async_bulk`
or equivalent, to close the W21 ⟨cp.async vs TMA⟩ gap and lift `mojo-matmul-bf16`
from 79.3 TF (49.6% of cuTile bf16) toward the cuTile 159.95 TF ceiling.

## Investigation method

Ran probe-compiles of every plausible candidate name in `std.gpu.sync` and
candidate sub-modules:

| Probe | Result |
|---|---|
| `from std.gpu.sync import *` | OK (module exists) |
| `from std.gpu.sync import cp_async_bulk` | NOPE (does not contain) |
| `from std.gpu.sync import async_copy_bulk` | NOPE |
| `from std.gpu.sync import tma_load` | NOPE |
| `from std.gpu.sync import tma_2d_load` | NOPE |
| `from std.gpu.sync import cp_tensor_async` | NOPE |
| `from std.gpu.sync import TMATile` | NOPE |
| `from std.gpu.sync import TMA` | NOPE |
| `from std.gpu.sync import tma_descriptor` | NOPE |
| `from std.gpu.sync import TensorMap` | NOPE |
| `from std.gpu.sync import cp_async_bulk_tensor` | NOPE |
| `from std.gpu.sync import wait_async` | NOPE |
| `from std.gpu.sync import wait_async_bulk` | NOPE |
| `from std.gpu.sync import fence_async_view` | NOPE |
| `from std.gpu.sync import barrier_async` | NOPE |
| `from std.gpu.sync import mbarrier_arrive` | **OK — exists** |
| `from std.gpu.sync import cp_async` | NOPE |
| `from std.gpu.tma import *` | ABSENT (module not located) |
| `from std.gpu.async_copy import *` | ABSENT |
| `from std.gpu.bulk import *` | ABSENT |
| `from std.gpu.compute.async import *` | parse error (`async` is a reserved keyword in Mojo) |

## Conclusion

Mojo 1.0.0b1's `std.gpu.sync` module exposes only **mbarrier-related
primitives** (mbarrier_arrive and similar), NOT cp.async.bulk / TMA /
cuTensorMap APIs. There is no module path through which a Mojo kernel
can issue UTMALDG SASS instructions today.

The closest Mojo primitive is `copy_dram_to_sram_async` (used in the
Wave 21 baseline), which lowers to plain cp.async (LDGSTS), not the TMA
bulk path that emits UTMALDG.

## Mojo dispatcher detail (from Wave 19/21/22.4 prior findings)

The `from std.gpu.compute.mma import mma` dispatcher already routes by
`(A_dtype, B_dtype, C_dtype, D_dtype, M, N, K)` tuple match against
`mma_nvidia` (sm_70-sm_90 path) and `mma_nvidia_sm100` (tcgen05/sm_100a).
Adding TMA support would need a parallel `cp_async_bulk` lowering hook
in the std.gpu.sync module. The closest existing infrastructure is the
mbarrier ops, which are the consumer side of any TMA pipeline; the
producer side (the cp.async.bulk.tensor instruction itself) has no
exposed Mojo binding.

## Recommendation

Defer until one of:

1. **Modular ships `cp_async_bulk` / TMA primitives in std.gpu.sync.** A PR
   that mirrors the bf16 `mma` lane onto a `cp_async_bulk_tensor_2d` lowering
   over the existing mbarrier infrastructure would unblock this. Worth
   filing as an upstream Mojo issue — Modular's roadmap for 1.x likely
   includes it for Hopper/Blackwell perf parity.
2. **Use `inlined_assembly` PTX inline** to emit `cp.async.bulk.tensor` and
   the matching `mbarrier.expect_tx` directly from Mojo. This is an
   experimental escape hatch (untested in 1.0.0b1; `inlined_assembly` does
   exist per the LLVM intrinsic mechanism that mma() uses, but the surface
   for arbitrary PTX is undocumented). Estimated effort: 200+ LOC of inline
   PTX + descriptor setup; high risk of subtle correctness bugs without
   `cuTensorMapEncodeTiled` (which Mojo would need to call from host).

## Comparable success path (different frontend)

Wave 22.10 succeeded at the same goal in a different frontend:
`cuda-attn-gdn-tma/` (CUDA C++) used `cuTensorMapEncodeTiled` host-side
+ `cp.async.bulk.tensor.2d` device-side and shipped at **1032 GB/s best
on RTX 5090 sm_120 (UTMALDG.2D x 2 in SASS)**, beating cuTile's 610 GB/s.
This proves the TMA API path is viable on consumer Blackwell; the gap is
purely the Mojo language binding, not a hardware constraint.

## Files in this cell

- `probe_imports.mojo` — generic GPU-detect probe (the original blocked
  subagent's run.sh references this; left here for documentation).
- `probe_cp_async_bulk_tensor.mojo` — the failing-import probe documenting
  the API gap.
- `BLOCKED.md` — this file.

No kernel was shipped because no API path exists in Mojo 1.0.0b1.

## Next-loop trigger

When `pixi run mojo --version` reports >= 1.1 OR Modular announces a
GPU stdlib expansion that includes TMA, re-run this probe; if it now
imports, port the Wave 21 `mojo-matmul-bf16/matmul_bf16.mojo` kernel
swapping `copy_dram_to_sram_async` for the new TMA primitive.
