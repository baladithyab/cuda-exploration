# libNVVM-vs-arch causal isolation (Wave 4 W4C)

## Hypothesis

The Wave 3 corrigendum fixed the v0 "safety tax" (safe ≈ 40 % of unchecked at
N=4096) by replacing the shadow libNVVM 7.0.1 with the modern libNVVM 22.0.0
from CUDA 13.2. That single fix conflated two variables:

- **libNVVM version**: 7.0.1 → 22.0.0
- **target arch**: PTX-JIT from compute_89 → native sm_120 cubin

Phase 8 reviewer #2 flagged this: was the safety tax caused by old libNVVM's
codegen quality, or by the PTX-JIT path itself? W4C isolates the variables.

## Method

Hold libNVVM constant (modern 22.0.0 via `LIBNVVM_PATH`) and force the old
arch via `CUDA_OXIDE_TARGET=sm_89`. Compare against the two archival
conditions.

## Four-condition table (N=4096, medians)

| Cond | libNVVM    | arch   | safe TF | unchk TF | safe/unchk | Source                              |
|------|------------|--------|---------|----------|-----------:|-------------------------------------|
| A    | 22.0.0     | sm_120 | 5.47    | 5.61     | 0.97 ×     | oxide-matmul/results.csv (current)  |
| B    | 7.0.1      | sm_89  | 2.01    | 4.96     | 0.41 ×     | `git show 468b446:…/results.csv`    |
| C    | 22.0.0     | sm_89  | —       | —        | —          | **unreachable** (see below)         |

## Result: Condition C is structurally unreachable

With the toolchain fixed at the current `cuda-oxide` revision (6de0509),
libNVVM 22.0.0 **rejects** the codegen backend's `.ll` output for any arch
below sm_120:

```
nvvmCompileProgram failed (9): oxide_matmul.ll (13, 30): parse expected type
```

Col 30 of line 13 is the opaque-pointer keyword `ptr` in a function
signature. The `.ll` carries `!nvvmir.version = !{i32 2, i32 0, …}` but uses
LLVM-15 opaque-pointer syntax. Modern libNVVM only accepts this combination
when targeting sm_120 / sm_121 (verified; sm_75/86/89/90/100/103/110 all
fail with the same parse error; sm_121 parses but cubin is rejected by the
RTX 5090 driver with error 209). The `.ll` itself is **byte-identical**
across `CUDA_OXIDE_TARGET` values — arch selection happens later in libNVVM,
not in rustc-codegen-cuda — so this is a libNVVM-side gate, not a codegen
choice.

## Conclusion: **inconclusive — cannot separate libNVVM-version from arch** on
this toolchain. The two variables are *mechanically* coupled: upgrading
libNVVM while keeping the emitted NVVM IR shape forces a sm_120-class arch.
The v0→v1 safety-tax recovery therefore remains a joint effect of
`(libNVVM 22, sm_120)`; neither factor can be cleanly attributed without
patching rustc-codegen-cuda to emit an older NVVM IR dialect. The Phase 8
hedge in the corrigendum ("caused by libNVVM swap *and/or* arch change")
stands and should not be tightened.

## Artifacts

- `oxide-matmul/run.log.compute89-newvm` — failing `CUDA_OXIDE_TARGET=sm_89`
  run, modern libNVVM.
- Clean-state verification: `CUDA_OXIDE_TARGET` unset, `cargo oxide build
  oxide-matmul` succeeds; subsequent waves inherit sm_120 default.
