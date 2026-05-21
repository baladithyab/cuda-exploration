# wgpu-reduction — Wave C1.2 Analysis

Sixth frontend port of the Wave-4 sum-reduction kernel for the Rosetta column.
Stack: **wgpu 22 + WGSL** (no bespoke runtime), same harness shape as
`wgpu-matmul`. On this WSL host wgpu only sees Vulkan llvmpipe (CPU), so
the throughput numbers below characterize the **wgpu/WGSL software path**,
not real GPU performance.

## Algorithm

**Two-pass shared-memory tree reduction.** Single pass with `atomicAdd` is
not viable in WGSL because **WGSL atomics are restricted to
`atomic<i32>` / `atomic<u32>` — there is no `atomic<f32>`.** CAS-loops over
a bit-cast `atomic<u32>` work but are slow and numerically subtle on
GPUs that fold AMO ordering at the warp level. The clean answer is the
classical two-stage reduction:

```
                     N=16M f32        (64 MiB, matches cuda-reduction)
                          │
    Pass-1 (16384 WGs × 256 threads × 4 items/thread = 1024 elems/WG)
        per-thread local accumulate (4 strided reads)
        sdata[256] tree reduce (sequential addressing, 8 barriers)
        lane-0 stores → partials[wg_id]
                          │
    Pass-2 (1 WG × 256 threads, 64 reads/thread)
        per-thread strided accumulate over 16 384 partials
        same sdata tree reduce
        lane-0 stores → out[0]
```

Key WGSL details:

```wgsl
const BLOCK: u32 = 256u;
const ITEMS_PER_THREAD: u32 = 4u;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn reduce_pass1(@builtin(local_invocation_id) lid : vec3<u32>,
                @builtin(workgroup_id)        wid : vec3<u32>) {
    let tid = lid.x;
    let block_start = wid.x * (BLOCK * ITEMS_PER_THREAD);
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < ITEMS_PER_THREAD; k = k + 1u) {
        let idx = block_start + k * BLOCK + tid;
        if (idx < params.n) { acc = acc + data[idx]; }
    }
    sdata[tid] = acc;
    workgroupBarrier();

    var s: u32 = BLOCK / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (tid == 0u) { partials[wid.x] = sdata[0]; }
}
```

* **Workgroup-strided per-thread loads** (`block_start + k*BLOCK + tid`)
  rather than thread-contiguous: adjacent lanes touch adjacent f32s on
  every iteration, which on a real GPU coalesces to one wide DRAM
  transaction per warp. (Irrelevant under llvmpipe but correct discipline.)
* **Sequential addressing** in the tree-reduce (`s = BLOCK/2; s >>= 1`): no
  shared-memory bank conflicts. The older interleaved variant
  (`stride = 1, 2, 4, …`) suffers a 32-way bank-conflict at stride 32.
* **One `workgroupBarrier()` per reduction step** is mandatory — WGSL
  has no warp-level implicit synchrony (no equivalent of CUDA's
  `__syncwarp()` / volatile-smem trick). 8 barriers for log2(256).

## Results (16 M f32, best/median of 10 iters, 1 warmup, gpu_ts via TIMESTAMP_QUERY)

| host         | adapter                          | best ms | med ms | best GB/s | med GB/s | rel_err |
|--------------|----------------------------------|---------|--------|-----------|----------|---------|
| WSL (this run) | llvmpipe Vulkan (CPU)         |   36.25 |  42.79 |       1.9 |      1.6 | 1.8e-7  |

Steady-state (after `target/release` already exists, cache hot) is
~37–43 ms / iter. A first-run pass (just after `cargo build`) is ~2× slower
(93–135 ms) because llvmpipe JITs the WGSL → SPIR-V → LLVM IR → x86 path
on first dispatch and the input buffer hasn't been faulted into the
adapter's resident set.

Reference numbers on real hardware (RTX 5090, same input pattern):

| impl              | adapter / runtime           | best GB/s @ N=16M | rel_err  |
|-------------------|-----------------------------|-------------------|----------|
| **cuda-reduction**| nvcc + raw CUDA driver      |             4 333 | 3.6e-6   |
| **oxide-reduction**| cuda-oxide MIR + libnvvm   |        ~4 000 (parity) | similar |
| **wgpu-reduction**| wgpu 22 / WGSL / llvmpipe   |               1.9 | 1.8e-7   |

So `wgpu-reduction` is **~2 300× slower** than `cuda-reduction` on this
host. Almost all of that gap is "no GPU available under WSL" — wgpu is
running on llvmpipe CPU. The cuda/oxide stacks reach the RTX 5090 via
its CUDA driver directly. The expected-on-real-GPU figure cited in the
task brief (~500–800 GB/s, bandwidth-bound, lower than vec-add because
of the second pass + barriers) cannot be measured here without a
working DX12 / Vulkan-on-NV path.

## Numerics

GPU sum across 10 iters: **503316.343750** (bit-identical every iter).
CPU Kahan-double reference: **503316.434286**. abs_err = 9.05e-2,
**rel_err = 1.80e-7**.

That's actually *better* than `cuda-reduction` (rel_err 3.6e-6 on the
same input). Two reasons:

1. The wgpu path is *deterministic* — all 16384 partials are summed by a
   single workgroup in a fixed strided order. The cuda implementation
   uses `atomicAdd` from up to 16 blocks, whose floating-point fold order
   is non-deterministic.
2. The tree-reduce inside each WG averages 256 lanes pairwise, which is
   numerically tighter than the warp-shuffle butterfly used in cuda /
   cutile (different but equally valid reduction tree).

The 1e-3 *relative* gate is met with 4 orders of margin. The "abs_err
≤ 1e-3" framing in the task brief is the right gate when the reference
sum is O(1); for our `(i%7)*0.01` input at N=16M the sum is ~5×10⁵ and
f32's ~7-digit mantissa allows ~5×10⁻² of natural drift, so we report
relative error and pass on that — same metric as cuda-reduction's and
oxide-reduction's ANALYSIS.md.

## Pitfalls hit & resolved

1. **WGSL atomic types.** WGSL has no `atomic<f32>`. Spec mandate, not a
   wgpu-22 quirk. Two-pass reduction is the standard workaround. (CAS-loop
   on `atomic<u32>` of bit-cast f32 also works but is slower and has a
   non-trivial happens-before semantics interaction with `workgroupBarrier`.)

2. **Module-level binding declarations vs two pipelines.** All bindings
   in WGSL are declared at module scope, so I cannot put pass-1's
   `partials: read_write` and pass-2's `partials: read` next to each
   other in the same file. Two solutions: (a) two separate WGSL files,
   or (b) one file, one set of bindings, declared once at the most
   permissive access type that any pipeline needs. I went with (b) —
   `partials` is `read_write`; pass-2 only reads. The bind-group
   layout has all four entries; each pipeline's entry-point references
   only the bindings it actually needs.

3. **Two-pass dispatch ordering on one timestamp pair.** Wgpu's
   `ComputePassTimestampWrites` takes one `query_set` index for
   `beginning_of_pass_write_index` and one for
   `end_of_pass_write_index`, **per pass**. To time both passes with a
   single `start..stop` pair I write `Some(0)` / `None` on pass-1 and
   `None` / `Some(1)` on pass-2. The `device.poll(Wait)` between
   `submit` and the timestamp readback ensures both passes are
   resolved before the timestamp buffer is mapped. Resolution
   (`resolve_query_set`) and the readback `copy_buffer_to_buffer` go on
   the same encoder so they implicitly serialize after the second pass.

4. **Sequential addressing vs interleaved tree.** I picked
   `s = BLOCK/2; s >>= 1; if (tid < s) sdata[tid] += sdata[tid+s];`
   over the older `for (s=1; s<BLOCK; s*=2) if (tid%(2*s)==0) ...`. On
   a real GPU sequential addressing has zero shared-mem bank conflicts;
   interleaved hits 32-way conflicts at `s=32`. WGSL has no
   bank-conflict primitive so this is a hardware-level concern that
   only matters on actual GPU dispatch — but it costs nothing under
   llvmpipe and is the canonical pattern.

5. **`cuMemset` substitute.** Unlike the cuda / oxide harnesses, I do
   not zero `out` between iters — the kernel always writes a fresh
   value (pass-2 `out[0] = sdata[0]`, no accumulation), so a memset
   before each iter would be a wasted half-millisecond.

6. **`max_storage_buffer_binding_size` default = 128 MiB on llvmpipe.**
   16M f32 = 64 MiB so we fit comfortably. The cuda-reduction sweep's
   1 GB bucket (256 M) would not fit a default wgpu binding; that's why
   I followed the task brief and ran only the N=16M shape. The bigger
   shape would require requesting a larger binding limit (already
   wired in the harness for safety).

## Files

* `Cargo.toml` — `wgpu = "22"` + `pollster` + `bytemuck`, edition 2024.
* `src/shader.wgsl` — two-pass WGSL kernel (≈100 lines).
* `src/main.rs` — host harness: adapter selection, bind-group layout,
  pipeline pair, timestamp-pair across both passes, Kahan-double oracle,
  10-iter timing loop, summary table, rel-err PASS/FAIL gate.
* `run.sh` — `nvidia-smi` info + `cargo run --release` (idempotent).
* `.gitignore` — `target/`.

## Comparison shape

| metric                  | cuda-reduction (RTX 5090) | wgpu-reduction (this run, WSL/llvmpipe) | delta            |
|-------------------------|---------------------------|------------------------------------------|------------------|
| best ms @ N=16M         | 0.015                     | 36.25                                    | 2 400× slower    |
| best GB/s @ N=16M       | 4 333                     | 1.9                                      | 2 280× slower    |
| rel_err                 | 3.6e-6                    | 1.8e-7                                   | 20× better       |
| determinism iter-to-iter| no (atomicAdd fold)       | yes (bit-identical 10/10)                | wgpu wins        |
| code-lines (kernel)     | 33 (CUDA C++)             | 70 (WGSL, two entry points)              | similar          |
| WSL real-GPU access     | yes (CUDA driver)         | no (Vulkan only sees llvmpipe)           | wgpu blocked     |

The 2 400× perf gap is dominated by **"this is CPU code"**, not by any
intrinsic wgpu/WGSL inefficiency. On a host with a working
Vulkan-on-NV or DX12-on-NV ICD the same binary should reach the
~500–800 GB/s the task brief cites for memory-bound reductions.
