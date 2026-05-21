# wgpu-vecadd — Rosetta Stone Wave C1.1 (vecadd, WebGPU/WGSL frontend)

Sixth frontend port of the canonical bandwidth-bound kernel `c[i] = a[i] + b[i]`,
matching the cuda / cuda-oxide / cutile / mojo / oxide-vecadd cells. Compute stack
is `wgpu = "22"` + WGSL, identical to `wgpu-matmul` (Rosetta Stone Wave C1.0).

## What this cell does

- **Kernel** (`src/shader.wgsl`): 2D-dispatched 256-thread workgroup, one f32 add
  per thread, three storage-buffer binds (A read, B read, C read_write) plus a
  uniform binding for `n`. The flat thread index is reconstructed from the 2D
  grid using `num_workgroups.x * 256` as the row stride.
- **Host** (`src/main.rs`): mirrors `wgpu-matmul`'s scaffolding — Backends::all()
  enumeration, prefer non-CPU adapter, request `Features::TIMESTAMP_QUERY` when
  available, raise `max_storage_buffer_binding_size` to whatever the adapter
  caps it at (skipping any `N` whose 4-byte buffer would exceed it). Sweeps
  `N ∈ {1M, 16M, 64M, 256M}` with 1 warmup + 10 timed iters per N — same shape
  as `cuda-vecadd-bench` and `oxide-vecadd-bench`. Per-iter timing comes from
  the timestamp query when supported, with CPU-wall as fallback. After timing,
  the C buffer is read back in full and `max_abs_err` is computed against the
  host reference.

## Results (run.log captured 2026-05-21)

Adapter: **llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan, type=Cpu)**.

| N (elems)    | per-buf | dispatch grid | best ms | med ms | best GB/s | med GB/s | max_abs_err |
|--------------|--------:|---------------|--------:|-------:|----------:|---------:|------------:|
| 1,048,576    | 4 MiB   | (4096, 1, 1)  |   0.751 |  0.891 |   **16.75** |   14.12 | 0.0 |
| 16,777,216   | 64 MiB  | (65535, 2, 1) |  11.66  | 14.80  |   **17.26** |   13.60 | 0.0 |
| 67,108,864   | —       | —             |    skipped (256 MiB/buf > 128 MiB cap) |||| — |
| 268,435,456  | —       | —             |    skipped (1 GiB/buf > 128 MiB cap)   |||| — |

**Correctness: max_abs_err = 0.0 at every probed N.** Vecadd is bit-exact in
IEEE-754 single precision, so the only failure mode is "wrong indexing" — and
the 2D-grid bounds check passes. Both the 3-point probe (idx=0, n/2, n-1) and
the full-vector readback agree byte-for-byte with the host reference.

## vs cuda-vecadd-bench (Rosetta column delta)

At N=16M, where the cuda baseline is **best 2003 GB/s, median 1820 GB/s**:

| frontend     | best GB/s | vs cuda |
|--------------|----------:|--------:|
| cuda         |      2003 |   1.00× |
| oxide-vecadd |      ~1500 (W4B headline) | ~0.75× |
| **wgpu**     |  **17.26** |  **0.0086×**  (−99.1%) |

The task prompt anticipated `wgpu vs cuda likely −60..−70% due to Vulkan
overhead`. We blew past that: the actual delta is **−99.1%**, two orders of
magnitude beyond the expected band. The reason is **not** Vulkan overhead per
se — it's that on WSL there is no working GPU Vulkan ICD. The only `wgpu`
backend that resolves an adapter is Vulkan-backed by **Mesa llvmpipe**, which
is a CPU rasterizer/compute fallback. We are not measuring the RTX 5090 at all;
we are measuring an LLVM JIT-compiled CPU kernel on however many host cores
llvmpipe decides to use.

This is the same caveat `wgpu-matmul`'s `run.log` and `ANALYSIS.md` already
flag for the matmul case (its 0.005 TFLOPS reading vs nvcc's 1+ TFLOPS at the
same shape). The Rosetta column captures **what the WebGPU frontend produces
on this concrete WSL host** — not what the WebGPU programming model is capable
of on a hosted GPU. A re-bench on a native-Linux box with `wgpu` ↔ Vulkan ↔
NVIDIA's userspace driver should land in the −60..−70% band the task predicted.

## Pitfalls hit (and the fixes)

1. **`max_compute_workgroups_per_dimension = 65535`.** A pure 1D dispatch
   `(ceil(N/256), 1, 1)` validates fine for N ≤ 16,776,960 elements but the
   N=16M case (16,777,216) needs 65,536 workgroups along x — one over the
   default cap. Wgpu rejects with
   `Each current dispatch group size dimension ([65536, 1, 1]) must be less
   or equal to 65535`. Fix: 2D dispatch with `wg_x = min(total_wg, 65535)` and
   `wg_y = ceil(total_wg / 65535)`. The shader reconstructs the flat index as
   `gid.y * (num_workgroups.x * 256) + gid.x` so we can scale to N=64M+ without
   touching adapter limits. *(CUDA folks: same pitfall as `gridDim.x` cap on
   ancient capabilities, just at a wgpu-default value rather than a hardware
   limit.)*

2. **`max_storage_buffer_binding_size = 128 MiB` on llvmpipe (and most adapter
   defaults).** Requesting a higher limit only succeeds if the adapter
   advertises it. Llvmpipe hard-caps at 128 MiB regardless of what we ask, so
   `request_device` fails with `LimitsExceeded` if we request 1 GiB. Fix: clamp
   the requested limit to the adapter's `adapter.limits()` value, then skip any
   `N` whose 4-byte-per-elem buffer exceeds it. This rules out N=64M (256 MiB)
   and N=256M (1 GiB) on llvmpipe. A CUDA-driven adapter would lift this and
   we'd run the full sweep.

3. **WGSL has no atomic-vs-non-atomic gotcha here.** The kernel is purely
   stream-write — each thread owns a unique `c[idx]` — so the storage buffer
   binding is plain `var<storage, read_write>`, no `atomic<f32>` (which WGSL
   doesn't support anyway), no barriers, no sync. The only sync is wgpu's
   implicit one between `queue.submit` and the readback's `device.poll(Wait)`.

4. **Storage-buffer alignment.** WGSL requires storage buffer arrays of `f32`
   to be 4-byte-aligned (which they are by definition for `array<f32>`), and
   uniform structs to be 16-byte-aligned overall. We use a uniform of a single
   `u32` — wgpu pads it transparently. No explicit `align(16)` decorations
   were needed.

5. **Timestamp queries: feature-gated.** Llvmpipe happens to expose
   `Features::TIMESTAMP_QUERY` (the matmul cell verified this earlier), so
   `gpu_ts` is the reported number. On adapters without it, the code falls
   back to `cpu_wall` and labels the column accordingly. The two are within
   10–30% of each other on llvmpipe, with `cpu_wall` consistently higher
   because it includes `submit + poll(Wait)` overhead.

6. **WSL adapter selection.** Same as wgpu-matmul: enumerate all backends,
   prefer the first non-CPU candidate, fall back to CPU. On this WSL host the
   non-CPU branch never triggers — only llvmpipe is reachable.

## Files in this cell

- `Cargo.toml` — `wgpu = { version = "22", features = ["dx12"] }`, `pollster`,
  `bytemuck`, edition 2024 (matches `wgpu-matmul`).
- `src/shader.wgsl` — vecadd compute kernel (~10 SLOC + comments).
- `src/main.rs` — host orchestration, sweep, timestamp queries, readback,
  correctness probe, summary table.
- `run.sh` — `cargo build --release && ./target/release/wgpu-vecadd | tee
  run.log`.
- `run.log` — captured output from the run that produced the table above.
- `.gitignore` — `target/`.
- `ANALYSIS.md` — this file.

## Open question for the Rosetta column

The mojo / cutile / oxide / cuda cells all measure the RTX 5090's actual DRAM
bandwidth. Wgpu in this environment doesn't, so the "WebGPU column" entry is
**not directly comparable** to the others on this host. Two options for the
column-builder:

- **(a)** Run wgpu cells on a hosted Linux VM with a real Vulkan ICD on the
  GPU (NVIDIA's `nvidia-icd` or Mesa `radv`), regenerate the row, and footnote
  the WSL number as a CPU baseline.
- **(b)** Keep the WSL number, label the column "wgpu/llvmpipe (CPU
  fallback)" so the reader knows the comparison is across stacks, not against
  the same hardware.

Either is defensible; (a) gives a true Rosetta entry, (b) characterises the
"what does an end-user with a default WSL setup see when they run wgpu?"
experience, which is itself useful information.
