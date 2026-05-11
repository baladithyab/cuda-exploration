# cuTile setup on RTX 5090 + WSL2 + CUDA 13.2

**Wave 12 — cuTile axis, Phase A (smoke test).** Verified 2026-05-11 on Codeseys' RTX 5090 box.

## TL;DR

```bash
cd cutile-vecadd
python3.12 -m venv .venv
source .venv/bin/activate
pip install cuda-tile cupy-cuda13x  # v1.3.0 + v14.0.1
python smoke_test.py
```

Result: ✅ 1M-element vector add, correctness OK, sm_120 native.

## Pip package names (none of which match the README)

The official NVIDIA/cutile-python README example does `pip install cupy-cuda13x` and then says "make sure cuda toolkit 13.1+ is installed". It does **not** tell you the cuTile package name. Discovery sequence:

| Tried | Result |
|---|---|
| `pip install nvidia-cutile` | 404 — no such package |
| `pip install cutile-python` | 404 — repo name ≠ pip name |
| `pip install cuda-tile` | ✅ — this is the canonical name (v1.3.0, March 2026) |

After install: `import cuda.tile as ct`. Note the dot — package on disk is `cuda/tile/`, sibling of `cuda/` namespace already on the system.

## Launch API differs from the README example

**README example (does NOT work in v1.3.0):**

```python
@ct.kernel
def vector_add_kernel(a, b, result): ...

vector_add_kernel[(N // TILE_SIZE,)](a, b, out)  # TypeError: 'kernel' object is not subscriptable
```

**Working launch API:**

```python
import cuda.tile as ct
import cupy

stream = cupy.cuda.get_current_stream()
ct.launch(stream.ptr, (N // TILE_SIZE,), vector_add_kernel, (a, b, out))
cupy.cuda.runtime.deviceSynchronize()
```

`ct.launch` signature (from `cuda.tile._cext.launch.__doc__`):

```
launch(stream, grid, kernel, kernel_args, /)
```

- `stream`: CUDA stream pointer (`cupy.cuda.get_current_stream().ptr` works)
- `grid`: tuple of up to 3 grid dimensions
- `kernel`: the `@ct.kernel`-decorated function
- `kernel_args`: positional args as a **tuple** (NOT *args splat)

The Triton-style `kernel[grid](args)` syntax in the README is either aspirational or from an internal pre-release. v1.3.0 ships with the `ct.launch(...)` form.

## Smoke test results

```
device 0 name: NVIDIA GeForce RTX 5090
device 0 compute capability: sm_120
N=1048576, TILE_SIZE=16, grid=65536
first launch wall time (incl JIT compile): 639.29 ms
correctness: OK  (max abs err = 0.00e+00)
timed 10 iters (cudaEvent): best=0.0332 ms median=0.0340 ms
effective bandwidth: best=378.5 GB/s median=370.6 GB/s
```

**JIT compile cost: 639 ms on first launch** — significant for short scripts but amortizes
across iterations. cuTile compiles to PTX (presumably via libNVVM) and JITs to SASS at first launch.

**Bandwidth: 21% of HBM peak (1792 GB/s)** at TILE_SIZE=16 (=64 B per block, 65k blocks of 1 thread effectively). This is launch-overhead-bound, NOT a cuTile codegen problem.
For the head-to-head matchup against cuda-oxide (1608 GB/s) and nvcc (1609 GB/s) on the same vec-add kernel,
we will need:

1. **Larger TILE_SIZE** (1024 or 4096 elements per tile) to bring the per-launch work up to GiB-class arithmetic.
2. **Bigger N** (256 MB+) to cache-overflow.

This is consistent with cuda-oxide's bench config: 256 MB buffers, multi-thousand-element block work.

## Pitfalls captured for the skill / next phase

1. **Pip package name is `cuda-tile`** (not `nvidia-cutile`, not `cutile-python`).
2. **README launch syntax doesn't work in v1.3.0 (`pip install cuda-tile==1.3.0`)** — use `ct.launch(stream_ptr, grid, kernel, args_tuple)`.
3. **TILE_SIZE matters a lot** — small tiles drown in launch overhead. Match cuda-oxide's per-block work to get apples-to-apples bandwidth comparisons.
4. **CuPy 14.0.1 with `cupy-cuda13x` reports `runtimeGetVersion() == 13000`** even though the system has CUDA 13.2 (driver/runtime forward compat). Note for future bug reports — don't assume the version string matches the toolkit dir.
5. **JIT cost amortizes**: 639 ms first launch, then sub-ms thereafter. For micro-benchmarks always discard the first 1-3 iters as warmup.
