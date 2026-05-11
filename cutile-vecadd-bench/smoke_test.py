"""cuTile smoke test on RTX 5090 / WSL2 / CUDA 13.2.

Adapted directly from the official NVIDIA/cutile-python README example.
Goal: confirm cuTile + CuPy + RTX 5090 (sm_120 Blackwell) actually run end-to-end
before committing to a multi-kernel head-to-head with cuda-oxide / nvcc.
"""

import sys
import time

import cuda.tile as ct
import cupy
import numpy as np


def main() -> int:
    print(f"cupy version: {cupy.__version__}")
    print(f"cupy CUDA runtime: {cupy.cuda.runtime.runtimeGetVersion()}")
    print(f"device 0 name: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    cc_major, cc_minor = cupy.cuda.runtime.getDeviceProperties(0)["major"], cupy.cuda.runtime.getDeviceProperties(0)["minor"]
    print(f"device 0 compute capability: sm_{cc_major}{cc_minor}")
    print(f"cuda.tile module: {ct}")
    print(f"cuda.tile public API (first 30): {sorted(x for x in dir(ct) if not x.startswith('_'))[:30]}")
    print()

    TILE_SIZE = 16
    N = 1 << 20  # 1M elements

    @ct.kernel
    def vector_add_kernel(a, b, result):
        block_id = ct.bid(0)
        a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
        b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
        result_tile = a_tile + b_tile
        ct.store(result, index=(block_id,), tile=result_tile)

    rng = cupy.random.default_rng(42)
    a = rng.random(N, dtype=cupy.float32)
    b = rng.random(N, dtype=cupy.float32)
    out = cupy.empty(N, dtype=cupy.float32)

    print(f"Launching vector_add_kernel: N={N}, TILE_SIZE={TILE_SIZE}, grid={N // TILE_SIZE}")

    # cuTile v1.3.0 launch API: ct.launch(stream, grid, kernel, args)
    grid = (N // TILE_SIZE,)
    stream = cupy.cuda.get_current_stream()

    t0 = time.perf_counter()
    ct.launch(stream.ptr, grid, vector_add_kernel, (a, b, out))
    cupy.cuda.runtime.deviceSynchronize()
    elapsed_first = (time.perf_counter() - t0) * 1000.0
    print(f"first launch wall time (incl JIT compile): {elapsed_first:.2f} ms")

    expected = a + b
    ok = cupy.allclose(out, expected, atol=1e-6, rtol=1e-6)
    max_err = float(cupy.max(cupy.abs(out - expected)))
    print(f"correctness: {'OK' if ok else 'FAIL'}  (max abs err = {max_err:.2e})")
    if not ok:
        print(f"  first 8 expected: {expected[:8].get()}")
        print(f"  first 8 got:      {out[:8].get()}")
        return 1

    # Warm timed iterations
    for _ in range(3):
        ct.launch(stream.ptr, grid, vector_add_kernel, (a, b, out))
    cupy.cuda.runtime.deviceSynchronize()

    iters = 10
    starts = [cupy.cuda.Event() for _ in range(iters)]
    ends = [cupy.cuda.Event() for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        ct.launch(stream.ptr, grid, vector_add_kernel, (a, b, out))
        ends[i].record()
    cupy.cuda.runtime.deviceSynchronize()
    times_ms = [cupy.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(iters)]
    times_ms.sort()
    median = times_ms[iters // 2]
    best = times_ms[0]
    bytes_moved = 3 * N * 4  # read a, read b, write out
    bw_best = bytes_moved / (best * 1e-3) / 1e9  # GB/s
    bw_med = bytes_moved / (median * 1e-3) / 1e9
    print()
    print(f"timed {iters} iters (cudaEvent): "
          f"best={best:.4f} ms median={median:.4f} ms")
    print(f"effective bandwidth: best={bw_best:.1f} GB/s median={bw_med:.1f} GB/s "
          f"(HBM peak ~1792 GB/s on RTX 5090)")

    print()
    print("smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
