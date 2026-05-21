# Wave 19 Phase C1 -- mojo-matmul (naive f32)
#
# Naive matmul mirroring cuda-matmul/matmul.cu exactly: 16x16 block, one
# output per thread, no shared memory. Sizes {1024, 2048, 4096}, 1 warmup
# + 10 timed iters, ctx.execution_time for cudaEvent timing.
#
# Goal: a "Mojo" entry in the headline compute-bound table. Expected behavior
# (per Wave 18 Phase B character): scalar FFMA path, parity with nvcc naive
# (~6.4 TFLOPS at N=4096 from Wave 1 baseline).
#
# Memory layout: A row-major NxN, B row-major NxN, C row-major NxN.
# C[row*N + col] = sum_k A[row*N + k] * B[k*N + col]

from std.math import ceildiv
from std.sys import has_accelerator

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

comptime float_dtype = DType.float32
comptime BS: Int = 16


def matmul_kernel(
    a: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    n: Int,
):
    var row = Int(block_idx.y * block_dim.y + thread_idx.y)
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    if row >= n or col >= n:
        return
    var acc: Float32 = 0.0
    var k: Int = 0
    while k < n:
        acc = acc + a[row * n + k] * b[k * n + col]
        k = k + 1
    c[row * n + col] = acc


def run_one(ctx: DeviceContext, n: Int) raises:
    """Allocate, init, time num_iters launches, report TFLOPS."""

    var elems = n * n
    var a_dev = ctx.enqueue_create_buffer[float_dtype](elems)
    var b_dev = ctx.enqueue_create_buffer[float_dtype](elems)
    var c_dev = ctx.enqueue_create_buffer[float_dtype](elems)

    var a_host = ctx.enqueue_create_host_buffer[float_dtype](elems)
    var b_host = ctx.enqueue_create_host_buffer[float_dtype](elems)
    ctx.synchronize()

    # Init: small magnitudes so the result fits in fp32 without blowing up
    # at N=4096 (mean of N float32 products of ~0.5 each ≈ 0.25 N → at
    # N=4096 ≈ 1024). Use a known seedable pattern.
    for i in range(elems):
        a_host[i] = Float32(((i * 2654435761) % 1024)) * 0.001
        b_host[i] = Float32((((i + 17) * 2654435761) % 1024)) * 0.001

    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.synchronize()

    var a_ptr = a_dev.unsafe_ptr()
    var b_ptr = b_dev.unsafe_ptr()
    var c_ptr = c_dev.unsafe_ptr()

    var grid_x = ceildiv(n, BS)
    var grid_y = ceildiv(n, BS)
    var num_iters = 10

    # Warmup -- covers JIT, populates any caches.
    ctx.enqueue_function[matmul_kernel, matmul_kernel](
        a_ptr, b_ptr, c_ptr, n,
        grid_dim=(grid_x, grid_y), block_dim=(BS, BS),
    )
    ctx.synchronize()

    @parameter
    def body(ctx: DeviceContext) raises -> None:
        ctx.enqueue_function[matmul_kernel, matmul_kernel](
            a_ptr, b_ptr, c_ptr, n,
            grid_dim=(grid_x, grid_y), block_dim=(BS, BS),
        )

    var elapsed_ns = ctx.execution_time[body](num_iters)
    ctx.synchronize()

    # TFLOPS = 2*N^3 ops / time
    var flops_per_iter: Float64 = 2.0 * Float64(n) * Float64(n) * Float64(n)
    var total_flops: Float64 = flops_per_iter * Float64(num_iters)
    var elapsed_s: Float64 = Float64(elapsed_ns) * 1e-9
    var avg_ms: Float64 = Float64(elapsed_ns) / Float64(num_iters) / 1e6
    var tflops: Float64 = total_flops / elapsed_s / 1e12

    print("[mojo-matmul] N=", n,
          " iters=", num_iters,
          " avg_ms/iter=", avg_ms,
          " TFLOPS=", tflops)


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU: ", ctx.name())

        run_one(ctx, 1024)
        run_one(ctx, 2048)
        run_one(ctx, 4096)
