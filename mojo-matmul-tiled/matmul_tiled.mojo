# Wave C1.4 -- mojo-matmul-tiled: classical FFMA-tiled matmul in Mojo
#
# Rounds out the matmul-tiled cross-frontend column alongside cuda-matmul-tiled,
# oxide-matmul-tiled, cutile-matmul-tiled, wgpu-matmul-tiled.
#
# This is the CLASSICAL FFMA tile: BM=BN=BK=16, one output per thread, smem
# staging via copy_dram_to_sram_async, plain *+ FFMA inner loop, NO mma() and
# NO TensorCore. The mojo-matmul-bf16 cell (Wave 21) uses hand-rolled MMA at
# 79.3 TFLOPS — this cell occupies the FFMA-only data point on the spectrum.
#
# Tile shape:
#   BM=BN=16, BK=16, 256 threads/block (one output per thread).
#   Each thread accumulates one C[row,col] in register.
# Per K-tile-pass:
#   Cooperative copy_dram_to_sram_async of A_smem[16x16] and B_smem[16x16].
#   Inner loop: for kk in 0..BK: acc += A_smem[ty,kk] * B_smem[kk,tx].
# Barrier between K-tile iterations.
#
# Shape: M=N=K=4096. Reports min/median/max ms + TFLOPS_median + TFLOPS_best
# and runs sampled correctness (1024 cells, Knuth LCG).

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import (
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async


def matmul_tiled_kernel[
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    A: LayoutTensor[DType.float32, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.float32, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    """C = A * B, classical FFMA-tiled.

    BM=BN=BK=16; one thread per output element. SMEM tiles staged with
    copy_dram_to_sram_async; inner K-loop is plain f32 FFMA.
    """
    comptime K = A.shape[1]()

    # Use flat 1D thread index (Mojo's copy_dram_to_sram_async dispatches
    # by linear lane id; 2D block_dim works but flat is the canonical form
    # used by the rest of the cuda-exploration project).
    var tid = Int(thread_idx.x)
    var ty = tid // BN
    var tx = tid % BN

    # Per-block A,B smem tiles (row-major, BM x BK and BK x BN).
    var A_smem = LayoutTensor[
        DType.float32,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.float32,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Per-block output tile.
    var C_block_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

    # Per-thread f32 accumulator register.
    var acc: Float32 = 0.0

    # K-loop over BK-sized slabs.
    for k_i in range(K // BK):
        barrier()

        # Async DRAM->SMEM cooperative copy of A and B tiles.
        # BM=BK=BN=16, vectorize[1,4] -> 4 vec-cols. 256 threads / 4 = 64
        # vec-loads -> thread_layout=(16,4). Per SKILL caveat:
        # tile_inner_dim/4 = 16/4 = 4; 4 % 4 == 0 OK.
        var A_dram_tile = A.tile[BM, BK](Int(block_idx.y), k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout=Layout.row_major(16, 4)](
            A_smem.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(16, 4)](
            B_smem.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()
        barrier()

        # Inner K-loop: classical FFMA.
        comptime for kk in range(BK):
            acc = acc + A_smem[ty, kk][0] * B_smem[kk, tx][0]

    # Final barrier before writing to C (defensive — last load above does
    # not need to be fenced against the inner compute since compute reads
    # only its own SMEM slice, but a barrier here is harmless and keeps
    # the kernel structure uniform).
    barrier()

    # Write per-thread output element.
    C_block_tile[ty, tx] = acc


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ----- Tile shape -----
        comptime BM = 16
        comptime BN = 16
        comptime BK = 16
        comptime BLOCK_THREADS = BM * BN  # 256, one output per thread

        # ----- Problem size -----
        comptime M = 4096
        comptime N = 4096
        comptime K = 4096
        comptime layout_a = Layout.row_major(M, K)
        comptime layout_b = Layout.row_major(K, N)
        comptime layout_c = Layout.row_major(M, N)

        comptime ftype = DType.float32

        # ----- Buffers -----
        var a_dev = ctx.enqueue_create_buffer[ftype](M * K)
        var b_dev = ctx.enqueue_create_buffer[ftype](K * N)
        var c_dev = ctx.enqueue_create_buffer[ftype](M * N)
        var a_host = ctx.enqueue_create_host_buffer[ftype](M * K)
        var b_host = ctx.enqueue_create_host_buffer[ftype](K * N)
        var c_host = ctx.enqueue_create_host_buffer[ftype](M * N)
        ctx.synchronize()

        # ----- Init: deterministic small magnitudes (mirror mojo-matmul). -----
        for i in range(M * K):
            a_host[i] = Float32(((i * 2654435761) % 1024)) * 0.001
        for i in range(K * N):
            b_host[i] = Float32((((i + 17) * 2654435761) % 1024)) * 0.001
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.synchronize()

        var A_lt = LayoutTensor[ftype, layout_a, MutAnyOrigin](a_dev.unsafe_ptr())
        var B_lt = LayoutTensor[ftype, layout_b, MutAnyOrigin](b_dev.unsafe_ptr())
        var C_lt = LayoutTensor[ftype, layout_c, MutAnyOrigin](c_dev.unsafe_ptr())

        comptime kernel = matmul_tiled_kernel[
            layout_a, layout_b, layout_c, BM, BN, BK,
        ]

        # ----- Warmup -----
        ctx.enqueue_function[kernel, kernel](
            A_lt, B_lt, C_lt,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.synchronize()

        # ----- Timed run: 10 iters, per-iter ctx.execution_time, median. -----
        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[kernel, kernel](
                A_lt, B_lt, C_lt,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(BLOCK_THREADS,),
            )

        var num_iters = 10
        var iter_ms = SIMD[DType.float64, 16](0.0)
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6
        ctx.synchronize()

        # Insertion sort iter_ms[0:num_iters].
        for ii in range(1, num_iters):
            var key = iter_ms[ii]
            var jj = ii - 1
            while jj >= 0 and iter_ms[jj] > key:
                iter_ms[jj + 1] = iter_ms[jj]
                jj -= 1
            iter_ms[jj + 1] = key

        var median_ms = iter_ms[num_iters // 2]
        var min_ms = iter_ms[0]
        var max_ms_iter = iter_ms[num_iters - 1]

        # ----- Copy back -----
        ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
        ctx.synchronize()

        # ----- Sampled correctness (1024 cells, Knuth LCG, high bits) -----
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_i = -1
        var fail_j = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0
        var num_samples = 1024
        for s in range(num_samples):
            var seed = s * 2654435761
            var i = (((seed >> 20) % M) + M) % M
            var j = (((seed >> 11) % N) + N) % N
            var expected: Float32 = 0.0
            for kk in range(K):
                expected += a_host[i * K + kk] * b_host[kk * N + j]
            var got = c_host[i * N + j]
            var abs_err = abs(got - expected)
            var ref_abs = abs(expected)
            var rel_err: Float32 = 0.0
            if ref_abs > 0.0:
                rel_err = abs_err / ref_abs
            if abs_err > max_err:
                max_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            # Acceptance: max_abs_err <= 1e-3 (per task spec).
            if abs_err > 1e-3 and fail_i < 0:
                fail_i = i
                fail_j = j
                fail_got = got
                fail_ref = expected

        # ----- Report -----
        var flops_per_iter: Float64 = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var median_s: Float64 = median_ms * 1e-3
        var tflops_median: Float64 = flops_per_iter / median_s / 1e12
        var tflops_best: Float64 = flops_per_iter / (min_ms * 1e-3) / 1e12

        print("[mojo-matmul-tiled] M=N=K=", M,
              " dtype=f32 FFMA-only",
              " BM=", BM, " BN=", BN, " BK=", BK,
              " min_ms=", min_ms,
              " median_ms=", median_ms,
              " max_ms=", max_ms_iter,
              " TFLOPS_median=", tflops_median,
              " TFLOPS_best=", tflops_best)
        print("[mojo-matmul-tiled] correctness: max_abs_err=", max_err,
              " max_rel_err=", max_rel_err)
        if fail_i >= 0:
            print("[mojo-matmul-tiled] FAIL at (", fail_i, ",", fail_j,
                  "): got=", fail_got, " ref=", fail_ref)
        else:
            print("[mojo-matmul-tiled] correctness PASSED at M=N=K=", M)
