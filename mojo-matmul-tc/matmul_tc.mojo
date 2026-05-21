# Wave 19 Phase C3 -- mojo-matmul-tc: Tensor Core matmul probe on sm_120
#
# Adapted from modular/max/examples/custom_ops/kernels/matrix_multiplication.mojo
# (the canonical `tensor_core_matrix_multiplication` from the Mojo GPU tutorial).
# We strip out the MAX graph wrapper and run as a standalone Mojo program with
# raw DeviceContext, mirroring the Wave 18 mojo-matmul/ harness shape.
#
# CRITICAL HYPOTHESIS this kernel tests:
#   Does Mojo's `from layout.tensor_core import TensorCore` actually emit
#   `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` (or equivalent) on sm_120?
#   If yes -> Mojo can engage tensor cores on consumer Blackwell, joining
#   cuTile in the TC-capable club (currently cuda-oxide is shut out per
#   Wave 14 finding).
#   If no -> hard error at compile/runtime, which itself is a finding.
#
# Tile shape: bf16 inputs, f32 output, MMA_M=16, MMA_N=8, MMA_K=8 (legacy
# NVIDIA mma.sync that runs sm_70+ through sm_120). The MAX dispatcher
# uses MMA_M=16/MMA_N=8/MMA_K=4 for f32 inputs, but we're going bf16 input
# + f32 accumulator which is the standard m16n8k8 / m16n8k16 path.

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from layout.tensor_core import TensorCore
from std.utils.index import Index


def tc_matmul[
    a_type: DType,
    c_type: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[a_type, layout_a, MutAnyOrigin],
    B: LayoutTensor[a_type, layout_b, MutAnyOrigin],
    C: LayoutTensor[c_type, layout_c, MutAnyOrigin],
):
    """Tensor-Core GEMM kernel C = A * B.

    Direct port of `tensor_core_matrix_multiplication` from the Mojo manual.
    Uses TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]() for
    arch-aware MMA dispatch.
    """
    comptime M = C.shape[0]()
    comptime N = C.shape[1]()
    comptime K = A.shape[1]()

    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    var C_warp_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[
        WM, WN
    ](warp_y, warp_x)

    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0
    ), "Warp tile must be a multiple of MMA shape"

    var mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    var A_smem = LayoutTensor[
        a_type,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        a_type,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_reg = (
        LayoutTensor[
            c_type,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    for k_i in range(K // BK):
        barrier()

        var A_dram_tile = A.tile[BM, BK](Int(block_idx.y), k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_smem.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            B_smem.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()
        barrier()

        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    var c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    var a_reg = mma_op.load_a(A_mma_tile)
                    var b_reg = mma_op.load_b(B_mma_tile)
                    var d_reg_m_n = mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                    c_reg_m_n.copy_from(d_reg_m_n)

    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
            mma_op.store_d(C_mma_tile, c_reg_m_n)


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU: ", ctx.name())

        # Match the official Mojo tutorial NVIDIA shape: f32-in/f32-out
        # using TF32 tensor cores. MMA_K=4 selects the m16n8k4 TF32 path
        # which works sm_80+ through sm_120. (bf16/f16 mixed-precision needs
        # a different code path -- the layout.tensor_core wrapper requires
        # A.dtype == C.dtype, see Wave 19 ANALYSIS.md.)
        comptime a_type = DType.float32
        comptime c_type = DType.float32

        comptime BM = 64
        comptime BN = 64
        comptime BK = 32      # OPTIMIZED_BLOCK_SIZE on NVIDIA
        comptime WM = 32
        comptime WN = WARP_SIZE  # 32 on NVIDIA
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 4       # TF32 path on Ampere/Ada/Blackwell
        comptime NUM_WARPS = (BM // WM) * (BN // WN)

        # 4096x4096x4096 to match cuBLAS hgemm baseline
        comptime M = 4096
        comptime N = 4096
        comptime K = 4096
        comptime layout_a = Layout.row_major(M, K)
        comptime layout_b = Layout.row_major(K, N)
        comptime layout_c = Layout.row_major(M, N)

        var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
        var b_dev = ctx.enqueue_create_buffer[a_type](K * N)
        var c_dev = ctx.enqueue_create_buffer[c_type](M * N)

        # Init via host buffer + copy
        var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
        var b_host = ctx.enqueue_create_host_buffer[a_type](K * N)
        ctx.synchronize()
        for i in range(M * K):
            a_host[i] = Float32(((i * 2654435761) % 256)) * 0.001
        for i in range(K * N):
            b_host[i] = Float32((((i + 17) * 2654435761) % 256)) * 0.001
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.synchronize()

        var A_lt = LayoutTensor[a_type, layout_a, MutAnyOrigin](
            a_dev.unsafe_ptr()
        )
        var B_lt = LayoutTensor[a_type, layout_b, MutAnyOrigin](
            b_dev.unsafe_ptr()
        )
        var C_lt = LayoutTensor[c_type, layout_c, MutAnyOrigin](
            c_dev.unsafe_ptr()
        )

        comptime kernel = tc_matmul[
            a_type,
            c_type,
            layout_a,
            layout_b,
            layout_c,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
        ]

        # Warmup
        ctx.enqueue_function[kernel, kernel](
            A_lt, B_lt, C_lt,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_WARPS * WARP_SIZE),
        )
        ctx.synchronize()

        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[kernel, kernel](
                A_lt, B_lt, C_lt,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_WARPS * WARP_SIZE),
            )

        var num_iters = 10
        var elapsed_ns = ctx.execution_time[body](num_iters)
        ctx.synchronize()

        var flops_per_iter: Float64 = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var total_flops: Float64 = flops_per_iter * Float64(num_iters)
        var elapsed_s: Float64 = Float64(elapsed_ns) * 1e-9
        var avg_ms: Float64 = Float64(elapsed_ns) / Float64(num_iters) / 1e6
        var tflops: Float64 = total_flops / elapsed_s / 1e12

        print("[mojo-matmul-tc] M=N=K=", M,
              " a_type=f32 c_type=f32 (TF32 TC path)",
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K,
              " avg_ms/iter=", avg_ms,
              " TFLOPS=", tflops)
