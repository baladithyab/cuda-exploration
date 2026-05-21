# Wave 22.3 -- mojo-matmul-bf16 padded-smem variant.
#
# Variant of the W21 baseline (matmul_bf16.mojo) with PADDED shared-memory
# layouts to test R2's W21 hypothesis that 5-10% perf may be available by
# breaking the natural bank-conflict pattern when ldmatrix-style smem
# loads are issued against `Layout.row_major(BM, BK)`.
#
# Strategy:
#   - Allocate A_smem with Layout.row_major(BM, BK_PAD) where BK_PAD = BK + 8.
#     The +8 bf16 elements per row = +16 bytes, which shifts each successive
#     row by an odd number of 32B banks => no two rows alias the same bank.
#   - Same for B_smem with Layout.row_major(BK_PAD, BN). (Padding the K-dim
#     for B places the padding at the *end of each "row" of the (BK,BN) tile,
#     which here is at the bottom of the tile -- but we still pad the leading
#     dim's stride to BN+0 ... no, B is row-major(BK,BN) so the leading dim
#     is BK. We pad as row-major(BK, BN+8) to give 16B inter-row offset
#     after vectorize[1,4].)
#   - For the cp.async copy, build a (BM, BK) NON-padded view of the padded
#     memory using a custom Layout(IntTuple(BM,BK), IntTuple(BK_PAD,1)).
#     The copy primitive then operates on a shape-matched DRAM<->SMEM pair.
#   - For the MMA inner loop, do the same: warp-tile slicing happens through
#     the non-padded *view*, so .tile[WM,BK] and .tile[MMA_M,MMA_K] both have
#     clean divisibility.
#
# Acceptance for this wave: compile, PASS correctness at M=N=K=64, SASS still
# shows HMMA.16816.F32.BF16 > 0, and report the LDS instruction-mix counts
# (LDS.U.32 / LDS.U.128 / LDS.U16 etc) so we can compare to W21 baseline.

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
    warp_id,
    lane_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.compute.mma import mma
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from layout.int_tuple import IntTuple
from layout.tensor_core import TensorCore
from std.utils.index import Index


def matmul_bf16_padded_kernel[
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    BK_PAD: Int,
    BN_PAD: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[DType.bfloat16, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.bfloat16, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    """C = A * B with bf16 inputs and f32 accumulator on Blackwell sm_120.

    Padded-smem variant of W21 baseline. Each smem row is padded to break
    the natural bank-conflict alignment of row_major(64,32) bf16.
    """
    comptime K = A.shape[1]()

    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    # Load-A and load-B helpers as bf16/bf16 TensorCore (uniform dtype, OK).
    var loader = TensorCore[DType.bfloat16, DType.bfloat16, Index(MMA_M, MMA_N, MMA_K)]()

    # Per-block C tile -> per-warp output tile.
    var C_warp_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[
        WM, WN
    ](warp_y, warp_x)

    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and BK % MMA_K == 0
    ), "Warp tile and BK must be multiples of MMA shape"

    # ----- Padded shared memory.
    # A_smem: shape (BM, BK_PAD), stride (BK_PAD, 1). Backing memory =
    # BM*BK_PAD bf16 elements. Only first BK columns of each row are used;
    # the trailing pad columns are dead space inserted to displace the
    # natural mod-32B bank alignment between successive rows.
    var A_smem_padded = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BM, BK_PAD),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # B_smem: shape (BK, BN_PAD), stride (BN_PAD, 1). Backing memory =
    # BK*BN_PAD bf16 elements.
    var B_smem_padded = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BK, BN_PAD),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Non-padded (BM,BK) view of the padded A_smem -- same backing pointer,
    # but layout shape=(BM,BK) and stride=(BK_PAD,1). All subsequent slicing
    # (.tile[WM,BK], .tile[MMA_M,MMA_K]) goes through this view so divisibility
    # checks always succeed and the padding columns are simply skipped.
    alias A_view_layout = Layout(IntTuple(BM, BK), IntTuple(BK_PAD, 1))
    alias B_view_layout = Layout(IntTuple(BK, BN), IntTuple(BN_PAD, 1))
    var A_smem = LayoutTensor[
        DType.bfloat16,
        A_view_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](A_smem_padded.ptr)
    var B_smem = LayoutTensor[
        DType.bfloat16,
        B_view_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](B_smem_padded.ptr)

    # f32 accumulator: per-warp WM/MMA_M × WN/MMA_N tiles, each holding
    # 4 f32 per lane (the m16n8 output distribution: 4 regs/lane).
    var c_reg = (
        LayoutTensor[
            DType.float32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    # K-loop over BK-sized tile slabs.
    for k_i in range(K // BK):
        barrier()

        # Async DRAM→SMEM copy. Source is (BM,BK) DRAM tile, destination
        # is the (BM,BK) non-padded view of the padded smem -- copy primitive
        # respects the per-row stride from the destination's layout, so each
        # row lands at offset row*BK_PAD in smem with a BK-element gap.
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

        # Per-warp slice of the smem tiles (uses the non-padded view, so the
        # stride-(BK_PAD,1) propagates into the warp/mma tiles automatically).
        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

        # Inner MMA loop.
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    var a_lt = loader.load_a(A_mma_tile)
                    var b_lt = loader.load_b(B_mma_tile)

                    var a_frag = SIMD[DType.bfloat16, 8](0)
                    a_frag[0] = a_lt[0, 0][0]
                    a_frag[1] = a_lt[0, 1][0]
                    a_frag[2] = a_lt[0, 2][0]
                    a_frag[3] = a_lt[0, 3][0]
                    a_frag[4] = a_lt[0, 4][0]
                    a_frag[5] = a_lt[0, 5][0]
                    a_frag[6] = a_lt[0, 6][0]
                    a_frag[7] = a_lt[0, 7][0]
                    var b_frag = SIMD[DType.bfloat16, 4](0)
                    b_frag[0] = b_lt[0, 0][0]
                    b_frag[1] = b_lt[0, 1][0]
                    b_frag[2] = b_lt[0, 2][0]
                    b_frag[3] = b_lt[0, 3][0]

                    var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var c_frag = SIMD[DType.float32, 4](0)
                    c_frag[0] = c_reg_tile[0, 0][0]
                    c_frag[1] = c_reg_tile[0, 1][0]
                    c_frag[2] = c_reg_tile[0, 2][0]
                    c_frag[3] = c_reg_tile[0, 3][0]
                    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

                    mma(d_frag, a_frag, b_frag, c_frag)

                    c_reg_tile[0, 0] = d_frag[0]
                    c_reg_tile[0, 1] = d_frag[1]
                    c_reg_tile[0, 2] = d_frag[2]
                    c_reg_tile[0, 3] = d_frag[3]

    # ----- Hand-rolled epilogue per PTX 9.7.13.4.8 m16n8 distribution. -----
    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)

            comptime for i in range(4):
                var row = group_id + (i >> 1) * 8
                var col = (tid_in_grp << 1) + (i & 1)
                C_mma_tile[row, col] = c_reg_tile[0, i]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ----- Tile shape -----
        comptime BM = 64
        comptime BN = 64
        comptime BK = 32
        comptime BK_PAD = BK + 8  # 40 bf16 elements per A_smem row
        comptime BN_PAD = BN + 8  # 72 bf16 elements per B_smem row
        comptime WM = 32
        comptime WN = 32
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 16
        comptime NUM_WARPS = (BM // WM) * (BN // WN)
        comptime BLOCK_THREADS = NUM_WARPS * WARP_SIZE

        # ----- Problem size: M=N=K=64 (correctness focus). -----
        comptime M = 64
        comptime N = 64
        comptime K = 64
        comptime layout_a = Layout.row_major(M, K)
        comptime layout_b = Layout.row_major(K, N)
        comptime layout_c = Layout.row_major(M, N)

        comptime a_type = DType.bfloat16
        comptime c_type = DType.float32

        # ----- Buffers -----
        var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
        var b_dev = ctx.enqueue_create_buffer[a_type](K * N)
        var c_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
        var b_host = ctx.enqueue_create_host_buffer[a_type](K * N)
        var c_host = ctx.enqueue_create_host_buffer[c_type](M * N)
        ctx.synchronize()

        # Init A, B with deterministic small-magnitude bf16.
        for i in range(M * K):
            a_host[i] = (Float32(((i * 2654435761) % 256)) * 0.001).cast[a_type]()
        for i in range(K * N):
            b_host[i] = (Float32((((i + 17) * 2654435761) % 256)) * 0.001).cast[a_type]()
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.synchronize()

        var A_lt = LayoutTensor[a_type, layout_a, MutAnyOrigin](a_dev.unsafe_ptr())
        var B_lt = LayoutTensor[a_type, layout_b, MutAnyOrigin](b_dev.unsafe_ptr())
        var C_lt = LayoutTensor[c_type, layout_c, MutAnyOrigin](c_dev.unsafe_ptr())

        comptime kernel = matmul_bf16_padded_kernel[
            layout_a, layout_b, layout_c,
            BM, BN, BK, BK_PAD, BN_PAD, WM, WN, MMA_M, MMA_N, MMA_K,
        ]

        # ----- Single launch + SASS dump -----
        ctx.enqueue_function[kernel, kernel, _dump_sass=True](
            A_lt, B_lt, C_lt,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.synchronize()

        # ----- Copy back -----
        ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
        ctx.synchronize()

        # ----- Numerical correctness: full CPU reference at M=N=K=64. -----
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_i = -1
        var fail_j = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0
        for i in range(M):
            for j in range(N):
                var expected: Float32 = 0.0
                for kk in range(K):
                    expected += a_host[i * K + kk].cast[DType.float32]() * b_host[kk * N + j].cast[DType.float32]()
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
                if abs_err > 1e-2 + 1e-3 * ref_abs and fail_i < 0:
                    fail_i = i
                    fail_j = j
                    fail_got = got
                    fail_ref = expected

        print("[mojo-matmul-bf16-padded] M=N=K=", M,
              " BM=", BM, " BN=", BN, " BK=", BK,
              " BK_PAD=", BK_PAD, " BN_PAD=", BN_PAD,
              " a_type=bf16 c_type=f32",
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K)
        print("[mojo-matmul-bf16-padded] correctness: max_abs_err=", max_err,
              " max_rel_err=", max_rel_err)
        if fail_i >= 0:
            print("[mojo-matmul-bf16-padded] FAIL at (", fail_i, ",", fail_j,
                  "): got=", fail_got, " ref=", fail_ref)
        else:
            print("[mojo-matmul-bf16-padded] correctness PASSED at M=N=K=", M)
