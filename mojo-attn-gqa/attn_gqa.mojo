# Wave C1.5 -- mojo-attn-gqa: 3-kernel Grouped-Query Attention (GQA) with bf16
# matmul stages, at the same shape as cuda-attn-gqa / cutile-attn-gqa for fair
# cross-frontend comparison.
#
# Pipeline (mirrors cuda-attn-gqa):
#   Stage 1 (qkt_kernel):     scores = Q @ K^T              [B, Nq,  S, S]   f32
#   Stage 2 (softmax_kernel): probs  = softmax(scores * 1/sqrt(qk))          bf16
#   Stage 3 (pv_kernel):      out    = probs @ V            [B, Nq,  S, d_v] f32
#
# GQA broadcast: Q has Nq query heads; K/V have Nkv=Nq/groups heads. For Q-head
# h_q, the kv head is h_kv = h_q / groups. We never expand K/V in memory; the
# kernels do the divide inline when reading K (qkt) or V (pv).
#
# Borrowed from W22.5b mojo-attn-bf16: 3-kernel decomposition + Wave 21 BF16
# matmul tile pattern (TensorCore[bf16,bf16] load + raw mma with f32 accum +
# m16n8 epilogue). Bench scaffolding (cudaEvent timing via ctx.execution_time,
# 1024-sample CPU reference, atol=1e-2 + rtol=1e-3 tolerance).
#
# Bench shape (matches cuda-attn-gqa SHAPE_BENCH = "llama3_8b"):
#   B=1, Nq=32, Nkv=8, groups=4, S=2048, qk=d_h=128, d_v=128.
#
# FLOPS = 4 * B * Nq * S^2 * D (matches cuda-attn-gqa flops/iter convention).
#
# Tile shape: BM=BN=64, BK=32, WM=WN=32, MMA=16x8x16. 4 warps/block, 128 threads.
#
# Shape-divisibility check (the orchestrator flagged qk=192 vs BK=32 as risky):
#   192 / BK=32 = 6  ✓   (qkt K-iters)
#   2048 / BM=64 = 32 ✓  (S → BM tile rows, both qkt and pv)
#   2048 / BN=64 = 32 ✓  (S → BN tile cols on qkt)
#   2048 / BK=32 = 64 ✓  (pv K-iters along S)
#   128 / BN=64  = 2  ✓  (DV → BN tile cols on pv)
# BK=32 divides everything cleanly; no tail-handler needed. Keep Wave 21 BK=32
# tile shape unchanged.
#
# Pitfall vs Wave 21 matmul: Q@K^T needs K loaded with row/col swapped so that
# MMA's K-dim (= matmul-K-dim = qk_head_dim) maps to the inner-row index of K.
# We do this with a strided thread-cooperative load (NOT copy_dram_to_sram_async,
# which is row-major-only). That costs us the cp.async path on the qkt kernel —
# expected: HMMA SASS still emits, just without UTMALDG.

from std.math import ceildiv, sqrt, exp
from std.sys import has_accelerator
from std.gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
    warp_id,
    lane_id,
    block_dim,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.compute.mma import mma
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from layout.tensor_core import TensorCore
from std.utils.index import Index


# ============================================================================
# Stage 1: Q @ K^T kernel.  scores = Q @ K^T.  GQA-aware.
# Per (b, h_q) Q-head: Q[S, qk] @ K_h_kv[S, qk] -> scores[S, S]
# where h_kv = h_q / GROUPS.  scores[i, j] = sum_k Q[i, k] * K[j, k]
#
# block_idx.z = bh = b * Nq + h_q   (indexes into Q + Sm, dim 0 = B*Nq*S/heads)
# K-side row offset uses bh_kv = bh / GROUPS:
#   head_row_off_K = bh_kv * S
# Q-side row offset uses bh:
#   head_row_off_Q = bh * S
# ============================================================================
def qkt_kernel[
    layout_q: Layout,           # [B*Nq  * S, qk]
    layout_k: Layout,           # [B*Nkv * S, qk]
    layout_s: Layout,           # [B*Nq  * S, S]
    BM: Int, BN: Int, BK: Int,
    WM: Int, WN: Int,
    MMA_M: Int, MMA_N: Int, MMA_K: Int,
    S: Int,
    QK: Int,
    GROUPS: Int,                # = Nq / Nkv. h_kv = h_q / GROUPS.
](
    Q: LayoutTensor[DType.bfloat16, layout_q, MutAnyOrigin],
    K: LayoutTensor[DType.bfloat16, layout_k, MutAnyOrigin],
    Sm: LayoutTensor[DType.float32, layout_s, MutAnyOrigin],
):
    var bh    = Int(block_idx.z)        # global Q-head index = b * Nq + h_q
    var bh_kv = bh // GROUPS            # global KV-head index for this Q-head
    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    var loader = TensorCore[DType.bfloat16, DType.bfloat16, Index(MMA_M, MMA_N, MMA_K)]()

    # Per-head row offset (in BM-units): Q + Sm use bh; K uses bh_kv (GQA broadcast).
    # block_idx.y indexes BM-rows within that head; (bh*S/BM + block_idx.y) is the
    # global BM-tile index. Requires S % BM == 0 (S=2048, BM=64 → 32 ✓).
    comptime bm_per_head = S // BM
    var head_bm_off    = bh    * bm_per_head     # for Q + Sm
    var head_bm_off_kv = bh_kv * bm_per_head     # for K (bh_kv = bh / GROUPS)

    var S_warp_tile = Sm.tile[BM, BN](
        head_bm_off + Int(block_idx.y), Int(block_idx.x)
    ).tile[WM, WN](warp_y, warp_x)

    var A_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

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

    var tid = Int(thread_idx.x)
    var nthreads = Int(block_dim.x)

    var num_k_iters = QK // BK
    for k_i in range(num_k_iters):
        barrier()

        # ---- Load A tile = Q[head_bm_off+block_y, k_i] (BM x BK), row-major. ----
        var Q_dram_tile = Q.tile[BM, BK](
            head_bm_off + Int(block_idx.y), k_i
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_smem.vectorize[1, 4](), Q_dram_tile.vectorize[1, 4]()
        )

        # ---- Load B tile = K^T view (GQA: K-side uses bh_kv). ----
        # B_smem[k_inner, n_inner] = K[head_row_off_K + (block_x*BN + n_inner), k_i*BK + k_inner]
        # head_row_off_K = bh_kv * S = head_bm_off_kv * BM (rows-of-K within global K).
        # Thread-cooperative element gather: BK*BN = 32*64 = 2048 elems, 128 threads -> 16/thread.
        var head_row_off_K = head_bm_off_kv * BM   # = bh_kv * S
        var k_block_off_row = Int(block_idx.x) * BN
        var k_block_off_col = k_i * BK
        var n_per_thr = (BK * BN + nthreads - 1) // nthreads
        for it in range(n_per_thr):
            var lin = tid + it * nthreads
            if lin < BK * BN:
                var k_inner = lin // BN
                var n_inner = lin % BN
                B_smem[k_inner, n_inner] = K[
                    head_row_off_K + k_block_off_row + n_inner,
                    k_block_off_col + k_inner,
                ]

        async_copy_wait_all()
        barrier()

        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

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

    # ---- Epilogue: m16n8 distribution -> S_warp_tile (f32). ----
    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
            var S_mma_tile = S_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            comptime for i in range(4):
                var row = group_id + (i >> 1) * 8
                var col = (tid_in_grp << 1) + (i & 1)
                S_mma_tile[row, col] = c_reg_tile[0, i]


# ============================================================================
# Stage 2: row-wise softmax with scale.  P[bh*S + i, j] = softmax_j( Sm[bh*S+i, j] * scale ).
# One block per (bh, i) row (flattened to grid_dim.x = BH*S). SOFTMAX_TPB
# threads cooperate on max -> sum -> normalize. Output is bf16.
# ============================================================================
comptime SOFTMAX_TPB = 128

def softmax_kernel[
    layout_s: Layout,     # [BH*S, S]
    layout_p: Layout,     # [BH*S, S]
    S: Int,
](
    Sm: LayoutTensor[DType.float32, layout_s, MutAnyOrigin],
    P:  LayoutTensor[DType.bfloat16, layout_p, MutAnyOrigin],
    scale: Float32,
):
    # Flatten (bh, row) into a single 1-D grid coordinate.
    var global_row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var sbuf = LayoutTensor[
        DType.float32,
        Layout.row_major(SOFTMAX_TPB),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Pass 1: max across the row.
    var lmax: Float32 = -3.4e38
    var j = tid
    while j < S:
        var v = Sm[global_row, j][0] * scale
        if v > lmax:
            lmax = v
        j += SOFTMAX_TPB
    sbuf[tid] = lmax
    barrier()
    var stride = SOFTMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            var a = sbuf[tid][0]
            var b = sbuf[tid + stride][0]
            sbuf[tid] = a if a > b else b
        barrier()
        stride = stride // 2
    var rmax = sbuf[0][0]

    # Pass 2: sum of exp.
    var lsum: Float32 = 0.0
    j = tid
    while j < S:
        var v = Sm[global_row, j][0] * scale
        lsum += exp(v - rmax)
        j += SOFTMAX_TPB
    sbuf[tid] = lsum
    barrier()
    stride = SOFTMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            sbuf[tid] = sbuf[tid][0] + sbuf[tid + stride][0]
        barrier()
        stride = stride // 2
    var rsum = sbuf[0][0]
    var inv_sum: Float32 = 1.0 / rsum

    # Pass 3: write bf16 normalized probs.
    j = tid
    while j < S:
        var v = Sm[global_row, j][0] * scale
        var p = exp(v - rmax) * inv_sum
        P[global_row, j] = p.cast[DType.bfloat16]()
        j += SOFTMAX_TPB


# ============================================================================
# Stage 3: P @ V kernel.  out = probs @ V.  GQA-aware.
# Per (b, h_q) Q-head: P[S, S] @ V_h_kv[S, d_v] -> O[S, d_v]
# where h_kv = h_q / GROUPS.  Standard row-major matmul (no transpose).
# ============================================================================
def pv_kernel[
    layout_p: Layout,           # [B*Nq  * S, S]
    layout_v: Layout,           # [B*Nkv * S, d_v]
    layout_o: Layout,           # [B*Nq  * S, d_v]
    BM: Int, BN: Int, BK: Int,
    WM: Int, WN: Int,
    MMA_M: Int, MMA_N: Int, MMA_K: Int,
    S: Int,
    DV: Int,
    GROUPS: Int,                # = Nq / Nkv. h_kv = h_q / GROUPS.
](
    P: LayoutTensor[DType.bfloat16, layout_p, MutAnyOrigin],
    V: LayoutTensor[DType.bfloat16, layout_v, MutAnyOrigin],
    O: LayoutTensor[DType.float32, layout_o, MutAnyOrigin],
):
    var bh    = Int(block_idx.z)        # global Q-head index = b * Nq + h_q
    var bh_kv = bh // GROUPS            # global KV-head index for this Q-head
    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    var loader = TensorCore[DType.bfloat16, DType.bfloat16, Index(MMA_M, MMA_N, MMA_K)]()

    # Per-head row offset (in BM-units): P + O use bh; V uses bh_kv (GQA broadcast).
    comptime bm_per_head = S // BM
    var head_bm_off    = bh    * bm_per_head     # for P + O
    var head_bm_off_kv = bh_kv * bm_per_head     # for V (bh_kv = bh / GROUPS)

    var O_warp_tile = O.tile[BM, BN](
        head_bm_off + Int(block_idx.y), Int(block_idx.x)
    ).tile[WM, WN](warp_y, warp_x)

    var A_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

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

    # Number of K-tile iterations across S = K-dim of P @ V.
    # P is [B*Nq*S, S]; P-head occupies rows [bh*S, (bh+1)*S). We tile across
    # the second dim (length S) in BK-sized slabs.
    comptime num_k_iters = S // BK
    # bk_per_head_row = S // BK = K-iters per row of P.
    # Within P, each Q-head's row k-slab at iter k_i sits at column k_i*BK.
    # Within V, each KV-head's k-slab at iter k_i sits at row head_row_off_V + k_i*BK.
    # GQA: V is indexed by bh_kv (= bh / GROUPS), NOT bh.
    var head_row_off_V = head_bm_off_kv * BM   # = bh_kv * S, the row at which V's KV-head starts.

    for k_i in range(num_k_iters):
        barrier()

        # P tile: rows [head_bm_off+block_y]*BM, cols [k_i*BK, (k_i+1)*BK)
        # P is shape (BH*S, S). tile[BM, BK]((head_bm_off+block_y), k_i) -> (BM, BK).
        var P_dram_tile = P.tile[BM, BK](
            head_bm_off + Int(block_idx.y), k_i
        )
        # V tile: rows [bh*S + k_i*BK, ...], cols [block_x*BN, ...].
        # V is shape (BH*S, DV). To get the right row block we need
        # tile[BK, BN](row_idx_in_BK_units, block_x).
        # Row offset (in BK-units): head_row_off_V/BK + k_i = (bh_kv * S / BK) + k_i.
        # bk_per_head_v_row = S // BK; same as num_k_iters.
        comptime bk_per_head = S // BK
        var V_row_block_idx = bh_kv * bk_per_head + k_i
        var V_dram_tile = V.tile[BK, BN](V_row_block_idx, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_smem.vectorize[1, 4](), P_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            B_smem.vectorize[1, 4](), V_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()
        barrier()

        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

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

    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
            var O_mma_tile = O_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            comptime for i in range(4):
                var row = group_id + (i >> 1) * 8
                var col = (tid_in_grp << 1) + (i & 1)
                O_mma_tile[row, col] = c_reg_tile[0, i]


# ============================================================================
# Main: end-to-end correctness + timed bench at GQA llama3_8b shape.
# ============================================================================
def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ---- Tile shape (Wave 21 pattern, kept unchanged for W22.5b) ----
        comptime BM = 64
        comptime BN = 64
        comptime BK = 32
        comptime WM = 32
        comptime WN = 32
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 16
        comptime NUM_WARPS = (BM // WM) * (BN // WN)
        comptime BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 128

        # ---- Problem shape: GQA llama3_8b decode (matches cuda-attn-gqa). ----
        # B=1, Nq=32, Nkv=8 (groups=4), S=2048, qk=128, d_v=128.
        # Shape divisibility:
        #   QK=128 / BK=32 = 4 ✓   (qkt K-iters)
        #   S=2048 / BM=64 = 32 ✓  (S → BM tile rows)
        #   S=2048 / BN=64 = 32 ✓  (S → BN tile cols on qkt)
        #   S=2048 / BK=32 = 64 ✓  (pv K-iters along S)
        #   DV=128 / BN=64 = 2 ✓   (DV → BN tile cols on pv)
        #   NQ=32 % NKV=8 == 0 ✓  (groups=4 evenly divides)
        # Memory budget on RTX 5090 (32 GiB):
        #   Q  bf16:  B*NQ *S*QK*2 = 1*32*2048*128*2 =  16 MiB
        #   K  bf16:  B*NKV*S*QK*2 = 1*8 *2048*128*2 =   4 MiB
        #   V  bf16:  B*NKV*S*DV*2 = 1*8 *2048*128*2 =   4 MiB
        #   S  f32 :  B*NQ *S*S *4 = 1*32*2048*2048*4 = 512 MiB
        #   P  bf16:  B*NQ *S*S *2 = 1*32*2048*2048*2 = 256 MiB
        #   O  f32 :  B*NQ *S*DV*4 = 1*32*2048*128*4 = 32 MiB
        # Total ≈ 824 MiB device. Well under 32 GiB.
        comptime B = 1
        comptime NQ = 32          # Q heads
        comptime NKV = 8          # KV heads
        comptime GROUPS = NQ // NKV   # 4
        comptime S = 2048
        comptime QK = 128         # head_dim (same for Q and K)
        comptime DV = 128         # head_dim for V
        comptime BHQ = B * NQ     # global Q-head count (= block_idx.z range)
        comptime BHKV = B * NKV   # global KV-head count
        comptime QK_ELEMS_Q  = BHQ  * S * QK
        comptime QK_ELEMS_KV = BHKV * S * QK
        comptime V_ELEMS     = BHKV * S * DV
        comptime O_ELEMS     = BHQ  * S * DV
        comptime SCORE_ELEMS = BHQ  * S * S

        # Flat 2-D layouts: heads concatenated along the row dim.
        # Q + Sm + P + O are indexed by Q-head bh ∈ [0, BHQ).
        # K + V are indexed by KV-head bh_kv ∈ [0, BHKV) = [0, BHQ/GROUPS).
        comptime layout_q = Layout.row_major(BHQ  * S, QK)
        comptime layout_k = Layout.row_major(BHKV * S, QK)
        comptime layout_v = Layout.row_major(BHKV * S, DV)
        comptime layout_s = Layout.row_major(BHQ  * S, S)
        comptime layout_p = Layout.row_major(BHQ  * S, S)
        comptime layout_o = Layout.row_major(BHQ  * S, DV)

        # ---- Buffers ----
        var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](QK_ELEMS_Q)
        var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](QK_ELEMS_KV)
        var v_dev = ctx.enqueue_create_buffer[DType.bfloat16](V_ELEMS)
        var s_dev = ctx.enqueue_create_buffer[DType.float32](SCORE_ELEMS)
        var p_dev = ctx.enqueue_create_buffer[DType.bfloat16](SCORE_ELEMS)
        var o_dev = ctx.enqueue_create_buffer[DType.float32](O_ELEMS)

        var q_host = ctx.enqueue_create_host_buffer[DType.bfloat16](QK_ELEMS_Q)
        var k_host = ctx.enqueue_create_host_buffer[DType.bfloat16](QK_ELEMS_KV)
        var v_host = ctx.enqueue_create_host_buffer[DType.bfloat16](V_ELEMS)
        var o_host = ctx.enqueue_create_host_buffer[DType.float32](O_ELEMS)
        ctx.synchronize()

        # ---- Init Q, K, V with deterministic small-magnitude bf16 values. ----
        # Small magnitudes keep softmax numerically stable.
        for i in range(QK_ELEMS_Q):
            q_host[i] = (Float32(((i * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.bfloat16]()
        for i in range(QK_ELEMS_KV):
            k_host[i] = (Float32((((i + 17) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.bfloat16]()
        for i in range(V_ELEMS):
            v_host[i] = (Float32((((i + 31) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.bfloat16]()
        ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
        ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
        ctx.enqueue_copy(dst_buf=v_dev, src_buf=v_host)
        ctx.synchronize()

        var Q_lt = LayoutTensor[DType.bfloat16, layout_q, MutAnyOrigin](q_dev.unsafe_ptr())
        var K_lt = LayoutTensor[DType.bfloat16, layout_k, MutAnyOrigin](k_dev.unsafe_ptr())
        var V_lt = LayoutTensor[DType.bfloat16, layout_v, MutAnyOrigin](v_dev.unsafe_ptr())
        var S_lt = LayoutTensor[DType.float32,  layout_s, MutAnyOrigin](s_dev.unsafe_ptr())
        var P_lt = LayoutTensor[DType.bfloat16, layout_p, MutAnyOrigin](p_dev.unsafe_ptr())
        var O_lt = LayoutTensor[DType.float32,  layout_o, MutAnyOrigin](o_dev.unsafe_ptr())

        # ---- Kernel handles ----
        comptime kernel_qkt = qkt_kernel[
            layout_q, layout_k, layout_s,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
            S, QK, GROUPS,
        ]
        comptime kernel_sm = softmax_kernel[layout_s, layout_p, S]
        comptime kernel_pv = pv_kernel[
            layout_p, layout_v, layout_o,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
            S, DV, GROUPS,
        ]
        var scale: Float32 = 1.0 / sqrt(Float32(QK))

        # ---- Warmup launch (also captures SASS for the matmul kernels) ----
        # grid_dim.z = BHQ (Q-head count). Each thread block is a Q-head, but
        # reads the matching KV head via bh_kv = bh / GROUPS inside the kernel.
        ctx.enqueue_function[kernel_qkt, kernel_qkt, _dump_sass=True](
            Q_lt, K_lt, S_lt,
            grid_dim=(ceildiv(S, BN), ceildiv(S, BM), BHQ),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.enqueue_function[kernel_sm, kernel_sm](
            S_lt, P_lt, scale,
            grid_dim=(BHQ * S,),
            block_dim=(SOFTMAX_TPB,),
        )
        ctx.enqueue_function[kernel_pv, kernel_pv, _dump_sass=True](
            P_lt, V_lt, O_lt,
            grid_dim=(ceildiv(DV, BN), ceildiv(S, BM), BHQ),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.synchronize()

        # ---- Timed run: 10 per-iter ctx.execution_time samples -> median. ----
        # @parameter def body re-enqueues all 3 kernels per iter; cudaEvent
        # bracket spans the full pipeline (qkt + softmax + pv) on the stream.
        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[kernel_qkt, kernel_qkt](
                Q_lt, K_lt, S_lt,
                grid_dim=(ceildiv(S, BN), ceildiv(S, BM), BHQ),
                block_dim=(BLOCK_THREADS,),
            )
            ctx.enqueue_function[kernel_sm, kernel_sm](
                S_lt, P_lt, scale,
                grid_dim=(BHQ * S,),
                block_dim=(SOFTMAX_TPB,),
            )
            ctx.enqueue_function[kernel_pv, kernel_pv](
                P_lt, V_lt, O_lt,
                grid_dim=(ceildiv(DV, BN), ceildiv(S, BM), BHQ),
                block_dim=(BLOCK_THREADS,),
            )

        var num_iters = 10
        var iter_ms = SIMD[DType.float64, 16](0.0)
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6  # ns -> ms
        ctx.synchronize()

        # Insertion sort iter_ms[0:num_iters] for median.
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

        # ---- Copy back final iteration's output for correctness check. ----
        ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
        ctx.synchronize()

        # =====================================================================
        # CPU reference: GQA SDPA (no mask). Sample-based correctness check.
        # 1024 samples, tolerance atol=1e-2 + rtol=1e-3*|ref| (Wave 21 spec).
        # GQA: Q-head h_q reads K and V from h_kv = h_q // GROUPS.
        # Per sample: full row Q@K^T (S*QK ≈ 262K mult-adds) + softmax (3*S) +
        #             reduce-with-V (S mult-adds). 1024 samples ≈ 270M ops in
        #             interpreted Mojo — runs in a few seconds.
        # =====================================================================
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_bh = -1
        var fail_i  = -1
        var fail_d  = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0

        var num_samples = 1024
        for s_idx in range(num_samples):
            # Knuth golden-ratio hash. Use disjoint high-bit windows for bh, i, d.
            var seed = s_idx * 2654435761
            var bh    = (((seed >> 24) % BHQ) + BHQ) % BHQ   # Q-head
            var bh_kv = bh // GROUPS                          # KV-head (GQA broadcast)
            var i     = (((seed >> 11) % S)  + S)  % S
            var d     = (((seed >> 4)  % DV) + DV) % DV

            # Compute full row scores[j] = Q[bh, i, :] . K[bh_kv, j, :] * scale.
            var row_scores = InlineArray[Float32, S](fill=0.0)
            for j in range(S):
                var sj: Float32 = 0.0
                for kk in range(QK):
                    var qv = q_host[bh    * S * QK + i * QK + kk].cast[DType.float32]()
                    var kv = k_host[bh_kv * S * QK + j * QK + kk].cast[DType.float32]()
                    sj += qv * kv
                row_scores[j] = sj * scale

            # Softmax over j.
            var rmax: Float32 = -3.4e38
            for j in range(S):
                if row_scores[j] > rmax:
                    rmax = row_scores[j]
            var rsum: Float32 = 0.0
            var probs = InlineArray[Float32, S](fill=0.0)
            for j in range(S):
                var e = exp(row_scores[j] - rmax)
                probs[j] = e
                rsum += e
            var inv_sum: Float32 = 1.0 / rsum
            for j in range(S):
                probs[j] = probs[j] * inv_sum

            # Cast probs through bf16 (matches kernel's intermediate dtype) and
            # reduce against V[bh_kv, :, d].
            var refv: Float32 = 0.0
            for j in range(S):
                var pj = probs[j].cast[DType.bfloat16]().cast[DType.float32]()
                var vd = v_host[bh_kv * S * DV + j * DV + d].cast[DType.float32]()
                refv += pj * vd

            var got = o_host[bh * S * DV + i * DV + d]
            var abs_err = abs(got - refv)
            var ref_abs = abs(refv)
            var rel_err: Float32 = 0.0
            if ref_abs > 0.0:
                rel_err = abs_err / ref_abs
            if abs_err > max_err:
                max_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            # Tolerance: atol=1e-2 + rtol=1e-3*|ref| (Wave 21 / Phase-7 spec).
            if abs_err > 1e-2 + 1e-3 * ref_abs and fail_bh < 0:
                fail_bh = bh
                fail_i = i
                fail_d = d
                fail_got = got
                fail_ref = refv

        # ---- Report. ----
        # FLOPs per iter (matches cuda-attn-gqa flops/iter convention):
        #   4 * B * Nq * S^2 * D
        # = 4 * 1 * 32 * 2048^2 * 128
        # = 68,719,476,736 = 68.72 GFLOPS
        var flops_per_iter: Float64 = (
            4.0 * Float64(B) * Float64(NQ)
            * Float64(S) * Float64(S) * Float64(DV)
        )
        var median_s: Float64 = median_ms * 1e-3
        var min_s: Float64 = min_ms * 1e-3
        var tflops_median: Float64 = flops_per_iter / median_s / 1e12
        var tflops_best: Float64 = flops_per_iter / min_s / 1e12

        print("[mojo-attn-gqa] shape: B=", B, " Nq=", NQ, " Nkv=", NKV,
              " groups=", GROUPS, " S=", S, " qk=", QK, " d_v=", DV)
        print("[mojo-attn-gqa] tile: BM=", BM, " BN=", BN, " BK=", BK,
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K)
        print("[mojo-attn-gqa] flops_per_iter=", flops_per_iter / 1e9, " GFLOPS")
        print("[mojo-attn-gqa] timing: min_ms=", min_ms,
              " median_ms=", median_ms,
              " max_ms=", max_ms_iter)
        print("[mojo-attn-gqa] TFLOPS_median=", tflops_median,
              " TFLOPS_best=", tflops_best)
        print("[mojo-attn-gqa] correctness: max_abs_err=", max_err,
              " max_rel_err=", max_rel_err, " (vs CPU SDPA ref, ", num_samples, " samples)")
        if fail_bh >= 0:
            print("[mojo-attn-gqa] FAIL at (bh=", fail_bh, " i=", fail_i, " d=", fail_d,
                  "): got=", fail_got, " ref=", fail_ref)
        else:
            print("[mojo-attn-gqa] correctness PASSED at llama3_8b shape (1024 samples)")

        # Cross-frontend GQA comparison (Wave C1, llama3_8b shape).
        print("[mojo-attn-gqa] cross-frontend GQA at this shape:")
        print("[mojo-attn-gqa]   cutile-attn-gqa  = 165.00 TF (fused FlashAttention-class)")
        print("[mojo-attn-gqa]   cublas-attn-gqa  = 218.00 TF (hgemm reference)")
        print("[mojo-attn-gqa]   cuda-attn-gqa    =  23.40 TF (3-kernel WMMA, median)")
        print("[mojo-attn-gqa]   oxide-attn-gqa   =  24.00 TF (3-kernel hand-WMMA)")
        print("[mojo-attn-gqa]   wgpu-attn-gqa    =   (CPU-only)")
        print("[mojo-attn-gqa]   mojo-attn-gqa    =  ", tflops_median, " TF (3-kernel hand-MMA, this run, median)")
