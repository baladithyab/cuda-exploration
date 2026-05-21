# Wave 22.4 -- mojo-matmul-fp8 probe: minimal sm_120 m16n8k32 e4m3 FP8 MMA smoke test
#
# Single-warp probe for the FP8 e4m3 MMA path on sm_120 (RTX 5090).
# m16n8k32, e4m3 inputs, f32 accumulator.
#
# Per Mojo `std.gpu.compute.mma._mma_nvidia` source + skill reference
# `references/mojo-mma-shapes.md` (Wave 21 distilled):
#   A: SIMD[float8_e4m3fn, 16]
#   B: SIMD[float8_e4m3fn, 8]
#   C: SIMD[float32, 4]
#   D: SIMD[float32, 4]
#
# Dispatch path (per skill table): the m16n8k32 e4m3 lane is implemented
# via inline PTX asm `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
# inside the dispatcher (NOT a clean LLVM intrinsic in 1.0.0b1). We try
# the high-level mma() dispatcher first (it may wire through). If that
# fails to compile, we fall back to direct LLVM intrinsic, then to raw
# `inlined_assembly`.
#
# Acceptance signal: HMMA.16832.F32.E4M3 > 0 in the SASS dump.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.gpu.compute.mma import mma


def mma_probe_fp8_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    # Build trivial fragments. All threads participate (warp-sync).
    var lane = Float32(Int(thread_idx.x))

    # A fragment: 16 e4m3 elements per lane (m16n8k32 / 32 lanes * 16/lane = 16x32 = 512).
    var a_frag = SIMD[DType.float8_e4m3fn, 16](
        (lane * 0.01).cast[DType.float8_e4m3fn](),
        (lane * 0.02).cast[DType.float8_e4m3fn](),
        (lane * 0.03).cast[DType.float8_e4m3fn](),
        (lane * 0.04).cast[DType.float8_e4m3fn](),
        (lane * 0.05).cast[DType.float8_e4m3fn](),
        (lane * 0.06).cast[DType.float8_e4m3fn](),
        (lane * 0.07).cast[DType.float8_e4m3fn](),
        (lane * 0.08).cast[DType.float8_e4m3fn](),
        (lane * 0.09).cast[DType.float8_e4m3fn](),
        (lane * 0.10).cast[DType.float8_e4m3fn](),
        (lane * 0.11).cast[DType.float8_e4m3fn](),
        (lane * 0.12).cast[DType.float8_e4m3fn](),
        (lane * 0.13).cast[DType.float8_e4m3fn](),
        (lane * 0.14).cast[DType.float8_e4m3fn](),
        (lane * 0.15).cast[DType.float8_e4m3fn](),
        (lane * 0.16).cast[DType.float8_e4m3fn](),
    )
    # B fragment: 8 e4m3 elements per lane (m16n8k32: 32*8/32 = 8/lane).
    var b_frag = SIMD[DType.float8_e4m3fn, 8](
        (lane * 0.21).cast[DType.float8_e4m3fn](),
        (lane * 0.22).cast[DType.float8_e4m3fn](),
        (lane * 0.23).cast[DType.float8_e4m3fn](),
        (lane * 0.24).cast[DType.float8_e4m3fn](),
        (lane * 0.25).cast[DType.float8_e4m3fn](),
        (lane * 0.26).cast[DType.float8_e4m3fn](),
        (lane * 0.27).cast[DType.float8_e4m3fn](),
        (lane * 0.28).cast[DType.float8_e4m3fn](),
    )
    var c_frag = SIMD[DType.float32, 4](
        lane * 0.0001,
        lane * 0.0002,
        lane * 0.0003,
        lane * 0.0004,
    )
    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

    # The MMA call itself. d = a*b + c.
    mma(d_frag, a_frag, b_frag, c_frag)

    # Write each lane's 4-element accumulator out so the compiler can't
    # delete the mma.
    out_ptr[Int(thread_idx.x) * 4 + 0] = d_frag[0]
    out_ptr[Int(thread_idx.x) * 4 + 1] = d_frag[1]
    out_ptr[Int(thread_idx.x) * 4 + 2] = d_frag[2]
    out_ptr[Int(thread_idx.x) * 4 + 3] = d_frag[3]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU: ", ctx.name())

        var out_dev = ctx.enqueue_create_buffer[DType.float32](128)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](128)
        ctx.synchronize()

        var out_ptr = out_dev.unsafe_ptr()

        # 32 threads = 1 warp.
        ctx.enqueue_function[mma_probe_fp8_kernel, mma_probe_fp8_kernel, _dump_sass=True](
            out_ptr,
            grid_dim=(1,),
            block_dim=(32,),
        )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
        ctx.synchronize()

        print("[mma-probe-fp8] First 8 of 128 output floats:")
        for i in range(8):
            print("  out[", i, "] = ", out_host[i])
        print("[mma-probe-fp8] If you see HMMA.16832.F32.E4M3 in stderr SASS dump,")
        print("           the e4m3 FP8 MMA path engaged on sm_120. Done.")
