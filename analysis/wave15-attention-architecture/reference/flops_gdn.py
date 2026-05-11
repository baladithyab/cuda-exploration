"""Wave 16 — GDN decode FLOPS and bytes model.

Single-timestep decode of Gated DeltaNet. Per MECHANISMS.md §5:

    S_t = alpha_t · (I - beta_t · k_t k_t^T) · S_{t-1}  +  beta_t · k_t v_t^T
    o_t = S_t^T · q_t

Expanding the update into per-head matmul-shaped ops (size d_k × d_v state):

  (a) scale state by scalar gate:       S *= alpha           d_k·d_v muls
  (b) compute k_t · S (row vector):     u = k_t^T · S        2·d_k·d_v    (inner product across d_k, d_v outputs)
  (c) delta:     (v - beta·u) · k_t^T   (outer product)     2·d_k·d_v    (one mul + one add per output elem) + d_v scale
  (d) output:    o = S^T · q_t                              2·d_k·d_v    (after update)

Dominant cost: ~6 · d_k · d_v FLOPS per (batch, head) for the matmul-shaped
ops (three d_k×d_v passes: b, c, d, each ≈ 2·d_k·d_v).

Per-token / per-head: 6 · d_k · d_v FLOPS (approx; we ignore O(d) elementwise
scales, exp-gate, etc. — they do not affect the memory-bound headline).

Total per kernel invocation:
    gdn_decode_flops = 6 · batch · n_heads · d_k · d_v

Bytes moved across HBM for one decode step (the dominant cost; GDN decode
at batch-1 is memory-bound because the whole state round-trips):

  - Read  S_in:  d_k · d_v · 4 bytes (f32)                per (batch, head)
  - Write S_out: d_k · d_v · 4 bytes (f32)                per (batch, head)
  - Read  q, k:  2 · d_k    · 2 bytes (f16)               per (batch, head)
  - Read  v, a, g (alpha, gate): (d_v + 2) · 2 bytes      per (batch, head)
  - Write o:     d_v        · 2 bytes (f16)               per (batch, head)

State traffic dominates. For d_k = d_v = 256 and n_heads = 16 at batch-1:
state_bytes = 2 · 256·256·4·16 = 2 MB moved per token.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shapes_gdn import GDNShape  # noqa: E402


def gdn_decode_flops(shape: GDNShape) -> int:
    """FLOPS for one timestep of GDN decode.

    Three ~d_k × d_v matmul-shaped passes per (batch, head):
      (1) k · S    → u          2·d_k·d_v
      (2) (v - beta·u) ⊗ k → ΔS 2·d_k·d_v
      (3) S · q    → o          2·d_k·d_v
    = 6 · d_k · d_v per (batch, head).

    Returns: int FLOPS count for the whole invocation.
    """
    return 6 * shape.batch * shape.n_heads * shape.d_k * shape.d_v


def gdn_decode_bytes(shape: GDNShape, state_dtype_bytes: int = 4,
                     io_dtype_bytes: int = 2) -> int:
    """Bytes moved across HBM for one decode timestep.

    State (S_in read + S_out write) is the dominant term; I/O tensors
    (q, k, v, alpha, gate, o) are negligible at d=256.
    """
    bh = shape.batch * shape.n_heads
    state = 2 * shape.d_k * shape.d_v * state_dtype_bytes  # read + write
    # q, k at d_k f16; v, o at d_v f16; alpha, gate scalars (2 f16 values).
    io = (2 * shape.d_k + 2 * shape.d_v + 2) * io_dtype_bytes
    return bh * (state + io)


def gdn_decode_arith_intensity(shape: GDNShape) -> float:
    """FLOPS per byte for one GDN decode step.

    On RTX 5090 (roofline ridge at f16 ≈ 62 flops/byte, at f32 ≈ 38),
    GDN decode sits FAR below this — classic memory-bound kernel.
    """
    return gdn_decode_flops(shape) / gdn_decode_bytes(shape)


def gdn_decode_gbps(shape: GDNShape, gpu_ms: float,
                    state_dtype_bytes: int = 4, io_dtype_bytes: int = 2) -> float:
    """Effective HBM bandwidth (GB/s) for a measured gpu_ms.

    Headline metric for GDN decode: since the kernel is BW-bound,
    report GB/s relative to the 5090's peak (~1.79 TB/s = 1790 GB/s).
    """
    bytes_moved = gdn_decode_bytes(shape, state_dtype_bytes, io_dtype_bytes)
    return bytes_moved / (gpu_ms * 1e-3) / 1e9


def gdn_decode_tflops(shape: GDNShape, gpu_ms: float) -> float:
    return gdn_decode_flops(shape) / (gpu_ms * 1e-3) / 1e12


if __name__ == "__main__":
    from shapes_gdn import SHAPE_CORRECTNESS, SHAPE_QWEN3_NEXT_DECODE

    print(f"{'shape':<24} {'GFLOPS':>8} {'KB moved':>10} {'AI (f/b)':>10}")
    for s in [SHAPE_CORRECTNESS, SHAPE_QWEN3_NEXT_DECODE]:
        flops = gdn_decode_flops(s)
        bytes_ = gdn_decode_bytes(s)
        ai = gdn_decode_arith_intensity(s)
        print(
            f"{s.name:<24} {flops / 1e9:>8.3f} {bytes_ / 1024:>10.1f} {ai:>10.2f}"
        )
    print()
    print("RTX 5090 ridge (f16 FMA peak / HBM BW): ~62 flops/byte")
    print("  GDN decode AI ≈ 0.75  → strongly memory-bound")
