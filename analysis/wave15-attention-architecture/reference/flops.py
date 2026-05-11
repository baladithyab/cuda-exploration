"""Wave 15 — FLOPS model.

Single source of truth for all FLOPS calculations across attention
mechanisms. Eliminates per-cell copy-paste errors that silently scale
TFLOPS wrong (the #1 way to ship a false headline).

Each function returns the number of fused-multiply-add equivalents
(2 FLOPS per FMA). For mechanisms with non-FMA work (softmax, exp,
gating), we count only the matmul FLOPS — the dominant contribution
on real GPUs. Document this consistently in ANALYSIS.md.

Conventions:
  - Forward pass only.
  - Causal masking does NOT halve FLOPS — modern fused kernels
    skip masked positions, but the standard practice in published
    perf tables (FlashAttention papers, etc.) is to report uncausal
    FLOPS for apples-to-apples comparison. We follow that.
  - 2N^3 per matmul (the standard) — N output × N inner-product × 2
    (one mul + one add).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from shapes import GQAShape  # noqa: E402


def gqa_attention_flops(shape: GQAShape) -> int:
    """Forward-pass FLOPS for GQA attention.

    Three matmuls dominate; softmax/exp negligible by FLOP count.

      1) QK^T:  (batch * n_q * seq * seq * d_head) × 2
                Each Q head dots against its KV-group's K. Q heads
                share K, so we still do n_q matmuls — the SAVINGS in
                GQA are MEMORY (KV cache), not compute (matmul ops).
      2) softmax: O(batch * n_q * seq^2). ~3 FLOPS each (max, exp, /sum).
                  Negligible vs matmul.
      3) PV:    (batch * n_q * seq * seq * d_head) × 2

    Total compute ≈ 2 × (2 × batch × n_q × seq² × d_head) FLOPS
                  = 4 × batch × n_q × seq² × d_head
    """
    return 4 * shape.batch * shape.n_q * shape.seq * shape.seq * shape.d_head


def gqa_attention_tflops(shape: GQAShape, gpu_ms: float) -> float:
    """Convert per-iter gpu time to TFLOPS for the given GQA shape."""
    return gqa_attention_flops(shape) / (gpu_ms * 1e-3) / 1e12


def gqa_attention_bytes(shape: GQAShape, dtype_bytes: int = 2) -> int:
    """Bytes moved across HBM for one forward pass at given dtype.

    Read Q, K, V, write O. Plus softmax intermediates if not fused
    (we ignore those — assume fused kernel).

    Useful for memory-bound regime sanity check on large shapes.
    """
    q = shape.batch * shape.seq * shape.n_q * shape.d_head
    kv = shape.batch * shape.seq * shape.n_kv * shape.d_head * 2  # K and V
    o = shape.batch * shape.seq * shape.n_q * shape.d_head
    return (q + kv + o) * dtype_bytes


if __name__ == "__main__":
    # Sanity check: the canonical bench shape should require an
    # achievable amount of compute. Llama-3 8B inference at seq=2048
    # is well-characterized publicly.
    from shapes import SHAPE_CORRECTNESS, SHAPE_BENCH

    for s in [SHAPE_CORRECTNESS, SHAPE_BENCH]:
        flops = gqa_attention_flops(s)
        bytes_f16 = gqa_attention_bytes(s, dtype_bytes=2)
        ai = flops / bytes_f16  # arithmetic intensity
        print(
            f"{s.name:<14} flops={flops / 1e9:7.2f} GFLOPS "
            f"bytes(f16)={bytes_f16 / 1e6:6.1f} MB "
            f"AI={ai:6.1f} flops/byte"
        )
