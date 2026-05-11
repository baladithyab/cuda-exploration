"""Wave 16.3 — FLOPS model for MLA (Multi-Head Latent Attention).

Counts only the attention-core matmul work (the fused kernel's job), not the
up-projection matmuls that produce Q / K / V from the latent c_KV. Those are
separate GEMMs upstream of the attention kernel and are benchmarked via
cuBLAS / cuTile matmul cells, not here.

The two matmuls inside the attention kernel are:

  QK^T : per head,  (seq × qk_head_dim) × (qk_head_dim × seq)  → (seq × seq)
         = 2 · seq² · qk_head_dim FLOPS per head
  PV   : per head,  (seq × seq) × (seq × d_v)                  → (seq × d_v)
         = 2 · seq² · d_v       FLOPS per head

Multiply by n_h query heads and batch. Softmax work is negligible vs matmul.

Note on qk_head_dim vs MVP padding: the `mla_attention_flops` function uses
the **true** qk_head_dim = d_h + d_rope. The cuTile MVP pads this to 256 in
the kernel for power-of-two alignment — that padding is extra waste-compute
the kernel does, *not* real work. TFLOPS therefore reports "effective" work
done, which is what lets the number be compared to FlashMLA / cuBLAS peaks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shapes_mla import MLAShape  # noqa: E402


def mla_attention_flops(shape: MLAShape) -> int:
    """Forward-pass FLOPS for the fused MLA attention core.

    FLOPS = 2 · batch · n_h · seq² · (qk_head_dim + d_v)
          = 2 · batch · n_h · seq² · (d_h + d_rope + d_v)
    """
    return (
        2
        * shape.batch
        * shape.n_h
        * shape.seq
        * shape.seq
        * (shape.qk_head_dim + shape.d_v)
    )


def mla_attention_tflops(shape: MLAShape, gpu_ms: float) -> float:
    return mla_attention_flops(shape) / (gpu_ms * 1e-3) / 1e12


def mla_attention_bytes(shape: MLAShape, dtype_bytes: int = 2) -> int:
    """HBM bytes for one forward pass at given dtype (pedagogical MVP layout).

    Reads the full Q, K, V tensors (not the latent) because the MVP kernel
    does not do weight absorption. This is an *upper* bound on traffic — a
    real MLA prefill would stream only `c_KV` + `k_rope` (~576 floats/token)
    and up-project inside the kernel, cutting KV traffic by ~50×.
    """
    q = shape.batch * shape.seq * shape.n_h * shape.qk_head_dim
    k = shape.batch * shape.seq * shape.n_h * shape.qk_head_dim
    v = shape.batch * shape.seq * shape.n_h * shape.d_v
    o = shape.batch * shape.seq * shape.n_h * shape.d_v
    return (q + k + v + o) * dtype_bytes


if __name__ == "__main__":
    from shapes_mla import SHAPE_CORRECTNESS_MLA, SHAPE_BENCH_MLA

    for s in [SHAPE_CORRECTNESS_MLA, SHAPE_BENCH_MLA]:
        flops = mla_attention_flops(s)
        bytes_f16 = mla_attention_bytes(s, dtype_bytes=2)
        ai = flops / bytes_f16
        print(
            f"{s.name:<18} flops={flops / 1e9:8.2f} GFLOPS "
            f"bytes(f16)={bytes_f16 / 1e6:7.1f} MB "
            f"AI={ai:7.1f} flops/byte"
        )
