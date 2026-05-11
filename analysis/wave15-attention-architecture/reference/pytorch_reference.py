"""Wave 15 — PyTorch reference for Grouped-Query Attention (GQA).

This module is the **correctness oracle** for every GQA cell across
every frontend. cuTile, nvcc, cuBLAS-3-kernel — they all have to
match the output of `gqa_reference(...)` to within the per-dtype
tolerance defined in tolerances.py.

Implementation choice: use `torch.nn.functional.scaled_dot_product_attention`
with `enable_gqa=True`. This is PyTorch's canonical GQA, fused-impl
when possible. Plus a hand-written `gqa_naive(...)` as a witness for
the witness — if the two disagree, something is wrong with PyTorch's
GQA, not our kernels.

Run this as a script: `python pytorch_reference.py` to dump expected
output to inputs/gqa_<shape>_expected.npy alongside the Q/K/V tensors.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as a script without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from shapes import GQAShape, SHAPE_CORRECTNESS, SHAPE_BENCH, all_shapes  # noqa: E402


def gqa_reference_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """PyTorch SDPA GQA reference.

    Shapes:
        q: (batch, n_q,  seq, d_head)
        k: (batch, n_kv, seq, d_head)
        v: (batch, n_kv, seq, d_head)
        returns: (batch, n_q, seq, d_head)

    enable_gqa=True tells PyTorch to broadcast K,V over the head
    dimension (Q has more heads than KV). Equivalent to repeat_interleave(
    k, n_q // n_kv, dim=1) but without the memory copy.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False, enable_gqa=True
    )


def gqa_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Hand-written GQA, line-by-line for clarity. Used to cross-check
    PyTorch's SDPA result.

    Maximally explicit: explicit broadcast of KV from n_kv to n_q via
    repeat_interleave.
    """
    batch, n_q, seq, d_head = q.shape
    _, n_kv, _, _ = k.shape
    assert n_q % n_kv == 0
    groups = n_q // n_kv

    # Broadcast K, V from n_kv heads to n_q heads.
    k_g = k.repeat_interleave(groups, dim=1)  # (batch, n_q, seq, d_head)
    v_g = v.repeat_interleave(groups, dim=1)  # (batch, n_q, seq, d_head)

    scale = 1.0 / math.sqrt(d_head)
    # scores: (batch, n_q, seq, seq)
    scores = torch.matmul(q, k_g.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_g)  # (batch, n_q, seq, d_head)
    return out


def make_qkv(
    shape: GQAShape, dtype: torch.dtype, seed: int = 0xC0FFEE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate seeded Q, K, V tensors. Same seed → identical inputs
    every run, so cross-impl comparisons are deterministic.

    Q,K,V are scaled to keep attention scores in a reasonable range
    (avoid softmax saturation at f16/bf16).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / math.sqrt(shape.d_head)
    q = torch.randn(
        (shape.batch, shape.n_q, shape.seq, shape.d_head), generator=g, dtype=torch.float32
    ) * scale
    k = torch.randn(
        (shape.batch, shape.n_kv, shape.seq, shape.d_head), generator=g, dtype=torch.float32
    ) * scale
    v = torch.randn(
        (shape.batch, shape.n_kv, shape.seq, shape.d_head), generator=g, dtype=torch.float32
    )  # V doesn't need pre-scale; softmax-weighted average preserves magnitude
    return q.to(dtype), k.to(dtype), v.to(dtype)


def main() -> int:
    out_dir = Path(__file__).parent.parent / "inputs"
    out_dir.mkdir(exist_ok=True)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute cap: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print()

    for shape in all_shapes():
        shape.assert_valid()
        # Generate canonical f32 inputs for this shape.
        q32, k32, v32 = make_qkv(shape, torch.float32)
        # f16 view for kernels that want half precision.
        q16, k16, v16 = q32.to(torch.float16), k32.to(torch.float16), v32.to(torch.float16)

        # Compute the reference output at f32 (highest accuracy).
        if torch.cuda.is_available():
            q_dev, k_dev, v_dev = q32.cuda(), k32.cuda(), v32.cuda()
            with torch.no_grad():
                out_sdpa = gqa_reference_torch(q_dev, k_dev, v_dev).cpu()
                out_naive = gqa_naive(q32, k32, v32)  # CPU
        else:
            out_sdpa = gqa_reference_torch(q32, k32, v32)
            out_naive = gqa_naive(q32, k32, v32)

        # Cross-check SDPA vs naive on f32 — should match to ~1e-5.
        max_err = float((out_sdpa - out_naive).abs().max())
        rel_err = max_err / float(out_naive.abs().max() + 1e-30)
        ok = max_err < 1e-4
        print(
            f"[{shape.name}] SDPA vs naive  max_abs_err={max_err:.3e}  "
            f"rel_err={rel_err:.3e}  {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            return 1

        # Save canonical inputs + expected outputs.
        prefix = out_dir / f"gqa_{shape.name}"
        np.save(f"{prefix}_q_f32.npy", q32.numpy())
        np.save(f"{prefix}_k_f32.npy", k32.numpy())
        np.save(f"{prefix}_v_f32.npy", v32.numpy())
        np.save(f"{prefix}_q_f16.npy", q16.numpy())
        np.save(f"{prefix}_k_f16.npy", k16.numpy())
        np.save(f"{prefix}_v_f16.npy", v16.numpy())
        np.save(f"{prefix}_expected_f32.npy", out_naive.numpy())
        print(
            f"  wrote {prefix}_{{q,k,v}}_{{f32,f16}}.npy and _expected_f32.npy "
            f"(Q={shape.q_size * 4 / 1e6:.1f}MB f32)"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
