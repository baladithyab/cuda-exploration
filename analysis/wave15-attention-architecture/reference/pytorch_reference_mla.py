"""Wave 16.3 — PyTorch reference for MLA (Multi-Head Latent Attention).

Correctness oracle for every MLA cell. Generates the "already up-projected"
Q, K, V tensors that the pedagogical MLA kernel consumes, plus the expected
attention output computed by PyTorch at f32.

Design note. The full MLA pipeline is:

    h_t → c_KV_t (latent) → (W_UK, W_UV) → per-head K, V  → attention → O

For the pedagogical kernel MVP we skip both the down- and up-projections and
feed the kernel Q, K, V *directly*. The latent-cache memory win of MLA is a
system-level property (what gets cached during decode), not something the
attention-core kernel itself implements. So the reference generates Q, K, V
with the correct MLA shapes (qk_head_dim = d_h + d_rope = 192, v_head_dim =
128, all n_h heads independent) and compares against pedagogical softmax
attention.

Two variants are cross-checked:
  - SDPA: torch.nn.functional.scaled_dot_product_attention (scale = 1/√qk_head_dim)
  - Hand-written matmul+softmax loop (witness-for-witness)

Both run at f32 on GPU if available, CPU otherwise. Cross-check must agree
to 1e-4 before expected output is saved.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shapes_mla import (  # noqa: E402
    MLAShape,
    SHAPE_CORRECTNESS_MLA,
    SHAPE_BENCH_MLA,
    all_shapes,
)


def mla_reference_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_head_dim: int
) -> torch.Tensor:
    """PyTorch SDPA reference.

    Shapes:
        q: (batch, n_h, seq, qk_head_dim)
        k: (batch, n_h, seq, qk_head_dim)
        v: (batch, n_h, seq, d_v)
        returns: (batch, n_h, seq, d_v)

    NOTE: scale here is 1/√qk_head_dim per DeepSeek-V3 (not 1/√d_h). See
    MECHANISMS.md §1 line 87: "softmax(q_t K_{≤t}^T / sqrt(qk_head_dim))".
    PyTorch's SDPA default scale is 1/√last_dim_of_q = 1/√qk_head_dim, which
    matches. Verified explicit.
    """
    scale = 1.0 / math.sqrt(qk_head_dim)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=scale
    )


def mla_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_head_dim: int
) -> torch.Tensor:
    """Hand-written MLA attention, line-by-line. Used as witness-for-witness.

    Uses qk_head_dim for the softmax scale (NOT d_h). The "nope + rope" split
    is irrelevant at the attention stage once Q and K are concatenated; only
    the scale formula cares.
    """
    scale = 1.0 / math.sqrt(qk_head_dim)
    # scores: (batch, n_h, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    # out: (batch, n_h, seq, d_v)
    out = torch.matmul(probs, v)
    return out


def make_qkv_mla(
    shape: MLAShape, dtype: torch.dtype, seed: int = 0xC0FFEE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate seeded Q, K, V for MLA.

    Scales Q and K to keep attention scores in range for f16 softmax. V is
    left unscaled — the softmax output is a convex combination so V's scale
    is preserved.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    qk = shape.qk_head_dim
    scale = 1.0 / math.sqrt(qk)
    q = torch.randn(
        (shape.batch, shape.n_h, shape.seq, qk),
        generator=g, dtype=torch.float32,
    ) * scale
    k = torch.randn(
        (shape.batch, shape.n_h, shape.seq, qk),
        generator=g, dtype=torch.float32,
    ) * scale
    v = torch.randn(
        (shape.batch, shape.n_h, shape.seq, shape.d_v),
        generator=g, dtype=torch.float32,
    )
    return q.to(dtype), k.to(dtype), v.to(dtype)


def main() -> int:
    out_dir = Path(__file__).parent.parent / "inputs"
    out_dir.mkdir(exist_ok=True)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute cap: sm_{cap[0]}{cap[1]}")
    print()

    for shape in all_shapes():
        shape.assert_valid()
        print(
            f"[{shape.name}] batch={shape.batch} seq={shape.seq} "
            f"n_h={shape.n_h} qk_head_dim={shape.qk_head_dim} d_v={shape.d_v}"
        )
        q32, k32, v32 = make_qkv_mla(shape, torch.float32)
        q16, k16, v16 = (
            q32.to(torch.float16),
            k32.to(torch.float16),
            v32.to(torch.float16),
        )

        if torch.cuda.is_available():
            q_dev, k_dev, v_dev = q32.cuda(), k32.cuda(), v32.cuda()
            with torch.no_grad():
                out_sdpa = mla_reference_sdpa(
                    q_dev, k_dev, v_dev, shape.qk_head_dim
                ).cpu()
                out_naive = mla_naive(q32, k32, v32, shape.qk_head_dim)  # CPU
        else:
            out_sdpa = mla_reference_sdpa(q32, k32, v32, shape.qk_head_dim)
            out_naive = mla_naive(q32, k32, v32, shape.qk_head_dim)

        # Cross-check SDPA vs naive at f32. Should match to ~1e-5 (tiny drift
        # is just PyTorch SDPA using a slightly different reduction order).
        max_err = float((out_sdpa - out_naive).abs().max())
        rel_err = max_err / float(out_naive.abs().max() + 1e-30)
        ok = max_err < 1e-4
        print(
            f"  SDPA vs naive  max_abs_err={max_err:.3e}  rel={rel_err:.3e}  "
            f"{'OK' if ok else 'FAIL'}"
        )
        if not ok:
            print("  ABORT: oracle disagreement", file=sys.stderr)
            return 1

        # Save inputs and expected output. Prefix `mla_` to coexist with the
        # GQA inputs in the same directory.
        prefix = out_dir / f"mla_{shape.name}"
        np.save(f"{prefix}_q_f32.npy", q32.numpy())
        np.save(f"{prefix}_k_f32.npy", k32.numpy())
        np.save(f"{prefix}_v_f32.npy", v32.numpy())
        np.save(f"{prefix}_q_f16.npy", q16.numpy())
        np.save(f"{prefix}_k_f16.npy", k16.numpy())
        np.save(f"{prefix}_v_f16.npy", v16.numpy())
        np.save(f"{prefix}_expected_f32.npy", out_naive.numpy())
        print(
            f"  wrote {prefix}_{{q,k,v}}_{{f32,f16}}.npy and _expected_f32.npy "
            f"(Q={shape.q_size * 2 / 1e6:.1f}MB f16)"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
