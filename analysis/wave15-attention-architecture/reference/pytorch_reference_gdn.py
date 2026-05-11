"""Wave 16 — PyTorch reference for Gated DeltaNet (GDN) single-timestep decode.

Correctness oracle for every GDN decode kernel. Implements the exact
recurrence from MECHANISMS.md §5:

    S_t = alpha_t · (I - beta_t · k_t · k_t^T) · S_{t-1}  +  beta_t · k_t · v_t^T
    o_t = S_t^T · q_t

Per head, per batch:
    S      (d_k, d_v)  f32 state
    q      (d_k,)      f16 input
    k      (d_k,)      f16 input  (pre-normalized upstream; see FLA)
    v      (d_v,)      f16 input
    alpha  ()          f16 scalar — the forget gate α_t ∈ (0, 1], typically exp(g)
    beta   ()          f16 scalar — the delta-rule step size β_t

Output:
    o      (d_v,)      f16
    S_out  (d_k, d_v)  f32 updated state

This is a hand-written reference (no FLA dependency) so it runs anywhere
cutile-vecadd-bench's .venv runs (torch only).  We cross-check against
an "expanded" form to guard against typos.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shapes_gdn import GDNShape, SHAPE_CORRECTNESS, SHAPE_QWEN3_NEXT_DECODE, all_shapes  # noqa: E402


def gdn_decode_reference(
    q: torch.Tensor,       # (B, H, d_k)  f16
    k: torch.Tensor,       # (B, H, d_k)  f16
    v: torch.Tensor,       # (B, H, d_v)  f16
    alpha: torch.Tensor,   # (B, H)       f16
    beta: torch.Tensor,    # (B, H)       f16
    S_in: torch.Tensor,    # (B, H, d_k, d_v) f32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference single-timestep GDN decode.

    Returns (o, S_out):
        o     : (B, H, d_v)      f16  (output token)
        S_out : (B, H, d_k, d_v) f32  (updated recurrent state)

    Matmuls are done in f32 for reference precision; inputs are upcast
    from f16 internally. This is intentional — the GPU kernel will
    mix f16 inputs with f32 accumulator and we want the reference to
    track the GPU's best achievable numerics.
    """
    qf = q.float()
    kf = k.float()
    vf = v.float()
    af = alpha.float()  # (B, H)
    bf = beta.float()   # (B, H)

    B, H, d_k = qf.shape
    _, _, d_v = vf.shape

    # (1) Scale state by scalar forget gate α_t.
    #   S_scaled[b, h] = alpha[b, h] * S_in[b, h]
    a_bcast = af.view(B, H, 1, 1)
    S_scaled = a_bcast * S_in  # (B, H, d_k, d_v)

    # (2) Compute u = k_t^T · S_scaled (reads the "current prediction" from
    #     the gated state). Shape (B, H, d_v).
    u = torch.einsum("bhk,bhkv->bhv", kf, S_scaled)  # (B, H, d_v)

    # (3) Residual (v_t - u) — this is the delta: what the new token says
    #     minus what the state already predicts. Note: u already contains
    #     the α factor because we read it from S_scaled.
    residual = vf - u  # (B, H, d_v)

    # (4) Update state with outer product:
    #       S_out = α·S  + β · k_t · (v_t - α·(k_t^T·S))^T
    #     Expanding: = α·S + β k v^T - α·β k (k^T S)
    #               = α·(I - β k k^T)·S + β k v^T       ← canonical equation
    b_bcast = bf.view(B, H, 1)
    outer = torch.einsum("bhk,bhv->bhkv", kf, residual)
    S_out = S_scaled + b_bcast.unsqueeze(-1) * outer

    # (5) Output: o = S_out^T · q  →  o[b, h, v] = sum_k q[k] · S_out[k, v]
    o_f32 = torch.einsum("bhk,bhkv->bhv", qf, S_out)

    return o_f32.to(torch.float16), S_out


def gdn_decode_reference_canonical(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor, S_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal transcription of the equation:
        S_out = alpha · (I - beta · k k^T) · S_in  +  beta · k v^T

    Used only as a witness to cross-check gdn_decode_reference.
    """
    qf, kf, vf = q.float(), k.float(), v.float()
    af, bf = alpha.float(), beta.float()
    B, H, d_k = qf.shape
    d_v = vf.shape[-1]

    # I - beta · k k^T   → (B, H, d_k, d_k)
    kkT = torch.einsum("bhi,bhj->bhij", kf, kf)
    I = torch.eye(d_k, dtype=torch.float32, device=kf.device).view(1, 1, d_k, d_k)
    M = I - bf.view(B, H, 1, 1) * kkT  # (B, H, d_k, d_k)

    # alpha · M · S_in
    MS = torch.einsum("bhij,bhjv->bhiv", M, S_in)
    aMS = af.view(B, H, 1, 1) * MS

    # + beta · k v^T
    kvT = torch.einsum("bhk,bhv->bhkv", kf, vf)
    S_out = aMS + bf.view(B, H, 1, 1) * kvT

    # o = S_out^T · q
    o_f32 = torch.einsum("bhk,bhkv->bhv", qf, S_out)
    return o_f32.to(torch.float16), S_out


def make_inputs(
    shape: GDNShape, seed: int = 0xC0FFEE
) -> dict[str, torch.Tensor]:
    """Generate seeded decode inputs for a GDN shape.

    Inputs are scaled to keep the state magnitude bounded. alpha is
    drawn from (0.85, 0.98) to mimic a realistic forget gate (exp of
    a small negative log-gate). beta is drawn from ~Uniform(-0.5, 0.5);
    the FLA reference uses `beta = sigmoid(linear(x))` in practice
    but for the kernel math only the scalar value matters.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, H, d_k, d_v = shape.batch, shape.n_heads, shape.d_k, shape.d_v

    scale_k = 1.0 / math.sqrt(d_k)
    q = torch.randn((B, H, d_k), generator=g, dtype=torch.float32) * scale_k
    k = torch.randn((B, H, d_k), generator=g, dtype=torch.float32) * scale_k
    v = torch.randn((B, H, d_v), generator=g, dtype=torch.float32)
    # alpha ∈ (0.85, 0.98) — per-head scalar forget gate.
    alpha = 0.85 + 0.13 * torch.rand((B, H), generator=g, dtype=torch.float32)
    # beta ∈ (-0.5, 0.5)
    beta = -0.5 + torch.rand((B, H), generator=g, dtype=torch.float32)
    # Initial state: small random f32 to test state round-trip correctly.
    S_in = torch.randn((B, H, d_k, d_v), generator=g, dtype=torch.float32) * 0.1

    return {
        "q_f16": q.to(torch.float16),
        "k_f16": k.to(torch.float16),
        "v_f16": v.to(torch.float16),
        "alpha_f16": alpha.to(torch.float16),
        "beta_f16": beta.to(torch.float16),
        "S_in_f32": S_in,
        # f32 copies (not strictly needed, but match wave-15 GQA convention
        # of shipping both).
        "q_f32": q,
        "k_f32": k,
        "v_f32": v,
        "alpha_f32": alpha,
        "beta_f32": beta,
    }


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
        inp = make_inputs(shape)

        q16, k16, v16 = inp["q_f16"], inp["k_f16"], inp["v_f16"]
        alpha16, beta16 = inp["alpha_f16"], inp["beta_f16"]
        S_in = inp["S_in_f32"]

        # Run reference on GPU if available (matches what the kernel will see).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q_d = q16.to(device); k_d = k16.to(device); v_d = v16.to(device)
        a_d = alpha16.to(device); b_d = beta16.to(device); S_d = S_in.to(device)

        with torch.no_grad():
            o_fused, S_fused = gdn_decode_reference(q_d, k_d, v_d, a_d, b_d, S_d)
            o_can, S_can = gdn_decode_reference_canonical(
                q_d, k_d, v_d, a_d, b_d, S_d
            )

        o_fused = o_fused.cpu(); S_fused = S_fused.cpu()
        o_can = o_can.cpu(); S_can = S_can.cpu()

        # Cross-check: the two forms must agree to within f32 roundoff.
        # Canonical materializes (I - β k k^T) of shape d_k·d_k, so it has
        # roughly sqrt(d_k) · eps more accumulated error than the fused form.
        # For d_k up to 256 and state~0.1 magnitude this is well under 1e-3.
        do = (o_fused.float() - o_can.float()).abs().max().item()
        ds = (S_fused - S_can).abs().max().item()
        ok = do < 1e-3 and ds < 1e-3
        print(
            f"[{shape.name}] fused-vs-canonical  "
            f"d_o={do:.3e}  d_S={ds:.3e}  {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            return 1

        # Save inputs + expected outputs.
        prefix = out_dir / f"gdn_{shape.name}"
        np.save(f"{prefix}_q_f16.npy", q16.numpy())
        np.save(f"{prefix}_k_f16.npy", k16.numpy())
        np.save(f"{prefix}_v_f16.npy", v16.numpy())
        np.save(f"{prefix}_alpha_f16.npy", alpha16.numpy())
        np.save(f"{prefix}_beta_f16.npy", beta16.numpy())
        np.save(f"{prefix}_S_in_f32.npy", S_in.numpy())
        np.save(f"{prefix}_q_f32.npy", inp["q_f32"].numpy())
        np.save(f"{prefix}_k_f32.npy", inp["k_f32"].numpy())
        np.save(f"{prefix}_v_f32.npy", inp["v_f32"].numpy())
        np.save(f"{prefix}_alpha_f32.npy", inp["alpha_f32"].numpy())
        np.save(f"{prefix}_beta_f32.npy", inp["beta_f32"].numpy())
        np.save(f"{prefix}_o_expected_f16.npy", o_fused.numpy())
        np.save(f"{prefix}_S_out_expected_f32.npy", S_fused.numpy())
        state_mb = S_in.numel() * 4 / 1e6
        print(
            f"  wrote {prefix}_{{q,k,v,alpha,beta}}_{{f16,f32}}.npy, "
            f"_S_in_f32.npy, _o_expected_f16.npy, _S_out_expected_f32.npy "
            f"(state={state_mb:.2f}MB f32)"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
