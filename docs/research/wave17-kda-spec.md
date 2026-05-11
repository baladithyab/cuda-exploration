# Wave 17 — KDA (Kimi Delta Attention) Specification

Ground-truth reference for a cuTile implementation of KDA's single-timestep
decode, prepared as a delta against the existing GDN decode fused kernel at
`cutile-attn-gdn/main.py`. Primary sources:

- Paper: *Kimi Linear: An Expressive, Efficient Attention Architecture*,
  arXiv:2510.26692 (Moonshot, 2025-10-30).
- Reference PyTorch kernel family: `fla-org/flash-linear-attention`,
  `fla/ops/kda/` (+ `fla/layers/kda.py` for the full layer).
- Reference CUTLASS kernel: `MoonshotAI/FlashKDA` (v1, 2026-04, CHUNK=16, CUTLASS-based).
- Reference model config: `moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json`.

## 1. KDA recurrence (canonical, verbatim)

From the paper §3 and the arXiv PDF:

> At its core lies Kimi Delta Attention (KDA) … KDA introduces a **channel-wise
> variant** in which each feature dimension maintains an independent forgetting
> rate … KDA refines GDN's scalar decay by introducing a fine-grained
> diagonalized gate Diag(αₜ).

The recurrence (paper Eq. in §3, mirrored in `fla/ops/kda/naive.py` and the
educational repo `hwilner/kimi-delta-attention`):

```
S_t = ( I − β_t k_t k_tᵀ ) · Diag(α_t) · S_{t−1}  +  β_t k_t v_tᵀ
o_t = S_tᵀ q_t
```

where `α_t ∈ (0,1]^{d_k}` is the **per-channel** (per-key-dim) decay vector
for the current token. Equivalently, with `g_t = log(α_t)` (the log-space
gate the kernel actually consumes):

```
S_t = Diag(exp(g_t)) · S_{t−1} + β_t k_t (v_t − k_tᵀ S_{t−1} decayed)ᵀ
```

The `fla` naive implementation fuses the decay into the state update rather
than the error residual — see §4 for the exact Python.

## 2. Diff vs GDN

GDN (Gated DeltaNet, Yang 2024; Qwen3-Next uses this) recurrence:

```
S_t = α_t · ( I − β_t k_t k_tᵀ ) · S_{t−1}  +  β_t k_t v_tᵀ         (scalar α_t ∈ ℝ)
```

| aspect                  | GDN                                  | KDA                                   |
|-------------------------|--------------------------------------|----------------------------------------|
| gate rank               | scalar per head                      | **vector per head, length d_k**        |
| gate tensor shape       | `[B, T, H]`                          | **`[B, T, H_v, d_k]`**                 |
| state decay op          | `S *= alpha`  (broadcast scalar)     | **`S *= exp(g)[..., None]`** (broadcast a d_k-vector over d_v cols) |
| gate preprocessing      | typically `alpha = sigmoid(x·W)`     | **`g = −exp(A_log) · softplus(f_proj(x) + dt_bias)`**; `A_log ∈ ℝ^{H_v}`, `dt_bias ∈ ℝ^{H_v · d_k}` |
| DPLR structure          | diagonal decay + rank-1 keyed update | constrained DPLR: decay & learning rate tied to `k_t` (enables chunkwise parallel)  |
| positional awareness    | none (scalar decay is positionless)  | **encodes position via channel-wise decay** → Kimi-Linear uses NoPE for MLA layers |
| FLA gate-in-kernel path | scalar broadcast, cheap              | per-dim decay is **pre-gated** into tensors before the WY matmul |

The output equation `o = Sᵀ q` is identical in form; only the state update
changes.

## 3. Canonical shapes (Kimi-Linear-48B)

From `config.json` at HF `moonshotai/Kimi-Linear-48B-A3B-Instruct`:

```
"linear_attn_config": {
    "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
    "kda_layers":       [1..27 minus above],   # 20 KDA layers, 7 MLA
    "head_dim":         128,
    "num_heads":        32,
    "short_conv_kernel_size": 4
}
```

Derived shapes:
- **n_heads = 32**
- **d_k = 128** (head_dim for KDA)
- **d_v = 128** (FLA `expand_v=1.0` default; config doesn't specify override and v_head_dim=128)
- **gate dim per head = d_k = 128**
- **state per head = d_k × d_v = 128 × 128 = 16 384 f32 = 64 KB**

FlashKDA kernel currently **requires K = V = 128**, confirming this is *the*
shape. Per-head state footprint matches GDN at d_k=256, d_v=256 (in fact
smaller — 64 KB vs 256 KB), so the shared-memory story is easier for KDA than
for our Qwen3-Next GDN bench shape.

**Canonical bench shape proposal for cutile-attn-kda:**
```
B=1, n_heads=32, d_k=128, d_v=128   (one "Kimi-Linear decode step")
```
Per-token state traffic: `32 × 128 × 128 × 4 B = 2.0 MB` R + 2.0 MB W = **4 MB**
(vs ~32 MB for our Qwen3-Next GDN bench). A single-step decode of KDA is
**~8× less memory-bound per head** than the GDN-Qwen shape.

## 4. Canonical Python reference

**Path:** `fla/ops/kda/naive.py`, function `naive_recurrent_kda`. License: MIT.
Authors: Songlin Yang, Yu Zhang, Zhiyuan Li.

Signature and inner loop (verbatim, lines ~10–64):

```python
def naive_recurrent_kda(
    q, k, v, g, beta,
    scale=None, initial_state=None, output_final_state=False,
):
    # q, k   : [B, T, H,  K]
    # v      : [B, T, HV, V]            # HV must be divisible by H (GVA)
    # g      : [B, T, HV, K]            # per-dim log-gate
    # beta   : [B, T, HV]               # scalar per (head, token)
    # initial_state: [B, HV, K, V]
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q,k,v,g,beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None: S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:,i], k[:,i], v[:,i], g[:,i], beta[:,i]
        S = S * g_i[..., None].exp()                    # Diag(exp(g)) · S
        S = S + torch.einsum(
            'b h k, b h v -> b h k v',
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),         # residual, post-decay
        )
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    return o, (S if output_final_state else None)
```

And the gate preprocessing (from `naive_kda_gate` in the same dir):

```python
g = -A_log.view(H, 1).float().exp() * F.softplus(g.float() + dt_bias.view(H,-1))
```

With `safe_gate=True` (FlashKDA fast-path) the activation instead becomes
`g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`; `lower_bound` is
typically −5 (decay ≥ exp(−5) ≈ 0.0067).

Note on the residual: FLA computes `v − (k·S)` **after** decaying S by
`exp(g)`. Algebraically this matches the paper since `Diag(α)` commutes out
of `k·S` only via `k·Diag(α)·S_prev`; the `fla` ordering is `S := α·S_prev`
then `residual := v − kᵀS`. Confirm during kernel implementation — the
ordering changes the inner-loop dependency chain.

## 5. Implementation delta vs cutile-attn-gdn — honest assessment

Current GDN decode kernel inner body (from `cutile-attn-gdn/main.py:141-159`):

```python
s_scaled = s_tile * alpha_f32                     # scalar broadcast over (d_k, BV)
u_acc = ct.mma(k_f32, s_scaled, zeros)            # (1, BV)
residual = v_f32 - u_acc
outer_acc = ct.mma(k_col, residual, zeros)        # (d_k, BV)
s_out = s_scaled + beta_f32 * outer_acc
o_acc = ct.mma(q_f32, s_out, zeros)
```

For KDA the **only** mathematical change is line 1: replace the scalar-times-
tile broadcast with a vector-times-row broadcast. Everything downstream is
unchanged because decay on the K-axis commutes through both the `k·S` reduce
(K-axis contraction) and the `k ⊗ residual` outer product (K-axis outer).

**KDA inner body:**
```python
g_tile = g_view.load((bh, 0))                 # (1, d_k) f16 log-gate
g_f32  = g_tile.astype(ct.float32)
alpha  = ct.exp(g_f32)                        # (1, d_k)  per-channel decay
# Broadcast alpha (1, d_k) along BV → scale each row of S by alpha[k].
s_scaled = s_tile * ct.transpose(alpha)       # (d_k, 1) × (d_k, BV) broadcast
# … rest identical to GDN
```

**Code delta size (honest count, not including tests/bench scaffolding):**

| change                                                  | ~LOC |
|---------------------------------------------------------|------|
| drop `Alpha` scalar arg, add `G` gate arg               | ±3   |
| load `g_tile` (1, d_k), upcast, `ct.exp`                | 3    |
| transpose alpha to (d_k, 1) and replace scalar mul       | 1    |
| update `KernelSignature` parameter list                 | ±2   |
| input plumbing: load `g_f16/f32.npy` instead of `alpha` | ~6   |
| update input generator / reference to emit `g` & honor `Diag(α)` recurrence | ~15 |
| update `shapes_kda.py` (new file) with 32×128×128       | ~10  |
| bench harness reuses GDN scaffolding verbatim           | 0    |

**Total kernel delta: ~10 LOC.**
**Total repo delta (kernel + shapes + reference + inputs): ~40–60 LOC.**

The "~30-LOC delta on GDN-decode" recommendation from Wave 15 is **accurate
for the kernel itself and mildly optimistic for the full harness**. There is
**no inner-loop restructuring** required because:

1. The decay commutes correctly through both the `mma(k, s, …)` reduction
   and the `mma(k_col, residual, …)` outer product on the K-axis.
2. Per-head state tile dims (128 × 128 = 64 KB f32) are **strictly smaller**
   than our current Qwen3-Next GDN bench (256 × 64 = 64 KB f32 — same)
   or the larger d_k variants — so shared-memory feasibility is at least as
   good as GDN.
3. The gate preprocessing (`-exp(A_log) * softplus(x + dt_bias)`) lives
   **outside** the fused kernel — the kernel sees pre-activation `g` in log
   space and calls `exp()` once. This matches FlashKDA's `fwd` API, which
   takes `g` as `bf16 [B, T, H, K]` "Gate before activation" and applies
   `A_log`/`dt_bias` internally.

### Caveats (things that will NOT be 30-LOC)

- **Chunked / multi-timestep KDA** (matching FlashKDA's CHUNK=16 CUTLASS
  kernel and `fla.ops.kda.chunk_kda`) is a **much larger** engineering task.
  The chunked form requires a bespoke DPLR chunkwise algorithm — `naive_chunk_kda`
  is ~80 lines of Python, the CUTLASS impl is a K1 (token-parallel) + K2
  (head-parallel) two-kernel split (per `FlashKDA/docs/20260420-flashkda-v1-deep-dive.md`).
  **Wave 17 should stay decode-only (T=1)** to preserve the ~30-LOC delta.
- **GVA (Grouped Value Attention):** FLA's KDA supports `num_v_heads >
  num_heads` (q/k repeat-interleaved). Kimi-Linear-48B has num_heads=32 and
  num_v_heads=32 (no GVA) — safe to ignore for Wave 17.
- **Numerical correctness:** the `fla` naive applies decay to S *before*
  computing the residual; reproduce that ordering in the reference to match.
- **f16 vs bf16:** FlashKDA uses bf16; our GDN harness uses f16. Inputs will
  need regeneration with `g` as f16/bf16 log-gate values (not `alpha = exp(g)`
  pre-exponentiated) if we want to match the kernel API semantics exactly.

## 6. Recommended Wave 17 plan (distilled)

1. Copy `cutile-attn-gdn/` → `cutile-attn-kda/`.
2. Swap `alpha: (B·H, 1)` for `g: (B·H, 1, d_k)` in I/O + signature.
3. Inner loop: compute `alpha = exp(g)`, broadcast along BV, everything else
   unchanged. ~10 LOC kernel delta.
4. New `shapes_kda.py` with `SHAPE_KIMI_LINEAR_DECODE(B=1, H=32, d_k=128, d_v=128)`.
5. Reference generator: call `naive_recurrent_kda` (MIT, vendor into repo or
   install fla), dump f16 inputs + f32 state.
6. Bench against FlashKDA only if we install the CUTLASS package (optional —
   not required to validate the cuTile impl).

**Expected outcome:** single-step KDA decode on RTX 5090 should hit
memory-bound regime at smaller traffic than GDN-Qwen-decode (4 MB vs 32 MB),
so likely dominated by launch overhead and/or register pressure rather than
HBM BW. That's the interesting finding for Wave 17, not the algorithm itself.
