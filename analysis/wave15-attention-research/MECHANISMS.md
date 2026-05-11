# Wave 15 — Attention Mechanisms Research

Scope: pin down math / models / canonical implementations for six attention
mechanisms to drive a cross-frontend CUDA kernel benchmarking project
(cuda-oxide / cuTile / nvcc + Triton baseline) on an RTX 5090 (sm_120).

Every section below is grounded in a primary source (paper, GitHub repo, or
official model card). Where the task wording was ambiguous (DSA-V4 / HCA-V4),
I surface the disambiguation explicitly and cite the canonical name used by
the paper.

> **Disambiguation up front**
>
> - **MLA** — Multi-Head *Latent* Attention (DeepSeek-V2 / V3 / V3.1 / R1).
> - **GQA** — Grouped-Query Attention (Ainslie et al., Google; used by
>   Llama 2/3/4, Mistral, Qwen<=3, GPT-OSS, etc.).
> - **"DSA-DeepSeekV4"** — the user's shorthand. In the papers, the V3.2
>   mechanism is **DSA = DeepSeek *Sparse* Attention** (lightning-indexer +
>   top-k on MLA latent KV). The V4 paper calls the *evolved* version
>   **CSA = Compressed Sparse Attention** (4× block compression + indexer +
>   sliding window). I cover DSA (V3.2) **and** its direct V4 descendant CSA
>   in one section since they're the same family.
> - **"HCA-DeepSeekV4"** — the V4 paper coins **HCA = Heavily Compressed
>   Attention**: a 128× block compression followed by *dense* MQA-style
>   attention over the compressed stream. It is only defined in the V4
>   technical report (Apr 2026) and is paired with CSA in alternating layers.
> - **GDN** — Gated DeltaNet (Yang et al., ICLR 2025, arXiv 2412.06464).
>   Used in Qwen3-Next / Qwen3.5 (3:1 GDN:full ratio).
> - **KDA** — Kimi Delta Attention (Moonshot AI, arXiv 2510.26692, "Kimi
>   Linear"). A *fine-grained-gated* GDN variant. Used in Kimi-Linear
>   (48B-A3B) in a 3:1 KDA:MLA hybrid. Note: the user wrote "K2 / K2.6"; the
>   publicly open-sourced KDA reference is **Kimi Linear**, not Kimi K2
>   (which is an MLA model). K2.5 / K2.6 community write-ups describe them
>   as MLA-based; the KDA kernel lives in **fla-org/flash-linear-attention**
>   under `fla/ops/kda/`.

---

## 1. MLA — Multi-Head Latent Attention

**Origin paper.** DeepSeek-V2, arXiv 2405.04434 (2024). Re-used unchanged in
DeepSeek-V3 (Dec 2024), R1, V3.1, V3.2. Also adopted by GLM-5 and (per
community analysis) Kimi K2.5 / K2.6 (GLM-5 additionally layers DSA on top).

**Idea.** Instead of caching full per-head K and V (as MHA/GQA do), cache a
single low-rank *latent* vector `c_KV ∈ R^{d_c}` per token and *reconstruct*
the per-head K and V on the fly at attention time. RoPE is not commutative
with the up-projection, so MLA uses a **decoupled RoPE** branch: a small
MQA-style RoPE sub-head is stored separately.

**Shapes (DeepSeek-V3 config — authoritative).**

| symbol | value | meaning |
|---|---|---|
| `n_h`              | 128  | # query heads |
| `d_h = qk_nope_head_dim` | 128 | per-head "nope" dim |
| `qk_rope_head_dim` | 64   | per-head RoPE dim |
| `qk_head_dim`      | 192  | 128 + 64 |
| `v_head_dim`       | 128  | per-head V dim |
| `kv_lora_rank d_c` | 512  | latent KV rank (what's cached) |
| `q_lora_rank`      | 1536 | latent Q rank |

Per-token KV cache = `d_c + qk_rope_head_dim = 576` floats (vs. `2 · 128 · 128
= 32 768` for MHA) — a ~98% reduction; confirmed by both the DeepSeek-V2
paper and community measurements (madsys-dev profile: 81.92 kB/token MHA →
1.15 kB/token MLA).

**Forward equations.** For input `h_t ∈ R^d`:

```
# Queries — also compressed via low-rank (prevents activation-memory blow-up)
c_Q_t      = W_DQ  h_t          ∈ R^{q_lora_rank}
q_nope_t   = W_UQ  c_Q_t        ∈ R^{n_h · d_h}      (reshape → [n_h, 128])
q_rope_t   = RoPE(W_QR c_Q_t)   ∈ R^{n_h · 64}       (reshape → [n_h, 64])
q_t        = concat(q_nope_t, q_rope_t)              ∈ R^{n_h · 192}

# Latent KV (what we cache)
c_KV_t     = W_DKV h_t          ∈ R^{d_c=512}
k_rope_t   = RoPE(W_KR h_t)     ∈ R^{64}             (shared across heads, MQA-style)

# Reconstructed per-head K and V from the cache
k_nope_t   = W_UK  c_KV_t       ∈ R^{n_h · d_h}
v_t        = W_UV  c_KV_t       ∈ R^{n_h · v_head_dim}
k_t        = concat(k_nope_t, broadcast(k_rope_t))   ∈ R^{n_h · 192}

# Attention (per-head, softmax-scaled MHA)
o_t = softmax( q_t K_{≤t}^T / sqrt(qk_head_dim) ) V_{≤t}
u_t = W_O vec(o_t)
```

**Weight-absorption trick (decode).** `W_UK` and `W_UV` can be folded into
`W_UQ` and `W_O` respectively (no nonlinearity between them), so the kernel
can compute `q_t^T (W_UK c_KV_t)` without ever materializing full K. This is
what FlashMLA exploits.

**Canonical reference implementations.**

- Paper code / inference: DeepSeek-V3 repo `inference/model.py::MLA.forward`
  — <https://github.com/deepseek-ai/DeepSeek-V3>
- Pedagogical PyTorch: `sebastianraschka/LLMs-from-scratch/ch04/05_mla/`
- Clean annotated blog + Julia port: <https://liorsinai.github.io/machine-learning/2025/02/22/mla.html>
- Ablation/pedagogy: <https://planetbanatt.net/articles/mla.html> (+ github.com/ambisinister/mla-experiments)

**Fused-CUDA implementations.**

- **FlashMLA** (DeepSeek, official): <https://github.com/deepseek-ai/FlashMLA>
  — Hopper-tuned, but portable CUDA C++; uses the weight-absorption trick.
- vLLM has a hand-tuned MLA kernel in `vllm/attention/backends/mla/`.
- SGLang has an MLA backend (verda.com blog has a walkthrough).

**Compute/memory profile.**

- Prefill: compute-bound (matmuls of `q (n_h·192) · K^T (T·192)` + up-projects).
  Ratio of matmul FLOPs / softmax FLOPs ≈ same as MHA (softmax is per-row).
- Decode: *dramatically* memory-bandwidth-bound if using compressed cache
  correctly: you only stream `d_c + 64 = 576` floats/token versus 2·128·128
  for MHA. With weight absorption, the per-step decode attention becomes a
  small matmul `[n_h, d_c] · [T, d_c]^T` then softmax then `[n_h,T] · [T, d_c]`
  then up-project to `v_head_dim` — so KV traffic shrinks ~50× at the cost
  of two extra matmuls per step.

**Kernel difficulty.**

- The math itself is just standard softmax attention; the *win* is in the
  per-token up-projection fused into the attention matmul. Writing the naive
  MLA kernel is straightforward; writing the *efficient* MLA kernel with
  weight absorption is substantially harder because you need to emit two
  different GEMM shapes (decode vs prefill).
- RoPE on the separate 64-dim sub-head is a small side-band op.
- **Estimated LOC for a cross-frontend benchmark kernel (no weight
  absorption, just pedagogical MLA):** ~300–400 LOC per frontend.

---

## 2. GQA — Grouped-Query Attention

**Origin paper.** Ainslie et al., "GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints", arXiv 2305.13245 (EMNLP 2023).

**Idea.** Keep `n_q` query heads as in MHA but have only `n_kv < n_q` KV
heads; each KV head is shared by `g = n_q / n_kv` query heads (MQA is the
special case `n_kv = 1`, MHA is `n_kv = n_q`).

**Shapes.**

- Llama-3 70B: `n_q=64`, `n_kv=8`, `d_h=128`, `g=8`.
- Llama-3 8B:  `n_q=32`, `n_kv=8`, `d_h=128`, `g=4`.
- Mistral-7B:  `n_q=32`, `n_kv=8`, `d_h=128`, `g=4`.

**Forward equations.**

```
Q = X W_Q  ∈ R^{T × n_q · d_h}          reshape → [T, n_q, d_h]
K = X W_K  ∈ R^{T × n_kv · d_h}         reshape → [T, n_kv, d_h]
V = X W_V  ∈ R^{T × n_kv · d_h}         reshape → [T, n_kv, d_h]

For each query head h, let g(h) = h // (n_q / n_kv):
  S_{h}    = Q_h K_{g(h)}^T / sqrt(d_h)          ∈ R^{T × T}
  P_{h}    = softmax(S_{h} + mask)
  O_{h}    = P_{h} V_{g(h)}                       ∈ R^{T × d_h}
```

**Canonical reference implementations.**

- Paper code + official Google release inside T5X.
- Reference in every Llama-family codebase, e.g. meta-llama/llama
  `model.py::Attention.forward`.
- Pedagogical: sebastianraschka `ch03_02_gqa`,
  <https://www.intoai.pub/p/grouped-query-attention>.

**Fused-CUDA implementations.**

- **Flash Attention 2/3** — native GQA support via `num_kv_heads` argument:
  <https://github.com/Dao-AILab/flash-attention>
- xformers `memory_efficient_attention` (handles GQA).
- PyTorch `F.scaled_dot_product_attention` (all three backends support GQA).
- cuDNN's Fused Flash Attention supports GQA since 9.x.
- NVIDIA's **cuTile** and **cuda-oxide** FMHA samples already target GQA
  as their primary shape.

**Compute/memory profile.**

- GQA is *almost exactly* the FlashAttention roofline. The only change vs.
  MHA is that K,V are read `g` times fewer bytes — purely a bandwidth win
  during decode (prefill is compute-bound the same way MHA is).
- Matmul FLOPs: `2 · T^2 · n_q · d_h` (same as MHA). Softmax FLOPs:
  `O(T^2 · n_q)` per sequence. Arithmetic intensity roofline ≈
  FLOPs / bytes ≈ `n_q · d_h` FLOPs per KV-byte read.

**Kernel difficulty.**

- **Easiest of the six.** Every modern attention kernel (FA2, FA3, cuDNN,
  triton tutorial 06) already implements it. A clean textbook FlashAttention
  forward pass with GQA broadcast is ~250–350 LOC per frontend.

---

## 3. DSA (DeepSeek-V3.2) + CSA (DeepSeek-V4) — "DSA-DeepSeekV4"

**Disambiguation.** The literal string "DSA-DeepSeekV4" conflates two
mechanisms: DSA is the V3.2 mechanism, and V4 *evolves* it into CSA. The V4
paper spells both out explicitly and uses CSA, not DSA, in V4.

### 3a. DSA (V3.2-Exp + V3.2)

**Origin paper.** "DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with
DeepSeek Sparse Attention" — GitHub
<https://github.com/deepseek-ai/DeepSeek-V3.2-Exp> (paper PDF in repo root,
Sept 2025). V3.2 full paper: arXiv 2512.02556 (Dec 2025).

**Idea.** DSA = **MLA + per-query top-k sparse selection over the cached
latent KV sequence**, driven by a cheap FP8 "lightning indexer". MLA
structure is kept 1:1 — only which past tokens participate is pruned.

**Architecture (from the DeepWiki dump of the V3.2-Exp repo).**

- **Indexer** runs *in parallel* to MLA, with its own tiny projections
  `wq_b, wk`. Config: `index_n_heads = 64`, `index_head_dim = 128`,
  `index_topk = 2048`.
- Indexer KV cache is FP8-quantized (separate from the MLA latent cache).
- Per query position `t`:
  ```
  q_idx = W_IQ h_t         ∈ R^{64 × 128}    (FP8)
  k_idx_j = W_IK h_j       ∈ R^{128}         (FP8, cached)
  score_t_j = Σ_h  ReLU( ⟨q_idx[h], k_idx_j⟩ ) · s_q · s_k
  topk_t   = top-2048 j in [0,t] by score_t_j
  ```
  (ReLU → sum-across-heads is a non-standard "lightning" score — cheaper
  than softmax and numerically friendly for FP8.)
- The MLA attention is then computed *only* over `topk_t`, i.e. a standard
  MLA with an additive `-inf` mask on non-selected positions.

**Complexity.** Attention compute drops from O(T²) to O(T · 2048). Indexer
itself is O(T²) in *tiny* FLOPs (FP8, 64×128 per token) and O(T) memory.

**Canonical reference.**

- `inference/model.py::MLA.forward` and `::Indexer.forward` in the V3.2-Exp
  repo.
- `inference/kernel.py::fp8_index` / `fp8_index_kernel` (TileLang). Block
  dims 512×128, 4-stage pipelining.
- Production kernel variants: `DeepGEMM` + `FlashMLA` (same repo family),
  plus day-0 vLLM support.

**Fused-CUDA existing.**

- DeepSeek's own TileLang kernel (research).
- vLLM backend for DSA (production).
- The "FP4 lightning indexer" phrasing from the V4 HF blog is V4-specific;
  V3.2 uses FP8. Fireworks/AI have an interactive visualizer:
  <https://demos.fireworks.ai/deepseek-v3.2>.

**Compute/memory profile.**

- Indexer: FP8 GEMM, tiny d=128, top-k sort. Bandwidth-bound at long T.
- Main attention: same as MLA but with a sparsity pattern that is **not**
  contiguous (top-k indices are scattered). This is exactly the pattern that
  FA variants with arbitrary masks struggle with.
- Matmul / softmax ratio: dominated by the `[n_h, d_c] · [2048, d_c]^T`
  matmul → softmax over 2048 → `[n_h, 2048] · [2048, v_head_dim]`. Softmax
  is a much smaller fraction than dense MHA because T→2048.

**Kernel difficulty.**

- **Hard.** Three sub-kernels to fuse (indexer score, top-k, sparse MLA).
- Top-k on-device is itself a non-trivial primitive across frontends (cuTile
  and cuda-oxide may not expose a one-liner — may need a radix/bitonic impl).
- The sparse-gather attention kernel is the hardest piece: you need
  FlashAttention logic *with* a per-query index list, so the K/V tiles are
  scatter-loaded, breaking the usual contiguous-tile assumption of
  FlashAttention.
- **Estimated LOC:** 800–1200 per frontend for a correct end-to-end DSA.

### 3b. CSA (V4) — V4's successor to DSA

**Origin paper.** "DeepSeek-V4: Towards Highly Efficient Million-Token
Context Intelligence", April 24 2026 (PDF at huggingface.co/deepseek-ai/
DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf). HF blog:
<https://huggingface.co/blog/deepseekv4>.

**Idea.** Same "indexer + top-k" shape as DSA, but:

1. **Compress first**: collapse every 4 consecutive tokens into a single
   compressed KV entry via softmax-gated pooling with a learned positional
   bias (so the sequence the indexer looks at is 4× shorter).
2. **Lightning indexer** (now in FP4, ReLU-scored multi-head dot product)
   picks top-k *compressed blocks* per query.
3. **Sliding-window branch** handles the most recent uncompressed tokens
   (for local detail).
4. KV cache: FP8 entries, BF16 only for the RoPE dims. Indexer runs in FP4.

**Layer layout (V4-Pro, 61 layers).** Layers 0–1: HCA. Layers 2–60: alternate
CSA / HCA. MTP block at end: sliding-window only.

**Canonical reference.**

- Official: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro> (paper PDF +
  modeling code).
- Also inherited from the V3.2-Exp indexer pipeline — same `fp8_index_kernel`
  shape with an additional 4-token pooling stage.

**Kernel difficulty.**

- **Extreme.** Everything DSA has + a fused compressor + FP4 tensor core path
  (Blackwell-only without emulation — 5090 has FP4 tensor cores, so
  hardware-feasible, but cuTile/cuda-oxide FP4 support varies) + a parallel
  sliding-window branch that must be aggregated into the same output.
- **Estimated LOC:** 1500+ per frontend.

---

## 4. HCA (DeepSeek-V4) — "HCA-DeepSeekV4"

**Disambiguation.** There are three unrelated things abbreviated "HCA" in
the ML literature (Hierarchical Causal Attention in some 2023 papers,
Hybrid Cache Attention, etc.). **In the DeepSeek-V4 paper, HCA = Heavily
Compressed Attention** — this is the only one meant here.

**Origin paper.** Same as CSA: DeepSeek-V4 technical report, April 2026.

**Idea.** **128× sequence compression + dense MQA over the compressed
stream.** The rationale is that long-distance context doesn't need
token-level resolution; a heavy compressor (factor 128) produces a sequence
short enough that *dense* attention is cheap, so the sparse selection of DSA
is dropped entirely. A sliding-window branch (same as CSA) provides local
fidelity.

**Architecture (from HF blog Figure 4).**

```
# For a sequence of length T:
compressed_K, compressed_V   ∈ R^{T/128 × d_kv}     via learned pool op
# Dense MQA (single KV head; query has n_q heads)
out_long  = softmax( Q · compressed_K^T / sqrt(d) ) · compressed_V

# Sliding-window branch
out_local = FlashAttention( Q, K_local, V_local, window=W )   # e.g. W=2048

out = out_long + out_local         # summed into same residual (alpha-gated)
```

**Compute/memory profile.**

- Attention sequence length is effectively `T/128 + W`. At T=1M and W=2048,
  that's ~9.8k — *dense* softmax is trivially fast.
- KV cache: 128× smaller for the long branch; constant-W for the local branch.
- Arithmetic intensity: dominated by the compressor itself (a fused pool
  over 128-token windows into a single `d_kv`-dim output).

**Canonical reference.**

- DeepSeek-V4-Pro modeling code on HF. (No separate open-source kernel has
  been released as of Apr 2026 — vLLM will be first.)

**Kernel difficulty.**

- **Medium.** Conceptually simpler than DSA/CSA because once the compressor
  runs, the core attention is a vanilla (small-T) FlashAttention.
- The **compressor** is a pooling+softmax op with a learned positional bias
  — non-standard and must be written from scratch.
- Two attention paths (long + sliding-window) must be summed; if done
  naively that's two kernel launches. A fused version is a research-grade
  effort.
- **Estimated LOC:** 600–900 per frontend for the unfused version, 1200+ for
  the fused version.

---

## 5. GDN — Gated DeltaNet

**Origin paper.** Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving
Mamba2 with Delta Rule", ICLR 2025, arXiv **2412.06464** (Dec 2024).

**Idea.** Linear attention with a *recurrent state matrix* `S_t ∈ R^{d_k × d_v}`
updated per token via the **delta rule** plus a scalar per-head forget gate
`α_t ∈ [0,1]`. KV cache is replaced by a fixed-size `S` independent of T.

**State recurrence (exact, per head).**

```
S_t = α_t · ( I - β_t k_t k_t^T ) · S_{t-1}  +  β_t · k_t v_t^T
o_t = S_t^T q_t
```

with `α_t = exp(g_t)` where `g_t` is a learned log-forget-gate in
R, and `β_t ∈ R` is a learned scalar (the "prediction-step" size). Note that
`(I - β k k^T) S_{t-1}` subtracts out the *current prediction* before writing
— this is the delta rule (Schlag 2021 → Yang 2024).

**Qwen3-Next / Qwen3.5 config (used in production).**

- Hybrid 3:1: most layers are GDN, one-in-four is Gated Attention (full
  softmax attn). Source: HF blog "Qwen3.5: Nobody Agrees on Attention
  Anymore" (mlabonne).
- `head_dim` typically 256 for GDN (per `fla-org/flash-linear-attention`
  defaults). State size per head = `d_k · d_v = 256 · 256 = 64k floats`.

**Chunkwise-parallel training algorithm.** Yang et al. derive a WY-style
factorization so that each chunk of size `BT` (default 64) can be computed
as a sequence of small matmuls + a local triangular solve. The inter-chunk
state recurrence is:

```
S_{[t+1]} = γ_C_[t] · S_[t]  +  ( Γ^{i→C}_[t] ⊙ K_[t] )^T · ( U_[t] - W_[t] · S_[t] )
```

where `W_[t], U_[t]` are the WY-representation outputs of
`chunk_gated_delta_rule_fwd_kkt_solve_kernel`.

**Canonical reference implementation.**

- `fla-org/flash-linear-attention`, path
  `fla/ops/gated_delta_rule/` — Triton kernels.
  - Fused recurrent (decode): `fused_recurrent_gated_delta_rule_fwd_kernel`.
  - Chunked (training/prefill): `chunk_gated_delta_rule_fwd_kkt_solve_kernel`,
    `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`.
- Pedagogical: `rasbt/LLMs-from-scratch/ch04/08_deltanet/README.md`.
- Reference HF integration: `transformers/src/.../qwen3_next/modular_qwen3_next.py`.

**Fused-CUDA implementations.** None publicly as CUDA C++; FLA's Triton
kernels are the de-facto standard. An FPGA paper (arXiv 2603.05931)
demonstrates that on-GPU GDN decode is memory-bound (the 2 MB state must
round-trip through HBM every token at batch-1).

**Compute/memory profile.**

- Decode: O(d_k · d_v) compute per token, O(d_k · d_v) state memory traffic.
  Memory-bound at batch-1 because the whole state must be streamed.
- Prefill/training (chunkwise): compute-bound if `BT` is large enough — most
  work is `K · K^T` (BT×BT) and `K · S` (BT×d_k ⊗ d_k×d_v) matmuls.
- Matmul / softmax: **no softmax at all**. The only elementwise nonlinearity
  is the gate `exp(g_t)` and (optionally) a normalizer like SiLU on `q,k`.

**Kernel difficulty.**

- **Fused recurrent decode kernel: moderate.** ~200 LOC in Triton. Three
  ops per token: scale state, compute prediction, write outer product.
  Needs `tl.dot` for the inner matmul and `tl.exp` for the gate. No softmax,
  no top-k, no RoPE.
- **Chunkwise training kernel: much harder.** The WY representation requires
  a small triangular solve of shape `[BT, BT]` (usually done by backward
  substitution in registers or shared memory). This is the "scan inside
  a tile" pattern that cuTile/cuda-oxide may not support directly.
- **Estimated LOC (decode only, linear-time kernel):** 250–400 per frontend.
- **Estimated LOC (chunkwise):** 700–1000 per frontend.

---

## 6. KDA — Kimi Delta Attention

**Origin paper.** "Kimi Linear: An Expressive, Efficient Attention
Architecture", arXiv **2510.26692** (Oct 2025). Team: Moonshot AI.
Model: Kimi-Linear-48B-A3B (base + instruct). Repo:
<https://github.com/MoonshotAI/Kimi-Linear>. CUTLASS kernel ("FlashKDA"):
marktechpost coverage, Apr 2026, open-sourced under FLA.

**Idea.** KDA = GDN **with a per-channel (diagonal) gate instead of a
scalar**. Everything else (delta rule, state shape, WY chunking) is the same.

**State recurrence (exact, per head).**

```
S_t = ( I - β_t k_t k_t^T ) · Diag(α_t) · S_{t-1}  +  β_t · k_t v_t^T
o_t = S_t^T q_t
```

with `α_t ∈ R^{d_k}` (a **vector**, one gate per key dim) — this is the
"finer-grained gating" compared to GDN's scalar `α_t`.

**Gate parameterization (from fla-org/kda).** Two modes:

- Standard: `g_t = -exp(A_log) · softplus(g_input + dt_bias)` then
  `α_t = exp(g_t)`.
- Safe: `α_t = lower_bound · sigmoid( exp(A_log) · (g_input + dt_bias) )`.

`A_log, dt_bias` are learned per-head-per-key-dim parameters (shape `[n_h, d_k]`).

**Kimi-Linear shapes.**

- `n_h` query heads per KDA layer, `d_k = 128` (typical).
- `d_v = expand_v · d_k` (expansion ratio usually 2, so `d_v = 256`).
- Hybrid 3:1 KDA:MLA layer pattern (75% KDA layers). KV cache saving: 75%.
- 48B total / 3B active params, 1M context length.

**Canonical reference implementation.**

- `fla-org/flash-linear-attention`, path `fla/ops/kda/`.
  - Naive PyTorch: `fla/ops/kda/naive.py::naive_recurrent_kda` and
    `naive_chunk_kda`.
  - Fused recurrent kernel: `fused_recurrent_kda_fwd_kernel`.
  - Chunked: `chunk_kda_fwd_kernel_intra_token_parallel`,
    `chunk_kda_fwd_kernel_inter_solve_fused`,
    `recompute_w_u_fwd_kda_kernel`.
- Gate cumsum helper: `kda_gate_chunk_cumsum` (because the decay is now
  per-channel, the cumsum is done over `d_k`-wide vectors).

**Fused-CUDA implementations.**

- **FlashKDA** (Moonshot, CUTLASS-based) — Apr 2026; variable-length
  batching, benchmarked on H20. This is the "official" CUDA C++ kernel.
  Coverage: <https://www.marktechpost.com/2026/04/30/moonshot-ai-open-sources-flashkda-cutlass-kernels-...>.
- FLA's Triton kernels (cross-platform reference).

**Compute/memory profile.**

- Essentially identical to GDN: O(d_k · d_v) per token, memory-bound at
  batch-1 decode.
- The per-channel gate adds a `d_k`-way elementwise multiply per token, not
  a performance change.
- `expand_v=2` doubles the state and therefore the decode bandwidth vs GDN
  with equal head_dim — so KDA is somewhat heavier than GDN at decode.

**Kernel difficulty.**

- **Strictly harder than GDN** because:
  1. The gate `α_t` is a vector — the decay can no longer be folded into a
     single scalar multiply; you need a per-dim multiply inside the state
     update, and the chunkwise cumsum becomes a vectorized cumsum.
  2. The WY representation for chunkwise now has an `Diag(α)` term mixed in
     between the two sides of the delta subtraction.
  3. Safe-mode gate has an `exp(A_log)` prefactor that needs careful FP
     handling (numerical stability is a real concern per the paper's ablations).
- **Estimated LOC (decode):** 350–500 per frontend.
- **Estimated LOC (chunkwise):** 1000–1500 per frontend.

---

## Cross-cutting summary table

| Mechanism | Canonical repo / kernel | Compute vs memory | LOC (decode) | LOC (full) | Cross-frontend risk |
|---|---|---|---|---|---|
| MLA | DeepSeek-V3 + FlashMLA | compute-bound prefill / BW-bound decode | ~250 | ~400 | **low** |
| GQA | FlashAttention 2/3, cuDNN, Triton tutorial 06 | FlashAttn roofline | ~250 | ~350 | **very low** |
| DSA (V3.2) | DeepSeek-V3.2-Exp (TileLang) | indexer BW + sparse attn | ~500 | ~1000 | **high** (top-k + sparse gather) |
| CSA (V4) | DeepSeek-V4 modeling code | compressor + sparse attn + sliding win | ~700 | ~1500 | **very high** (FP4, three sub-ops) |
| HCA (V4) | DeepSeek-V4 modeling code | compressor + small dense attn | ~400 | ~900 | **medium** (compressor is novel) |
| GDN | fla-org/flash-linear-attention `ops/gated_delta_rule/` | BW-bound decode | ~250 | ~900 | **medium** (WY/tri-solve for full) |
| KDA | fla-org/flash-linear-attention `ops/kda/` + FlashKDA | BW-bound decode | ~400 | ~1400 | **medium-high** (vector gate + tri-solve) |

### Feature availability sanity check

- **cuda-oxide / cuTile / nvcc**: all support standard FP16/BF16 tensor
  cores, async shared-memory copy, and FlashAttention-style tiling. Good
  for GQA and MLA and GDN-decode.
- **FP4**: only on sm_100+ tensor cores (RTX 5090 is sm_120, so it has FP4
  MMA). However, **cuda-oxide and cuTile FP4 support is unproven as of
  this writing**, and FP4 is *required* for V4's lightning indexer and
  expert weights. This pushes CSA/HCA into "nvcc-only" territory.
- **On-device top-k**: not a built-in primitive in cuTile. Would need a
  custom warp-level bitonic sort or a two-pass threshold. This pushes
  DSA/CSA kernels up by another 200+ LOC.
- **Triangular solve inside a tile**: needed by GDN/KDA chunkwise. Also not
  a primitive in any of the four frontends — must be hand-written. But
  decode-only kernels avoid this.

---

## Recommendation — which 2–3 to prioritize

**Choose GQA + MLA + GDN-decode.**

- **GQA** (required baseline). Every frontend has hello-world FlashAttention
  samples. Tiny LOC. Validates the tiling/softmax/masking primitives across
  cuda-oxide, cuTile, nvcc, and Triton. *This is the calibration benchmark.*
- **MLA** (headline "modern-transformer" kernel). Well-understood math, a
  reference CUDA kernel exists (FlashMLA), and the implementation stays
  under 500 LOC if you skip weight absorption. Gives a cross-frontend story
  on a non-trivial attention shape (192-dim Q/K, 128-dim V, decoupled RoPE)
  that every frontend should be able to hit.
- **GDN-decode** (fundamentally different regime). Tests whether the
  frontends can express a *recurrent-state* kernel with an outer-product
  update and a gated linear scan — no softmax, no top-k, no triangular
  solve. This is the cleanest linear-attention kernel and therefore the
  cleanest test for "can frontend X handle a stateful recurrence".
  Reference: `fla-org/flash-linear-attention` Triton kernel, ~200 LOC.

**Why not KDA first?** KDA is a tiny delta (vector gate) on top of GDN. If
the GDN-decode kernel works in all four frontends, KDA is a ~30-LOC
extension; we get it "for free" without spending a full wave on it. So KDA
should be kept as a **stretch goal** rather than a day-one target.

**Why not DSA/CSA/HCA first?** They require three capabilities simultaneously
— on-device top-k, sparse-gather attention, and (for V4) FP4 tensor cores.
Any one of these being unavailable in cuda-oxide or cuTile blows the budget.
The right move is to do DSA/HCA as a *second-wave* deliverable after the
GQA+MLA+GDN spine is proven.

### Final 5-bullet session plan

- **Kernel 1 — GQA**: canonical FlashAttention-2-style forward, Llama-3-8B
  shape (`n_q=32, n_kv=8, d_h=128`). Target: <300 LOC per frontend. Reference
  implementation: Triton tutorial 06 + Dao-AILab/flash-attention.
- **Kernel 2 — MLA (pedagogical, no weight absorption)**: DeepSeek-V3 shape
  (`n_h=128, d_h=128, rope=64, d_c=512, v=128`). Target: <500 LOC per
  frontend. Reference: DeepSeek-V3 `MLA.forward` + FlashMLA source.
- **Kernel 3 — GDN fused-recurrent decode**: Qwen3-Next shape (`d_k=d_v=256`).
  Target: <300 LOC per frontend. Reference:
  `fla/ops/gated_delta_rule/fused_recurrent.py`.
- **Stretch — KDA decode**: same code as GDN decode, swap scalar gate for a
  `d_k`-vector gate; piggy-backs on GDN kernel infrastructure.
- **Out of scope this wave — DSA, CSA, HCA**: defer until the three-kernel
  spine is validated; they require FP4 tensor cores and/or on-device top-k
  which are not guaranteed to be available across all four frontends.
