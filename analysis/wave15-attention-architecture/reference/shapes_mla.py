"""Wave 16.3 — canonical MLA (Multi-Head Latent Attention) shape definitions.

MLA is the DeepSeek-V3 mechanism. The distinguishing feature is that K and V
are reconstructed at attention time from a low-rank latent `c_KV` via up-
projection weights, and the KV cache stores only `c_KV` plus a small decoupled
RoPE sub-head, *not* the full per-head K and V. For the pedagogical MVP here
we bypass weight absorption and treat the kernel's inputs as the already-
projected Q / K / V tensors — so the attention itself looks like MHA with a
non-square per-head shape (qk_head_dim = 192, v_head_dim = 128).

Authoritative shape numbers come from the DeepSeek-V3 config (see
`analysis/wave15-attention-research/MECHANISMS.md` §1):

    n_h = 128
    d_h (qk_nope_head_dim) = 128
    d_rope (qk_rope_head_dim) = 64
    qk_head_dim = d_h + d_rope = 192
    v_head_dim = 128
    d_c (kv_lora_rank) = 512
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLAShape:
    """Multi-Head Latent Attention shape parameters.

    For MVP (no weight-absorption) the kernel consumes already-up-projected Q,
    K, V. Both Q and K have per-head dim = d_h + d_rope. V has per-head dim =
    d_v. All heads are independent (no GQA-style sharing); the 'latent'
    property of MLA only manifests at the cache stage, which the pedagogical
    kernel doesn't model.
    """

    name: str
    batch: int
    seq: int
    n_h: int        # # query / KV heads (MLA up-projects to full n_h heads)
    d_h: int        # per-head "nope" dim (shared between Q and K)
    d_rope: int     # per-head RoPE dim (shared between Q and K)
    d_v: int        # per-head V dim
    d_c: int        # latent rank (what gets cached; unused in MVP kernel)

    @property
    def qk_head_dim(self) -> int:
        return self.d_h + self.d_rope

    @property
    def q_size(self) -> int:
        return self.batch * self.seq * self.n_h * self.qk_head_dim

    @property
    def k_size(self) -> int:
        return self.batch * self.seq * self.n_h * self.qk_head_dim

    @property
    def v_size(self) -> int:
        return self.batch * self.seq * self.n_h * self.d_v

    def assert_valid(self) -> None:
        assert self.d_h > 0 and self.d_rope > 0 and self.d_v > 0
        assert self.n_h > 0 and self.seq > 0 and self.batch > 0
        assert self.d_c > 0


# Small correctness shape — fast numerical check. Small n_h so the reference
# CPU-naive attention finishes instantly. Keep the (d_h, d_rope, d_v) ratios
# representative so the padding cost is exercised on a tiny shape.
SHAPE_CORRECTNESS_MLA = MLAShape(
    name="correctness_mla",
    batch=1,
    seq=128,
    n_h=4,
    d_h=64,
    d_rope=32,   # qk_head_dim = 96
    d_v=64,
    d_c=128,
)

# DeepSeek-V3 canonical bench shape. n_h=128 is the real head count.
# At seq=2048 this is the canonical FlashMLA prefill benchmark size.
SHAPE_BENCH_MLA = MLAShape(
    name="deepseek_v3",
    batch=1,
    seq=2048,
    n_h=128,
    d_h=128,
    d_rope=64,   # qk_head_dim = 192
    d_v=128,
    d_c=512,
)


def all_shapes() -> list[MLAShape]:
    return [SHAPE_CORRECTNESS_MLA, SHAPE_BENCH_MLA]


if __name__ == "__main__":
    for s in all_shapes():
        s.assert_valid()
        print(
            f"{s.name:<18} batch={s.batch} seq={s.seq:<5} "
            f"n_h={s.n_h:<4} d_h={s.d_h:<4} d_rope={s.d_rope:<3} "
            f"qk={s.qk_head_dim:<4} d_v={s.d_v:<4} d_c={s.d_c:<4} "
            f"Q={s.q_size * 2 / 1e6:.1f}MB(f16) "
            f"K={s.k_size * 2 / 1e6:.1f}MB(f16) "
            f"V={s.v_size * 2 / 1e6:.1f}MB(f16)"
        )
