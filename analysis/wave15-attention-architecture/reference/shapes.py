"""Wave 15 — canonical attention shape definitions.

Two shape sets used everywhere in the wave-15 attention bench:

  - SHAPE_CORRECTNESS — small, used for fast numerical correctness
    checks against PyTorch reference. Should run in <100 ms.
  - SHAPE_BENCH       — Llama-3-8B-style, used for performance numbers.

Each shape set has both GQA-specific values (n_q, n_kv) and the
shared (batch, seq, d_head) values. Other mechanisms (MLA, GDN, …)
get their own shape sets in their own modules but reuse the same
batch and seq from here so the FLOPS comparison stays apples-to-
apples for sequence-length-dependent costs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GQAShape:
    """Grouped-Query Attention shape parameters.

    Q has `n_q` heads. K and V each have `n_kv` heads (`n_kv ≤ n_q`).
    Each Q head shares its KV with `n_q // n_kv` other Q heads in its
    group. d_head is the per-head dimension for both Q and K (typical
    GQA keeps `d_q == d_k == d_v`).
    """

    name: str
    batch: int
    seq: int
    n_q: int
    n_kv: int
    d_head: int

    @property
    def n_groups(self) -> int:
        return self.n_q // self.n_kv

    @property
    def q_size(self) -> int:
        return self.batch * self.seq * self.n_q * self.d_head

    @property
    def kv_size(self) -> int:
        return self.batch * self.seq * self.n_kv * self.d_head

    def assert_valid(self) -> None:
        assert self.n_q % self.n_kv == 0, (
            f"{self.name}: n_q ({self.n_q}) must be divisible by n_kv ({self.n_kv})"
        )


# Fast numerical-correctness shape. Pytorch reference + cell impl
# should both finish in well under 1 second.
SHAPE_CORRECTNESS = GQAShape(
    name="correctness",
    batch=1,
    seq=128,
    n_q=4,
    n_kv=2,
    d_head=64,
)

# Llama-3-8B canonical bench shape.
SHAPE_BENCH = GQAShape(
    name="llama3_8b",
    batch=1,
    seq=2048,
    n_q=32,
    n_kv=8,
    d_head=128,
)


def all_shapes() -> list[GQAShape]:
    return [SHAPE_CORRECTNESS, SHAPE_BENCH]


if __name__ == "__main__":
    for s in all_shapes():
        s.assert_valid()
        print(
            f"{s.name:<14} batch={s.batch} seq={s.seq:<5} "
            f"n_q={s.n_q:<3} n_kv={s.n_kv:<3} d_head={s.d_head:<4} "
            f"groups={s.n_groups} "
            f"Q={s.q_size * 2 / 1e6:.1f}MB(f16) "
            f"KV={s.kv_size * 2 / 1e6:.1f}MB(f16) ea"
        )
