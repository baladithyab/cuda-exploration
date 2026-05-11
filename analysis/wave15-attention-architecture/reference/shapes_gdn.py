"""Wave 16 — canonical GDN (Gated DeltaNet) shape definitions.

GDN, introduced in Yang et al. "Gated Delta Networks" (ICLR 2025,
arXiv 2412.06464), maintains a per-head recurrent state matrix
`S ∈ R^{d_k × d_v}` updated per token via a gated delta rule.

Used in Qwen3-Next / Qwen3.5 as the dominant layer type of a 3:1
GDN:full-attention hybrid. Per `fla-org/flash-linear-attention`
defaults: `head_dim = 256` (so `d_k = d_v = 256`).

Two shape sets here:
  - SHAPE_CORRECTNESS — small, fast PyTorch-reference comparison.
    Head dim 64 so the d_k × d_v state matrix is only 64 × 64 f32.
  - SHAPE_QWEN3_NEXT_DECODE — canonical decode-bench shape. One
    timestep at a time; what each kernel invocation processes.

We target only the **decode** regime: a single timestep.  The state
recurrence for prefill over many timesteps has a serial dependency,
so prefill is done with the chunkwise-parallel algorithm which is
out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GDNShape:
    """Gated-DeltaNet decode shape parameters.

    Decode processes ONE timestep per kernel invocation for a batch of
    independent sequences. Each (batch, head) owns its own state tile
    S of shape (d_k, d_v), which is read from HBM at the start of the
    kernel and written back at the end.

    Fields:
        name    identifier used in filenames and logs
        batch   number of independent sequences processed together
        n_heads number of GDN heads (each has its own state tile)
        d_k     key head dimension (also the first dim of S)
        d_v     value head dimension (also the second dim of S)

    Note: d_k and d_v are held independent here (Qwen3-Next sets both
    to 256). FLA's fused_recurrent kernel also allows d_k != d_v.
    """

    name: str
    batch: int
    n_heads: int
    d_k: int
    d_v: int

    @property
    def state_elems(self) -> int:
        """Number of f32 scalars in the state tensor (one per head, per batch)."""
        return self.batch * self.n_heads * self.d_k * self.d_v

    @property
    def state_bytes_f32(self) -> int:
        return self.state_elems * 4

    @property
    def q_elems(self) -> int:
        return self.batch * self.n_heads * self.d_k

    @property
    def k_elems(self) -> int:
        return self.batch * self.n_heads * self.d_k

    @property
    def v_elems(self) -> int:
        return self.batch * self.n_heads * self.d_v

    @property
    def o_elems(self) -> int:
        return self.batch * self.n_heads * self.d_v

    def assert_valid(self) -> None:
        assert self.d_k > 0 and self.d_v > 0
        assert self.n_heads > 0 and self.batch > 0


# Fast numerical-correctness shape. Small state (64×64 = 16KB f32 per head)
# so full PyTorch reference runs in <10 ms.
SHAPE_CORRECTNESS = GDNShape(
    name="correctness",
    batch=2,
    n_heads=4,
    d_k=64,
    d_v=64,
)

# Qwen3-Next canonical decode shape. Per HF blog + FLA defaults:
#   num_v_heads = 16 (Qwen3-Next-80B-A3B-Base config)
#   head_dim   = 256
# Batch 1 is the worst case for decode bandwidth (full state round-trip
# every token at batch-1 is what makes GDN decode memory-bound).
SHAPE_QWEN3_NEXT_DECODE = GDNShape(
    name="qwen3_next_decode",
    batch=1,
    n_heads=16,
    d_k=256,
    d_v=256,
)


def all_shapes() -> list[GDNShape]:
    return [SHAPE_CORRECTNESS, SHAPE_QWEN3_NEXT_DECODE]


if __name__ == "__main__":
    for s in all_shapes():
        s.assert_valid()
        print(
            f"{s.name:<24} B={s.batch} H={s.n_heads:<3} "
            f"d_k={s.d_k:<4} d_v={s.d_v:<4} "
            f"state(f32)={s.state_bytes_f32 / 1024:.1f}KB"
        )
