"""Wave 15 — per-dtype correctness tolerances (atol, rtol).

These thresholds were calibrated against PyTorch's
`scaled_dot_product_attention` reference at SHAPE_CORRECTNESS.
Anything tighter starts producing false positives from softmax FP
sensitivity; anything looser doesn't catch real bugs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


TOLERANCES: dict[str, Tolerance] = {
    "f32": Tolerance(atol=1e-5, rtol=1e-5),
    "tf32": Tolerance(atol=1e-3, rtol=1e-3),  # tf32 inputs reduce mantissa precision
    "bf16": Tolerance(atol=1e-2, rtol=1e-2),  # bf16 has only 7 mantissa bits
    "f16": Tolerance(atol=5e-3, rtol=5e-3),  # f16 has 10 mantissa bits but smaller range
}


def get(dtype: str) -> Tolerance:
    return TOLERANCES[dtype]
