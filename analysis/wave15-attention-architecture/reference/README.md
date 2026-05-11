# Wave 15 attention shared infra

This directory contains the reference implementation and shared
infrastructure for the Wave 15 attention-kernel benchmarking effort.
Every cell (e.g. `cutile-attn-gqa/`, `cuda-attn-gqa/`,
`cublas-attn-gqa/`) consumes from here and produces results that get
compared against this reference.

## Layout

```
analysis/wave15-attention-architecture/
├── PLAN.md                          # the architecture proposal (read this first)
├── reference/
│   ├── shapes.py                    # canonical GQAShape dataclass + 2 shape sets
│   ├── tensors.py                   # NOT YET — for now use pytorch_reference.py
│   ├── tolerances.py                # per-dtype atol/rtol thresholds
│   ├── flops.py                     # single source of truth for FLOPS calculation
│   ├── pytorch_reference.py         # PyTorch SDPA + naive GQA; correctness oracle
│   └── reference_run.log            # captured stdout of running pytorch_reference.py
└── inputs/                          # .gitignore'd — regenerate via pytorch_reference.py
    ├── .gitignore
    ├── gqa_correctness_q_f32.npy    # (1, 4, 128, 64)
    ├── gqa_correctness_k_f32.npy    # (1, 2, 128, 64)
    ├── gqa_correctness_v_f32.npy    # (1, 2, 128, 64)
    ├── gqa_correctness_{q,k,v}_f16.npy
    ├── gqa_correctness_expected_f32.npy
    ├── gqa_llama3_8b_q_f32.npy      # (1, 32, 2048, 128)  — 32 MB
    └── ... f16 variants and expected output
```

## Regenerating inputs

```bash
cd analysis/wave15-attention-architecture/reference
/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python pytorch_reference.py
```

This is deterministic (seed = 0xC0FFEE). Same inputs every time.

## Cross-cell correctness contract

Every cell's bench harness must:

1. Load `gqa_<shape>_q_*.npy`, `_k_*.npy`, `_v_*.npy` from `inputs/`
2. Run its kernel
3. Compare against `gqa_<shape>_expected_f32.npy` using
   `tolerances.get(dtype)`
4. Print `OK` or `FAIL: max_abs_err=... rel_err=...`

Any cell that doesn't pass at SHAPE_CORRECTNESS does NOT report bench
numbers — broken kernels don't get published TFLOPS.

## Canonical shapes

Defined in `reference/shapes.py`:

| name          | batch | seq  | n_q  | n_kv | d_head | groups | purpose      |
|---------------|------:|-----:|-----:|-----:|-------:|-------:|--------------|
| correctness   | 1     | 128  | 4    | 2    | 64     | 2      | numerical    |
| llama3_8b     | 1     | 2048 | 32   | 8    | 128    | 4      | bench        |

## FLOPS model

Defined in `reference/flops.py` — `gqa_attention_flops(shape)`. **Use
this everywhere**; do NOT recompute the formula in each cell. Llama-3-8B
canonical bench shape: 68.72 GFLOPS per forward pass, AI=1638 flops/byte
at f16 → strongly compute-bound (RTX 5090 f16 ridge is ~62 flops/byte).
Expected ceiling: ~218 TFLOPS (cuBLAS hgemm peak from Wave 14.1).

## Tolerances

`reference/tolerances.py`:

- f32:  atol=1e-5, rtol=1e-5
- tf32: atol=1e-3, rtol=1e-3
- bf16: atol=1e-2, rtol=1e-2
- f16:  atol=5e-3, rtol=5e-3

These match Wave 13's mixed-precision-matmul tolerances; calibrated
against PyTorch SDPA at SHAPE_CORRECTNESS.
