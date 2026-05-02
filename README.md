# FlashKDA-mlx

Standalone MLX / Metal implementation, tests, and benchmark reproduction
harness for FlashKDA-style Kimi Delta Attention on Apple Silicon.

This repository is intentionally independent from the original CUDA/CUTLASS
FlashKDA source tree ([MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)).
It contains only the MLX package, MLX benchmark scripts, torch-reference
parity fixtures, and MLX-side documentation.

## Headline numbers (Apple M3 Max, MLX 0.31.2)

`flash_kda_mlx` beats both MLX-LM-backed baselines on every bench-scale
row at `T=8192`, `D=128`:

| Heads | Case | `flash_kda_mlx` (ms) | vs `mlx_chunk_kda` | vs `mlx_chunk_gdn` |
|---:|---|---:|:---:|:---:|
| 96 | Fixed | 49.8 | **4.86×** | **2.80×** |
| 96 | Varlen, mixed `[1300, 547, 2048, 963, 271, 3063]` | 52.5 | **2.81×** | **1.68×** |
| 96 | Varlen, uniform `1024 × 8` | 44.7 | **2.41×** | **1.69×** |
| 64 | Fixed | 31.5 | **4.32×** | **2.29×** |
| 64 | Varlen, mixed | 34.1 | **2.44×** | **1.67×** |
| 64 | Varlen, uniform | 31.1 | **2.20×** | **1.69×** |

Cumulative speedup vs the no-Metal `mx.compile`-only baseline:
**5.7–6.4×** at bench scale. See `BENCHMARK_MLX.md` for the full
report (provenance, env flags, variance band) and `STATUS.md` for the
shipped optimization track.

> Apple GPU MLX timings are **not** numerically comparable to the CUDA/H20
> table in `BENCHMARK_H20.md` — column semantics correspond, absolute
> numbers do not.

## Contents

- `flash_kda_mlx/`: public MLX operators, reference path, optimized path, and Metal kernels.
- `flash_kda_mlx/baselines/`: MLX-LM-backed `chunk_kda` and `chunk_gdn` baseline adapters.
- `tests/`: torch-reference parity, API, Metal-kernel, and benchmark harness tests.
- `benchmarks/`: MLX benchmark runners and report generator.
- `scripts/`: torch oracle and fixture generation for MLX parity tests.
- `BENCHMARK_MLX.md`: current Apple GPU benchmark report using the H20-shaped cases.
- `docs/20260501-flashkda-mlx-v1-deep-dive.md`: design deep dive — chunk-size choice, two-kernel decomposition (prepare + cross-chunk recurrence), numerical-precision contract, and accuracy comparison vs. `mlx_chunk_kda`.
- `plan.md`: consolidated TDD migration, optimization, and benchmark-reproduction plan.
- `STATUS.md`: as-shipped state of the plan (current PR/phase status, measured tolerances, current benchmark numbers).

## Setup

```bash
uv sync
```

The default environment installs MLX, MLX-LM, NumPy, PyTorch, and pytest.
PyTorch is used only for local CPU oracle tests and fixture generation; CUDA
is not required.

## Test

```bash
uv run pytest
```

Useful focused subsets:

```bash
uv run pytest tests/test_parity_fixtures.py
uv run pytest tests/test_chunk_baseline_torch_reference.py
uv run pytest tests/test_optimized_parity.py
```

## Benchmark

Current production MLX benchmark flags:

```bash
MLX_KDA_ENABLE_METAL_PREPARE=fused4 \
MLX_KDA_ENABLE_METAL_RECURRENCE=1 \
uv run python benchmarks/generate_benchmark_mlx_md.py \
  --output BENCHMARK_MLX.md
```

The report uses the same H20-shaped cases as the original CUDA report for
column-role correspondence, but it is an MLX-on-Apple-GPU benchmark. It should
be read as an MLX-local performance comparison, not as a CUDA-device timing
claim.

## Parity Policy

- `flash_kda_mlx` is validated against checked-in torch-reference KDA fixtures.
- `mlx_chunk_kda` and `mlx_chunk_gdn` are validated against small local
  PyTorch gated-delta reference cases.
- CUDA device fixtures are not required for this repository's production gate.
