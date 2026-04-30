# FlashKDA-mlx

Standalone MLX / Metal implementation, tests, and benchmark reproduction
harness for FlashKDA-style Kimi Delta Attention on Apple Silicon.

This repository is intentionally independent from the original CUDA/CUTLASS
FlashKDA source tree. It contains only the MLX package, MLX benchmark scripts,
torch-reference parity fixtures, and MLX-side documentation.

## Contents

- `flash_kda_mlx/`: public MLX operators, reference path, optimized path, and Metal kernels.
- `flash_kda_mlx/baselines/`: MLX-LM-backed `chunk_kda` and `chunk_gdn` baseline adapters.
- `tests/`: torch-reference parity, API, Metal-kernel, and benchmark harness tests.
- `benchmarks/`: MLX benchmark runners and report generator.
- `scripts/`: torch oracle and fixture generation for MLX parity tests.
- `BENCHMARK_MLX.md`: current Apple GPU benchmark report using the H20-shaped cases.
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
uv run --no-config pytest
```

Useful focused subsets:

```bash
uv run --no-config pytest tests/test_parity_fixtures.py
uv run --no-config pytest tests/test_chunk_baseline_torch_reference.py
uv run --no-config pytest tests/test_optimized_parity.py
```

## Benchmark

Current production MLX benchmark flags:

```bash
MLX_KDA_ENABLE_METAL_PREPARE=fused4 \
MLX_KDA_ENABLE_METAL_RECURRENCE=1 \
uv run --no-config python benchmarks/generate_benchmark_mlx_md.py \
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
