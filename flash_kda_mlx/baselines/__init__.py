"""FLA-shaped baseline adapters for the MLX benchmark report.

Each adapter wraps an MLX implementation (native MLX or via ``mlx-lm``)
with the same kwarg surface the CUDA benchmark uses against FLA (see
``benchmarks/bench_fwd.py``). The goal is column-for-column comparability
in the MLX-side report. Correctness is validated by the local
torch-reference gated-delta parity in
``tests/test_chunk_baseline_torch_reference.py``; CUDA-host FLA fixture
parity is not pursued because the MLX side has no CUDA device.
"""
