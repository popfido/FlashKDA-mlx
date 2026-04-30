# KDA forward benchmark (MLX / Apple GPU)

- Generated: 2026-04-30

- Command: `MLX_KDA_ENABLE_METAL_PREPARE=fused4 MLX_KDA_ENABLE_METAL_RECURRENCE=1 uv run --no-config python benchmarks/generate_benchmark_mlx_md.py --output BENCHMARK_MLX.md --strict-equivalence`

- Benchmark settings: `warmup=3`, `iters=10`, `repeats=1`

> **Hardware caveat.** MLX timings on Apple GPU are not numerically comparable to the CUDA/H20 table in `BENCHMARK_H20.md`. Column semantics mirror the CUDA report; absolute numbers do not.

- MLX configuration: `backend=optimized`, `CHUNK=16`, `lower_bound=-5`, `D=128`. Timed regions force execution with `mx.eval(...)`.
- Metal recurrence: `MLX_KDA_ENABLE_METAL_RECURRENCE=1`, mode `simdgroup`, enabled=True, has_kernel=True, cross_chunk_active=True, packed_active=True, flat_ragged_active=True.
- Metal prepare: `MLX_KDA_ENABLE_METAL_PREPARE=fused4`, mode `fused4`, enabled=True, has_kernel=True, prepare_kernel_loaded=True, fused_active=True.
- Input construction (default `cuda_correspond=True`) matches `benchmarks/bench_fwd.py`: `q/k/v/g/beta` in bf16, `out` as bf16 zeros, `initial_state = arange(N*H*D*D).reshape(N,H,D,D).to(bf16)`, `final_state` as matching bf16 zeros. FLA baselines receive `initial_state.astype(fp32)` at the call site, mirroring CUDA's `initial_state.float()`.
- `mlx_chunk_kda` / `mlx_chunk_gdn` columns wrap the MLX-LM gated-delta backbone with FLA-matching kwargs via `flash_kda_mlx.baselines.chunk_kda.chunk_kda_mlx` and `flash_kda_mlx.baselines.chunk_gdn.chunk_gdn_mlx`. `use_kernel=True` (Metal path) with `transpose_state_layout=True` and `use_qk_l2norm_in_kernel=True`.
- Varlen uses the fused4 flat-ragged path for `flash_kda_mlx`: metadata-indirected token-major prepare plus one flat-ragged recurrence/direct-write-output dispatch across sequences. Baseline adapters still use per-sequence unpacked execution.
- GDN gate construction: `g_gdn = mx.random.normal((1, T_total, H), dtype=fp32)` drawn from a key distinct from the `g`/`q`/`k`/`v` seed stream — an independent deterministic scalar per-head gate, mirroring CUDA bench's `torch.randn((1, T_total, H), dtype=torch.float32)`.

> **Torch-reference parity.** `flash_kda_mlx` is validated against the checked-in KDA fixtures in `tests/test_parity_fixtures.py`. `mlx_chunk_kda` and `mlx_chunk_gdn` are validated against small fixed/varlen PyTorch gated-delta reference cases in `tests/test_chunk_baseline_torch_reference.py`. CUDA device fixtures are not required for this MLX-side benchmark gate.

> **Speedup convention.** Columns `Speedup vs chunk_kda` and `Speedup vs gdn` use `baseline_mean / flash_kda_mlx_mean`, matching `BENCHMARK_H20.md`. A value > 1 means `flash_kda_mlx` is faster than the baseline; < 1 means slower. These ratios depend on the Metal recurrence/prepare switches recorded above.

## Benchmark Column Roles

| Original `BENCHMARK_H20.md` column | MLX equivalent | Status |
|---|---|---|
| `flash_kda` mean (ms) | `flash_kda_mlx.fwd(..., backend="optimized")`, `CHUNK=16` | Validated against the torch-reference KDA fixtures. |
| `fla_chunk_kda` mean (ms) | `flash_kda_mlx.baselines.chunk_kda.chunk_kda_mlx` (MLX-LM backbone, FLA-matching kwargs) | Validated against local torch-reference fixed/varlen cases. |
| Speedup vs `chunk_kda` | `mlx_chunk_kda_mean / flash_kda_mlx_mean` | Reported; apples-to-apples across MLX methods but not cross-framework validated. |
| `fla_chunk_gdn` mean (ms) | `flash_kda_mlx.baselines.chunk_gdn.chunk_gdn_mlx` (MLX-LM backbone, FLA-matching kwargs) | Validated against local torch-reference fixed/varlen cases. |
| Speedup vs `gdn` | `mlx_chunk_gdn_mean / flash_kda_mlx_mean` | Reported; same caveat as above. |

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda_mlx` mean (ms) | `mlx_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `mlx_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|--------------------:|--------------------------:|-----------------------:|--------------------------:|-----------------:|
| Fixed | 50.2945 | 242.1577 | 4.81× | 142.2082 | 2.83× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 53.7382 | 146.9932 | 2.74× | 91.5073 | 1.70× |
| Varlen, `seq_lens`=`1024 x 8` | 50.4482 | 117.4850 | 2.33× | 80.7292 | 1.60× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda_mlx` mean (ms) | `mlx_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `mlx_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|--------------------:|--------------------------:|-----------------------:|--------------------------:|-----------------:|
| Fixed | 34.3557 | 163.2694 | 4.75× | 88.7781 | 2.58× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 40.1443 | 92.9966 | 2.32× | 61.5364 | 1.53× |
| Varlen, `seq_lens`=`1024 x 8` | 35.4228 | 77.8676 | 2.20× | 58.5038 | 1.65× |
