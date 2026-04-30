"""Shared benchmark/profile utilities for the MLX rewrite track.

Principles (plan.md §"MLX-specific execution rule"):

* MLX is lazy — every timed region must call ``mx.eval(*outputs)`` so the
  measurement reflects GPU work, not graph construction.
* Warmup iterations prime kernel compilation / autotuning caches before the
  measured run.
* Results are reported as ``{median_ms, mean_ms, p90_ms, n_iters}`` dicts
  so callers can pivot easily.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import mlx.core as mx
import numpy as np

# Make tests helpers importable (shared case construction).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _helpers import make_inputs, make_varlen_inputs  # noqa: E402


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    name: str
    kind: str  # "fixed" | "varlen"
    T: int
    H: int
    seq_lens: tuple[int, ...] | None = None
    has_state: bool = False


DEFAULT_CASES: tuple[Case, ...] = (
    Case("fixed_T64_H1",          "fixed", T=64,   H=1),
    Case("fixed_T256_H1",         "fixed", T=256,  H=1),
    Case("fixed_T256_H4",         "fixed", T=256,  H=4),
    Case("fixed_T1024_H4",        "fixed", T=1024, H=4),
    Case("fixed_T1024_H4_state",  "fixed", T=1024, H=4, has_state=True),
    Case("varlen_mixed_H4",       "varlen", T=0, H=4,
         seq_lens=(37, 16, 97, 64)),
    # Cross-sequence packed varlen targets: larger N benefits more from
    # amortising the pre-compute across the sequence axis.
    Case("varlen_N8_mixed_H4",    "varlen", T=0, H=4,
         seq_lens=(37, 16, 97, 64, 128, 17, 256, 80)),
    Case("varlen_N16_T64_H4",     "varlen", T=0, H=4,
         seq_lens=tuple([64] * 16)),
)


STRESS_CASES: tuple[Case, ...] = (
    Case("fixed_T4096_H8",        "fixed", T=4096, H=8),
    Case("fixed_T4096_H8_state",  "fixed", T=4096, H=8, has_state=True),
)


# Benchmark-scale cases mirroring generate_benchmark_mlx_md.py so
# profilers can run section timings on the same shapes reported in
# BENCHMARK_MLX.md. These are opt-in (not in DEFAULT_CASES / STRESS_CASES)
# — they're expensive and only needed for targeted Phase 2 profiling.
BENCH_CASES: tuple[Case, ...] = (
    Case("bench_fixed_T8192_H64_state",  "fixed", T=8192, H=64, has_state=True),
    Case("bench_fixed_T8192_H96_state",  "fixed", T=8192, H=96, has_state=True),
    Case("bench_varlen_mixed_H64_state", "varlen", T=0, H=64,
         seq_lens=(1300, 547, 2048, 963, 271, 3063),
         has_state=True),
    Case("bench_varlen_mixed_H96_state", "varlen", T=0, H=96,
         seq_lens=(1300, 547, 2048, 963, 271, 3063),
         has_state=True),
    Case("bench_varlen_uniform_H64_state", "varlen", T=0, H=64,
         seq_lens=tuple([1024] * 8), has_state=True),
    Case("bench_varlen_uniform_H96_state", "varlen", T=0, H=96,
         seq_lens=tuple([1024] * 8), has_state=True),
)


# Lookup table for --cases selection on the bench driver.
CASE_BY_NAME: dict[str, Case] = {
    c.name: c for c in (*DEFAULT_CASES, *STRESS_CASES, *BENCH_CASES)
}


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def build_call_kwargs(
    case: Case,
    seed: int = 0,
    *,
    cuda_correspond: bool = True,
) -> dict[str, Any]:
    """Construct a kwargs dict suitable for ``flash_kda_mlx.fwd``.

    Note: varlen cases set ``case.T = 0`` as a placeholder — the real total
    token count comes from ``sum(case.seq_lens)`` computed here, and the
    output placeholder is sized accordingly.

    Args:
        case: the benchmark case (fixed or varlen).
        seed: base seed for deterministic input construction.
        cuda_correspond: when True (default), construct inputs matching
            ``benchmarks/bench_fwd.py`` — ``q/k/v/g/beta`` in bf16, ``out``
            as bf16 zeros, ``initial_state`` as bf16 ``arange`` reshaped to
            ``[N, H, D, D]``, ``final_state`` as bf16 zeros, and an
            ``initial_state_fp32`` companion for FLA-style baselines that
            CUDA bench feeds via ``initial_state.float()``. Also adds an
            independent deterministic scalar GDN gate ``g_gdn`` of shape
            ``[1, T_total, H]`` drawn from ``mx.random.normal`` with a
            seed stream distinct from ``g``.

            When False, preserves the legacy fp32 convenience path: all
            inputs in fp32, ``out`` as fp32 zeros, and ``initial_state``
            from ``standard_normal * 0.1``. No ``g_gdn`` or
            ``initial_state_fp32`` keys are added; callers that need a
            GDN gate must synthesize one themselves. This mode is useful
            for pure MLX-only development where CUDA correspondence is
            not required.
    """
    if case.kind == "fixed":
        inputs = make_inputs(
            T=case.T, H=case.H, D=128, seed=seed,
            dtype=mx.bfloat16 if cuda_correspond else mx.float32,
        )
        T_total = case.T
        N = 1
    elif case.kind == "varlen":
        assert case.seq_lens is not None, "varlen case must define seq_lens"
        inputs = make_varlen_inputs(
            list(case.seq_lens), H=case.H, D=128, seed=seed,
            dtype=mx.bfloat16 if cuda_correspond else mx.float32,
        )
        T_total = sum(case.seq_lens)
        N = len(case.seq_lens)
    else:
        raise ValueError(f"unknown case kind: {case.kind}")

    state_dtype = mx.bfloat16 if cuda_correspond else mx.float32
    out_dtype = mx.bfloat16 if cuda_correspond else mx.float32

    kwargs: dict[str, Any] = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T_total, case.H, 128), dtype=out_dtype),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
    }
    if case.has_state:
        if cuda_correspond:
            # CUDA bench: arange(N*H*D*D, fp32).reshape(N,H,D,D).to(bf16).
            init = (
                mx.arange(N * case.H * 128 * 128, dtype=mx.float32)
                .reshape(N, case.H, 128, 128)
                .astype(mx.bfloat16)
            )
            kwargs["initial_state"] = init
            kwargs["final_state"] = mx.zeros(
                (N, case.H, 128, 128), dtype=state_dtype
            )
            # Companion for FLA-style baselines (CUDA uses ``initial_state.float()``).
            kwargs["initial_state_fp32"] = init.astype(mx.float32)
        else:
            rng = np.random.default_rng(seed + 12345)
            init_np = rng.standard_normal(
                (N, case.H, 128, 128)
            ).astype(np.float32) * 0.1
            kwargs["initial_state"] = mx.array(init_np, dtype=mx.float32)
            kwargs["final_state"] = mx.zeros(
                (N, case.H, 128, 128), dtype=mx.float32
            )
    if case.kind == "varlen":
        kwargs["cu_seqlens"] = inputs["cu_seqlens"]

    if cuda_correspond:
        # Independent deterministic scalar-per-head GDN gate, drawn from a
        # distinct key so it does not perturb the q/k/v/g seed stream.
        # CUDA bench: ``torch.randn((1, T_total, H), dtype=torch.float32)``.
        gdn_key = mx.random.key(_derive_gdn_seed(seed))
        kwargs["g_gdn"] = mx.random.normal(
            shape=(1, T_total, case.H), dtype=mx.float32, key=gdn_key,
        )

    mx.eval(*[v for v in kwargs.values() if isinstance(v, mx.array)])
    return kwargs


# Disjoint from the ``seed + 12345`` offset used for legacy state construction,
# and from the input seed itself, so g_gdn is provably uncorrelated with g.
_GDN_SEED_OFFSET = 0x6D4E_0001


def _derive_gdn_seed(seed: int) -> int:
    return (int(seed) + _GDN_SEED_OFFSET) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_fn(
    fn: Callable[[], tuple[mx.array, mx.array | None]],
    *,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> dict[str, float | int]:
    """Time ``fn`` over ``n_iters`` runs after ``n_warmup`` warmup calls.

    ``fn`` must return an MLX array (or tuple of MLX arrays); we call
    ``mx.eval`` inside the timed region so the measurement reflects real
    GPU work, not lazy-graph construction.
    """
    for _ in range(n_warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*[x for x in out if isinstance(x, mx.array)])
        else:
            mx.eval(out)

    samples: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*[x for x in out if isinstance(x, mx.array)])
        else:
            mx.eval(out)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": samples[int(math.ceil(0.9 * len(samples))) - 1],
        "min_ms": samples[0],
        "n_iters": n_iters,
    }


def format_result_row(case_name: str, backend: str, stats: dict) -> str:
    return (
        f"| {case_name:<26s} | {backend:<11s} | "
        f"{stats['median_ms']:>9.2f} | "
        f"{stats['mean_ms']:>9.2f} | "
        f"{stats['p90_ms']:>9.2f} | "
        f"{stats['min_ms']:>9.2f} |"
    )


TABLE_HEADER = (
    "| Case                       | Backend     | median_ms | mean_ms   | p90_ms    | min_ms    |\n"
    "|----------------------------|-------------|-----------|-----------|-----------|-----------|"
)


# ---------------------------------------------------------------------------
# Metal trace capture
# ---------------------------------------------------------------------------

def capture_metal_trace(
    fn: Callable[[], Any],
    trace_path: str | Path,
    *,
    n_warmup: int = 2,
) -> Path:
    """Capture an MLX GPU trace to ``trace_path`` (``.gputrace`` bundle).

    Open with Instruments on macOS or with Xcode's GPU debugger. Warmup runs
    outside the capture region so JIT/compilation noise doesn't pollute the
    trace.
    """
    for _ in range(n_warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*[x for x in out if isinstance(x, mx.array)])
        else:
            mx.eval(out)

    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        # mx.metal.start_capture refuses to overwrite; rotate.
        import shutil
        shutil.rmtree(trace_path, ignore_errors=True)

    try:
        mx.metal.start_capture(str(trace_path))
    except RuntimeError as exc:
        msg = str(exc)
        if "Capture layer is not inserted" in msg:
            raise RuntimeError(
                "Metal capture layer not inserted. Re-run with MTL_CAPTURE_ENABLED=1 set:\n"
                "  MTL_CAPTURE_ENABLED=1 uv run --no-config python -m benchmarks.profile_fwd ..."
            ) from exc
        raise
    try:
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*[x for x in out if isinstance(x, mx.array)])
        else:
            mx.eval(out)
    finally:
        mx.metal.stop_capture()
    return trace_path


__all__ = [
    "Case", "DEFAULT_CASES", "STRESS_CASES", "BENCH_CASES", "CASE_BY_NAME",
    "build_call_kwargs", "time_fn",
    "format_result_row", "TABLE_HEADER",
    "capture_metal_trace",
]
