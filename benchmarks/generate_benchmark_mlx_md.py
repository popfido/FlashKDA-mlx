#!/usr/bin/env python3
"""Generate a CUDA-benchmark-corresponding table for the MLX implementation.

The CUDA report in ``BENCHMARK_H20.md`` compares ``flash_kda`` against FLA
``chunk_kda`` and ``chunk_gated_delta_rule`` at ``T=8192`` for ``H=96``
and ``H=64``. H20 is only the hardware used for that original CUDA
timing. This MLX report runs on Apple GPU and mirrors the CUDA table's
cases and method roles:

- ``flash_kda_mlx``       — ``flash_kda_mlx.fwd(..., backend="optimized")`` with CHUNK=16,
                      the MLX equivalent of CUDA ``flash_kda``.
- ``mlx_chunk_kda`` — ``flash_kda_mlx.baselines.chunk_kda.chunk_kda_mlx``,
                      the MLX equivalent of FLA ``chunk_kda``.
- ``mlx_chunk_gdn`` — ``flash_kda_mlx.baselines.chunk_gdn.chunk_gdn_mlx``,
                      the MLX equivalent of FLA ``chunk_gated_delta_rule``.

Speedup columns mirror the CUDA report: ``mlx_chunk_kda / flash_kda_mlx`` and
``mlx_chunk_gdn / flash_kda_mlx``.

**Input construction.** Under the default ``cuda_correspond=True`` mode
in ``_harness.build_call_kwargs``, inputs match ``benchmarks/bench_fwd.py``:
``q/k/v/g/beta`` are bf16, ``out`` is bf16 zeros, ``initial_state`` is
``arange(N*H*D*D).reshape(N,H,D,D).to(bf16)`` and ``final_state`` is the
matching bf16 zeros. For FLA-style baselines (``chunk_kda_mlx`` /
``chunk_gdn_mlx``), ``initial_state`` is cast to fp32 at the call site —
mirroring CUDA bench's ``initial_state.float()``. The scalar per-head
GDN gate ``g_gdn`` of shape ``[1, T_total, H]`` is drawn independently
from ``mx.random.normal`` with a seed stream distinct from ``g``,
matching CUDA's independent ``torch.randn`` draw.

**Hardware note.** MLX timings on Apple GPU are not numerically
comparable to the CUDA/H20 table in ``BENCHMARK_H20.md``. Column
semantics mirror the CUDA report; absolute numbers do not.

**FLA torch-reference parity for the baseline columns is covered by small
local tests.**
``mlx_chunk_kda`` and ``mlx_chunk_gdn`` wrap the MLX-LM gated-delta
backbone with FLA-matching kwargs and are checked against a sequential
PyTorch gated-delta oracle in fixed and varlen cases. CUDA device fixture
parity is no longer a production gate.

All timed regions force MLX execution with ``mx.eval``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks._harness import Case, build_call_kwargs
from flash_kda_mlx import optimized


DEFAULT_OUT = REPO_ROOT / "BENCHMARK_MLX.md"
MISSING_MARK = "—"

FIXED_CASES = ([8192],)
VARLEN_CASES = (
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
)


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingStats:
    """Result of timing a callable.

    ``mean_ms`` / ``min_ms`` / ``max_ms`` / ``median_ms`` are ``nan`` when
    the call failed; ``error`` then carries a short reason string. This
    mirrors the policy in ``bench_chunk`` below: benchmark drivers fall
    through to ``MISSING_MARK`` rather than aborting the full run.
    """

    mean_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    samples: int
    error: str | None = None


_NAN_STATS = TimingStats(
    mean_ms=float("nan"),
    min_ms=float("nan"),
    max_ms=float("nan"),
    median_ms=float("nan"),
    samples=0,
    error=None,
)


def _eval_output(out: Any) -> None:
    """Force evaluation of MLX arrays returned by a timed callable."""
    if isinstance(out, tuple):
        mx.eval(*[x for x in out if isinstance(x, mx.array)])
    elif isinstance(out, mx.array):
        mx.eval(out)


def _collect_timings(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> TimingStats:
    """Time ``fn`` and return :class:`TimingStats`.

    Each call's result is forced to materialize with ``mx.eval`` inside
    the timed region so we measure actual GPU work, not lazy-graph
    construction. Exceptions are captured as ``TimingStats(..., error=...)``
    so the caller can record a "—" cell without aborting the whole run.
    """
    try:
        for _ in range(max(warmup, 0)):
            _eval_output(fn())

        samples: list[float] = []
        for _ in range(max(repeats, 1)):
            for _ in range(max(iters, 0)):
                t0 = time.perf_counter()
                _eval_output(fn())
                samples.append((time.perf_counter() - t0) * 1e3)
    except Exception as exc:  # noqa: BLE001 — benchmark driver policy
        return TimingStats(
            mean_ms=float("nan"),
            min_ms=float("nan"),
            max_ms=float("nan"),
            median_ms=float("nan"),
            samples=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    if not samples:
        return _NAN_STATS

    samples.sort()
    return TimingStats(
        mean_ms=statistics.fmean(samples),
        min_ms=samples[0],
        max_ms=samples[-1],
        median_ms=statistics.median(samples),
        samples=len(samples),
        error=None,
    )


# ---------------------------------------------------------------------------
# Method callables
# ---------------------------------------------------------------------------

def _call_flash_kda_mlx(kwargs: dict[str, Any], chunk: int) -> Any:
    return optimized.fwd_optimized(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g"],
        beta=kwargs["beta"],
        scale=kwargs["scale"],
        out_like=kwargs["out"],
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        initial_state=kwargs.get("initial_state"),
        final_state_like=kwargs.get("final_state"),
        cu_seqlens=kwargs.get("cu_seqlens"),
        _chunk=chunk,
    )


def _fla_initial_state(kwargs: dict[str, Any]) -> Any:
    """Return the fp32 initial state FLA-style baselines expect.

    CUDA ``benchmarks/bench_fwd.py`` feeds ``initial_state.float()`` to
    both ``chunk_kda`` and ``chunk_gated_delta_rule``; ``_harness.py``
    pre-casts this as ``initial_state_fp32`` under ``cuda_correspond=True``.
    When that key is absent (legacy fp32 convenience path), fall back to
    whatever ``initial_state`` is present.
    """
    return kwargs.get("initial_state_fp32", kwargs.get("initial_state"))


def _call_chunk_kda(kwargs: dict[str, Any], chunk_kda_fn: Callable) -> Any:
    return chunk_kda_fn(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g"],
        beta=kwargs["beta"],
        scale=kwargs["scale"],
        initial_state=_fla_initial_state(kwargs),
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        transpose_state_layout=True,
        cu_seqlens=kwargs.get("cu_seqlens"),
        use_kernel=True,
    )


def _call_chunk_gdn(kwargs: dict[str, Any], chunk_gdn_fn: Callable) -> Any:
    return chunk_gdn_fn(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g_gdn"],
        beta=kwargs["beta"],
        scale=kwargs["scale"],
        initial_state=_fla_initial_state(kwargs),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        cu_seqlens=kwargs.get("cu_seqlens"),
        use_kernel=True,
    )


def _try_import_chunk_kda() -> tuple[Callable | None, str | None]:
    try:
        from flash_kda_mlx.baselines.chunk_kda import chunk_kda_mlx
        return chunk_kda_mlx, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _try_import_chunk_gdn() -> tuple[Callable | None, str | None]:
    try:
        from flash_kda_mlx.baselines.chunk_gdn import chunk_gdn_mlx
        return chunk_gdn_mlx, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def _case_name(seq_lens: list[int]) -> str:
    if len(seq_lens) == 1:
        return "Fixed"
    if len(set(seq_lens)) == 1:
        return f"Varlen, `seq_lens`=`{seq_lens[0]} x {len(seq_lens)}`"
    return f"Varlen, `seq_lens`={list(seq_lens)}"


def _make_case(seq_lens: list[int], H: int) -> Case:
    if len(seq_lens) == 1:
        return Case(
            name=f"fixed_T{seq_lens[0]}_H{H}_state",
            kind="fixed",
            T=seq_lens[0],
            H=H,
            has_state=True,
        )
    return Case(
        name=f"varlen_T{sum(seq_lens)}_H{H}_N{len(seq_lens)}_state",
        kind="varlen",
        T=0,
        H=H,
        seq_lens=tuple(seq_lens),
        has_state=True,
    )


# ---------------------------------------------------------------------------
# Format helpers (pure, tested in tests/test_benchmark_formatting.py)
# ---------------------------------------------------------------------------

def _fmt_ms(x: float | int | None) -> str:
    if x is None:
        return MISSING_MARK
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return MISSING_MARK
    if math.isnan(xf):
        return MISSING_MARK
    return f"{xf:.4f}"


def _fmt_speedup(base_ms: float | int | None, target_ms: float | int | None) -> str:
    """Format ``base_ms / target_ms`` as ``X.XX×``.

    ``base_ms`` is the method we're comparing *against* (larger = slower);
    ``target_ms`` is the MLX path under test. A ratio > 1 means the target
    is faster. Returns ``MISSING_MARK`` for any nan / zero / ``None``.
    """
    if base_ms is None or target_ms is None:
        return MISSING_MARK
    try:
        base = float(base_ms)
        target = float(target_ms)
    except (TypeError, ValueError):
        return MISSING_MARK
    if math.isnan(base) or math.isnan(target) or base <= 0 or target <= 0:
        return MISSING_MARK
    return f"{base / target:.2f}×"


def _metal_recurrence_status() -> str:
    mode = getattr(optimized, "_METAL_MODE", "unknown")
    enabled = getattr(optimized, "_ENABLE_METAL_RECURRENCE", False)
    has_kernel = getattr(optimized, "_HAS_METAL_KERNEL", False)
    cross_active = optimized._metal_cross_chunk_active()
    packed_active = optimized._metal_cross_chunk_packed_active()
    flat_ragged_active_fn = getattr(
        optimized, "_metal_cross_chunk_flat_ragged_active", lambda: False,
    )
    env = os.environ.get("MLX_KDA_ENABLE_METAL_RECURRENCE", "0")
    return (
        f"`MLX_KDA_ENABLE_METAL_RECURRENCE={env}`, mode `{mode}`, "
        f"enabled={enabled}, has_kernel={has_kernel}, "
        f"cross_chunk_active={cross_active}, packed_active={packed_active}, "
        f"flat_ragged_active={flat_ragged_active_fn()}"
    )


def _metal_prepare_status() -> str:
    mode = getattr(optimized, "_METAL_PREPARE_MODE", "unknown")
    enabled = getattr(optimized, "_ENABLE_METAL_PREPARE", False)
    has_kernel = getattr(optimized, "_HAS_METAL_PREPARE", False)
    prepare_loaded_fn = getattr(optimized, "_metal_prepare_active", lambda: False)
    fused_active_fn = getattr(optimized, "_metal_prepare_fused_active", lambda: False)
    env = os.environ.get("MLX_KDA_ENABLE_METAL_PREPARE", "0")
    return (
        f"`MLX_KDA_ENABLE_METAL_PREPARE={env}`, mode `{mode}`, "
        f"enabled={enabled}, has_kernel={has_kernel}, "
        f"prepare_kernel_loaded={prepare_loaded_fn()}, "
        f"fused_active={fused_active_fn()}"
    )


def _varlen_strategy_note() -> str:
    mode = getattr(optimized, "_METAL_PREPARE_MODE", "unknown")
    flat_ragged_active_fn = getattr(
        optimized, "_metal_cross_chunk_flat_ragged_active", lambda: False,
    )
    if mode == "fused4":
        if flat_ragged_active_fn():
            return (
                "- Varlen uses the fused4 flat-ragged path for `flash_kda_mlx`: "
                "metadata-indirected token-major prepare plus one "
                "flat-ragged recurrence/direct-write-output dispatch across "
                "sequences. Baseline adapters still use per-sequence "
                "unpacked execution."
            )
        return (
            "- Varlen uses the fused4 flat-ragged prepare path for `flash_kda_mlx` "
            "with per-sequence recurrence fallback. Baseline adapters still "
            "use per-sequence unpacked execution."
        )
    return (
        "- Varlen uses the current packed strategy for `flash_kda_mlx` and "
        "per-sequence unpacked execution inside the baseline adapters. "
        "The mixed-varlen `H=96` row therefore has different memory "
        "pressure from the fixed and uniform-varlen rows."
    )


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------

@dataclass
class CaseRow:
    case: str
    T: int
    H: int
    D: int
    seq_lens: list[int]
    flash_kda_mlx: TimingStats | None = None
    mlx_chunk_kda: TimingStats | None = None
    mlx_chunk_gdn: TimingStats | None = None
    notes: dict[str, str] = field(default_factory=dict)


def run_case(
    seq_lens: list[int],
    H: int,
    args: argparse.Namespace,
    *,
    chunk_kda_fn: Callable | None,
    chunk_gdn_fn: Callable | None,
) -> CaseRow:
    case = _make_case(seq_lens, H)
    kwargs = build_call_kwargs(case, seed=args.seed)

    row = CaseRow(
        case=_case_name(seq_lens),
        T=sum(seq_lens),
        H=H,
        D=128,
        seq_lens=list(seq_lens),
    )

    # --- Primary: flash_kda_mlx (CHUNK=16) ---
    row.flash_kda_mlx = _collect_timings(
        lambda: _call_flash_kda_mlx(kwargs, 16),
        warmup=args.warmup, iters=args.iters, repeats=args.repeats,
    )
    _print_timing(row.case, H, "flash_kda_mlx CHUNK=16", row.flash_kda_mlx)

    # --- mlx_chunk_kda column ---
    if not args.skip_chunk_kda:
        if chunk_kda_fn is None:
            row.notes["mlx_chunk_kda"] = "import failed"
            row.mlx_chunk_kda = _NAN_STATS
        else:
            row.mlx_chunk_kda = _collect_timings(
                lambda: _call_chunk_kda(kwargs, chunk_kda_fn),
                warmup=args.warmup, iters=args.iters, repeats=args.repeats,
            )
            _print_timing(row.case, H, "mlx_chunk_kda", row.mlx_chunk_kda)

    # --- mlx_chunk_gdn column ---
    if not args.skip_gdn:
        if chunk_gdn_fn is None:
            row.notes["mlx_chunk_gdn"] = "import failed"
            row.mlx_chunk_gdn = _NAN_STATS
        else:
            row.mlx_chunk_gdn = _collect_timings(
                lambda: _call_chunk_gdn(kwargs, chunk_gdn_fn),
                warmup=args.warmup, iters=args.iters, repeats=args.repeats,
            )
            _print_timing(row.case, H, "mlx_chunk_gdn", row.mlx_chunk_gdn)

    # Release temporaries before the next big case.
    del kwargs
    mx.clear_cache()
    gc.collect()
    return row


def _print_timing(case: str, H: int, label: str, stats: TimingStats) -> None:
    if stats.error is not None:
        print(f"H={H:<3d} {case:<52s} {label:<22s} ERROR: {stats.error}")
        return
    if stats.samples == 0:
        print(f"H={H:<3d} {case:<52s} {label:<22s} (no samples)")
        return
    print(
        f"H={H:<3d} {case:<52s} {label:<22s} "
        f"mean={stats.mean_ms:.4f} ms "
        f"min={stats.min_ms:.4f} ms max={stats.max_ms:.4f} ms"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_primary_table(rows: list[CaseRow], H: int) -> list[str]:
    section_rows = [r for r in rows if r.H == H]
    if not section_rows:
        return []
    T_values = {r.T for r in section_rows}
    assert len(T_values) == 1, f"section rows have mixed T: {T_values}"
    T = T_values.pop()
    lines = [
        f"### `T={T}`, `H={H}`, `D=128`",
        "",
        (
            "| Case | `flash_kda_mlx` mean (ms) | `mlx_chunk_kda` mean (ms) | "
            "Speedup vs `chunk_kda` | `mlx_chunk_gdn` mean (ms) | "
            "Speedup vs `gdn` |"
        ),
        (
            "|------|--------------------:|--------------------------:|"
            "-----------------------:|--------------------------:|"
            "-----------------:|"
        ),
    ]
    for row in section_rows:
        case_cell = row.case.replace("|", "\\|")
        mlx_mean = row.flash_kda_mlx.mean_ms if row.flash_kda_mlx else None
        ck_mean = row.mlx_chunk_kda.mean_ms if row.mlx_chunk_kda else None
        gdn_mean = row.mlx_chunk_gdn.mean_ms if row.mlx_chunk_gdn else None
        ck_cell = _fmt_ms(ck_mean)
        gdn_cell = _fmt_ms(gdn_mean)
        ck_note = row.notes.get("mlx_chunk_kda")
        gdn_note = row.notes.get("mlx_chunk_gdn")
        if ck_note and ck_cell == MISSING_MARK:
            ck_cell = f"{MISSING_MARK} ({ck_note})"
        if gdn_note and gdn_cell == MISSING_MARK:
            gdn_cell = f"{MISSING_MARK} ({gdn_note})"
        lines.append(
            f"| {case_cell} | {_fmt_ms(mlx_mean)} | {ck_cell} | "
            f"{_fmt_speedup(ck_mean, mlx_mean)} | {gdn_cell} | "
            f"{_fmt_speedup(gdn_mean, mlx_mean)} |"
        )
    lines.append("")
    return lines


def render_markdown(
    rows: list[CaseRow],
    args: argparse.Namespace,
    generated_at: str,
    command: str,
) -> str:
    lines: list[str] = [
        "# KDA forward benchmark (MLX / Apple GPU)",
        "",
        f"- Generated: {generated_at}",
        "",
        f"- Command: `{command}`",
        "",
        (
            f"- Benchmark settings: `warmup={args.warmup}`, "
            f"`iters={args.iters}`, `repeats={args.repeats}`"
        ),
        "",
        "> **Hardware caveat.** MLX timings on Apple GPU are not "
        "numerically comparable to the CUDA/H20 table in "
        "`BENCHMARK_H20.md`. Column semantics mirror the CUDA report; "
        "absolute numbers do not.",
        "",
        "- MLX configuration: `backend=optimized`, `CHUNK=16`, "
        "`lower_bound=-5`, `D=128`. Timed regions force execution with "
        "`mx.eval(...)`.",
        f"- Metal recurrence: {_metal_recurrence_status()}.",
        f"- Metal prepare: {_metal_prepare_status()}.",
        "- Input construction (default `cuda_correspond=True`) matches "
        "`benchmarks/bench_fwd.py`: `q/k/v/g/beta` in bf16, `out` as bf16 "
        "zeros, `initial_state = arange(N*H*D*D).reshape(N,H,D,D).to(bf16)`, "
        "`final_state` as matching bf16 zeros. FLA baselines receive "
        "`initial_state.astype(fp32)` at the call site, mirroring CUDA's "
        "`initial_state.float()`.",
        "- `mlx_chunk_kda` / `mlx_chunk_gdn` columns wrap the MLX-LM "
        "gated-delta backbone with FLA-matching kwargs via "
        "`flash_kda_mlx.baselines.chunk_kda.chunk_kda_mlx` and "
        "`flash_kda_mlx.baselines.chunk_gdn.chunk_gdn_mlx`. `use_kernel=True` "
        "(Metal path) with `transpose_state_layout=True` and "
        "`use_qk_l2norm_in_kernel=True`.",
        _varlen_strategy_note(),
        "- GDN gate construction: `g_gdn = mx.random.normal((1, T_total, H), "
        "dtype=fp32)` drawn from a key distinct from the `g`/`q`/`k`/`v` "
        "seed stream — an independent deterministic scalar per-head gate, "
        "mirroring CUDA bench's `torch.randn((1, T_total, H), "
        "dtype=torch.float32)`.",
        "",
        "> **Torch-reference parity.** `flash_kda_mlx` is validated against "
        "the checked-in KDA fixtures in `tests/test_parity_fixtures.py`. "
        "`mlx_chunk_kda` and `mlx_chunk_gdn` are validated against small "
        "fixed/varlen PyTorch gated-delta reference cases in "
        "`tests/test_chunk_baseline_torch_reference.py`. CUDA device "
        "fixtures are not required for this MLX-side benchmark gate.",
        "",
        "> **Speedup convention.** Columns `Speedup vs chunk_kda` and "
        "`Speedup vs gdn` use `baseline_mean / flash_kda_mlx_mean`, matching "
        "`BENCHMARK_H20.md`. A value > 1 means `flash_kda_mlx` is faster than "
        "the baseline; < 1 means slower. These ratios depend on the "
        "Metal recurrence/prepare switches recorded above.",
        "",
        "## Benchmark Column Roles",
        "",
        "| Original `BENCHMARK_H20.md` column | MLX equivalent | Status |",
        "|---|---|---|",
        (
            "| `flash_kda` mean (ms) | `flash_kda_mlx.fwd(..., "
            "backend=\"optimized\")`, `CHUNK=16` | Validated against "
            "the torch-reference KDA fixtures. |"
        ),
        (
            "| `fla_chunk_kda` mean (ms) | "
            "`flash_kda_mlx.baselines.chunk_kda.chunk_kda_mlx` (MLX-LM "
            "backbone, FLA-matching kwargs) | "
            "Validated against local torch-reference fixed/varlen cases. |"
        ),
        (
            "| Speedup vs `chunk_kda` | "
            "`mlx_chunk_kda_mean / flash_kda_mlx_mean` | Reported; apples-to-apples "
            "across MLX methods but not cross-framework validated. |"
        ),
        (
            "| `fla_chunk_gdn` mean (ms) | "
            "`flash_kda_mlx.baselines.chunk_gdn.chunk_gdn_mlx` (MLX-LM "
            "backbone, FLA-matching kwargs) | "
            "Validated against local torch-reference fixed/varlen cases. |"
        ),
        (
            "| Speedup vs `gdn` | `mlx_chunk_gdn_mean / flash_kda_mlx_mean` | "
            "Reported; same caveat as above. |"
        ),
        "",
    ]

    # Primary table per H.
    for H in args.H:
        lines.extend(_render_primary_table(rows, H))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate CUDA-corresponding MLX benchmark table.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--H", type=int, nargs="+", default=[96, 64])
    p.add_argument(
        "--mode", choices=["fixed", "varlen", "all"], default="all",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--skip-chunk-kda", action="store_true",
        help="Omit the mlx_chunk_kda column (e.g., if mlx-lm is unavailable).",
    )
    p.add_argument(
        "--skip-gdn", action="store_true",
        help="Omit the mlx_chunk_gdn column (e.g., if mlx-lm is unavailable).",
    )
    p.add_argument(
        "--strict-equivalence", action="store_true",
        help=(
            "Exit with status 1 if any non-skipped baseline column fails "
            "to produce timings (import failure or adapter error). Default: "
            "warn and render '—' in the missing cells."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    seq_sets: list[list[int]] = []
    if args.mode in ("fixed", "all"):
        seq_sets.extend([list(x) for x in FIXED_CASES])
    if args.mode in ("varlen", "all"):
        seq_sets.extend([list(x) for x in VARLEN_CASES])

    # Resolve optional baselines once up front.
    chunk_kda_fn: Callable | None = None
    chunk_gdn_fn: Callable | None = None

    if not args.skip_chunk_kda:
        chunk_kda_fn, err_msg = _try_import_chunk_kda()
        if chunk_kda_fn is None:
            print(
                f"WARNING: mlx_chunk_kda unavailable — {err_msg}",
                file=sys.stderr,
            )
            if args.strict_equivalence:
                print("strict-equivalence set; aborting.", file=sys.stderr)
                return 1

    if not args.skip_gdn:
        chunk_gdn_fn, err_msg = _try_import_chunk_gdn()
        if chunk_gdn_fn is None:
            print(
                f"WARNING: mlx_chunk_gdn unavailable — {err_msg}",
                file=sys.stderr,
            )
            if args.strict_equivalence:
                print("strict-equivalence set; aborting.", file=sys.stderr)
                return 1

    rows: list[CaseRow] = []
    for H in args.H:
        for seq_lens in seq_sets:
            row = run_case(
                seq_lens, H, args,
                chunk_kda_fn=chunk_kda_fn,
                chunk_gdn_fn=chunk_gdn_fn,
            )
            # Strict-equivalence check: surface adapter runtime errors too.
            if args.strict_equivalence:
                if (
                    not args.skip_chunk_kda and row.mlx_chunk_kda
                    and row.mlx_chunk_kda.error is not None
                ):
                    print(
                        f"strict-equivalence: mlx_chunk_kda failed on "
                        f"H={H}, {row.case}: {row.mlx_chunk_kda.error}",
                        file=sys.stderr,
                    )
                    return 1
                if (
                    not args.skip_gdn and row.mlx_chunk_gdn
                    and row.mlx_chunk_gdn.error is not None
                ):
                    print(
                        f"strict-equivalence: mlx_chunk_gdn failed on "
                        f"H={H}, {row.case}: {row.mlx_chunk_gdn.error}",
                        file=sys.stderr,
                    )
                    return 1
            rows.append(row)

    command = "uv run --no-config python benchmarks/generate_benchmark_mlx_md.py"
    metal_env = os.environ.get("MLX_KDA_ENABLE_METAL_RECURRENCE")
    prepare_env = os.environ.get("MLX_KDA_ENABLE_METAL_PREPARE")
    if metal_env is not None:
        command = f"MLX_KDA_ENABLE_METAL_RECURRENCE={metal_env} " + command
    if prepare_env is not None:
        command = f"MLX_KDA_ENABLE_METAL_PREPARE={prepare_env} " + command
    if cli_argv:
        command += " " + " ".join(cli_argv)
    generated_at = _dt.date.today().isoformat()
    md = render_markdown(rows, args, generated_at, command)
    args.output.write_text(md)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
