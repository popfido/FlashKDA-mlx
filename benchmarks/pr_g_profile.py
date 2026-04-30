"""PR G post-Phase4 profile — picks Branch A/B/C from the future-work tree.

Three measurements per case, output as one merged JSON + markdown table:

1. ``flash_kda_mlx fwd_optimized`` end-to-end at
   ``MLX_KDA_ENABLE_METAL_RECURRENCE=0`` and ``=1``. Captures how much
   wall-clock the Phase 3b/4 simdgroup Metal recurrence saved.

2. ``chunk_kda_mlx(use_kernel=True)`` and ``chunk_kda_mlx(use_kernel=False)``
   — the MLX-LM-backed FLA ``chunk_kda`` adapter. ``use_kernel=True``
   dispatches to the Metal ``gated_delta_kernel``; ``False`` uses the
   pure-MLX ops fallback.

3. ``chunk_gdn_mlx(use_kernel=True)`` and ``chunk_gdn_mlx(use_kernel=False)``
   — the MLX-LM-backed FLA ``chunk_gated_delta_rule`` adapter, same
   toggle.

Implementation note: the env var ``MLX_KDA_ENABLE_METAL_RECURRENCE`` is
read at ``flash_kda_mlx.optimized`` import time and binds the Metal kernel
function pointers. Toggling within one Python process is therefore not
sufficient. We launch one subprocess per env mode for the flash_kda_mlx
measurements; adapter timings are env-independent and run in-process.

Usage::

    uv run --no-config python -m benchmarks.pr_g_profile
    uv run --no-config python -m benchmarks.pr_g_profile \\
        --cases fixed_T4096_H8 bench_fixed_T8192_H64_state
    uv run --no-config python -m benchmarks.pr_g_profile \\
        --json benchmarks/results/pr_g_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from ._harness import (
    CASE_BY_NAME,
    Case,
    build_call_kwargs,
    time_fn,
)


# Default case set: paired with the report cases used in §3 of
# section_timings_report.md, plus the bench-scale rows from
# BENCHMARK_MLX.md so the branch decision is grounded in
# bench-realistic shapes.
DEFAULT_CASE_NAMES: tuple[str, ...] = (
    # From §3 of the existing section-timings report:
    "fixed_T4096_H8",
    "varlen_mixed_H4",
    # Bench-scale (from BENCHMARK_MLX.md):
    "bench_fixed_T8192_H64_state",
    "bench_fixed_T8192_H96_state",
    "bench_varlen_uniform_H64_state",
    "bench_varlen_uniform_H96_state",
    "bench_varlen_mixed_H64_state",
    "bench_varlen_mixed_H96_state",
)


# ---------------------------------------------------------------------------
# Subprocess helper for the flash_kda_mlx fwd A/B
# ---------------------------------------------------------------------------

def _run_flash_kda_mlx_subprocess(
    case_names: list[str],
    metal_mode: str,
    n_warmup: int,
    n_iters: int,
) -> list[dict[str, Any]]:
    """Run ``benchmarks.bench_fwd`` in a subprocess with the requested
    ``MLX_KDA_ENABLE_METAL_RECURRENCE`` value, parse and return the JSON
    rows for the optimized backend only.
    """
    out_path = Path("/tmp") / f"pr_g_flash_kda_mlx_metal_{metal_mode}.json"
    if out_path.exists():
        out_path.unlink()

    env = dict(os.environ)
    env["MLX_KDA_ENABLE_METAL_RECURRENCE"] = metal_mode

    cmd = [
        sys.executable, "-m", "benchmarks.bench_fwd",
        "--backend", "optimized",
        "--n-iters", str(n_iters),
        "--cases", *case_names,
        "--json", str(out_path),
    ]
    print(
        f"  → flash_kda_mlx subprocess MLX_KDA_ENABLE_METAL_RECURRENCE={metal_mode}: "
        f"{' '.join(cmd)}"
    )
    subprocess.run(cmd, env=env, check=True)

    rows = json.loads(out_path.read_text())
    return [
        {
            "case": r["case"],
            "method": "flash_kda_mlx",
            "variant": f"metal={metal_mode}",
            "median_ms": r["median_ms"],
            "mean_ms": r["mean_ms"],
            "p90_ms": r["p90_ms"],
            "min_ms": r["min_ms"],
            "n_iters": r["n_iters"],
        }
        for r in rows
        if r["backend"] == "optimized"
    ]


# ---------------------------------------------------------------------------
# In-process adapter timings
# ---------------------------------------------------------------------------

def _fla_initial_state(kwargs: dict[str, Any]) -> mx.array | None:
    return kwargs.get("initial_state_fp32", kwargs.get("initial_state"))


def _time_chunk_kda(case: Case, *, use_kernel: bool, n_warmup: int, n_iters: int) -> dict[str, Any]:
    from flash_kda_mlx.baselines.chunk_kda import chunk_kda_mlx

    kwargs = build_call_kwargs(case)

    def go(kw: dict[str, Any] = kwargs) -> tuple[mx.array, mx.array | None]:
        return chunk_kda_mlx(
            q=kw["q"], k=kw["k"], v=kw["v"], g=kw["g"], beta=kw["beta"],
            scale=kw["scale"],
            initial_state=_fla_initial_state(kw),
            output_final_state=True,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            A_log=kw["A_log"], dt_bias=kw["dt_bias"],
            lower_bound=kw["lower_bound"],
            transpose_state_layout=True,
            cu_seqlens=kw.get("cu_seqlens"),
            use_kernel=use_kernel,
        )

    return time_fn(go, n_warmup=n_warmup, n_iters=n_iters)


def _time_chunk_gdn(case: Case, *, use_kernel: bool, n_warmup: int, n_iters: int) -> dict[str, Any]:
    from flash_kda_mlx.baselines.chunk_gdn import chunk_gdn_mlx

    kwargs = build_call_kwargs(case)

    def go(kw: dict[str, Any] = kwargs) -> tuple[mx.array, mx.array | None]:
        return chunk_gdn_mlx(
            q=kw["q"], k=kw["k"], v=kw["v"], g=kw["g_gdn"], beta=kw["beta"],
            scale=kw["scale"],
            initial_state=_fla_initial_state(kw),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            transpose_state_layout=True,
            cu_seqlens=kw.get("cu_seqlens"),
            use_kernel=use_kernel,
        )

    return time_fn(go, n_warmup=n_warmup, n_iters=n_iters)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_METHOD_ORDER = (
    ("flash_kda_mlx", "metal=0"),
    ("flash_kda_mlx", "metal=1"),
    ("chunk_kda_mlx", "use_kernel=False"),
    ("chunk_kda_mlx", "use_kernel=True"),
    ("chunk_gdn_mlx", "use_kernel=False"),
    ("chunk_gdn_mlx", "use_kernel=True"),
)


def _format_table(rows: list[dict[str, Any]]) -> str:
    by_case: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case"], {})[(r["method"], r["variant"])] = r

    out: list[str] = []
    out.append(
        "| Case | "
        + " | ".join(f"{m}<br>{v}" for m, v in _METHOD_ORDER)
        + " | Metal speedup<br>(=0/=1) | flash_kda_mlx(=1) vs<br>chunk_kda(kernel=T) |"
    )
    out.append("|" + "---|" * (1 + len(_METHOD_ORDER) + 2))
    for case_name in sorted(by_case):
        cells: list[str] = []
        meds: dict[tuple[str, str], float] = {}
        for key in _METHOD_ORDER:
            r = by_case[case_name].get(key)
            if r is None:
                cells.append("—")
            else:
                cells.append(f"{r['median_ms']:.2f}")
                meds[key] = r["median_ms"]
        # Metal speedup ratio (off / on); >1 means Metal is faster.
        try:
            metal_ratio = meds[("flash_kda_mlx", "metal=0")] / meds[("flash_kda_mlx", "metal=1")]
            cells.append(f"{metal_ratio:.2f}×")
        except (KeyError, ZeroDivisionError):
            cells.append("—")
        try:
            adapter = meds[("chunk_kda_mlx", "use_kernel=True")]
            flash_kda_mlx = meds[("flash_kda_mlx", "metal=1")]
            cells.append(f"{adapter / flash_kda_mlx:.2f}×")
        except (KeyError, ZeroDivisionError):
            cells.append("—")
        out.append(f"| {case_name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None,
                        help=f"Case names. Default: {list(DEFAULT_CASE_NAMES)}")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-iters", type=int, default=5)
    parser.add_argument(
        "--json",
        type=str,
        default=str(Path(__file__).parent / "results" / "pr_g_profile.json"),
    )
    parser.add_argument("--skip-mlx-kda", action="store_true",
                        help="Skip the flash_kda_mlx subprocess A/B (debug only)")
    parser.add_argument("--skip-adapters", action="store_true",
                        help="Skip the chunk_kda_mlx / chunk_gdn_mlx timings")
    args = parser.parse_args(argv)

    case_names = args.cases or list(DEFAULT_CASE_NAMES)
    unknown = [c for c in case_names if c not in CASE_BY_NAME]
    if unknown:
        raise SystemExit(
            f"Unknown case(s): {unknown}. "
            f"Available: {sorted(CASE_BY_NAME)}"
        )
    cases = [CASE_BY_NAME[c] for c in case_names]

    # Record env / version metadata so the report is auditable.
    try:
        import mlx_lm
        mlx_lm_version = mlx_lm.__version__
        mlx_lm_path = mlx_lm.__file__
    except Exception as exc:  # noqa: BLE001
        mlx_lm_version = f"unavailable: {exc}"
        mlx_lm_path = ""

    metadata = {
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "mlx_lm_version": mlx_lm_version,
        "mlx_lm_path": mlx_lm_path,
        "n_warmup": args.n_warmup,
        "n_iters": args.n_iters,
        "case_names": case_names,
    }
    print("# PR G — post-Phase4 profile")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    rows: list[dict[str, Any]] = []

    if not args.skip_flash_kda_mlx:
        print("\n## flash_kda_mlx subprocess A/B (MLX_KDA_ENABLE_METAL_RECURRENCE)")
        for mode in ("0", "1"):
            rows.extend(
                _run_flash_kda_mlx_subprocess(
                    case_names, mode, args.n_warmup, args.n_iters
                )
            )

    if not args.skip_adapters:
        print("\n## chunk_kda_mlx / chunk_gdn_mlx in-process timings")
        for case in cases:
            for use_kernel in (False, True):
                stats = _time_chunk_kda(
                    case, use_kernel=use_kernel,
                    n_warmup=args.n_warmup, n_iters=args.n_iters,
                )
                rows.append({
                    "case": case.name,
                    "method": "chunk_kda_mlx",
                    "variant": f"use_kernel={use_kernel}",
                    **stats,
                })
                print(
                    f"  {case.name} chunk_kda_mlx use_kernel={use_kernel}: "
                    f"median {stats['median_ms']:.2f} ms"
                )
            for use_kernel in (False, True):
                stats = _time_chunk_gdn(
                    case, use_kernel=use_kernel,
                    n_warmup=args.n_warmup, n_iters=args.n_iters,
                )
                rows.append({
                    "case": case.name,
                    "method": "chunk_gdn_mlx",
                    "variant": f"use_kernel={use_kernel}",
                    **stats,
                })
                print(
                    f"  {case.name} chunk_gdn_mlx use_kernel={use_kernel}: "
                    f"median {stats['median_ms']:.2f} ms"
                )

    print("\n## Combined median table\n")
    print(_format_table(rows))

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2))
    print(f"\nWrote JSON to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
