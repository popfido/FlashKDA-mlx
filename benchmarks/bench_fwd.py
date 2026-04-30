"""Timed benchmark driver for ``flash_kda_mlx.fwd``.

Usage::

    uv run --no-config python -m benchmarks.bench_fwd
    uv run --no-config python -m benchmarks.bench_fwd --stress
    uv run --no-config python -m benchmarks.bench_fwd --backend reference
    uv run --no-config python -m benchmarks.bench_fwd --backend all

Results print a markdown table to stdout. ``--json`` emits machine-readable
output to ``benchmarks/results/latest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

import flash_kda_mlx

from ._harness import (
    CASE_BY_NAME,
    DEFAULT_CASES,
    STRESS_CASES,
    TABLE_HEADER,
    Case,
    build_call_kwargs,
    format_result_row,
    time_fn,
)


def _available_backends() -> list[str]:
    """Return backends the running ``flash_kda_mlx`` understands.

    We inspect the ``fwd`` signature so a single bench script works before and
    after the optimized backend lands. "reference" is always available.
    """
    import inspect
    sig = inspect.signature(flash_kda_mlx.fwd)
    if "backend" in sig.parameters:
        return ["reference", "optimized"]
    return ["reference"]


def _call(kwargs: dict, backend: str):
    if backend == "reference":
        from flash_kda_mlx import reference
        return reference.fwd_reference(
            q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
            g=kwargs["g"], beta=kwargs["beta"],
            scale=kwargs["scale"], out_like=kwargs["out"],
            A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
            lower_bound=kwargs["lower_bound"],
            initial_state=kwargs.get("initial_state"),
            final_state_like=kwargs.get("final_state"),
            cu_seqlens=kwargs.get("cu_seqlens"),
        )
    if backend == "optimized":
        from flash_kda_mlx import optimized
        return optimized.fwd_optimized(
            q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
            g=kwargs["g"], beta=kwargs["beta"],
            scale=kwargs["scale"], out_like=kwargs["out"],
            A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
            lower_bound=kwargs["lower_bound"],
            initial_state=kwargs.get("initial_state"),
            final_state_like=kwargs.get("final_state"),
            cu_seqlens=kwargs.get("cu_seqlens"),
        )
    raise ValueError(f"unknown backend: {backend}")


def run(cases: list[Case], backends: list[str], n_iters: int) -> list[dict]:
    rows = []
    for case in cases:
        kwargs = build_call_kwargs(case)
        for backend in backends:
            # Default-argument capture prevents the late-binding pitfall if
            # this loop is ever parallelized.
            stats = time_fn(
                lambda b=backend, kw=kwargs: _call(kw, b),
                n_iters=n_iters,
            )
            rows.append({
                "case": case.name, "backend": backend, **stats,
            })
            print(format_result_row(case.name, backend, stats))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", action="store_true",
                        help="Run larger stress cases.")
    parser.add_argument("--backend", default="all",
                        choices=["reference", "optimized", "all"])
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--json", type=str, default=None,
                        help="Write JSON results to this path (alias: --output).")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to this path.")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Subset of case names to run (space-separated). "
                             "Defaults to DEFAULT_CASES (+ STRESS_CASES if --stress).")
    args = parser.parse_args(argv)

    available = _available_backends()
    if args.backend == "all":
        backends = available
    elif args.backend not in available:
        print(f"Backend '{args.backend}' not available (have: {available}). "
              f"Falling back to 'reference'.", file=sys.stderr)
        backends = ["reference"]
    else:
        backends = [args.backend]

    if args.cases is not None:
        unknown = [c for c in args.cases if c not in CASE_BY_NAME]
        if unknown:
            print(f"Unknown case(s): {unknown}. "
                  f"Available: {sorted(CASE_BY_NAME)}", file=sys.stderr)
            return 2
        cases = [CASE_BY_NAME[name] for name in args.cases]
    else:
        cases = list(DEFAULT_CASES) + (list(STRESS_CASES) if args.stress else [])

    print(f"# MLX fwd benchmarks — device={mx.default_device()}  "
          f"backends={backends}  n_iters={args.n_iters}\n")
    print(TABLE_HEADER)
    rows = run(cases, backends, n_iters=args.n_iters)

    output_path = args.output or args.json
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote JSON results to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
