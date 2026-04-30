"""Metal GPU-trace capture for ``flash_kda_mlx.fwd``.

Usage::

    uv run --no-config python -m benchmarks.profile_fwd \\
        --case fixed_T1024_H4 --backend reference \\
        --out benchmarks/traces/ref_T1024_H4.gputrace

Open the resulting ``.gputrace`` bundle with Xcode's GPU debugger or
Instruments' Metal trace template.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._harness import (
    DEFAULT_CASES,
    STRESS_CASES,
    Case,
    build_call_kwargs,
    capture_metal_trace,
)
from .bench_fwd import _available_backends, _call


def _find_case(name: str) -> Case:
    for case in (*DEFAULT_CASES, *STRESS_CASES):
        if case.name == name:
            return case
    raise SystemExit(
        f"Unknown case: {name!r}. "
        f"Choose from: {[c.name for c in (*DEFAULT_CASES, *STRESS_CASES)]}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="fixed_T1024_H4")
    parser.add_argument("--backend", default="reference")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "traces" / "fwd.gputrace"),
        help="Output .gputrace bundle path.",
    )
    args = parser.parse_args(argv)

    available = _available_backends()
    if args.backend not in available:
        print(f"Backend '{args.backend}' not available. Available: {available}",
              file=sys.stderr)
        return 2

    case = _find_case(args.case)
    kwargs = build_call_kwargs(case)
    trace = capture_metal_trace(
        lambda: _call(kwargs, args.backend),
        trace_path=args.out,
    )
    print(f"Wrote Metal trace to: {trace}")
    print("Open with: open -a Xcode <trace>  or import into Instruments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
