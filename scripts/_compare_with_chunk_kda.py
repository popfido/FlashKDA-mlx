"""Generate the docs/assets/compare_with_chunk_kda.png deep-dive figure.

One-shot helper for ``docs/2026MMDD-flashkda-mlx-v1-deep-dive.md``. For every
checked-in torch-oracle parity fixture (``tests/fixtures/*.npz``) we run both
``flash_kda_mlx.fwd(backend="optimized")`` and ``chunk_kda_mlx`` (the
MLX-LM-backed adapter that fills the ``mlx_chunk_kda`` benchmark column) and
compare each against the saved ``out_expected`` produced by
``scripts/torch_ref_cpu.py``. The output is a markdown table on stdout plus a
PNG bar chart written under ``docs/assets/``.

Run::

    MLX_KDA_ENABLE_METAL_PREPARE=fused4 \\
    MLX_KDA_ENABLE_METAL_RECURRENCE=1 \\
    uv run --no-config --with matplotlib python scripts/_compare_with_chunk_kda.py

The script is intentionally not part of the package or the test suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _helpers import fixture_to_mlx_inputs, load_fixture, to_numpy  # noqa: E402

import flash_kda_mlx  # noqa: E402
from flash_kda_mlx.baselines.chunk_kda import chunk_kda_mlx  # noqa: E402


CASES: tuple[tuple[str, str], ...] = (
    ("fixed__T16__H1__no_state__seed0",                                     "T=16  H=1  fixed"),
    ("fixed__T64__H1__no_state__seed0",                                     "T=64  H=1  fixed"),
    ("fixed__T64__H4__no_state__seed0",                                     "T=64  H=4  fixed"),
    ("fixed__T256__H1__no_state__seed0",                                    "T=256 H=1  fixed"),
    ("fixed__T256__H4__no_state__seed0",                                    "T=256 H=4  fixed"),
    ("fixed__T256__H4__state_in_out_bf16__seed0",                           "T=256 H=4  state_bf16"),
    ("fixed__T256__H4__state_in_out_fp32__seed0",                           "T=256 H=4  state_fp32"),
    ("varlen__T214__H1__no_state__seed0__varlen_37_16_97_64",               "varlen H=1  no_state"),
    ("varlen__T214__H4__no_state__seed0__varlen_37_16_97_64",               "varlen H=4  no_state"),
    ("varlen__T214__H4__state_in_out_bf16__seed0__varlen_37_16_97_64",      "varlen H=4  state_bf16"),
    ("varlen__T214__H4__state_in_out_fp32__seed0__varlen_37_16_97_64",      "varlen H=4  state_fp32"),
)


@dataclass(frozen=True)
class Result:
    label: str
    flash_max: float
    flash_mean: float
    chunk_max: float
    chunk_mean: float


def _run_flash(kwargs: dict, expected_shape: tuple[int, ...]) -> mx.array:
    out = mx.zeros(expected_shape, dtype=mx.float32)
    final_state = None
    if "initial_state" in kwargs:
        final_state = mx.zeros_like(kwargs["initial_state"])
    res = flash_kda_mlx.fwd(
        out=out, final_state=final_state, backend="optimized", **kwargs
    )
    mx.eval(res.out)
    return res.out


def _run_chunk_kda(kwargs: dict) -> mx.array:
    # Adapter expects FLA-style kwargs and a [N,H,D,D] state in FLA layout.
    init_state = kwargs.get("initial_state")
    out, _ = chunk_kda_mlx(
        kwargs["q"], kwargs["k"], kwargs["v"], kwargs["g"], kwargs["beta"],
        scale=kwargs["scale"],
        initial_state=init_state,
        output_final_state=False,
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
    mx.eval(out)
    return out


def _err(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return float(diff.max()), float(diff.mean())


def evaluate() -> list[Result]:
    results: list[Result] = []
    for name, label in CASES:
        fx = load_fixture(name)
        kwargs = fixture_to_mlx_inputs(fx)
        expected = fx["out_expected"]

        flash_out = _run_flash(kwargs, expected.shape)
        chunk_out = _run_chunk_kda(kwargs)

        flash_max, flash_mean = _err(to_numpy(flash_out), expected)
        chunk_max, chunk_mean = _err(to_numpy(chunk_out), expected)
        results.append(Result(label, flash_max, flash_mean, chunk_max, chunk_mean))
    return results


def print_table(results: list[Result]) -> None:
    print("| Case | flash_kda_mlx max\\|Δ\\| | chunk_kda max\\|Δ\\| | flash mean\\|Δ\\| | chunk mean\\|Δ\\| |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.label} | {r.flash_max:.2e} | {r.chunk_max:.2e} | "
            f"{r.flash_mean:.2e} | {r.chunk_mean:.2e} |"
        )


def render_png(results: list[Result], out_path: Path) -> None:
    labels = [r.label for r in results]
    flash_vals = [r.flash_max for r in results]
    chunk_vals = [r.chunk_max for r in results]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=140)
    bars_flash = ax.bar(
        x - width / 2, flash_vals, width,
        label="flash_kda_mlx (this work)",
        color="#1f77b4",
    )
    bars_chunk = ax.bar(
        x + width / 2, chunk_vals, width,
        label="mlx_chunk_kda (mlx-lm gated_delta_kernel)",
        color="#d62728",
    )

    ax.set_yscale("log")
    ax.set_ylabel("max |output − torch oracle|  (log scale)")
    ax.set_title(
        "Accuracy vs. CPU torch reference oracle — flash_kda_mlx vs. mlx_chunk_kda\n"
        "Lower is better. bf16 round-off floor ≈ 3.9e-3.",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.axhline(3.9e-3, linestyle="--", linewidth=0.8, color="gray",
               label="bf16 round-off floor (3.9e-3)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4, alpha=0.6)

    for bars, vals in ((bars_flash, flash_vals), (bars_chunk, chunk_vals)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v * 1.15,
                f"{v:.1e}",
                ha="center", va="bottom", fontsize=7,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    results = evaluate()
    print_table(results)
    render_png(results, REPO_ROOT / "docs" / "assets" / "compare_with_chunk_kda.png")


if __name__ == "__main__":
    main()
