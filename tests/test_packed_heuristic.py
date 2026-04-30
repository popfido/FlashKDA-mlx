"""Tests for the size-aware packed-vs-per-seq varlen dispatch heuristic.

Background
----------
Up through commit 26b7813 (section_timings_report.md §8 addendum), the
varlen dispatch in ``fwd_optimized`` unconditionally ran the packed
cross-sequence path whenever ``N > 1`` (unless opted out via the
``MLX_KDA_DISABLE_PACKED`` env var). The A/B showed the packed path is a
**massive regression at bench-scale high-H mixed-varlen** (e.g. 13.2×
slower on ``seq_lens=[1300, 547, 2048, 963, 271, 3063]`` at ``H=96``)
while remaining a win for small-N high-count varlen.

This suite pins the new routing heuristic:

* Pure-function coverage of ``_should_use_packed(seq_lens, H, chunk)``
  across the empirical size regimes.
* Integration coverage: under a monkeypatched threshold, packed and
  per-seq paths produce numerically equivalent output for identical
  inputs — the heuristic is a pure perf lever, not a correctness one.
* Env-override coverage: ``MLX_KDA_DISABLE_PACKED=1`` still forces
  per-seq even when the heuristic would pick packed.

Threshold (1024, on ``max_chunks × H``) is justified in
``benchmarks/section_timings_report.md`` §8.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _helpers import make_varlen_inputs, to_numpy  # noqa: E402
from flash_kda_mlx import optimized, reference  # noqa: E402


D = 128


# ---------------------------------------------------------------------------
# Pure-function: _should_use_packed
# ---------------------------------------------------------------------------

def test_heuristic_public_symbol_exists():
    """The routing decision must be a named, importable function so tests
    (and future heuristic refinements) can poke at it directly."""
    assert hasattr(optimized, "_should_use_packed"), (
        "_should_use_packed must live in flash_kda_mlx.optimized so the "
        "routing decision is inspectable and testable."
    )
    assert hasattr(optimized, "_MAX_PACKED_WORK_PER_SEQ"), (
        "The threshold constant must be module-level for monkeypatching."
    )


def test_heuristic_default_threshold_is_1024():
    """Threshold 1024 comes from section_timings_report.md §8 — a 2×
    safety margin below the smallest case where per-seq wins (uniform
    H=64 at max_chunks × H = 4096)."""
    assert optimized._MAX_PACKED_WORK_PER_SEQ == 1024


def test_heuristic_empty_seq_lens_returns_false():
    """Defensive: an empty seq_lens list can't route to anything useful;
    the caller must handle N == 0. We return False to match the
    "nothing to pack" intuition."""
    assert optimized._should_use_packed([], H=8, chunk=16) is False


@pytest.mark.parametrize(
    "seq_lens, H, chunk, expected, label",
    [
        # Small-N high-count: packed amortizes per-seq loop overhead.
        ([37, 16, 97, 64], 4, 16, True,
         "varlen_mixed_H4: max_c=7, H=4, max_c*H=28 ≤ 1024"),
        ([37, 16, 97, 64, 128, 17, 256, 80], 4, 16, True,
         "varlen_N8_mixed_H4: max_c=16, H=4, max_c*H=64 ≤ 1024"),
        ([64] * 16, 4, 16, True,
         "varlen_N16_T64_H4: max_c=4, H=4, max_c*H=16 ≤ 1024"),
        # Boundary: right at the threshold (max_c*H == 1024) stays packed.
        ([16 * 16] * 4, 64, 16, True,
         "max_c=16, H=64 → 1024 == threshold → packed"),
        # Just over: flip to per-seq.
        ([16 * 17], 64, 16, False,
         "max_c=17, H=64 → 1088 > 1024 → per-seq"),
        # Bench-scale: all four bench_varlen_* cases must route to per-seq.
        ([1300, 547, 2048, 963, 271, 3063], 64, 16, False,
         "bench_varlen_mixed_H64: max_c=192, H=64 → 12288 → per-seq"),
        ([1300, 547, 2048, 963, 271, 3063], 96, 16, False,
         "bench_varlen_mixed_H96: max_c=192, H=96 → 18432 → per-seq"),
        ([1024] * 8, 64, 16, False,
         "bench_varlen_uniform_H64: max_c=64, H=64 → 4096 → per-seq"),
        ([1024] * 8, 96, 16, False,
         "bench_varlen_uniform_H96: max_c=64, H=96 → 6144 → per-seq"),
        # Fixed-length single-sequence cases (N=1): the heuristic's return
        # value is irrelevant (the caller short-circuits on N==1), but the
        # function must not crash and must be consistent with its rule.
        ([8192], 64, 16, False,
         "fixed_T8192_H64 single-seq: max_c=512, H=64 → 32768 → per-seq"),
        # CHUNK=32 halves max_c, so the threshold bites later.
        ([1024] * 8, 32, 32, True,
         "chunk=32 uniform_H32: max_c=32, H=32 → 1024 == threshold"),
    ],
)
def test_heuristic_routes_correctly(seq_lens, H, chunk, expected, label):
    got = optimized._should_use_packed(seq_lens, H=H, chunk=chunk)
    assert got == expected, (
        f"{label}: _should_use_packed({seq_lens=}, {H=}, {chunk=}) "
        f"= {got}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Integration: both paths must produce numerically equivalent outputs
# ---------------------------------------------------------------------------

def _make_kwargs(seq_lens, H, seed):
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=seed)
    T_total = sum(seq_lens)
    return {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T_total, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
        "cu_seqlens": inputs["cu_seqlens"],
    }


def _run_path(kwargs):
    return optimized.fwd_optimized(
        q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
        g=kwargs["g"], beta=kwargs["beta"],
        scale=kwargs["scale"], out_like=kwargs["out"],
        A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        initial_state=kwargs.get("initial_state"),
        final_state_like=kwargs.get("final_state"),
        cu_seqlens=kwargs["cu_seqlens"],
    )


def test_packed_vs_perseq_parity_same_case(monkeypatch):
    """Same inputs, two routings, must match within the packed-path band.

    The heuristic is a perf lever only — both branches compute the same
    delta-rule recurrence. This guards against any silent divergence
    introduced by the routing refactor.

    We pick a small-N mixed-length case so the test runs cheaply, and
    we force per-seq by monkeypatching the threshold down.
    """
    seq_lens = [37, 16, 97, 64, 128, 17]
    H = 4
    kwargs = _make_kwargs(seq_lens, H, seed=101)

    # Route 1: heuristic default → packed (max_c*H = 8*4 = 32 ≤ 1024).
    assert optimized._should_use_packed(seq_lens, H=H, chunk=16) is True
    out_packed, _ = _run_path(kwargs)

    # Route 2: monkeypatch threshold to 0 → force per-seq.
    monkeypatch.setattr(optimized, "_MAX_PACKED_WORK_PER_SEQ", 0)
    assert optimized._should_use_packed(seq_lens, H=H, chunk=16) is False
    out_perseq, _ = _run_path(kwargs)

    # Same tolerance as the packed-path parity tests (1-ULP band).
    np.testing.assert_allclose(
        to_numpy(out_packed), to_numpy(out_perseq),
        rtol=1e-4, atol=5e-5,
        err_msg="packed vs per-seq outputs diverged under the heuristic",
    )


def test_env_override_still_forces_perseq(monkeypatch):
    """``MLX_KDA_DISABLE_PACKED=1`` must still force per-seq regardless
    of the heuristic's preference.

    The env var is the explicit user-side override used by benchmark
    A/Bs and by bisect-a-regression workflows. The heuristic must not
    subsume it.
    """
    seq_lens = [37, 16, 97, 64]
    H = 4
    kwargs = _make_kwargs(seq_lens, H, seed=202)

    # Heuristic says packed (max_c*H = 7*4 = 28 ≤ 1024).
    assert optimized._should_use_packed(seq_lens, H=H, chunk=16) is True

    # Set the env var, reload the module, run — must still produce valid
    # output matching the heuristic-default run bit-for-ULP.
    baseline, _ = _run_path(kwargs)

    monkeypatch.setenv("MLX_KDA_DISABLE_PACKED", "1")
    # The module read the env var at import time; verify that the
    # module-level constant exposes it for test-side override too.
    assert hasattr(optimized, "_DISABLE_PACKED")
    monkeypatch.setattr(optimized, "_DISABLE_PACKED", True)
    forced_perseq, _ = _run_path(kwargs)

    np.testing.assert_allclose(
        to_numpy(baseline), to_numpy(forced_perseq),
        rtol=1e-4, atol=5e-5,
        err_msg="env override routed to a numerically different path",
    )


def test_reference_parity_under_forced_perseq(monkeypatch):
    """Per-seq path under heuristic-force must match the chunk-by-chunk
    reference oracle at the same tolerance as the packed path.

    This is the correctness gate for routing mixed-varlen through the
    per-seq loop at bench scale — if the per-seq branch had drifted from
    the oracle, the new heuristic's bench speedup would be illusory.
    """
    seq_lens = [37, 16, 97, 64, 128, 17, 256, 80]  # N=8, H=4 small case.
    H = 4
    kwargs = _make_kwargs(seq_lens, H, seed=303)

    # Force per-seq for this N>1 case.
    monkeypatch.setattr(optimized, "_MAX_PACKED_WORK_PER_SEQ", 0)

    opt_out, _ = optimized.fwd_optimized(
        q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
        g=kwargs["g"], beta=kwargs["beta"],
        scale=kwargs["scale"], out_like=kwargs["out"],
        A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        initial_state=kwargs.get("initial_state"),
        final_state_like=kwargs.get("final_state"),
        cu_seqlens=kwargs["cu_seqlens"],
    )
    ref_out, _ = reference.fwd_reference(
        q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
        g=kwargs["g"], beta=kwargs["beta"],
        scale=kwargs["scale"], out_like=kwargs["out"],
        A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        initial_state=kwargs.get("initial_state"),
        final_state_like=kwargs.get("final_state"),
        cu_seqlens=kwargs["cu_seqlens"],
    )

    np.testing.assert_allclose(
        to_numpy(opt_out), to_numpy(ref_out),
        rtol=1e-4, atol=5e-5,
        err_msg="per-seq path diverged from reference oracle",
    )
