"""Unit tests for formatting / timing helpers in
``benchmarks.generate_benchmark_mlx_md``.

These tests cover pure-function logic (formatting, case-name rendering,
timing aggregation sentinel, CLI parsing). Wall-clock-dependent GPU
benchmark runs are *not* exercised here — they would be flaky and slow
by design.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmarks.generate_benchmark_mlx_md import (
    MISSING_MARK,
    TimingStats,
    _case_name,
    _collect_timings,
    _fmt_ms,
    _fmt_speedup,
    parse_args,
)


# ---------------------------------------------------------------------------
# _fmt_ms
# ---------------------------------------------------------------------------

class TestFmtMs:
    def test_formats_positive_float(self) -> None:
        assert _fmt_ms(12.34567) == "12.3457"

    def test_formats_integer(self) -> None:
        assert _fmt_ms(1) == "1.0000"

    def test_none_renders_as_missing_mark(self) -> None:
        assert _fmt_ms(None) == MISSING_MARK

    def test_nan_renders_as_missing_mark(self) -> None:
        assert _fmt_ms(float("nan")) == MISSING_MARK


# ---------------------------------------------------------------------------
# _fmt_speedup
# ---------------------------------------------------------------------------

class TestFmtSpeedup:
    def test_basic_ratio(self) -> None:
        # ratio = baseline_ms / target_ms; higher means target is faster.
        # _fmt_speedup(baseline, target) => baseline/target ×
        assert _fmt_speedup(10.0, 2.5) == "4.00×"

    def test_sub_unity(self) -> None:
        assert _fmt_speedup(2.5, 10.0) == "0.25×"

    def test_none_left_is_missing(self) -> None:
        assert _fmt_speedup(None, 3.0) == MISSING_MARK

    def test_none_right_is_missing(self) -> None:
        assert _fmt_speedup(3.0, None) == MISSING_MARK

    def test_zero_denominator_is_missing(self) -> None:
        assert _fmt_speedup(10.0, 0) == MISSING_MARK

    def test_zero_numerator_is_missing(self) -> None:
        assert _fmt_speedup(0, 10.0) == MISSING_MARK

    def test_nan_is_missing(self) -> None:
        assert _fmt_speedup(float("nan"), 2.0) == MISSING_MARK
        assert _fmt_speedup(2.0, float("nan")) == MISSING_MARK


# ---------------------------------------------------------------------------
# _case_name
# ---------------------------------------------------------------------------

class TestCaseName:
    def test_fixed(self) -> None:
        assert _case_name([8192]) == "Fixed"

    def test_uniform_varlen(self) -> None:
        assert _case_name([1024] * 8) == "Varlen, `seq_lens`=`1024 x 8`"

    def test_mixed_varlen(self) -> None:
        seq = [1300, 547, 2048, 963, 271, 3063]
        expected = (
            "Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063]"
        )
        assert _case_name(seq) == expected


# ---------------------------------------------------------------------------
# _collect_timings
# ---------------------------------------------------------------------------

class TestCollectTimings:
    def test_returns_timing_stats_for_trivial_fn(self) -> None:
        # Local import: mlx.core has non-trivial init cost and the other
        # tests in this module are pure-Python formatting checks.
        import mlx.core as mx

        def fn() -> mx.array:
            return mx.sum(mx.zeros(8))

        stats = _collect_timings(fn, warmup=1, iters=3, repeats=1)
        assert isinstance(stats, TimingStats)
        assert stats.error is None
        # Timings should be non-negative and well-ordered.
        assert stats.samples == 3
        assert stats.mean_ms >= 0.0
        assert stats.min_ms >= 0.0
        assert stats.max_ms >= stats.min_ms
        assert stats.mean_ms <= stats.max_ms + 1e-9
        assert stats.median_ms >= stats.min_ms
        assert stats.median_ms <= stats.max_ms + 1e-9

    def test_catches_exception_and_returns_error_stats(self) -> None:
        def fn() -> mx.array:
            raise RuntimeError("boom")

        stats = _collect_timings(fn, warmup=0, iters=1, repeats=1)
        assert isinstance(stats, TimingStats)
        assert stats.error is not None
        assert "boom" in stats.error
        assert stats.samples == 0
        assert math.isnan(stats.mean_ms)

    def test_fmt_ms_handles_error_stats(self) -> None:
        """Rendering an error TimingStats via _fmt_ms should emit MISSING_MARK."""
        def fn() -> mx.array:
            raise RuntimeError("fail")

        stats = _collect_timings(fn, warmup=0, iters=1, repeats=1)
        assert _fmt_ms(stats.mean_ms) == MISSING_MARK


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.warmup == 3
        assert args.iters == 10
        assert args.repeats == 1
        assert args.H == [96, 64]
        assert args.mode == "all"
        assert args.seed == 0
        assert args.skip_chunk_kda is False
        assert args.skip_gdn is False
        assert args.strict_equivalence is False
        assert isinstance(args.output, Path)

    def test_skip_flags(self) -> None:
        args = parse_args(["--skip-chunk-kda", "--skip-gdn"])
        assert args.skip_chunk_kda is True
        assert args.skip_gdn is True

    def test_strict_equivalence(self) -> None:
        args = parse_args(["--strict-equivalence"])
        assert args.strict_equivalence is True

    @pytest.mark.parametrize("flag", ["--include-chunk32", "--chunks"])
    def test_retired_chunk32_flags_rejected(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([flag])

    def test_output_override(self) -> None:
        args = parse_args(["--output", "/tmp/report.md"])
        assert args.output == Path("/tmp/report.md")

    def test_custom_H(self) -> None:
        args = parse_args(["--H", "32", "16"])
        assert args.H == [32, 16]

    def test_mode_choices(self) -> None:
        args = parse_args(["--mode", "fixed"])
        assert args.mode == "fixed"
        with pytest.raises(SystemExit):
            parse_args(["--mode", "bogus"])

    def test_parse_args_skip_and_strict_together(self) -> None:
        # Skip takes precedence over strict for the skipped baseline —
        # no error is raised when the baseline is deliberately omitted.
        args = parse_args(["--skip-chunk-kda", "--strict-equivalence"])
        assert args.skip_chunk_kda is True
        assert args.strict_equivalence is True

    def test_parse_args_skip_all_baselines(self) -> None:
        # Skipping both baselines is legal — should still run flash_kda_mlx
        # and produce a minimal table.
        args = parse_args(["--skip-chunk-kda", "--skip-gdn"])
        assert args.skip_chunk_kda is True
        assert args.skip_gdn is True
