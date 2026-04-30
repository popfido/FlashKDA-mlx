"""API/shape-contract tests for ``flash_kda_mlx.fwd``.

These tests lock the public surface. They must be runnable against a
stub implementation — the rules they encode are about *validation*, not
about the algorithm itself. They should continue to pass through every PR.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

import flash_kda_mlx
from _helpers import make_inputs, make_varlen_inputs


D = 128


def _base_inputs(T: int = 16, H: int = 1):
    """Construct a full, valid kwarg set. Callers mutate what they want to
    break and splat the rest directly into ``flash_kda_mlx.fwd``.
    """
    inp = make_inputs(T=T, H=H, D=D, seed=0)
    out = mx.zeros((1, T, H, D), dtype=mx.float32)
    return {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"],
        "scale": inp["scale"],
        "out": out,
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
    }


def _call_fwd(**overrides):
    """Call flash_kda_mlx.fwd with a small legal default set unless overridden."""
    kwargs = _base_inputs()
    kwargs.update(overrides)
    return flash_kda_mlx.fwd(**kwargs)


# ---------------------------------------------------------------------------
# Happy-path contract
# ---------------------------------------------------------------------------

def test_fwd_returns_output_and_optional_state():
    """fwd returns (out, final_state); final_state is None if not requested."""
    out, final = _call_fwd()
    assert isinstance(out, mx.array)
    assert final is None
    assert out.shape == (1, 16, 1, D)


def test_fwd_rank_is_4d():
    """q/k/v/g must be 4D [B, T, H, D]."""
    kwargs = _base_inputs()
    kwargs["q"] = mx.zeros((16, 1, D), dtype=mx.float32)  # 3D
    with pytest.raises(ValueError, match="q"):
        flash_kda_mlx.fwd(**kwargs)


def test_fwd_requires_d_128():
    """D != 128 must be rejected (hardcoded constraint)."""
    inputs = make_inputs(T=16, H=1, D=64)
    out = mx.zeros((1, 16, 1, 64), dtype=mx.float32)
    with pytest.raises(ValueError, match="D"):
        flash_kda_mlx.fwd(
            q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], beta=inputs["beta"],
            scale=inputs["scale"], out=out,
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            lower_bound=inputs["lower_bound"],
        )


def test_varlen_requires_b_equals_1():
    """When cu_seqlens is provided, B must be 1."""
    inputs = make_varlen_inputs([16, 16], H=1, D=D)
    q_b2 = mx.broadcast_to(inputs["q"], (2,) + tuple(inputs["q"].shape[1:]))
    beta_b2 = mx.broadcast_to(
        inputs["beta"], (2,) + tuple(inputs["beta"].shape[1:])
    )
    out = mx.zeros(q_b2.shape, dtype=mx.float32)
    with pytest.raises(ValueError, match="B=1"):
        flash_kda_mlx.fwd(
            q=q_b2, k=q_b2, v=q_b2, g=q_b2, beta=beta_b2,
            scale=inputs["scale"], out=out,
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            lower_bound=inputs["lower_bound"],
            cu_seqlens=inputs["cu_seqlens"],
        )


def test_beta_rank_is_3d():
    """beta must be [B, T, H]."""
    kwargs = _base_inputs()
    kwargs["beta"] = mx.zeros((1, 16), dtype=mx.float32)
    with pytest.raises(ValueError, match="beta"):
        flash_kda_mlx.fwd(**kwargs)


def test_state_shape_fixed_mode():
    """In fixed mode, state shape must be [B, H, D, D]."""
    kwargs = _base_inputs()
    kwargs["initial_state"] = mx.zeros((1, 1, D, D // 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="initial_state"):
        flash_kda_mlx.fwd(**kwargs)


@pytest.mark.parametrize("backend", ["reference", "optimized"])
def test_backend_kwarg_dispatches(backend):
    """Both backends reachable via the public fwd() API."""
    result = _call_fwd(backend=backend)
    assert result.out.shape == (1, 16, 1, D)
    assert result.final_state is None


def test_backend_rejects_unknown():
    """Unknown backend values are rejected."""
    with pytest.raises(ValueError, match="backend"):
        _call_fwd(backend="nonexistent")


# ---------------------------------------------------------------------------
# FwdResult typed return
# ---------------------------------------------------------------------------

def test_fwd_result_is_named_tuple():
    """fwd returns a FwdResult NamedTuple with .out / .final_state fields."""
    result = _call_fwd()
    assert isinstance(result, flash_kda_mlx.FwdResult)
    assert isinstance(result.out, mx.array)
    assert result.final_state is None


def test_fwd_result_tuple_destructuring():
    """Legacy ``out, final = fwd(...)`` destructuring continues to work."""
    out, final = _call_fwd()
    assert isinstance(out, mx.array)
    assert final is None


def test_fwd_result_positional_indexing():
    """result[0] is result.out, result[1] is result.final_state."""
    result = _call_fwd()
    assert result[0] is result.out
    assert result[1] is result.final_state


# ---------------------------------------------------------------------------
# Input-type checks
# ---------------------------------------------------------------------------

def test_fwd_rejects_non_array_q():
    """Passing a python list for q raises TypeError."""
    kwargs = _base_inputs()
    kwargs["q"] = [[[[0.0] * D]]]
    with pytest.raises(TypeError, match="q"):
        flash_kda_mlx.fwd(**kwargs)


def test_fwd_rejects_non_int_cu_seqlens():
    """cu_seqlens must be int32/int64, not float."""
    inputs = make_varlen_inputs([16, 16], H=1, D=D)
    bad_cu = mx.array([0, 16, 32], dtype=mx.float32)
    out = mx.zeros(inputs["q"].shape, dtype=mx.float32)
    with pytest.raises(TypeError, match="cu_seqlens"):
        flash_kda_mlx.fwd(
            q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"],
            beta=inputs["beta"],
            scale=inputs["scale"], out=out,
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            lower_bound=inputs["lower_bound"],
            cu_seqlens=bad_cu,
        )


# ---------------------------------------------------------------------------
# Shape validation — previously-missing checks
# ---------------------------------------------------------------------------

def test_fwd_rejects_kvg_shape_mismatch():
    """k/v/g must match q's shape exactly."""
    kwargs = _base_inputs(T=16, H=2)
    # Give k a different H. Same rank, wrong shape.
    kwargs["k"] = mx.zeros((1, 16, 1, D), dtype=mx.float32)
    with pytest.raises(ValueError, match="k"):
        flash_kda_mlx.fwd(**kwargs)


def test_fwd_rejects_out_shape_mismatch():
    """out placeholder shape must match q."""
    kwargs = _base_inputs()
    kwargs["out"] = mx.zeros((1, 32, 1, D), dtype=mx.float32)
    with pytest.raises(ValueError, match="out"):
        flash_kda_mlx.fwd(**kwargs)


def test_fwd_rejects_wrong_a_log_shape():
    """A_log must be [H]."""
    kwargs = _base_inputs(H=4)
    kwargs["A_log"] = mx.zeros((2,), dtype=mx.float32)  # wrong H
    with pytest.raises(ValueError, match="A_log"):
        flash_kda_mlx.fwd(**kwargs)


def test_fwd_rejects_wrong_dt_bias_shape():
    """dt_bias must be [H, D]."""
    kwargs = _base_inputs(H=4)
    kwargs["dt_bias"] = mx.zeros((4, D // 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="dt_bias"):
        flash_kda_mlx.fwd(**kwargs)


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------

def test_fwd_rejects_nan_scale():
    """scale must be finite."""
    kwargs = _base_inputs()
    kwargs["scale"] = float("nan")
    with pytest.raises(ValueError, match="scale"):
        flash_kda_mlx.fwd(**kwargs)


def test_fwd_rejects_inf_lower_bound():
    """lower_bound must be finite."""
    kwargs = _base_inputs()
    kwargs["lower_bound"] = float("-inf")
    with pytest.raises(ValueError, match="lower_bound"):
        flash_kda_mlx.fwd(**kwargs)


# ---------------------------------------------------------------------------
# cu_seqlens value validation
# ---------------------------------------------------------------------------

def test_fwd_rejects_cu_seqlens_not_starting_at_zero():
    """cu_seqlens[0] must equal 0."""
    inputs = make_varlen_inputs([16, 16], H=1, D=D)
    bad_cu = mx.array([1, 17, 32], dtype=mx.int32)
    out = mx.zeros(inputs["q"].shape, dtype=mx.float32)
    with pytest.raises(ValueError, match="cu_seqlens"):
        flash_kda_mlx.fwd(
            q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"],
            beta=inputs["beta"],
            scale=inputs["scale"], out=out,
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            lower_bound=inputs["lower_bound"],
            cu_seqlens=bad_cu,
        )


def test_fwd_rejects_cu_seqlens_wrong_total():
    """cu_seqlens[-1] must equal B*T."""
    inputs = make_varlen_inputs([16, 16], H=1, D=D)
    # Total is 32, give 30 instead.
    bad_cu = mx.array([0, 16, 30], dtype=mx.int32)
    out = mx.zeros(inputs["q"].shape, dtype=mx.float32)
    with pytest.raises(ValueError, match="cu_seqlens"):
        flash_kda_mlx.fwd(
            q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"],
            beta=inputs["beta"],
            scale=inputs["scale"], out=out,
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            lower_bound=inputs["lower_bound"],
            cu_seqlens=bad_cu,
        )


def test_fwd_rejects_non_monotonic_cu_seqlens():
    """cu_seqlens must be non-decreasing."""
    # Need total = B*T=32 and start at 0, but have a dip in between.
    inputs = make_varlen_inputs([16, 16], H=1, D=D)
    bad_cu = mx.array([0, 20, 16, 32], dtype=mx.int32)
    # Rebuild inputs with N=3 cu_seqlens, T=32.
    out = mx.zeros(inputs["q"].shape, dtype=mx.float32)
    with pytest.raises(ValueError, match="cu_seqlens"):
        flash_kda_mlx.fwd(
            q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"],
            beta=inputs["beta"],
            scale=inputs["scale"], out=out,
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            lower_bound=inputs["lower_bound"],
            cu_seqlens=bad_cu,
        )


# ---------------------------------------------------------------------------
# __all__ / public surface
# ---------------------------------------------------------------------------

def test_public_surface():
    """flash_kda_mlx exports fwd, FwdResult, and Backend."""
    assert "fwd" in flash_kda_mlx.__all__
    assert "FwdResult" in flash_kda_mlx.__all__
    assert "Backend" in flash_kda_mlx.__all__
    assert hasattr(flash_kda_mlx, "fwd")
    assert hasattr(flash_kda_mlx, "FwdResult")
    assert hasattr(flash_kda_mlx, "Backend")
