"""Regression guard: ``fwd_optimized`` must match ``fwd_reference`` tightly.

The optimized path is a *structural* transformation of the reference — same
math, same bf16 cast boundaries, same chunk-local numerics — so tolerances
here are orders of magnitude tighter than the oracle-parity tolerances in
``test_parity_fixtures.py``. Any drift larger than ``PARITY_ATOL`` means the
optimized path has silently changed semantics.

If ``flash_kda_mlx.optimized`` is not yet importable, the whole module is skipped
so this guard can land in a RED state without breaking CI.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from _helpers import make_inputs, make_varlen_inputs, to_numpy

flash_kda_mlx_optimized = pytest.importorskip(
    "flash_kda_mlx.optimized",
    reason="flash_kda_mlx.optimized not implemented yet (PR6).",
)
from flash_kda_mlx import reference  # noqa: E402


D = 128

# Tolerances for optimized↔reference parity. The two paths evaluate the same
# expression graph at the same fp32/bf16/fp16 cast boundaries, so we expect
# bit-exact equality modulo MLX dispatcher reordering of independent matmuls.
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-5


def _run_reference(kwargs):
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


def _run_optimized(kwargs):
    return flash_kda_mlx_optimized.fwd_optimized(
        q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
        g=kwargs["g"], beta=kwargs["beta"],
        scale=kwargs["scale"], out_like=kwargs["out"],
        A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        initial_state=kwargs.get("initial_state"),
        final_state_like=kwargs.get("final_state"),
        cu_seqlens=kwargs.get("cu_seqlens"),
    )


def _assert_tight(actual, expected, label):
    act = to_numpy(actual).astype(np.float32)
    exp = to_numpy(expected).astype(np.float32)
    diff = np.abs(act - exp)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / (np.abs(exp) + 1e-12)).max()) if diff.size else 0.0
    np.testing.assert_allclose(
        act, exp,
        rtol=PARITY_RTOL, atol=PARITY_ATOL,
        err_msg=f"{label}: max_abs={max_abs:.2e} max_rel={max_rel:.2e}",
    )


# ---------------------------------------------------------------------------
# Fixed length, no state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [16, 17, 64, 97, 256])
@pytest.mark.parametrize("H", [1, 4])
def test_parity_fixed_no_state(T, H):
    inputs = make_inputs(T=T, H=H, D=D, seed=0)
    kwargs = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
    }
    ref_out, _ = _run_reference(kwargs)
    opt_out, _ = _run_optimized(kwargs)
    _assert_tight(opt_out, ref_out, label=f"out T={T} H={H}")


# ---------------------------------------------------------------------------
# Fixed length, with state (in+out, both dtypes)
# ---------------------------------------------------------------------------

def _initial_state(N, H, seed, dtype):
    rng = np.random.default_rng(seed + 777)
    arr = rng.standard_normal((N, H, D, D)).astype(np.float32) * 0.1
    return mx.array(arr, dtype=dtype)


@pytest.mark.parametrize("T,H", [(64, 1), (256, 4), (17, 4), (97, 1)])
@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16],
                          ids=["fp32", "bf16"])
def test_parity_fixed_with_state(T, H, state_dtype):
    inputs = make_inputs(T=T, H=H, D=D, seed=1)
    kwargs = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
        "initial_state": _initial_state(1, H, seed=2, dtype=state_dtype),
        "final_state": mx.zeros((1, H, D, D), dtype=state_dtype),
    }
    ref_out, ref_final = _run_reference(kwargs)
    opt_out, opt_final = _run_optimized(kwargs)
    _assert_tight(opt_out, ref_out, label=f"out T={T} H={H} {state_dtype}")
    assert opt_final is not None and ref_final is not None
    _assert_tight(opt_final, ref_final, label=f"final T={T} H={H} {state_dtype}")


# ---------------------------------------------------------------------------
# Varlen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_lens", [
    [16, 16],
    [37, 16, 97, 64],
    [1, 15, 16, 17],
])
@pytest.mark.parametrize("H", [1, 4])
def test_parity_varlen_no_state(seq_lens, H):
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=3)
    T_total = sum(seq_lens)
    kwargs = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T_total, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
        "cu_seqlens": inputs["cu_seqlens"],
    }
    ref_out, _ = _run_reference(kwargs)
    opt_out, _ = _run_optimized(kwargs)
    _assert_tight(opt_out, ref_out, label=f"varlen out {seq_lens} H={H}")


# ---------------------------------------------------------------------------
# Batched fixed-length (B > 1) — exercises the B>1 cu_seqlens branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,T,H", [(2, 64, 1), (2, 128, 4), (3, 64, 2)])
def test_parity_batched_with_state(B, T, H):
    """Multi-chunk, multi-sequence batched mode with state in+out.

    Catches any regression in inter-chunk state threading: T=64 has 4 chunks,
    T=128 has 8 chunks, all sharing a state across those chunks. If the
    optimized path's per-chunk state propagation is wrong, final_state will
    diverge from the reference far outside the one-bf16-ULP packed-path band.
    """
    inputs = make_inputs(T=T, H=H, D=D, seed=9)
    # Broadcast to a real B-dimension batch by tiling with a different seed
    # per-batch so each sequence sees independent work.
    tiled = {}
    seed_by_name = {"q": 101, "k": 102, "v": 103, "g": 104}
    for name in ("q", "k", "v", "g"):
        arr = to_numpy(inputs[name])  # [1, T, H, D]
        rng = np.random.default_rng(seed_by_name[name])
        extras = [arr] + [
            arr + rng.standard_normal(arr.shape).astype(np.float32) * 0.01
            for _ in range(B - 1)
        ]
        tiled[name] = mx.array(np.concatenate(extras, axis=0))
    beta_arr = to_numpy(inputs["beta"])
    rng = np.random.default_rng(42)
    beta_extras = [beta_arr] + [
        beta_arr + rng.standard_normal(beta_arr.shape).astype(np.float32) * 0.01
        for _ in range(B - 1)
    ]
    tiled["beta"] = mx.array(np.concatenate(beta_extras, axis=0))

    kwargs = {
        **tiled,
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "scale": inputs["scale"], "lower_bound": inputs["lower_bound"],
        "out": mx.zeros((B, T, H, D), dtype=mx.float32),
        "initial_state": _initial_state(B, H, seed=11, dtype=mx.float32),
        "final_state": mx.zeros((B, H, D, D), dtype=mx.float32),
    }
    ref_out, ref_final = _run_reference(kwargs)
    opt_out, opt_final = _run_optimized(kwargs)
    _assert_stress(opt_out, ref_out, label=f"batched out B={B} T={T} H={H}")
    _assert_stress(opt_final, ref_final, label=f"batched final B={B} T={T} H={H}")


@pytest.mark.parametrize("seq_lens", [[37, 16, 97, 64]])
@pytest.mark.parametrize("H", [1, 4])
def test_parity_varlen_with_state(seq_lens, H):
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=4)
    T_total = sum(seq_lens)
    N = len(seq_lens)
    kwargs = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T_total, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
        "cu_seqlens": inputs["cu_seqlens"],
        "initial_state": _initial_state(N, H, seed=5, dtype=mx.float32),
        "final_state": mx.zeros((N, H, D, D), dtype=mx.float32),
    }
    ref_out, ref_final = _run_reference(kwargs)
    opt_out, opt_final = _run_optimized(kwargs)
    _assert_tight(opt_out, ref_out, label=f"varlen out {seq_lens} H={H}")
    _assert_tight(opt_final, ref_final, label=f"varlen final {seq_lens} H={H}")


# ---------------------------------------------------------------------------
# Cross-sequence packed varlen stress (plan.md §Phase 8 next-step candidate).
# These exercise the multi-sequence packed pre-compute path that amortises
# setup over N>1 sequences. They use a slightly relaxed parity band
# (``STRESS_*``) because the packed path replaces many independent
# per-sequence matmuls with one batched matmul over a leading N axis; the
# MLX dispatcher may reorder those operands, and bf16 is non-associative,
# so one ULP of drift (~3e-5 near magnitude 1) can land on a tiny fraction
# of elements. That's the same dispatcher-order noise the H-axis batching
# already introduces, pushed one step deeper.
# ---------------------------------------------------------------------------

# ULP-sized slack above the 1e-5 bit-exact band used by the single-sequence
# tests above. Drift of this size is a dispatcher reorder, not a semantic bug.
STRESS_ATOL = 5e-5
STRESS_RTOL = 1e-4


def _assert_stress(actual, expected, label):
    act = to_numpy(actual).astype(np.float32)
    exp = to_numpy(expected).astype(np.float32)
    diff = np.abs(act - exp)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / (np.abs(exp) + 1e-12)).max()) if diff.size else 0.0
    np.testing.assert_allclose(
        act, exp,
        rtol=STRESS_RTOL, atol=STRESS_ATOL,
        err_msg=f"{label}: max_abs={max_abs:.2e} max_rel={max_rel:.2e}",
    )


@pytest.mark.parametrize("has_state", [False, True], ids=["no_state", "with_state"])
def test_parity_varlen_N8_mixed(has_state):
    """N=8 mixed lengths, H=4 — main target for cross-sequence packing."""
    seq_lens = [37, 16, 97, 64, 128, 17, 256, 80]
    H = 4
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=21)
    T_total = sum(seq_lens)
    N = len(seq_lens)
    kwargs = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T_total, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
        "cu_seqlens": inputs["cu_seqlens"],
    }
    if has_state:
        kwargs["initial_state"] = _initial_state(N, H, seed=22, dtype=mx.float32)
        kwargs["final_state"] = mx.zeros((N, H, D, D), dtype=mx.float32)

    ref_out, ref_final = _run_reference(kwargs)
    opt_out, opt_final = _run_optimized(kwargs)
    _assert_stress(opt_out, ref_out, label=f"N8 out state={has_state}")
    if has_state:
        assert opt_final is not None and ref_final is not None
        _assert_stress(opt_final, ref_final, label=f"N8 final state={has_state}")


def test_parity_varlen_N16_balanced():
    """N=16 balanced lengths (all 64) — degenerate case where packing has
    no padding overhead; every chunk is used by every sequence."""
    seq_lens = [64] * 16
    H = 4
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=31)
    T_total = sum(seq_lens)
    kwargs = {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"],
        "g": inputs["g"], "beta": inputs["beta"],
        "scale": inputs["scale"],
        "out": mx.zeros((1, T_total, H, D), dtype=mx.float32),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "lower_bound": inputs["lower_bound"],
        "cu_seqlens": inputs["cu_seqlens"],
    }
    ref_out, _ = _run_reference(kwargs)
    opt_out, _ = _run_optimized(kwargs)
    _assert_stress(opt_out, ref_out, label="N16 balanced out")


def test_parity_batched_B4_T256_H4():
    """Batched B=4 fixed T=256 H=4 — routes through the N>1 packed path now
    that cu_seqlens is synthesised from B for B>1 inputs."""
    B, T, H = 4, 256, 4
    inputs = make_inputs(T=T, H=H, D=D, seed=41)
    tiled = {}
    perturb_seeds = {"q": 51, "k": 52, "v": 53, "g": 54}
    for name in ("q", "k", "v", "g"):
        arr = to_numpy(inputs[name])  # [1, T, H, D]
        rng = np.random.default_rng(perturb_seeds[name])
        extras = [arr] + [
            arr + rng.standard_normal(arr.shape).astype(np.float32) * 0.01
            for _ in range(B - 1)
        ]
        tiled[name] = mx.array(np.concatenate(extras, axis=0))
    beta_arr = to_numpy(inputs["beta"])
    rng = np.random.default_rng(99)
    beta_extras = [beta_arr] + [
        beta_arr + rng.standard_normal(beta_arr.shape).astype(np.float32) * 0.01
        for _ in range(B - 1)
    ]
    tiled["beta"] = mx.array(np.concatenate(beta_extras, axis=0))

    kwargs = {
        **tiled,
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "scale": inputs["scale"], "lower_bound": inputs["lower_bound"],
        "out": mx.zeros((B, T, H, D), dtype=mx.float32),
    }
    ref_out, _ = _run_reference(kwargs)
    opt_out, _ = _run_optimized(kwargs)
    _assert_stress(opt_out, ref_out, label="B4 T256 H4 out")
