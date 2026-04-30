"""Regression tests for the CUDA-corresponding benchmark harness (plan PR B).

These tests pin ``build_call_kwargs`` to the input-construction contract of
``benchmarks/bench_fwd.py`` so that the generated ``BENCHMARK_MLX.md``
is column-for-column comparable to the CUDA ``BENCHMARK_H20.md`` report.

The harness exposes a ``cuda_correspond`` flag:

* ``cuda_correspond=True`` (default): bf16 ``q/k/v/g/beta`` + bf16 ``out`` +
  ``initial_state = arange(N*H*D*D).reshape(N,H,D,D).to(bf16)``, bf16
  ``final_state``, an ``initial_state_fp32`` companion for FLA baselines,
  and an independent deterministic scalar GDN gate ``g_gdn``.
* ``cuda_correspond=False``: legacy fp32 convenience path used during
  MLX-only development; no ``g_gdn`` / ``initial_state_fp32``.

These tests must pass before the benchmark report can be treated as a
reproduction of the CUDA table. See ``plan.md`` §PR B.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks._harness import Case, build_call_kwargs  # noqa: E402

D = 128


def _to_np(a: mx.array) -> np.ndarray:
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    mx.eval(a)
    return np.asarray(a)


# ---------------------------------------------------------------------------
# cuda_correspond=True (default) — must match benchmarks/bench_fwd.py
# ---------------------------------------------------------------------------

def _fixed_case(T: int = 64, H: int = 4, has_state: bool = True) -> Case:
    return Case(
        name=f"fixed_T{T}_H{H}" + ("_state" if has_state else ""),
        kind="fixed",
        T=T, H=H,
        has_state=has_state,
    )


def _varlen_case(seq_lens: tuple[int, ...], H: int = 4,
                 has_state: bool = True) -> Case:
    return Case(
        name=f"varlen_T{sum(seq_lens)}_H{H}_N{len(seq_lens)}"
             + ("_state" if has_state else ""),
        kind="varlen",
        T=0, H=H,
        seq_lens=seq_lens,
        has_state=has_state,
    )


def test_default_is_cuda_correspond():
    """The plan calls for CUDA correspondence as the primary path."""
    kwargs = build_call_kwargs(_fixed_case(), seed=0)
    # If default were legacy fp32, q would be fp32; we require bf16.
    assert kwargs["q"].dtype == mx.bfloat16


@pytest.mark.parametrize("name", ["q", "k", "v", "g", "beta"])
def test_cuda_correspond_qkvgbeta_bf16(name):
    kwargs = build_call_kwargs(_fixed_case(), seed=0)
    assert kwargs[name].dtype == mx.bfloat16, (
        f"{name} must be bf16 under cuda_correspond=True "
        f"(matches bench_fwd.py), got {kwargs[name].dtype}"
    )


def test_cuda_correspond_A_log_and_dt_bias_fp32():
    """CUDA bench leaves A_log/dt_bias in fp32 regardless of q/k/v dtype."""
    kwargs = build_call_kwargs(_fixed_case(), seed=0)
    assert kwargs["A_log"].dtype == mx.float32
    assert kwargs["dt_bias"].dtype == mx.float32


def test_cuda_correspond_out_is_bf16_zeros_like_q():
    kwargs = build_call_kwargs(_fixed_case(T=64, H=4), seed=0)
    out = kwargs["out"]
    assert out.dtype == mx.bfloat16
    assert out.shape == (1, 64, 4, D)
    # out = zeros_like(q)
    assert np.allclose(_to_np(out), 0.0)


def test_cuda_correspond_initial_state_is_bf16_arange():
    """CUDA bench: initial_state = arange(N*H*D*D).reshape(N,H,D,D).to(bf16).

    After bf16 round-trip, small integers (0..255) survive exactly.
    """
    case = _fixed_case(T=64, H=4, has_state=True)
    kwargs = build_call_kwargs(case, seed=0)
    state = kwargs["initial_state"]
    assert state.dtype == mx.bfloat16
    assert state.shape == (1, 4, D, D)
    flat = _to_np(state).reshape(-1)
    # First bf16-exact values must equal the integers 0..255.
    assert np.allclose(flat[:256], np.arange(256, dtype=np.float32))


def test_cuda_correspond_initial_state_varlen_uses_N():
    case = _varlen_case(seq_lens=(37, 16, 97, 64), H=4, has_state=True)
    kwargs = build_call_kwargs(case, seed=0)
    state = kwargs["initial_state"]
    assert state.dtype == mx.bfloat16
    assert state.shape == (4, 4, D, D)  # N=4, H=4
    # First block is still arange from 0.
    flat = _to_np(state).reshape(-1)
    assert float(flat[0]) == 0.0
    assert float(flat[1]) == 1.0


def test_cuda_correspond_final_state_is_bf16_zeros_matching_initial():
    kwargs = build_call_kwargs(_fixed_case(has_state=True), seed=0)
    fs = kwargs["final_state"]
    is_ = kwargs["initial_state"]
    assert fs.dtype == is_.dtype == mx.bfloat16
    assert fs.shape == is_.shape
    assert np.allclose(_to_np(fs), 0.0)


def test_cuda_correspond_initial_state_fp32_companion():
    """FLA baselines receive ``initial_state.float()`` — we pre-cast this as
    ``initial_state_fp32`` to avoid per-iter work inside the timed region."""
    kwargs = build_call_kwargs(_fixed_case(has_state=True), seed=0)
    assert "initial_state_fp32" in kwargs
    is_fp32 = kwargs["initial_state_fp32"]
    assert is_fp32.dtype == mx.float32
    # Values must equal initial_state.astype(fp32) exactly (bf16 → fp32 is lossless).
    np.testing.assert_array_equal(
        _to_np(is_fp32), _to_np(kwargs["initial_state"].astype(mx.float32)),
    )


def test_cuda_correspond_no_state_omits_state_keys():
    kwargs = build_call_kwargs(_fixed_case(has_state=False), seed=0)
    assert "initial_state" not in kwargs
    assert "final_state" not in kwargs
    assert "initial_state_fp32" not in kwargs


def test_cuda_correspond_g_gdn_shape_and_dtype():
    T = 64
    H = 4
    kwargs = build_call_kwargs(_fixed_case(T=T, H=H), seed=0)
    g_gdn = kwargs["g_gdn"]
    assert g_gdn.shape == (1, T, H)
    assert g_gdn.dtype == mx.float32


def test_cuda_correspond_g_gdn_varlen_uses_total_tokens():
    seq_lens = (37, 16, 97, 64)
    H = 4
    kwargs = build_call_kwargs(_varlen_case(seq_lens, H=H), seed=0)
    g_gdn = kwargs["g_gdn"]
    assert g_gdn.shape == (1, sum(seq_lens), H)
    assert g_gdn.dtype == mx.float32


def test_cuda_correspond_g_gdn_is_independent_of_g():
    """g_gdn must NOT be a reduction of g — it's an independent draw.

    The previous derivation ``g_gdn = mean(g, axis=-1)`` gave every method
    in a case a shared random seed, which (a) couples baseline measurements
    to `g`'s statistics and (b) diverges from CUDA bench's independent
    ``torch.randn((1, T_total, H), dtype=fp32)``.
    """
    kwargs = build_call_kwargs(_fixed_case(T=256, H=4), seed=0)
    g_gdn = _to_np(kwargs["g_gdn"])
    g = _to_np(kwargs["g"])
    g_reduced = g.astype(np.float32).mean(axis=-1)
    # If g_gdn is truly independent it must differ materially from any
    # reduction of g. We assert the L2 distance is far above bf16 ULP noise.
    diff = np.linalg.norm(g_gdn - g_reduced)
    assert diff > 1.0, (
        f"g_gdn ({g_gdn.mean():.4f}/{g_gdn.std():.4f}) is suspiciously close "
        f"to mean(g, -1) ({g_reduced.mean():.4f}/{g_reduced.std():.4f}); "
        f"diff L2 = {diff:.4f}. g_gdn must be an independent draw."
    )


def test_cuda_correspond_g_gdn_determinism():
    """Same seed must yield bit-identical g_gdn."""
    a = build_call_kwargs(_fixed_case(), seed=7)["g_gdn"]
    b = build_call_kwargs(_fixed_case(), seed=7)["g_gdn"]
    np.testing.assert_array_equal(_to_np(a), _to_np(b))


def test_cuda_correspond_g_gdn_seed_sensitivity():
    """Different seeds must yield different g_gdn (confirms seed is wired)."""
    a = build_call_kwargs(_fixed_case(), seed=7)["g_gdn"]
    b = build_call_kwargs(_fixed_case(), seed=8)["g_gdn"]
    assert not np.array_equal(_to_np(a), _to_np(b))


def test_cuda_correspond_varlen_cu_seqlens_preserved():
    seq_lens = (37, 16, 97, 64)
    kwargs = build_call_kwargs(_varlen_case(seq_lens), seed=0)
    assert "cu_seqlens" in kwargs
    cu = _to_np(kwargs["cu_seqlens"]).astype(np.int64)
    expected = np.zeros(len(seq_lens) + 1, dtype=np.int64)
    expected[1:] = np.cumsum(seq_lens)
    np.testing.assert_array_equal(cu, expected)


# ---------------------------------------------------------------------------
# cuda_correspond=False — legacy fp32 convenience path preserved
# ---------------------------------------------------------------------------

def test_legacy_mode_keeps_fp32_and_random_state():
    kwargs = build_call_kwargs(_fixed_case(has_state=True), seed=0,
                               cuda_correspond=False)
    for name in ("q", "k", "v", "g", "beta", "out", "initial_state",
                 "final_state"):
        assert kwargs[name].dtype == mx.float32, f"{name} must stay fp32"
    # Legacy state is randn * 0.1 — not arange — so values are in ~[-1, 1]·0.1.
    state = _to_np(kwargs["initial_state"])
    assert abs(state).max() < 2.0  # loose sanity bound


def test_legacy_mode_omits_cuda_only_keys():
    kwargs = build_call_kwargs(_fixed_case(has_state=True), seed=0,
                               cuda_correspond=False)
    assert "g_gdn" not in kwargs
    assert "initial_state_fp32" not in kwargs
