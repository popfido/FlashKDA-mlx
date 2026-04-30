"""CHUNK=32 prototype guard (plan.md §"CHUNK = 32 or 64 prototype").

Asserts three things and nothing tighter:

1. **Finiteness.** The CHUNK=32 path must not produce NaN or inf for any
   input the CHUNK=16 path handles. This is the one hard parity guarantee
   — the ``_ex2_pos_safe`` clamp at CHUNK=32 (see ``optimized.py``) exists
   precisely to avoid the ``0 * inf = NaN`` path triggered by
   ``lower_bound=-5`` spanning 32 tokens of cumulative decay.

2. **Bounded mean-drift.** Mean absolute deviation vs. CHUNK=16 stays
   inside ``MEAN_DRIFT_BUDGET``. Individual elements can drift by several
   bf16 ULPs at chunk boundaries — this is the **intrinsic cost** of
   halving the state-update cadence (plan.md §"CHUNK = 32"), not a bug.
   Tight per-element parity would require regenerating the
   ``torch_ref_cpu.py`` oracle at CHUNK=32 as well.

3. **Shape/dtype preservation.** Return shapes/dtypes at CHUNK=32 are
   identical to CHUNK=16, so the opt-in env var is a drop-in for any
   consumer whose tolerance envelope can absorb the drift.

The measured drift envelope at ``lower_bound=-5`` across the case matrix
below (M3 Max, MLX 0.31.2):

- ``max |Δ|``  roughly 2–5e-2 (~6 bf16 ULPs at typical-magnitude entries)
- ``mean |Δ|`` roughly 6–8e-4

The test tolerances below are set 2× the observed max to allow some
headroom for MLX-dispatcher reorder noise without masking a real
regression.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from _helpers import make_inputs, make_varlen_inputs, to_numpy

flash_kda_mlx_optimized = pytest.importorskip(
    "flash_kda_mlx.optimized",
    reason="flash_kda_mlx.optimized not importable.",
)

D = 128

# Observed ``out`` drift is ~7e-4 mean / ~5e-2 max; budget is ~1.5× mean and
# ~2× max to absorb dispatcher reorder without masking a regression.
MEAN_DRIFT_BUDGET = 1.0e-3
MAX_DRIFT_BUDGET = 1.0e-1

# Final-state drift is an order of magnitude wider than ``out`` because the
# per-chunk ``g_total`` at ``lower_bound=-5`` crosses the fp32-subnormal
# ftz boundary (~2^-126) for CHUNK=32 but not CHUNK=16. Consequence: the
# ``state_bf * g_total_exp`` decay term becomes **literally zero** at
# CHUNK=32 where CHUNK=16 keeps a ~2^-115 contribution. That changes the
# long-horizon state semantics, not just its last-ULP precision. Budget
# reflects the measured envelope (~2.5e-2 mean / ~2e-1 max). Consumers who
# rely on state for long-context continuity should stay on CHUNK=16.
STATE_MEAN_DRIFT_BUDGET = 5.0e-2
STATE_MAX_DRIFT_BUDGET = 1.0  # Some positions can flip from ~O(1) value to 0.


def _call_opt(kwargs: dict, chunk: int):
    return flash_kda_mlx_optimized.fwd_optimized(
        q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
        g=kwargs["g"], beta=kwargs["beta"],
        scale=kwargs["scale"], out_like=kwargs["out"],
        A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        initial_state=kwargs.get("initial_state"),
        final_state_like=kwargs.get("final_state"),
        cu_seqlens=kwargs.get("cu_seqlens"),
        _chunk=chunk,
    )


def _assert_finite_and_bounded(
    out16, out32, label, *,
    mean_budget: float = MEAN_DRIFT_BUDGET,
    max_budget: float = MAX_DRIFT_BUDGET,
):
    v16 = to_numpy(out16).astype(np.float32)
    v32 = to_numpy(out32).astype(np.float32)
    assert np.all(np.isfinite(v32)), f"{label}: CHUNK=32 produced non-finite"
    assert v16.shape == v32.shape, f"{label}: shape mismatch {v16.shape} vs {v32.shape}"

    diff = np.abs(v16 - v32)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0

    assert mean_abs <= mean_budget, (
        f"{label}: mean drift {mean_abs:.2e} exceeds budget {mean_budget:.0e}"
    )
    assert max_abs <= max_budget, (
        f"{label}: max drift {max_abs:.2e} exceeds budget {max_budget:.0e}"
    )


# ---------------------------------------------------------------------------
# Validation: bad values rejected
# ---------------------------------------------------------------------------

def test_chunk_arg_rejects_invalid():
    """The ``_chunk`` kwarg is an allowlist; 17 is not 16 or 32."""
    inp = make_inputs(T=16, H=1, D=D, seed=0)
    kw = {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"], "scale": inp["scale"],
        "out": mx.zeros((1, 16, 1, D), dtype=mx.float32),
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
    }
    with pytest.raises(ValueError, match="MLX_KDA_CHUNK"):
        _call_opt(kw, chunk=17)


# ---------------------------------------------------------------------------
# Finiteness + drift vs. CHUNK=16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [16, 32, 64, 97, 256, 1024])
@pytest.mark.parametrize("H", [1, 4])
def test_chunk32_vs_chunk16_fixed_no_state(T, H):
    """Fixed-length, no state — main drift envelope."""
    inp = make_inputs(T=T, H=H, D=D, seed=0)
    kw = {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"], "scale": inp["scale"],
        "out": mx.zeros((1, T, H, D), dtype=mx.float32),
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
    }
    out16, _ = _call_opt(kw, chunk=16)
    out32, _ = _call_opt(kw, chunk=32)
    _assert_finite_and_bounded(out16, out32, label=f"fixed T={T} H={H}")


@pytest.mark.parametrize("T,H", [(64, 1), (256, 4), (1024, 4)])
def test_chunk32_vs_chunk16_fixed_with_state(T, H):
    """State in+out path exercises both out and final_state."""
    rng = np.random.default_rng(11)
    init = rng.standard_normal((1, H, D, D)).astype(np.float32) * 0.1
    inp = make_inputs(T=T, H=H, D=D, seed=1)
    kw = {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"], "scale": inp["scale"],
        "out": mx.zeros((1, T, H, D), dtype=mx.float32),
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
        "initial_state": mx.array(init, dtype=mx.float32),
        "final_state": mx.zeros((1, H, D, D), dtype=mx.float32),
    }
    out16, final16 = _call_opt(kw, chunk=16)
    out32, final32 = _call_opt(kw, chunk=32)
    _assert_finite_and_bounded(out16, out32, label=f"fixed+state out T={T} H={H}")
    assert final16 is not None and final32 is not None
    _assert_finite_and_bounded(
        final16, final32, label=f"fixed+state final T={T} H={H}",
        mean_budget=STATE_MEAN_DRIFT_BUDGET,
        max_budget=STATE_MAX_DRIFT_BUDGET,
    )


@pytest.mark.parametrize("seq_lens", [[37, 16, 97, 64], [64, 64, 64, 64]])
@pytest.mark.parametrize("H", [1, 4])
def test_chunk32_vs_chunk16_varlen(seq_lens, H):
    """Varlen path — packed path at CHUNK=32 must also stay finite."""
    inp = make_varlen_inputs(seq_lens, H=H, D=D, seed=3)
    T_total = sum(seq_lens)
    kw = {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"], "scale": inp["scale"],
        "out": mx.zeros((1, T_total, H, D), dtype=mx.float32),
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
        "cu_seqlens": inp["cu_seqlens"],
    }
    out16, _ = _call_opt(kw, chunk=16)
    out32, _ = _call_opt(kw, chunk=32)
    _assert_finite_and_bounded(out16, out32, label=f"varlen {seq_lens} H={H}")


# ---------------------------------------------------------------------------
# dtype/shape preservation
# ---------------------------------------------------------------------------

def test_chunk32_preserves_output_dtype():
    inp = make_inputs(T=128, H=2, D=D, seed=7)
    kw = {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"], "scale": inp["scale"],
        "out": mx.zeros((1, 128, 2, D), dtype=mx.float32),
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
    }
    out16, _ = _call_opt(kw, chunk=16)
    out32, _ = _call_opt(kw, chunk=32)
    assert out16.dtype == out32.dtype
    assert out16.shape == out32.shape


def test_chunk32_preserves_state_dtype_bf16():
    H = 2
    rng = np.random.default_rng(17)
    init = rng.standard_normal((1, H, D, D)).astype(np.float32) * 0.1
    inp = make_inputs(T=64, H=H, D=D, seed=8)
    kw = {
        "q": inp["q"], "k": inp["k"], "v": inp["v"], "g": inp["g"],
        "beta": inp["beta"], "scale": inp["scale"],
        "out": mx.zeros((1, 64, H, D), dtype=mx.float32),
        "A_log": inp["A_log"], "dt_bias": inp["dt_bias"],
        "lower_bound": inp["lower_bound"],
        "initial_state": mx.array(init, dtype=mx.bfloat16),
        "final_state": mx.zeros((1, H, D, D), dtype=mx.bfloat16),
    }
    _, final16 = _call_opt(kw, chunk=16)
    _, final32 = _call_opt(kw, chunk=32)
    assert final16 is not None and final32 is not None
    assert final16.dtype == final32.dtype == mx.bfloat16
    assert final16.shape == final32.shape
