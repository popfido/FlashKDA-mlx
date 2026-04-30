"""Phase 3 tests for the cross-chunk Metal recurrence kernels.

Phase 3 collapses the per-chunk Python loop into a SINGLE Metal dispatch
per head. State lives in threadgroup memory across all chunks, eliminating
the per-chunk device memory round-trip that Phase 2 still pays.

Two routings are exercised:

* ``metal_recurrence_cross_chunk_scalar`` (Phase 3a) — scalar inner products
  cooperatively computed by 1024 threads per head, with simd_sum / direct
  element sums. Mirrors ``mlx-lm``'s ``gated_delta_kernel`` structure.
* ``metal_recurrence_cross_chunk_simdgroup`` (Phase 3b) — same architecture
  but matmuls tiled via ``simdgroup_matrix<float, 8, 8>``. Added in a
  later commit; the parametrization below references it so the tests turn
  on automatically once it lands.

Correctness oracle: the Python ``mx.compile``-wrapped
``_recurrence_body_single`` applied chunk-by-chunk in a Python loop, from
``flash_kda_mlx.optimized._mx_compiled_body_single``. Both paths share the
same arithmetic and bf16-cast boundaries, so parity is expected at
``rtol=atol=1e-5``.

Skipped on M1/M2 per plan §8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flash_kda_mlx import _metal_recurrence as _mr  # noqa: E402
from flash_kda_mlx._metal_recurrence import HAS_METAL_KERNEL  # noqa: E402
from flash_kda_mlx.reference import _q_bf16  # noqa: E402


D = 128
CHUNK = 16


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal cross-chunk path disabled (M3+ only, per plan §8)",
)


# Phase 3b (simdgroup) lands in a follow-up commit. Use getattr so the test
# file imports cleanly in the Phase-3a-only window, and the simdgroup
# parametrization skips until the symbol exists.
_SCALAR = getattr(_mr, "metal_recurrence_cross_chunk_scalar", None)
_SIMDGROUP = getattr(_mr, "metal_recurrence_cross_chunk_simdgroup", None)


ROUTINGS = []
if _SCALAR is not None:
    ROUTINGS.append(pytest.param(_SCALAR, id="scalar"))
if _SIMDGROUP is not None:
    ROUTINGS.append(pytest.param(_SIMDGROUP, id="simdgroup"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mx_loop_reference(
    state_in: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
) -> tuple[mx.array, mx.array]:
    """Reference: python loop over chunks calling the mx.compile body.

    ``_mx_compiled_body_single`` is the same body the Phase 2 kernel
    replaces, so using it here keeps the oracle on the same arithmetic
    path the optimized module uses when the Metal flag is OFF.
    """
    # Import lazily so the test module still loads when optimized.py has
    # the env-triggered routing toggles in a weird state.
    from flash_kda_mlx.optimized import _mx_compiled_body_single

    n_chunks = k_decayed.shape[0]
    H = k_decayed.shape[1]
    chunk = k_decayed.shape[2]
    D = k_decayed.shape[3]
    state = state_in
    outs: list[mx.array] = []
    for c in range(n_chunks):
        out_h, state = _mx_compiled_body_single(
            state,
            k_decayed[c],
            q_decayed[c],
            k_restored[c],
            Mqk[c],
            INV_bf[c],
            vc[c],
            beta_bf16[c],
            g_total_exp[c],
        )
        outs.append(out_h)
    # PR H follow-on 3 Phase A: the cross-chunk simdgroup kernel writes
    # directly to [T_total, H, D] (= [n_chunks*CHUNK, H, D]) layout, so
    # the reference must materialize the same layout for parity.
    out_stack = mx.stack(outs, axis=0)  # [n_chunks, H, CHUNK, D]
    out_all = out_stack.transpose(0, 2, 1, 3).reshape(
        n_chunks * chunk, H, D
    )
    return out_all, state


def _make_inputs(n_chunks: int, H: int, seed: int) -> dict[str, mx.array]:
    """Generate small, well-scaled random bf16-valued inputs.

    Matches the shape contract of ``_precompute_chunk_tensors`` after
    its ``[n_chunks, H, ...]`` leading layout, plus an initial state
    sized ``[H, D, D]``.

    Scale chosen so |state| stays bounded across ``n_chunks`` iterations
    — larger scales (e.g., ``mx.random.normal`` at unit variance) let
    ``delta_s + state * g_total_exp`` grow unboundedly over multi-chunk
    sequences, which stresses fp32 accumulation but not kernel logic.
    """
    mx.random.seed(seed)
    s = 0.1  # bounded dynamic range

    data = {
        "state_in":    _q_bf16(mx.random.normal((H, D, D)) * s),
        "k_decayed":   _q_bf16(mx.random.normal((n_chunks, H, CHUNK, D)) * s),
        "q_decayed":   _q_bf16(mx.random.normal((n_chunks, H, CHUNK, D)) * s),
        "k_restored":  _q_bf16(mx.random.normal((n_chunks, H, CHUNK, D)) * s),
        "Mqk":         _q_bf16(mx.random.normal((n_chunks, H, CHUNK, CHUNK)) * s),
        "INV_bf":      _q_bf16(mx.random.normal((n_chunks, H, CHUNK, CHUNK)) * s),
        "vc":          _q_bf16(mx.random.normal((n_chunks, H, CHUNK, D)) * s),
        "beta_bf16":   _q_bf16(mx.random.normal((n_chunks, H, CHUNK, 1)) * s),
        "g_total_exp": _q_bf16(mx.abs(mx.random.normal((n_chunks, H, D, 1))) * s),
    }
    mx.eval(*data.values())
    return data


def _call(fn, data: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
    return fn(
        data["state_in"],
        data["k_decayed"],
        data["q_decayed"],
        data["k_restored"],
        data["Mqk"],
        data["INV_bf"],
        data["vc"],
        data["beta_bf16"],
        data["g_total_exp"],
    )


# ---------------------------------------------------------------------------
# Parity — cross-chunk kernel must match Python-loop-over-compiled-body
# ---------------------------------------------------------------------------

PARITY_CASES: list[tuple[int, int, int]] = [
    # (n_chunks, H, seed)
    (1, 1, 0),   # single chunk, single head — exercises init-load + one body
    (2, 1, 1),   # carries state once across chunks
    (2, 4, 2),   # multi-head parallelism
    (4, 2, 3),   # multiple carries
    (8, 4, 4),   # deeper chunk loop
    (1, 8, 5),   # H=8 — same dispatch geometry, different occupancy
]


@requires_metal
@pytest.mark.parametrize("fn", ROUTINGS)
@pytest.mark.parametrize("n_chunks, H, seed", PARITY_CASES)
def test_cross_chunk_parity(fn, n_chunks, H, seed):
    data = _make_inputs(n_chunks, H, seed)

    ref_out, ref_state = _mx_loop_reference(**data)
    got_out, got_state = _call(fn, data)
    mx.eval(ref_out, ref_state, got_out, got_state)

    ref_out_np = np.asarray(ref_out)
    got_out_np = np.asarray(got_out)
    ref_state_np = np.asarray(ref_state)
    got_state_np = np.asarray(got_state)

    max_abs_out = float(np.abs(got_out_np - ref_out_np).max())
    max_abs_state = float(np.abs(got_state_np - ref_state_np).max())

    np.testing.assert_allclose(
        got_out_np, ref_out_np,
        rtol=1e-5, atol=1e-5,
        err_msg=(
            f"out_h drift: n_chunks={n_chunks} H={H} seed={seed} "
            f"max_abs={max_abs_out:.3e}"
        ),
    )
    np.testing.assert_allclose(
        got_state_np, ref_state_np,
        rtol=1e-5, atol=1e-5,
        err_msg=(
            f"new_state drift: n_chunks={n_chunks} H={H} seed={seed} "
            f"max_abs={max_abs_state:.3e}"
        ),
    )


@requires_metal
@pytest.mark.parametrize("fn", ROUTINGS)
def test_cross_chunk_parity_zeros_are_zeros(fn):
    """Trivial input — zeros in → zeros out + state preserved."""
    H, n_chunks = 2, 3
    z_state = mx.zeros((H, D, D), dtype=mx.float32)
    z_chunk_d = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
    z_chunk_c = mx.zeros((n_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
    z_beta = mx.zeros((n_chunks, H, CHUNK, 1), dtype=mx.float32)
    z_gte = mx.zeros((n_chunks, H, D, 1), dtype=mx.float32)

    out, state = fn(
        z_state, z_chunk_d, z_chunk_d, z_chunk_d,
        z_chunk_c, z_chunk_c, z_chunk_d, z_beta, z_gte,
    )
    mx.eval(out, state)
    assert np.all(np.asarray(out) == 0.0)
    # state stays zero: delta_s=0 + state*0 = 0 → q_bf = 0
    assert np.all(np.asarray(state) == 0.0)


@requires_metal
@pytest.mark.parametrize("fn", ROUTINGS)
def test_cross_chunk_parity_state_is_preserved_when_chunk_is_identity(fn):
    """When the chunk input is zero, state should pass through unchanged
    modulo the ``g_total_exp`` multiplier (which we set to 1 here).

    Regression guard for the cross-chunk carry.
    """
    H, n_chunks = 1, 2
    rng = np.random.default_rng(777)
    init = rng.standard_normal((H, D, D)).astype(np.float32) * 0.1
    init_bf = _q_bf16(mx.array(init))
    mx.eval(init_bf)

    z_chunk_d = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
    z_chunk_c = mx.zeros((n_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
    z_beta = mx.zeros((n_chunks, H, CHUNK, 1), dtype=mx.float32)
    ones_gte = mx.ones((n_chunks, H, D, 1), dtype=mx.float32)

    out, state = fn(
        init_bf, z_chunk_d, z_chunk_d, z_chunk_d,
        z_chunk_c, z_chunk_c, z_chunk_d, z_beta, ones_gte,
    )
    mx.eval(out, state)

    # U=0, delta_s=0 at each chunk; state_new = q_bf(0 + state * 1) = state.
    np.testing.assert_array_equal(
        np.asarray(state),
        np.asarray(init_bf),
    )
    # out is q_bf(q_bf(q_decayed@state_T) + q_bf(Mqk@U)); both are zero.
    assert np.all(np.asarray(out) == 0.0)


# ---------------------------------------------------------------------------
# Dtype / shape assertions
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("fn", ROUTINGS)
def test_cross_chunk_rejects_wrong_dtype(fn):
    H, n_chunks = 1, 1
    data = _make_inputs(n_chunks, H, seed=0)
    data["state_in"] = data["state_in"].astype(mx.bfloat16)
    with pytest.raises(AssertionError):
        _call(fn, data)


@requires_metal
@pytest.mark.parametrize("fn", ROUTINGS)
def test_cross_chunk_rejects_wrong_chunk(fn):
    """CHUNK≠16 is rejected — Phase 3 assumes the 16-chunk simdgroup grid."""
    H, n_chunks = 1, 1
    mx.random.seed(0)
    bad_chunk = 32
    state_in = _q_bf16(mx.random.normal((H, D, D)) * 0.1)
    k_decayed = _q_bf16(mx.random.normal((n_chunks, H, bad_chunk, D)) * 0.1)
    with pytest.raises(AssertionError):
        fn(
            state_in, k_decayed, k_decayed, k_decayed,
            mx.zeros((n_chunks, H, bad_chunk, bad_chunk), dtype=mx.float32),
            mx.zeros((n_chunks, H, bad_chunk, bad_chunk), dtype=mx.float32),
            k_decayed,
            mx.zeros((n_chunks, H, bad_chunk, 1), dtype=mx.float32),
            mx.zeros((n_chunks, H, D, 1), dtype=mx.float32),
        )


