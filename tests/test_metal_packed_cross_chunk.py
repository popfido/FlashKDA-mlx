"""Phase 4 tests for the packed cross-chunk Metal recurrence kernel.

Phase 4 extends the Phase 3b simdgroup cross-chunk kernel with an extra
``N`` (sequence) leading axis plus an ``active_mask`` (derived from
``n_chunks_per_seq``) to match the packed variant's state-freeze
semantics documented in ``STATUS.md`` §"Cross-sequence varlen packing
(Option A — mask-based)".

Correctness oracle: the Python ``_mx_compiled_body_packed`` applied
chunk-by-chunk in a Python loop, from ``flash_kda_mlx.optimized``. Same
arithmetic, same bf16-cast boundaries, same freeze rule.

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

_PACKED = getattr(_mr, "metal_recurrence_cross_chunk_packed", None)


# ---------------------------------------------------------------------------
# Reference: Python loop over mx.compile body
# ---------------------------------------------------------------------------

def _mx_loop_reference_packed(
    state_in: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
    n_chunks_per_seq: mx.array,
) -> tuple[mx.array, mx.array]:
    """Python oracle: replicates ``_run_packed``'s inner loop using the
    ``mx.compile``-wrapped packed body (same semantics that
    ``MLX_KDA_ENABLE_METAL_RECURRENCE=0`` executes).
    """
    from flash_kda_mlx.optimized import _recurrence_body_packed  # mx.compile version

    N = state_in.shape[0]
    max_chunks = k_decayed.shape[1]
    H = k_decayed.shape[2]
    chunk = k_decayed.shape[3]
    D = k_decayed.shape[4]
    state = state_in
    outs: list[mx.array] = []
    for c in range(max_chunks):
        active = (n_chunks_per_seq > c).astype(mx.float32).reshape(N, 1, 1, 1)
        out_h, state = _recurrence_body_packed(
            state,
            k_decayed[:, c],
            q_decayed[:, c],
            k_restored[:, c],
            Mqk[:, c],
            INV_bf[:, c],
            vc[:, c],
            beta_bf16[:, c],
            g_total_exp[:, c],
            active,
        )
        outs.append(out_h)
    # PR H follow-on 3 Phase A: the packed cross-chunk simdgroup kernel
    # writes directly to [N, padded_T, H, D] layout, so the reference must
    # materialize the same layout for parity.
    out_stack = mx.stack(outs, axis=1)  # [N, max_chunks, H, CHUNK, D]
    out_all = out_stack.transpose(0, 1, 3, 2, 4).reshape(
        N, max_chunks * chunk, H, D
    )
    return out_all, state


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _make_inputs(
    N: int, max_chunks: int, H: int, seed: int
) -> dict[str, mx.array]:
    mx.random.seed(seed)
    s = 0.1
    return {
        "state_in":    _q_bf16(mx.random.normal((N, H, D, D)) * s),
        "k_decayed":   _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, D)) * s),
        "q_decayed":   _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, D)) * s),
        "k_restored":  _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, D)) * s),
        "Mqk":         _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, CHUNK)) * s),
        "INV_bf":      _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, CHUNK)) * s),
        "vc":          _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, D)) * s),
        "beta_bf16":   _q_bf16(mx.random.normal((N, max_chunks, H, CHUNK, 1)) * s),
        "g_total_exp": _q_bf16(mx.abs(mx.random.normal((N, max_chunks, H, D, 1))) * s),
    }


def _call_packed(fn, data: dict[str, mx.array], n_chunks_per_seq: mx.array):
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
        n_chunks_per_seq,
    )


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

# Packed parity uses the looser ULP-wide band the existing packed tests
# use (test_optimized_parity.STRESS_ATOL): MLX matmul dispatcher reorder
# on N-axis batches is known to cost at most one bf16 ULP.
PACKED_ATOL = 5e-5
PACKED_RTOL = 1e-4


PARITY_CASES: list[tuple[int, int, int, tuple[int, ...], int]] = [
    # (N, max_chunks, H, n_chunks_per_seq, seed)
    (1, 1, 1, (1,), 0),
    (2, 2, 1, (1, 2), 1),      # sequence 0 frozen after chunk 1
    (2, 4, 4, (3, 4), 2),
    (4, 4, 4, (2, 4, 3, 1), 3),
    (8, 6, 4, (1, 2, 3, 4, 5, 6, 6, 6), 4),
    (3, 8, 2, (8, 4, 1), 5),
]


@requires_metal
@pytest.mark.skipif(_PACKED is None, reason="Phase 4 kernel not yet implemented")
@pytest.mark.parametrize("N, max_chunks, H, n_chunks_per_seq, seed", PARITY_CASES)
def test_packed_parity(N, max_chunks, H, n_chunks_per_seq, seed):
    data = _make_inputs(N, max_chunks, H, seed)
    ncs = mx.array(n_chunks_per_seq, dtype=mx.int32)
    mx.eval(*data.values(), ncs)

    ref_out, ref_state = _mx_loop_reference_packed(**data, n_chunks_per_seq=ncs)
    got_out, got_state = _call_packed(_PACKED, data, ncs)
    mx.eval(ref_out, ref_state, got_out, got_state)

    # Compare only the ACTIVE chunk slots of out_h per sequence — padded-
    # tail chunks hold garbage in both paths (the reference path produces
    # whatever falls out of computing with masked state; the Metal kernel
    # does the same work unconditionally). The caller trims via seq_lens.
    # Layout is now [N, padded_T, H, D] so per-seq slicing is
    # ``[n, :nc*CHUNK]`` (PR H follow-on 3 Phase A).
    ref_out_np = np.asarray(ref_out)
    got_out_np = np.asarray(got_out)
    for n in range(N):
        nc = int(n_chunks_per_seq[n])
        if nc == 0:
            continue
        np.testing.assert_allclose(
            got_out_np[n, : nc * CHUNK],
            ref_out_np[n, : nc * CHUNK],
            rtol=PACKED_RTOL, atol=PACKED_ATOL,
            err_msg=f"out_h seq {n}: N={N} max_chunks={max_chunks} H={H}",
        )

    ref_state_np = np.asarray(ref_state)
    got_state_np = np.asarray(got_state)
    np.testing.assert_allclose(
        got_state_np, ref_state_np,
        rtol=PACKED_RTOL, atol=PACKED_ATOL,
        err_msg=f"new_state: N={N} max_chunks={max_chunks} H={H}",
    )


# ---------------------------------------------------------------------------
# State-freeze regression guard
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.skipif(_PACKED is None, reason="Phase 4 kernel not yet implemented")
def test_packed_state_frozen_matches_input_for_zero_chunks():
    """Sequence with n_chunks_per_seq=0 must have state == state_in."""
    N, max_chunks, H = 3, 4, 2
    data = _make_inputs(N, max_chunks, H, seed=99)
    ncs = mx.array([0, 2, 0], dtype=mx.int32)
    mx.eval(*data.values(), ncs)

    _, got_state = _call_packed(_PACKED, data, ncs)
    mx.eval(got_state)

    # Seq 0 & 2: state stays at state_in exactly (no chunks consumed).
    # Values are bf16-valued fp32 in both sides, so equality is bit-exact.
    np.testing.assert_array_equal(
        np.asarray(got_state)[[0, 2]],
        np.asarray(data["state_in"])[[0, 2]],
    )


@requires_metal
@pytest.mark.skipif(_PACKED is None, reason="Phase 4 kernel not yet implemented")
def test_packed_state_frozen_matches_last_active_chunk():
    """For seq n with n_chunks_per_seq[n]=k, state after the kernel must
    equal the state produced by running the first k chunks of that seq.

    Regression guard against padded-chunk state leakage.
    """
    N, max_chunks, H = 4, 5, 2
    data = _make_inputs(N, max_chunks, H, seed=123)
    ncs_full = mx.array([max_chunks] * N, dtype=mx.int32)
    ncs_half = mx.array([2, 3, 1, 4], dtype=mx.int32)
    mx.eval(*data.values(), ncs_full, ncs_half)

    # Reference: run with ncs_half — state frozen past each seq's limit.
    _, ref_state = _mx_loop_reference_packed(**data, n_chunks_per_seq=ncs_half)
    # Metal: same.
    _, got_state = _call_packed(_PACKED, data, ncs_half)
    mx.eval(ref_state, got_state)

    np.testing.assert_allclose(
        np.asarray(got_state), np.asarray(ref_state),
        rtol=PACKED_RTOL, atol=PACKED_ATOL,
    )


# ---------------------------------------------------------------------------
# Dtype / shape assertions
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.skipif(_PACKED is None, reason="Phase 4 kernel not yet implemented")
def test_packed_rejects_wrong_chunk():
    N, max_chunks, H, bad_chunk = 2, 1, 1, 32
    mx.random.seed(0)
    state_in = _q_bf16(mx.random.normal((N, H, D, D)) * 0.1)
    k = _q_bf16(mx.random.normal((N, max_chunks, H, bad_chunk, D)) * 0.1)
    zeros_c = mx.zeros((N, max_chunks, H, bad_chunk, bad_chunk), dtype=mx.float32)
    zeros_beta = mx.zeros((N, max_chunks, H, bad_chunk, 1), dtype=mx.float32)
    zeros_gte = mx.zeros((N, max_chunks, H, D, 1), dtype=mx.float32)
    ncs = mx.array([max_chunks] * N, dtype=mx.int32)
    with pytest.raises(AssertionError):
        _PACKED(
            state_in, k, k, k, zeros_c, zeros_c, k, zeros_beta, zeros_gte, ncs,
        )
