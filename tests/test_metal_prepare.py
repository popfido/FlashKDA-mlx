"""Parity tests for the Metal chunk pre-compute kernel (PR H Branch A).

PHASE 1 STATUS
==============
The kernel currently writes zeros (skeleton). Phase 1 tests therefore
check ONLY that the kernel:

* compiles and dispatches without error,
* allocates outputs at the correct shapes and dtypes,
* preserves the ``vc`` passthrough.

The numeric parity tests against ``_precompute_core`` are
``@pytest.mark.xfail`` with a Phase 1 reason; Phase 2's arithmetic
implementation flips them to xpass.

Skipped on M1/M2 per ``HAS_METAL_KERNEL`` (same gate as the recurrence
kernels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flash_kda_mlx._metal_prepare import HAS_METAL_KERNEL, metal_prepare_chunk  # noqa: E402
from flash_kda_mlx.optimized import _precompute_core  # noqa: E402
from flash_kda_mlx.reference import _l2_normalize, _q_bf16  # noqa: E402


D = 128
CHUNK = 16


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal prepare kernel disabled (M3+ only)",
)


# ---------------------------------------------------------------------------
# Inputs that mirror what _run_single hands to _precompute_chunk_tensors:
# q/k are L2-normalised + bf16-quantised, g is post-activation, beta is
# pre-sigmoid.
# ---------------------------------------------------------------------------

def _make_inputs(n_chunks: int, H: int, seed: int) -> dict[str, mx.array]:
    mx.random.seed(seed)
    s = 0.1
    # q/k go through L2-norm + bf16 in the real path. Match that here so
    # the Phase 2 parity check sees a realistic input distribution.
    q_raw = mx.random.normal((n_chunks * CHUNK, H, D)) * s
    k_raw = mx.random.normal((n_chunks * CHUNK, H, D)) * s
    q = _q_bf16(_l2_normalize(q_raw))
    k = _q_bf16(_l2_normalize(k_raw))
    v = _q_bf16(mx.random.normal((n_chunks * CHUNK, H, D)) * s)
    # g: arbitrary post-activation; we use small scaled values so
    # cumsum * ex2 doesn't underflow the bf16 representation.
    g = -mx.abs(mx.random.normal((n_chunks * CHUNK, H, D))) * 0.1
    beta = mx.random.normal((n_chunks * CHUNK, H)) * s

    # Reshape to [n_chunks, H, CHUNK, D] / [n_chunks, H, CHUNK] as the
    # kernel expects (also matches the layout _precompute_chunk_tensors
    # produces internally before calling _precompute_core).
    def _to_chunks_hd(x: mx.array) -> mx.array:
        return x.reshape(n_chunks, CHUNK, H, D).transpose(0, 2, 1, 3)

    qc = _to_chunks_hd(q)
    kc = _to_chunks_hd(k)
    vc = _to_chunks_hd(v)
    gc = _to_chunks_hd(g)
    bc = beta.reshape(n_chunks, CHUNK, H).transpose(0, 2, 1)

    scale = _q_bf16(mx.array([0.125], dtype=mx.float32))[0]

    return {
        "k": kc, "q": qc, "v": vc, "g": gc, "beta": bc,
        "scale_bf16_rt": scale,
        # For the parity oracle (_precompute_core takes the same shapes
        # but as positional args).
        "_oracle_inputs": (gc, qc, kc, vc, bc),
    }


# ---------------------------------------------------------------------------
# Phase 1: structural checks
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("n_chunks, H", [(1, 1), (4, 2), (8, 4)])
def test_prepare_kernel_dispatches_and_allocates_shapes(n_chunks, H):
    """Phase 1 contract: kernel runs, outputs exist, shapes/dtypes match
    ``_precompute_core``'s return dict.
    """
    data = _make_inputs(n_chunks, H, seed=0)
    pre = metal_prepare_chunk(
        k=data["k"], q=data["q"], v=data["v"],
        g=data["g"], beta=data["beta"],
        scale_bf16_rt=data["scale_bf16_rt"],
    )

    expected_shapes = {
        "k_decayed":   (n_chunks, H, CHUNK, D),
        "q_decayed":   (n_chunks, H, CHUNK, D),
        "k_restored":  (n_chunks, H, CHUNK, D),
        "Mqk":         (n_chunks, H, CHUNK, CHUNK),
        "INV_bf":      (n_chunks, H, CHUNK, CHUNK),
        "vc":          (n_chunks, H, CHUNK, D),
        "beta_bf16":   (n_chunks, H, CHUNK, 1),
        "g_total_exp": (n_chunks, H, D, 1),
    }
    for name, shape in expected_shapes.items():
        assert name in pre, f"missing key {name}"
        assert pre[name].shape == shape, (
            f"{name}: got {pre[name].shape}, want {shape}"
        )
        assert pre[name].dtype == mx.float32, (
            f"{name}: got {pre[name].dtype}, want fp32"
        )


@requires_metal
def test_prepare_kernel_vc_is_passthrough():
    """``vc`` must be the same array object the caller passed in (zero
    extra work; matches ``_precompute_core``'s passthrough)."""
    data = _make_inputs(1, 1, seed=1)
    pre = metal_prepare_chunk(
        k=data["k"], q=data["q"], v=data["v"],
        g=data["g"], beta=data["beta"],
        scale_bf16_rt=data["scale_bf16_rt"],
    )
    # Same object identity OR exact value equality (MLX arrays are
    # immutable; identity is the strict check).
    assert pre["vc"] is data["v"], "vc passthrough must be the same array"


@requires_metal
def test_prepare_kernel_rejects_chunk_neq_16():
    """Phase 1 supports CHUNK=16 only — Neumann nilpotent closure baked
    in. Other CHUNK values must fail the assertion."""
    n_chunks, H, bad_chunk = 1, 1, 32
    k = mx.zeros((n_chunks, H, bad_chunk, D), dtype=mx.float32)
    q = mx.zeros((n_chunks, H, bad_chunk, D), dtype=mx.float32)
    v = mx.zeros((n_chunks, H, bad_chunk, D), dtype=mx.float32)
    g = mx.zeros((n_chunks, H, bad_chunk, D), dtype=mx.float32)
    beta = mx.zeros((n_chunks, H, bad_chunk), dtype=mx.float32)
    scale = mx.array([0.125], dtype=mx.float32)
    with pytest.raises(AssertionError):
        metal_prepare_chunk(
            k=k, q=q, v=v, g=g, beta=beta, scale_bf16_rt=scale,
        )


# ---------------------------------------------------------------------------
# Phase 2 will land: parity vs ``_precompute_core``. Marked xfail until
# the kernel produces real arithmetic.
# ---------------------------------------------------------------------------

PARITY_CASES = [
    (1, 1, 0),
    (4, 2, 1),
    (8, 4, 2),
]


# Per-output tolerances. Pointwise outputs (no reduction across elements)
# match bit-exact at the bf16-precision level because the kernel applies
# the SAME bf16 round-trip pattern as ``_precompute_core``. Outputs derived
# from a fp32 reduction (cumsum, matmul, Neumann inverse) are subject to
# a ~1-bf16-ULP shift due to Metal vs MLX reduction-order differences:
#
#   * cumsum: my Metal does sequential left-to-right per d-slot; MLX
#     likely uses a parallel scan with different rounding tail.
#   * matmul: simdgroup_matrix accumulates per-K-tile in fp32 fragments;
#     mx.matmul has its own dispatcher choice.
#   * Neumann: sequential scalar matmul vs MLX's batched matmul fp16-cast.
#
# At bf16 precision a 1-ULP shift can amplify through ex2 to ~2^-10 ≈
# 9.77e-4 at value ~0.15. Use 2e-3 / 5e-3 (abs / rel) for the affected
# outputs — loose enough to absorb 1 bf16 ULP, tight enough to detect
# real arithmetic bugs (which would mismatch at >> 1 ULP across many
# elements, not 1-2 elements at the boundary).
_BIT_EXACT_OUTPUTS = ("k_decayed", "q_decayed", "beta_bf16")
_BF16_ULP_OUTPUTS = ("k_restored", "g_total_exp", "Mqk", "INV_bf")


@requires_metal
@pytest.mark.parametrize("n_chunks, H, seed", PARITY_CASES)
def test_prepare_kernel_parity_with_precompute_core(n_chunks, H, seed):
    data = _make_inputs(n_chunks, H, seed)
    gc, qc, kc, vc, bc = data["_oracle_inputs"]
    ref = _precompute_core(
        gc, qc, kc, vc, bc,
        H=H, D=D, chunk=CHUNK,
        scale_bf16_rt=data["scale_bf16_rt"],
    )

    got = metal_prepare_chunk(
        k=data["k"], q=data["q"], v=data["v"],
        g=data["g"], beta=data["beta"],
        scale_bf16_rt=data["scale_bf16_rt"],
    )

    # Strict bit-exact gate: outputs that don't depend on the cumsum's
    # reduction order match Phase 3b's recurrence-kernel tolerance.
    for name in _BIT_EXACT_OUTPUTS:
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=1e-5, atol=1e-5,
            err_msg=f"{name}: n_chunks={n_chunks} H={H} seed={seed}",
        )

    # Loose gate: 1 bf16 ULP band for cumsum-sensitive outputs.
    for name in _BF16_ULP_OUTPUTS:
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=5e-3, atol=2e-3,
            err_msg=f"{name}: n_chunks={n_chunks} H={H} seed={seed}",
        )
