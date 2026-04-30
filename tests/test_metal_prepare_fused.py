"""Parity tests for the fused Metal prepare kernel (PR H follow-on 1).

The "fused" variant is a partial fusion: L2-norm + bf16-round of q/k
runs inside the kernel (saving the MLX reduction dispatch), while the
KDA gate activation stays in MLX because the varlen packed path zero-
pads g after activation and emulating that inside the kernel would
need per-chunk valid-token masks.

Oracle: MLX-side L2-norm + bf16 of q/k (NOT applied — the caller
passes raw q/k) + pre-activated g + ``_precompute_core``. The kernel
does the L2-norm inside; the reference applies MLX's L2-norm then
calls ``_precompute_core`` identically.

Tolerance: all outputs get the same 1-bf16-ULP band as
``test_metal_prepare.py``'s reduction bucket — the L2-norm reduction
inside the kernel uses simd_sum whose rounding order differs from
MLX's ``mx.rsqrt(sum(x*x, axis=-1))`` by ~1 bf16 ULP, and that
propagates everywhere.

Skipped on M1/M2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flash_kda_mlx._metal_prepare import (  # noqa: E402
    HAS_METAL_KERNEL,
    metal_prepare_chunk_fused,
)
from flash_kda_mlx.optimized import _precompute_core  # noqa: E402
from flash_kda_mlx.reference import (  # noqa: E402
    LOG2E,
    _ex2_ftz,
    _l2_normalize,
    _q_bf16,
)


D = 128
CHUNK = 16


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal fused prepare kernel disabled (M3+ only)",
)


# ---------------------------------------------------------------------------
# Raw inputs (pre-section-(a)): q/k fp32 unnormalised, g fp32 raw, beta
# fp32 pre-sigmoid, A_log/dt_bias/lower_bound from the usual KDA ranges.
# ---------------------------------------------------------------------------

def _make_inputs(n_chunks: int, H: int, seed: int) -> dict[str, mx.array]:
    mx.random.seed(seed)
    s = 0.1
    q_raw_flat = mx.random.normal((n_chunks * CHUNK, H, D)) * s
    k_raw_flat = mx.random.normal((n_chunks * CHUNK, H, D)) * s
    v_flat = _q_bf16(mx.random.normal((n_chunks * CHUNK, H, D)) * s)
    g_raw_flat = -mx.abs(mx.random.normal((n_chunks * CHUNK, H, D))) * 0.1
    beta_flat = mx.random.normal((n_chunks * CHUNK, H)) * s

    A_log = mx.random.normal((H,)) * 0.1
    dt_bias = mx.random.normal((H, D)) * 0.01
    lower_bound = -5.0

    # Pre-activate g in MLX (matches fwd_optimized's fused-mode flow):
    # caller applies gate activation; kernel does only L2-norm.
    g_bias = g_raw_flat + dt_bias[None, :, :].astype(mx.float32)
    a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)
    g_activated_flat = (lower_bound * LOG2E) * mx.sigmoid(
        a_log_exp[None, :, None] * g_bias
    )

    def _to_chunks_hd(x: mx.array) -> mx.array:
        return x.reshape(n_chunks, CHUNK, H, D).transpose(0, 2, 1, 3)

    q_raw = _to_chunks_hd(q_raw_flat)
    k_raw = _to_chunks_hd(k_raw_flat)
    v = _to_chunks_hd(v_flat)
    g_activated = _to_chunks_hd(g_activated_flat)
    beta = beta_flat.reshape(n_chunks, CHUNK, H).transpose(0, 2, 1)

    scale_bf16_rt = _q_bf16(mx.array([0.125], dtype=mx.float32))[0]

    return {
        "q_raw": q_raw, "k_raw": k_raw, "v": v,
        "g_activated": g_activated, "beta": beta,
        "scale_bf16_rt": scale_bf16_rt,
    }


def _oracle_full_pipeline(d: dict) -> dict[str, mx.array]:
    """Python path: L2-norm + bf16 on q/k, pre-activated g from caller,
    then ``_precompute_core``. Gate activation is applied by the test
    helper before the kernel runs — the kernel's caller (production
    ``fwd_optimized``) does the same."""
    q_n = _q_bf16(_l2_normalize(d["q_raw"]))
    k_n = _q_bf16(_l2_normalize(d["k_raw"]))
    return _precompute_core(
        d["g_activated"], q_n, k_n, d["v"], d["beta"],
        H=k_n.shape[1], D=D, chunk=CHUNK,
        scale_bf16_rt=d["scale_bf16_rt"],
    )


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("n_chunks, H", [(1, 1), (4, 2), (8, 4)])
def test_fused_kernel_shapes(n_chunks, H):
    d = _make_inputs(n_chunks, H, seed=0)
    pre = metal_prepare_chunk_fused(
        k_raw=d["k_raw"], q_raw=d["q_raw"], v=d["v"],
        g_activated=d["g_activated"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
    )
    expected = {
        "k_decayed":   (n_chunks, H, CHUNK, D),
        "q_decayed":   (n_chunks, H, CHUNK, D),
        "k_restored":  (n_chunks, H, CHUNK, D),
        "Mqk":         (n_chunks, H, CHUNK, CHUNK),
        "INV_bf":      (n_chunks, H, CHUNK, CHUNK),
        "vc":          (n_chunks, H, CHUNK, D),
        "beta_bf16":   (n_chunks, H, CHUNK, 1),
        "g_total_exp": (n_chunks, H, D, 1),
    }
    for name, shape in expected.items():
        assert pre[name].shape == shape, f"{name}: {pre[name].shape} != {shape}"
        assert pre[name].dtype == mx.float32


# ---------------------------------------------------------------------------
# Parity against full Python pipeline.
# ---------------------------------------------------------------------------

# All fused outputs are affected by the L2-norm reduction and the
# activation sigmoid, both of which can differ by ~1 bf16 ULP from MLX.
# Use the loose bucket uniformly here.
_FUSED_ATOL = 2e-3
_FUSED_RTOL = 5e-3


PARITY_CASES = [
    (1, 1, 0),
    (4, 2, 1),
    (8, 4, 2),
]


@requires_metal
@pytest.mark.parametrize("n_chunks, H, seed", PARITY_CASES)
def test_fused_kernel_parity(n_chunks, H, seed):
    d = _make_inputs(n_chunks, H, seed)
    ref = _oracle_full_pipeline(d)

    got = metal_prepare_chunk_fused(
        k_raw=d["k_raw"], q_raw=d["q_raw"], v=d["v"],
        g_activated=d["g_activated"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
    )

    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=_FUSED_RTOL, atol=_FUSED_ATOL,
            err_msg=f"{name}: n_chunks={n_chunks} H={H} seed={seed}",
        )
