"""Parity tests for the token-major full-fusion Metal prepare kernel
(PR K-a, ``fused3``).

The v3 kernel is functionally equivalent to ``fused_v2`` (full-fusion:
L2-norm + KDA gate activation + sections (b)-(g) in one Metal dispatch).
It differs only in the input layout: q/k/v/g arrive as token-major
``[T_total, H, D]`` and beta as ``[T_total, H]``, eliminating the
``[n_chunks, CHUNK, H, D] -> [n_chunks, H, CHUNK, D]`` transpose +
``ensure_row_contiguous`` copy that fused2 forces on every forward.

Output buffers are tile-major ``[n_chunks, H, CHUNK, ...]`` (same as v2),
so the recurrence kernel is unaffected.

Oracle: ``metal_prepare_chunk_fused_v2`` itself, which already passes
parity vs. the full Python pipeline at ``rtol=5e-3 atol=2e-3``. We
compare v3 to v2 byte-for-byte (tighter atol/rtol where applicable)
since the only difference is input addressing — same arithmetic, same
reduction order.

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
    metal_prepare_chunk_fused_v2,
    metal_prepare_chunk_fused_v3,
)
from flash_kda_mlx.optimized import (  # noqa: E402
    _precompute_core,
    _valid_tokens_per_chunk_packed,
    _valid_tokens_per_chunk_single,
)
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
    reason="Metal fused v3 prepare kernel disabled (M3+ only)",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
#
# Shared raw inputs as token-major flat buffers. The v2 fixture below
# applies the standard tile-major reshape; v3 takes the flat buffers
# directly.

def _make_inputs(n_chunks: int, H: int, seed: int) -> dict:
    mx.random.seed(seed)
    s = 0.1
    T_total = n_chunks * CHUNK
    q_raw_flat = mx.random.normal((T_total, H, D)) * s
    k_raw_flat = mx.random.normal((T_total, H, D)) * s
    v_flat = _q_bf16(mx.random.normal((T_total, H, D)) * s)
    g_raw_flat = -mx.abs(mx.random.normal((T_total, H, D))) * 0.1
    beta_flat = mx.random.normal((T_total, H)) * s

    A_log = mx.random.normal((H,)) * 0.1
    dt_bias = mx.random.normal((H, D)) * 0.01
    lower_bound = -5.0

    a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)
    lower_bound_log2e = mx.array([lower_bound * LOG2E], dtype=mx.float32)

    scale_bf16_rt = _q_bf16(mx.array([0.125], dtype=mx.float32))[0]

    # Token-major (v3 inputs):
    #   q/k/v/g : [T_total, H, D]
    #   beta    : [T_total, H]
    return {
        "q_tm": q_raw_flat, "k_tm": k_raw_flat, "v_tm": v_flat,
        "g_tm": g_raw_flat, "beta_tm": beta_flat,
        "scale_bf16_rt": scale_bf16_rt,
        "a_log_exp": a_log_exp, "dt_bias": dt_bias,
        "lower_bound_log2e": lower_bound_log2e,
        "A_log": A_log, "lower_bound": lower_bound,
        "T_total": T_total, "H": H, "n_chunks": n_chunks,
    }


def _to_v2_inputs(d: dict) -> dict:
    """Apply the tile-major reshape that ``fused_v2`` requires."""
    n_chunks = d["n_chunks"]
    H = d["H"]

    def _to_chunks_hd(x: mx.array) -> mx.array:
        return x.reshape(n_chunks, CHUNK, H, D).transpose(0, 2, 1, 3)

    return {
        "q_raw": _to_chunks_hd(d["q_tm"]),
        "k_raw": _to_chunks_hd(d["k_tm"]),
        "v": _to_chunks_hd(d["v_tm"]),
        "g_raw": _to_chunks_hd(d["g_tm"]),
        "beta": d["beta_tm"].reshape(n_chunks, CHUNK, H).transpose(0, 2, 1),
    }


def _full_pipeline_oracle(d: dict, valid_tokens_per_chunk: list[int]) -> dict:
    """MLX-side full pipeline (matches the v2 oracle)."""
    v2 = _to_v2_inputs(d)
    q_n = _q_bf16(_l2_normalize(v2["q_raw"]))
    k_n = _q_bf16(_l2_normalize(v2["k_raw"]))

    n_chunks = d["n_chunks"]
    H = d["H"]
    a_log_exp_full = d["a_log_exp"][None, :, None, None]
    dt_bias_full = d["dt_bias"][None, :, None, :]
    lb_log2e = d["lower_bound_log2e"][0]
    g_act = lb_log2e * mx.sigmoid(a_log_exp_full * (v2["g_raw"] + dt_bias_full))

    mask_rows = mx.zeros((n_chunks, CHUNK), dtype=mx.float32)
    for c_idx, valid in enumerate(valid_tokens_per_chunk):
        if valid > 0:
            ones = mx.ones((valid,), dtype=mx.float32)
            zeros = mx.zeros((CHUNK - valid,), dtype=mx.float32)
            mask_rows[c_idx] = mx.concatenate([ones, zeros], axis=0)
    mask = mask_rows.reshape(n_chunks, 1, CHUNK, 1)
    g_act_masked = g_act * mask

    return _precompute_core(
        g_act_masked, q_n, k_n, v2["v"], v2["beta"],
        H=H, D=D, chunk=CHUNK,
        scale_bf16_rt=d["scale_bf16_rt"],
    )


def _run_v3(d: dict, valid_arr: mx.array) -> dict:
    return metal_prepare_chunk_fused_v3(
        k_raw=d["k_tm"], q_raw=d["q_tm"], v=d["v_tm"],
        g_raw=d["g_tm"], beta=d["beta_tm"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=valid_arr,
    )


def _run_v2(d: dict, valid_arr: mx.array) -> dict:
    v2 = _to_v2_inputs(d)
    return metal_prepare_chunk_fused_v2(
        k_raw=v2["k_raw"], q_raw=v2["q_raw"], v=v2["v"],
        g_raw=v2["g_raw"], beta=v2["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=valid_arr,
    )


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("n_chunks, H", [(1, 1), (4, 2), (8, 4)])
def test_fused_v3_shapes(n_chunks, H):
    d = _make_inputs(n_chunks, H, seed=0)
    valid_arr = mx.array([CHUNK] * n_chunks, dtype=mx.int32)
    pre = _run_v3(d, valid_arr)
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
    # Most outputs are fp32; vc inherits the input v dtype (bf16 here from
    # _q_bf16). Verify both groups separately.
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        assert pre[name].dtype == mx.float32, f"{name}: dtype={pre[name].dtype}"
    # ``v_tm`` was rounded via _q_bf16 which returns fp32; vc should inherit fp32.
    assert pre["vc"].dtype == d["v_tm"].dtype


@requires_metal
def test_fused_v3_vc_dtype_bf16():
    """Verify that when v is supplied as bf16, vc output preserves bf16."""
    d = _make_inputs(2, 2, seed=42)
    d["v_tm"] = d["v_tm"].astype(mx.bfloat16)
    valid_arr = mx.array([CHUNK, CHUNK], dtype=mx.int32)
    pre = _run_v3(d, valid_arr)
    assert pre["vc"].dtype == mx.bfloat16


# ---------------------------------------------------------------------------
# Parity vs. fused_v2 (the only thing that differs is input addressing).
# ---------------------------------------------------------------------------
#
# v3 reproduces v2 reductions byte-for-byte (same g_in / k_in / q_in
# values reach the same threadgroup elements via different addressing).
# We require atol=0 — anything else is a bug in the index math.

@requires_metal
@pytest.mark.parametrize("n_chunks, H, seed", [
    (1, 1, 0),
    (4, 2, 1),
    (8, 4, 2),
    (16, 8, 3),  # bench-shape-ish: T=256, H=8
])
def test_fused_v3_matches_fused_v2_all_valid(n_chunks, H, seed):
    d = _make_inputs(n_chunks, H, seed)
    valid_arr = mx.array([CHUNK] * n_chunks, dtype=mx.int32)
    got = _run_v3(d, valid_arr)
    ref = _run_v2(d, valid_arr)
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp", "vc"):
        np.testing.assert_array_equal(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            err_msg=f"{name}: n_chunks={n_chunks} H={H} seed={seed}",
        )


@requires_metal
@pytest.mark.parametrize("seq_len, H, seed", [
    (13, 2, 11),    # 1 partial chunk
    (37, 4, 12),    # 3 chunks, last has 5 valid tokens
    (96, 4, 13),    # exact multiple of CHUNK (no partial)
    (130, 4, 14),   # 9 chunks, last has 2 valid tokens
])
def test_fused_v3_matches_fused_v2_partial_last_chunk(seq_len, H, seed):
    n_chunks = (seq_len + CHUNK - 1) // CHUNK
    d = _make_inputs(n_chunks, H, seed)
    valid_list = _valid_tokens_per_chunk_single(seq_len, n_chunks, CHUNK)
    valid_arr = mx.array(valid_list, dtype=mx.int32)
    got = _run_v3(d, valid_arr)
    ref = _run_v2(d, valid_arr)
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp", "vc"):
        np.testing.assert_array_equal(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            err_msg=f"{name}: seq_len={seq_len} H={H} seed={seed}",
        )


# ---------------------------------------------------------------------------
# Parity vs. full Python pipeline. This is the ground-truth gate; tolerance
# matches the v2 band (kernel-side reductions differ from MLX by ~1 bf16 ULP).
# ---------------------------------------------------------------------------

_FUSED3_ATOL = 2e-3
_FUSED3_RTOL = 5e-3


@requires_metal
@pytest.mark.parametrize("n_chunks, H, seed", [(1, 1, 0), (4, 2, 1), (8, 4, 2)])
def test_fused_v3_parity_vs_pipeline_all_valid(n_chunks, H, seed):
    d = _make_inputs(n_chunks, H, seed)
    valid_list = [CHUNK] * n_chunks
    valid_arr = mx.array(valid_list, dtype=mx.int32)
    ref = _full_pipeline_oracle(d, valid_list)
    got = _run_v3(d, valid_arr)
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=_FUSED3_RTOL, atol=_FUSED3_ATOL,
            err_msg=f"{name}: n_chunks={n_chunks} H={H} seed={seed}",
        )


@requires_metal
@pytest.mark.parametrize("seq_len, H, seed", [
    (13, 2, 11),
    (37, 4, 12),
    (96, 4, 13),
    (130, 4, 14),
])
def test_fused_v3_parity_vs_pipeline_partial_last_chunk(seq_len, H, seed):
    n_chunks = (seq_len + CHUNK - 1) // CHUNK
    d = _make_inputs(n_chunks, H, seed)
    valid_list = _valid_tokens_per_chunk_single(seq_len, n_chunks, CHUNK)
    valid_arr = mx.array(valid_list, dtype=mx.int32)
    ref = _full_pipeline_oracle(d, valid_list)
    got = _run_v3(d, valid_arr)
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=_FUSED3_RTOL, atol=_FUSED3_ATOL,
            err_msg=f"{name}: seq_len={seq_len} H={H} seed={seed}",
        )


# ---------------------------------------------------------------------------
# Varlen padded-token regression: confirm fully-padded chunks (valid=0)
# write zero-derived outputs (g_total_exp = 1.0 since cumsum=0, exp(0)=1).
# This exercises the gate-zero-pad mask path that was the reason
# follow-on 1 kept activation in MLX.
# ---------------------------------------------------------------------------

@requires_metal
def test_fused_v3_padded_chunk_semantics():
    n_chunks = 4
    H = 2
    d = _make_inputs(n_chunks, H, seed=99)
    # First two chunks valid, last two fully padded.
    valid_list = [CHUNK, CHUNK, 0, 0]
    valid_arr = mx.array(valid_list, dtype=mx.int32)

    got = _run_v3(d, valid_arr)
    ref = _full_pipeline_oracle(d, valid_list)

    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=_FUSED3_RTOL, atol=_FUSED3_ATOL,
            err_msg=f"{name}",
        )

    # Spot-check: padded chunks have g_total = 0, so g_total_exp = exp2(0) = 1.0.
    g_total_exp = np.asarray(got["g_total_exp"])  # [n_chunks, H, D, 1]
    np.testing.assert_allclose(
        g_total_exp[2:], np.ones_like(g_total_exp[2:]),
        rtol=0, atol=0,
        err_msg="fully-padded chunks must produce g_total_exp == 1.0",
    )


# ---------------------------------------------------------------------------
# Cross-mode parity: the public env-gated path picks fused3 when the env
# var is set. Run a single end-to-end step via the optimized forward and
# compare against fused2 for the same inputs.
# ---------------------------------------------------------------------------
#
# (We skip this here — it duplicates test_optimized_parity coverage and the
# subprocess-isolation cost is high. The kernel-level parity above is
# sufficient to gate PR K-a.)
