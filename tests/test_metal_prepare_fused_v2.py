"""Parity tests for the full-fusion Metal prepare kernel (PR H follow-on 2).

The "fused2" variant adds KDA gate activation to the partial-fusion
kernel (which only fuses Q/K L2-norm). A per-chunk valid-token count
masks padded varlen positions inside the kernel so the cumsum in
stage 2 does not propagate nonzero values from padded tokens.

Oracle: full Python pipeline — L2-norm + bf16 of q/k, gate activation
of g (``lower_bound * LOG2E * sigmoid(_ex2_ftz(A_log * LOG2E) *
(g + dt_bias))``), then ``_precompute_core``.

Tolerance: same loose 1-bf16-ULP band as the partial-fusion test
(``test_metal_prepare_fused.py``). The kernel reduction-order
differences (simd_sum L2-norm and per-element sigmoid) propagate
through every output.

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
    reason="Metal fused v2 prepare kernel disabled (M3+ only)",
)


# ---------------------------------------------------------------------------
# Inputs (raw — pre-section-(a)).
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

    a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)
    lower_bound_log2e = mx.array([lower_bound * LOG2E], dtype=mx.float32)

    def _to_chunks_hd(x: mx.array) -> mx.array:
        return x.reshape(n_chunks, CHUNK, H, D).transpose(0, 2, 1, 3)

    q_raw = _to_chunks_hd(q_raw_flat)
    k_raw = _to_chunks_hd(k_raw_flat)
    v = _to_chunks_hd(v_flat)
    g_raw = _to_chunks_hd(g_raw_flat)
    beta = beta_flat.reshape(n_chunks, CHUNK, H).transpose(0, 2, 1)

    scale_bf16_rt = _q_bf16(mx.array([0.125], dtype=mx.float32))[0]

    return {
        "q_raw": q_raw, "k_raw": k_raw, "v": v,
        "g_raw": g_raw, "beta": beta,
        "scale_bf16_rt": scale_bf16_rt,
        "a_log_exp": a_log_exp, "dt_bias": dt_bias,
        "lower_bound_log2e": lower_bound_log2e,
        "A_log": A_log, "lower_bound": lower_bound,
    }


def _full_pipeline_oracle(d: dict, valid_tokens_per_chunk: list[int]) -> dict[str, mx.array]:
    """MLX-side pipeline: L2-norm q/k + gate activation (with the same
    per-token mask the kernel applies internally), then _precompute_core."""
    q_n = _q_bf16(_l2_normalize(d["q_raw"]))
    k_n = _q_bf16(_l2_normalize(d["k_raw"]))

    # Apply gate activation on the [n_chunks, H, CHUNK, D] grid then
    # mask invalid token rows to zero (mirroring the kernel-side stage 0z).
    n_chunks, H, _, _ = d["g_raw"].shape
    a_log_exp_full = d["a_log_exp"][None, :, None, None]  # [1, H, 1, 1]
    dt_bias_full = d["dt_bias"][None, :, None, :]         # [1, H, 1, D]
    lb_log2e = d["lower_bound_log2e"][0]
    g_act = lb_log2e * mx.sigmoid(a_log_exp_full * (d["g_raw"] + dt_bias_full))

    # Build a [n_chunks, 1, CHUNK, 1] mask from the per-chunk valid count.
    mask_rows = mx.zeros((n_chunks, CHUNK), dtype=mx.float32)
    for c_idx, valid in enumerate(valid_tokens_per_chunk):
        if valid > 0:
            ones = mx.ones((valid,), dtype=mx.float32)
            zeros = mx.zeros((CHUNK - valid,), dtype=mx.float32)
            mask_rows[c_idx] = mx.concatenate([ones, zeros], axis=0)
    mask = mask_rows.reshape(n_chunks, 1, CHUNK, 1)
    g_act_masked = g_act * mask

    return _precompute_core(
        g_act_masked, q_n, k_n, d["v"], d["beta"],
        H=k_n.shape[1], D=D, chunk=CHUNK,
        scale_bf16_rt=d["scale_bf16_rt"],
    )


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("n_chunks, H", [(1, 1), (4, 2), (8, 4)])
def test_fused_v2_shapes(n_chunks, H):
    d = _make_inputs(n_chunks, H, seed=0)
    valid = [CHUNK] * n_chunks
    valid_arr = mx.array(valid, dtype=mx.int32)
    pre = metal_prepare_chunk_fused_v2(
        k_raw=d["k_raw"], q_raw=d["q_raw"], v=d["v"],
        g_raw=d["g_raw"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=valid_arr,
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
# Parity against full Python pipeline (all-valid case — single full chunks).
# ---------------------------------------------------------------------------

# Loose tolerance: kernel-side L2-norm reduction + per-element sigmoid
# both differ from MLX by ~1 bf16 ULP and that propagates through cumsum
# / matmul outputs.
_FUSED2_ATOL = 2e-3
_FUSED2_RTOL = 5e-3


@requires_metal
@pytest.mark.parametrize("n_chunks, H, seed", [(1, 1, 0), (4, 2, 1), (8, 4, 2)])
def test_fused_v2_parity_all_valid(n_chunks, H, seed):
    d = _make_inputs(n_chunks, H, seed)
    valid_list = [CHUNK] * n_chunks  # all chunks fully valid
    ref = _full_pipeline_oracle(d, valid_list)

    valid_arr = mx.array(valid_list, dtype=mx.int32)
    got = metal_prepare_chunk_fused_v2(
        k_raw=d["k_raw"], q_raw=d["q_raw"], v=d["v"],
        g_raw=d["g_raw"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=valid_arr,
    )

    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=_FUSED2_RTOL, atol=_FUSED2_ATOL,
            err_msg=f"{name}: n_chunks={n_chunks} H={H} seed={seed}",
        )


# ---------------------------------------------------------------------------
# Parity with partial-validity chunks (varlen-shaped). The last chunk has
# fewer valid tokens; both the kernel and the oracle apply the mask.
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("seq_len, H, seed", [
    (13, 2, 11),    # 1 partial chunk (n_chunks=1, last_valid=13)
    (37, 4, 12),    # 3 chunks, last has 5 valid tokens
    (96, 4, 13),    # exact multiple of CHUNK (no partial)
    (130, 4, 14),   # 9 chunks, last has 2 valid tokens
])
def test_fused_v2_parity_partial_last_chunk(seq_len, H, seed):
    n_chunks = (seq_len + CHUNK - 1) // CHUNK
    d = _make_inputs(n_chunks, H, seed)

    valid_list = _valid_tokens_per_chunk_single(seq_len, n_chunks, CHUNK)
    valid_arr = mx.array(valid_list, dtype=mx.int32)

    ref = _full_pipeline_oracle(d, valid_list)
    got = metal_prepare_chunk_fused_v2(
        k_raw=d["k_raw"], q_raw=d["q_raw"], v=d["v"],
        g_raw=d["g_raw"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=valid_arr,
    )

    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=_FUSED2_RTOL, atol=_FUSED2_ATOL,
            err_msg=f"{name}: seq_len={seq_len} H={H} seed={seed}",
        )


# ---------------------------------------------------------------------------
# Helpers — sanity checks on _valid_tokens_per_chunk_*
# ---------------------------------------------------------------------------

def test_valid_tokens_per_chunk_single():
    # exact multiple
    assert _valid_tokens_per_chunk_single(64, 4, 16) == [16, 16, 16, 16]
    # partial last
    assert _valid_tokens_per_chunk_single(50, 4, 16) == [16, 16, 16, 2]
    # single full chunk
    assert _valid_tokens_per_chunk_single(16, 1, 16) == [16]
    # single partial chunk
    assert _valid_tokens_per_chunk_single(7, 1, 16) == [7]
    # zero seq
    assert _valid_tokens_per_chunk_single(0, 0, 16) == []


def test_valid_tokens_per_chunk_packed():
    # two seqs, max_chunks=4: seq0 spans 4 chunks (last partial), seq1 spans 2 chunks.
    seq_lens = [50, 17]   # n_chunks_per_seq=[4, 2], partials=[2, 1]
    out = _valid_tokens_per_chunk_packed(seq_lens, n_chunks_max=4, chunk=16)
    expected = [16, 16, 16, 2,    # seq0 chunks
                16,  1,  0, 0]    # seq1 chunks (2 valid, 2 padded)
    assert out == expected
