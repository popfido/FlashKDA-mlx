"""Parity tests for the flat-ragged token-major Metal prepare kernel
(PR M Option A, ``fused4``).

The v4 kernel is functionally and arithmetically identical to
``fused_v3`` (full-fusion: L2-norm + KDA gate activation + sections
(b)-(g) in one Metal dispatch). It differs only in how each output
chunk computes its source-token starting index:

* v3: ``tok_chunk_base = chunk_id * CHUNK`` — implicit, every chunk
  reads from a CHUNK-aligned offset in a per-seq padded buffer.
* v4: ``tok_chunk_base = chunk_token_start_in[chunk_id]`` — explicit
  via a tiny int32 metadata table, allowing chunks to point at
  arbitrary offsets in a SINGLE flat ``[T_total, H, D]`` buffer
  spanning all packed sequences. Avoids the per-seq pad-and-stack
  allocation that fused3 forces in the packed varlen path.

Output buffers are tile-major ``[n_total_chunks, H, CHUNK, ...]``
(same as v3), so the recurrence kernel is unaffected.

Oracles:
* For "v4 mimics v3" cases (chunk_token_start = chunk_id * CHUNK), we
  compare v4 against v3 byte-for-byte (atol=0). Same arithmetic, same
  reduction order, only addressing differs.
* For genuine flat-ragged cases (multiple seqs with different bos
  offsets), we compare v4 against per-seq fused_v3 calls byte-for-byte.

End-to-end parity vs. the per-seq optimized forward is also exercised.

Skipped on M1/M2.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flash_kda_mlx._metal_prepare import (  # noqa: E402
    HAS_METAL_KERNEL,
    metal_prepare_chunk_fused_v3,
    metal_prepare_chunk_fused_v4,
)
from flash_kda_mlx.optimized import (  # noqa: E402
    _valid_tokens_per_chunk_single,
)
from flash_kda_mlx.reference import (  # noqa: E402
    LOG2E,
    _ex2_ftz,
    _q_bf16,
)


D = 128
CHUNK = 16


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal fused v4 prepare kernel disabled (M3+ only)",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _make_inputs(T_total: int, H: int, seed: int) -> dict:
    mx.random.seed(seed)
    s = 0.1
    q_flat = mx.random.normal((T_total, H, D)) * s
    k_flat = mx.random.normal((T_total, H, D)) * s
    v_flat = _q_bf16(mx.random.normal((T_total, H, D)) * s)
    g_flat = -mx.abs(mx.random.normal((T_total, H, D))) * 0.1
    beta_flat = mx.random.normal((T_total, H)) * s

    A_log = mx.random.normal((H,)) * 0.1
    dt_bias = mx.random.normal((H, D)) * 0.01
    lower_bound = -5.0

    a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)
    lower_bound_log2e = mx.array([lower_bound * LOG2E], dtype=mx.float32)
    scale_bf16_rt = _q_bf16(mx.array([0.125], dtype=mx.float32))[0]

    return {
        "q": q_flat, "k": k_flat, "v": v_flat,
        "g": g_flat, "beta": beta_flat,
        "scale_bf16_rt": scale_bf16_rt,
        "a_log_exp": a_log_exp, "dt_bias": dt_bias,
        "lower_bound_log2e": lower_bound_log2e,
        "T_total": T_total, "H": H,
    }


def _run_v4(d: dict, valid: list[int], starts: list[int]) -> dict:
    return metal_prepare_chunk_fused_v4(
        k_raw=d["k"], q_raw=d["q"], v=d["v"], g_raw=d["g"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=mx.array(valid, dtype=mx.int32),
        chunk_token_start=mx.array(starts, dtype=mx.int32),
    )


def _run_v3_on_padded_seq(d: dict, bos: int, eos: int, valid: list[int]) -> dict:
    """Slice [bos:eos] from the flat buffers, pad to a CHUNK multiple,
    and call fused_v3. Returns the v3 outputs (which have an n_chunks
    based on the padded length).
    """
    seq_len = eos - bos
    n_chunks = (seq_len + CHUNK - 1) // CHUNK
    padded_T = n_chunks * CHUNK

    def _pad(x: mx.array) -> mx.array:
        cur = x.shape[0]
        if cur == padded_T:
            return x
        pad_shape = list(x.shape)
        pad_shape[0] = padded_T - cur
        pad = mx.zeros(tuple(pad_shape), dtype=x.dtype)
        return mx.concatenate([x, pad], axis=0)

    q = _pad(d["q"][bos:eos])
    k = _pad(d["k"][bos:eos])
    v = _pad(d["v"][bos:eos])
    g = _pad(d["g"][bos:eos])
    beta = _pad(d["beta"][bos:eos])

    return metal_prepare_chunk_fused_v3(
        k_raw=k, q_raw=q, v=v, g_raw=g, beta=beta,
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=mx.array(valid, dtype=mx.int32),
    )


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("n_chunks, H", [(1, 1), (4, 2), (8, 4)])
def test_fused_v4_shapes(n_chunks, H):
    T_total = n_chunks * CHUNK
    d = _make_inputs(T_total, H, seed=0)
    valid = [CHUNK] * n_chunks
    starts = [c * CHUNK for c in range(n_chunks)]
    pre = _run_v4(d, valid, starts)
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
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp"):
        assert pre[name].dtype == mx.float32
    # vc preserves the input v dtype.
    assert pre["vc"].dtype == d["v"].dtype


@requires_metal
def test_fused_v4_vc_dtype_bf16():
    """When v is supplied as bf16, vc output preserves bf16."""
    d = _make_inputs(2 * CHUNK, 2, seed=42)
    d["v"] = d["v"].astype(mx.bfloat16)
    starts = [0, CHUNK]
    valid = [CHUNK, CHUNK]
    pre = _run_v4(d, valid, starts)
    assert pre["vc"].dtype == mx.bfloat16


# ---------------------------------------------------------------------------
# Parity vs. fused_v3 when chunk_token_start mimics v3 addressing
# (chunk_token_start[c] = c * CHUNK). v4 and v3 should produce
# byte-identical outputs since only addressing differs.
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("n_chunks, H, seed", [
    (1, 1, 0),
    (4, 2, 1),
    (8, 4, 2),
    (16, 8, 3),
])
def test_fused_v4_matches_fused_v3_aligned_starts(n_chunks, H, seed):
    T_total = n_chunks * CHUNK
    d = _make_inputs(T_total, H, seed)
    valid = [CHUNK] * n_chunks
    starts = [c * CHUNK for c in range(n_chunks)]

    got = _run_v4(d, valid, starts)
    ref = metal_prepare_chunk_fused_v3(
        k_raw=d["k"], q_raw=d["q"], v=d["v"], g_raw=d["g"], beta=d["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=mx.array(valid, dtype=mx.int32),
    )
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
    (96, 4, 13),    # exact multiple of CHUNK
    (130, 4, 14),   # 9 chunks, last has 2 valid tokens
])
def test_fused_v4_matches_fused_v3_partial_last_chunk(seq_len, H, seed):
    """Single-seq parity: aligned starts with a partial last chunk.

    v4 has STRICTER input semantics than v3: v4 zeroes invalid rows
    (c >= valid_count) at every read site, because v4 inputs come from
    a flat unpadded buffer where invalid rows belong to the next
    sequence (or are simply unused). v3 by contrast assumes the caller
    has zero-padded invalid rows.

    To compare them on a partial-last-chunk single-seq case, we zero the
    invalid rows of v3's input first — mirroring what the fused3 caller
    does (``_pad_to_multiple`` followed by recurrence on padded buffers).
    """
    n_chunks = (seq_len + CHUNK - 1) // CHUNK
    T_total = n_chunks * CHUNK
    d = _make_inputs(T_total, H, seed)
    valid = _valid_tokens_per_chunk_single(seq_len, n_chunks, CHUNK)
    starts = [c * CHUNK for c in range(n_chunks)]

    got = _run_v4(d, valid, starts)

    # Build a v3-compatible input by zeroing rows >= seq_len (the trailing
    # part of the last partial chunk). For aligned starts the only invalid
    # rows are the [seq_len, T_total) tail.
    def _zero_invalid(x: mx.array) -> mx.array:
        if seq_len == T_total:
            return x
        head = x[:seq_len]
        zero_shape = list(x.shape)
        zero_shape[0] = T_total - seq_len
        return mx.concatenate(
            [head, mx.zeros(tuple(zero_shape), dtype=x.dtype)], axis=0,
        )

    ref = metal_prepare_chunk_fused_v3(
        k_raw=_zero_invalid(d["k"]),
        q_raw=_zero_invalid(d["q"]),
        v=_zero_invalid(d["v"]),
        g_raw=_zero_invalid(d["g"]),
        beta=_zero_invalid(d["beta"]),
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=mx.array(valid, dtype=mx.int32),
    )
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp", "vc"):
        np.testing.assert_array_equal(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            err_msg=f"{name}: seq_len={seq_len} H={H} seed={seed}",
        )


# ---------------------------------------------------------------------------
# Genuine flat-ragged: multiple sequences in one buffer, chunks point at
# per-seq starts. Compare v4 (one dispatch) against N fused_v3 calls
# (per-seq, with each seq padded to its own n_chunks*CHUNK).
# ---------------------------------------------------------------------------

@requires_metal
@pytest.mark.parametrize("seq_lens, H, seed", [
    ([16, 32], 2, 21),                 # 2 seqs, all chunks full, 3 total
    ([13, 27, 50], 4, 22),             # 3 seqs, last chunks partial, 8 total
    ([100, 50, 75, 30], 4, 23),        # 4 seqs, varied
    ([1, 2, 3], 2, 24),                # tiny seqs, 1 partial chunk each
    ([16, 16, 16, 16], 8, 25),         # uniform full chunks
    ([200, 150, 100], 8, 26),          # bench-ish
])
def test_fused_v4_flat_ragged_matches_per_seq_fused_v3(seq_lens, H, seed):
    T_total = sum(seq_lens)
    d = _make_inputs(T_total, H, seed)

    # Build flat-ragged metadata as fwd_optimized would.
    cu = [0]
    for sl in seq_lens:
        cu.append(cu[-1] + sl)
    starts: list[int] = []
    valid: list[int] = []
    seq_chunk_offsets = [0]
    for n, sl in enumerate(seq_lens):
        bos = cu[n]
        n_chunks_n = (sl + CHUNK - 1) // CHUNK
        for c in range(n_chunks_n):
            starts.append(bos + c * CHUNK)
            valid.append(min(CHUNK, sl - c * CHUNK))
        seq_chunk_offsets.append(seq_chunk_offsets[-1] + n_chunks_n)
    total_chunks = seq_chunk_offsets[-1]

    got_all = _run_v4(d, valid, starts)
    assert got_all["k_decayed"].shape[0] == total_chunks

    # Compare each sequence's chunk range to fused_v3 invoked on that seq alone.
    for n, sl in enumerate(seq_lens):
        bos, eos = cu[n], cu[n + 1]
        cs, ce = seq_chunk_offsets[n], seq_chunk_offsets[n + 1]
        n_chunks_n = ce - cs
        valid_n = _valid_tokens_per_chunk_single(sl, n_chunks_n, CHUNK)

        ref = _run_v3_on_padded_seq(d, bos, eos, valid_n)

        for name in ("k_decayed", "q_decayed", "k_restored",
                     "Mqk", "INV_bf", "beta_bf16", "g_total_exp", "vc"):
            got_slice = np.asarray(got_all[name][cs:ce])
            np.testing.assert_array_equal(
                got_slice,
                np.asarray(ref[name]),
                err_msg=f"{name}: seq={n} (bos={bos}, sl={sl}) "
                        f"seq_lens={seq_lens} seed={seed}",
            )


# ---------------------------------------------------------------------------
# Defensive: explicitly verify a chunk pointing at a NON-aligned token
# offset (mid-buffer) reads the right tokens. The flat-ragged test above
# already covers this transitively, but a single-seq mid-offset check
# pins down the addressing math directly.
# ---------------------------------------------------------------------------

@requires_metal
def test_fused_v4_arbitrary_offset_addressing():
    H = 2
    T_total = 64
    d = _make_inputs(T_total, H, seed=77)

    # One chunk reads tokens [3:19] (offset 3, length CHUNK=16).
    starts = [3]
    valid = [CHUNK]
    got = _run_v4(d, valid, starts)

    # Reference: shift the input buffers by 3 tokens then run v3 with
    # aligned starts.
    d_shift = dict(d)
    d_shift["q"]    = d["q"][3:3 + CHUNK]
    d_shift["k"]    = d["k"][3:3 + CHUNK]
    d_shift["v"]    = d["v"][3:3 + CHUNK]
    d_shift["g"]    = d["g"][3:3 + CHUNK]
    d_shift["beta"] = d["beta"][3:3 + CHUNK]
    ref = metal_prepare_chunk_fused_v3(
        k_raw=d_shift["k"], q_raw=d_shift["q"], v=d_shift["v"],
        g_raw=d_shift["g"], beta=d_shift["beta"],
        scale_bf16_rt=d["scale_bf16_rt"],
        a_log_exp=d["a_log_exp"], dt_bias=d["dt_bias"],
        lower_bound_log2e=d["lower_bound_log2e"],
        valid_tokens_per_chunk=mx.array(valid, dtype=mx.int32),
    )
    for name in ("k_decayed", "q_decayed", "k_restored",
                 "Mqk", "INV_bf", "beta_bf16", "g_total_exp", "vc"):
        np.testing.assert_array_equal(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            err_msg=f"{name} mismatch on arbitrary offset",
        )


# ---------------------------------------------------------------------------
# E2E parity: opt-in fused4 path must produce the same forward result
# as the default (fused3 / per-seq) path on bench-shape varlen inputs.
#
# Run in a subprocess so that env vars are picked up at import time.
# ---------------------------------------------------------------------------


_E2E_HARNESS = r"""
import json, os, sys
sys.path.insert(0, "{repo_root}")
import mlx.core as mx
import numpy as np
import flash_kda_mlx
from flash_kda_mlx.reference import _q_bf16

mx.random.seed(0)
H, D = {H}, 128

# Build varlen pack — mimics bench_varlen_mixed_H{H}.
seq_lens = {seq_lens}
T_total = sum(seq_lens)
cu = [0]
for sl in seq_lens:
    cu.append(cu[-1] + sl)
cu_arr = mx.array(cu, dtype=mx.int64)

q = (mx.random.normal((1, T_total, H, D)) * 0.1).astype(mx.bfloat16)
k = (mx.random.normal((1, T_total, H, D)) * 0.1).astype(mx.bfloat16)
v = (mx.random.normal((1, T_total, H, D)) * 0.1).astype(mx.bfloat16)
g = (-mx.abs(mx.random.normal((1, T_total, H, D))) * 0.1).astype(mx.bfloat16)
beta = (mx.random.normal((1, T_total, H)) * 0.1).astype(mx.bfloat16)
A_log = mx.random.normal((H,)).astype(mx.float32) * 0.1
dt_bias = mx.random.normal((H, D)).astype(mx.float32) * 0.01
out_buf = mx.zeros_like(q)

mx.eval(q, k, v, g, beta, A_log, dt_bias)

r = flash_kda_mlx.fwd(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=0.125, out=out_buf,
    A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    initial_state=None, final_state=None,
    cu_seqlens=cu_arr,
    backend="optimized",
)
out = np.asarray(r.out)
np.save("{out_path}", out)
"""


def _run_e2e_subprocess(env_overrides: dict, out_path: str, seq_lens, H: int) -> np.ndarray:
    repo_root = str(REPO_ROOT)
    code = _E2E_HARNESS.format(
        repo_root=repo_root, H=H, seq_lens=list(seq_lens), out_path=out_path,
    )
    env = os.environ.copy()
    # Strip any pre-existing prepare/recurrence env that might leak in.
    for k in ("MLX_KDA_ENABLE_METAL_PREPARE", "MLX_KDA_ENABLE_METAL_RECURRENCE",
              "MLX_KDA_DISABLE_PACKED"):
        env.pop(k, None)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"E2E subprocess failed (env={env_overrides}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return np.load(out_path)


@requires_metal
@pytest.mark.parametrize("seq_lens, H", [
    ([512, 256, 128, 64], 8),         # bench-ish mixed varlen
    ([300, 200, 150], 4),
    ([16, 32, 48, 64], 2),            # uniform-CHUNK boundary cases
    ([13, 27, 50], 4),                # partial last chunks
])
def test_fused_v4_e2e_parity_vs_per_seq(tmp_path, seq_lens, H):
    """Forward through ``flash_kda_mlx.fwd(backend='optimized')``: fused4 path
    must match the per-seq path (``MLX_KDA_DISABLE_PACKED=1`` disables
    the heuristic-packed path so we always use per-seq fused3).
    """
    out_ref_path = str(tmp_path / "ref.npy")
    out_v4_path = str(tmp_path / "v4.npy")

    out_ref = _run_e2e_subprocess(
        env_overrides={
            "MLX_KDA_ENABLE_METAL_PREPARE": "fused3",
            "MLX_KDA_ENABLE_METAL_RECURRENCE": "1",
            "MLX_KDA_DISABLE_PACKED": "1",  # force per-seq path as oracle
        },
        out_path=out_ref_path, seq_lens=seq_lens, H=H,
    )

    out_v4 = _run_e2e_subprocess(
        env_overrides={
            "MLX_KDA_ENABLE_METAL_PREPARE": "fused4",
            "MLX_KDA_ENABLE_METAL_RECURRENCE": "1",
        },
        out_path=out_v4_path, seq_lens=seq_lens, H=H,
    )

    # Same kernel arithmetic on both sides — only the prepare-dispatch
    # consolidation differs. Reduction order across chunks is preserved
    # because v4 chunks are addressed in the same per-seq order. Tight
    # tolerance: 1 bf16 ULP band as fused3.
    np.testing.assert_allclose(
        out_v4, out_ref, rtol=5e-3, atol=2e-3,
        err_msg=f"E2E mismatch: seq_lens={seq_lens} H={H}",
    )
