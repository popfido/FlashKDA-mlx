"""Packed-sequence (cu_seqlens) semantics tests (PR4 target)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import flash_kda_mlx
from _helpers import make_varlen_inputs, to_numpy


D = 128


@pytest.mark.parametrize("seq_lens", [
    [16, 16],                  # exact tile boundaries
    [37, 16, 97, 64],          # mixed tails + exact
    [1, 15, 16, 17],           # tiny sequences, tail-heavy
])
@pytest.mark.parametrize("H", [1, 4])
def test_varlen_runs_and_is_finite(seq_lens, H):
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=0)
    T_total = sum(seq_lens)
    out = mx.zeros((1, T_total, H, D), dtype=mx.float32)

    result, final = flash_kda_mlx.fwd(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=inputs["scale"], out=out,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
        cu_seqlens=inputs["cu_seqlens"],
    )
    assert result.shape == (1, T_total, H, D)
    assert final is None
    assert np.isfinite(to_numpy(result)).all()


def test_varlen_state_shape_is_n_not_b():
    """In varlen mode, state shape is [N, H, D, D] not [B, H, D, D]."""
    seq_lens = [37, 16, 97, 64]
    N = len(seq_lens)
    H = 1
    inputs = make_varlen_inputs(seq_lens, H=H, D=D, seed=0)
    T_total = sum(seq_lens)

    initial = mx.zeros((N, H, D, D), dtype=mx.float32)
    final_placeholder = mx.zeros((N, H, D, D), dtype=mx.float32)
    out = mx.zeros((1, T_total, H, D), dtype=mx.float32)

    result, final = flash_kda_mlx.fwd(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=inputs["scale"], out=out,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
        initial_state=initial,
        final_state=final_placeholder,
        cu_seqlens=inputs["cu_seqlens"],
    )
    assert result.shape == (1, T_total, H, D)
    assert final is not None
    assert final.shape == (N, H, D, D)
