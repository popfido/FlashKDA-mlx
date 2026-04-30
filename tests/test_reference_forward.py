"""Small fixed-length, no-state forward tests.

These drive PR2. They are deliberately lightweight — a real reference
comparison against frozen oracle output belongs in
``test_parity_fixtures.py``. Here we only check:

* the call runs without error,
* the result has the expected shape and dtype,
* the result is finite.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import flash_kda_mlx
from _helpers import make_inputs, to_numpy


D = 128


@pytest.mark.parametrize("T", [16, 17, 64, 97, 256])
@pytest.mark.parametrize("H", [1, 4])
def test_fwd_runs_and_is_finite(T, H):
    inputs = make_inputs(T=T, H=H, D=D, seed=0)
    out = mx.zeros((1, T, H, D), dtype=mx.float32)
    result, final = flash_kda_mlx.fwd(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=inputs["scale"], out=out,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
    )
    assert result.shape == (1, T, H, D)
    assert result.dtype == mx.float32
    assert final is None

    np_out = to_numpy(result)
    assert np.isfinite(np_out).all(), f"non-finite values for T={T}, H={H}"


def test_fwd_deterministic_under_same_seed():
    inputs_a = make_inputs(T=64, H=1, D=D, seed=7)
    inputs_b = make_inputs(T=64, H=1, D=D, seed=7)
    out = mx.zeros((1, 64, 1, D), dtype=mx.float32)
    r_a, _ = flash_kda_mlx.fwd(
        q=inputs_a["q"], k=inputs_a["k"], v=inputs_a["v"],
        g=inputs_a["g"], beta=inputs_a["beta"],
        scale=inputs_a["scale"], out=out,
        A_log=inputs_a["A_log"], dt_bias=inputs_a["dt_bias"],
        lower_bound=inputs_a["lower_bound"],
    )
    r_b, _ = flash_kda_mlx.fwd(
        q=inputs_b["q"], k=inputs_b["k"], v=inputs_b["v"],
        g=inputs_b["g"], beta=inputs_b["beta"],
        scale=inputs_b["scale"], out=out,
        A_log=inputs_b["A_log"], dt_bias=inputs_b["dt_bias"],
        lower_bound=inputs_b["lower_bound"],
    )
    np.testing.assert_array_equal(to_numpy(r_a), to_numpy(r_b))
