"""State input/output semantics tests (PR3 target).

Covers the four ``(has_in, has_out)`` combinations plus bf16/fp32 external
dtypes. The actual numerical check lives in ``test_parity_fixtures.py``;
here we validate the state-shape contract and that the returned ``final``
tensor is either ``None`` (when not requested) or a properly-shaped array.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import flash_kda_mlx
from _helpers import make_inputs, to_numpy


D = 128


def _initial_state(B, H, seed, dtype):
    rng = np.random.default_rng(seed + 1_000_000)
    arr = rng.standard_normal((B, H, D, D)).astype(np.float32) * 0.1
    return mx.array(arr, dtype=dtype)


@pytest.mark.parametrize("has_in,has_out", [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
], ids=["in+out", "in_only", "out_only", "no_state"])
@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_state_shape_contract(has_in, has_out, state_dtype):
    T, H = 64, 1
    inputs = make_inputs(T=T, H=H, D=D, seed=0)
    out = mx.zeros((1, T, H, D), dtype=mx.float32)

    initial = _initial_state(1, H, 0, state_dtype) if has_in else None
    final_placeholder = (
        mx.zeros((1, H, D, D), dtype=state_dtype) if has_out else None
    )

    result, final = flash_kda_mlx.fwd(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=inputs["scale"], out=out,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
        initial_state=initial,
        final_state=final_placeholder,
    )

    assert result.shape == (1, T, H, D)
    if has_out:
        assert final is not None
        assert final.shape == (1, H, D, D)
        assert final.dtype == state_dtype
        np_final = to_numpy(final)
        assert np.isfinite(np_final).all()
    else:
        assert final is None
