"""Shared test helpers for the MLX rewrite track.

Principles enforced here:

* MLX is lazy — every public helper that returns an array calls ``mx.eval``
  on its result. This matches plan.md §"MLX-specific execution rule".
* Seeds are always explicit. Randomness lives only in the fixture generator
  and in tests that parametrize over a small, named set of seeds.
* Numerical comparisons go through ``assert_allclose_mlx`` so tolerance
  policy is centralized and easy to tune in one place during PR5.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Random input construction (MLX-native, deterministic)
# ---------------------------------------------------------------------------

def make_inputs(
    T: int,
    H: int,
    D: int = 128,
    seed: int = 0,
    *,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    """Build a deterministic set of MLX inputs matching FlashKDA's contract.

    For MLX v1 we operate in fp32 end-to-end (see STATUS.md §"Internal dtype").
    ``dtype`` is kept as a parameter for Phase-6 dtype expansion but should
    default to fp32.
    """
    rng = np.random.default_rng(seed)

    def randn(*shape):
        return np.asarray(rng.standard_normal(shape), dtype=np.float32)

    def uniform(*shape, lo=0.0, hi=1.0):
        return np.asarray(rng.uniform(lo, hi, size=shape), dtype=np.float32)

    q_np = randn(1, T, H, D)
    k_np = randn(1, T, H, D)
    q_np = q_np / np.linalg.norm(q_np, axis=-1, keepdims=True)
    k_np = k_np / np.linalg.norm(k_np, axis=-1, keepdims=True)
    v_np = randn(1, T, H, D)
    g_np = randn(1, T, H, D)
    beta_np = randn(1, T, H)
    A_log_np = uniform(H)
    dt_bias_np = uniform(H, D)

    inputs: dict[str, mx.array] = {
        "q": mx.array(q_np, dtype=dtype),
        "k": mx.array(k_np, dtype=dtype),
        "v": mx.array(v_np, dtype=dtype),
        "g": mx.array(g_np, dtype=dtype),
        "beta": mx.array(beta_np, dtype=dtype),
        "A_log": mx.array(A_log_np, dtype=mx.float32),
        "dt_bias": mx.array(dt_bias_np, dtype=mx.float32),
        "scale": 1.0 / math.sqrt(D),
        "lower_bound": -5.0,
    }
    mx.eval(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        inputs["A_log"], inputs["dt_bias"],
    )
    return inputs


def make_varlen_inputs(
    seq_lens: list[int],
    H: int,
    D: int = 128,
    seed: int = 0,
    *,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    T_total = sum(seq_lens)
    inputs = make_inputs(T_total, H, D, seed=seed, dtype=dtype)
    cu = np.zeros(len(seq_lens) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(seq_lens)
    inputs["cu_seqlens"] = mx.array(cu)
    mx.eval(inputs["cu_seqlens"])
    return inputs


# ---------------------------------------------------------------------------
# MLX evaluation discipline
# ---------------------------------------------------------------------------

def eval_all(*arrays: mx.array) -> None:
    """Force MLX evaluation for correctness/timing boundaries."""
    mx.eval(*arrays)


def to_numpy(a: mx.array) -> np.ndarray:
    """Materialize an MLX array as a numpy array (forces eval).

    bfloat16 is not representable in numpy, so cast through fp32 when needed.
    """
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    mx.eval(a)
    return np.asarray(a)


# ---------------------------------------------------------------------------
# Fixture I/O
# ---------------------------------------------------------------------------

def load_fixture(name: str) -> dict[str, Any]:
    """Load a parity fixture ``.npz`` and return a dict of numpy arrays + scalars.

    Fixtures carry:

    * inputs: ``q, k, v, g, beta, A_log, dt_bias`` (numpy arrays)
    * scalars: ``scale, lower_bound`` (Python floats)
    * optional: ``initial_state, final_state_expected, cu_seqlens``
    * always: ``out_expected``
    * metadata: ``meta`` (json-encoded string)
    """
    path = FIXTURES_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture {path} missing. Run `uv run --no-config python scripts/generate_parity_fixtures.py`."
        )
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    # Unwrap 0-d scalars that numpy stores as arrays.
    for key in ("scale", "lower_bound"):
        if key in data:
            data[key] = float(data[key])
    return data


def fixture_to_mlx_inputs(fx: dict[str, Any]) -> dict[str, Any]:
    """Convert a fixture dict into the keyword-argument set passed to flash_kda_mlx.fwd.

    Fixtures always store state tensors as fp32 numpy (numpy has no bf16).
    We reconstruct the intended state dtype from the ``state_fp32`` flag
    so the fwd contract (``initial_state.dtype == final_state.dtype``)
    holds in bf16 cases too.
    """
    kwargs: dict[str, Any] = {
        "q": mx.array(fx["q"]),
        "k": mx.array(fx["k"]),
        "v": mx.array(fx["v"]),
        "g": mx.array(fx["g"]),
        "beta": mx.array(fx["beta"]),
        "A_log": mx.array(fx["A_log"]),
        "dt_bias": mx.array(fx["dt_bias"]),
        "scale": float(fx["scale"]),
        "lower_bound": float(fx["lower_bound"]),
    }
    state_fp32 = "state_fp32" in fx and int(fx["state_fp32"]) == 1
    state_dtype = mx.float32 if state_fp32 else mx.bfloat16
    if "initial_state" in fx:
        kwargs["initial_state"] = mx.array(fx["initial_state"]).astype(state_dtype)
    if "cu_seqlens" in fx:
        kwargs["cu_seqlens"] = mx.array(fx["cu_seqlens"])
    mx.eval(*[v for v in kwargs.values() if isinstance(v, mx.array)])
    return kwargs


# ---------------------------------------------------------------------------
# Tolerance policy
# ---------------------------------------------------------------------------

# Default tolerances for MLX↔oracle parity. Tuned in PR5.
DEFAULT_RTOL = 5e-3
DEFAULT_ATOL = 5e-3


def assert_allclose_mlx(
    actual: mx.array | np.ndarray,
    expected: mx.array | np.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    err_msg: str = "",
) -> None:
    """Tolerance-based comparison that evaluates MLX arrays first."""
    if isinstance(actual, mx.array):
        actual = to_numpy(actual)
    if isinstance(expected, mx.array):
        expected = to_numpy(expected)
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
    )
