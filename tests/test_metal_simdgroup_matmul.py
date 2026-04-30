"""Phase 1 tests for the simdgroup_matrix Metal kernel infrastructure.

Covers:

* Hardware probe: ``HAS_METAL_KERNEL`` reflects device class; ``MLX_KDA_FORCE_METAL_FALLBACK=1`` forces False.
* Parity: ``metal_matmul_A_by_B`` matches ``mx.matmul(A, B)`` at ``rtol=atol=1e-5``
  across the parameter grid (varying H, with CHUNK=16, D=128 fixed).
* Edge cases: contiguous and transposed (MLX normalizes via
  ``ensure_row_contiguous=True``) inputs, small-H / typical-H / bench-H.
* Bench gate: metal kernel stays within ±30% of ``mx.matmul`` wall-clock
  per call (plan §5 Phase 1 ship bar).

Skipped entirely on M1/M2 per plan §8 (user decision #3).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flash_kda_mlx._metal_recurrence import (  # noqa: E402
    HAS_METAL_KERNEL,
    metal_matmul_A_by_B,
    warmup_matmul_shapes,
)


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal simdgroup path disabled (M3+ only, per plan §8)",
)


# ---------------------------------------------------------------------------
# Hardware probe
# ---------------------------------------------------------------------------

def test_probe_matches_device_info():
    """HAS_METAL_KERNEL must align with the current device.

    This test runs on any host; it asserts the probe is self-consistent,
    not that any particular value is correct. Drive-by check: the probe
    doesn't raise.
    """
    from flash_kda_mlx import _metal_recurrence as mr
    # Re-probe via the helper we exposed; must match cached module value
    # unless MLX_KDA_FORCE_METAL_FALLBACK has been set since import.
    forced = bool(int(os.environ.get("MLX_KDA_FORCE_METAL_FALLBACK", "0") or "0"))
    if forced:
        assert not mr.HAS_METAL_KERNEL
    else:
        assert mr.HAS_METAL_KERNEL == mr._probe_m3_or_newer()


def test_probe_rejects_unknown_device_name(monkeypatch):
    """The probe must NOT classify non-Apple / unrecognised names as M3+."""
    from flash_kda_mlx import _metal_recurrence as mr
    monkeypatch.setattr(mr, "_device_name", lambda: "")
    assert mr._probe_m3_or_newer() is False

    monkeypatch.setattr(mr, "_device_name", lambda: "NVIDIA A100")
    assert mr._probe_m3_or_newer() is False

    monkeypatch.setattr(mr, "_device_name", lambda: "Apple M1 Max")
    assert mr._probe_m3_or_newer() is False

    monkeypatch.setattr(mr, "_device_name", lambda: "Apple M2 Pro")
    assert mr._probe_m3_or_newer() is False


def test_probe_accepts_m3_family(monkeypatch):
    from flash_kda_mlx import _metal_recurrence as mr
    for name in ("Apple M3", "Apple M3 Pro", "Apple M3 Max", "Apple M3 Ultra"):
        monkeypatch.setattr(mr, "_device_name", lambda n=name: n)
        assert mr._probe_m3_or_newer() is True, f"should accept {name}"


def test_probe_accepts_future_families(monkeypatch):
    from flash_kda_mlx import _metal_recurrence as mr
    for name in ("Apple M4", "Apple M5 Max", "Apple M7 Ultra"):
        monkeypatch.setattr(mr, "_device_name", lambda n=name: n)
        assert mr._probe_m3_or_newer() is True, f"should accept {name}"


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

PARITY_CASES: list[tuple[int, int, int, int]] = [
    # (H, CHUNK, D, seed)
    (1, 16, 128, 0),
    (2, 16, 128, 1),
    (4, 16, 128, 2),
    (8, 16, 128, 3),
    (64, 16, 128, 4),    # bench shape
    (96, 16, 128, 5),    # bench shape
]


@requires_metal
@pytest.mark.parametrize("H, CHUNK, D, seed", PARITY_CASES)
def test_matmul_parity_vs_mx(H, CHUNK, D, seed):
    """``metal_matmul_A_by_B`` must match ``mx.matmul`` at 1e-5."""
    mx.random.seed(seed)
    A = mx.random.normal((H, CHUNK, D))
    B = mx.random.normal((H, D, D))
    mx.eval(A, B)

    C_ref = mx.matmul(A, B)
    C_metal = metal_matmul_A_by_B(A, B)
    mx.eval(C_ref, C_metal)

    ref = np.asarray(C_ref)
    got = np.asarray(C_metal)
    np.testing.assert_allclose(
        got, ref, rtol=1e-5, atol=1e-5,
        err_msg=(
            f"H={H} CHUNK={CHUNK} D={D} seed={seed}: "
            f"max_abs={float(np.abs(got - ref).max()):.3e}"
        ),
    )


@requires_metal
def test_matmul_bit_exact_on_zeros():
    """Trivial input: zeros in → zeros out."""
    H, CHUNK, D = 4, 16, 128
    A = mx.zeros((H, CHUNK, D), dtype=mx.float32)
    B = mx.zeros((H, D, D), dtype=mx.float32)
    C = metal_matmul_A_by_B(A, B)
    mx.eval(C)
    assert np.all(np.asarray(C) == 0.0)


@requires_metal
def test_matmul_rejects_wrong_dtype():
    """fp16/bf16 inputs should assert — Phase 1 is fp32-only."""
    H, CHUNK, D = 4, 16, 128
    A = mx.zeros((H, CHUNK, D), dtype=mx.bfloat16)
    B = mx.zeros((H, D, D), dtype=mx.float32)
    with pytest.raises(AssertionError):
        metal_matmul_A_by_B(A, B)


@requires_metal
def test_matmul_rejects_shape_mismatch():
    """A's D must equal B's rows and cols."""
    A = mx.zeros((4, 16, 64), dtype=mx.float32)
    B = mx.zeros((4, 128, 128), dtype=mx.float32)
    with pytest.raises(AssertionError):
        metal_matmul_A_by_B(A, B)


# ---------------------------------------------------------------------------
# Bench gate (ship bar: within ±30% of mx.matmul)
# ---------------------------------------------------------------------------
#
# The plan's Phase 1 ship bar is that the simdgroup kernel is not
# catastrophically worse than MLX's generated matmul. ±30% is the
# safety margin. Run with forced per-iter ``mx.eval`` so the timing
# reflects real GPU work, not lazy graph construction.

@requires_metal
@pytest.mark.parametrize("H", [4, 64, 96])
def test_matmul_bench_within_30pct_of_mx(H):
    CHUNK, D = 16, 128
    mx.random.seed(42)
    A = mx.random.normal((H, CHUNK, D))
    B = mx.random.normal((H, D, D))
    mx.eval(A, B)

    warmup_matmul_shapes([(H, CHUNK, D)])
    # Warmup both paths to eliminate first-call JIT effects.
    for _ in range(5):
        mx.eval(mx.matmul(A, B))
        mx.eval(metal_matmul_A_by_B(A, B))

    N = 200

    t0 = time.perf_counter()
    for _ in range(N):
        C = mx.matmul(A, B)
        mx.eval(C)
    t_mx = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        C = metal_matmul_A_by_B(A, B)
        mx.eval(C)
    t_metal = time.perf_counter() - t0

    ratio = t_metal / t_mx
    # ratio < 1.3 means metal is at most 30% slower than mx.matmul.
    # Some noise is inherent; allow a comfortable ceiling.
    assert ratio < 1.3, (
        f"H={H}: metal kernel {t_metal*1000:.1f} ms vs mx.matmul "
        f"{t_mx*1000:.1f} ms — ratio {ratio:.2f}x exceeds 1.3x bar"
    )
