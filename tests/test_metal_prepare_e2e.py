"""E2E parity test for the Metal prepare kernel (PR H Branch A Phase 3).

The Metal prepare kernel matches ``_precompute_core``'s arithmetic at
the bf16 level (see ``test_metal_prepare.py``), but it differs by
~1 bf16 ULP on cumsum-/matmul-derived outputs because Metal's sequential
cumsum and simdgroup_matrix accumulator differ from MLX's parallel scan
and ``mx.matmul``. Those 1-ULP shifts propagate through the per-chunk
recurrence and accumulate (random walk), so ``test_optimized_parity``'s
strict 1e-5 gate would break.

This file validates the loose-but-production-realistic gate:
``fwd_optimized`` with ``MLX_KDA_ENABLE_METAL_PREPARE=1`` must agree
with the off-path within the production torch-reference tolerance band
(atol=5e-3, rtol=2e-3 for ``out``; rtol=1e-2 for ``final_state`` on
benchmark-shape state magnitudes).

Subprocess isolation is required because the env var is read at module
import time and binds the prepare kernel function pointer.

Skipped on M1/M2 (no Metal prepare kernel available).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import gate uses the OFF-path's HAS_METAL_KERNEL probe so the test
# auto-skips on M1/M2 the same way the prepare-only tests do.
from flash_kda_mlx._metal_recurrence import HAS_METAL_KERNEL  # noqa: E402


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal prepare kernel disabled (M3+ only)",
)


# Tolerance band accepted for production optimized-path changes.
E2E_ATOL = 5e-3
E2E_RTOL = 2e-3


# ---------------------------------------------------------------------------
# Subprocess runner — env var must be set before import.
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = r"""
import json, sys
import mlx.core as mx
import numpy as np
sys.path.insert(0, {repo_root!r})
from tests._helpers import make_inputs, make_varlen_inputs
import flash_kda_mlx

case_kind = {case_kind!r}
H = {H}
seed = {seed}

if case_kind == "fixed":
    T = {T}
    inputs = make_inputs(T=T, H=H, D=128, seed=seed, dtype=mx.bfloat16)
    T_total, N = T, 1
    cu_seqlens = None
else:
    seq_lens = {seq_lens!r}
    inputs = make_varlen_inputs(list(seq_lens), H=H, D=128, seed=seed,
                                dtype=mx.bfloat16)
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = inputs["cu_seqlens"]

initial_state = mx.zeros((N, H, 128, 128), dtype=mx.bfloat16)
final_state = mx.zeros((N, H, 128, 128), dtype=mx.bfloat16)
out = mx.zeros((1, T_total, H, 128), dtype=mx.bfloat16)

result = flash_kda_mlx.fwd(
    backend="optimized",
    q=inputs["q"], k=inputs["k"], v=inputs["v"],
    g=inputs["g"], beta=inputs["beta"],
    scale=inputs["scale"], out=out,
    A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
    lower_bound=inputs["lower_bound"],
    initial_state=initial_state, final_state=final_state,
    cu_seqlens=cu_seqlens,
)
out_arr = result.out.astype(mx.float32)
fs_arr  = result.final_state.astype(mx.float32) if result.final_state is not None else None
mx.eval(out_arr, fs_arr)

np.savez({npz_path!r},
         out=np.asarray(out_arr),
         final_state=np.asarray(fs_arr) if fs_arr is not None else np.zeros(0))
print("OK")
"""


def _run_one(case_kind, *, H, seed, T=None, seq_lens=None, prepare_mode):
    """``prepare_mode`` ∈ {"0", "1", "fused"}."""
    npz_path = Path(
        f"/tmp/prepare_e2e_{case_kind}_H{H}_p{prepare_mode}.npz"
    )
    if npz_path.exists():
        npz_path.unlink()

    runner = _RUNNER_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        case_kind=case_kind,
        H=H, seed=seed,
        T=T or 0, seq_lens=tuple(seq_lens or ()),
        npz_path=str(npz_path),
    )
    env = dict(os.environ)
    env["MLX_KDA_ENABLE_METAL_PREPARE"] = prepare_mode
    # Keep the recurrence path consistent across both runs so the only
    # variable is the prepare kernel.
    env.setdefault("MLX_KDA_ENABLE_METAL_RECURRENCE", "1")

    result = subprocess.run(
        [sys.executable, "-c", runner],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        f"runner failed (prepare={enable_prepare}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    data = np.load(npz_path)
    return {"out": data["out"], "final_state": data["final_state"]}


# Cases: small fixed + small varlen. Full bench-scale comparison is the
# job of pr_g_profile.py / a future bench step, not this regression test.
# The N=4 / N=8 short varlen rows route through ``_run_packed`` (Phase 2b
# size-aware threshold) and therefore exercise the packed-path Metal
# prepare wire-up added in PR H Branch A Phase 4.
E2E_CASES = [
    ("fixed", {"H": 4, "T": 256, "seed": 0}),
    ("fixed", {"H": 8, "T": 1024, "seed": 1}),
    ("varlen", {"H": 4, "seq_lens": (37, 16, 97, 64), "seed": 2}),
    ("varlen", {"H": 4, "seq_lens": (64, 64, 64, 64, 64, 64, 64, 64), "seed": 3}),
    ("varlen", {"H": 4, "seq_lens": (37, 16, 97, 64, 128, 17, 256, 80), "seed": 4}),
]


@requires_metal
@pytest.mark.parametrize("kind, params", E2E_CASES)
def test_metal_prepare_e2e_matches_off_path(kind, params):
    """fwd_optimized with MLX_KDA_ENABLE_METAL_PREPARE=1 must agree with
    the off-path within the production tolerance band on representative
    short cases.
    """
    off = _run_one(kind, prepare_mode="0", **params)
    on  = _run_one(kind, prepare_mode="1", **params)

    np.testing.assert_allclose(
        on["out"], off["out"],
        rtol=E2E_RTOL, atol=E2E_ATOL,
        err_msg=f"out mismatch: kind={kind} {params}",
    )
    if off["final_state"].size > 0:
        np.testing.assert_allclose(
            on["final_state"], off["final_state"],
            rtol=E2E_RTOL, atol=E2E_ATOL,
            err_msg=f"final_state mismatch: kind={kind} {params}",
        )


@requires_metal
@pytest.mark.parametrize("kind, params", E2E_CASES)
def test_metal_prepare_fused_e2e_matches_off_path(kind, params):
    """fwd_optimized with MLX_KDA_ENABLE_METAL_PREPARE=fused must agree
    with the off-path within the production tolerance band. The fused
    kernel additionally runs section (a) L2-norm inline.
    """
    off = _run_one(kind, prepare_mode="0", **params)
    fused = _run_one(kind, prepare_mode="fused", **params)

    np.testing.assert_allclose(
        fused["out"], off["out"],
        rtol=E2E_RTOL, atol=E2E_ATOL,
        err_msg=f"fused out mismatch: kind={kind} {params}",
    )
    if off["final_state"].size > 0:
        np.testing.assert_allclose(
            fused["final_state"], off["final_state"],
            rtol=E2E_RTOL, atol=E2E_ATOL,
            err_msg=f"fused final_state mismatch: kind={kind} {params}",
        )


@requires_metal
@pytest.mark.parametrize("kind, params", E2E_CASES)
def test_metal_prepare_fused_v2_e2e_matches_off_path(kind, params):
    """fwd_optimized with MLX_KDA_ENABLE_METAL_PREPARE=fused2 must agree
    with the off-path within the production tolerance band. The full-
    fusion kernel runs both section (a) halves (L2-norm AND KDA gate
    activation) inline; padded varlen positions are masked via the
    per-chunk valid-token count so the activate-then-zero-pad MLX
    semantics are preserved.
    """
    off = _run_one(kind, prepare_mode="0", **params)
    fused2 = _run_one(kind, prepare_mode="fused2", **params)

    np.testing.assert_allclose(
        fused2["out"], off["out"],
        rtol=E2E_RTOL, atol=E2E_ATOL,
        err_msg=f"fused2 out mismatch: kind={kind} {params}",
    )
    if off["final_state"].size > 0:
        np.testing.assert_allclose(
            fused2["final_state"], off["final_state"],
            rtol=E2E_RTOL, atol=E2E_ATOL,
            err_msg=f"fused2 final_state mismatch: kind={kind} {params}",
        )
