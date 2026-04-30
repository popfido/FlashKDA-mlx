"""Generate parity fixtures for the MLX rewrite track.

Runs the CPU torch oracle (``scripts/torch_ref_cpu.py``) over a frozen
matrix of Phase-0 cases from ``plan.md`` and writes the inputs + expected
outputs as ``.npz`` files under ``tests/fixtures/``.

Usage::

    uv run --no-config python scripts/generate_parity_fixtures.py
    uv run --no-config python scripts/generate_parity_fixtures.py --stress

Fixture naming: ``<mode>__T<...>__H<...>__<state>__seed<...>.npz``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from torch_ref_cpu import torch_ref_cpu  # noqa: E402

FIXTURES_DIR = _THIS.parent.parent / "tests" / "fixtures"

D = 128
LOWER_BOUND = -5.0


# ---------------------------------------------------------------------------
# Input construction — must stay identical to tests/_helpers.make_inputs
# so that MLX-side inputs for the same seed match the fixtures exactly.
# ---------------------------------------------------------------------------

def _make_numpy_inputs(T: int, H: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def randn(*shape):
        return np.asarray(rng.standard_normal(shape), dtype=np.float32)

    def uniform(*shape, lo=0.0, hi=1.0):
        return np.asarray(rng.uniform(lo, hi, size=shape), dtype=np.float32)

    q = randn(1, T, H, D)
    k = randn(1, T, H, D)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    v = randn(1, T, H, D)
    g = randn(1, T, H, D)
    beta = randn(1, T, H)
    A_log = uniform(H)
    dt_bias = uniform(H, D)

    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias)


def _to_torch_bf16(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(torch.bfloat16)


def _initial_state_from_seed(N: int, H: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    rng = np.random.default_rng(seed + 1_000_000)
    vals = rng.standard_normal((N, H, D, D)).astype(np.float32) * 0.1
    return torch.from_numpy(vals).to(dtype).contiguous()


# ---------------------------------------------------------------------------
# Oracle driver
# ---------------------------------------------------------------------------

def _run_oracle(
    inputs_np: dict[str, np.ndarray],
    *,
    cu_seqlens_np: np.ndarray | None = None,
    initial_state: torch.Tensor | None = None,
    want_final_state: bool = False,
    state_fp32: bool = False,
) -> dict[str, np.ndarray]:
    q = _to_torch_bf16(inputs_np["q"])
    k = _to_torch_bf16(inputs_np["k"])
    v = _to_torch_bf16(inputs_np["v"])
    g = _to_torch_bf16(inputs_np["g"])
    beta = _to_torch_bf16(inputs_np["beta"])
    A_log = torch.from_numpy(inputs_np["A_log"])
    dt_bias = torch.from_numpy(inputs_np["dt_bias"])

    B, T_seq = q.shape[0], q.shape[1]
    out = torch.zeros_like(q)

    cu_seqlens_t = torch.from_numpy(cu_seqlens_np).to(torch.long) if cu_seqlens_np is not None else None

    if cu_seqlens_t is not None:
        N = cu_seqlens_t.numel() - 1
    else:
        N = B

    state_dtype = torch.float32 if state_fp32 else torch.bfloat16
    final_state = (
        torch.zeros(N, A_log.shape[0], D, D, dtype=state_dtype)
        if want_final_state
        else None
    )

    scale = 1.0 / math.sqrt(D)

    torch_ref_cpu(
        q, k, v, g, beta, scale, out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_state, final_state=final_state,
        cu_seqlens=cu_seqlens_t,
    )

    payload: dict[str, np.ndarray] = {
        "q": inputs_np["q"],
        "k": inputs_np["k"],
        "v": inputs_np["v"],
        "g": inputs_np["g"],
        "beta": inputs_np["beta"],
        "A_log": inputs_np["A_log"],
        "dt_bias": inputs_np["dt_bias"],
        "scale": np.float32(scale),
        "lower_bound": np.float32(LOWER_BOUND),
        "out_expected": out.to(torch.float32).numpy(),
    }
    if cu_seqlens_np is not None:
        payload["cu_seqlens"] = cu_seqlens_np.astype(np.int64)
    if initial_state is not None:
        payload["initial_state"] = initial_state.to(torch.float32).numpy()
    if final_state is not None:
        payload["final_state_expected"] = final_state.to(torch.float32).numpy()
        payload["state_fp32"] = np.int32(1 if state_fp32 else 0)
    return payload


# ---------------------------------------------------------------------------
# Case matrix (frozen)
# ---------------------------------------------------------------------------

BASE_H = [1, 4]
BASE_T = [16, 17, 64, 97, 256]
BASE_SEEDS = [0]

VARLEN_SEQ_LENS = [37, 16, 97, 64]   # exact-tile + tails

STRESS_CASES = [
    dict(T=8192, H=96),
]


def _write(name: str, payload: dict[str, np.ndarray], overwrite: bool) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURES_DIR / f"{name}.npz"
    if path.exists() and not overwrite:
        return
    np.savez(path, **payload)
    print(f"  wrote {path.name} ({path.stat().st_size // 1024} KiB)")


def _case_name(mode: str, T: int, H: int, state: str, seed: int, extra: str = "") -> str:
    parts = [mode, f"T{T}", f"H{H}", state, f"seed{seed}"]
    if extra:
        parts.append(extra)
    return "__".join(parts)


def generate_basic(overwrite: bool) -> None:
    print("Basic fixtures (fixed-length):")
    for H in BASE_H:
        for T in BASE_T:
            for seed in BASE_SEEDS:
                inputs = _make_numpy_inputs(T, H, seed)

                # no-state
                _write(
                    _case_name("fixed", T, H, "no_state", seed),
                    _run_oracle(inputs, want_final_state=False),
                    overwrite,
                )

                # in+out bf16 state
                init_bf16 = _initial_state_from_seed(1, H, seed, torch.bfloat16)
                _write(
                    _case_name("fixed", T, H, "state_in_out_bf16", seed),
                    _run_oracle(
                        inputs,
                        initial_state=init_bf16.clone(),
                        want_final_state=True,
                        state_fp32=False,
                    ),
                    overwrite,
                )

                # in-only bf16 (no final_state returned)
                _write(
                    _case_name("fixed", T, H, "state_in_only_bf16", seed),
                    _run_oracle(inputs, initial_state=init_bf16.clone(), want_final_state=False),
                    overwrite,
                )

                # out-only bf16 (final_state but no initial_state)
                _write(
                    _case_name("fixed", T, H, "state_out_only_bf16", seed),
                    _run_oracle(inputs, want_final_state=True, state_fp32=False),
                    overwrite,
                )

                # in+out fp32 state
                init_fp32 = _initial_state_from_seed(1, H, seed, torch.float32)
                _write(
                    _case_name("fixed", T, H, "state_in_out_fp32", seed),
                    _run_oracle(
                        inputs,
                        initial_state=init_fp32.clone(),
                        want_final_state=True,
                        state_fp32=True,
                    ),
                    overwrite,
                )


def generate_varlen(overwrite: bool) -> None:
    print("Varlen fixtures:")
    seq_lens = VARLEN_SEQ_LENS
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu = np.zeros(N + 1, dtype=np.int64)
    cu[1:] = np.cumsum(seq_lens)
    for H in BASE_H:
        for seed in BASE_SEEDS:
            inputs = _make_numpy_inputs(T_total, H, seed)
            extra = "varlen_" + "_".join(str(s) for s in seq_lens)

            # no-state
            _write(
                _case_name("varlen", T_total, H, "no_state", seed, extra),
                _run_oracle(inputs, cu_seqlens_np=cu, want_final_state=False),
                overwrite,
            )
            # in+out bf16
            init_bf16 = _initial_state_from_seed(N, H, seed, torch.bfloat16)
            _write(
                _case_name("varlen", T_total, H, "state_in_out_bf16", seed, extra),
                _run_oracle(
                    inputs,
                    cu_seqlens_np=cu,
                    initial_state=init_bf16.clone(),
                    want_final_state=True,
                    state_fp32=False,
                ),
                overwrite,
            )
            # in+out fp32
            init_fp32 = _initial_state_from_seed(N, H, seed, torch.float32)
            _write(
                _case_name("varlen", T_total, H, "state_in_out_fp32", seed, extra),
                _run_oracle(
                    inputs,
                    cu_seqlens_np=cu,
                    initial_state=init_fp32.clone(),
                    want_final_state=True,
                    state_fp32=True,
                ),
                overwrite,
            )


def generate_stress(overwrite: bool) -> None:
    print("Stress fixtures:")
    for case in STRESS_CASES:
        T, H = case["T"], case["H"]
        inputs = _make_numpy_inputs(T, H, 0)
        _write(
            _case_name("fixed", T, H, "no_state", 0, "stress"),
            _run_oracle(inputs, want_final_state=False),
            overwrite,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", action="store_true", help="also generate stress cases (slow)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing fixtures")
    args = parser.parse_args()

    torch.manual_seed(0)

    generate_basic(args.overwrite)
    generate_varlen(args.overwrite)
    if args.stress:
        generate_stress(args.overwrite)

    print(f"\nDone. Fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
