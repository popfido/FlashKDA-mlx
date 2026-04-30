"""Input validation for :func:`flash_kda_mlx.fwd`.

The rules here encode the public contract frozen by
``tests/test_api_contract.py`` and plan.md §"First supported subset" /
§"Cases to freeze first". Error messages are designed to help users
diagnose the offending argument fast: every message names the argument,
shows expected vs. observed, and includes a hint where one common mistake
applies.

Dtype policy is intentionally narrow (plan.md §Phase 6 is deferred): we
accept floating dtypes for q/k/v/g/beta and cast internally — we do not
reject fp16 or bf16 inputs today.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


D_FIXED = 128

_FLOAT_DTYPES = (mx.float16, mx.float32, mx.bfloat16)
_INT_DTYPES = (mx.int32, mx.int64)


def _is_array(x, name: str) -> None:
    if not isinstance(x, mx.array):
        raise TypeError(
            f"{name} must be an mlx.core.array, got {type(x).__name__}"
        )


def _check_float(x: mx.array, name: str) -> None:
    if x.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"{name} must have a float dtype (float16/float32/bfloat16), "
            f"got {x.dtype}"
        )


def _check_finite_scalar(value, name: str) -> None:
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be a real number, got {value!r}") from e
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {f}")


def _validate(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    out_like: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    initial_state: Optional[mx.array],
    final_state_like: Optional[mx.array],
    cu_seqlens: Optional[mx.array],
    *,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = None,
) -> None:
    """Validate inputs for :func:`flash_kda_mlx.fwd`.

    Keyword-only ``scale`` and ``lower_bound`` are optional so callers that
    already pre-validated them (legacy path) do not break. The public
    ``fwd`` always forwards them, which covers the user-facing entry point.
    """
    # ----- type checks --------------------------------------------------
    for name, tensor in (
        ("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
        ("out", out_like), ("A_log", A_log), ("dt_bias", dt_bias),
    ):
        _is_array(tensor, name)
    for name, tensor in (
        ("initial_state", initial_state),
        ("final_state", final_state_like),
        ("cu_seqlens", cu_seqlens),
    ):
        if tensor is not None:
            _is_array(tensor, name)

    # ----- q/k/v/g shape + dtype ---------------------------------------
    if q.ndim != 4:
        raise ValueError(
            f"q must be 4D [B, T, H, D], got {q.ndim}D with shape {tuple(q.shape)}"
        )
    for name, tensor in (("k", k), ("v", v), ("g", g)):
        if tensor.shape != q.shape:
            raise ValueError(
                f"{name} must match q shape [B, T, H, D]={tuple(q.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
        _check_float(tensor, name)

    B, T_seq, H, D = (int(x) for x in q.shape)

    if T_seq < 1:
        raise ValueError(
            f"T (sequence length) must be >= 1, got T={T_seq}. "
            "Empty sequences are not supported."
        )
    if H < 1:
        raise ValueError(f"H (num heads) must be >= 1, got H={H}")

    if D != D_FIXED:
        raise ValueError(
            f"D must be {D_FIXED} (hardcoded head dim), got D={D}. "
            "FlashKDA v1 only supports D=128; pad or project your tensors."
        )

    # ----- beta ---------------------------------------------------------
    expected_beta = (B, T_seq, H)
    if beta.ndim != 3 or tuple(beta.shape) != expected_beta:
        raise ValueError(
            f"beta must be [B, T, H]={expected_beta}, got {tuple(beta.shape)}"
        )
    _check_float(beta, "beta")

    # ----- out placeholder ---------------------------------------------
    if tuple(out_like.shape) != tuple(q.shape):
        raise ValueError(
            f"out must match q shape [B, T, H, D]={tuple(q.shape)}, "
            f"got {tuple(out_like.shape)}. "
            "Note: MLX is immutable — out is a shape/dtype placeholder only; "
            "fwd returns the computed output."
        )

    # ----- A_log / dt_bias ---------------------------------------------
    if tuple(A_log.shape) != (H,):
        raise ValueError(
            f"A_log must be [H={H}], got {tuple(A_log.shape)}"
        )
    if tuple(dt_bias.shape) != (H, D):
        raise ValueError(
            f"dt_bias must be [H={H}, D={D}], got {tuple(dt_bias.shape)}"
        )

    # ----- scalars ------------------------------------------------------
    if scale is not None:
        _check_finite_scalar(scale, "scale")
    if lower_bound is not None:
        _check_finite_scalar(lower_bound, "lower_bound")

    # ----- cu_seqlens ---------------------------------------------------
    if cu_seqlens is not None:
        if cu_seqlens.dtype not in _INT_DTYPES:
            raise TypeError(
                f"cu_seqlens must have int32 or int64 dtype, got {cu_seqlens.dtype}"
            )
        if cu_seqlens.ndim != 1:
            raise ValueError(
                f"cu_seqlens must be 1D, got {cu_seqlens.ndim}D with shape "
                f"{tuple(cu_seqlens.shape)}"
            )
        if cu_seqlens.shape[0] < 2:
            raise ValueError(
                f"cu_seqlens must have at least 2 entries (N+1), "
                f"got length {cu_seqlens.shape[0]}"
            )
        if B != 1:
            raise ValueError(
                f"varlen mode requires B=1; got B={B}. "
                "If you meant batched mode, drop cu_seqlens."
            )
        # Materialise to check values. cu_seqlens is small (N+1), so this is cheap.
        mx.eval(cu_seqlens)
        cu_list = [int(cu_seqlens[i].item()) for i in range(cu_seqlens.shape[0])]
        if cu_list[0] != 0:
            raise ValueError(
                f"cu_seqlens[0] must be 0, got {cu_list[0]}"
            )
        if cu_list[-1] != B * T_seq:
            raise ValueError(
                f"cu_seqlens[-1] must equal B*T={B * T_seq}, got {cu_list[-1]}"
            )
        for i in range(1, len(cu_list)):
            if cu_list[i] < cu_list[i - 1]:
                raise ValueError(
                    f"cu_seqlens must be monotonically non-decreasing, "
                    f"got cu_seqlens[{i - 1}]={cu_list[i - 1]} > "
                    f"cu_seqlens[{i}]={cu_list[i]}"
                )

    # ----- state shapes -------------------------------------------------
    if cu_seqlens is not None:
        N = int(cu_seqlens.shape[0]) - 1
        state_context = "varlen mode"
    else:
        N = B
        state_context = "batched mode"
    expected_state = (N, H, D, D)
    for name, tensor in (
        ("initial_state", initial_state),
        ("final_state", final_state_like),
    ):
        if tensor is None:
            continue
        if tuple(tensor.shape) != expected_state:
            raise ValueError(
                f"{name} must be [N, H, D, D]={expected_state} in {state_context}, "
                f"got {tuple(tensor.shape)}"
            )
    if initial_state is not None and final_state_like is not None:
        if initial_state.dtype != final_state_like.dtype:
            raise ValueError(
                "initial_state and final_state must share dtype, got "
                f"initial_state.dtype={initial_state.dtype} vs "
                f"final_state.dtype={final_state_like.dtype}"
            )
