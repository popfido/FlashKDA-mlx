"""Public MLX operators for FlashKDA.

This module is the public entry point. It dispatches to either the pure-MLX
reference (``flash_kda_mlx.reference``) or the optimized path
(``flash_kda_mlx.optimized``) via the ``backend`` kwarg. Default is
``"optimized"`` — the reference remains reachable as the correctness oracle
per plan.md §Phase 8.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import mlx.core as mx

from . import optimized, reference
from ._validation import _validate


__all__ = ["fwd", "FwdResult", "Backend"]


Backend = Literal["reference", "optimized"]


class FwdResult(NamedTuple):
    """Typed return of :func:`fwd`.

    ``FwdResult`` is a :class:`NamedTuple`, so existing callers that use
    tuple destructuring (``out, final = fwd(...)``) and positional indexing
    (``result[0]``) keep working unchanged. New call sites should prefer
    attribute access (``result.out``, ``result.final_state``) for clarity.
    """

    out: mx.array
    final_state: Optional[mx.array]


def fwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float,
    out: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    lower_bound: float,
    initial_state: Optional[mx.array] = None,
    final_state: Optional[mx.array] = None,
    cu_seqlens: Optional[mx.array] = None,
    *,
    backend: Backend = "optimized",
) -> FwdResult:
    """FlashKDA forward — MLX implementation.

    Mirrors ``flash_kda.fwd``'s signature (plan.md §Design-decision §4). MLX
    arrays are immutable, so unlike the CUDA path this function *returns*
    the computed output and (optionally) final state rather than writing
    them in place. ``out`` and ``final_state`` are accepted for signature
    compatibility but only their **shape/dtype** are consulted.

    Args:
        q: Queries, ``[B, T, H, D]``, float dtype. ``D`` must be 128.
        k: Keys, ``[B, T, H, D]``, same shape/dtype as ``q``.
        v: Values, ``[B, T, H, D]``, same shape/dtype as ``q``.
        g: Gate pre-activations, ``[B, T, H, D]``, same shape/dtype as ``q``.
        beta: Per-token mixing coefficients (pre-sigmoid), ``[B, T, H]``,
            float dtype.
        scale: Query scaling factor applied after L2-normalising q/k. Must
            be a finite real number.
        out: Output-shape placeholder, ``[B, T, H, D]``. Only the shape
            is consulted; the returned ``out`` array is freshly allocated.
        A_log: Per-head decay log-rate, ``[H]``, float32.
        dt_bias: Per-head gate bias added to ``g`` before the sigmoid,
            ``[H, D]``, float32.
        lower_bound: Floor for the gate activation (typically ``-5.0``).
            Must be a finite real number.
        initial_state: Optional prior recurrent state.

            * Batched mode (``cu_seqlens`` is ``None``): ``[B, H, D, D]``.
            * Varlen mode: ``[N, H, D, D]`` where ``N = len(cu_seqlens) - 1``.

            Dtype selects the on-chip state policy (``bfloat16`` or
            ``float32``).
        final_state: Shape/dtype placeholder for the returned final state.
            Same shape rules as ``initial_state``. If ``None``, the returned
            ``final_state`` is ``None`` (the kernel skips writing it back).
        cu_seqlens: 1-D int32/int64 cumulative sequence offsets with
            ``cu_seqlens[0] == 0`` and ``cu_seqlens[-1] == B * T``. When
            provided, ``B`` must be 1.
        backend: ``"optimized"`` (default) dispatches to the vectorized
            path. ``"reference"`` uses the chunk-by-chunk oracle
            implementation. Most cases match to within ``rtol=atol=1e-5``;
            packed ``N > 1`` batched/varlen cases use
            ``rtol=1e-4, atol=5e-5`` to allow one-bf16-ULP
            dispatcher-order drift. See
            ``tests/test_optimized_parity.py``.

    Returns:
        :class:`FwdResult` — a named tuple with fields ``out`` (shape
        ``[B, T, H, D]``) and ``final_state`` (shape-matching
        ``initial_state`` / ``final_state`` placeholder, or ``None`` when
        no placeholder was supplied).

    Raises:
        TypeError: If any tensor argument is not an :class:`mlx.core.array`,
            if ``cu_seqlens`` is not int32/int64, or if a scalar argument is
            not a real number.
        ValueError: On any shape mismatch, invalid ``D``, non-finite scalar,
            malformed ``cu_seqlens``, or unknown ``backend`` value.

    Example:
        >>> import mlx.core as mx
        >>> import flash_kda_mlx
        >>> B, T, H, D = 1, 16, 4, 128
        >>> q = mx.random.normal((B, T, H, D))
        >>> k, v, g = q, q, q
        >>> beta = mx.zeros((B, T, H))
        >>> out_buf = mx.zeros((B, T, H, D))
        >>> A_log = mx.zeros((H,))
        >>> dt_bias = mx.zeros((H, D))
        >>> result = flash_kda_mlx.fwd(
        ...     q=q, k=k, v=v, g=g, beta=beta,
        ...     scale=1.0 / (D ** 0.5), out=out_buf,
        ...     A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        ... )
        >>> result.out.shape
        (1, 16, 4, 128)
        >>> result.final_state is None
        True
    """
    # Validate at the public boundary so both backends see pre-checked inputs.
    _validate(
        q, k, v, g, beta, out, A_log, dt_bias,
        initial_state, final_state, cu_seqlens,
        scale=scale, lower_bound=lower_bound,
    )

    if backend == "reference":
        impl = reference.fwd_reference
    elif backend == "optimized":
        impl = optimized.fwd_optimized
    else:
        raise ValueError(
            f"backend must be 'reference' or 'optimized', got {backend!r}"
        )

    out_array, final_state_array = impl(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        out_like=out,
        A_log=A_log, dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state_like=final_state,
        cu_seqlens=cu_seqlens,
    )
    return FwdResult(out=out_array, final_state=final_state_array)
