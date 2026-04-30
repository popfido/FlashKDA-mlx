"""MLX-LM gated-delta adapter exposed via the FLA ``chunk_gated_delta_rule`` contract.

This adapter provides the ``mlx_chunk_gdn`` column for the MLX benchmark
report by wrapping ``mlx_lm.models.gated_delta.{gated_delta_kernel, gated_delta_ops}``
with the kwarg surface that ``benchmarks/bench_fwd.py`` uses to call FLA
``chunk_gated_delta_rule``:

* ``scale`` — applied to ``q`` before the kernel (FLA convention).
* ``use_qk_l2norm_in_kernel`` — L2-normalize ``q``/``k`` along the last axis
  with the same ``eps=1e-6`` formulation used by ``flash_kda_mlx.reference._l2_normalize``.
* ``transpose_state_layout`` — when ``True`` (FLA benchmark default), states
  are ``[N, H, D_k, D_v]``; mlx-lm's native layout is ``[B, H, D_v, D_k]``,
  so the adapter swaps the last two axes on input and output.
* ``cu_seqlens`` — varlen mode; the adapter slices per-sequence and
  reassembles outputs. ``B`` must be 1 in varlen mode.
* ``output_final_state=False`` returns ``(out, None)``.
* ``use_kernel`` — picks between ``gated_delta_kernel`` (Metal) and
  ``gated_delta_ops`` (pure MLX ops).

**Log-decay → multiplicative decay.** FLA ``chunk_gated_delta_rule``
accepts ``g`` in log-decay space while mlx-lm's ``gated_delta_kernel`` /
``gated_delta_ops`` consume ``g`` as multiplicative per-step decay
(typically in ``(0, 1]``). This adapter applies ``g_mlx = mx.exp(g_fla)``
by default; the choice is validated by the local torch-reference
gated-delta parity in
``tests/test_chunk_baseline_torch_reference.py``. Alternative
transforms (``exp2(g · RCP_LN2)``, for example) are not validated and
should only be revisited if local parity ever breaks.

Varlen uses per-sequence unpacking, not a fused packed kernel —
intentional for the baseline; the benchmark generator may revisit
this for performance.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops


__all__ = ["chunk_gdn_mlx"]


_GTransform = Literal["exp", "identity"]


def _l2_normalize_last(x: mx.array, *, eps: float = 1e-6) -> mx.array:
    """L2-normalize ``x`` along the last axis, returning fp32.

    Mirrors ``flash_kda_mlx.reference._l2_normalize`` exactly: reduce in fp32 with
    ``sum(x**2, -1) + eps`` and ``rsqrt``, and return the fp32 result. No
    round-trip back to the caller's dtype — a bf16 round-trip would silently
    weaken the norm.
    """
    xf = x.astype(mx.float32)
    sq = mx.sum(xf * xf, axis=-1, keepdims=True)
    return xf * mx.rsqrt(sq + eps)


def _apply_g_transform(g: mx.array, transform: _GTransform) -> mx.array:
    if transform == "exp":
        return mx.exp(g)
    if transform == "identity":
        return g
    raise ValueError(f"Unsupported _g_transform={transform!r}")


def _run_single(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g_mlx: mx.array,
    beta: mx.array,
    state_mlx: mx.array,
    use_kernel: bool,
) -> tuple[mx.array, mx.array]:
    """Call mlx-lm's kernel or ops on a single fixed-length sequence.

    ``state_mlx`` is in mlx-lm's native ``[B, H, D_v, D_k]`` layout.

    When ``use_kernel=True`` but Metal is unavailable, raises
    ``RuntimeError`` rather than silently falling back to the ops path — a
    benchmarking adapter that silently downgrades is a measurement hazard.
    """
    if use_kernel:
        if not mx.metal.is_available():
            raise RuntimeError(
                "chunk_gdn_mlx(use_kernel=True) requires Metal; "
                "pass use_kernel=False to force the ops path."
            )
        return gated_delta_kernel(q, k, v, g_mlx, beta, state_mlx, mask=None)
    return gated_delta_ops(q, k, v, g_mlx, beta, state_mlx, mask=None)


def chunk_gdn_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    *,
    scale: float,
    initial_state: mx.array | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    transpose_state_layout: bool = True,
    cu_seqlens: mx.array | None = None,
    use_kernel: bool = True,
    _g_transform: _GTransform = "exp",
) -> tuple[mx.array, mx.array | None]:
    """FLA-compatible adapter around mlx-lm's gated-delta primitives.

    Args:
        q, k, v: ``[B, T, H, D]`` float arrays. In FLA's GDN benchmark
            ``D = 128``.
        g: ``[B, T, H]`` log-decay (FLA convention). Transformed to
            multiplicative decay via ``mx.exp`` before being passed to mlx-lm.
        beta: ``[B, T, H]`` pre-sigmoid. Sigmoid is applied inside the
            adapter to match FLA's ``use_beta_sigmoid_in_kernel=True``
            convention (FLA's ``chunk_gated_delta_rule`` applies the
            sigmoid internally via its own default).
        scale: Multiplicative scale applied to ``q`` before the kernel.
        initial_state: Prior recurrent state. Shape depends on
            ``transpose_state_layout``:

            * ``True`` (FLA default): ``[N, H, D_k, D_v]``.
            * ``False``: ``[B, H, D_v, D_k]`` (mlx-lm native).

            ``N = len(cu_seqlens) - 1`` in varlen mode, else ``N = B``.
            Dtype is preserved end-to-end.
        output_final_state: If ``False``, return ``(out, None)``. The
            adapter still allocates a state internally because mlx-lm's
            kernel path requires it.
        use_qk_l2norm_in_kernel: L2-normalize ``q``/``k`` along the last
            axis (``eps=1e-6``) before the kernel.
        transpose_state_layout: Swap the last two axes of state on input
            and output. True matches FLA's benchmark call.
        cu_seqlens: ``[N+1]`` int32/int64 cumulative offsets. When set,
            ``B`` must be 1. Per-sequence unpacked execution.
        use_kernel: Pick ``gated_delta_kernel`` (Metal) when ``True``,
            ``gated_delta_ops`` when ``False``. The kernel path also
            requires ``mx.metal.is_available()``.
        _g_transform: Log→multiplicative transform. ``"exp"`` (default) or
            ``"identity"``. Non-default transforms are for future FLA
            parity tuning and are not validated.

    Returns:
        ``(out, final_state)``. ``out`` shape is ``[B, T, H, D]``.
        ``final_state`` is ``None`` when ``output_final_state=False``;
        otherwise its shape matches ``initial_state``'s layout policy.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [B, T, H, D]; got {q.shape}")
    B, T, H, D = q.shape
    if k.shape != (B, T, H, D) or v.shape != (B, T, H, D):
        raise ValueError(
            f"k/v shape must match q; got k={k.shape}, v={v.shape}"
        )
    if g.shape != (B, T, H):
        raise ValueError(
            f"g must be [B, T, H] scalar per-head log-decay; got {g.shape}"
        )
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must be [B, T, H]; got {beta.shape}")

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("cu_seqlens requires B == 1")
        # Number of sequences.
        N = int(cu_seqlens.shape[0]) - 1
    else:
        N = B

    # --- dtype policy ---
    # Preserve state dtype end-to-end; allocate fp32 zeros when unset.
    if initial_state is None:
        state_dtype = mx.float32
        state_fla_shape = (N, H, D, D)  # layout is irrelevant for zeros
        state_fla = mx.zeros(state_fla_shape, dtype=state_dtype)
    else:
        state_dtype = initial_state.dtype
        state_fla = initial_state

    # Move state to mlx-lm's native [B/N, H, D_v, D_k] layout.
    # When transpose_state_layout=True the caller passed [N, H, D_k, D_v]
    # and we swap the last two axes. When False, caller already passes the
    # native mlx-lm layout.
    if transpose_state_layout:
        state_mlx_native = mx.transpose(state_fla, (0, 1, 3, 2))
    else:
        state_mlx_native = state_fla

    # --- q/k preprocessing ---
    if use_qk_l2norm_in_kernel:
        q_n = _l2_normalize_last(q)
        k_n = _l2_normalize_last(k)
    else:
        q_n = q
        k_n = k

    # Scale is applied to q in FLA (equivalent to scaling logits by `scale`).
    if scale != 1.0:
        q_n = q_n * scale

    # Log-decay -> multiplicative decay.
    g_mlx = _apply_g_transform(g, _g_transform)

    # Beta is pre-sigmoid on the FLA surface; mlx-lm's primitives expect
    # beta already in (0, 1)-ish range. Apply sigmoid to match FLA's
    # use_beta_sigmoid_in_kernel=True default.
    beta_mlx = mx.sigmoid(beta)

    # --- dispatch ---
    if cu_seqlens is None:
        out_arr, state_out_native = _run_single(
            q=q_n, k=k_n, v=v,
            g_mlx=g_mlx, beta=beta_mlx,
            state_mlx=state_mlx_native,
            use_kernel=use_kernel,
        )
    else:
        cu_list = [int(cu_seqlens[i].item()) for i in range(N + 1)]
        per_seq_outs: list[mx.array] = []
        per_seq_finals: list[mx.array] = []
        for n in range(N):
            bos, eos = cu_list[n], cu_list[n + 1]
            q_n_s = q_n[:, bos:eos]
            k_n_s = k_n[:, bos:eos]
            v_s = v[:, bos:eos]
            g_s = g_mlx[:, bos:eos]
            beta_s = beta_mlx[:, bos:eos]
            state_s = state_mlx_native[n:n + 1]  # [1, H, D_v, D_k]
            out_s, fin_s = _run_single(
                q=q_n_s, k=k_n_s, v=v_s,
                g_mlx=g_s, beta=beta_s,
                state_mlx=state_s,
                use_kernel=use_kernel,
            )
            per_seq_outs.append(out_s)
            per_seq_finals.append(fin_s)
        out_arr = mx.concatenate(per_seq_outs, axis=1)  # along T
        state_out_native = mx.concatenate(per_seq_finals, axis=0)  # along N

    # Restore caller's state layout.
    if transpose_state_layout:
        state_out_fla = mx.transpose(state_out_native, (0, 1, 3, 2))
    else:
        state_out_fla = state_out_native

    # Preserve dtype: mlx-lm returns state in input state dtype already, but
    # a defensive cast protects against future drift.
    if state_out_fla.dtype != state_dtype:
        state_out_fla = state_out_fla.astype(state_dtype)

    final = state_out_fla if output_final_state else None
    return out_arr, final
