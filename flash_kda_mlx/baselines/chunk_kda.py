"""MLX-LM gated-delta adapter exposed via the FLA ``chunk_kda`` contract.

This adapter provides the ``mlx_chunk_kda`` column for the MLX benchmark
report by wrapping ``mlx_lm.models.gated_delta.{gated_delta_kernel,
gated_delta_ops, compute_g}`` with the kwarg surface that
``benchmarks/bench_fwd.py`` uses to call FLA ``chunk_kda``:

* ``scale`` — applied to ``q`` before the kernel (FLA convention).
* ``use_qk_l2norm_in_kernel`` — L2-normalize ``q``/``k`` along the last axis
  with ``eps=1e-6`` (matching ``flash_kda_mlx.reference._l2_normalize``).
* ``use_beta_sigmoid_in_kernel`` — apply ``mx.sigmoid(beta)`` inside the
  adapter when ``True``; otherwise the caller has already sigmoided.
* ``use_gate_in_kernel`` — when ``True``, the adapter computes per-dim
  multiplicative decay from ``(g, A_log, dt_bias, lower_bound)``. When
  ``False``, the caller passes pre-computed multiplicative ``g``.
* ``A_log`` / ``dt_bias`` / ``lower_bound`` — used only when
  ``use_gate_in_kernel=True``.
* ``transpose_state_layout`` — when ``True`` (FLA benchmark default), states
  are ``[N, H, D_k, D_v]``; mlx-lm's native layout is ``[B, H, D_v, D_k]``,
  so the adapter swaps the last two axes on input and output.
* ``cu_seqlens`` — varlen mode; the adapter slices per-sequence and
  reassembles outputs. ``B`` must be 1 in varlen mode.
* ``output_final_state=False`` returns ``(out, None)``.
* ``use_kernel`` — picks between ``gated_delta_kernel`` (Metal) and
  ``gated_delta_ops`` (pure MLX ops).

**Gate formula.** When ``use_gate_in_kernel=True`` the per-step decay is:

    g_decay = exp(-exp(A_log) * softplus(clamp(g + dt_bias, min=lower_bound)))

matching FLA's ``chunk_kda`` with ``use_gate_in_kernel=True`` and
``use_lower_bound=True``. FLA's chunked Triton path does chunk-local
cumsum / exp2 manipulations under the hood, so this per-step form may not
be bit-equivalent. The production correctness gate is **local
torch-reference parity** in
``tests/test_chunk_baseline_torch_reference.py``; FLA-CUDA fixture
parity is not pursued (no CUDA device on the MLX side).

**Per-dim gate support.** mlx-lm's ``gated_delta_kernel`` / ``gated_delta_ops``
both accept ``g`` of shape ``[B, T, H, D_k]`` via their vectorized-gate
path (see ``mlx_lm.models.gated_delta._make_gated_delta_kernel`` with
``vectorized=True`` and ``gated_delta_ops``'s ``g.ndim == 3`` branch in
``_gated_delta_step_ops``). This adapter always passes the vectorized form.

Varlen uses per-sequence unpacking, not a fused packed kernel. For the
PR C scaffold this is intentional; the benchmark generator (PR D) may
revisit for performance once parity is confirmed.

The structure closely mirrors the sibling GDN adapter in
``flash_kda_mlx/baselines/chunk_gdn.py``; keep them in sync when adjusting
shared semantics (L2-norm formulation, transpose policy, varlen loop,
Metal-required guard, dtype policy).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops


__all__ = ["chunk_kda_mlx"]


def _l2_normalize_last(x: mx.array, *, eps: float = 1e-6) -> mx.array:
    """L2-normalize ``x`` along the last axis, returning fp32.

    Mirrors ``flash_kda_mlx.reference._l2_normalize``: reduce in fp32 with
    ``sum(x**2, -1) + eps`` and ``rsqrt``. No round-trip back to the caller's
    dtype — a bf16 round-trip would silently weaken the norm.
    """
    xf = x.astype(mx.float32)
    sq = mx.sum(xf * xf, axis=-1, keepdims=True)
    return xf * mx.rsqrt(sq + eps)


def _kda_gate(
    g: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    lower_bound: float,
) -> mx.array:
    """Compute per-step multiplicative KDA decay.

    Formula: ``exp(-exp(A_log) * softplus(clamp(g + dt_bias, min=lower_bound)))``.

    This matches mlx-lm's ``compute_g(A_log, a, dt_bias)`` with the
    additional ``lower_bound`` clamp applied to ``(g + dt_bias)`` before the
    softplus — FLA's ``chunk_kda`` applies an equivalent clamp when
    ``use_lower_bound=True``. Computed inline (rather than calling
    ``compute_g`` on a pre-shifted tensor) because the clamp is the
    non-trivial piece and inlining keeps the data flow explicit.

    Args:
        g: ``[B, T, H, D]`` fp32.
        A_log: ``[H]`` — per-head log decay rate.
        dt_bias: ``[H, D]`` — per-head, per-dim bias.
        lower_bound: Lower clamp applied to ``g + dt_bias``.

    Returns:
        ``[B, T, H, D]`` fp32 multiplicative decay.
    """
    g_f = g.astype(mx.float32)
    # dt_bias: [H, D] -> [1, 1, H, D]
    pre = g_f + dt_bias.astype(mx.float32).reshape(1, 1, *dt_bias.shape)
    pre_clamped = mx.maximum(pre, lower_bound)
    # A_log: [H] -> [1, 1, H, 1]
    a_log_exp = mx.exp(A_log.astype(mx.float32)).reshape(1, 1, -1, 1)
    return mx.exp(-a_log_exp * nn.softplus(pre_clamped))


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
                "chunk_kda_mlx(use_kernel=True) requires Metal; "
                "pass use_kernel=False to force the ops path."
            )
        return gated_delta_kernel(q, k, v, g_mlx, beta, state_mlx, mask=None)
    return gated_delta_ops(q, k, v, g_mlx, beta, state_mlx, mask=None)


def chunk_kda_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    *,
    scale: float,
    initial_state: mx.array | None = None,
    output_final_state: bool = True,
    use_gate_in_kernel: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    A_log: mx.array | None = None,
    dt_bias: mx.array | None = None,
    lower_bound: float = -5.0,
    transpose_state_layout: bool = True,
    cu_seqlens: mx.array | None = None,
    use_kernel: bool = True,
) -> tuple[mx.array, mx.array | None]:
    """FLA-compatible adapter around mlx-lm's gated-delta primitives.

    Args:
        q, k, v: ``[B, T, H, D]`` float arrays. In FLA's KDA benchmark
            ``D = 128``.
        g: ``[B, T, H, D]`` — semantics depend on ``use_gate_in_kernel``:

            * ``True`` (default): pre-activation gate (log-scale) that is
              transformed to multiplicative decay inside the adapter via
              the KDA gate formula.
            * ``False``: already multiplicative per-step decay, forwarded
              directly to mlx-lm's vectorized gate path.

        beta: ``[B, T, H]``. When ``use_beta_sigmoid_in_kernel=True``,
            this is pre-sigmoid and the adapter applies ``mx.sigmoid``;
            otherwise the caller has already sigmoided.
        scale: Multiplicative scale applied to ``q`` before the kernel.
        initial_state: Prior recurrent state. Shape depends on
            ``transpose_state_layout``:

            * ``True`` (FLA default): ``[N, H, D_k, D_v]``.
            * ``False``: ``[N, H, D_v, D_k]`` (mlx-lm native).

            ``N = len(cu_seqlens) - 1`` in varlen mode, else ``N = B``.
            Dtype is preserved end-to-end.
        output_final_state: If ``False``, return ``(out, None)``. The
            adapter still allocates a state internally because mlx-lm's
            primitives require one.
        use_gate_in_kernel: When ``True`` (default / FLA benchmark),
            compute per-dim decay from ``(g, A_log, dt_bias, lower_bound)``.
            When ``False``, ``g`` is already multiplicative.
        use_qk_l2norm_in_kernel: L2-normalize ``q``/``k`` along the last
            axis (``eps=1e-6``) before the kernel.
        use_beta_sigmoid_in_kernel: Apply ``mx.sigmoid(beta)`` when ``True``.
        A_log: ``[H]`` fp32. Required when ``use_gate_in_kernel=True``.
        dt_bias: ``[H, D]`` fp32. Required when ``use_gate_in_kernel=True``.
        lower_bound: Lower clamp for ``g + dt_bias`` before softplus.
            Typically ``-5.0`` in the FLA benchmark. Unused when
            ``use_gate_in_kernel=False``.
        transpose_state_layout: Swap the last two axes of state on input
            and output. ``True`` matches FLA's benchmark call.
        cu_seqlens: ``[N+1]`` int32/int64 cumulative offsets. When set,
            ``B`` must be 1. Per-sequence unpacked execution.
        use_kernel: Pick ``gated_delta_kernel`` (Metal) when ``True``,
            ``gated_delta_ops`` when ``False``. The kernel path also
            requires ``mx.metal.is_available()``.

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
    if g.shape != (B, T, H, D):
        raise ValueError(
            f"g must be [B, T, H, D] per-dim gate; got {g.shape}"
        )
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must be [B, T, H]; got {beta.shape}")

    if use_gate_in_kernel:
        if A_log is None or dt_bias is None:
            raise ValueError(
                "use_gate_in_kernel=True requires A_log and dt_bias"
            )
        if A_log.shape != (H,):
            raise ValueError(f"A_log must be [H]; got {A_log.shape}")
        if dt_bias.shape != (H, D):
            raise ValueError(f"dt_bias must be [H, D]; got {dt_bias.shape}")

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("cu_seqlens requires B == 1")
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
    if transpose_state_layout:
        state_mlx_native = mx.transpose(state_fla, (0, 1, 3, 2))
    else:
        state_mlx_native = state_fla

    # --- q/k preprocessing ---
    if use_qk_l2norm_in_kernel:
        q_n = _l2_normalize_last(q)
        k_n = _l2_normalize_last(k)
    else:
        q_n = q.astype(mx.float32)
        k_n = k.astype(mx.float32)

    # Scale is applied to q in FLA (equivalent to scaling logits by `scale`).
    if scale != 1.0:
        q_n = q_n * scale

    # v promoted to fp32 to match the L2-normalized q/k path.
    v_f = v.astype(mx.float32)

    # --- gate computation ---
    if use_gate_in_kernel:
        g_mlx = _kda_gate(g, A_log, dt_bias, lower_bound)
    else:
        g_mlx = g.astype(mx.float32)

    # --- beta ---
    if use_beta_sigmoid_in_kernel:
        beta_mlx = mx.sigmoid(beta.astype(mx.float32))
    else:
        beta_mlx = beta.astype(mx.float32)

    # --- dispatch ---
    if cu_seqlens is None:
        out_arr, state_out_native = _run_single(
            q=q_n, k=k_n, v=v_f,
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
            v_s = v_f[:, bos:eos]
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

    # Preserve dtype end-to-end.
    if state_out_fla.dtype != state_dtype:
        state_out_fla = state_out_fla.astype(state_dtype)

    final = state_out_fla if output_final_state else None
    return out_arr, final
