"""Pure-MLX reference implementation of FlashKDA forward.

Mirrors the algorithmic spec in ``scripts/torch_ref_cpu.py`` — chunk-by-chunk,
with bf16 quantization at the same cast boundaries the oracle uses — so
parity fixtures compare cleanly under the tolerances in
``tests/test_parity_fixtures.py``.

Readability over speed. Optimization is plan.md §Phase 8.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from ._validation import D_FIXED, _validate  # re-exported for callers


LOG2E = 1.4426950408889634
CHUNK = 16


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _as_int(x) -> int:
    if isinstance(x, mx.array):
        return int(x.item())
    return int(x)


def _q_bf16(x: mx.array) -> mx.array:
    """Round-trip through bf16, returning an fp32 array.

    The oracle quantizes to bf16 at many intermediate points; the MLX path
    preserves the same *quantization pattern* so parity tolerances stay tight,
    while keeping arithmetic in fp32 for MLX-friendliness.
    """
    return x.astype(mx.bfloat16).astype(mx.float32)


def _ex2_ftz(x: mx.array) -> mx.array:
    """Base-2 exponent flushed to zero for subnormals (matches ex2.approx.ftz.f32).

    MLX has no ``mx.exp2``; use ``2**x`` / ``mx.power(2, x)`` in fp32.
    """
    x_f = x.astype(mx.float32)
    y = mx.power(mx.array(2.0, dtype=mx.float32), x_f)
    tiny = 1.1754943508222875e-38  # float32 tiny
    return mx.where(mx.abs(y) < tiny, mx.zeros_like(y), y)


def _l2_normalize(x: mx.array) -> mx.array:
    """L2 normalize along the last axis, matching the oracle's ``+ 1e-6`` eps.

    Uses a straightforward reduction rather than the CUDA kernel's warp-shuffle
    tree — acceptable for PR2 since our parity target is the CPU oracle, which
    itself does not use the approximate PTX reduction on CPU.
    """
    x_f = x.astype(mx.float32)
    sq = mx.sum(x_f * x_f, axis=-1, keepdims=True)
    return x_f * mx.rsqrt(sq + 1e-6)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def fwd_reference(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float,
    out_like: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    lower_bound: float,
    initial_state: Optional[mx.array],
    final_state_like: Optional[mx.array],
    cu_seqlens: Optional[mx.array],
) -> tuple[mx.array, Optional[mx.array]]:
    _validate(q, k, v, g, beta, out_like, A_log, dt_bias,
              initial_state, final_state_like, cu_seqlens)

    B, T_seq, H, D = q.shape
    T_total = B * T_seq

    q = q.reshape(T_total, H, D).astype(mx.float32)
    k = k.reshape(T_total, H, D).astype(mx.float32)
    v = v.reshape(T_total, H, D).astype(mx.float32)
    g = g.reshape(T_total, H, D).astype(mx.float32)
    beta = beta.reshape(T_total, H).astype(mx.float32)

    # Sequence bookkeeping
    if cu_seqlens is not None:
        cu = cu_seqlens
    else:
        if B > 1:
            cu = mx.arange(0, B * T_seq + 1, T_seq, dtype=mx.int64)
        else:
            cu = mx.array([0, T_total], dtype=mx.int64)
    mx.eval(cu)
    cu_list = [int(cu[i].item()) for i in range(cu.shape[0])]
    N = len(cu_list) - 1

    # Determine external state dtype (bf16 by default, fp32 if caller asked)
    want_final = final_state_like is not None
    state_fp32 = False
    if initial_state is not None and initial_state.dtype == mx.float32:
        state_fp32 = True
    if final_state_like is not None and final_state_like.dtype == mx.float32:
        state_fp32 = True

    # --- L2 normalize q, k ---
    q = _q_bf16(_l2_normalize(q))
    k = _q_bf16(_l2_normalize(k))

    # --- Gate activation ---
    # g = g + dt_bias  (broadcast over T)
    g = g + dt_bias[None, :, :].astype(mx.float32)
    a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)  # [H]
    a_log_exp = a_log_exp[None, :, None]                    # [1, H, 1]
    g = (lower_bound * LOG2E) * mx.sigmoid(a_log_exp * g)   # [T_total, H, D] fp32

    # --- Working recurrent state, always carried as bf16 on-chip semantics ---
    if initial_state is not None:
        work_state = initial_state.astype(mx.bfloat16).astype(mx.float32)
    else:
        work_state = mx.zeros((N, H, D, D), dtype=mx.float32)
    mx.eval(work_state)

    # --- Precompute per-token constants ---
    scale_bf16_rt = _q_bf16(mx.array([scale], dtype=mx.float32))[0]  # scalar fp32 bf16-rounded

    # --- Per-sequence / per-chunk / per-head loop ---
    # We build the output as a list of [actual_len, H, D] tiles per sequence,
    # then concatenate to [T_total, H, D]. MLX arrays are immutable so this
    # is cleaner than writing into a pre-allocated buffer.
    all_out_chunks: list[mx.array] = []

    eye_chunk = mx.eye(CHUNK, dtype=mx.float32)

    for seq_idx in range(N):
        bos = cu_list[seq_idx]
        eos = cu_list[seq_idx + 1]
        seq_len = eos - bos
        n_chunks = (seq_len + CHUNK - 1) // CHUNK

        seq_out_chunks: list[mx.array] = []

        for chunk_idx in range(n_chunks):
            t0 = bos + chunk_idx * CHUNK
            actual_len = min(CHUNK, eos - t0)

            # Pad chunks to CHUNK-size along axis 0
            if actual_len == CHUNK:
                g_chunk = g[t0:t0 + CHUNK]          # [CHUNK, H, D]
                q_chunk = q[t0:t0 + CHUNK]
                k_chunk = k[t0:t0 + CHUNK]
                v_chunk = v[t0:t0 + CHUNK]
                beta_chunk = beta[t0:t0 + CHUNK]    # [CHUNK, H]
            else:
                pad_n = CHUNK - actual_len
                g_chunk = mx.concatenate(
                    [g[t0:t0 + actual_len], mx.zeros((pad_n, H, D), dtype=g.dtype)], axis=0,
                )
                q_chunk = mx.concatenate(
                    [q[t0:t0 + actual_len], mx.zeros((pad_n, H, D), dtype=q.dtype)], axis=0,
                )
                k_chunk = mx.concatenate(
                    [k[t0:t0 + actual_len], mx.zeros((pad_n, H, D), dtype=k.dtype)], axis=0,
                )
                v_chunk = mx.concatenate(
                    [v[t0:t0 + actual_len], mx.zeros((pad_n, H, D), dtype=v.dtype)], axis=0,
                )
                beta_chunk = mx.concatenate(
                    [beta[t0:t0 + actual_len], mx.zeros((pad_n, H), dtype=beta.dtype)], axis=0,
                )

            # Transpose to put H first: shape [H, CHUNK, D] / [H, CHUNK]
            g_chunk = g_chunk.transpose(1, 0, 2)
            q_chunk = q_chunk.transpose(1, 0, 2)
            k_chunk = k_chunk.transpose(1, 0, 2)
            v_chunk = v_chunk.transpose(1, 0, 2)
            beta_chunk = beta_chunk.transpose(1, 0)  # [H, CHUNK]

            # Per-head output for this chunk: list of [CHUNK, D]
            per_head_outs: list[mx.array] = []

            for h in range(H):
                gc = g_chunk[h]            # [CHUNK, D] fp32
                qc = q_chunk[h]
                kc = k_chunk[h]
                vc = v_chunk[h]
                bc = beta_chunk[h]         # [CHUNK]

                g_cumsum = mx.cumsum(gc, axis=0)                       # [CHUNK, D] fp32
                g_total = g_cumsum[-1:, :]                             # [1, D] fp32

                ex_pos = _q_bf16(_ex2_ftz(g_cumsum))                   # bf16-quantised
                ex_neg = _q_bf16(_ex2_ftz(-g_cumsum))                  # bf16-quantised
                ex_gtot = _q_bf16(_ex2_ftz(g_total))                   # bf16-quantised

                k_decayed = _q_bf16(kc * ex_pos)                       # bf16
                q_decayed = _q_bf16(_q_bf16(qc * ex_pos) * scale_bf16_rt)
                k_inv = _q_bf16(kc * ex_neg)
                k_restored = _q_bf16(k_inv * ex_gtot)

                # L = k_decayed @ k_inv.T  (fp32 accumulate → fp16 cast)
                L = mx.matmul(k_decayed, k_inv.transpose(1, 0))        # [CHUNK, CHUNK] fp32
                L = L.astype(mx.float16).astype(mx.float32)            # round through fp16

                # Mqk = q_decayed @ k_inv.T  (bf16 semantics)
                Mqk = _q_bf16(mx.matmul(q_decayed, k_inv.transpose(1, 0)))

                beta_act = mx.sigmoid(bc)                              # fp32 [CHUNK]
                beta_bf16 = _q_bf16(beta_act)[:, None]                 # [CHUNK, 1]
                beta_fp16 = (beta_act.astype(mx.float16).astype(mx.float32))[:, None]

                # Mask before fp16 cast to avoid overflow-becomes-NaN via mask*inf.
                # The oracle does cast→tril which zeros the upper tri literally; we
                # match that by using mx.tril (assignment-style) instead of
                # multiplying by a {0,1} mask, which would propagate inf to NaN.
                L = mx.tril(L, k=-1) * beta_fp16
                L = L.astype(mx.float16).astype(mx.float32)            # keep fp16-scale arithmetic
                Mqk = mx.tril(Mqk)
                Mqk = _q_bf16(Mqk)

                # Neumann series inverse in fp16 domain
                INV = (eye_chunk - L).astype(mx.float16).astype(mx.float32)

                def _fp16_mm(a, b):
                    return mx.matmul(
                        a.astype(mx.float16), b.astype(mx.float16)
                    ).astype(mx.float32)

                L2 = _fp16_mm(L, L)
                INV = INV + _fp16_mm(INV, L2)
                L4 = _fp16_mm(L2, L2)
                INV = INV + _fp16_mm(INV, L4)
                L8 = _fp16_mm(L4, L4)
                INV = INV + _fp16_mm(INV, L8)

                INV_bf = _q_bf16(INV)                                   # cast to bf16

                state_slice = work_state[seq_idx, h]                    # [D, D] fp32 (bf16-valued)
                state_bf = _q_bf16(state_slice)

                # v' = v - k_decayed @ state.T ; then v' *= beta
                vdiff = vc - _q_bf16(mx.matmul(k_decayed, state_bf.transpose(1, 0)))
                vdiff = _q_bf16(vdiff)
                vdiff = _q_bf16(vdiff * beta_bf16)

                U = _q_bf16(mx.matmul(INV_bf, vdiff))
                out_h = _q_bf16(mx.matmul(q_decayed, state_bf.transpose(1, 0)))
                out_h = _q_bf16(out_h + _q_bf16(mx.matmul(Mqk, U)))

                # delta_s = k_restored.T @ U  (fp32 accumulate)
                delta_s = mx.matmul(k_restored.transpose(1, 0), U)      # [D, D] fp32

                g_total_exp = _ex2_ftz(g_total).reshape(D, 1)           # [D, 1] fp32
                new_state_fp32 = delta_s + state_bf.transpose(1, 0) * g_total_exp
                # .T then to bf16 → store
                new_state = _q_bf16(new_state_fp32.transpose(1, 0))

                # Mutate work_state[seq_idx, h]. MLX supports advanced index set.
                work_state[seq_idx, h] = new_state

                per_head_outs.append(out_h)  # [CHUNK, D] fp32(bf16-valued)

            # Stack per-head: [CHUNK, H, D]
            chunk_out = mx.stack(per_head_outs, axis=1)
            # Trim to actual_len
            if actual_len < CHUNK:
                chunk_out = chunk_out[:actual_len]
            seq_out_chunks.append(chunk_out)

        # Concatenate chunks for this sequence
        if seq_out_chunks:
            all_out_chunks.append(mx.concatenate(seq_out_chunks, axis=0))

    # Assemble full output [T_total, H, D] then reshape to [B, T_seq, H, D]
    if all_out_chunks:
        out_flat = mx.concatenate(all_out_chunks, axis=0)
    else:
        out_flat = mx.zeros((T_total, H, D), dtype=mx.float32)
    out = out_flat.reshape(B, T_seq, H, D)

    # Final state handling
    final: Optional[mx.array] = None
    if want_final:
        if state_fp32:
            final = work_state.astype(mx.float32)
        else:
            final = _q_bf16(work_state).astype(mx.bfloat16)

    return out, final
