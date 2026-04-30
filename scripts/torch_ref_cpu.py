"""Pure-torch, device-agnostic port of ``tests/torch_ref.py``.

This is the algorithmic oracle used by the MLX rewrite track. It differs from
``tests/torch_ref.py`` in two carefully scoped places so it can run without
CUDA:

* ``sigmoid_tanh_fp32`` (CUDA inline extension calling the ``tanh.approx.f32``
  PTX intrinsic) → ``torch.sigmoid`` on fp32. Mathematically identical; the
  PTX intrinsic is an *approximation*, so CPU values are strictly closer to
  the true sigmoid than the CUDA kernel.
* ``matmul_fp16acc`` (cuBLAS ``cublasGemmEx`` with ``CUBLAS_COMPUTE_16F``) →
  ``torch.matmul`` with fp16 inputs. On CPU PyTorch accumulates in fp32,
  which is again strictly more precise than the kernel path.

Everything else — the L2 normalisation reduction tree, the base-2 exponent
(``fp32_ex2_ftz``), the bf16 state storage semantics, the chunk loop, the
Neumann-series inverse construction — is preserved line-for-line against the
CUDA reference so MLX parity tolerances can be tight.

Consequences for parity testing:

* Outputs from this oracle will NOT be bit-exact against the CUDA kernel.
  They will be within the "CUDA sigmoid approximation + fp16 GEMM" envelope,
  typically a few ulp in bf16.
* MLX parity uses ``numpy.testing.assert_allclose`` with tolerances tuned in
  ``tests/test_parity_fixtures.py``.
"""

from __future__ import annotations

import torch

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Numeric helpers (same signatures as tests/torch_ref.py)
# ---------------------------------------------------------------------------

def fp32_ex2_ftz(x: torch.Tensor) -> torch.Tensor:
    """Base-2 exponent, flush-to-zero for subnormals (matches ex2.approx.ftz.f32)."""
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    ret = torch.special.exp2(x)
    tiny = torch.finfo(torch.float32).tiny
    ret = torch.where(ret.abs() < tiny, torch.zeros_like(ret), ret)
    return ret


def fp32_fma(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Emulate an fp32 FMA via fp64 intermediate, matching torch_ref.py."""
    assert c.dtype == torch.float32
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def l2_normalize_kernel_match(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize matching the kernel's warp-shuffle tree reduction with FMA.

    ``x`` has shape ``[..., D]`` with ``D == 128``. Returns the normalized
    tensor with the original dtype. The reduction tree mirrors the CUDA
    kernel so comparisons at identical precision remain tight.
    """
    x_f32 = x.float()
    groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)

    partials = torch.zeros(*x_f32.shape[:-1], 16, dtype=torch.float32, device=x.device)
    for i in range(8):
        partials = fp32_fma(partials, groups[..., i], groups[..., i])

    for offset in [8, 4, 2, 1]:
        indices = torch.arange(16, device=x.device) ^ offset
        partials = partials + partials[..., indices]

    inv_norm = torch.rsqrt(partials[..., 0:1] + 1e-6)
    return (x_f32 * inv_norm).to(x.dtype)


def _sigmoid_cpu(x: torch.Tensor) -> torch.Tensor:
    """Replacement for the CUDA tanh.approx.f32-based sigmoid.

    Always computes in fp32 to match the precision of the CUDA path's
    ``float`` intermediates. Returns fp32.
    """
    return torch.sigmoid(x.to(torch.float32))


def _matmul_fp16acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Replacement for cuBLAS ``CUBLAS_COMPUTE_16F`` GEMM.

    The CUDA path intentionally accumulates in fp16 to stay within the
    Neumann-series dynamic-range argument from the deep-dive doc. On CPU we
    cannot request fp16 accumulation, so we accept fp32 accumulation and let
    MLX parity tolerances absorb the difference.
    """
    assert a.shape[-1] == b.shape[-2]
    return torch.matmul(a.to(torch.float16), b.to(torch.float16))


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def torch_ref_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> None:
    """Device-agnostic port of ``torch_ref`` — in-place writes to ``out``/``final_state``.

    Input tensors must be on the same device (CPU or MPS). Shapes/dtypes
    follow ``flash_kda.fwd`` exactly:

    * ``q/k/v/g``: bf16, ``[B, T, H, D]``
    * ``beta``: bf16, ``[B, T, H]``
    * ``A_log``: fp32, ``[H]``
    * ``dt_bias``: fp32, ``[H, D]``
    * ``out``: bf16, ``[B, T, H, D]`` (written in place)
    * ``initial_state`` / ``final_state``: ``[N, H, D, D]`` bf16 or fp32
    * ``cu_seqlens``: int64 ``[N+1]`` (``B`` must be 1 when provided)
    """
    assert q.dim() == 4, f"Expected 4D input [B, T, H, D], got {q.dim()}D"
    B = q.shape[0]
    if cu_seqlens is not None:
        assert B == 1, f"B must be 1 when cu_seqlens is provided, got B={B}"

    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    g = g.reshape(-1, *g.shape[2:])
    beta = beta.reshape(-1, *beta.shape[2:])
    out = out.reshape(-1, *out.shape[2:])

    device = q.device
    if B > 1:
        T_seq = q.shape[0] // B
        cu_seqlens = torch.arange(0, B * T_seq + 1, T_seq, dtype=torch.long, device=device)

    _, H, D = q.shape
    CHUNK = 16
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device=device)

    q = l2_normalize_kernel_match(q)
    k = l2_normalize_kernel_match(k)

    if A_log is not None:
        assert dt_bias is not None
        assert A_log.dtype == torch.float32
        assert g.dtype == torch.bfloat16
        assert dt_bias.dtype == torch.float32
        g = g.to(torch.float32) + dt_bias.unsqueeze(0)
        a_log_exp = fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)
        gate_scalar = lower_bound * LOG2E
        g = gate_scalar * _sigmoid_cpu(a_log_exp * g)

    state_fp32 = (
        (initial_state is not None and initial_state.dtype == torch.float32)
        or (final_state is not None and final_state.dtype == torch.float32)
    )

    if cu_seqlens is None:
        T = q.shape[0]
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)

    N = len(cu_seqlens) - 1

    if initial_state is not None:
        work_state = initial_state.to(torch.bfloat16).clone()
    else:
        work_state = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=device)

    for seq_idx in range(N):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        seq_len = eos - bos
        n_chunks = (seq_len + CHUNK - 1) // CHUNK

        for chunk_idx in range(n_chunks):
            t0 = bos + chunk_idx * CHUNK
            actual_len = min(CHUNK, eos - t0)

            for h in range(H):
                g_chunk = torch.zeros(CHUNK, D, dtype=g.dtype, device=device)
                q_chunk = torch.zeros(CHUNK, D, dtype=q.dtype, device=device)
                k_chunk = torch.zeros(CHUNK, D, dtype=k.dtype, device=device)
                v_chunk = torch.zeros(CHUNK, D, dtype=v.dtype, device=device)
                beta_chunk = torch.zeros(CHUNK, dtype=beta.dtype, device=device)

                g_chunk[:actual_len] = g[t0:t0 + actual_len, h, :]
                q_chunk[:actual_len] = q[t0:t0 + actual_len, h, :]
                k_chunk[:actual_len] = k[t0:t0 + actual_len, h, :]
                v_chunk[:actual_len] = v[t0:t0 + actual_len, h, :]
                beta_chunk[:actual_len] = beta[t0:t0 + actual_len, h]

                g_cumsum = g_chunk.cumsum(dim=0)
                g_total = g_cumsum[-1:]
                k_decayed = k_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
                q_decayed = q_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
                neg_g_cumsum_bf16 = fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
                k_inv = k_chunk * neg_g_cumsum_bf16
                g_total_exp_bf16 = fp32_ex2_ftz(g_total).to(torch.bfloat16)
                k_restored = k_inv * g_total_exp_bf16

                # GEMM in fp32 then cast to fp16 (matches torch_ref.py line)
                L = torch.matmul(k_decayed.float(), k_inv.float().t()).to(torch.float16)
                Mqk = torch.matmul(q_decayed, k_inv.t())

                beta_activated = _sigmoid_cpu(beta_chunk)
                beta_val_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)
                beta_val_fp16 = beta_activated.to(torch.float16).unsqueeze(-1)
                L = torch.tril(L, diagonal=-1) * beta_val_fp16
                Mqk = torch.tril(Mqk)

                INV = torch.eye(CHUNK, dtype=torch.float16, device=device) - L
                L2 = _matmul_fp16acc(L, L)
                INV = INV + _matmul_fp16acc(INV, L2)
                L4 = _matmul_fp16acc(L2, L2)
                INV = INV + _matmul_fp16acc(INV, L4)
                L8 = _matmul_fp16acc(L4, L4)
                INV = INV + _matmul_fp16acc(INV, L8)

                INV = INV.to(torch.bfloat16)

                state_slice = work_state[seq_idx, h]
                v_chunk = v_chunk - torch.matmul(k_decayed, state_slice.t())
                v_chunk = v_chunk * beta_val_bf16

                U = torch.matmul(INV, v_chunk)
                _out = torch.matmul(q_decayed, state_slice.t())
                _out = _out + torch.matmul(Mqk, U)

                delta_s = torch.matmul(k_restored.float().t(), U.float())

                g_total_exp = fp32_ex2_ftz(g_total).squeeze(0).unsqueeze(-1)
                work_state[seq_idx, h] = fp32_fma(
                    delta_s, state_slice.to(torch.float32).t(), g_total_exp
                ).to(torch.bfloat16).t()

                out[t0:t0 + actual_len, h] = _out[:actual_len]

    if final_state is not None:
        if state_fp32:
            final_state.copy_(work_state.to(torch.float32))
        else:
            final_state.copy_(work_state)
