"""Torch-reference parity for the MLX-LM-backed chunk baselines.

These tests validate the local MLX baseline adapters without requiring CUDA
fixtures. The oracle is the sequential gated-delta recurrence expressed in
PyTorch, with the same FLA-surface preprocessing that the adapters expose.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from flash_kda_mlx.baselines.chunk_gdn import chunk_gdn_mlx
from flash_kda_mlx.baselines.chunk_kda import chunk_kda_mlx

from _helpers import make_inputs, make_varlen_inputs, to_numpy


def _mt(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(to_numpy(x).astype(np.float32))


def _state_mt(x: mx.array | None) -> torch.Tensor | None:
    if x is None:
        return None
    state = _mt(x)
    return state.to(torch.bfloat16) if x.dtype == mx.bfloat16 else state


def _l2_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(torch.sum(xf * xf, dim=-1, keepdim=True) + eps)


def _run_gated_delta_torch(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential gated-delta recurrence matching mlx-lm's ops path."""
    _, T, _, _ = q.shape
    ys: list[torch.Tensor] = []
    for t in range(T):
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        gt = g[:, t]
        bt = beta[:, t]
        decay = gt[..., None, None] if gt.ndim == 2 else gt[..., None, :]
        state = state * decay
        kv_mem = (state * kt[..., None, :]).sum(dim=-1)
        delta = (vt - kv_mem) * bt[..., None]
        state = state + kt[..., None, :] * delta[..., None]
        ys.append((state * qt[..., None, :]).sum(dim=-1))
    return torch.stack(ys, dim=1), state


def _state_to_native(
    state_fla: torch.Tensor | None,
    *,
    N: int,
    H: int,
    D: int,
    transpose_state_layout: bool,
) -> tuple[torch.Tensor, torch.dtype]:
    if state_fla is None:
        return torch.zeros((N, H, D, D), dtype=torch.float32), torch.float32
    state_dtype = state_fla.dtype
    if transpose_state_layout:
        return state_fla.transpose(-1, -2).float(), state_dtype
    return state_fla.float(), state_dtype


def _state_from_native(
    state_native: torch.Tensor,
    *,
    state_dtype: torch.dtype,
    transpose_state_layout: bool,
) -> torch.Tensor:
    state = state_native.transpose(-1, -2) if transpose_state_layout else state_native
    return state.to(state_dtype)


def _kda_gate_torch(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    pre = g.float() + dt_bias.float().reshape(1, 1, *dt_bias.shape)
    pre = torch.maximum(pre, torch.tensor(lower_bound, dtype=pre.dtype))
    a = torch.exp(A_log.float()).reshape(1, 1, -1, 1)
    return torch.exp(-a * torch.nn.functional.softplus(pre))


def _chunk_kda_torch_reference(
    inputs: dict,
    *,
    initial_state: mx.array | None = None,
    cu_seqlens: mx.array | None = None,
    transpose_state_layout: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    q = _l2_torch(_mt(inputs["q"])) * float(inputs["scale"])
    k = _l2_torch(_mt(inputs["k"]))
    v = _mt(inputs["v"])
    beta = torch.sigmoid(_mt(inputs["beta"]))
    g = _kda_gate_torch(
        _mt(inputs["g"]),
        _mt(inputs["A_log"]),
        _mt(inputs["dt_bias"]),
        float(inputs["lower_bound"]),
    )
    B, T, H, D = q.shape
    N = int(cu_seqlens.shape[0]) - 1 if cu_seqlens is not None else B
    state_fla = _state_mt(initial_state)
    state_native, state_dtype = _state_to_native(
        state_fla, N=N, H=H, D=D, transpose_state_layout=transpose_state_layout,
    )

    if cu_seqlens is None:
        out, fin_native = _run_gated_delta_torch(
            q=q, k=k, v=v, g=g, beta=beta, state=state_native,
        )
    else:
        cu = [int(cu_seqlens[i].item()) for i in range(N + 1)]
        outs: list[torch.Tensor] = []
        fins: list[torch.Tensor] = []
        for n in range(N):
            bos, eos = cu[n], cu[n + 1]
            out_n, fin_n = _run_gated_delta_torch(
                q=q[:, bos:eos],
                k=k[:, bos:eos],
                v=v[:, bos:eos],
                g=g[:, bos:eos],
                beta=beta[:, bos:eos],
                state=state_native[n:n + 1],
            )
            outs.append(out_n)
            fins.append(fin_n)
        out = torch.cat(outs, dim=1)
        fin_native = torch.cat(fins, dim=0)

    final = _state_from_native(
        fin_native, state_dtype=state_dtype, transpose_state_layout=transpose_state_layout,
    )
    return out.detach().numpy(), final.float().detach().numpy()


def _chunk_gdn_torch_reference(
    inputs: dict,
    *,
    g_scalar: mx.array,
    initial_state: mx.array | None = None,
    cu_seqlens: mx.array | None = None,
    transpose_state_layout: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    q = _l2_torch(_mt(inputs["q"])) * float(inputs["scale"])
    k = _l2_torch(_mt(inputs["k"]))
    v = _mt(inputs["v"])
    g = torch.exp(_mt(g_scalar))
    beta = torch.sigmoid(_mt(inputs["beta"]))
    B, _, H, D = q.shape
    N = int(cu_seqlens.shape[0]) - 1 if cu_seqlens is not None else B
    state_fla = _state_mt(initial_state)
    state_native, state_dtype = _state_to_native(
        state_fla, N=N, H=H, D=D, transpose_state_layout=transpose_state_layout,
    )

    if cu_seqlens is None:
        out, fin_native = _run_gated_delta_torch(
            q=q, k=k, v=v, g=g, beta=beta, state=state_native,
        )
    else:
        cu = [int(cu_seqlens[i].item()) for i in range(N + 1)]
        outs: list[torch.Tensor] = []
        fins: list[torch.Tensor] = []
        for n in range(N):
            bos, eos = cu[n], cu[n + 1]
            out_n, fin_n = _run_gated_delta_torch(
                q=q[:, bos:eos],
                k=k[:, bos:eos],
                v=v[:, bos:eos],
                g=g[:, bos:eos],
                beta=beta[:, bos:eos],
                state=state_native[n:n + 1],
            )
            outs.append(out_n)
            fins.append(fin_n)
        out = torch.cat(outs, dim=1)
        fin_native = torch.cat(fins, dim=0)

    final = _state_from_native(
        fin_native, state_dtype=state_dtype, transpose_state_layout=transpose_state_layout,
    )
    return out.detach().numpy(), final.float().detach().numpy()


def _state(N: int, H: int, D: int, *, seed: int, dtype: mx.Dtype) -> mx.array:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((N, H, D, D)).astype(np.float32) * 0.1
    state = mx.array(arr).astype(dtype)
    mx.eval(state)
    return state


def _scalar_gate(T: int, H: int, *, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-2.0, 0.0, size=(1, T, H)).astype(np.float32)
    g = mx.array(arr)
    mx.eval(g)
    return g


@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16])
def test_chunk_kda_fixed_matches_torch_reference(state_dtype):
    T, H, D = 9, 2, 16
    inputs = make_inputs(T, H, D, seed=100, dtype=mx.bfloat16)
    initial_state = _state(1, H, D, seed=101, dtype=state_dtype)

    out, final = chunk_kda_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=initial_state,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
        transpose_state_layout=True,
        use_kernel=False,
    )
    ref_out, ref_final = _chunk_kda_torch_reference(
        inputs, initial_state=initial_state, transpose_state_layout=True,
    )

    np.testing.assert_allclose(to_numpy(out), ref_out, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(to_numpy(final), ref_final, rtol=3e-3, atol=3e-3)


def test_chunk_kda_varlen_matches_torch_reference():
    seq_lens = [5, 9, 4]
    H, D = 2, 16
    inputs = make_varlen_inputs(seq_lens, H, D, seed=110, dtype=mx.bfloat16)
    initial_state = _state(len(seq_lens), H, D, seed=111, dtype=mx.float32)

    out, final = chunk_kda_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=initial_state,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
        transpose_state_layout=True,
        cu_seqlens=inputs["cu_seqlens"],
        use_kernel=False,
    )
    ref_out, ref_final = _chunk_kda_torch_reference(
        inputs, initial_state=initial_state,
        cu_seqlens=inputs["cu_seqlens"], transpose_state_layout=True,
    )

    np.testing.assert_allclose(to_numpy(out), ref_out, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(to_numpy(final), ref_final, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16])
def test_chunk_gdn_fixed_matches_torch_reference(state_dtype):
    T, H, D = 9, 2, 16
    inputs = make_inputs(T, H, D, seed=200, dtype=mx.bfloat16)
    g = _scalar_gate(T, H, seed=201)
    initial_state = _state(1, H, D, seed=202, dtype=state_dtype)

    out, final = chunk_gdn_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=g, beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=initial_state,
        transpose_state_layout=True,
        use_kernel=False,
    )
    ref_out, ref_final = _chunk_gdn_torch_reference(
        inputs, g_scalar=g, initial_state=initial_state,
        transpose_state_layout=True,
    )

    np.testing.assert_allclose(to_numpy(out), ref_out, rtol=7e-4, atol=7e-4)
    np.testing.assert_allclose(to_numpy(final), ref_final, rtol=3e-3, atol=3e-3)


def test_chunk_gdn_varlen_matches_torch_reference():
    seq_lens = [5, 9, 4]
    H, D = 2, 16
    inputs = make_varlen_inputs(seq_lens, H, D, seed=210, dtype=mx.bfloat16)
    g = _scalar_gate(sum(seq_lens), H, seed=211)
    initial_state = _state(len(seq_lens), H, D, seed=212, dtype=mx.float32)

    out, final = chunk_gdn_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=g, beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=initial_state,
        transpose_state_layout=True,
        cu_seqlens=inputs["cu_seqlens"],
        use_kernel=False,
    )
    ref_out, ref_final = _chunk_gdn_torch_reference(
        inputs, g_scalar=g, initial_state=initial_state,
        cu_seqlens=inputs["cu_seqlens"], transpose_state_layout=True,
    )

    np.testing.assert_allclose(to_numpy(out), ref_out, rtol=7e-4, atol=7e-4)
    np.testing.assert_allclose(to_numpy(final), ref_final, rtol=3e-3, atol=3e-3)
