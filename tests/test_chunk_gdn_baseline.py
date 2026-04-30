"""Contract / shape / dtype tests for ``flash_kda_mlx.baselines.chunk_gdn``.

Scope: the MLX-LM GDN adapter (PR B) is a scaffold that must match the kwarg
contract of ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` as called
from ``benchmarks/bench_fwd.py``. Small torch-reference parity lives in
``tests/test_chunk_baseline_torch_reference.py``; this file keeps the
broader adapter contract coverage.

Dimensions:

* ``B, T, H, D`` choices follow the FlashKDA benchmark contract (``D = 128``).
* Tiny shapes (``T ∈ {16, 32}``, ``H ∈ {2, 4}``) keep per-test runtime low on
  CPU-only macOS hosts; the adapter doesn't need large shapes for a contract
  check.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from flash_kda_mlx.baselines.chunk_gdn import chunk_gdn_mlx

from _helpers import make_inputs, make_varlen_inputs, to_numpy


def _scalar_gate(T: int, H: int, *, seed: int = 0) -> mx.array:
    """Build a [1, T, H] log-space gate with deterministic seed.

    FLA's ``chunk_gated_delta_rule`` treats ``g`` as log-decay; we keep
    values in roughly ``[-5, 0]`` so ``exp(g)`` lands in ``(0, 1]``.
    """
    rng = np.random.default_rng(seed)
    g_np = rng.uniform(-5.0, 0.0, size=(1, T, H)).astype(np.float32)
    g = mx.array(g_np)
    mx.eval(g)
    return g


def _call_adapter(
    inputs: dict,
    *,
    g_scalar: mx.array,
    initial_state: mx.array | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    transpose_state_layout: bool = True,
    cu_seqlens: mx.array | None = None,
    use_kernel: bool = True,
):
    """Thin wrapper collapsing the common call shape used across tests."""
    return chunk_gdn_mlx(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=g_scalar,
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        transpose_state_layout=transpose_state_layout,
        cu_seqlens=cu_seqlens,
        use_kernel=use_kernel,
    )


# ---------------------------------------------------------------------------
# 1. Fixed-batch shape contract (the RED test for Step 1).
# ---------------------------------------------------------------------------

def test_fixed_batch_shape_contract_minimal():
    B, T, H, D = 1, 16, 2, 128
    inputs = make_inputs(T, H, D, seed=0)
    g = _scalar_gate(T, H, seed=1)

    out, final_state = _call_adapter(inputs, g_scalar=g, use_kernel=False)

    assert isinstance(out, mx.array)
    assert out.shape == (B, T, H, D), f"out shape {out.shape} != {(B, T, H, D)}"
    assert out.dtype in (mx.float32, mx.bfloat16, mx.float16)
    assert bool(mx.any(out != 0).item()), "adapter returned all-zeros output"

    assert final_state is not None
    # transpose_state_layout=True -> [N, H, Dk, Dv] where N=B=1 here.
    assert final_state.shape == (B, H, D, D)
    assert final_state.dtype == mx.float32  # default when initial_state is None


# ---------------------------------------------------------------------------
# 2. L2-norm flag: enabling the flag with un-normalized input matches
#    disabling the flag with externally-L2-normalized input.
# ---------------------------------------------------------------------------

def test_qk_l2norm_flag_matches_external_normalization():
    T, H, D = 16, 2, 128
    rng = np.random.default_rng(42)

    # Build *un-normalized* q/k (the default `make_inputs` already L2-normalizes,
    # so we rebuild to isolate the flag effect).
    q_np = rng.standard_normal((1, T, H, D)).astype(np.float32)
    k_np = rng.standard_normal((1, T, H, D)).astype(np.float32)
    v_np = rng.standard_normal((1, T, H, D)).astype(np.float32)
    beta_np = rng.standard_normal((1, T, H)).astype(np.float32)

    q = mx.array(q_np); k = mx.array(k_np); v = mx.array(v_np)
    beta = mx.array(beta_np)
    g = _scalar_gate(T, H, seed=3)
    scale = 1.0 / math.sqrt(D)
    mx.eval(q, k, v, beta, g)

    # Path A: flag ON, raw q/k.
    out_a, _ = chunk_gdn_mlx(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_kernel=False,
    )

    # Path B: flag OFF, externally L2-normalize q/k the same way the
    # reference path does (see ``flash_kda_mlx.reference._l2_normalize`` — eps=1e-6,
    # rsqrt). The adapter must match that definition.
    def _l2(x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        sq = mx.sum(xf * xf, axis=-1, keepdims=True)
        return (xf * mx.rsqrt(sq + 1e-6)).astype(x.dtype)

    out_b, _ = chunk_gdn_mlx(
        q=_l2(q), k=_l2(k), v=v, g=g, beta=beta,
        scale=scale,
        use_qk_l2norm_in_kernel=False,
        use_kernel=False,
    )

    np.testing.assert_allclose(to_numpy(out_a), to_numpy(out_b), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. State layout transpose: transpose_state_layout flips last two state axes
#    on both input and output.
# ---------------------------------------------------------------------------

def test_transpose_state_layout_axes_swap():
    B, T, H, D = 1, 16, 2, 128
    inputs = make_inputs(T, H, D, seed=7)
    g = _scalar_gate(T, H, seed=9)

    # Build a deterministic fp32 initial state in FLA's transposed layout
    # [N, H, Dk, Dv]. Call once with transpose=True (adapter un-transposes
    # on input and re-transposes on output); call again with transpose=False
    # but pass the manually transposed tensor so mlx-lm sees the same data.
    rng = np.random.default_rng(11)
    state_fla = mx.array(
        rng.standard_normal((B, H, D, D)).astype(np.float32)
    )  # [N, H, Dk, Dv]
    mx.eval(state_fla)

    out_t, final_t = _call_adapter(
        inputs, g_scalar=g, initial_state=state_fla,
        transpose_state_layout=True, use_kernel=False,
    )

    # When transpose=False, mlx-lm expects [B, H, Dv, Dk]. Pre-swap axes.
    state_mlx = mx.transpose(state_fla, (0, 1, 3, 2))
    out_f, final_f = _call_adapter(
        inputs, g_scalar=g, initial_state=state_mlx,
        transpose_state_layout=False, use_kernel=False,
    )

    # Outputs should match regardless of layout convention — the tensor data
    # fed into mlx-lm is the same, only the caller's layout differs.
    np.testing.assert_allclose(to_numpy(out_t), to_numpy(out_f), rtol=1e-5, atol=1e-6)

    # The returned final_state, under transpose=True, must be the axis-swap
    # of the transpose=False return.
    np.testing.assert_allclose(
        to_numpy(final_t),
        to_numpy(mx.transpose(final_f, (0, 1, 3, 2))),
        rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 4. Scale: passing scale=s with unit q matches scale=1 with q pre-scaled.
# ---------------------------------------------------------------------------

def test_scale_applied_to_q():
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=13)
    g = _scalar_gate(T, H, seed=14)
    s = 0.5

    out_scale, _ = chunk_gdn_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=g, beta=inputs["beta"],
        scale=s,
        # Disable L2-norm so the scale effect is not washed out by the norm.
        use_qk_l2norm_in_kernel=False,
        use_kernel=False,
    )

    out_prescaled, _ = chunk_gdn_mlx(
        q=inputs["q"] * s, k=inputs["k"], v=inputs["v"],
        g=g, beta=inputs["beta"],
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        use_kernel=False,
    )

    np.testing.assert_allclose(
        to_numpy(out_scale), to_numpy(out_prescaled), rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 5. Varlen unpack matches concatenated fixed-batch calls.
# ---------------------------------------------------------------------------

def test_varlen_matches_fixed_batch_concat():
    H, D = 2, 128
    seq_lens = [16, 16]
    N = len(seq_lens)
    T_total = sum(seq_lens)

    varlen_in = make_varlen_inputs(seq_lens, H, D, seed=21)
    g_total = _scalar_gate(T_total, H, seed=22)

    # Per-sequence initial states (one per sequence).
    rng = np.random.default_rng(23)
    init_state_fla = mx.array(
        rng.standard_normal((N, H, D, D)).astype(np.float32)
    )  # [N, H, Dk, Dv]
    mx.eval(init_state_fla)

    out_var, final_var = chunk_gdn_mlx(
        q=varlen_in["q"], k=varlen_in["k"], v=varlen_in["v"],
        g=g_total, beta=varlen_in["beta"],
        scale=varlen_in["scale"],
        initial_state=init_state_fla,
        cu_seqlens=varlen_in["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        use_kernel=False,
    )

    # Run each sequence as its own fixed-batch call.
    per_seq_outs = []
    per_seq_finals = []
    cu_list = [int(varlen_in["cu_seqlens"][i].item()) for i in range(N + 1)]
    for n in range(N):
        bos, eos = cu_list[n], cu_list[n + 1]
        q_n = varlen_in["q"][:, bos:eos]
        k_n = varlen_in["k"][:, bos:eos]
        v_n = varlen_in["v"][:, bos:eos]
        beta_n = varlen_in["beta"][:, bos:eos]
        g_n = g_total[:, bos:eos]
        h0_n = init_state_fla[n:n + 1]  # [1, H, Dk, Dv]

        out_n, fin_n = chunk_gdn_mlx(
            q=q_n, k=k_n, v=v_n, g=g_n, beta=beta_n,
            scale=varlen_in["scale"],
            initial_state=h0_n,
            use_qk_l2norm_in_kernel=True,
            transpose_state_layout=True,
            use_kernel=False,
        )
        per_seq_outs.append(out_n)
        per_seq_finals.append(fin_n)

    out_concat = mx.concatenate(per_seq_outs, axis=1)  # along T
    final_concat = mx.concatenate(per_seq_finals, axis=0)  # along N

    np.testing.assert_allclose(to_numpy(out_var), to_numpy(out_concat), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(final_var), to_numpy(final_concat), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. output_final_state=False returns None for the state slot.
# ---------------------------------------------------------------------------

def test_output_final_state_false():
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=31)
    g = _scalar_gate(T, H, seed=32)

    out, final_state = _call_adapter(
        inputs, g_scalar=g, output_final_state=False, use_kernel=False,
    )
    assert final_state is None
    assert out.shape == (1, T, H, D)


# ---------------------------------------------------------------------------
# 7. Kernel vs ops paths agree numerically (sanity check on mlx-lm itself).
# ---------------------------------------------------------------------------

def test_kernel_vs_ops_equivalence():
    if not mx.metal.is_available():
        pytest.skip("Metal unavailable; kernel path cannot be exercised")

    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=41)
    g = _scalar_gate(T, H, seed=42)

    out_k, final_k = _call_adapter(inputs, g_scalar=g, use_kernel=True)
    out_o, final_o = _call_adapter(inputs, g_scalar=g, use_kernel=False)

    np.testing.assert_allclose(to_numpy(out_k), to_numpy(out_o), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(to_numpy(final_k), to_numpy(final_o), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. State dtype preservation: fp32 stays fp32, bf16 stays bf16.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16])
def test_state_dtype_preserved(state_dtype):
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=51)
    g = _scalar_gate(T, H, seed=52)

    rng = np.random.default_rng(53)
    init_state = mx.array(
        rng.standard_normal((1, H, D, D)).astype(np.float32)
    ).astype(state_dtype)
    mx.eval(init_state)

    _, final_state = _call_adapter(
        inputs, g_scalar=g, initial_state=init_state, use_kernel=False,
    )
    assert final_state is not None
    assert final_state.dtype == state_dtype, (
        f"final_state dtype {final_state.dtype} != initial_state dtype {state_dtype}"
    )
