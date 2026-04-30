"""Contract / shape / dtype tests for ``flash_kda_mlx.baselines.chunk_kda``.

Scope: the MLX-LM KDA adapter (PR C) is a scaffold that must match the kwarg
contract of ``fla.ops.kda.chunk_kda`` as called from ``benchmarks/bench_fwd.py``.
Small torch-reference parity lives in
``tests/test_chunk_baseline_torch_reference.py``; this file keeps the
broader adapter contract coverage.

Dimensions:

* ``B, T, H, D`` choices follow the FlashKDA benchmark contract (``D = 128``).
* Tiny shapes (``T ∈ {16, 32}``, ``H ∈ {1, 2}``) keep per-test runtime low on
  CPU-only macOS hosts; the adapter doesn't need large shapes for a contract
  check.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from flash_kda_mlx.baselines.chunk_kda import chunk_kda_mlx

from _helpers import (
    fixture_to_mlx_inputs,
    load_fixture,
    make_inputs,
    make_varlen_inputs,
    to_numpy,
)


def _call_adapter(
    inputs: dict,
    *,
    initial_state: mx.array | None = None,
    output_final_state: bool = True,
    use_gate_in_kernel: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    transpose_state_layout: bool = True,
    cu_seqlens: mx.array | None = None,
    use_kernel: bool = False,
    lower_bound: float | None = None,
    g_override: mx.array | None = None,
    beta_override: mx.array | None = None,
):
    """Thin wrapper collapsing the common call shape used across tests.

    ``use_kernel=False`` by default so these tests exercise the ops path and
    remain runnable on CPU-only macOS CI.
    """
    g = g_override if g_override is not None else inputs["g"]
    beta = beta_override if beta_override is not None else inputs["beta"]
    lb = lower_bound if lower_bound is not None else inputs["lower_bound"]
    return chunk_kda_mlx(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=g,
        beta=beta,
        scale=inputs["scale"],
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=lb,
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

    out, final_state = _call_adapter(inputs, use_kernel=False)

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
    g_np = rng.standard_normal((1, T, H, D)).astype(np.float32)
    beta_np = rng.standard_normal((1, T, H)).astype(np.float32)
    A_log_np = rng.uniform(size=(H,)).astype(np.float32)
    dt_bias_np = rng.uniform(size=(H, D)).astype(np.float32)

    q = mx.array(q_np); k = mx.array(k_np); v = mx.array(v_np)
    g = mx.array(g_np); beta = mx.array(beta_np)
    A_log = mx.array(A_log_np); dt_bias = mx.array(dt_bias_np)
    scale = 1.0 / math.sqrt(D)
    mx.eval(q, k, v, g, beta, A_log, dt_bias)

    # Path A: flag ON, raw q/k.
    out_a, _ = chunk_kda_mlx(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        use_kernel=False,
    )

    # Path B: flag OFF, externally L2-normalize q/k the same way the
    # reference path does (eps=1e-6, rsqrt). The adapter must match that
    # definition.
    def _l2(x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        sq = mx.sum(xf * xf, axis=-1, keepdims=True)
        return (xf * mx.rsqrt(sq + 1e-6)).astype(x.dtype)

    out_b, _ = chunk_kda_mlx(
        q=_l2(q), k=_l2(k), v=v, g=g, beta=beta,
        scale=scale,
        use_qk_l2norm_in_kernel=False,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
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

    rng = np.random.default_rng(11)
    state_fla = mx.array(
        rng.standard_normal((B, H, D, D)).astype(np.float32)
    )  # [N, H, Dk, Dv]
    mx.eval(state_fla)

    out_t, final_t = _call_adapter(
        inputs, initial_state=state_fla, transpose_state_layout=True,
        use_kernel=False,
    )

    # When transpose=False, mlx-lm expects [B, H, Dv, Dk]. Pre-swap axes.
    state_mlx = mx.transpose(state_fla, (0, 1, 3, 2))
    out_f, final_f = _call_adapter(
        inputs, initial_state=state_mlx, transpose_state_layout=False,
        use_kernel=False,
    )

    np.testing.assert_allclose(to_numpy(out_t), to_numpy(out_f), rtol=1e-5, atol=1e-6)
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
    s = 0.5

    out_scale, _ = chunk_kda_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=s,
        use_qk_l2norm_in_kernel=False,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
        use_kernel=False,
    )

    out_prescaled, _ = chunk_kda_mlx(
        q=inputs["q"] * s, k=inputs["k"], v=inputs["v"],
        g=inputs["g"], beta=inputs["beta"],
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        lower_bound=inputs["lower_bound"],
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

    varlen_in = make_varlen_inputs(seq_lens, H, D, seed=21)

    rng = np.random.default_rng(23)
    init_state_fla = mx.array(
        rng.standard_normal((N, H, D, D)).astype(np.float32)
    )  # [N, H, Dk, Dv]
    mx.eval(init_state_fla)

    out_var, final_var = chunk_kda_mlx(
        q=varlen_in["q"], k=varlen_in["k"], v=varlen_in["v"],
        g=varlen_in["g"], beta=varlen_in["beta"],
        scale=varlen_in["scale"],
        initial_state=init_state_fla,
        cu_seqlens=varlen_in["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        A_log=varlen_in["A_log"], dt_bias=varlen_in["dt_bias"],
        lower_bound=varlen_in["lower_bound"],
        transpose_state_layout=True,
        use_kernel=False,
    )

    per_seq_outs = []
    per_seq_finals = []
    cu_list = [int(varlen_in["cu_seqlens"][i].item()) for i in range(N + 1)]
    for n in range(N):
        bos, eos = cu_list[n], cu_list[n + 1]
        q_n = varlen_in["q"][:, bos:eos]
        k_n = varlen_in["k"][:, bos:eos]
        v_n = varlen_in["v"][:, bos:eos]
        g_n = varlen_in["g"][:, bos:eos]
        beta_n = varlen_in["beta"][:, bos:eos]
        h0_n = init_state_fla[n:n + 1]

        out_n, fin_n = chunk_kda_mlx(
            q=q_n, k=k_n, v=v_n, g=g_n, beta=beta_n,
            scale=varlen_in["scale"],
            initial_state=h0_n,
            use_qk_l2norm_in_kernel=True,
            A_log=varlen_in["A_log"], dt_bias=varlen_in["dt_bias"],
            lower_bound=varlen_in["lower_bound"],
            transpose_state_layout=True,
            use_kernel=False,
        )
        per_seq_outs.append(out_n)
        per_seq_finals.append(fin_n)

    out_concat = mx.concatenate(per_seq_outs, axis=1)
    final_concat = mx.concatenate(per_seq_finals, axis=0)

    np.testing.assert_allclose(to_numpy(out_var), to_numpy(out_concat), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(final_var), to_numpy(final_concat), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. output_final_state=False returns None for the state slot.
# ---------------------------------------------------------------------------

def test_output_final_state_false():
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=31)

    out, final_state = _call_adapter(
        inputs, output_final_state=False, use_kernel=False,
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

    out_k, final_k = _call_adapter(inputs, use_kernel=True)
    out_o, final_o = _call_adapter(inputs, use_kernel=False)

    np.testing.assert_allclose(to_numpy(out_k), to_numpy(out_o), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(to_numpy(final_k), to_numpy(final_o), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. State dtype preservation: fp32 stays fp32, bf16 stays bf16.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16])
def test_state_dtype_preserved(state_dtype):
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=51)

    rng = np.random.default_rng(53)
    init_state = mx.array(
        rng.standard_normal((1, H, D, D)).astype(np.float32)
    ).astype(state_dtype)
    mx.eval(init_state)

    _, final_state = _call_adapter(
        inputs, initial_state=init_state, use_kernel=False,
    )
    assert final_state is not None
    assert final_state.dtype == state_dtype, (
        f"final_state dtype {final_state.dtype} != initial_state dtype {state_dtype}"
    )


# ---------------------------------------------------------------------------
# 9. use_gate_in_kernel=False: caller provides pre-computed multiplicative g.
# ---------------------------------------------------------------------------

def test_use_gate_in_kernel_false_matches_precomputed_gate():
    """Flag-off with pre-computed multiplicative gate matches flag-on with raw g.

    Proves the adapter's gate-formula path can be bypassed cleanly by a
    caller that has already computed the per-step decay.
    """
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=61)
    lb = inputs["lower_bound"]

    # Path A: default — adapter computes the gate internally.
    out_a, final_a = _call_adapter(
        inputs, use_gate_in_kernel=True, use_kernel=False,
    )

    # Path B: pre-compute the decay exactly as the adapter would, then pass
    # flag=False so the adapter forwards it through untouched.
    pre = inputs["g"] + inputs["dt_bias"].reshape(1, 1, H, D)
    pre_clamped = mx.maximum(pre, lb)
    a_log_exp = mx.exp(inputs["A_log"].astype(mx.float32)).reshape(1, 1, -1, 1)
    gate = mx.exp(-a_log_exp * nn.softplus(pre_clamped))
    mx.eval(gate)

    out_b, final_b = _call_adapter(
        inputs, use_gate_in_kernel=False,
        g_override=gate, use_kernel=False,
    )

    np.testing.assert_allclose(to_numpy(out_a), to_numpy(out_b), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(final_a), to_numpy(final_b), rtol=1e-5, atol=1e-6)


def test_gate_off_does_not_require_A_log_dt_bias():
    """With ``use_gate_in_kernel=False``, ``A_log``/``dt_bias`` are unused.

    Exercises the precondition that the adapter must accept ``A_log=None``
    and ``dt_bias=None`` whenever the gate is pre-computed by the caller.
    The other gate-off test always passes ``A_log``/``dt_bias`` through
    ``_call_adapter``, so it doesn't cover this path.
    """
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=99)
    gate = mx.ones((1, T, H, D), dtype=mx.float32)  # trivial (no-decay) gate
    mx.eval(gate)

    out, _ = chunk_kda_mlx(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g=gate, beta=inputs["beta"],
        scale=inputs["scale"],
        use_gate_in_kernel=False,
        A_log=None, dt_bias=None,
        use_kernel=False,
    )
    assert out.shape == (1, T, H, D)
    assert bool(mx.any(out != 0).item())


# ---------------------------------------------------------------------------
# 10. use_beta_sigmoid_in_kernel=False: caller passes pre-sigmoided beta.
# ---------------------------------------------------------------------------

def test_use_beta_sigmoid_in_kernel_false_matches_presigmoided_beta():
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=71)

    out_a, final_a = _call_adapter(
        inputs, use_beta_sigmoid_in_kernel=True, use_kernel=False,
    )

    beta_presig = mx.sigmoid(inputs["beta"])
    mx.eval(beta_presig)
    out_b, final_b = _call_adapter(
        inputs, use_beta_sigmoid_in_kernel=False,
        beta_override=beta_presig, use_kernel=False,
    )

    np.testing.assert_allclose(to_numpy(out_a), to_numpy(out_b), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(final_a), to_numpy(final_b), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. lower_bound effect: lb=-5 vs lb=-1e9 must differ when pre-clamp inputs
#     fall below -5. Proves the clamp is actually applied.
# ---------------------------------------------------------------------------

def test_lower_bound_clamp_observable():
    T, H, D = 16, 2, 128
    inputs = make_inputs(T, H, D, seed=81)

    # Force some of g + dt_bias below -5 by shifting g strongly negative.
    g_shifted = inputs["g"] - 10.0
    mx.eval(g_shifted)

    # Sanity: pre-clamp values must actually dip below -5, otherwise this
    # test would prove nothing.
    pre = g_shifted + inputs["dt_bias"].reshape(1, 1, H, D)
    assert float(mx.min(pre).item()) < -5.0

    out_clamped, _ = _call_adapter(
        inputs, g_override=g_shifted,
        lower_bound=-5.0, use_kernel=False,
    )
    out_unclamped, _ = _call_adapter(
        inputs, g_override=g_shifted,
        lower_bound=-1e9, use_kernel=False,
    )

    diff = float(mx.max(mx.abs(out_clamped - out_unclamped)).item())
    assert diff > 1e-5, (
        f"lower_bound=-5 and lower_bound=-1e9 produced identical outputs "
        f"(max abs diff {diff}); the clamp must not be applied"
    )


# ---------------------------------------------------------------------------
# 12. Sanity vs ``flash_kda_mlx.reference``: the adapter's output should be in
#     the same ballpark as the FlashKDA correctness oracle. This is a SMOKE
#     test, not a parity test — torch-reference parity for this adapter lives
#     in test_chunk_baseline_torch_reference.py.
# ---------------------------------------------------------------------------

def test_sanity_vs_flash_kda_mlx_reference_loose():
    """Loose smoke check against the reference oracle.

    Tolerances here are intentionally generous (atol=0.5). The adapter and
    the reference follow different chunking / quantization strategies, so
    they are not expected to match to numerical precision. We only assert
    that the adapter produces finite values in roughly the same magnitude
    as the oracle.
    """
    import flash_kda_mlx  # local import keeps top-level import graph tight

    # Pick a small fixed-batch fixture with fp32 state.
    fx = load_fixture("fixed__T16__H1__state_in_out_fp32__seed0")
    kwargs = fixture_to_mlx_inputs(fx)

    # Reference/optimized signature requires out/final_state placeholders.
    q = kwargs["q"]
    out_buf = mx.zeros(q.shape, dtype=mx.float32)
    final_buf = mx.zeros_like(kwargs["initial_state"]) if "initial_state" in kwargs else None

    ref_kwargs = dict(kwargs)
    ref_result = flash_kda_mlx.fwd(
        out=out_buf,
        final_state=final_buf,
        **ref_kwargs,
        backend="reference",
    )

    # Adapter call — beta/g/A_log/dt_bias come straight from the fixture.
    beta = mx.array(fx["beta"])
    mx.eval(beta)
    adapter_kwargs = dict(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g"],
        beta=beta,
        scale=kwargs["scale"],
        initial_state=kwargs.get("initial_state"),
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        lower_bound=kwargs["lower_bound"],
        transpose_state_layout=True,
        cu_seqlens=kwargs.get("cu_seqlens"),
        use_kernel=False,
    )
    out_adapter, _ = chunk_kda_mlx(**adapter_kwargs)

    out_adapter_np = to_numpy(out_adapter)
    out_ref_np = to_numpy(ref_result.out)

    assert np.all(np.isfinite(out_adapter_np)), "adapter output contains non-finite values"
    assert np.all(np.isfinite(out_ref_np)), "reference output contains non-finite values"

    max_abs_diff = float(np.max(np.abs(out_adapter_np - out_ref_np)))
    # Loose bound — different chunking strategy, not a parity test.
    assert max_abs_diff < 0.5, (
        f"adapter vs reference max|diff| = {max_abs_diff:.4f} exceeds loose "
        f"smoke bound 0.5; suggests the adapter is algorithmically broken"
    )
    # Mean-abs bound guards against catastrophic regressions where *every*
    # element is off by ~max_abs_diff (a max-only check would miss those).
    # Measured mean|diff| ≈ 0.001; 0.05 leaves ~50× headroom.
    mean_abs_diff = float(np.mean(np.abs(out_adapter_np - out_ref_np)))
    assert mean_abs_diff < 0.05, (
        f"adapter vs reference mean|diff| = {mean_abs_diff:.4f} exceeds 0.05"
    )
    print(
        f"[sanity_vs_reference] max_abs_diff = {max_abs_diff:.6f} "
        f"mean_abs_diff = {mean_abs_diff:.6f}"
    )
