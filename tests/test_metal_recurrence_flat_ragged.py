"""PR M Option B parity tests for the flat-ragged cross-chunk simdgroup
Metal recurrence kernel.

Oracle: N independent calls to ``metal_recurrence_cross_chunk_simdgroup``
on per-seq slices of the flat-ragged prepare buffers (exactly the path
that PR M Option A currently runs). The flat-ragged kernel must produce
byte-exact outputs and per-seq states identical to the per-seq oracle.

Skipped on M1/M2 per plan §8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flash_kda_mlx import _metal_recurrence as _mr  # noqa: E402
from flash_kda_mlx._metal_recurrence import HAS_METAL_KERNEL  # noqa: E402


D = 128
CHUNK = 16


requires_metal = pytest.mark.skipif(
    not HAS_METAL_KERNEL,
    reason="Metal cross-chunk path disabled (M3+ only, per plan §8)",
)

_FLAT = getattr(_mr, "metal_recurrence_cross_chunk_flat_ragged", None)
_SINGLE = getattr(_mr, "metal_recurrence_cross_chunk_simdgroup", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_flat_ragged_inputs(
    seq_lens: list[int],
    H: int,
    *,
    seed: int = 0,
) -> tuple[mx.array, dict[str, mx.array], mx.array, mx.array, list[int]]:
    """Build random flat-ragged prepare-style inputs for a list of seq_lens.

    Returns (state_in, prepare_buffers, seq_chunk_start, seq_token_start,
    n_chunks_per_seq_list).
    """
    rng = np.random.default_rng(seed)
    N = len(seq_lens)
    n_chunks_per_seq = [(sl + CHUNK - 1) // CHUNK for sl in seq_lens]
    total_chunks = sum(n_chunks_per_seq)
    T_total = sum(seq_lens)

    seq_chunk_offsets = [0]
    for nc in n_chunks_per_seq:
        seq_chunk_offsets.append(seq_chunk_offsets[-1] + nc)
    tok_offsets = [0]
    for sl in seq_lens:
        tok_offsets.append(tok_offsets[-1] + sl)

    seq_chunk_start = mx.array(seq_chunk_offsets, dtype=mx.int32)
    seq_token_start = mx.array(tok_offsets, dtype=mx.int32)

    def _r32(*shape: int) -> mx.array:
        return mx.array(
            rng.standard_normal(size=shape, dtype=np.float32) * 0.1,
            dtype=mx.float32,
        )

    state_in = _r32(N, H, D, D)
    k_decayed = _r32(total_chunks, H, CHUNK, D)
    q_decayed = _r32(total_chunks, H, CHUNK, D)
    k_restored = _r32(total_chunks, H, CHUNK, D)
    Mqk = _r32(total_chunks, H, CHUNK, CHUNK)
    INV_bf = _r32(total_chunks, H, CHUNK, CHUNK)
    vc = _r32(total_chunks, H, CHUNK, D)
    beta_bf16 = _r32(total_chunks, H, CHUNK, 1)
    # ``g_total_exp`` is an exp() result in the real pipeline, so it's
    # strictly positive. Use abs to mirror that.
    g_total_exp = mx.abs(_r32(total_chunks, H, D, 1)) + 0.5

    prepare = {
        "k_decayed": k_decayed,
        "q_decayed": q_decayed,
        "k_restored": k_restored,
        "Mqk": Mqk,
        "INV_bf": INV_bf,
        "vc": vc,
        "beta_bf16": beta_bf16,
        "g_total_exp": g_total_exp,
    }
    mx.eval(state_in, *prepare.values(), seq_chunk_start, seq_token_start)
    return state_in, prepare, seq_chunk_start, seq_token_start, n_chunks_per_seq


def _per_seq_oracle(
    state_in: mx.array,
    prepare: dict[str, mx.array],
    seq_lens: list[int],
    n_chunks_per_seq: list[int],
) -> tuple[mx.array, mx.array]:
    """Reference: N independent calls to the single-seq simdgroup kernel,
    matching exactly what PR M Option A's per-seq recurrence loop runs.
    Returns (out_flat[T_total, H, D], new_state[N, H, D, D])."""
    N = state_in.shape[0]
    H = state_in.shape[1]
    seq_chunk_offsets = [0]
    for nc in n_chunks_per_seq:
        seq_chunk_offsets.append(seq_chunk_offsets[-1] + nc)
    seq_outs: list[mx.array] = []
    new_state_slices: list[mx.array] = []
    for n in range(N):
        cs, ce = seq_chunk_offsets[n], seq_chunk_offsets[n + 1]
        seq_len = seq_lens[n]
        out_h, ns = _SINGLE(
            state_in[n],
            prepare["k_decayed"][cs:ce],
            prepare["q_decayed"][cs:ce],
            prepare["k_restored"][cs:ce],
            prepare["Mqk"][cs:ce],
            prepare["INV_bf"][cs:ce],
            prepare["vc"][cs:ce],
            prepare["beta_bf16"][cs:ce],
            prepare["g_total_exp"][cs:ce],
        )
        # out_h has shape [n_chunks*CHUNK, H, D]; trim to seq_len.
        if out_h.shape[0] > seq_len:
            out_h = out_h[:seq_len]
        seq_outs.append(out_h)
        new_state_slices.append(ns)
    out_flat = (
        seq_outs[0] if N == 1 else mx.concatenate(seq_outs, axis=0)
    )
    new_state = mx.stack(new_state_slices, axis=0)
    mx.eval(out_flat, new_state)
    return out_flat, new_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_metal
@pytest.mark.parametrize(
    "seq_lens, H",
    [
        ([64, 64], 4),
        ([48, 80], 4),
        ([15, 17, 31, 49], 4),
        ([16, 32, 48, 64], 4),
        ([7, 16, 1], 4),                                    # ragged tails incl 1-token seq
        ([128, 64, 32, 96], 8),
        ([100, 33, 200, 17], 8),
        ([512, 256, 128], 4),
    ],
    ids=[
        "uniform_N2_H4", "ragged_N2_H4", "ragged_N4_tiny",
        "uniform_chunks_N4", "with_singleton", "mixed_N4_H8",
        "wide_mixed_H8", "long_seqs",
    ],
)
def test_flat_ragged_byte_exact_vs_per_seq(seq_lens, H):
    state_in, prepare, scs, sts, ncs = _build_flat_ragged_inputs(
        seq_lens, H=H, seed=0,
    )
    T_total = sum(seq_lens)
    N = len(seq_lens)

    out_flat, new_state = _FLAT(
        state_in,
        prepare["k_decayed"],
        prepare["q_decayed"],
        prepare["k_restored"],
        prepare["Mqk"],
        prepare["INV_bf"],
        prepare["vc"],
        prepare["beta_bf16"],
        prepare["g_total_exp"],
        scs,
        sts,
        T_total,
    )
    mx.eval(out_flat, new_state)

    out_ref, new_state_ref = _per_seq_oracle(state_in, prepare, seq_lens, ncs)

    assert out_flat.shape == (T_total, H, D)
    assert out_flat.shape == out_ref.shape
    assert new_state.shape == (N, H, D, D)
    assert new_state.shape == new_state_ref.shape

    # Byte-exact: the flat-ragged kernel runs the SAME math on the SAME inputs;
    # only the dispatch geometry and output indexing differ.
    out_diff = mx.max(mx.abs(out_flat - out_ref)).item()
    state_diff = mx.max(mx.abs(new_state - new_state_ref)).item()
    assert out_diff == 0.0, f"output max-abs diff = {out_diff}"
    assert state_diff == 0.0, f"new_state max-abs diff = {state_diff}"


@requires_metal
def test_flat_ragged_handles_n_equals_1():
    """N=1 should match the single-seq kernel exactly (seq_chunk_start
    is just [0, n_chunks])."""
    state_in, prepare, scs, sts, ncs = _build_flat_ragged_inputs(
        [129], H=4, seed=42,
    )
    T_total = 129
    out_flat, new_state = _FLAT(
        state_in,
        prepare["k_decayed"], prepare["q_decayed"], prepare["k_restored"],
        prepare["Mqk"], prepare["INV_bf"], prepare["vc"],
        prepare["beta_bf16"], prepare["g_total_exp"],
        scs, sts, T_total,
    )
    out_ref, new_state_ref = _per_seq_oracle(state_in, prepare, [129], ncs)
    mx.eval(out_flat, new_state, out_ref, new_state_ref)
    assert mx.max(mx.abs(out_flat - out_ref)).item() == 0.0
    assert mx.max(mx.abs(new_state - new_state_ref)).item() == 0.0


@requires_metal
def test_flat_ragged_does_not_corrupt_neighbor_rows():
    """Tail rows of seq[k] (where chunk*CHUNK > seq_len) MUST NOT be
    written — otherwise they would clobber seq[k+1]'s leading rows in
    the flat T_total output buffer."""
    seq_lens = [17, 33, 25]               # all have partial-last-chunks
    state_in, prepare, scs, sts, ncs = _build_flat_ragged_inputs(
        seq_lens, H=4, seed=7,
    )
    T_total = sum(seq_lens)
    out_flat, new_state = _FLAT(
        state_in,
        prepare["k_decayed"], prepare["q_decayed"], prepare["k_restored"],
        prepare["Mqk"], prepare["INV_bf"], prepare["vc"],
        prepare["beta_bf16"], prepare["g_total_exp"],
        scs, sts, T_total,
    )
    out_ref, new_state_ref = _per_seq_oracle(
        state_in, prepare, seq_lens, ncs,
    )
    mx.eval(out_flat, new_state, out_ref, new_state_ref)

    # Per-seq slice equality.
    tok_off = 0
    for sl in seq_lens:
        sl_out = out_flat[tok_off:tok_off + sl]
        sl_ref = out_ref[tok_off:tok_off + sl]
        diff = mx.max(mx.abs(sl_out - sl_ref)).item()
        assert diff == 0.0, f"slice [{tok_off}:{tok_off+sl}] diff={diff}"
        tok_off += sl


@requires_metal
def test_flat_ragged_state_independence_per_seq():
    """Each seq's final state must depend ONLY on its own chunks and
    its own initial state slot — never on other seqs' inputs.

    Verify by perturbing seq 1's prepare buffers and confirming seq 0's
    new_state is unchanged.
    """
    seq_lens = [32, 64]
    state_in, prepare, scs, sts, ncs = _build_flat_ragged_inputs(
        seq_lens, H=4, seed=11,
    )
    T_total = sum(seq_lens)
    _, ns_a = _FLAT(
        state_in,
        prepare["k_decayed"], prepare["q_decayed"], prepare["k_restored"],
        prepare["Mqk"], prepare["INV_bf"], prepare["vc"],
        prepare["beta_bf16"], prepare["g_total_exp"],
        scs, sts, T_total,
    )
    mx.eval(ns_a)

    # Perturb seq 1's chunks (chunks 2..5 because seq 0 has 2 chunks).
    rng = np.random.default_rng(99)
    k_dec_perturbed = mx.array(prepare["k_decayed"])
    seq1_chunks = mx.array(
        rng.standard_normal(
            size=(4, 4, CHUNK, D), dtype=np.float32,
        ) * 0.1,
        dtype=mx.float32,
    )
    # Replace rows for seq 1 (chunks 2..6 = chunks for seq 1 of len 64 = 4 chunks).
    k_dec_perturbed_np = np.array(k_dec_perturbed, copy=True)
    k_dec_perturbed_np[2:6] = np.array(seq1_chunks)
    k_dec_perturbed = mx.array(k_dec_perturbed_np)

    _, ns_b = _FLAT(
        state_in,
        k_dec_perturbed, prepare["q_decayed"], prepare["k_restored"],
        prepare["Mqk"], prepare["INV_bf"], prepare["vc"],
        prepare["beta_bf16"], prepare["g_total_exp"],
        scs, sts, T_total,
    )
    mx.eval(ns_b)

    # seq 0's state must be IDENTICAL since we only perturbed seq 1's inputs.
    seq0_diff = mx.max(mx.abs(ns_a[0] - ns_b[0])).item()
    assert seq0_diff == 0.0, f"seq 0 contaminated by seq 1 perturbation: {seq0_diff}"
    # seq 1's state SHOULD differ (sanity check that perturbation took effect).
    seq1_diff = mx.max(mx.abs(ns_a[1] - ns_b[1])).item()
    assert seq1_diff > 0.0, "seq 1 perturbation had no effect — test is broken"


@requires_metal
def test_flat_ragged_large_h_96():
    """H=96 stresses the grid_z * H * 1024 dispatch shape."""
    seq_lens = [64, 32, 96]
    state_in, prepare, scs, sts, ncs = _build_flat_ragged_inputs(
        seq_lens, H=96, seed=3,
    )
    T_total = sum(seq_lens)
    out_flat, new_state = _FLAT(
        state_in,
        prepare["k_decayed"], prepare["q_decayed"], prepare["k_restored"],
        prepare["Mqk"], prepare["INV_bf"], prepare["vc"],
        prepare["beta_bf16"], prepare["g_total_exp"],
        scs, sts, T_total,
    )
    out_ref, new_state_ref = _per_seq_oracle(state_in, prepare, seq_lens, ncs)
    mx.eval(out_flat, new_state, out_ref, new_state_ref)
    assert mx.max(mx.abs(out_flat - out_ref)).item() == 0.0
    assert mx.max(mx.abs(new_state - new_state_ref)).item() == 0.0
