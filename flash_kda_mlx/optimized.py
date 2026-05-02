"""Optimized MLX FlashKDA forward path (plan.md §Phase 8).

Same algorithm, same cast boundaries, same Neumann-series inverse as
:mod:`flash_kda_mlx.reference`. The difference is structural: we vectorize what
the reference spells out as Python loops.

Vectorization axes:

1. **Per-head**: all ``H`` heads evaluate in one leading-axis matmul.
2. **Per-chunk pre-compute**: every state-independent quantity for an entire
   sequence (``g_cumsum``, ``k_decayed``, ``k_inv``, ``k_restored``, ``L``,
   ``INV``, ``Mqk``, beta casts) is built once across all ``n_chunks`` chunks
   in one vectorized dispatch chain. Only the state-carrying recurrence
   (``vdiff → U → out → new_state``) stays in a Python loop.
3. **Per-sequence (packed)**: when ``N > 1`` (varlen with multiple sequences
   or ``B > 1``) the outer sequence loop over pre-compute is folded into a
   batched axis. Each sequence is padded to the same ``max_chunks * CHUNK``
   length, stacked into ``[N, padded, H, D]``, and pre-computed in a single
   dispatch chain. The sequential recurrence then runs chunk-by-chunk over
   the shared ``max_chunks`` axis with a per-sequence validity mask so each
   sequence stops updating state past its own chunk count.
4. **Per-sequence (single)**: when ``N == 1`` we skip the packing overhead
   and run the original fast path directly.

Correctness is guarded by ``tests/test_optimized_parity.py`` at
``rtol=atol=1e-5`` against ``fwd_reference`` for the single-sequence cases
and at a one-ULP-wider band for N>1 packed cases (see that file for why).

Per plan.md §Phase 8 "Important rule": the readable reference remains the
correctness oracle — this module is *in addition to* it.
"""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx

from .reference import (
    CHUNK,
    D_FIXED,
    LOG2E,
    _ex2_ftz,
    _l2_normalize,
    _q_bf16,
    _validate,
)


# Opt-out for the cross-sequence packed path. Useful for A/B benchmarking and
# bisecting potential regressions — not a public API. Set
# ``MLX_KDA_DISABLE_PACKED=1`` to revert to the per-sequence Python loop.
# This is a hard override: when True, per-seq runs unconditionally.
_DISABLE_PACKED = bool(int(os.environ.get("MLX_KDA_DISABLE_PACKED", "0") or "0"))


# Opt-out for PR M Option B's flat-ragged recurrence kernel. When True the
# flat-ragged prepare path (Option A) keeps running but the recurrence
# falls back to the per-seq Python loop + mx.concatenate. Used for A/B
# benchmarking the recurrence-side savings independently of the prepare
# consolidation. Not a public API.
_DISABLE_FLAT_RAGGED_RECURRENCE = bool(
    int(os.environ.get("MLX_KDA_DISABLE_FLAT_RAGGED_RECURRENCE", "0") or "0")
)


# Size-aware packed-vs-per-seq routing threshold (plan PR E, Phase 2b).
#
# The packed strategy pads every sequence to ``max_chunks * chunk`` and runs
# one cross-sequence recurrence. Its cost model is:
#
#   packed_cost  ≈ N × max_chunks × H × per_chunk_kernel_cost
#   per_seq_cost ≈ Σ_n (ceil(len_n / chunk) × H × per_chunk_kernel_cost)
#                  + N × per_seq_python_launch_overhead
#
# Packed wins when amortising the per-seq Python launch overhead across N
# buys more than the padding penalty. Benchmark A/B in
# ``benchmarks/section_timings_report.md`` §8 shows the crossover
# tracks ``max_chunks × H`` cleanly:
#
#   * max_chunks × H ≤ 64:   packed 1.08–2.21× faster (small-N, short T).
#   * max_chunks × H == 4096: within ±10% (uniform varlen at bench scale).
#   * max_chunks × H ≥ 12288: packed 2.24–13.2× slower (mixed-varlen at
#                              bench scale — 55% of chunks are padded).
#
# Threshold 1024 sits 2× below the smallest observed case where per-seq
# wins (4096) and well above the largest small-case max_chunks × H (64),
# so the heuristic routes all seven observed cases correctly with room
# for the regime in between to absorb noise.
_MAX_PACKED_WORK_PER_SEQ = 1024


def _should_use_packed(seq_lens: list[int], H: int, chunk: int) -> bool:
    """Route varlen dispatch based on ``max_chunks × H``.

    Returns ``True`` if the packed cross-sequence path is expected to be
    faster than the per-sequence Python loop at this workload, ``False``
    otherwise.

    Empty ``seq_lens`` returns ``False`` defensively — callers must
    short-circuit ``N == 0`` upstream; this fall-through just makes the
    function safe to call on degenerate inputs.

    The ``_DISABLE_PACKED`` env-var hard-override is checked by the
    caller, not here: this function encodes the heuristic, not policy.

    See ``tests/test_packed_heuristic.py`` for boundary coverage and
    ``benchmarks/section_timings_report.md`` §8 for the A/B data
    that picked 1024 as the threshold.
    """
    if not seq_lens:
        return False
    max_chunks = max((sl + chunk - 1) // chunk for sl in seq_lens)
    return max_chunks * H <= _MAX_PACKED_WORK_PER_SEQ

# Opt-out for the ``mx.compile``-wrapped recurrence step. The wrapped body is
# arithmetically identical but MLX fuses its ~10 ops into fewer dispatches,
# cutting per-chunk launch overhead. Set ``MLX_KDA_DISABLE_COMPILE=1`` to
# revert to the inline body for A/B.
_DISABLE_COMPILE = bool(int(os.environ.get("MLX_KDA_DISABLE_COMPILE", "0") or "0"))

# Opt-in Metal-kernel replacement for the per-chunk recurrence body.
# M3+ only; skipped on older hardware via _metal_recurrence.HAS_METAL_KERNEL
# auto-detect.
#
# MLX_KDA_ENABLE_METAL_RECURRENCE values (production envelope after the
# 2026-04-30 cleanup; see plan.md §8.3 for the historical Phase 2
# per-chunk layer that was retired once Phase 3 superseded it):
#
#   "0" | "" | "off"  : Python + mx.compile per-chunk body (baseline)
#   "1" | "simdgroup" : Phase 3b cross-chunk simdgroup kernel (DEFAULT)
#   "scalar"          : Phase 3a cross-chunk scalar kernel — kept as a
#                       future-hardware A/B anchor (mlx-lm's own
#                       gated_delta_kernel chose scalar+simd_sum over
#                       simdgroup_matrix; on chips where that calculus
#                       differs, scalar can win).
#
# The cross-chunk modes collapse the entire chunk loop for a sequence into
# ONE Metal dispatch per head. They require CHUNK=16 (hardcoded simdgroup
# tile geometry); CHUNK≠16 falls through to the mx.compile path.

_METAL_MODE_ALIASES = {
    "0": "off",
    "": "off",
    "off": "off",
    "false": "off",
    "1": "simdgroup",       # Phase 3c: winner of the A/B is the default
    "simdgroup": "simdgroup",
    "scalar": "scalar",
}
_METAL_MODE_RAW = os.environ.get("MLX_KDA_ENABLE_METAL_RECURRENCE", "0").strip().lower()
if _METAL_MODE_RAW not in _METAL_MODE_ALIASES:
    raise ValueError(
        f"MLX_KDA_ENABLE_METAL_RECURRENCE={_METAL_MODE_RAW!r} is not a "
        f"recognised mode. Valid: {sorted(set(_METAL_MODE_ALIASES))}"
    )
_METAL_MODE = _METAL_MODE_ALIASES[_METAL_MODE_RAW]
_ENABLE_METAL_RECURRENCE = _METAL_MODE != "off"

# ---------------------------------------------------------------------------
# Recurrence step (one chunk)
# ---------------------------------------------------------------------------
#
# Pulled out of the per-chunk Python loop so MLX can fuse the ~10 ops of a
# single iteration into one compiled graph. The body is a pure function of
# its inputs — no captures, no side effects — so ``mx.compile`` can trace it
# once per input-shape tuple and reuse the cached kernel for every chunk.
#
# Profile (``benchmarks/section_timings_report.md``): the sequential
# chunk recurrence owns 74–82% of E2E on long sequences because each chunk
# iteration submits ~10 MLX graph ops with a hard state-dependency between
# chunks. Fusing the body collapses those into fewer dispatches per chunk
# without changing arithmetic order (same bf16 cast boundaries).


def _recurrence_body_single(
    state_slice: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
) -> tuple[mx.array, mx.array]:
    """One chunk of the single-sequence recurrence. Returns ``(out_h, new_state)``.

    All tensors follow the ``[H, ...]`` layout:
    state in/out ``[H, D, D]`` fp32 (bf16-valued); ``out_h`` ``[H, CHUNK, D]``.
    """
    state_bf = _q_bf16(state_slice)
    state_bf_T = state_bf.transpose(0, 2, 1)

    vdiff = vc - _q_bf16(mx.matmul(k_decayed, state_bf_T))
    vdiff = _q_bf16(vdiff)
    vdiff = _q_bf16(vdiff * beta_bf16)

    U = _q_bf16(mx.matmul(INV_bf, vdiff))
    out_h = _q_bf16(mx.matmul(q_decayed, state_bf_T))
    out_h = _q_bf16(out_h + _q_bf16(mx.matmul(Mqk, U)))

    delta_s = mx.matmul(k_restored.transpose(0, 2, 1), U)
    new_state_fp32 = delta_s + state_bf_T * g_total_exp
    new_state = _q_bf16(new_state_fp32.transpose(0, 2, 1))
    return out_h, new_state


def _recurrence_body_packed(
    state: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
    active_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """One chunk of the packed multi-sequence recurrence.

    Leading axes ``[N, H, ...]``; ``active_mask`` is ``[N, 1, 1, 1]`` fp32
    keeping a per-sequence state frozen once ``chunk_idx`` exceeds its own
    chunk count (see ``STATUS.md`` §"Cross-sequence varlen packing
    (Option A — mask-based)"). Returns ``(out_h, new_state)``.
    """
    state_bf = _q_bf16(state)
    state_bf_T = state_bf.transpose(0, 1, 3, 2)

    vdiff = vc - _q_bf16(mx.matmul(k_decayed, state_bf_T))
    vdiff = _q_bf16(vdiff)
    vdiff = _q_bf16(vdiff * beta_bf16)

    U = _q_bf16(mx.matmul(INV_bf, vdiff))
    out_h = _q_bf16(mx.matmul(q_decayed, state_bf_T))
    out_h = _q_bf16(out_h + _q_bf16(mx.matmul(Mqk, U)))

    delta_s = mx.matmul(k_restored.transpose(0, 1, 3, 2), U)
    new_state_fp32 = delta_s + state_bf_T * g_total_exp
    new_state = _q_bf16(new_state_fp32.transpose(0, 1, 3, 2))
    new_state = mx.where(active_mask > 0, new_state, state)
    return out_h, new_state


if not _DISABLE_COMPILE:
    _recurrence_body_single = mx.compile(_recurrence_body_single)
    _recurrence_body_packed = mx.compile(_recurrence_body_packed)


_mx_compiled_body_single = _recurrence_body_single  # preserve for fallback

# Metal kernel handles (set at import if the hardware supports it AND the
# env var selects a Metal mode). ``None`` means: take the mx.compile path.
_HAS_METAL_KERNEL = False
_metal_cross_chunk_fn = None  # Phase 3a / 3b: one call per sequence
_metal_cross_chunk_packed_fn = None  # Phase 4: one call for all N sequences
_metal_cross_chunk_flat_ragged_fn = None  # PR M Option B: one call across N flat-ragged seqs

if _ENABLE_METAL_RECURRENCE:
    from ._metal_recurrence import HAS_METAL_KERNEL as _HAS_METAL_KERNEL  # noqa: E402

    if _HAS_METAL_KERNEL:
        if _METAL_MODE == "scalar":
            # Phase 3a: the chunk loop itself lives inside the Metal kernel.
            # ``_recurrence_body_single`` stays at its mx.compile definition
            # (used only as a fallback for CHUNK≠16); the cross-chunk entry
            # point replaces the Python loop in ``_run_single``.
            from ._metal_recurrence import (  # noqa: E402
                metal_recurrence_cross_chunk_scalar as _metal_cross_chunk_fn,
            )
        elif _METAL_MODE == "simdgroup":
            # Phase 3b + Phase 4: simdgroup for both single-seq and packed.
            from ._metal_recurrence import (  # noqa: E402
                metal_recurrence_cross_chunk_simdgroup as _metal_cross_chunk_fn,
            )
            try:
                from ._metal_recurrence import (  # noqa: E402
                    metal_recurrence_cross_chunk_packed as _metal_cross_chunk_packed_fn,
                )
            except ImportError:
                # Phase 4 optional; leave packed path on mx.compile if absent.
                _metal_cross_chunk_packed_fn = None
            try:
                from ._metal_recurrence import (  # noqa: E402
                    metal_recurrence_cross_chunk_flat_ragged as _metal_cross_chunk_flat_ragged_fn,
                )
            except ImportError:
                # PR M Option B optional; leaves flat-ragged path on the
                # per-seq simdgroup loop if the kernel hasn't shipped yet.
                _metal_cross_chunk_flat_ragged_fn = None


def _metal_cross_chunk_active() -> bool:
    """True when the cross-chunk single-sequence kernel should be used.

    Used by ``_run_single`` to collapse its per-chunk Python loop into one
    Metal dispatch. ``False`` on non-M3+ hardware, when the env var selects
    per-chunk or off, or when the future simdgroup path isn't compiled.
    """
    return _HAS_METAL_KERNEL and _metal_cross_chunk_fn is not None


def _metal_cross_chunk_packed_active() -> bool:
    """True when the packed cross-chunk kernel should be used.

    Used by ``_run_packed`` to collapse its per-chunk Python loop over N
    sequences into one Metal dispatch. Only active under
    ``MLX_KDA_ENABLE_METAL_RECURRENCE=simdgroup`` (or its alias ``=1``)
    since Phase 4 only ships the simdgroup variant.
    """
    return _HAS_METAL_KERNEL and _metal_cross_chunk_packed_fn is not None


def _metal_cross_chunk_flat_ragged_active() -> bool:
    """True when the PR M Option B flat-ragged cross-chunk kernel is loaded.

    Used by the ``use_flat_ragged`` branch in ``fwd_optimized`` to replace
    the per-seq Python recurrence loop AND the trailing
    ``mx.concatenate(seq_outs)`` with a single Metal dispatch over the
    flat-ragged prepare buffers.
    """
    return _HAS_METAL_KERNEL and _metal_cross_chunk_flat_ragged_fn is not None


# ---------------------------------------------------------------------------
# Metal prepare kernel — env-gated, default OFF.
# ---------------------------------------------------------------------------
# MLX_KDA_ENABLE_METAL_PREPARE values (production envelope after
# 2026-04-30 cleanup; see plan.md §8.4 for the historical fused / fused2
# layers that were retired once fused3/fused4 superseded them):
#
#   "0" | "" | "off"               : disabled (baseline)
#   "1" | "on" | "basic"           : basic prepare kernel — sections
#                                    (b)-(g); caller runs section (a) in
#                                    MLX. Strict 1e-5 tolerance band —
#                                    the unique strict-band Metal-prepare
#                                    bisect anchor.
#   "fused3" | "4" | "token_major" : full-fusion prepare kernel reading
#                                    token-major [T_total, H, D] inputs
#                                    directly. 1-bf16-ULP fused2-band
#                                    drift carried through to consumers.
#   "fused4" | "5" | "flat_ragged" : flat-ragged metadata-indirected
#                                    full-fusion prepare. PROD DEFAULT.
#                                    Single-seq (N=1) callers route to
#                                    fused3 internally; multi-seq packed
#                                    varlen callers use the flat-ragged
#                                    indirection.
#
# Each Metal mode introduces a 1-bf16-ULP shift on cumsum-/matmul-derived
# outputs due to reduction-order differences between Metal sequential
# cumsum / simdgroup_matrix and MLX's parallel scan / mx.matmul. That
# shift propagates through the per-chunk recurrence and breaks
# ``test_optimized_parity``'s 1e-5 gate, hence the default-off opt-in.
# ``fused3``/``fused4`` additionally add a ~1-bf16-ULP shift on
# k_decayed/q_decayed (in-kernel L2-norm) and g-derived outputs
# (in-kernel sigmoid).

_METAL_PREPARE_MODE_ALIASES = {
    "0": "off", "": "off", "off": "off", "false": "off",
    "1": "basic", "on": "basic", "basic": "basic",
    "fused3": "fused3", "4": "fused3", "token_major": "fused3",
    "fused4": "fused4", "5": "fused4", "flat_ragged": "fused4",
}
_METAL_PREPARE_MODE_RAW = os.environ.get("MLX_KDA_ENABLE_METAL_PREPARE", "0").strip().lower()
if _METAL_PREPARE_MODE_RAW not in _METAL_PREPARE_MODE_ALIASES:
    raise ValueError(
        f"MLX_KDA_ENABLE_METAL_PREPARE={_METAL_PREPARE_MODE_RAW!r} is not a "
        f"recognised mode. Valid: {sorted(set(_METAL_PREPARE_MODE_ALIASES))}"
    )
_METAL_PREPARE_MODE = _METAL_PREPARE_MODE_ALIASES[_METAL_PREPARE_MODE_RAW]
_ENABLE_METAL_PREPARE = _METAL_PREPARE_MODE != "off"

_metal_prepare_fn = None             # basic variant — (b)-(g) only
_metal_prepare_fused_v3_fn = None    # full fusion (token-major) — PR K-a
_metal_prepare_fused_v4_fn = None    # flat-ragged token-major — PR M Option A

if _ENABLE_METAL_PREPARE:
    try:
        from ._metal_prepare import (  # noqa: E402
            HAS_METAL_KERNEL as _HAS_METAL_PREPARE,
            metal_prepare_chunk as _metal_prepare_fn,
        )
        if _METAL_PREPARE_MODE == "fused3":
            from ._metal_prepare import (  # noqa: E402
                metal_prepare_chunk_fused_v3 as _metal_prepare_fused_v3_fn,
            )
        if _METAL_PREPARE_MODE == "fused4":
            # fused4 also uses the v3 kernel internally for the N==1
            # single-seq fast path: when N==1 the chunk_token_start
            # collapses to {0, CHUNK, 2*CHUNK, ...} which is exactly
            # what v3 already computes, so we route to v3 there to keep
            # the established tile-major recurrence path.
            from ._metal_prepare import (  # noqa: E402
                metal_prepare_chunk_fused_v3 as _metal_prepare_fused_v3_fn,
                metal_prepare_chunk_fused_v4 as _metal_prepare_fused_v4_fn,
            )
    except ImportError:
        _HAS_METAL_PREPARE = False
        _metal_prepare_fn = None
        _metal_prepare_fused_v3_fn = None
        _metal_prepare_fused_v4_fn = None
else:
    _HAS_METAL_PREPARE = False


def _metal_prepare_active() -> bool:
    """True when the basic Metal prepare kernel should be used.

    Strict 1e-5 tolerance band — the unique strict-band Metal-prepare
    bisect anchor. Higher-numbered fused modes (fused3 / fused4) carry
    the documented 1-bf16-ULP fused2-band drift.
    """
    return _HAS_METAL_PREPARE and _metal_prepare_fn is not None


def _metal_prepare_fused_active() -> bool:
    """True when any fused prepare mode (fused3 / fused4) is active.

    Used by ``fwd_optimized`` for the entry-side bf16 preservation +
    L2-norm skip path: both fused kernels read q/k as raw values and
    L2-norm internally.
    """
    return _HAS_METAL_PREPARE and (
        _metal_prepare_fused_v3_fn is not None
        or _metal_prepare_fused_v4_fn is not None
    )


def _metal_prepare_fused_v3_active() -> bool:
    """True when the token-major full-fusion (v3) prepare kernel is selected.

    Functionally equivalent to v2 but reads token-major
    ``[T_total, H, D]`` inputs directly, eliminating the tile-major
    transpose+contiguous-copy that fused2 forces on every forward
    (~1 GB / 8.6 ms at single-seq H=64; PR K-a).

    NB: under ``fused4`` the v3 kernel is ALSO loaded so the single-seq
    (N==1) fast path inside ``_run_single`` keeps using it directly —
    fused4's value is on the multi-seq packed varlen path. Both
    ``_metal_prepare_fused_v3_active`` and
    ``_metal_prepare_fused_v4_active`` may be True simultaneously when
    ``MLX_KDA_ENABLE_METAL_PREPARE=fused4``.
    """
    return _HAS_METAL_PREPARE and _metal_prepare_fused_v3_fn is not None


def _metal_prepare_fused_v4_active() -> bool:
    """True when the flat-ragged (v4) prepare kernel is selected.

    Functionally equivalent to v3 but reads chunks at arbitrary token
    offsets via a per-chunk int32 metadata table, allowing one Metal
    dispatch to span multiple unpadded packed sequences (PR M Option A).
    Activates a new branch in ``_run_packed`` that runs prepare once on
    the flat ``[T_total, H, D]`` buffer, then runs per-seq recurrence on
    flat-buffer slices.
    """
    return _HAS_METAL_PREPARE and _metal_prepare_fused_v4_fn is not None


# ---------------------------------------------------------------------------
# Small kernels
# ---------------------------------------------------------------------------

def _fp16_mm(a: mx.array, b: mx.array) -> mx.array:
    """fp16 matmul returning fp32 (matches reference._fp16_mm semantics)."""
    return mx.matmul(a.astype(mx.float16), b.astype(mx.float16)).astype(mx.float32)


def _pad_to_multiple(x: mx.array, multiple: int, axis: int = 0) -> tuple[mx.array, int]:
    """Pad ``x`` along ``axis`` with zeros so its length is a multiple of ``multiple``.

    Returns ``(padded, pad_len)``.
    """
    n = x.shape[axis]
    rem = n % multiple
    if rem == 0:
        return x, 0
    pad_len = multiple - rem
    pad_shape = list(x.shape)
    pad_shape[axis] = pad_len
    pad = mx.zeros(tuple(pad_shape), dtype=x.dtype)
    return mx.concatenate([x, pad], axis=axis), pad_len


def _pad_to_length(x: mx.array, target: int, axis: int = 0) -> mx.array:
    """Pad ``x`` along ``axis`` with zeros up to length ``target``.

    ``target`` must be >= ``x.shape[axis]``.
    """
    cur = x.shape[axis]
    if cur == target:
        return x
    pad_shape = list(x.shape)
    pad_shape[axis] = target - cur
    pad = mx.zeros(tuple(pad_shape), dtype=x.dtype)
    return mx.concatenate([x, pad], axis=axis)


# ---------------------------------------------------------------------------
# Pre-compute phase: everything that does not depend on recurrent state
#
# The helper handles both the single-sequence case ([padded_T, H, D] input,
# returning [n_chunks, H, CHUNK, ...]) and the packed-multi-sequence case
# ([N, padded_T, H, D] input, returning [N, n_chunks, H, CHUNK, ...]). MLX
# matmul already batches leading axes, so the math is identical either way;
# only the reshape + transpose signature differs.
# ---------------------------------------------------------------------------

def _valid_tokens_per_chunk_single(seq_len: int, n_chunks: int, chunk: int) -> list[int]:
    """Build the per-chunk valid-token count list for a single sequence.

    The last chunk has ``seq_len - (n_chunks - 1) * chunk`` valid tokens
    (or ``chunk`` if seq_len is an exact multiple); all earlier chunks
    are full. Total length is ``n_chunks``.
    """
    if n_chunks <= 0:
        return []
    if seq_len <= 0:
        return [0] * n_chunks
    full = [chunk] * (n_chunks - 1)
    last = seq_len - (n_chunks - 1) * chunk
    return full + [last]


def _valid_tokens_per_chunk_packed(
    seq_lens: list[int], n_chunks_max: int, chunk: int
) -> list[int]:
    """Build the per-chunk valid-token count list across packed sequences.

    Each sequence contributes ``n_chunks_max`` entries: the seq's
    full chunks, its partial last chunk (if any), and zeros for the
    chunks that exist only because of packing-pad to ``n_chunks_max``.
    Total length is ``len(seq_lens) * n_chunks_max``.
    """
    out: list[int] = []
    for sl in seq_lens:
        if sl <= 0:
            out.extend([0] * n_chunks_max)
            continue
        n_chunks_seq = (sl + chunk - 1) // chunk
        full = [chunk] * (n_chunks_seq - 1)
        last = sl - (n_chunks_seq - 1) * chunk
        seq_valid = full + [last]
        seq_valid.extend([0] * (n_chunks_max - n_chunks_seq))
        out.extend(seq_valid)
    return out


def _precompute_chunk_tensors(
    g_seq: mx.array,
    q_seq: mx.array,
    k_seq: mx.array,
    v_seq: mx.array,
    beta_seq: mx.array,
    *,
    H: int,
    D: int,
    chunk: int,
    scale_bf16_rt: mx.array,
    # Full-fusion (fused3 / fused4) mode params; required iff a fused
    # mode is active so the kernel can run section (a) inline:
    a_log_exp: "mx.array | None" = None,
    dt_bias: "mx.array | None" = None,
    lower_bound_log2e: "mx.array | None" = None,
    seq_len: "int | None" = None,
) -> dict[str, mx.array]:
    """Vectorize all chunk-local, state-independent computations for a sequence.

    Inputs are already padded to a multiple of ``chunk`` rows on axis 0.
    All returned tensors have leading axes ``[n_chunks, H, chunk, ...]``.

    Kernel dispatch (CHUNK=16 only; MLX graph fallback otherwise):

    * ``MLX_KDA_ENABLE_METAL_PREPARE=fused4`` / ``fused3``: full-fusion
      Metal kernel runs section (a) (q/k L2-norm + KDA gate activation)
      AND sections (b)-(g) in one dispatch. Inputs are RAW (post fp32
      cast or bf16 passthrough). See ``_metal_prepare`` for the
      kernel sources.
    * ``MLX_KDA_ENABLE_METAL_PREPARE=1`` (basic): sections (b)-(g)
      only. ``q_seq`` / ``k_seq`` are already post-section-(a).
    * ``MLX_KDA_ENABLE_METAL_PREPARE=0`` (default): pure MLX graph path.
    """
    padded_T = g_seq.shape[0]
    assert padded_T % chunk == 0
    n_chunks = padded_T // chunk

    # PR K-a: fused3 reads token-major inputs directly. Skip the
    # tile-major transpose+contiguous-copy that fused/fused2 force.
    if _metal_prepare_fused_v3_active():
        assert a_log_exp is not None and dt_bias is not None, (
            "fused3 mode requires a_log_exp and dt_bias"
        )
        assert lower_bound_log2e is not None and seq_len is not None, (
            "fused3 mode requires lower_bound_log2e and seq_len"
        )
        valid_list = _valid_tokens_per_chunk_single(seq_len, n_chunks, chunk)
        valid_arr = mx.array(valid_list, dtype=mx.int32)
        return _metal_prepare_fused_v3_fn(
            k_raw=k_seq, q_raw=q_seq, v=v_seq, g_raw=g_seq, beta=beta_seq,
            scale_bf16_rt=scale_bf16_rt,
            a_log_exp=a_log_exp, dt_bias=dt_bias,
            lower_bound_log2e=lower_bound_log2e,
            valid_tokens_per_chunk=valid_arr,
        )

    # Reshape [padded_T, H, *] → [n_chunks, chunk, H, *] → [n_chunks, H, chunk, *]
    def _to_chunks_hd(x: mx.array) -> mx.array:
        # x has shape [padded_T, H, D]
        return x.reshape(n_chunks, chunk, H, D).transpose(0, 2, 1, 3)

    gc = _to_chunks_hd(g_seq)          # [n_chunks, H, chunk, D]
    qc = _to_chunks_hd(q_seq)
    kc = _to_chunks_hd(k_seq)
    vc = _to_chunks_hd(v_seq)

    # beta: [padded_T, H] → [n_chunks, chunk, H] → [n_chunks, H, chunk]
    bc = beta_seq.reshape(n_chunks, chunk, H).transpose(0, 2, 1)

    if _metal_prepare_active():
        # Basic Metal kernel: one dispatch per (chunk, head) tile, output
        # dict has the same schema as ``_precompute_core``.
        # scale_bf16_rt is a 0-D scalar; the kernel wrapper accepts it as
        # either [1] or () shape and reshapes internally.
        return _metal_prepare_fn(
            k=kc, q=qc, v=vc, g=gc, beta=bc,
            scale_bf16_rt=scale_bf16_rt,
        )

    return _precompute_core(gc, qc, kc, vc, bc,
                            H=H, D=D, chunk=chunk,
                            scale_bf16_rt=scale_bf16_rt)


def _precompute_chunk_tensors_packed(
    g_pack: mx.array,
    q_pack: mx.array,
    k_pack: mx.array,
    v_pack: mx.array,
    beta_pack: mx.array,
    *,
    H: int,
    D: int,
    chunk: int,
    scale_bf16_rt: mx.array,
    # Fused2 (full-fusion) mode params; required iff fused2 is active:
    a_log_exp: "mx.array | None" = None,
    dt_bias: "mx.array | None" = None,
    lower_bound_log2e: "mx.array | None" = None,
    seq_lens: "list[int] | None" = None,
) -> dict[str, mx.array]:
    """Packed variant: pre-compute across N sequences in one dispatch chain.

    Inputs are ``[N, padded_T, H, D]`` (beta is ``[N, padded_T, H]``) where
    ``padded_T = max_chunks * chunk`` is the shared padded length. Returns
    tensors with leading axes ``[N, n_chunks, H, chunk, ...]``.

    Kernel dispatch mirrors the single-seq function: partial-fusion when
    selected (raw q/k plus pre-activated g), otherwise basic Metal kernel,
    otherwise pure MLX graph.
    """
    N = g_pack.shape[0]
    padded_T = g_pack.shape[1]
    assert padded_T % chunk == 0
    n_chunks = padded_T // chunk

    # PR K-a: fused3 reads token-major inputs directly. Flatten
    # [N, padded_T, H, *] -> [N*padded_T, H, *] (lazy reshape, no copy
    # because the first two axes are already row-contiguous in the
    # caller's pack stack), then call the v3 kernel.  Output is tile-major
    # [N*n_chunks, H, CHUNK, ...]; we reshape back to [N, n_chunks, ...].
    if _metal_prepare_fused_v3_active():
        assert a_log_exp is not None and dt_bias is not None, (
            "fused3 mode requires a_log_exp and dt_bias"
        )
        assert lower_bound_log2e is not None and seq_lens is not None, (
            "fused3 mode requires lower_bound_log2e and seq_lens"
        )
        T_total = N * padded_T

        g_tm = g_pack.reshape(T_total, H, D)
        q_tm = q_pack.reshape(T_total, H, D)
        k_tm = k_pack.reshape(T_total, H, D)
        v_tm = v_pack.reshape(T_total, H, D)
        b_tm = beta_pack.reshape(T_total, H)

        valid_list = _valid_tokens_per_chunk_packed(seq_lens, n_chunks, chunk)
        valid_arr = mx.array(valid_list, dtype=mx.int32)
        pre_flat = _metal_prepare_fused_v3_fn(
            k_raw=k_tm, q_raw=q_tm, v=v_tm, g_raw=g_tm, beta=b_tm,
            scale_bf16_rt=scale_bf16_rt,
            a_log_exp=a_log_exp, dt_bias=dt_bias,
            lower_bound_log2e=lower_bound_log2e,
            valid_tokens_per_chunk=valid_arr,
        )
        # pre_flat outputs are [N*n_chunks, H, CHUNK, ...]; downstream
        # consumers expect [N, n_chunks, H, CHUNK, ...].
        return {
            "k_decayed":   pre_flat["k_decayed"]  .reshape(N, n_chunks, H, chunk, D),
            "q_decayed":   pre_flat["q_decayed"]  .reshape(N, n_chunks, H, chunk, D),
            "k_restored":  pre_flat["k_restored"] .reshape(N, n_chunks, H, chunk, D),
            "Mqk":         pre_flat["Mqk"]        .reshape(N, n_chunks, H, chunk, chunk),
            "INV_bf":      pre_flat["INV_bf"]     .reshape(N, n_chunks, H, chunk, chunk),
            "vc":          pre_flat["vc"]         .reshape(N, n_chunks, H, chunk, D),
            "beta_bf16":   pre_flat["beta_bf16"]  .reshape(N, n_chunks, H, chunk, 1),
            "g_total_exp": pre_flat["g_total_exp"].reshape(N, n_chunks, H, D, 1),
        }

    # Reshape [N, padded_T, H, *] → [N, n_chunks, chunk, H, *]
    #                            → [N, n_chunks, H, chunk, *]
    def _to_chunks_hd(x: mx.array) -> mx.array:
        return x.reshape(N, n_chunks, chunk, H, D).transpose(0, 1, 3, 2, 4)

    gc = _to_chunks_hd(g_pack)          # [N, n_chunks, H, chunk, D]
    qc = _to_chunks_hd(q_pack)
    kc = _to_chunks_hd(k_pack)
    vc = _to_chunks_hd(v_pack)

    # beta: [N, padded_T, H] → [N, n_chunks, chunk, H] → [N, n_chunks, H, chunk]
    bc = beta_pack.reshape(N, n_chunks, chunk, H).transpose(0, 1, 3, 2)

    def _flatten_for_kernel():
        gc_flat = gc.reshape(N * n_chunks, H, chunk, D)
        qc_flat = qc.reshape(N * n_chunks, H, chunk, D)
        kc_flat = kc.reshape(N * n_chunks, H, chunk, D)
        vc_flat = vc.reshape(N * n_chunks, H, chunk, D)
        bc_flat = bc.reshape(N * n_chunks, H, chunk)
        return gc_flat, qc_flat, kc_flat, vc_flat, bc_flat

    def _reshape_back(pre_flat):
        return {
            "k_decayed":   pre_flat["k_decayed"]  .reshape(N, n_chunks, H, chunk, D),
            "q_decayed":   pre_flat["q_decayed"]  .reshape(N, n_chunks, H, chunk, D),
            "k_restored":  pre_flat["k_restored"] .reshape(N, n_chunks, H, chunk, D),
            "Mqk":         pre_flat["Mqk"]        .reshape(N, n_chunks, H, chunk, chunk),
            "INV_bf":      pre_flat["INV_bf"]     .reshape(N, n_chunks, H, chunk, chunk),
            "vc":          pre_flat["vc"]         .reshape(N, n_chunks, H, chunk, D),
            "beta_bf16":   pre_flat["beta_bf16"]  .reshape(N, n_chunks, H, chunk, 1),
            "g_total_exp": pre_flat["g_total_exp"].reshape(N, n_chunks, H, D, 1),
        }

    if _metal_prepare_active():
        gc_flat, qc_flat, kc_flat, vc_flat, bc_flat = _flatten_for_kernel()
        pre_flat = _metal_prepare_fn(
            k=kc_flat, q=qc_flat, v=vc_flat, g=gc_flat, beta=bc_flat,
            scale_bf16_rt=scale_bf16_rt,
        )
        return _reshape_back(pre_flat)

    return _precompute_core(gc, qc, kc, vc, bc,
                            H=H, D=D, chunk=chunk,
                            scale_bf16_rt=scale_bf16_rt)


def _precompute_core(
    gc: mx.array,
    qc: mx.array,
    kc: mx.array,
    vc: mx.array,
    bc: mx.array,
    *,
    H: int,
    D: int,
    chunk: int,
    scale_bf16_rt: mx.array,
) -> dict[str, mx.array]:
    """Shape-agnostic body of the pre-compute phase.

    Accepts tensors whose last three axes are ``[..., H, chunk, D]`` (or
    ``[..., H, chunk]`` for beta). Everything MLX does here batches leading
    axes, so the same body serves both single-sequence and packed inputs.
    ``chunk`` selects the nilpotent-closure length of the Neumann series:
    ``16`` → 3 factors, ``32`` → 4 factors (adds ``(I + L^16)``).
    """
    # Axis of length chunk is the second-to-last (``-2``) for the 4D-value
    # tensors. Cumsum along that axis is independent per [..., H, :, D] tile.
    g_cumsum = mx.cumsum(gc, axis=-2)                       # [..., H, chunk, D]
    # g_total is the last token's cumsum, with chunk axis kept as size 1.
    g_total = g_cumsum[..., -1:, :]                         # [..., H, 1, D]

    ex_pos = _q_bf16(_ex2_ftz(g_cumsum))
    ex_neg = _q_bf16(_ex2_ftz(-g_cumsum))
    ex_gtot = _q_bf16(_ex2_ftz(g_total))

    k_decayed = _q_bf16(kc * ex_pos)
    q_decayed = _q_bf16(_q_bf16(qc * ex_pos) * scale_bf16_rt)
    k_inv = _q_bf16(kc * ex_neg)
    k_restored = _q_bf16(k_inv * ex_gtot)

    # Swap the last two axes of k_inv: [..., H, chunk, D] → [..., H, D, chunk]
    k_inv_T = mx.swapaxes(k_inv, -1, -2)

    L = mx.matmul(k_decayed, k_inv_T)                       # [..., H, chunk, chunk]
    L = L.astype(mx.float16).astype(mx.float32)

    Mqk = _q_bf16(mx.matmul(q_decayed, k_inv_T))

    beta_act = mx.sigmoid(bc)                               # [..., H, chunk]
    beta_bf16 = _q_bf16(beta_act)[..., None]                # [..., H, chunk, 1]
    beta_fp16 = (beta_act.astype(mx.float16).astype(mx.float32))[..., None]

    # tril acts on last two axes; broadcast beta_fp16 column-wise.
    L = mx.tril(L, k=-1) * beta_fp16
    L = L.astype(mx.float16).astype(mx.float32)
    Mqk = _q_bf16(mx.tril(Mqk))

    # Neumann-series inverse in fp16 domain, batched over all leading axes.
    # Broadcast the identity across whatever leading dims L has.
    # Product = (I - L)·∏_{k=1..log2(chunk)-1} (I + L^(2^k)). For strictly
    # lower-triangular L (nilpotent with L^chunk = 0) this is exact.
    eye_chunk = mx.eye(chunk, dtype=mx.float32)
    INV = (eye_chunk - L).astype(mx.float16).astype(mx.float32)

    L2 = _fp16_mm(L, L)
    INV = INV + _fp16_mm(INV, L2)
    L4 = _fp16_mm(L2, L2)
    INV = INV + _fp16_mm(INV, L4)
    L8 = _fp16_mm(L4, L4)
    INV = INV + _fp16_mm(INV, L8)

    INV_bf = _q_bf16(INV)                                   # [..., H, chunk, chunk]

    # g_total_exp laid out for the state-update broadcast. For the single-
    # sequence case this is [n_chunks, H, D, 1]; for the packed case it is
    # [N, n_chunks, H, D, 1]. Either way we just squeeze the CHUNK axis and
    # swap it with D.
    # g_total has shape [..., H, 1, D]; swap last two → [..., H, D, 1].
    g_total_exp = mx.swapaxes(_ex2_ftz(g_total), -1, -2)

    mx.eval(
        k_decayed, q_decayed, k_restored,
        Mqk, INV_bf, beta_bf16,
        vc, g_total_exp,
    )

    return {
        "k_decayed": k_decayed,
        "q_decayed": q_decayed,
        "k_restored": k_restored,
        "Mqk": Mqk,
        "INV_bf": INV_bf,
        "vc": vc,
        "beta_bf16": beta_bf16,
        "g_total_exp": g_total_exp,
    }


# ---------------------------------------------------------------------------
# Single-sequence recurrence (fast path for N == 1)
# ---------------------------------------------------------------------------

def _run_single(
    g_seq: mx.array,
    q_seq: mx.array,
    k_seq: mx.array,
    v_seq: mx.array,
    beta_seq: mx.array,
    state_in: mx.array,
    *,
    seq_len: int,
    H: int,
    D: int,
    chunk: int,
    scale_bf16_rt: mx.array,
    a_log_exp: "mx.array | None" = None,
    dt_bias: "mx.array | None" = None,
    lower_bound_log2e: "mx.array | None" = None,
) -> tuple[mx.array, mx.array]:
    """Run one sequence of length ``seq_len`` and return ``(out_trimmed, state_out)``.

    ``*_seq`` tensors are padded to a multiple of ``chunk`` on axis 0.
    ``state_in`` is ``[H, D, D]`` fp32-valued.

    When the Metal cross-chunk kernel (Phase 3) is active, the entire
    per-chunk Python loop collapses into ONE Metal dispatch with state
    held in threadgroup memory across chunks.

    The ``a_log_exp`` / ``dt_bias`` / ``lower_bound_log2e`` triple is only
    consumed by the fused2 (full-fusion) Metal prepare kernel; other
    paths receive pre-activated ``g_seq`` from the caller.
    """
    pre = _precompute_chunk_tensors(
        g_seq=g_seq, q_seq=q_seq, k_seq=k_seq, v_seq=v_seq, beta_seq=beta_seq,
        H=H, D=D, chunk=chunk, scale_bf16_rt=scale_bf16_rt,
        a_log_exp=a_log_exp, dt_bias=dt_bias,
        lower_bound_log2e=lower_bound_log2e, seq_len=seq_len,
    )
    n_chunks = g_seq.shape[0] // chunk

    # -----------------------------------------------------------------
    # Phase 3 fast path: one Metal kernel per head, chunks inside shader.
    # -----------------------------------------------------------------
    if _metal_cross_chunk_active():
        seq_out, state_out = _metal_cross_chunk_fn(
            state_in,
            pre["k_decayed"], pre["q_decayed"], pre["k_restored"],
            pre["Mqk"], pre["INV_bf"], pre["vc"],
            pre["beta_bf16"], pre["g_total_exp"],
        )
        # Kernel writes directly to [padded_T, H, D] (PR H follow-on 3
        # Phase A) — no transpose+reshape needed.
        if seq_out.shape[0] > seq_len:
            seq_out = seq_out[:seq_len]
        return seq_out, state_out

    # -----------------------------------------------------------------
    # Fallback: per-chunk loop (mx.compile or Phase 2 Metal body).
    # -----------------------------------------------------------------
    state_slice = state_in                            # [H, D, D] fp32
    chunk_outs: list[mx.array] = []
    for chunk_idx in range(n_chunks):
        out_h, state_slice = _recurrence_body_single(
            state_slice,
            pre["k_decayed"][chunk_idx],       # [H, CHUNK, D]
            pre["q_decayed"][chunk_idx],
            pre["k_restored"][chunk_idx],
            pre["Mqk"][chunk_idx],             # [H, CHUNK, CHUNK]
            pre["INV_bf"][chunk_idx],
            pre["vc"][chunk_idx],              # [H, CHUNK, D]
            pre["beta_bf16"][chunk_idx],       # [H, CHUNK, 1]
            pre["g_total_exp"][chunk_idx],     # [H, D, 1]
        )
        chunk_outs.append(out_h.transpose(1, 0, 2))   # [CHUNK, H, D]

    seq_out = mx.concatenate(chunk_outs, axis=0)      # [padded_T, H, D]
    if seq_out.shape[0] > seq_len:
        seq_out = seq_out[:seq_len]
    return seq_out, state_slice


# ---------------------------------------------------------------------------
# Packed multi-sequence recurrence (Option A: mask-based)
# ---------------------------------------------------------------------------

def _run_packed(
    g_pack: mx.array,
    q_pack: mx.array,
    k_pack: mx.array,
    v_pack: mx.array,
    beta_pack: mx.array,
    state_in: mx.array,
    *,
    seq_lens: list[int],
    n_chunks_per_seq: list[int],
    H: int,
    D: int,
    chunk: int,
    scale_bf16_rt: mx.array,
    a_log_exp: "mx.array | None" = None,
    dt_bias: "mx.array | None" = None,
    lower_bound_log2e: "mx.array | None" = None,
) -> tuple[list[mx.array], mx.array]:
    """Run all N sequences in parallel using a packed pre-compute.

    Returns ``(per_seq_outputs, state_out)`` where ``per_seq_outputs`` is a
    Python list of ``[seq_lens[n], H, D]`` tensors and ``state_out`` is
    ``[N, H, D, D]`` fp32-valued.

    Correctness contract (Option A): for chunks past a sequence's own
    ``n_chunks_per_seq[n]``, the per-sequence state must remain at the value
    it held after chunk ``n_chunks_per_seq[n] - 1``. We enforce this via an
    ``mx.where`` on the state update, gated by a ``[N, 1, 1, 1]`` validity
    mask that becomes ``0`` once chunk_idx >= n_chunks_for_seq[n].
    """
    N = g_pack.shape[0]
    padded_T = g_pack.shape[1]
    assert padded_T % chunk == 0
    max_chunks = padded_T // chunk

    pre = _precompute_chunk_tensors_packed(
        g_pack=g_pack, q_pack=q_pack, k_pack=k_pack, v_pack=v_pack,
        beta_pack=beta_pack, H=H, D=D, chunk=chunk, scale_bf16_rt=scale_bf16_rt,
        a_log_exp=a_log_exp, dt_bias=dt_bias,
        lower_bound_log2e=lower_bound_log2e, seq_lens=seq_lens,
    )

    # ``n_chunks_arr[n]`` tells us how many chunks sequence n actually spans.
    # At chunk_idx, the mask is 1 where chunk_idx < n_chunks_arr[n], else 0.
    n_chunks_arr = mx.array(n_chunks_per_seq, dtype=mx.int32)  # [N]
    mx.eval(n_chunks_arr)

    # -----------------------------------------------------------------
    # Phase 4 fast path: one Metal kernel per forward, N sequences in
    # parallel via the grid z-axis. State freeze is handled inside the
    # shader by gating step 6 on c < n_chunks_per_seq[seq_id].
    # -----------------------------------------------------------------
    if _metal_cross_chunk_packed_active():
        chunk_stack, state_out = _metal_cross_chunk_packed_fn(
            state_in,
            pre["k_decayed"], pre["q_decayed"], pre["k_restored"],
            pre["Mqk"], pre["INV_bf"], pre["vc"],
            pre["beta_bf16"], pre["g_total_exp"],
            n_chunks_arr,
        )
        # Kernel writes directly to [N, padded_T, H, D] (PR H follow-on 3
        # Phase A) — no transpose+reshape needed.
        mx.eval(chunk_stack, state_out)
        per_seq_outs: list[mx.array] = []
        for n in range(N):
            per_seq_outs.append(chunk_stack[n, :seq_lens[n]])
        return per_seq_outs, state_out

    # -----------------------------------------------------------------
    # Fallback: per-chunk loop using ``_recurrence_body_packed``.
    # -----------------------------------------------------------------
    state = state_in                                            # [N, H, D, D] fp32
    # Pre-allocate per-chunk output buffer for the packed axis so the
    # per-sequence trim after the loop is a simple slice.
    per_chunk_outs: list[mx.array] = []                         # each [N, H, CHUNK, D]

    for chunk_idx in range(max_chunks):
        # Option A: gate the state update per-sequence. For sequences whose
        # chunk_idx is past their own n_chunks_per_seq[n], keep ``state``
        # unchanged so trailing padded chunks cannot corrupt it. The mask is
        # broadcast to [N, 1, 1, 1] so it applies to the whole [H, D, D]
        # per-sequence state block.
        active = (n_chunks_arr > chunk_idx).astype(mx.float32).reshape(N, 1, 1, 1)

        out_h, state = _recurrence_body_packed(
            state,
            pre["k_decayed"][:, chunk_idx],      # [N, H, CHUNK, D]
            pre["q_decayed"][:, chunk_idx],
            pre["k_restored"][:, chunk_idx],
            pre["Mqk"][:, chunk_idx],            # [N, H, CHUNK, CHUNK]
            pre["INV_bf"][:, chunk_idx],
            pre["vc"][:, chunk_idx],             # [N, H, CHUNK, D]
            pre["beta_bf16"][:, chunk_idx],      # [N, H, CHUNK, 1]
            pre["g_total_exp"][:, chunk_idx],    # [N, H, D, 1]
            active,
        )
        per_chunk_outs.append(out_h)                    # [N, H, CHUNK, D]

    # Stack chunks and slice per-sequence outputs.
    # chunk_stack: [max_chunks, N, H, CHUNK, D] → [N, max_chunks, H, CHUNK, D]
    chunk_stack = mx.stack(per_chunk_outs, axis=0).transpose(1, 0, 2, 3, 4)
    # Reshape the (max_chunks, CHUNK) pair back into a contiguous padded_T
    # axis and move H/D to trailing: [N, padded_T, H, D].
    chunk_stack = chunk_stack.transpose(0, 1, 3, 2, 4).reshape(N, padded_T, H, D)
    mx.eval(chunk_stack, state)

    per_seq_outs: list[mx.array] = []
    for n in range(N):
        per_seq_outs.append(chunk_stack[n, :seq_lens[n]])

    return per_seq_outs, state


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def fwd_optimized(
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

    chunk = CHUNK

    B, T_seq, H, D = q.shape
    T_total = B * T_seq

    # PR H follow-on 3 Phase B: under fused prepare modes the Metal kernel
    # reads bf16 inputs directly (Metal implicit bfloat→float on load) so
    # the entry-side fp32 promotion is unnecessary — saving ~268 MB of
    # bandwidth at bench scale (4 casts × 67 MB at H=64).
    _entry_skip_fp32 = _metal_prepare_fused_active()
    if _entry_skip_fp32:
        q = q.reshape(T_total, H, D)
        k = k.reshape(T_total, H, D)
        v = v.reshape(T_total, H, D)
        g = g.reshape(T_total, H, D)
        beta = beta.reshape(T_total, H)
    else:
        q = q.reshape(T_total, H, D).astype(mx.float32)
        k = k.reshape(T_total, H, D).astype(mx.float32)
        v = v.reshape(T_total, H, D).astype(mx.float32)
        g = g.reshape(T_total, H, D).astype(mx.float32)
        beta = beta.reshape(T_total, H).astype(mx.float32)

    if cu_seqlens is not None:
        cu = cu_seqlens
        # tolist() is one device→host sync; the prior loop did N+1
        # serial cu[i].item() syncs that gated every prepare-kernel launch.
        cu_list = cu.tolist()
    else:
        # Synthesize segmentation in pure Python — avoids the host→device
        # build + device→host readback round-trip for the trivial case.
        if B > 1:
            cu_list = list(range(0, B * T_seq + 1, T_seq))
        else:
            cu_list = [0, T_total]
        cu = mx.array(cu_list, dtype=mx.int64)
    N = len(cu_list) - 1

    want_final = final_state_like is not None
    state_fp32 = False
    if initial_state is not None and initial_state.dtype == mx.float32:
        state_fp32 = True
    if final_state_like is not None and final_state_like.dtype == mx.float32:
        state_fp32 = True

    # Section (a): L2-norm Q/K + KDA gate activation.
    # Three Metal prepare modes affect what runs here vs. inside the
    # kernel (see plan.md §8.4 for the historical fused/fused2 layers
    # that fused3/fused4 superseded):
    #   * basic    : section (a) runs entirely in MLX.
    #   * fused3   : kernel does both (token-major inputs, PR K-a).
    #                MLX only precomputes the small per-head broadcast
    #                tensors (a_log_exp, dt_bias_h, lower_bound_log2e)
    #                and threads them through to the kernel, which
    #                masks padded varlen positions inside.
    #   * fused4   : same as fused3 but flat-ragged token-major (PR M
    #                Option A). The N==1 single-seq path internally
    #                routes to v3; the N>1 packed path routes to v4
    #                with chunk_token_start metadata.
    _fused_active = _metal_prepare_fused_active()
    _fused_v3_active = _metal_prepare_fused_v3_active()
    _fused_v4_active = _metal_prepare_fused_v4_active()

    if not _fused_active:
        # fused3/fused4 want kernel-side L2-norm; basic and off-path
        # need the MLX-side L2-norm.
        q = _q_bf16(_l2_normalize(q))
        k = _q_bf16(_l2_normalize(k))

    if _fused_v3_active or _fused_v4_active:
        # Precompute the per-head and scalar broadcasts once; the kernel
        # consumes these directly. ``g`` stays raw (post fp32 cast).
        a_log_exp_h = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)        # [H]
        dt_bias_h = dt_bias.astype(mx.float32)                          # [H, D]
        lower_bound_log2e = mx.array([lower_bound * LOG2E], dtype=mx.float32)
        mx.eval(a_log_exp_h, dt_bias_h, lower_bound_log2e)
    else:
        a_log_exp_h = None
        dt_bias_h = None
        lower_bound_log2e = None
        g = g + dt_bias[None, :, :].astype(mx.float32)
        a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)
        a_log_exp = a_log_exp[None, :, None]
        g = (lower_bound * LOG2E) * mx.sigmoid(a_log_exp * g)

    if initial_state is not None:
        work_state = initial_state.astype(mx.bfloat16).astype(mx.float32)
    else:
        work_state = mx.zeros((N, H, D, D), dtype=mx.float32)
    mx.eval(work_state)

    scale_bf16_rt = _q_bf16(mx.array([scale], dtype=mx.float32))[0]

    # -------------------------------------------------------------------
    # Dispatch: N==1 → single-sequence fast path (zero packing overhead).
    #           N>1  → packed path iff _should_use_packed routes there;
    #                  otherwise per-sequence Python loop.
    #
    # Rationale and A/B data: benchmarks/section_timings_report.md §8.
    # Summary: the packed path wins for small-N / short-sequence / low-H
    # workloads where per-seq Python-launch overhead dominates, but loses
    # (up to 13× on bench_varlen_mixed_H96) at bench scale with uneven
    # seq_lens because 55% of its chunks are padded work.
    # -------------------------------------------------------------------
    seq_lens_list = [cu_list[i + 1] - cu_list[i] for i in range(N)]
    use_packed = (
        N > 1
        and not _DISABLE_PACKED
        and _should_use_packed(seq_lens_list, H=H, chunk=chunk)
    )

    # -------------------------------------------------------------------
    # PR M Option A: flat-ragged prepare. Runs ONE Metal prepare dispatch
    # over the FLAT [T_total, H, D] inputs spanning all packed sequences,
    # then runs per-seq simdgroup recurrence on slices of the prepare
    # output. Avoids:
    #   * the ~30-50 MB [N, max_padded_T, H, D] zero-pad allocation that
    #     packed-fused3 forces (the killer that reverted simplified PR M);
    #   * the per-seq Python launch overhead on N independent prepare
    #     dispatches that the per-seq path costs.
    #
    # Output concat is ~1.7-2.5 ms — bounded and small, unlike the
    # serialized-DRAM concat of simplified PR M.
    #
    # Only the prepare dispatch is consolidated; recurrence stays per-seq
    # via the existing simdgroup kernel. A future PR (Option B) could fuse
    # the recurrence too via a flat-ragged variant.
    # -------------------------------------------------------------------
    use_flat_ragged = (
        _metal_prepare_fused_v4_active()
       
        and N > 1
        and not _DISABLE_PACKED
        and _metal_cross_chunk_active()
    )

    if use_flat_ragged:
        # Build per-chunk metadata. Each sequence contributes
        # ceil(seq_len / chunk) chunks (NO inter-seq padding chunks).
        chunk_token_starts: list[int] = []
        valid_per_chunk: list[int] = []
        seq_chunk_offsets: list[int] = [0]
        for n in range(N):
            bos = cu_list[n]
            sl = seq_lens_list[n]
            n_chunks_n = (sl + chunk - 1) // chunk
            for c in range(n_chunks_n):
                chunk_token_starts.append(bos + c * chunk)
                # Tail chunk: min(chunk, sl - c*chunk).
                valid_per_chunk.append(min(chunk, sl - c * chunk))
            seq_chunk_offsets.append(seq_chunk_offsets[-1] + n_chunks_n)
        total_chunks = seq_chunk_offsets[-1]

        chunk_token_start_arr = mx.array(chunk_token_starts, dtype=mx.int32)
        valid_arr = mx.array(valid_per_chunk, dtype=mx.int32)
        seq_chunk_start_arr = mx.array(seq_chunk_offsets, dtype=mx.int32)
        mx.eval(chunk_token_start_arr, valid_arr, seq_chunk_start_arr)

        # One Metal dispatch over the flat [T_total, H, D] inputs.
        pre_flat = _metal_prepare_fused_v4_fn(
            k_raw=k, q_raw=q, v=v, g_raw=g, beta=beta,
            scale_bf16_rt=scale_bf16_rt,
            a_log_exp=a_log_exp_h, dt_bias=dt_bias_h,
            lower_bound_log2e=lower_bound_log2e,
            valid_tokens_per_chunk=valid_arr,
            chunk_token_start=chunk_token_start_arr,
        )

        # PR M Option B: one Metal recurrence dispatch across all N seqs,
        # direct-writing into the flat [T_total, H, D] output. Replaces
        # the per-seq Python loop AND the trailing mx.concatenate.
        if (
            _metal_cross_chunk_flat_ragged_active()
            and not _DISABLE_FLAT_RAGGED_RECURRENCE
        ):
            # ``cu`` is the canonical seq_token_start[N+1]; cast to int32
            # to match the kernel's binding type.
            seq_token_start_arr = cu.astype(mx.int32)
            mx.eval(seq_token_start_arr)

            out_flat, new_state = _metal_cross_chunk_flat_ragged_fn(
                work_state,
                pre_flat["k_decayed"],
                pre_flat["q_decayed"],
                pre_flat["k_restored"],
                pre_flat["Mqk"],
                pre_flat["INV_bf"],
                pre_flat["vc"],
                pre_flat["beta_bf16"],
                pre_flat["g_total_exp"],
                seq_chunk_start_arr,
                seq_token_start_arr,
                T_total,
            )
            work_state = new_state
        else:
            # Fallback: per-seq recurrence loop on flat-ragged prepare slices.
            # Kept on the cold path so PR M Option A keeps working when the
            # Option B kernel isn't loaded (e.g. on a fresh checkout running
            # an env where the import failed).
            seq_outs: list[mx.array] = []
            for n in range(N):
                cs, ce = seq_chunk_offsets[n], seq_chunk_offsets[n + 1]
                seq_len = seq_lens_list[n]
                if seq_len == 0:
                    continue
                seq_out, new_state_slice = _metal_cross_chunk_fn(
                    work_state[n],
                    pre_flat["k_decayed"]  [cs:ce],
                    pre_flat["q_decayed"]  [cs:ce],
                    pre_flat["k_restored"] [cs:ce],
                    pre_flat["Mqk"]        [cs:ce],
                    pre_flat["INV_bf"]     [cs:ce],
                    pre_flat["vc"]         [cs:ce],
                    pre_flat["beta_bf16"]  [cs:ce],
                    pre_flat["g_total_exp"][cs:ce],
                )
                if seq_out.shape[0] > seq_len:
                    seq_out = seq_out[:seq_len]
                work_state[n] = new_state_slice
                seq_outs.append(seq_out)
            out_flat = mx.concatenate(seq_outs, axis=0)
    elif not use_packed:
        # Per-sequence loop. Identical to the original unpacked optimized
        # path, just structured around _run_single.
        seq_outs: list[mx.array] = []
        for seq_idx in range(N):
            bos, eos = cu_list[seq_idx], cu_list[seq_idx + 1]
            seq_len = eos - bos
            g_seq, _ = _pad_to_multiple(g[bos:eos], chunk, axis=0)
            q_seq, _ = _pad_to_multiple(q[bos:eos], chunk, axis=0)
            k_seq, _ = _pad_to_multiple(k[bos:eos], chunk, axis=0)
            v_seq, _ = _pad_to_multiple(v[bos:eos], chunk, axis=0)
            beta_seq, _ = _pad_to_multiple(beta[bos:eos], chunk, axis=0)

            seq_out, new_state_slice = _run_single(
                g_seq, q_seq, k_seq, v_seq, beta_seq,
                state_in=work_state[seq_idx],
                seq_len=seq_len, H=H, D=D, chunk=chunk,
                scale_bf16_rt=scale_bf16_rt,
                a_log_exp=a_log_exp_h, dt_bias=dt_bias_h,
                lower_bound_log2e=lower_bound_log2e,
            )
            work_state[seq_idx] = new_state_slice
            seq_outs.append(seq_out)
        out_flat = seq_outs[0] if N == 1 else mx.concatenate(seq_outs, axis=0)
    else:
        # Pack: each sequence padded independently to ``max_chunks*chunk``.
        seq_lens = seq_lens_list
        n_chunks_per_seq = [(sl + chunk - 1) // chunk for sl in seq_lens]
        max_chunks = max(n_chunks_per_seq)
        max_padded = max_chunks * chunk

        def _gather_pack(arr: mx.array, last_shape: tuple) -> mx.array:
            # ``arr`` is [T_total, *last_shape]. Returns [N, max_padded, *last_shape].
            per_seq = []
            for n in range(N):
                bos, eos = cu_list[n], cu_list[n + 1]
                slice_ = arr[bos:eos]
                padded = _pad_to_length(slice_, max_padded, axis=0)
                per_seq.append(padded)
            return mx.stack(per_seq, axis=0)

        g_pack = _gather_pack(g, (H, D))
        q_pack = _gather_pack(q, (H, D))
        k_pack = _gather_pack(k, (H, D))
        v_pack = _gather_pack(v, (H, D))
        beta_pack = _gather_pack(beta, (H,))

        mx.eval(g_pack, q_pack, k_pack, v_pack, beta_pack)

        per_seq_outs, new_state = _run_packed(
            g_pack=g_pack, q_pack=q_pack, k_pack=k_pack, v_pack=v_pack,
            beta_pack=beta_pack,
            state_in=work_state,
            seq_lens=seq_lens,
            n_chunks_per_seq=n_chunks_per_seq,
            H=H, D=D, chunk=chunk, scale_bf16_rt=scale_bf16_rt,
            a_log_exp=a_log_exp_h, dt_bias=dt_bias_h,
            lower_bound_log2e=lower_bound_log2e,
        )
        work_state = new_state
        out_flat = mx.concatenate(per_seq_outs, axis=0)

    out = out_flat.reshape(B, T_seq, H, D)

    final: Optional[mx.array] = None
    if want_final:
        if state_fp32:
            final = work_state.astype(mx.float32)
        else:
            # ``work_state`` is already bf16-valued fp32 (recurrence kernel
            # quantizes via ``bfloat()`` on every state write); the prior
            # ``_q_bf16(...)`` round-trip was numerically a no-op so the
            # direct bf16 cast suffices (PR H follow-on 3 Phase A).
            final = work_state.astype(mx.bfloat16)

    return out, final


__all__ = ["fwd_optimized"]
