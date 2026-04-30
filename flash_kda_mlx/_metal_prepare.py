"""Metal kernel infrastructure for chunk pre-compute (PR H Branch A).

Scope (this file)
=================
Two Metal prepare kernels compute the state-independent chunk tensors
for one ``[H, CHUNK, D]`` chunk-tile per threadgroup:

* the basic kernel starts after section (a): L2-normalised Q/K and
  activated gate values are provided by the caller;
* the partial-fusion kernel additionally moves Q/K L2-norm + bf16
  rounding into Metal. Gate activation remains in MLX to preserve
  varlen padding semantics.

Both kernels implement sections (b)-(g) of
``flash_kda_mlx.optimized._precompute_core``:

* (b) ``g_cumsum`` along chunk axis + ``ex2`` of ``±g_cumsum`` and
  ``g_total``.
* (c) ``k_decayed``, ``q_decayed`` (scaled), ``k_inv``, ``k_restored``
  with bf16 round-trip boundaries matching ``_q_bf16(...)``.
* (d) ``L = k_decayed @ k_inv.T``, ``tril``, ``* beta_fp16``,
  ``fp16`` round-trip.
* (e) Neumann series inverse: ``INV = (I - L) · (I + L^2) · (I + L^4) ·
  (I + L^8)`` in fp16 domain (CHUNK=16 nilpotent closure).
* (f) ``Mqk = q_decayed @ k_inv.T``, ``tril``, bf16 round-trip.
* (g) ``beta_bf16`` materialisation, ``g_total_exp`` reshape.

Each kernel writes 7 output buffers per chunk-tile; ``v`` is a
passthrough on the Python side (kept here to mirror the current ``pre``
dict shape).

Why this is the right kernel shape
----------------------------------
``benchmarks/section_timings_report.md`` §9.5 estimates pre-compute
(a-g) at ~25-45% of E2E post-Phase4. Each of those sections currently
submits one or more separate MLX graph nodes; collapsing them into a
single Metal dispatch removes the per-section launch and the bf16
round-trip materialisations between sub-graphs.

Status
------
Arithmetic is implemented for both the basic and partial-fusion
variants. They are opt-in through ``MLX_KDA_ENABLE_METAL_PREPARE``
(``1``/``basic`` or ``fused``) because reduction-order differences
introduce small bf16-ULP shifts relative to the pure MLX graph.

Hardware gate
-------------
Same as ``_metal_recurrence``: M3+ only, gated via ``HAS_METAL_KERNEL``
imported from that module so we share the device probe.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# Reuse the M3+ probe from the recurrence module so we have one source
# of truth for hardware gating.
from flash_kda_mlx._metal_recurrence import HAS_METAL_KERNEL


# ---------------------------------------------------------------------------
# Kernel signature & dispatch geometry
# ---------------------------------------------------------------------------
#
# Grid contract:
#
#   grid        = (256, n_chunks, H)
#   threadgroup = (256, 1, 1)
#
# One threadgroup per (chunk, head) pair. 256 threads = 8 simdgroups,
# enough to:
#
#   * tile [CHUNK=16, D=128] inputs across simdgroups for the bf16-cast
#     pointwise sections (each thread handles 16*128/256 = 8 elements);
#   * do the [16,16] × [16,16] L / Mqk / Neumann matmuls inside one
#     simdgroup (4 mma ops each) without cross-simdgroup synchronisation;
#   * cooperatively reduce ``g_cumsum`` along the CHUNK axis (size 16) in
#     threadgroup memory.
#
# Threadgroup memory budget (M3 Max: 32 KB hard limit, per Phase 3a
# discovery):
#
#   * inputs we pin in TG memory (live across stages):
#       g_cumsum / ex_pos / ex_neg : 16*128*fp32 = 8 KB
#       k_tile, q_tile             : 16*128*fp32 = 8 KB each
#       L_tile, INV_tile, Mqk_tile : 16*16*fp32  = 1 KB each (~3 KB)
#     subtotal: ~27 KB, fits in 32 KB.
#   * v_tile, ex_gtot, beta_tile, g_total_exp_tile stream through
#     registers / are written direct-to-device (no TG residency).
#
# Inputs (device, contiguous):
#
#   k_in, q_in, v_in : [n_chunks, H, CHUNK, D]  fp32 (bf16-valued
#                                                   from L2-norm/_q_bf16)
#   g_in              : [n_chunks, H, CHUNK, D]  fp32 (post-activation
#                                                   gate; lower_bound
#                                                   already applied)
#   beta_in           : [n_chunks, H, CHUNK]      fp32 (pre-sigmoid)
#   scale_in          : [1] fp32 — scale_bf16_rt scalar, broadcast in
#                                  registers
#
# Outputs (device, contiguous):
#
#   k_decayed_out, q_decayed_out,
#   k_restored_out  : [n_chunks, H, CHUNK, D]   fp32 bf16-valued
#   Mqk_out          : [n_chunks, H, CHUNK, CHUNK] fp32 bf16-valued
#   INV_bf_out       : [n_chunks, H, CHUNK, CHUNK] fp32 bf16-valued
#   beta_bf16_out    : [n_chunks, H, CHUNK, 1]    fp32 bf16-valued
#   g_total_exp_out  : [n_chunks, H, D, 1]        fp32 (no bf16 cast —
#                                                    matches _ex2_ftz path
#                                                    in _precompute_core)
#
# (``v`` is a passthrough on the Python wrapper; the kernel does not
# touch it.)


_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// _q_bf16 in Metal: round-trip through bfloat (round-to-nearest-even),
// matches flash_kda_mlx.reference._q_bf16 (x.astype(bfloat16).astype(float32)).
inline float bf16_round(float x) {
    return float(bfloat(x));
}

// _ex2_ftz: 2^x in fp32, with subnormal flushing.
// MLX path: where(|y| < 2^-126, 0, y) where y = 2**x.
// 2^-126 = 1.1754943508222875e-38 (smallest positive normal fp32).
inline float ex2_ftz(float x) {
    float y = exp2(x);
    return (fabs(y) < 1.1754943508222875e-38f) ? 0.0f : y;
}

// _ex2_pos_safe: clamp input to 126.0 BEFORE exp2 to prevent fp32 overflow,
// then ftz. Matches optimized._ex2_pos_safe semantics. At CHUNK=16 with
// lower_bound=-5 this clamp is a no-op (|-g_cumsum| <= ~115), but kept
// to mirror Python bit-for-bit.
inline float ex2_pos_safe(float x) {
    float clamped = fmin(x, 126.0f);
    float y = exp2(clamped);
    return (fabs(y) < 1.1754943508222875e-38f) ? 0.0f : y;
}

// fp16 round-trip for the _fp16_mm operand-cast pattern.
inline float fp16_round(float x) {
    return float(half(x));
}
"""


# Phase 2 fused chunk pre-compute kernel. Each threadgroup owns one
# (chunk, head) pair and produces all 7 output buffers.
#
# Grid:        (256, n_chunks, H)
# Threadgroup: (256, 1, 1) = 8 simdgroups
#
# TG memory budget (M3 Max 32 KB hard limit):
#
#   g_cumsum_sm [CHUNK*D]  = 8 KB  (initially g, becomes cumsum, then ex_pos)
#   ex_neg_sm   [CHUNK*D]  = 8 KB
#   k_inv_sm    [CHUNK*D]  = 8 KB  (held for L and Mqk simdgroup matmuls)
#   L_sm        [CHUNK*CHUNK] = 1 KB
#   INV_sm      [CHUNK*CHUNK] = 1 KB
#   Lk_sm       [CHUNK*CHUNK] = 1 KB  (rotates L^2, L^4, L^8)
#   Tmp_sm      [CHUNK*CHUNK] = 1 KB  (matmul scratch)
#   Total                    = 28 KB
#
# Mapping of arithmetic to source-of-truth Python steps (see
# _precompute_core in flash_kda_mlx/optimized.py and the spec audit appended
# to the PR commit): every bf16/fp16 round-trip is at the same point
# in the expression tree as the Python path. Bit-exact at rtol=atol=1e-5
# under the Phase 3b convention.

_PREPARE_SOURCE_PHASE2 = """
    // Grid contract:
    //   grid        = (256, n_chunks, H)
    //   threadgroup = (256, 1, 1)

    const uint chunk_id = threadgroup_position_in_grid.y;
    const uint head_id  = threadgroup_position_in_grid.z;
    const uint tid      = thread_position_in_threadgroup.x;
    const uint tpg      = 256;
    const uint simd_id  = simdgroup_index_in_threadgroup;

    constexpr uint D_TILES   = uint(D) / 8;       // 16 for D=128
    constexpr uint K_D_TILES = uint(D) / 8;       // 16 K-tiles for L = k_dec @ k_inv.T
    constexpr uint C_TILES   = uint(CHUNK) / 8;   // 2 for CHUNK=16
    constexpr uint K_C_TILES = uint(CHUNK) / 8;   // 2 K-tiles for L^k = L^a @ L^a

    // Output offset bases (row-major [n_chunks, H, ...]).
    const uint kqv_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(D);
    const uint mqk_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(CHUNK);
    const uint beta_off = (chunk_id * uint(H) + head_id) * uint(CHUNK);
    const uint gtot_off = (chunk_id * uint(H) + head_id) * uint(D);

    threadgroup float g_cumsum_sm[uint(CHUNK) * uint(D)];
    threadgroup float ex_neg_sm  [uint(CHUNK) * uint(D)];
    threadgroup float k_inv_sm   [uint(CHUNK) * uint(D)];
    threadgroup float L_sm       [uint(CHUNK) * uint(CHUNK)];
    threadgroup float INV_sm     [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Lk_sm      [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Tmp_sm     [uint(CHUNK) * uint(CHUNK)];

    // ================================================================
    // STAGE 1: load g into g_cumsum_sm (will be overwritten in stages 2-4).
    // ================================================================
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        g_cumsum_sm[i] = g_in[kqv_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // STAGE 2: cumsum along chunk axis, in place in g_cumsum_sm.
    // (Spec step 1: g_cumsum = cumsum(gc, axis=-2))
    //
    // Threads tid in [0, D) each scan all CHUNK rows for one d-slot.
    // Threads tid >= D idle this stage.
    // ================================================================
    if (tid < uint(D)) {
        float acc = g_cumsum_sm[tid];   // c=0 stays as g[0, d]
        for (uint c = 1; c < uint(CHUNK); ++c) {
            acc += g_cumsum_sm[c * uint(D) + tid];
            g_cumsum_sm[c * uint(D) + tid] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // STAGE 3: g_total_exp output. Spec step 36:
    //   g_total_exp = swapaxes(_ex2_ftz(g_total), -1, -2)  (NO bf16 cast)
    // For this (chunk, head) the output slice is [D, 1] contiguous; we
    // write D values at gtot_off + d.
    // ================================================================
    if (tid < uint(D)) {
        float gt = g_cumsum_sm[(uint(CHUNK) - 1) * uint(D) + tid];
        g_total_exp_out[gtot_off + tid] = ex2_ftz(gt);
    }
    // No barrier needed — the next stage reads from g_cumsum_sm only.

    // ================================================================
    // STAGE 4: ex_neg, then OVERWRITE g_cumsum_sm with ex_pos.
    // (Spec steps 3-4)
    //   ex_neg = bf16(ex2_pos_safe(-g_cumsum))
    //   ex_pos = bf16(ex2_ftz(g_cumsum))
    // Order matters: ex_neg reads g_cumsum BEFORE we overwrite with ex_pos.
    // ================================================================
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        ex_neg_sm[i] = bf16_round(ex2_pos_safe(-cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        g_cumsum_sm[i] = bf16_round(ex2_ftz(cs));   // now holds ex_pos
    }
    // mem_device flag here is REQUIRED so stage 5's reads of
    // g_total_exp_out (written in stage 3 by tid<D, but read in stage 5
    // by ALL threads — tid=128+ reads what tid=0+ wrote) see the prior
    // device-side writes. mem_threadgroup alone does NOT order device
    // memory accesses across threads in the TG.
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);
    // Alias: g_cumsum_sm now == ex_pos.

    // ================================================================
    // STAGE 5: k_decayed, q_decayed (DOUBLE bf16), k_inv (TG), beta_bf16,
    //          k_restored. (Spec steps 6-10, 17)
    //
    // Reads from g_cumsum_sm (ex_pos) and ex_neg_sm. Writes outputs to
    // device and stages k_inv in TG for later L/Mqk matmuls.
    // ex_gtot[d] = bf16(ex2_ftz(g_total[d])) — we recompute by applying
    // bf16_round to the just-written raw g_total_exp_out value, avoiding
    // a second exp2 call.
    // ================================================================
    const float scale = scale_in[0];

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        const uint c = i / uint(D);
        const uint d = i % uint(D);

        const float k = k_in[kqv_off + i];
        const float q = q_in[kqv_off + i];
        const float ep = g_cumsum_sm[i];
        const float en = ex_neg_sm[i];

        // k_decayed = bf16(k * ep)
        k_decayed_out[kqv_off + i] = bf16_round(k * ep);

        // q_decayed = bf16(bf16(q * ep) * scale)  -- DOUBLE bf16
        const float q_int = bf16_round(q * ep);
        q_decayed_out[kqv_off + i] = bf16_round(q_int * scale);

        // k_inv = bf16(k * en) -- staged in TG for L and Mqk matmuls
        const float ki = bf16_round(k * en);
        k_inv_sm[i] = ki;

        // k_restored = bf16(k_inv * ex_gtot)
        // ex_gtot[d] derived from the raw ex2_ftz(g_total[d]) we wrote.
        const float gtot_raw = g_total_exp_out[gtot_off + d];
        const float ex_gtot_d = bf16_round(gtot_raw);
        k_restored_out[kqv_off + i] = bf16_round(ki * ex_gtot_d);

        // beta_bf16 output: only c=0 row writes the per-chunk beta vector.
        // beta_bf16 = bf16(sigmoid(beta_in)). We compute sigmoid here so
        // beta_act is derivable for the L stage's beta_fp16 in stage 7.
        if (c == 0 && d == 0) {
            // (only one thread per CHUNK row writes; CHUNK threads total)
            // No-op here; handled by per-row block below.
        }
    }
    // Write beta_bf16 separately — one thread per CHUNK row, no race.
    if (tid < uint(CHUNK)) {
        const float b = beta_in[beta_off + tid];
        const float b_act = 1.0f / (1.0f + exp(-b));
        beta_bf16_out[beta_off + tid] = bf16_round(b_act);
    }
    // Force device writes for k_decayed before stage 6 reads them via
    // simdgroup_load. Also covers k_inv_sm TG writes.
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // ================================================================
    // STAGE 6: L = k_decayed @ k_inv.T  -- simdgroup_matrix matmul.
    // (Spec step 12)
    //
    // L is [CHUNK, CHUNK] = 4 output 8x8 tiles. Use simd_id < 4
    // (one simdgroup per output tile). K-axis = D = 128 -> 16 K-tiles.
    // ================================================================
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;

        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            // A = k_decayed[tile_r*8:(tile_r+1)*8, k_tile*8:(k_tile+1)*8]
            const device float* A_src =
                k_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));

            // B = k_inv.T[k_tile*8:(k_tile+1)*8, tile_c*8:(tile_c+1)*8]
            //   = k_inv[tile_c*8:(tile_c+1)*8, k_tile*8:(k_tile+1)*8].T
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }

        threadgroup float* C_dst =
            &L_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // STAGE 7: L = fp16-round(L); tril(L, k=-1); L *= beta_fp16;
    //          L = fp16-round(L). (Spec steps 13, 16-20)
    //
    // [CHUNK, CHUNK] = 256 elements, 256 threads, 1 element each.
    // ================================================================
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);

        // Step 13: L → fp16 → fp32
        float l = fp16_round(L_sm[tid]);

        // Steps 16-19: tril(k=-1) and beta scale.
        // beta_fp16[i] = fp16(sigmoid(beta_in[i]))
        // (recompute here to avoid an extra TG slot for beta_act)
        if (j >= i) {
            l = 0.0f;
        }
        const float b = beta_in[beta_off + i];
        const float b_act = 1.0f / (1.0f + exp(-b));
        const float beta_fp16 = fp16_round(b_act);
        l = l * beta_fp16;

        // Step 20: L → fp16 → fp32 (final fp16-quantised L)
        L_sm[tid] = fp16_round(l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // STAGE 8: Neumann inverse.
    //   INV = fp16(I - L)
    //   L2 = fp16_mm(L, L);   INV += fp16_mm(INV, L2)
    //   L4 = fp16_mm(L2, L2); INV += fp16_mm(INV, L4)
    //   L8 = fp16_mm(L4, L4); INV += fp16_mm(INV, L8)
    //   INV_bf = bf16(INV)
    // (Spec steps 24-35)
    //
    // _fp16_mm: a→fp16, b→fp16, matmul, accum→fp32 (no fp32→fp16 on
    // the output). All matmuls are [CHUNK, CHUNK] @ [CHUNK, CHUNK];
    // K=CHUNK=16 fits in a per-thread scalar inner product with no
    // simdgroup matmul (matches Phase 3b's INV_bf @ vdiff pattern).
    // ================================================================

    // INV = fp16(I - L)
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        const float l = L_sm[tid];
        const float ident = (i == j) ? 1.0f : 0.0f;
        INV_sm[tid] = fp16_round(ident - l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lk_sm = L2 = fp16_mm(L, L). Operands fp16-cast on read.
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(L_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(L_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Lk_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tmp_sm = fp16_mm(INV, L2)
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // INV += Tmp
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tmp_sm = L4 = fp16_mm(L2, L2)
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Promote Tmp_sm (= L4) into Lk_sm so the next L^k matmul can read.
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tmp_sm = fp16_mm(INV, L4)
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // INV += Tmp
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tmp_sm = L8 = fp16_mm(L4, L4)
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];   // Lk_sm now holds L8
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tmp_sm = fp16_mm(INV, L8)
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // INV += Tmp; INV_bf = bf16(INV) -> device output
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
        INV_bf_out[mqk_off + tid] = bf16_round(INV_sm[tid]);
    }
    // No barrier — INV_bf is the final write for this output.

    // ================================================================
    // STAGE 9: Mqk = bf16(tril(bf16(q_decayed @ k_inv.T))).
    // (Spec steps 14-15, 21-22)
    //
    // Same simdgroup pattern as L. q_decayed is in device memory
    // (written in stage 5; stage 5's mem_device barrier covers visibility),
    // k_inv stays in TG (k_inv_sm).
    // ================================================================
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;

        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                q_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));

            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }

        threadgroup float* C_dst =
            &Tmp_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // bf16(tril(bf16(Mqk))) -> device. Steps 15, 21, 22 fuse: the second
    // bf16 cast is idempotent on a bf16-valued input but we replicate
    // the Python sequence exactly: mqk = bf16(matmul); mqk = bf16(tril(mqk))
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);

        // Step 15: bf16(matmul result)
        float m = bf16_round(Tmp_sm[tid]);
        // Step 21: tril (zero strict upper triangle)
        if (j > i) {
            m = 0.0f;
        }
        // Step 22: bf16 again (idempotent for bf16-valued input)
        Mqk_out[mqk_off + tid] = bf16_round(m);
    }
"""


# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _build_prepare_kernel(H: int, D: int, CHUNK: int):
    """Build (or fetch from MLX's cache) the prepare kernel.

    Template parameters bake (H, D, CHUNK) into the compiled shader.
    Same caching pattern as _metal_recurrence.
    """
    del H, D, CHUNK  # baked via template at call time; source is generic
    return mx.fast.metal_kernel(
        name="flash_kda_prepare_chunk",
        input_names=["k_in", "q_in", "v_in", "g_in", "beta_in", "scale_in"],
        output_names=[
            "k_decayed_out",
            "q_decayed_out",
            "k_restored_out",
            "Mqk_out",
            "INV_bf_out",
            "beta_bf16_out",
            "g_total_exp_out",
        ],
        header=_HEADER,
        source=_PREPARE_SOURCE_PHASE2,
        ensure_row_contiguous=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def metal_prepare_chunk(
    k: mx.array,
    q: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale_bf16_rt: mx.array,
) -> dict[str, mx.array]:
    """Run the basic chunk-pre-compute kernel.

    This is the basic prepare mode: the caller has already run section
    (a), and this kernel computes sections (b)-(g).

    Args:
        k, q, v, g: ``[n_chunks, H, CHUNK, D]`` fp32. ``q`` and ``k``
            are expected to already be L2-normalised + bf16-rounded by
            the caller; ``g`` is the post-activation KDA gate (after
            ``A_log`` / ``dt_bias`` / ``lower_bound``); ``v`` is a
            passthrough (the kernel does not touch it but is part of
            the pre-compute dict schema).
        beta: ``[n_chunks, H, CHUNK]`` fp32 (pre-sigmoid).
        scale_bf16_rt: ``[1]`` fp32 — the bf16-rounded scale scalar.

    Returns:
        Dict mirroring ``_precompute_core``'s return dict:
          ``k_decayed``, ``q_decayed``, ``k_restored``: [n_chunks, H, CHUNK, D]
          ``Mqk``, ``INV_bf``: [n_chunks, H, CHUNK, CHUNK]
          ``vc`` (passthrough of input ``v``): [n_chunks, H, CHUNK, D]
          ``beta_bf16``: [n_chunks, H, CHUNK, 1]
          ``g_total_exp``: [n_chunks, H, D, 1]

        All outputs are fp32 (matching ``_precompute_core``); bf16
        round-trips happen inside the kernel and the outputs hold
        bf16-representable fp32 values where the spec calls for it.
    """
    assert HAS_METAL_KERNEL, (
        "metal_prepare_chunk called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    assert k.dtype == mx.float32 and q.dtype == mx.float32, (
        f"k/q must be fp32 (bf16-valued); got k={k.dtype}, q={q.dtype}"
    )
    assert k.ndim == 4 and q.ndim == 4 and v.ndim == 4 and g.ndim == 4, (
        f"k/q/v/g must be 4-D; got "
        f"k={k.shape} q={q.shape} v={v.shape} g={g.shape}"
    )
    n_chunks, H, CHUNK, D = k.shape
    assert q.shape == (n_chunks, H, CHUNK, D)
    assert v.shape == (n_chunks, H, CHUNK, D)
    assert g.shape == (n_chunks, H, CHUNK, D)
    assert beta.shape == (n_chunks, H, CHUNK), (
        f"beta must be [n_chunks, H, CHUNK]; got {beta.shape}"
    )
    assert scale_bf16_rt.shape == (1,) or scale_bf16_rt.shape == (), (
        f"scale_bf16_rt must be scalar or [1]; got {scale_bf16_rt.shape}"
    )
    assert CHUNK == 16, (
        f"Metal prepare supports CHUNK=16 only (Neumann nilpotent closure); "
        f"got CHUNK={CHUNK}"
    )

    # Materialise scale as a [1] fp32 array so the kernel signature is
    # uniform across all six inputs.
    if scale_bf16_rt.shape == ():
        scale_in = mx.reshape(scale_bf16_rt, (1,))
    else:
        scale_in = scale_bf16_rt

    kernel = _build_prepare_kernel(H=H, D=D, CHUNK=CHUNK)

    outputs = kernel(
        inputs=[k, q, v, g, beta, scale_in],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(256, n_chunks, H),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (n_chunks, H, CHUNK, D),     # k_decayed
            (n_chunks, H, CHUNK, D),     # q_decayed
            (n_chunks, H, CHUNK, D),     # k_restored
            (n_chunks, H, CHUNK, CHUNK), # Mqk
            (n_chunks, H, CHUNK, CHUNK), # INV_bf
            (n_chunks, H, CHUNK, 1),     # beta_bf16
            (n_chunks, H, D, 1),         # g_total_exp
        ],
        output_dtypes=[mx.float32] * 7,
    )

    return {
        "k_decayed":   outputs[0],
        "q_decayed":   outputs[1],
        "k_restored":  outputs[2],
        "Mqk":         outputs[3],
        "INV_bf":      outputs[4],
        "vc":          v,                # passthrough
        "beta_bf16":   outputs[5],
        "g_total_exp": outputs[6],
    }


# ---------------------------------------------------------------------------
# FUSED variant — Q/K L2-norm from section (a) + sections (b)-(g).
# ---------------------------------------------------------------------------
#
# The "basic" kernel above assumes the caller has already run section (a)
# — L2-norm Q/K + KDA gate activation — before dispatching. The fused
# kernel accepts raw q/k and does the L2-norm + bf16-round inline. Gate
# activation still happens in MLX, which preserves the packed varlen
# zero-padding semantics without carrying valid-token masks into Metal.
# This saves the reduction-heavy MLX graph work from section (a).
#
# Additional stages vs the basic kernel:
#
#   * stage 0a: L2-norm of k per (c, h) row using simd_sum reduction
#               across D, then bf16_round. Written to ``k_decayed_out``
#               as scratch (same buffer is overwritten with the final
#               ``k * ex_pos`` in stage 5).
#   * stage 0b: same L2-norm pipeline for q, writing scratch to
#               ``q_decayed_out``.
#
# Stages 2-9 are bit-identical to the basic kernel except stage 5 reads
# the scratch L2-normed values from the output buffers instead of from
# k_in / q_in. The mem_device barrier after stage 0 ensures those writes
# are visible to stage 5 (same hazard pattern as the basic kernel's
# g_total_exp_out stage 3 → stage 5 dependency).
#
# TG memory budget is the same as basic.


_PREPARE_SOURCE_FUSED = """
    // Grid contract:
    //   grid        = (256, n_chunks, H)
    //   threadgroup = (256, 1, 1)
    //
    // PARTIAL-FUSION VARIANT (PR H follow-on 1):
    //   * Does L2-norm + bf16 of q/k inside the kernel (removes the
    //     MLX reduction + bf16 dispatches from section (a)).
    //   * Takes PRE-ACTIVATED g (gate activation stays in MLX because
    //     the varlen packed path zero-pads g AFTER activation, and
    //     emulating that inside the kernel would require per-chunk
    //     valid-token masks).

    const uint chunk_id = threadgroup_position_in_grid.y;
    const uint head_id  = threadgroup_position_in_grid.z;
    const uint tid      = thread_position_in_threadgroup.x;
    const uint tpg      = 256;
    const uint simd_id  = simdgroup_index_in_threadgroup;
    const uint simd_lane = thread_index_in_simdgroup;

    constexpr uint D_TILES   = uint(D) / 8;
    constexpr uint K_D_TILES = uint(D) / 8;
    constexpr uint C_TILES   = uint(CHUNK) / 8;

    const uint kqv_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(D);
    const uint mqk_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(CHUNK);
    const uint beta_off = (chunk_id * uint(H) + head_id) * uint(CHUNK);
    const uint gtot_off = (chunk_id * uint(H) + head_id) * uint(D);

    threadgroup float g_cumsum_sm[uint(CHUNK) * uint(D)];
    threadgroup float ex_neg_sm  [uint(CHUNK) * uint(D)];
    threadgroup float k_inv_sm   [uint(CHUNK) * uint(D)];
    threadgroup float L_sm       [uint(CHUNK) * uint(CHUNK)];
    threadgroup float INV_sm     [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Lk_sm      [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Tmp_sm     [uint(CHUNK) * uint(CHUNK)];

    // ================================================================
    // STAGE 0a: load pre-activated g into g_cumsum_sm
    // (same as basic kernel stage 1 — no activation in the kernel).
    // ================================================================
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        g_cumsum_sm[i] = g_in[kqv_off + i];
    }

    // ================================================================
    // STAGE 0c: L2-norm of k per (c, h) row via simd_sum reduction.
    //   Each simdgroup (32 threads) owns CHUNK / (tpg/32) = 16/8 = 2
    //   rows. For each row, 32 lanes handle D/32 = 4 elements each,
    //   compute partial sum-of-squares, reduce with simd_sum, then
    //   normalise and bf16-round.
    //   Output: scratch in k_decayed_out (overwritten in stage 5).
    // ================================================================
    {
        constexpr uint ROWS_PER_SIMD = uint(CHUNK) * 32 / 256;   // = 2 at CHUNK=16
        constexpr uint ELEMS_PER_LANE = uint(D) / 32;            // = 4 at D=128
        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {
            const uint c = simd_id * ROWS_PER_SIMD + rb;
            if (c >= uint(CHUNK)) continue;
            float partial = 0.0f;
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = k_in[kqv_off + c * uint(D) + d];
                partial += val * val;
            }
            const float row_sum = simd_sum(partial);
            const float rsqrt_val = rsqrt(row_sum + 1e-6f);
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = k_in[kqv_off + c * uint(D) + d];
                k_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);
            }
        }
    }

    // ================================================================
    // STAGE 0d: same L2-norm pipeline for q → q_decayed_out (scratch).
    // ================================================================
    {
        constexpr uint ROWS_PER_SIMD = uint(CHUNK) * 32 / 256;
        constexpr uint ELEMS_PER_LANE = uint(D) / 32;
        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {
            const uint c = simd_id * ROWS_PER_SIMD + rb;
            if (c >= uint(CHUNK)) continue;
            float partial = 0.0f;
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = q_in[kqv_off + c * uint(D) + d];
                partial += val * val;
            }
            const float row_sum = simd_sum(partial);
            const float rsqrt_val = rsqrt(row_sum + 1e-6f);
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = q_in[kqv_off + c * uint(D) + d];
                q_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);
            }
        }
    }

    // Device barrier: stage 5 reads scratch values from k_decayed_out /
    // q_decayed_out across TG threads (tid in stage 5 does not match
    // the simd-lane-based indexing of stage 0c/0d).
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // ================================================================
    // STAGE 2: cumsum along chunk axis, in place in g_cumsum_sm.
    //   (g_cumsum_sm already holds activated g from stage 0b; no
    //   stage 1 load needed.)
    // ================================================================
    if (tid < uint(D)) {
        float acc = g_cumsum_sm[tid];
        for (uint c = 1; c < uint(CHUNK); ++c) {
            acc += g_cumsum_sm[c * uint(D) + tid];
            g_cumsum_sm[c * uint(D) + tid] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 3: g_total_exp output. (Spec step 36)
    if (tid < uint(D)) {
        float gt = g_cumsum_sm[(uint(CHUNK) - 1) * uint(D) + tid];
        g_total_exp_out[gtot_off + tid] = ex2_ftz(gt);
    }

    // STAGE 4: ex_neg, then overwrite g_cumsum_sm with ex_pos.
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        ex_neg_sm[i] = bf16_round(ex2_pos_safe(-cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        g_cumsum_sm[i] = bf16_round(ex2_ftz(cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // ================================================================
    // STAGE 5 (modified): read L2-normed k, q from scratch output
    // buffers (written in stage 0c/0d); compute final k_decayed,
    // q_decayed, k_inv (TG), k_restored. (Spec steps 6-10, 17)
    // ================================================================
    const float scale = scale_in[0];

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        const uint d = i % uint(D);

        const float k = k_decayed_out[kqv_off + i];    // L2-normed, bf16
        const float q = q_decayed_out[kqv_off + i];    // L2-normed, bf16
        const float ep = g_cumsum_sm[i];
        const float en = ex_neg_sm[i];

        // k_decayed = bf16(k * ep) — overwrites scratch
        k_decayed_out[kqv_off + i] = bf16_round(k * ep);

        // q_decayed = bf16(bf16(q * ep) * scale) — DOUBLE bf16
        const float q_int = bf16_round(q * ep);
        q_decayed_out[kqv_off + i] = bf16_round(q_int * scale);

        // k_inv = bf16(k * en) -- staged in TG
        const float ki = bf16_round(k * en);
        k_inv_sm[i] = ki;

        // k_restored = bf16(k_inv * bf16(ex2_ftz(g_total[d])))
        const float gtot_raw = g_total_exp_out[gtot_off + d];
        const float ex_gtot_d = bf16_round(gtot_raw);
        k_restored_out[kqv_off + i] = bf16_round(ki * ex_gtot_d);
    }

    if (tid < uint(CHUNK)) {
        const float b = beta_in[beta_off + tid];
        const float b_act = 1.0f / (1.0f + exp(-b));
        beta_bf16_out[beta_off + tid] = bf16_round(b_act);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // ================================================================
    // STAGES 6-9: unchanged from the basic kernel.
    // ================================================================

    // STAGE 6: L = k_decayed @ k_inv.T
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                k_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }
        threadgroup float* C_dst =
            &L_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 7: L = fp16-round(L); tril(L, k=-1); L *= beta_fp16; fp16-round.
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float l = fp16_round(L_sm[tid]);
        if (j >= i) l = 0.0f;
        const float b = beta_in[beta_off + i];
        const float b_act = 1.0f / (1.0f + exp(-b));
        const float beta_fp16 = fp16_round(b_act);
        l = l * beta_fp16;
        L_sm[tid] = fp16_round(l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 8: Neumann inverse.
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        const float l = L_sm[tid];
        const float ident = (i == j) ? 1.0f : 0.0f;
        INV_sm[tid] = fp16_round(ident - l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(L_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(L_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Lk_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
        INV_bf_out[mqk_off + tid] = bf16_round(INV_sm[tid]);
    }

    // STAGE 9: Mqk = bf16(tril(bf16(q_decayed @ k_inv.T)))
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                q_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }
        threadgroup float* C_dst =
            &Tmp_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float m = bf16_round(Tmp_sm[tid]);
        if (j > i) m = 0.0f;
        Mqk_out[mqk_off + tid] = bf16_round(m);
    }
"""


@lru_cache(maxsize=16)
def _build_prepare_kernel_fused(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template
    return mx.fast.metal_kernel(
        name="flash_kda_prepare_chunk_fused",
        input_names=["k_in", "q_in", "v_in", "g_in", "beta_in", "scale_in"],
        output_names=[
            "k_decayed_out",
            "q_decayed_out",
            "k_restored_out",
            "Mqk_out",
            "INV_bf_out",
            "beta_bf16_out",
            "g_total_exp_out",
        ],
        header=_HEADER,
        source=_PREPARE_SOURCE_FUSED,
        ensure_row_contiguous=True,
    )


def metal_prepare_chunk_fused(
    k_raw: mx.array,
    q_raw: mx.array,
    v: mx.array,
    g_activated: mx.array,
    beta: mx.array,
    scale_bf16_rt: mx.array,
) -> dict[str, mx.array]:
    """Partial-fusion prepare kernel — L2-norm Q/K + sections (b)-(g).

    Moves the L2-norm + bf16-round of q/k into the kernel (saving the
    MLX reduction dispatch). Gate activation of ``g`` stays in MLX
    because the varlen packed path's zero-padding semantics would
    require per-chunk valid-token masks to emulate inside the kernel;
    the partial-fusion split avoids that complexity while still
    capturing the larger half of section (a)'s dispatch overhead.

    Args:
        q_raw, k_raw: ``[n_chunks, H, CHUNK, D]`` fp32 — pre-L2-norm,
            pre-bf16 (raw ``.astype(fp32)`` values from the caller).
        v: ``[n_chunks, H, CHUNK, D]`` fp32 — passthrough.
        g_activated: ``[n_chunks, H, CHUNK, D]`` fp32 — already post-
            section-(a) gate activation (``lower_bound * LOG2E *
            sigmoid(ex2_ftz(A_log * LOG2E) * (g + dt_bias))``).
        beta: ``[n_chunks, H, CHUNK]`` fp32 — pre-sigmoid.
        scale_bf16_rt: scalar fp32 — the bf16-rounded scale.

    Returns: the same dict schema as ``metal_prepare_chunk``.
    """
    assert HAS_METAL_KERNEL, (
        "metal_prepare_chunk_fused called on non-M3+ hardware"
    )
    assert k_raw.ndim == 4 and q_raw.ndim == 4, (
        f"k/q must be 4-D; got k={k_raw.shape} q={q_raw.shape}"
    )
    n_chunks, H, CHUNK, D = k_raw.shape
    assert q_raw.shape == (n_chunks, H, CHUNK, D)
    assert v.shape == (n_chunks, H, CHUNK, D)
    assert g_activated.shape == (n_chunks, H, CHUNK, D)
    assert beta.shape == (n_chunks, H, CHUNK)
    assert CHUNK == 16, f"fused path supports CHUNK=16 only; got {CHUNK}"

    def _as_1d(x: mx.array) -> mx.array:
        return mx.reshape(x, (1,)) if x.shape == () else x

    kernel = _build_prepare_kernel_fused(H=H, D=D, CHUNK=CHUNK)

    outputs = kernel(
        inputs=[k_raw, q_raw, v, g_activated, beta, _as_1d(scale_bf16_rt)],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(256, n_chunks, H),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (n_chunks, H, CHUNK, D),
            (n_chunks, H, CHUNK, D),
            (n_chunks, H, CHUNK, D),
            (n_chunks, H, CHUNK, CHUNK),
            (n_chunks, H, CHUNK, CHUNK),
            (n_chunks, H, CHUNK, 1),
            (n_chunks, H, D, 1),
        ],
        output_dtypes=[mx.float32] * 7,
    )

    return {
        "k_decayed":   outputs[0],
        "q_decayed":   outputs[1],
        "k_restored":  outputs[2],
        "Mqk":         outputs[3],
        "INV_bf":      outputs[4],
        "vc":          v,
        "beta_bf16":   outputs[5],
        "g_total_exp": outputs[6],
    }


# ---------------------------------------------------------------------------
# FUSED2 variant — full section (a) + sections (b)-(g).
# ---------------------------------------------------------------------------
#
# Extends the partial-fusion kernel by also moving KDA gate activation
# into Metal:
#
#   g_act[c,h,d] = lower_bound_log2e * sigmoid(a_log_exp[h] * (g_raw[c,h,d] + dt_bias[h,d]))
#
# Varlen padding semantics (the reason follow-on 1 kept activation in MLX)
# is preserved by a per-chunk valid-token count: positions with
# ``c >= valid_tokens_per_chunk[chunk_id]`` get ``g_act = 0`` so the
# cumsum in stage 2 does not propagate nonzero values from padded tokens.
# For single-seq fixed inputs every chunk has CHUNK valid tokens; for
# packed varlen the last chunk per sequence has fewer, and chunks beyond
# the sequence are fully invalid.
#
# Additional inputs vs the partial-fusion kernel:
#   * a_log_exp_in           : [H]         fp32   = ex2_ftz(A_log * LOG2E)
#   * dt_bias_in             : [H, D]      fp32
#   * lower_bound_log2e_in   : [1]         fp32   = lower_bound * LOG2E
#   * valid_tokens_per_chunk_in : [n_chunks] int32
#
# ``g_in`` is now the RAW gate tensor (post fp32 cast and reshape, but
# pre-activation); the kernel does the activation per-element.
#
# TG memory budget is unchanged from the partial-fusion variant (28 KB).


_PREPARE_SOURCE_FUSED_V2 = """
    // Grid contract:
    //   grid        = (256, n_chunks, H)
    //   threadgroup = (256, 1, 1)
    //
    // FULL-FUSION VARIANT (PR H follow-on 2): Q/K L2-norm + KDA gate
    // activation + sections (b)-(g), all inside one Metal dispatch.
    // Per-token validity is read from valid_tokens_per_chunk_in so
    // padded varlen positions get g_act = 0 (the cumsum in stage 2
    // therefore matches the MLX activate-then-zero-pad pipeline).

    const uint chunk_id = threadgroup_position_in_grid.y;
    const uint head_id  = threadgroup_position_in_grid.z;
    const uint tid      = thread_position_in_threadgroup.x;
    const uint tpg      = 256;
    const uint simd_id  = simdgroup_index_in_threadgroup;
    const uint simd_lane = thread_index_in_simdgroup;

    constexpr uint D_TILES   = uint(D) / 8;
    constexpr uint K_D_TILES = uint(D) / 8;
    constexpr uint C_TILES   = uint(CHUNK) / 8;

    const uint kqv_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(D);
    const uint mqk_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(CHUNK);
    const uint beta_off = (chunk_id * uint(H) + head_id) * uint(CHUNK);
    const uint gtot_off = (chunk_id * uint(H) + head_id) * uint(D);

    threadgroup float g_cumsum_sm[uint(CHUNK) * uint(D)];
    threadgroup float ex_neg_sm  [uint(CHUNK) * uint(D)];
    threadgroup float k_inv_sm   [uint(CHUNK) * uint(D)];
    threadgroup float L_sm       [uint(CHUNK) * uint(CHUNK)];
    threadgroup float INV_sm     [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Lk_sm      [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Tmp_sm     [uint(CHUNK) * uint(CHUNK)];

    // ================================================================
    // STAGE 0z (NEW): KDA gate activation with per-token validity mask.
    //   Each thread reads g_raw, applies activation only if its row is
    //   within the valid-token count for this chunk; otherwise writes 0.
    //   Output goes into g_cumsum_sm (consumed by stage 2 cumsum).
    // ================================================================
    {
        const float a_log_exp_h = a_log_exp_in[head_id];
        const float lb_log2e = lower_bound_log2e_in[0];
        const uint my_valid_count = uint(valid_tokens_per_chunk_in[chunk_id]);

        for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
            const uint c = i / uint(D);
            const uint d = i % uint(D);
            if (c < my_valid_count) {
                const float g_val = g_in[kqv_off + i];
                const float dt_b = dt_bias_in[head_id * uint(D) + d];
                const float pre = a_log_exp_h * (g_val + dt_b);
                const float sig = 1.0f / (1.0f + exp(-pre));
                g_cumsum_sm[i] = lb_log2e * sig;
            } else {
                g_cumsum_sm[i] = 0.0f;
            }
        }
    }

    // ================================================================
    // STAGE 0c: L2-norm of k → scratch in k_decayed_out (unchanged).
    // ================================================================
    {
        constexpr uint ROWS_PER_SIMD = uint(CHUNK) * 32 / 256;
        constexpr uint ELEMS_PER_LANE = uint(D) / 32;
        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {
            const uint c = simd_id * ROWS_PER_SIMD + rb;
            if (c >= uint(CHUNK)) continue;
            float partial = 0.0f;
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = k_in[kqv_off + c * uint(D) + d];
                partial += val * val;
            }
            const float row_sum = simd_sum(partial);
            const float rsqrt_val = rsqrt(row_sum + 1e-6f);
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = k_in[kqv_off + c * uint(D) + d];
                k_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);
            }
        }
    }

    // ================================================================
    // STAGE 0d: L2-norm of q → scratch in q_decayed_out (unchanged).
    // ================================================================
    {
        constexpr uint ROWS_PER_SIMD = uint(CHUNK) * 32 / 256;
        constexpr uint ELEMS_PER_LANE = uint(D) / 32;
        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {
            const uint c = simd_id * ROWS_PER_SIMD + rb;
            if (c >= uint(CHUNK)) continue;
            float partial = 0.0f;
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = q_in[kqv_off + c * uint(D) + d];
                partial += val * val;
            }
            const float row_sum = simd_sum(partial);
            const float rsqrt_val = rsqrt(row_sum + 1e-6f);
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = q_in[kqv_off + c * uint(D) + d];
                q_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);
            }
        }
    }

    // Device + TG barrier: stage 5 reads scratch from k_decayed_out /
    // q_decayed_out across TG threads; stage 2 reads g_cumsum_sm.
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // STAGE 2: cumsum along chunk axis, in place in g_cumsum_sm.
    if (tid < uint(D)) {
        float acc = g_cumsum_sm[tid];
        for (uint c = 1; c < uint(CHUNK); ++c) {
            acc += g_cumsum_sm[c * uint(D) + tid];
            g_cumsum_sm[c * uint(D) + tid] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 3: g_total_exp output.
    if (tid < uint(D)) {
        float gt = g_cumsum_sm[(uint(CHUNK) - 1) * uint(D) + tid];
        g_total_exp_out[gtot_off + tid] = ex2_ftz(gt);
    }

    // STAGE 4: ex_neg, then overwrite g_cumsum_sm with ex_pos.
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        ex_neg_sm[i] = bf16_round(ex2_pos_safe(-cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        g_cumsum_sm[i] = bf16_round(ex2_ftz(cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // STAGE 5: read L2-normed k, q from scratch; compute final
    //          k_decayed, q_decayed, k_inv (TG), k_restored.
    const float scale = scale_in[0];

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        const uint d = i % uint(D);

        const float k = k_decayed_out[kqv_off + i];
        const float q = q_decayed_out[kqv_off + i];
        const float ep = g_cumsum_sm[i];
        const float en = ex_neg_sm[i];

        k_decayed_out[kqv_off + i] = bf16_round(k * ep);

        const float q_int = bf16_round(q * ep);
        q_decayed_out[kqv_off + i] = bf16_round(q_int * scale);

        const float ki = bf16_round(k * en);
        k_inv_sm[i] = ki;

        const float gtot_raw = g_total_exp_out[gtot_off + d];
        const float ex_gtot_d = bf16_round(gtot_raw);
        k_restored_out[kqv_off + i] = bf16_round(ki * ex_gtot_d);
    }

    if (tid < uint(CHUNK)) {
        const float b = beta_in[beta_off + tid];
        const float b_act = 1.0f / (1.0f + exp(-b));
        beta_bf16_out[beta_off + tid] = bf16_round(b_act);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // STAGES 6-9: unchanged from the partial-fusion variant.

    // STAGE 6: L = k_decayed @ k_inv.T
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                k_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }
        threadgroup float* C_dst =
            &L_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 7: L = fp16(L); tril(L,k=-1); L *= beta_fp16; fp16(L).
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float l = fp16_round(L_sm[tid]);
        if (j >= i) l = 0.0f;
        const float b = beta_in[beta_off + i];
        const float b_act = 1.0f / (1.0f + exp(-b));
        const float beta_fp16 = fp16_round(b_act);
        l = l * beta_fp16;
        L_sm[tid] = fp16_round(l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 8: Neumann inverse.
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        const float l = L_sm[tid];
        const float ident = (i == j) ? 1.0f : 0.0f;
        INV_sm[tid] = fp16_round(ident - l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(L_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(L_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Lk_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
        INV_bf_out[mqk_off + tid] = bf16_round(INV_sm[tid]);
    }

    // STAGE 9: Mqk = bf16(tril(bf16(q_decayed @ k_inv.T)))
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                q_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }
        threadgroup float* C_dst =
            &Tmp_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float m = bf16_round(Tmp_sm[tid]);
        if (j > i) m = 0.0f;
        Mqk_out[mqk_off + tid] = bf16_round(m);
    }
"""


@lru_cache(maxsize=16)
def _build_prepare_kernel_fused_v2(H: int, D: int, CHUNK: int):
    del H, D, CHUNK
    return mx.fast.metal_kernel(
        name="flash_kda_prepare_chunk_fused_v2",
        input_names=[
            "k_in", "q_in", "v_in", "g_in", "beta_in", "scale_in",
            "a_log_exp_in", "dt_bias_in", "lower_bound_log2e_in",
            "valid_tokens_per_chunk_in",
        ],
        output_names=[
            "k_decayed_out",
            "q_decayed_out",
            "k_restored_out",
            "Mqk_out",
            "INV_bf_out",
            "beta_bf16_out",
            "g_total_exp_out",
        ],
        header=_HEADER,
        source=_PREPARE_SOURCE_FUSED_V2,
        ensure_row_contiguous=True,
    )


def metal_prepare_chunk_fused_v2(
    k_raw: mx.array,
    q_raw: mx.array,
    v: mx.array,
    g_raw: mx.array,
    beta: mx.array,
    scale_bf16_rt: mx.array,
    a_log_exp: mx.array,
    dt_bias: mx.array,
    lower_bound_log2e: mx.array,
    valid_tokens_per_chunk: mx.array,
) -> dict[str, mx.array]:
    """Full-fusion prepare kernel — section (a) + sections (b)-(g).

    Adds KDA gate activation to the partial-fusion kernel. A per-chunk
    valid-token count masks padded varlen positions so the cumsum in
    stage 2 does not propagate nonzero values from padded tokens.

    Args:
        k_raw, q_raw: ``[n_chunks, H, CHUNK, D]`` fp32 — pre-L2-norm.
        v: ``[n_chunks, H, CHUNK, D]`` fp32 — passthrough.
        g_raw: ``[n_chunks, H, CHUNK, D]`` fp32 — RAW gate (post fp32
            cast and reshape, pre-activation).
        beta: ``[n_chunks, H, CHUNK]`` fp32 — pre-sigmoid.
        scale_bf16_rt: scalar fp32 — bf16-rounded scale.
        a_log_exp: ``[H]`` fp32 — precomputed ``ex2_ftz(A_log * LOG2E)``.
        dt_bias: ``[H, D]`` fp32.
        lower_bound_log2e: ``[1]`` fp32 — precomputed ``lower_bound * LOG2E``.
        valid_tokens_per_chunk: ``[n_chunks]`` int32 — number of valid
            tokens per chunk (CHUNK for fully-valid chunks, fewer for
            the last chunk of a sequence, 0 for fully-padded chunks
            in the packed varlen path).

    Returns: same dict schema as ``metal_prepare_chunk``.
    """
    assert HAS_METAL_KERNEL, (
        "metal_prepare_chunk_fused_v2 called on non-M3+ hardware"
    )
    assert k_raw.ndim == 4 and q_raw.ndim == 4
    n_chunks, H, CHUNK, D = k_raw.shape
    assert q_raw.shape == (n_chunks, H, CHUNK, D)
    assert v.shape == (n_chunks, H, CHUNK, D)
    assert g_raw.shape == (n_chunks, H, CHUNK, D)
    assert beta.shape == (n_chunks, H, CHUNK)
    assert a_log_exp.shape == (H,), (
        f"a_log_exp must be [{H}]; got {a_log_exp.shape}"
    )
    assert dt_bias.shape == (H, D), (
        f"dt_bias must be [{H}, {D}]; got {dt_bias.shape}"
    )
    assert valid_tokens_per_chunk.shape == (n_chunks,), (
        f"valid_tokens_per_chunk must be [{n_chunks}]; "
        f"got {valid_tokens_per_chunk.shape}"
    )
    assert valid_tokens_per_chunk.dtype == mx.int32, (
        f"valid_tokens_per_chunk must be int32; got {valid_tokens_per_chunk.dtype}"
    )
    assert CHUNK == 16, f"fused v2 supports CHUNK=16 only; got {CHUNK}"

    def _as_1d(x: mx.array) -> mx.array:
        return mx.reshape(x, (1,)) if x.shape == () else x

    kernel = _build_prepare_kernel_fused_v2(H=H, D=D, CHUNK=CHUNK)

    outputs = kernel(
        inputs=[
            k_raw, q_raw, v, g_raw, beta, _as_1d(scale_bf16_rt),
            a_log_exp, dt_bias, _as_1d(lower_bound_log2e),
            valid_tokens_per_chunk,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(256, n_chunks, H),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (n_chunks, H, CHUNK, D),
            (n_chunks, H, CHUNK, D),
            (n_chunks, H, CHUNK, D),
            (n_chunks, H, CHUNK, CHUNK),
            (n_chunks, H, CHUNK, CHUNK),
            (n_chunks, H, CHUNK, 1),
            (n_chunks, H, D, 1),
        ],
        output_dtypes=[mx.float32] * 7,
    )

    return {
        "k_decayed":   outputs[0],
        "q_decayed":   outputs[1],
        "k_restored":  outputs[2],
        "Mqk":         outputs[3],
        "INV_bf":      outputs[4],
        "vc":          v,
        "beta_bf16":   outputs[5],
        "g_total_exp": outputs[6],
    }


# ---------------------------------------------------------------------------
# FUSED3 variant — full section (a) + sections (b)-(g) reading TOKEN-MAJOR.
# ---------------------------------------------------------------------------
#
# Functionally identical to ``fused_v2`` (full-fusion: L2-norm + KDA gate
# activation + sections (b)-(g) in one dispatch). The only difference is
# the layout of ``q_in`` / ``k_in`` / ``v_in`` / ``g_in`` / ``beta_in``:
# they are flat token-major buffers, not the tile-major reshape used by
# the v2 kernel.
#
# Why this exists (PR K-a). The PR J audit measured ~8.6 ms / 1.07 GB of
# forced contiguous copies on each forward at single-seq H=64 (bigger than
# the whole PR H follow-on 3 win). Root cause: the caller produces
# token-major ``[T_total, H, D]`` inputs and then runs
# ``reshape(n_chunks, CHUNK, H, D).transpose(0, 2, 1, 3)`` on q/k/v/g/beta
# to feed the v2 kernel's tile-major ``[n_chunks, H, CHUNK, D]`` contract.
# The transpose is a lazy stride change with no copy — but the kernel is
# built with ``ensure_row_contiguous=True`` so MLX inserts an actual
# contiguous copy before launch on every call.
#
# By having the kernel read the original token-major buffer directly we
# eliminate the transpose+copy entirely. Output buffers stay tile-major
# (``[n_chunks, H, CHUNK, ...]``) so the recurrence kernel sees the same
# contract; no transpose appears between prepare and recurrence either.
#
# Index translation (per chunk_id, c (row in chunk), head_id, d):
#
#   tile-major (v2):    kqv_off + c * D + d
#                       where kqv_off = (chunk_id * H + head_id) * CHUNK * D
#   token-major (v3):   ((chunk_id * CHUNK + c) * H + head_id) * D + d
#
# For beta (no D axis):
#   tile-major (v2):    beta_off + c
#                       where beta_off = (chunk_id * H + head_id) * CHUNK
#   token-major (v3):   (chunk_id * CHUNK + c) * H + head_id
#
# Output offsets are unchanged (the kernel WRITES tile-major into fresh
# allocations).
#
# Stages 0-9 are otherwise identical to v2: same threadgroup layout, same
# 28 KB TG memory budget, same stage-0z gate-zero-pad mask via
# ``valid_tokens_per_chunk``. Outputs match v2 byte-for-byte (modulo any
# reduction-order shifts inherent to the kernel — which would also affect
# v2; v3 is purely a layout change on the input side).


_PREPARE_SOURCE_FUSED_V3 = """
    // Grid contract:
    //   grid        = (256, n_chunks, H)
    //   threadgroup = (256, 1, 1)
    //
    // TOKEN-MAJOR FULL-FUSION VARIANT (PR K-a):
    //   * Inputs q_in / k_in / v_in / g_in are [T_total, H, D] flat
    //     buffers (T_total = n_chunks * CHUNK after caller-side padding).
    //   * Input beta_in is [T_total, H].
    //   * Outputs are tile-major [n_chunks, H, CHUNK, ...] — same as v2.
    //   * Eliminates the [n_chunks, CHUNK, H, D] -> [n_chunks, H, CHUNK, D]
    //     transpose + ensure_row_contiguous copy in the caller.

    const uint chunk_id = threadgroup_position_in_grid.y;
    const uint head_id  = threadgroup_position_in_grid.z;
    const uint tid      = thread_position_in_threadgroup.x;
    const uint tpg      = 256;
    const uint simd_id  = simdgroup_index_in_threadgroup;
    const uint simd_lane = thread_index_in_simdgroup;

    constexpr uint D_TILES   = uint(D) / 8;
    constexpr uint K_D_TILES = uint(D) / 8;
    constexpr uint C_TILES   = uint(CHUNK) / 8;

    // Output bases (tile-major, identical to v2).
    const uint kqv_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(D);
    const uint mqk_off  = (chunk_id * uint(H) + head_id) * uint(CHUNK) * uint(CHUNK);
    const uint beta_off = (chunk_id * uint(H) + head_id) * uint(CHUNK);
    const uint gtot_off = (chunk_id * uint(H) + head_id) * uint(D);

    // Token-major input bases for this chunk and head. The flat token
    // index for (chunk_id, c=row, head_id) is:
    //   t = chunk_id * CHUNK + c
    //   q_in[t * H * D + head_id * D + d]
    //   beta_in[t * H + head_id]
    const uint tok_chunk_base = chunk_id * uint(CHUNK);
    const uint tm_stride_t    = uint(H) * uint(D);   // distance between successive tokens
    const uint tm_head_off    = head_id * uint(D);

    threadgroup float g_cumsum_sm[uint(CHUNK) * uint(D)];
    threadgroup float ex_neg_sm  [uint(CHUNK) * uint(D)];
    threadgroup float k_inv_sm   [uint(CHUNK) * uint(D)];
    threadgroup float L_sm       [uint(CHUNK) * uint(CHUNK)];
    threadgroup float INV_sm     [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Lk_sm      [uint(CHUNK) * uint(CHUNK)];
    threadgroup float Tmp_sm     [uint(CHUNK) * uint(CHUNK)];

    // ================================================================
    // STAGE 0z: KDA gate activation with per-token validity mask.
    //   Reads g_in token-major. Writes g_cumsum_sm tile-major (TG).
    // ================================================================
    {
        const float a_log_exp_h = a_log_exp_in[head_id];
        const float lb_log2e = lower_bound_log2e_in[0];
        const uint my_valid_count = uint(valid_tokens_per_chunk_in[chunk_id]);

        for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
            const uint c = i / uint(D);
            const uint d = i % uint(D);
            if (c < my_valid_count) {
                const uint tm_idx = (tok_chunk_base + c) * tm_stride_t
                                    + tm_head_off + d;
                const float g_val = g_in[tm_idx];
                const float dt_b = dt_bias_in[head_id * uint(D) + d];
                const float pre = a_log_exp_h * (g_val + dt_b);
                const float sig = 1.0f / (1.0f + exp(-pre));
                g_cumsum_sm[i] = lb_log2e * sig;
            } else {
                g_cumsum_sm[i] = 0.0f;
            }
        }
    }

    // ================================================================
    // STAGE 0v: copy v from token-major to tile-major output buffer.
    //   This is a layout transform only — no arithmetic. Doing it in the
    //   kernel avoids an MLX-side ensure_row_contiguous copy on vc when
    //   the recurrence kernel later loads it.
    // ================================================================
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        const uint c = i / uint(D);
        const uint d = i % uint(D);
        const uint tm_idx = (tok_chunk_base + c) * tm_stride_t
                            + tm_head_off + d;
        vc_out[kqv_off + i] = v_in[tm_idx];
    }

    // ================================================================
    // STAGE 0c: L2-norm of k → scratch in k_decayed_out (tile-major).
    // ================================================================
    {
        constexpr uint ROWS_PER_SIMD = uint(CHUNK) * 32 / 256;
        constexpr uint ELEMS_PER_LANE = uint(D) / 32;
        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {
            const uint c = simd_id * ROWS_PER_SIMD + rb;
            if (c >= uint(CHUNK)) continue;
            const uint tm_row_base = (tok_chunk_base + c) * tm_stride_t + tm_head_off;
            float partial = 0.0f;
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = k_in[tm_row_base + d];
                partial += val * val;
            }
            const float row_sum = simd_sum(partial);
            const float rsqrt_val = rsqrt(row_sum + 1e-6f);
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = k_in[tm_row_base + d];
                k_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);
            }
        }
    }

    // ================================================================
    // STAGE 0d: L2-norm of q → scratch in q_decayed_out (tile-major).
    // ================================================================
    {
        constexpr uint ROWS_PER_SIMD = uint(CHUNK) * 32 / 256;
        constexpr uint ELEMS_PER_LANE = uint(D) / 32;
        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {
            const uint c = simd_id * ROWS_PER_SIMD + rb;
            if (c >= uint(CHUNK)) continue;
            const uint tm_row_base = (tok_chunk_base + c) * tm_stride_t + tm_head_off;
            float partial = 0.0f;
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = q_in[tm_row_base + d];
                partial += val * val;
            }
            const float row_sum = simd_sum(partial);
            const float rsqrt_val = rsqrt(row_sum + 1e-6f);
            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
                const uint d = simd_lane * ELEMS_PER_LANE + e;
                const float val = q_in[tm_row_base + d];
                q_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);
            }
        }
    }

    // Device + TG barrier: stage 5 reads scratch from k_decayed_out /
    // q_decayed_out across TG threads; stage 2 reads g_cumsum_sm.
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // STAGE 2: cumsum along chunk axis, in place in g_cumsum_sm.
    if (tid < uint(D)) {
        float acc = g_cumsum_sm[tid];
        for (uint c = 1; c < uint(CHUNK); ++c) {
            acc += g_cumsum_sm[c * uint(D) + tid];
            g_cumsum_sm[c * uint(D) + tid] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 3: g_total_exp output.
    if (tid < uint(D)) {
        float gt = g_cumsum_sm[(uint(CHUNK) - 1) * uint(D) + tid];
        g_total_exp_out[gtot_off + tid] = ex2_ftz(gt);
    }

    // STAGE 4: ex_neg, then overwrite g_cumsum_sm with ex_pos.
    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        ex_neg_sm[i] = bf16_round(ex2_pos_safe(-cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        float cs = g_cumsum_sm[i];
        g_cumsum_sm[i] = bf16_round(ex2_ftz(cs));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // STAGE 5: read L2-normed k, q from scratch (tile-major); compute
    //          final k_decayed, q_decayed, k_inv (TG), k_restored.
    const float scale = scale_in[0];

    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {
        const uint d = i % uint(D);

        const float k = k_decayed_out[kqv_off + i];
        const float q = q_decayed_out[kqv_off + i];
        const float ep = g_cumsum_sm[i];
        const float en = ex_neg_sm[i];

        k_decayed_out[kqv_off + i] = bf16_round(k * ep);

        const float q_int = bf16_round(q * ep);
        q_decayed_out[kqv_off + i] = bf16_round(q_int * scale);

        const float ki = bf16_round(k * en);
        k_inv_sm[i] = ki;

        const float gtot_raw = g_total_exp_out[gtot_off + d];
        const float ex_gtot_d = bf16_round(gtot_raw);
        k_restored_out[kqv_off + i] = bf16_round(ki * ex_gtot_d);
    }

    // beta_bf16: read beta_in token-major, write tile-major output.
    if (tid < uint(CHUNK)) {
        const uint tm_beta_idx = (tok_chunk_base + tid) * uint(H) + head_id;
        const float b = beta_in[tm_beta_idx];
        const float b_act = 1.0f / (1.0f + exp(-b));
        beta_bf16_out[beta_off + tid] = bf16_round(b_act);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                        mem_flags::mem_device);

    // STAGES 6-9: byte-identical to v2 (operate on TG / device tile-major).

    // STAGE 6: L = k_decayed @ k_inv.T
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                k_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }
        threadgroup float* C_dst =
            &L_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 7: L = fp16(L); tril(L,k=-1); L *= beta_fp16; fp16(L).
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float l = fp16_round(L_sm[tid]);
        if (j >= i) l = 0.0f;
        // beta sigmoid recomputed from token-major input here.
        const uint tm_beta_idx = (tok_chunk_base + i) * uint(H) + head_id;
        const float b = beta_in[tm_beta_idx];
        const float b_act = 1.0f / (1.0f + exp(-b));
        const float beta_fp16 = fp16_round(b_act);
        l = l * beta_fp16;
        L_sm[tid] = fp16_round(l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STAGE 8: Neumann inverse.
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        const float l = L_sm[tid];
        const float ident = (i == j) ? 1.0f : 0.0f;
        INV_sm[tid] = fp16_round(ident - l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(L_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(L_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Lk_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(Lk_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(CHUNK) * uint(CHUNK)) {
        Lk_sm[tid] = Tmp_sm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float acc = 0.0f;
        for (uint k = 0; k < uint(CHUNK); ++k) {
            const float a = fp16_round(INV_sm[i * uint(CHUNK) + k]);
            const float b = fp16_round(Lk_sm[k * uint(CHUNK) + j]);
            acc += a * b;
        }
        Tmp_sm[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        INV_sm[tid] = INV_sm[tid] + Tmp_sm[tid];
        INV_bf_out[mqk_off + tid] = bf16_round(INV_sm[tid]);
    }

    // STAGE 9: Mqk = bf16(tril(bf16(q_decayed @ k_inv.T)))
    if (simd_id < C_TILES * C_TILES) {
        const uint tile_r = simd_id / C_TILES;
        const uint tile_c = simd_id % C_TILES;
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                q_decayed_out + kqv_off + (tile_r * 8) * uint(D) + (k_tile * 8);
            simdgroup_load(A_tile, A_src, uint(D));
            const threadgroup float* B_src =
                &k_inv_sm[(tile_c * 8) * uint(D) + (k_tile * 8)];
            simdgroup_load(B_tile, B_src, uint(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }
        threadgroup float* C_dst =
            &Tmp_sm[(tile_r * 8) * uint(CHUNK) + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, uint(CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(CHUNK) * uint(CHUNK)) {
        const uint i = tid / uint(CHUNK);
        const uint j = tid % uint(CHUNK);
        float m = bf16_round(Tmp_sm[tid]);
        if (j > i) m = 0.0f;
        Mqk_out[mqk_off + tid] = bf16_round(m);
    }
"""


@lru_cache(maxsize=16)
def _build_prepare_kernel_fused_v3(H: int, D: int, CHUNK: int):
    del H, D, CHUNK
    return mx.fast.metal_kernel(
        name="flash_kda_prepare_chunk_fused_v3",
        input_names=[
            "k_in", "q_in", "v_in", "g_in", "beta_in", "scale_in",
            "a_log_exp_in", "dt_bias_in", "lower_bound_log2e_in",
            "valid_tokens_per_chunk_in",
        ],
        output_names=[
            "k_decayed_out",
            "q_decayed_out",
            "k_restored_out",
            "vc_out",
            "Mqk_out",
            "INV_bf_out",
            "beta_bf16_out",
            "g_total_exp_out",
        ],
        header=_HEADER,
        source=_PREPARE_SOURCE_FUSED_V3,
        ensure_row_contiguous=True,
    )


def metal_prepare_chunk_fused_v3(
    k_raw: mx.array,
    q_raw: mx.array,
    v: mx.array,
    g_raw: mx.array,
    beta: mx.array,
    scale_bf16_rt: mx.array,
    a_log_exp: mx.array,
    dt_bias: mx.array,
    lower_bound_log2e: mx.array,
    valid_tokens_per_chunk: mx.array,
) -> dict[str, mx.array]:
    """Token-major full-fusion prepare kernel (PR K-a).

    Functionally equivalent to ``metal_prepare_chunk_fused_v2`` — the
    arithmetic and reductions are bit-identical. Only the input layout
    differs: q/k/v/g arrive as ``[T_total, H, D]`` (token-major) and
    ``beta`` as ``[T_total, H]``, with ``T_total = n_chunks * CHUNK``.
    The kernel reads them directly without the
    ``reshape -> transpose(0, 2, 1, 3)`` step the v2 caller has to do —
    eliminating MLX's ~1 GB ``ensure_row_contiguous`` copy on every
    forward at bench scale (single-seq H=64).

    Outputs are tile-major ``[n_chunks, H, CHUNK, ...]`` (same as v2),
    so the recurrence kernel consumes them unchanged.

    Args:
        k_raw, q_raw, v: ``[T_total, H, D]`` fp32 (or bf16) — token-major.
        g_raw: ``[T_total, H, D]`` fp32 — RAW gate (post fp32 cast,
            pre-activation).
        beta: ``[T_total, H]`` fp32 — pre-sigmoid.
        scale_bf16_rt: scalar fp32 — bf16-rounded scale.
        a_log_exp: ``[H]`` fp32 — precomputed ``ex2_ftz(A_log * LOG2E)``.
        dt_bias: ``[H, D]`` fp32.
        lower_bound_log2e: ``[1]`` fp32 — precomputed ``lower_bound * LOG2E``.
        valid_tokens_per_chunk: ``[n_chunks]`` int32 — per-chunk valid
            token count (CHUNK for fully-valid chunks, fewer for the
            last chunk of a sequence, 0 for fully-padded chunks in the
            packed varlen path).

    Returns: same dict schema as ``metal_prepare_chunk`` and
    ``metal_prepare_chunk_fused_v2``.
    """
    assert HAS_METAL_KERNEL, (
        "metal_prepare_chunk_fused_v3 called on non-M3+ hardware"
    )
    assert k_raw.ndim == 3 and q_raw.ndim == 3 and v.ndim == 3 and g_raw.ndim == 3, (
        f"k/q/v/g must be 3-D token-major; got "
        f"k={k_raw.shape} q={q_raw.shape} v={v.shape} g={g_raw.shape}"
    )
    T_total, H, D = k_raw.shape
    assert q_raw.shape == (T_total, H, D)
    assert v.shape == (T_total, H, D)
    assert g_raw.shape == (T_total, H, D)
    assert beta.shape == (T_total, H), (
        f"beta must be [T_total={T_total}, H={H}]; got {beta.shape}"
    )
    CHUNK = 16
    assert T_total % CHUNK == 0, (
        f"T_total must be a multiple of CHUNK={CHUNK}; got T_total={T_total}"
    )
    n_chunks = T_total // CHUNK
    assert a_log_exp.shape == (H,), (
        f"a_log_exp must be [{H}]; got {a_log_exp.shape}"
    )
    assert dt_bias.shape == (H, D), (
        f"dt_bias must be [{H}, {D}]; got {dt_bias.shape}"
    )
    assert valid_tokens_per_chunk.shape == (n_chunks,), (
        f"valid_tokens_per_chunk must be [{n_chunks}]; "
        f"got {valid_tokens_per_chunk.shape}"
    )
    assert valid_tokens_per_chunk.dtype == mx.int32, (
        f"valid_tokens_per_chunk must be int32; got {valid_tokens_per_chunk.dtype}"
    )

    def _as_1d(x: mx.array) -> mx.array:
        return mx.reshape(x, (1,)) if x.shape == () else x

    kernel = _build_prepare_kernel_fused_v3(H=H, D=D, CHUNK=CHUNK)

    # vc carries the dtype of the caller-supplied v (bf16 or fp32). The
    # kernel writes ``v_in[tm_idx]`` straight into ``vc_out`` so output
    # dtype must equal input dtype to avoid a hidden Metal cast (and to
    # preserve PR H follow-on 3 Phase B's bf16 bandwidth savings).
    vc_out_dtype = v.dtype

    outputs = kernel(
        inputs=[
            k_raw, q_raw, v, g_raw, beta, _as_1d(scale_bf16_rt),
            a_log_exp, dt_bias, _as_1d(lower_bound_log2e),
            valid_tokens_per_chunk,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(256, n_chunks, H),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (n_chunks, H, CHUNK, D),     # k_decayed
            (n_chunks, H, CHUNK, D),     # q_decayed
            (n_chunks, H, CHUNK, D),     # k_restored
            (n_chunks, H, CHUNK, D),     # vc (token-major -> tile-major copy)
            (n_chunks, H, CHUNK, CHUNK), # Mqk
            (n_chunks, H, CHUNK, CHUNK), # INV_bf
            (n_chunks, H, CHUNK, 1),     # beta_bf16
            (n_chunks, H, D, 1),         # g_total_exp
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.float32,
            vc_out_dtype,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
    )

    return {
        "k_decayed":   outputs[0],
        "q_decayed":   outputs[1],
        "k_restored":  outputs[2],
        "vc":          outputs[3],
        "Mqk":         outputs[4],
        "INV_bf":      outputs[5],
        "beta_bf16":   outputs[6],
        "g_total_exp": outputs[7],
    }


# ===========================================================================
# fused_v4 — flat-ragged token-major prepare kernel (PR M Option A)
# ===========================================================================
#
# v4 is functionally and arithmetically identical to v3 (token-major
# full-fusion: L2-norm + KDA gate activation + sections (b)-(g) in one
# Metal dispatch). The ONLY difference is how it computes the per-chunk
# starting token index for input addressing:
#
#   v3 :  tok_chunk_base = chunk_id * CHUNK
#         (every chunk lives at a fixed CHUNK-aligned offset; varlen padding
#         is achieved by a CHUNK-aligned ``ensure_row_contiguous`` pack
#         before the call, then masking padded tokens via valid_tokens.)
#
#   v4 :  tok_chunk_base = chunk_token_start_in[chunk_id]
#         (chunks reference arbitrary token offsets in a single FLAT
#         ``[T_total, H, D]`` buffer — no padding allocation. The caller
#         passes a tiny [n_total_chunks] int32 metadata table that tells
#         each chunk which token to start reading from.)
#
# Use case: the packed varlen path used to allocate
# ``[N, max_padded_T, H, D]`` (with max_padded_T = max_chunks*CHUNK) and
# zero-pad each sequence to that length, costing a serialized DRAM copy.
# v4 reads the unpadded flat buffer ``[T_total, H, D]`` directly using
# the metadata table; the int32 table is ~2 KB even for thousands of
# chunks, so this is essentially free.
#
# Output layout is unchanged (tile-major ``[n_total_chunks, H, CHUNK, ...]``);
# the recurrence kernel consumes outputs unmodified.

# v4 = v3 with two surgical changes:
#   (1) tok_chunk_base reads from the per-chunk metadata table instead of
#       the implicit chunk_id*CHUNK formula (the address-mode change).
#   (2) Input reads on stages 0v, 0c, 0d, and beta (stages 0z and 7 row
#       masking) are gated on ``c < valid_count`` and zeroed otherwise.
#
# Why (2): the v3 caller GUARANTEES that input rows at c >= valid_count
# are zero-padded (each seq is independently padded to CHUNK before being
# stacked). v4 reads from a flat unpadded buffer where those rows belong
# to the NEXT sequence — junk reads would corrupt downstream state in
# the recurrence kernel. We patch v3's source so the masks live in the
# kernel itself (output rows for c >= valid_count are deterministic
# zero-derived, matching v3's behaviour exactly).
def _build_fused_v4_source(v3_src: str) -> str:
    src = v3_src

    # (1) Address-mode change.
    src = src.replace(
        "const uint tok_chunk_base = chunk_id * uint(CHUNK);",
        # In v4 we ALSO want to thread valid-count into the stage 0v / 0c /
        # 0d masks; the easiest way is to hoist a per-chunk valid-count
        # variable next to the new tok_chunk_base.
        "const uint tok_chunk_base = uint(chunk_token_start_in[chunk_id]);\n"
        "    const uint v4_valid_count = uint(valid_tokens_per_chunk_in[chunk_id]);",
    )

    # (2a) Stage 0v: copy v. For invalid rows, write 0 (matches v3's
    # input-zero-pad invariant). To support both vc_out=fp32 and vc_out=bf16,
    # we read v_in[tok_chunk_base+0+...] as a benign typed-zero source.
    # Simpler: cast a fp32 zero to the input dtype via the load path.
    src = src.replace(
        "    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {\n"
        "        const uint c = i / uint(D);\n"
        "        const uint d = i % uint(D);\n"
        "        const uint tm_idx = (tok_chunk_base + c) * tm_stride_t\n"
        "                            + tm_head_off + d;\n"
        "        vc_out[kqv_off + i] = v_in[tm_idx];\n"
        "    }",
        "    for (uint i = tid; i < uint(CHUNK) * uint(D); i += tpg) {\n"
        "        const uint c = i / uint(D);\n"
        "        const uint d = i % uint(D);\n"
        "        if (c < v4_valid_count) {\n"
        "            const uint tm_idx = (tok_chunk_base + c) * tm_stride_t\n"
        "                                + tm_head_off + d;\n"
        "            vc_out[kqv_off + i] = v_in[tm_idx];\n"
        "        } else {\n"
        "            // Zero invalid rows. The store dtype matches v_in's dtype\n"
        "            // (fp32 or bf16); writing the integer 0 lets the implicit\n"
        "            // conversion produce the correctly-typed zero in both cases.\n"
        "            vc_out[kqv_off + i] = 0;\n"
        "        }\n"
        "    }",
    )

    # (2b) Stage 0c: L2-norm of k. Mask both the read and the writeback.
    src = src.replace(
        "        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {\n"
        "            const uint c = simd_id * ROWS_PER_SIMD + rb;\n"
        "            if (c >= uint(CHUNK)) continue;\n"
        "            const uint tm_row_base = (tok_chunk_base + c) * tm_stride_t + tm_head_off;\n"
        "            float partial = 0.0f;\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = k_in[tm_row_base + d];\n"
        "                partial += val * val;\n"
        "            }\n"
        "            const float row_sum = simd_sum(partial);\n"
        "            const float rsqrt_val = rsqrt(row_sum + 1e-6f);\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = k_in[tm_row_base + d];\n"
        "                k_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);\n"
        "            }\n"
        "        }",
        "        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {\n"
        "            const uint c = simd_id * ROWS_PER_SIMD + rb;\n"
        "            if (c >= uint(CHUNK)) continue;\n"
        "            if (c >= v4_valid_count) {\n"
        "                // Zero invalid rows — mirrors v3's input-zero-pad invariant.\n"
        "                for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                    const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                    k_decayed_out[kqv_off + c * uint(D) + d] = 0.0f;\n"
        "                }\n"
        "                continue;\n"
        "            }\n"
        "            const uint tm_row_base = (tok_chunk_base + c) * tm_stride_t + tm_head_off;\n"
        "            float partial = 0.0f;\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = k_in[tm_row_base + d];\n"
        "                partial += val * val;\n"
        "            }\n"
        "            const float row_sum = simd_sum(partial);\n"
        "            const float rsqrt_val = rsqrt(row_sum + 1e-6f);\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = k_in[tm_row_base + d];\n"
        "                k_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);\n"
        "            }\n"
        "        }",
    )

    # (2c) Stage 0d: L2-norm of q.
    src = src.replace(
        "        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {\n"
        "            const uint c = simd_id * ROWS_PER_SIMD + rb;\n"
        "            if (c >= uint(CHUNK)) continue;\n"
        "            const uint tm_row_base = (tok_chunk_base + c) * tm_stride_t + tm_head_off;\n"
        "            float partial = 0.0f;\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = q_in[tm_row_base + d];\n"
        "                partial += val * val;\n"
        "            }\n"
        "            const float row_sum = simd_sum(partial);\n"
        "            const float rsqrt_val = rsqrt(row_sum + 1e-6f);\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = q_in[tm_row_base + d];\n"
        "                q_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);\n"
        "            }\n"
        "        }",
        "        for (uint rb = 0; rb < ROWS_PER_SIMD; ++rb) {\n"
        "            const uint c = simd_id * ROWS_PER_SIMD + rb;\n"
        "            if (c >= uint(CHUNK)) continue;\n"
        "            if (c >= v4_valid_count) {\n"
        "                for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                    const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                    q_decayed_out[kqv_off + c * uint(D) + d] = 0.0f;\n"
        "                }\n"
        "                continue;\n"
        "            }\n"
        "            const uint tm_row_base = (tok_chunk_base + c) * tm_stride_t + tm_head_off;\n"
        "            float partial = 0.0f;\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = q_in[tm_row_base + d];\n"
        "                partial += val * val;\n"
        "            }\n"
        "            const float row_sum = simd_sum(partial);\n"
        "            const float rsqrt_val = rsqrt(row_sum + 1e-6f);\n"
        "            for (uint e = 0; e < ELEMS_PER_LANE; ++e) {\n"
        "                const uint d = simd_lane * ELEMS_PER_LANE + e;\n"
        "                const float val = q_in[tm_row_base + d];\n"
        "                q_decayed_out[kqv_off + c * uint(D) + d] = bf16_round(val * rsqrt_val);\n"
        "            }\n"
        "        }",
    )

    # (2d) Beta read (the 1-D beta_bf16 writeback). To preserve byte-for-byte
    # parity with v3-on-zero-padded-input, treat invalid rows as if beta_in=0
    # (then run the standard sigmoid path). v3 reads 0 from the padded buffer
    # and writes bf16(sigmoid(0))=bf16(0.5); we mirror that exactly.
    src = src.replace(
        "    if (tid < uint(CHUNK)) {\n"
        "        const uint tm_beta_idx = (tok_chunk_base + tid) * uint(H) + head_id;\n"
        "        const float b = beta_in[tm_beta_idx];\n"
        "        const float b_act = 1.0f / (1.0f + exp(-b));\n"
        "        beta_bf16_out[beta_off + tid] = bf16_round(b_act);\n"
        "    }",
        "    if (tid < uint(CHUNK)) {\n"
        "        float b;\n"
        "        if (tid < v4_valid_count) {\n"
        "            const uint tm_beta_idx = (tok_chunk_base + tid) * uint(H) + head_id;\n"
        "            b = beta_in[tm_beta_idx];\n"
        "        } else {\n"
        "            b = 0.0f;\n"
        "        }\n"
        "        const float b_act = 1.0f / (1.0f + exp(-b));\n"
        "        beta_bf16_out[beta_off + tid] = bf16_round(b_act);\n"
        "    }",
    )

    # (2e) Beta read in stage 7 (the L *= beta_fp16 step). Same treatment.
    src = src.replace(
        "        // beta sigmoid recomputed from token-major input here.\n"
        "        const uint tm_beta_idx = (tok_chunk_base + i) * uint(H) + head_id;\n"
        "        const float b = beta_in[tm_beta_idx];\n"
        "        const float b_act = 1.0f / (1.0f + exp(-b));\n"
        "        const float beta_fp16 = fp16_round(b_act);\n"
        "        l = l * beta_fp16;",
        "        // beta sigmoid recomputed from token-major input here.\n"
        "        // For invalid rows, treat input as 0 (matches v3 zero-pad).\n"
        "        float b_stage7;\n"
        "        if (i < v4_valid_count) {\n"
        "            const uint tm_beta_idx = (tok_chunk_base + i) * uint(H) + head_id;\n"
        "            b_stage7 = beta_in[tm_beta_idx];\n"
        "        } else {\n"
        "            b_stage7 = 0.0f;\n"
        "        }\n"
        "        const float b_act = 1.0f / (1.0f + exp(-b_stage7));\n"
        "        const float beta_fp16 = fp16_round(b_act);\n"
        "        l = l * beta_fp16;",
    )

    return src


_PREPARE_SOURCE_FUSED_V4 = _build_fused_v4_source(_PREPARE_SOURCE_FUSED_V3)
assert _PREPARE_SOURCE_FUSED_V4 != _PREPARE_SOURCE_FUSED_V3, (
    "fused_v4 source should differ from fused_v3; "
    "the v3 source no longer contains the expected substitution targets — "
    "this builder needs an update to track v3 changes."
)


@lru_cache(maxsize=16)
def _build_prepare_kernel_fused_v4(H: int, D: int, CHUNK: int):
    del H, D, CHUNK
    return mx.fast.metal_kernel(
        name="flash_kda_prepare_chunk_fused_v4",
        input_names=[
            "k_in", "q_in", "v_in", "g_in", "beta_in", "scale_in",
            "a_log_exp_in", "dt_bias_in", "lower_bound_log2e_in",
            "valid_tokens_per_chunk_in",
            "chunk_token_start_in",
        ],
        output_names=[
            "k_decayed_out",
            "q_decayed_out",
            "k_restored_out",
            "vc_out",
            "Mqk_out",
            "INV_bf_out",
            "beta_bf16_out",
            "g_total_exp_out",
        ],
        header=_HEADER,
        source=_PREPARE_SOURCE_FUSED_V4,
        ensure_row_contiguous=True,
    )


def metal_prepare_chunk_fused_v4(
    k_raw: mx.array,
    q_raw: mx.array,
    v: mx.array,
    g_raw: mx.array,
    beta: mx.array,
    scale_bf16_rt: mx.array,
    a_log_exp: mx.array,
    dt_bias: mx.array,
    lower_bound_log2e: mx.array,
    valid_tokens_per_chunk: mx.array,
    chunk_token_start: mx.array,
) -> dict[str, mx.array]:
    """Flat-ragged token-major full-fusion prepare kernel (PR M Option A).

    Functionally and arithmetically equivalent to
    ``metal_prepare_chunk_fused_v3``. The only difference: each output
    chunk reads its source tokens starting at
    ``chunk_token_start[chunk_id]`` in the flat ``[T_total, H, D]``
    input buffer, instead of the implicit ``chunk_id * CHUNK`` of v3.

    This lets the caller hand the kernel a SINGLE flat unpadded buffer
    spanning all packed sequences, with chunks pointing into per-seq
    boundaries via the int32 metadata table — eliminating the
    ``[N, max_padded_T, H, D]`` zero-pad allocation that the packed
    fused3 path forces (~30-50 MB of serialized DRAM at bench scale).

    Args:
        k_raw, q_raw, v, g_raw: ``[T_total, H, D]`` — flat token-major
            buffers spanning all packed sequences (NO per-seq padding).
            ``T_total`` is the sum of caller-supplied sequence lengths.
        beta: ``[T_total, H]`` flat token-major.
        scale_bf16_rt: scalar fp32 — bf16-rounded scale.
        a_log_exp: ``[H]`` fp32 — precomputed ``ex2_ftz(A_log * LOG2E)``.
        dt_bias: ``[H, D]`` fp32.
        lower_bound_log2e: ``[1]`` fp32 — precomputed
            ``lower_bound * LOG2E``.
        valid_tokens_per_chunk: ``[n_total_chunks]`` int32 — per-chunk
            valid-token count. Mirrors v3 semantics: full chunks pass
            CHUNK, the partial last chunk of each sequence passes its
            tail length.
        chunk_token_start: ``[n_total_chunks]`` int32 — flat token
            index where each output chunk begins reading. For sequence
            ``n`` with bos=cu[n], the c-th chunk starts at
            ``cu[n] + c*CHUNK``.

    Returns: same dict schema as ``metal_prepare_chunk_fused_v3``.
    """
    assert HAS_METAL_KERNEL, (
        "metal_prepare_chunk_fused_v4 called on non-M3+ hardware"
    )
    assert k_raw.ndim == 3 and q_raw.ndim == 3 and v.ndim == 3 and g_raw.ndim == 3, (
        f"k/q/v/g must be 3-D token-major; got "
        f"k={k_raw.shape} q={q_raw.shape} v={v.shape} g={g_raw.shape}"
    )
    T_total, H, D = k_raw.shape
    assert q_raw.shape == (T_total, H, D)
    assert v.shape == (T_total, H, D)
    assert g_raw.shape == (T_total, H, D)
    assert beta.shape == (T_total, H), (
        f"beta must be [T_total={T_total}, H={H}]; got {beta.shape}"
    )
    CHUNK = 16
    # NB: T_total is NOT required to be a multiple of CHUNK in v4 —
    # the chunk_token_start table can point chunks at arbitrary offsets,
    # and the kernel reads token-by-token (token_id < T_total is the
    # only constraint, enforced via valid_tokens_per_chunk).
    assert valid_tokens_per_chunk.ndim == 1
    n_total_chunks = valid_tokens_per_chunk.shape[0]
    assert chunk_token_start.shape == (n_total_chunks,), (
        f"chunk_token_start must be [{n_total_chunks}]; "
        f"got {chunk_token_start.shape}"
    )
    assert valid_tokens_per_chunk.dtype == mx.int32, (
        f"valid_tokens_per_chunk must be int32; "
        f"got {valid_tokens_per_chunk.dtype}"
    )
    assert chunk_token_start.dtype == mx.int32, (
        f"chunk_token_start must be int32; got {chunk_token_start.dtype}"
    )
    assert a_log_exp.shape == (H,), (
        f"a_log_exp must be [{H}]; got {a_log_exp.shape}"
    )
    assert dt_bias.shape == (H, D), (
        f"dt_bias must be [{H}, {D}]; got {dt_bias.shape}"
    )

    def _as_1d(x: mx.array) -> mx.array:
        return mx.reshape(x, (1,)) if x.shape == () else x

    kernel = _build_prepare_kernel_fused_v4(H=H, D=D, CHUNK=CHUNK)

    # vc inherits the dtype of v — see fused_v3 docstring for rationale
    # (PR H follow-on 3 Phase B bf16 bandwidth preservation).
    vc_out_dtype = v.dtype

    outputs = kernel(
        inputs=[
            k_raw, q_raw, v, g_raw, beta, _as_1d(scale_bf16_rt),
            a_log_exp, dt_bias, _as_1d(lower_bound_log2e),
            valid_tokens_per_chunk,
            chunk_token_start,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(256, n_total_chunks, H),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (n_total_chunks, H, CHUNK, D),     # k_decayed
            (n_total_chunks, H, CHUNK, D),     # q_decayed
            (n_total_chunks, H, CHUNK, D),     # k_restored
            (n_total_chunks, H, CHUNK, D),     # vc
            (n_total_chunks, H, CHUNK, CHUNK), # Mqk
            (n_total_chunks, H, CHUNK, CHUNK), # INV_bf
            (n_total_chunks, H, CHUNK, 1),     # beta_bf16
            (n_total_chunks, H, D, 1),         # g_total_exp
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.float32,
            vc_out_dtype,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
    )

    return {
        "k_decayed":   outputs[0],
        "q_decayed":   outputs[1],
        "k_restored":  outputs[2],
        "vc":          outputs[3],
        "Mqk":         outputs[4],
        "INV_bf":      outputs[5],
        "beta_bf16":   outputs[6],
        "g_total_exp": outputs[7],
    }


__all__ = [
    "HAS_METAL_KERNEL",
    "metal_prepare_chunk",
    "metal_prepare_chunk_fused",
    "metal_prepare_chunk_fused_v2",
    "metal_prepare_chunk_fused_v3",
    "metal_prepare_chunk_fused_v4",
]
