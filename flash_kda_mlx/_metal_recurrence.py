"""Metal kernel infrastructure for the fused recurrence body (plan PR E Phase 2c).

Phase 1 scope (this file): build the machinery that later phases will reuse,
plus a single-matmul kernel that exercises the whole pipeline end-to-end.

The one matmul we implement is ``matmul_A_by_B(A, B) → A @ B`` at shapes
``[H, CHUNK, D] @ [H, D, D]`` — the shape appearing in matmuls #1 and #3
of ``_recurrence_body_single``. Pure fp32 semantics (equivalent to
``mx.matmul(A, B)``), no ``_q_bf16`` wrapping. Phases 2/3 layer the bf16
round-trips and the other four matmuls on top.

Hardware gate (plan §8, user decision #3)
-----------------------------------------
Only M3+ runs the Metal path. M1/M2 deterministically skip — no opt-in.
Detected at import via ``mx.device_info()``; result cached on the module.

Kernel caching
--------------
``mx.fast.metal_kernel`` keys on ``(source, header)``. We use
``template`` for shape specialization rather than f-string source
interpolation so the same source+header pair serves all ``(H, D, CHUNK)``
triples (MLX caches specialized compilations, see mlx-lm's
``gated_delta_kernel`` for the pattern).

Architecture reference
----------------------
``mlx/include/mlx/backend/metal/kernels/steel/gemm/mma.h`` documents the
subset of ``<metal_simdgroup_matrix>`` that's known to work under MLX's
driver: only ``simdgroup_matrix<float, 8, 8>`` and ``<half, 8, 8>``,
``simdgroup_multiply_accumulate(D, A, B, C)`` as in-place
``D = A*B + C``. We use ``float`` throughout (bf16 input values cast to
fp32 on simdgroup load — same fp32 accumulation semantics as ``mx.matmul``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _device_name() -> str:
    """Return the device name from the new or deprecated API."""
    try:
        info = mx.device_info()  # preferred (MLX ≥ 0.31.x)
    except AttributeError:
        try:
            info = mx.metal.device_info()  # deprecated fallback
        except Exception:  # noqa: BLE001
            return ""
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(info, dict):
        return str(info.get("device_name", ""))
    return ""


def _probe_m3_or_newer() -> bool:
    """Return True only if we're running on Apple M3 or newer.

    M1/M2 are excluded per plan decision #3 (no opt-in fallback for them).
    The plan also excludes non-Apple Metal devices; this probe returns
    False for any unrecognized name, which conservatively disables the
    Metal path rather than risking a crash on surprise hardware.
    """
    name = _device_name()
    if not name:
        return False
    # Match "Apple M3", "Apple M4", ..., up to M9 defensively. Rejects
    # "Apple M1", "Apple M2", "Apple M1 Max", etc.
    for family_num in range(3, 10):
        if f"Apple M{family_num}" in name:
            return True
    return False


# Force-override via env var, primarily for testing the disabled path on
# M3+ hardware without physical M1/M2 access.
_FORCE_FALLBACK = bool(int(os.environ.get("MLX_KDA_FORCE_METAL_FALLBACK", "0") or "0"))

HAS_METAL_KERNEL: bool = (not _FORCE_FALLBACK) and _probe_m3_or_newer()


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------
#
# Phase 1 matmul kernel. Dispatch geometry:
#
#   grid        = (1024, H, 1)        # 1024 threads × H heads
#   threadgroup = (1024, 1, 1)        # 32 simdgroups per threadgroup
#
# One threadgroup per head. 32 simdgroups cooperate on 32 output tiles
# (CHUNK/8 × D/8 = 2 × 16 = 32). Each simdgroup computes one 8×8 output
# tile via a K-loop of D/8 = 16 MMAs.

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""


_MATMUL_SOURCE = """
    // Grid contract:
    //   threadgroup_position_in_grid.y = head_id  (0..H-1)
    //   threads_per_threadgroup.x      = 1024     (32 simdgroups)
    //
    // Each simdgroup owns one 8x8 output tile.
    //   simd_id in [0, 32)
    //   tile_r = simd_id / (D/8)  in {0, 1}
    //   tile_c = simd_id % (D/8)  in [0, D/8)

    uint head_id = threadgroup_position_in_grid.y;
    uint simd_id = simdgroup_index_in_threadgroup;

    // Compile-time per-shape constants (substituted via template params):
    //   H, CHUNK, D are Metal template parameters baked at JIT time.
    constexpr uint CHUNK_TILES = CHUNK / 8;   // 2 for CHUNK=16
    constexpr uint D_TILES     = D / 8;       // 16 for D=128
    constexpr uint K_TILES     = D / 8;       // K = D for this matmul

    uint tile_r = simd_id / D_TILES;  // row-block of CHUNK
    uint tile_c = simd_id % D_TILES;  // col-block of D

    // Bounds: we have exactly CHUNK_TILES * D_TILES = 32 output tiles and
    // 32 simdgroups per threadgroup, so every simdgroup has work. No
    // guard needed for the common case; assert at kernel-build time.
    static_assert(CHUNK_TILES * D_TILES <= 32,
                  "Phase 1 kernel assumes one simdgroup per output tile");

    // Per-head strided base pointers.
    const device float* A_head = A + head_id * CHUNK * D;   // [CHUNK, D]
    const device float* B_head = B + head_id * D * D;       // [D, D]
    device float*       C_head = C + head_id * CHUNK * D;   // [CHUNK, D]

    // Accumulator tile (fp32).
    simdgroup_matrix<float, 8, 8> C_tile;
    C_tile = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // K-loop over D in 8-wide tiles.
    for (uint k_tile = 0; k_tile < K_TILES; ++k_tile) {
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;

        // Load A[tile_r*8 : tile_r*8+8, k_tile*8 : k_tile*8+8]
        // Source pointer is pre-offset to the top-left of the 8x8 tile.
        const device float* A_src =
            A_head + (tile_r * 8) * D + (k_tile * 8);
        simdgroup_load(A_tile, A_src, D);

        // Load B[k_tile*8 : k_tile*8+8, tile_c*8 : tile_c*8+8]
        const device float* B_src =
            B_head + (k_tile * 8) * D + (tile_c * 8);
        simdgroup_load(B_tile, B_src, D);

        // C_tile += A_tile @ B_tile
        simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
    }

    // Store C_tile to C[tile_r*8 : +8, tile_c*8 : +8]
    device float* C_dst =
        C_head + (tile_r * 8) * D + (tile_c * 8);
    simdgroup_store(C_tile, C_dst, D);
"""


# ---------------------------------------------------------------------------
# Kernel-factory cache
# ---------------------------------------------------------------------------
#
# MLX compiles the shader on first dispatch for each unique
# (name, source, header) triple AND (per mlx-lm observation) specializes
# further per ``template`` combination. We cache the factory object per
# (H, D, CHUNK) so Python-side dispatch overhead stays minimal.

@lru_cache(maxsize=16)
def _build_matmul_kernel(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template at call time; kernel is generic
    return mx.fast.metal_kernel(
        name="flash_kda_phase1_matmul",
        input_names=["A", "B"],
        output_names=["C"],
        header=_HEADER,
        source=_MATMUL_SOURCE,
        ensure_row_contiguous=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def metal_matmul_A_by_B(A: mx.array, B: mx.array) -> mx.array:
    """Compute ``A @ B`` batched over the leading H axis.

    Shapes:
      A : ``[H, CHUNK, D]`` fp32
      B : ``[H, D, D]`` fp32
      returns ``[H, CHUNK, D]`` fp32
    """
    assert HAS_METAL_KERNEL, (
        "metal_matmul_A_by_B called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    assert A.dtype == mx.float32 and B.dtype == mx.float32
    assert A.ndim == 3 and B.ndim == 3
    H, CHUNK, D_a = A.shape
    Hb, D_b, D_out = B.shape
    assert H == Hb and D_a == D_b == D_out, (
        f"shape mismatch: A={A.shape} B={B.shape}"
    )

    kernel = _build_matmul_kernel(H=H, D=D_a, CHUNK=CHUNK)

    outputs = kernel(
        inputs=[A, B],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D_a)],
        grid=(1024, H, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(H, CHUNK, D_a)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def warmup_matmul_shapes(shapes: list[tuple[int, int, int]]) -> None:
    """JIT-compile kernels for each ``(H, CHUNK, D)`` shape ahead of timing.

    Phase 1 compile cost can be 500ms–2s per shape on first dispatch;
    this helper runs a zero-input call for each shape so the bench
    timing doesn't absorb that hit.
    """
    if not HAS_METAL_KERNEL:
        return
    for H, CHUNK, D in shapes:
        A = mx.zeros((H, CHUNK, D), dtype=mx.float32)
        B = mx.zeros((H, D, D), dtype=mx.float32)
        out = metal_matmul_A_by_B(A, B)
        mx.eval(out)


# ---------------------------------------------------------------------------
# Phase 2: fused per-chunk body kernel (all 5 matmuls + bf16 round-trips)
# ---------------------------------------------------------------------------
#
# Dispatch geometry:
#   grid        = (1024, H, 1)
#   threadgroup = (1024, 1, 1)       # 32 simdgroups per head
#
# One threadgroup per head. 32 simdgroups cooperate on:
#   - step 0: load state_bf_T into threadgroup memory (bf16, 32 KB)
#   - step 1: matmul #1 k_decayed @ state_bf_T → simdgroup tiles →
#             vdiff_sm [CHUNK, D] after bf16 round-trips (matches Python
#             ``_recurrence_body_single`` lines 196-198)
#   - step 2: matmul #2 INV_bf @ vdiff → U_sm [CHUNK, D]
#   - step 3+4 fused: matmul #3 q_decayed @ state_bf_T and matmul #4
#             Mqk @ U per simdgroup tile; combine and write out_h
#   - step 5: matmul #5 U^T @ k_restored → delta_s^T into new_state
#             (the U^T trick avoids writing-then-transposing)
#   - step 6: per-thread state update:
#             new_state[a, b] = q_bf(new_state[a, b]
#                                    + q_bf(state[a, b]) * g_total_exp[b])
#
# Total threadgroup memory at H_*=1, CHUNK=16, D=128:
#   state_bf_T_sm: D*D*2 = 32768 B = 32 KB
#   vdiff_sm     : CHUNK*D*4 = 8192 B = 8 KB
#   U_sm         : CHUNK*D*4 = 8192 B = 8 KB
#                                        ----
#                                        48 KB   (fits in M3's 64 KB)

_RECURRENCE_SOURCE = """
    // Template params (MLX substitutes at JIT):
    //   H, CHUNK, D — int constants.
    constexpr uint CHUNK_TILES = CHUNK / 8;    // 2 for CHUNK=16
    constexpr uint D_TILES     = D / 8;        // 16 for D=128
    constexpr uint K_D_TILES   = D / 8;        // K=D matmuls
    constexpr uint K_C_TILES   = CHUNK / 8;    // K=CHUNK matmuls
    constexpr uint OUT_TILES_BODY = CHUNK_TILES * D_TILES;  // 32

    // state_slice is already bf16-valued fp32 on input (cast at
    // fwd_optimized.py:634 on initial_state, preserved by prior body
    // iterations). The Python reference's ``state_bf = _q_bf16(state_slice)``
    // is numerically a no-op on those values, so we can skip bf16 staging
    // and load state directly from device memory. simdgroup_load with
    // transpose=true serves matmuls #1 and #3 (both consume state_bf_T).
    threadgroup float  vdiff_sm[CHUNK * D];     // 8 KB — vdiff then qs stage
    threadgroup float  U_sm[CHUNK * D];         // 8 KB — U (needed thru step 5)
    threadgroup float  scratch_sm[CHUNK * D];   // 8 KB — mu stage

    uint head_id = threadgroup_position_in_grid.y;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint tid     = thread_position_in_threadgroup.x;
    uint tpg     = threads_per_threadgroup.x;

    const uint state_base  = head_id * D * D;
    const uint chunkD_base = head_id * CHUNK * D;
    const uint chunkC_base = head_id * CHUNK * CHUNK;
    const uint beta_base   = head_id * CHUNK;
    const uint gte_base    = head_id * D;

    const device float* k_dec_head  = k_decayed  + chunkD_base;
    const device float* q_dec_head  = q_decayed  + chunkD_base;
    const device float* k_rest_head = k_restored + chunkD_base;
    const device float* Mqk_head    = Mqk        + chunkC_base;
    const device float* INV_head    = INV_bf     + chunkC_base;
    // ``vc`` is the caller-provided ``v`` passthrough; under
    // PR H follow-on 3 Phase B it can be either fp32 or bf16. We avoid a
    // typed local pointer (which would force a fixed dtype) and instead
    // index ``vc`` inline below — Metal's implicit ``bfloat→float``
    // conversion handles both dtypes equivalently.
    const uint vc_off               = chunkD_base;
    const device float* beta_head   = beta_bf16  + beta_base;
    const device float* gte_head    = g_total_exp + gte_base;
    const device float* state_head  = state_slice + state_base;
    device float*       out_head    = out_h      + chunkD_base;
    device float*       ns_head     = new_state  + state_base;

    // ======================================================================
    // Step 1: Matmul #1 — partial = k_decayed @ state_bf_T via simdgroup.
    //         Then: vdiff = q_bf(q_bf(vc - q_bf(partial)) * beta)
    //         Output goes to vdiff_sm.
    //
    // state_bf_T is state^T (last-two transpose). To load the 8x8 tile
    // at state_bf_T[k_tile*8..+8, tile_c*8..+8], we load from state at
    // offset [(tile_c*8)*D + k_tile*8] with transpose=true. That gives
    // us the semantic tile where local (r, c) maps to state[tile_c*8+c,
    // k_tile*8+r] — exactly state_bf_T[k_tile*8+r, tile_c*8+c].
    //
    // Per-simdgroup tile: (tile_r, tile_c) in [0, CHUNK_TILES) × [0, D_TILES).
    // 32 simdgroups; exactly 32 tiles so each simd owns one.
    // ======================================================================
    {
        uint tile_r = simd_id / D_TILES;
        uint tile_c = simd_id % D_TILES;

        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> C_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            // A = k_decayed[tile_r*8..+8, k_tile*8..+8]
            const device float* A_src =
                k_dec_head + (tile_r * 8) * D + (k_tile * 8);
            simdgroup_load(A_tile, A_src, D);

            // B = state_bf_T[k_tile*8..+8, tile_c*8..+8]
            //   == state[tile_c*8..+8, k_tile*8..+8]^T (load + transpose)
            const device float* B_src =
                state_head + (tile_c * 8) * D + (k_tile * 8);
            simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
        }

        // Store the raw partial into vdiff_sm first — post-process in step 1b.
        threadgroup float* C_dst =
            &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
        simdgroup_store(C_tile, C_dst, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1b: per-thread post-process vdiff
    //   vdiff[i, j] = q_bf(q_bf(vc[i, j] - q_bf(partial[i, j])) * beta[i])
    for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
        uint i = idx / D;
        float p = vdiff_sm[idx];                      // partial[i, j]
        float v = float(vc[vc_off + idx]);            // bf16-or-fp32 read
        float t = float(bfloat(v - float(bfloat(p))));
        float b = beta_head[i];
        vdiff_sm[idx] = float(bfloat(t * b));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ======================================================================
    // Step 2: Matmul #2 — U = q_bf(INV_bf @ vdiff).
    //   Both inputs are small; K=CHUNK=16, 2 K-tiles. Use scalar inner product
    //   since simdgroup_matrix doesn't gain much for tiny K.
    //   (32 simdgroups = 32 output tiles, one per simd; matches [CHUNK, D].)
    //
    //   U[i, j] = sum_k INV_bf[i, k] * vdiff[k, j]
    // ======================================================================
    for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
        uint i = idx / D;
        uint j = idx % D;
        float acc = 0.0f;
        for (uint k = 0; k < CHUNK; ++k) {
            float iv = INV_head[i * CHUNK + k];
            float vv = vdiff_sm[k * D + j];
            acc += iv * vv;
        }
        U_sm[idx] = float(bfloat(acc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ======================================================================
    // Step 3+4: matmul #3 + matmul #4 + combine, all tile-by-tile.
    //   qs_tile  = q_decayed @ state_bf_T     // matmul #3 (simdgroup)
    //   mu_tile  = Mqk @ U                     // matmul #4 (simdgroup for D axis; scalar over CHUNK K is 2 tiles)
    //   out_tile = q_bf(q_bf(qs_tile) + q_bf(mu_tile))
    //   simdgroup_store out_tile to device.
    // ======================================================================
    {
        uint tile_r = simd_id / D_TILES;
        uint tile_c = simd_id % D_TILES;

        // Matmul #3: same shape as #1, same state_bf_T_sm operand.
        simdgroup_matrix<float, 8, 8> A_tile;
        simdgroup_matrix<float, 8, 8> B_tile;
        simdgroup_matrix<float, 8, 8> qs_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
            const device float* A_src =
                q_dec_head + (tile_r * 8) * D + (k_tile * 8);
            simdgroup_load(A_tile, A_src, D);
            // B = state_bf_T[k_tile*8..+8, tile_c*8..+8] via transposed load.
            const device float* B_src =
                state_head + (tile_c * 8) * D + (k_tile * 8);
            simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(qs_tile, A_tile, B_tile, qs_tile);
        }

        // Matmul #4 for the same tile: Mqk @ U. K=CHUNK=16, K_C_TILES=2.
        simdgroup_matrix<float, 8, 8> mu_tile =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
            const device float* A_src =
                Mqk_head + (tile_r * 8) * CHUNK + (k_tile * 8);
            simdgroup_load(A_tile, A_src, CHUNK);
            const threadgroup float* B_src =
                &U_sm[(k_tile * 8) * D + (tile_c * 8)];
            simdgroup_load(B_tile, B_src, D);
            simdgroup_multiply_accumulate(mu_tile, A_tile, B_tile, mu_tile);
        }

        // Stage qs_tile and mu_tile to TG memory so we can combine them
        // per-thread with the bf16 round-trips the Python reference
        // applies (``thread_elements``-based tile mutation is ABI-sensitive;
        // TG-staged per-thread combine is correct by construction).
        //   vdiff_sm was consumed by step 2 → safe to reuse as qs stage.
        //   U_sm must live through step 5 → use scratch_sm for mu.
        threadgroup float* qs_dst =
            &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
        simdgroup_store(qs_tile, qs_dst, D);
        threadgroup float* mu_dst =
            &scratch_sm[(tile_r * 8) * D + (tile_c * 8)];
        simdgroup_store(mu_tile, mu_dst, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3+4 combine: per-thread, out_h = q_bf(q_bf(qs) + q_bf(mu)).
    for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
        float q_val = float(bfloat(vdiff_sm[idx]));    // q_bf(qs)
        float m_val = float(bfloat(scratch_sm[idx]));  // q_bf(mu)
        out_head[idx] = float(bfloat(q_val + m_val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ======================================================================
    // Step 5: Matmul #5 — delta_s^T = U^T @ k_restored writes into new_state.
    //   Output shape [D, D]: 256 output tiles. 32 simdgroups → 8 tiles each.
    //   We compute the TRANSPOSE of delta_s (which simplifies step 6):
    //     new_state_pre[a, b] := delta_s_T[a, b] = sum_k U[k, a] * k_restored[k, b]
    //   i.e., new_state_pre = U^T @ k_restored. Achieved via simdgroup_load
    //   of U with transpose=true.
    // ======================================================================
    {
        for (uint tile_batch = 0; tile_batch < 8; ++tile_batch) {
            uint tile_idx = simd_id * 8 + tile_batch;
            uint tile_a = tile_idx / D_TILES;  // 0..D_TILES-1
            uint tile_b = tile_idx % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> C_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                // A = U^T[tile_a*8..+8, k_tile*8..+8]
                //   == U[k_tile*8..+8, tile_a*8..+8] loaded with transpose=true
                const threadgroup float* A_src =
                    &U_sm[(k_tile * 8) * D + (tile_a * 8)];
                simdgroup_load(A_tile, A_src, D, ulong2(0, 0), true);

                // B = k_restored[k_tile*8..+8, tile_b*8..+8]
                const device float* B_src =
                    k_rest_head + (k_tile * 8) * D + (tile_b * 8);
                simdgroup_load(B_tile, B_src, D);

                simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
            }

            // Store raw delta_s_T tile to new_state device memory.
            device float* C_dst =
                ns_head + (tile_a * 8) * D + (tile_b * 8);
            simdgroup_store(C_tile, C_dst, D);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ======================================================================
    // Step 6: per-thread state update in place.
    //   new_state[a, b] = q_bf(new_state[a, b]
    //                          + q_bf(state[a, b]) * g_total_exp[b])
    // ======================================================================
    for (uint idx = tid; idx < D * D; idx += tpg) {
        uint a = idx / D;
        uint b = idx % D;
        float pre = ns_head[idx];                          // delta_s_T[a, b]
        float stv = float(bfloat(state_head[a * D + b]));  // q_bf(state[a, b])
        float gte = gte_head[b];                           // g_total_exp[b, 0]
        ns_head[idx] = float(bfloat(pre + stv * gte));
    }
"""


@lru_cache(maxsize=16)
def _build_recurrence_kernel(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template at call time
    return mx.fast.metal_kernel(
        name="flash_kda_phase2_recurrence",
        input_names=[
            "state_slice",
            "k_decayed",
            "q_decayed",
            "k_restored",
            "Mqk",
            "INV_bf",
            "vc",
            "beta_bf16",
            "g_total_exp",
        ],
        output_names=["out_h", "new_state"],
        header=_HEADER,
        source=_RECURRENCE_SOURCE,
        ensure_row_contiguous=True,
    )


def metal_recurrence_body_single(
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
    """Phase 2 Metal kernel drop-in for ``_recurrence_body_single``.

    Shapes & dtypes must match the Python reference exactly:
      state_slice : [H, D, D] fp32 (bf16-valued)
      k_decayed / q_decayed / k_restored / vc : [H, CHUNK, D] fp32
      Mqk / INV_bf : [H, CHUNK, CHUNK] fp32
      beta_bf16 : [H, CHUNK, 1] fp32
      g_total_exp : [H, D, 1] fp32

    Returns ``(out_h, new_state)`` both fp32 with bf16-valued bits.
    """
    assert HAS_METAL_KERNEL, (
        "metal_recurrence_body_single called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    H = state_slice.shape[0]
    D = state_slice.shape[1]
    CHUNK = k_decayed.shape[1]

    kernel = _build_recurrence_kernel(H=H, D=D, CHUNK=CHUNK)

    out_h, new_state = kernel(
        inputs=[
            state_slice, k_decayed, q_decayed, k_restored,
            Mqk, INV_bf, vc, beta_bf16, g_total_exp,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(1024, H, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(H, CHUNK, D), (H, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out_h, new_state


def warmup_recurrence_shapes(shapes: list[tuple[int, int, int]]) -> None:
    """JIT-compile the Phase 2 kernel for each shape."""
    if not HAS_METAL_KERNEL:
        return
    for H, CHUNK, D in shapes:
        state = mx.zeros((H, D, D), dtype=mx.float32)
        k_dec = mx.zeros((H, CHUNK, D), dtype=mx.float32)
        q_dec = mx.zeros((H, CHUNK, D), dtype=mx.float32)
        k_rest = mx.zeros((H, CHUNK, D), dtype=mx.float32)
        Mqk = mx.zeros((H, CHUNK, CHUNK), dtype=mx.float32)
        INV = mx.zeros((H, CHUNK, CHUNK), dtype=mx.float32)
        vc = mx.zeros((H, CHUNK, D), dtype=mx.float32)
        beta = mx.zeros((H, CHUNK, 1), dtype=mx.float32)
        gte = mx.zeros((H, D, 1), dtype=mx.float32)
        out_h, ns = metal_recurrence_body_single(
            state, k_dec, q_dec, k_rest, Mqk, INV, vc, beta, gte,
        )
        mx.eval(out_h, ns)


# ---------------------------------------------------------------------------
# Phase 3a: cross-chunk scalar kernel
# ---------------------------------------------------------------------------
#
# Collapses the per-chunk Python dispatch loop into ONE Metal kernel per
# forward. State lives in threadgroup memory across all chunks for a head;
# the chunk loop iterates inside the shader.
#
# Architectural win vs Phase 2: state round-trip to device memory per chunk
# is eliminated. At T=8192, n_chunks=512, D=128 that saves ~2 GB of
# state-related device traffic per forward.
#
# Dispatch geometry:
#   grid        = (N_THREADS, H, 1)
#   threadgroup = (N_THREADS, 1, 1)     # 32 simdgroups per head
#
# Threadgroup memory budget (HARD 32 KB limit on M3, not the 64 KB the
# initial plan assumed — verified at runtime). Only state fits:
#
#   state_tg  : D*D × bf16 = 32 KB     (exactly at the limit)
#
# vdiff and U must spill to device memory. We reuse the ``new_state``
# output buffer as per-head scratch: its first ``2 × CHUNK × D = 16 KB``
# hosts vdiff then U during the chunk loop. After the last chunk,
# ``state_tg`` is written back over the whole 64 KB of ``new_state``, so
# the scratch region is overwritten before the kernel returns.
#
# The added device R/W per chunk is 3 × 8 KB (vdiff write+read, U
# write+read, out_h write) = 32 KB vs Phase 2's ~128 KB per-chunk state
# round-trip. Net: ~4× less device memory pressure per chunk.
#
# Scalar reduction strategy (mirrors mlx-lm's gated_delta_kernel): each
# thread owns one output element and executes the inner-k loop serially;
# no simd_sum required because each (i, j) cell is a sequential dot
# product, not a parallel reduction across threads.

_CROSS_CHUNK_SCALAR_N_THREADS = 1024


_CROSS_CHUNK_SCALAR_SOURCE = """
    // Template params: H, CHUNK, D are int constants.
    constexpr uint N_THREADS = 1024;
    constexpr uint CHUNK_D   = CHUNK * D;     // 2048 at CHUNK=16, D=128
    constexpr uint D_D       = D * D;         // 16384 at D=128

    // n_chunks is a runtime scalar: we don't specialize per-n_chunks because
    // varying T would cause repeated recompiles. Passed as a 1-element
    // ``const device int*``; read once at kernel entry.
    const uint n_chunks = uint(n_chunks_ptr[0]);

    uint head_id = threadgroup_position_in_grid.y;
    uint tid     = thread_position_in_threadgroup.x;

    // Threadgroup memory — state ONLY; vdiff/U live in device scratch.
    threadgroup bfloat state_tg[D * D];        // 32 KB (at the limit)

    // Per-head device base pointers. The per-chunk offsets are computed
    // inside the chunk loop.
    const device float* state_head = state_slice + head_id * D_D;
    device float*       ns_head    = new_state   + head_id * D_D;

    // Reuse the new_state output as per-head scratch during the chunk
    // loop. Layout within ns_head:
    //   [0,         CHUNK_D)      : vdiff scratch
    //   [CHUNK_D,   2*CHUNK_D)    : U scratch
    //   [2*CHUNK_D, D_D)          : unused during loop
    // After the loop, the WHOLE ns_head region is overwritten with the
    // final state_tg data, so scratch contamination is immaterial.
    device float* vdiff_buf = ns_head;
    device float* U_buf     = ns_head + CHUNK_D;

    // --------------------------------------------------------------------
    // Initialize TG state from device memory. state_slice is already
    // bf16-valued fp32 (caller contract, matches Phase 2). Storing as
    // bfloat preserves the bits exactly.
    // --------------------------------------------------------------------
    for (uint i = tid; i < D_D; i += N_THREADS) {
        state_tg[i] = bfloat(state_head[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --------------------------------------------------------------------
    // Chunk loop. Each iteration computes one chunk's out_h + updates
    // state_tg in place.
    // --------------------------------------------------------------------
    for (uint c = 0; c < n_chunks; ++c) {
        // Per-chunk, per-head base pointers. Input layout is
        // [n_chunks, H, CHUNK, D] for CHUNK*D tensors, etc.
        const device float* kdec_cb  =
            k_decayed   + c * H * CHUNK * D     + head_id * CHUNK * D;
        const device float* qdec_cb  =
            q_decayed   + c * H * CHUNK * D     + head_id * CHUNK * D;
        const device float* krest_cb =
            k_restored  + c * H * CHUNK * D     + head_id * CHUNK * D;
        // ``vc`` may be fp32 or bf16 (PR H follow-on 3 Phase B); index inline
        // to avoid forcing a typed local pointer.
        const uint vc_off            =
            c * H * CHUNK * D + head_id * CHUNK * D;
        const device float* Mqk_cb   =
            Mqk         + c * H * CHUNK * CHUNK + head_id * CHUNK * CHUNK;
        const device float* INV_cb   =
            INV_bf      + c * H * CHUNK * CHUNK + head_id * CHUNK * CHUNK;
        const device float* beta_cb  =
            beta_bf16   + c * H * CHUNK         + head_id * CHUNK;
        const device float* gte_cb   =
            g_total_exp + c * H * D             + head_id * D;
        // ``out_h`` layout is [T_total, H, D] (= [n_chunks*CHUNK, H, D])
        // — written directly so the caller can skip a transpose+reshape.
        // Per-chunk-row stride is H*D (PR H follow-on 3 Phase A).
        device float*       out_cb   =
            out_h       + c * CHUNK * H * D     + head_id * D;

        // ----------------------------------------------------------------
        // Step 1: vdiff[i, j] = q_bf(q_bf(vc[i,j] - q_bf(partial[i,j])) * beta[i])
        //   partial[i, j] = sum_k k_decayed[i, k] * state_bf_T[k, j]
        //                 = sum_k k_decayed[i, k] * state[j, k]
        //
        // Each thread owns 2 output cells (CHUNK_D / N_THREADS = 2). vdiff
        // is staged in device scratch (vdiff_buf = ns_head prefix).
        // ----------------------------------------------------------------
        for (uint idx = tid; idx < CHUNK_D; idx += N_THREADS) {
            uint i = idx / D;
            uint j = idx % D;
            float acc = 0.0f;
            for (uint k = 0; k < D; ++k) {
                acc += kdec_cb[i * D + k] * float(state_tg[j * D + k]);
            }
            float v = float(vc[vc_off + idx]);  // bf16-or-fp32 read
            float t = float(bfloat(v - float(bfloat(acc))));
            float b = beta_cb[i];
            vdiff_buf[idx] = float(bfloat(t * b));
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ----------------------------------------------------------------
        // Step 2: U[i, j] = q_bf(sum_k INV_bf[i, k] * vdiff[k, j])
        //   K=CHUNK=16, small inner loop.
        // ----------------------------------------------------------------
        for (uint idx = tid; idx < CHUNK_D; idx += N_THREADS) {
            uint i = idx / D;
            uint j = idx % D;
            float acc = 0.0f;
            for (uint k = 0; k < CHUNK; ++k) {
                acc += INV_cb[i * CHUNK + k] * vdiff_buf[k * D + j];
            }
            U_buf[idx] = float(bfloat(acc));
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ----------------------------------------------------------------
        // Step 3+4: out_h[i, j] = q_bf(q_bf(qs[i,j]) + q_bf(mu[i,j]))
        //   qs[i, j] = sum_k q_decayed[i, k] * state[j, k]
        //   mu[i, j] = sum_k Mqk[i, k] * U[k, j]
        // ----------------------------------------------------------------
        for (uint idx = tid; idx < CHUNK_D; idx += N_THREADS) {
            uint i = idx / D;
            uint j = idx % D;
            float qs = 0.0f;
            for (uint k = 0; k < D; ++k) {
                qs += qdec_cb[i * D + k] * float(state_tg[j * D + k]);
            }
            float mu = 0.0f;
            for (uint k = 0; k < CHUNK; ++k) {
                mu += Mqk_cb[i * CHUNK + k] * U_buf[k * D + j];
            }
            // Direct-write to [T_total, H, D] layout: stride H*D per row.
            out_cb[i * H * D + j] =
                float(bfloat(float(bfloat(qs)) + float(bfloat(mu))));
        }
        // No barrier needed before step 5: step 3+4 only writes to out_cb
        // (device, not used by step 5), and reads state_tg + U_buf (step 5
        // reads state_tg + U_buf; no write-after-read hazard yet since
        // step 5 writes only state_tg, which step 3+4 read already).
        //
        // BUT: step 5 writes state_tg while other threads in the same
        // threadgroup may still be executing step 3+4's reads of
        // state_tg. We need a threadgroup-wide sync.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ----------------------------------------------------------------
        // Step 5+6: state[a, b] = q_bf(delta_s_T[a, b] + state[a, b] * gte[b])
        //   delta_s_T[a, b] = sum_k k_restored[k, b] * U[k, a]
        //
        // Each thread updates D_D / N_THREADS = 16 state cells. Read-before-
        // write within the same thread is race-free (same cell, same thread).
        // ----------------------------------------------------------------
        for (uint idx = tid; idx < D_D; idx += N_THREADS) {
            uint a = idx / D;
            uint b = idx % D;
            float ds = 0.0f;
            for (uint k = 0; k < CHUNK; ++k) {
                ds += krest_cb[k * D + b] * U_buf[k * D + a];
            }
            float stv = float(state_tg[idx]);
            float gte = gte_cb[b];
            state_tg[idx] = bfloat(ds + stv * gte);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --------------------------------------------------------------------
    // Write final state back to device memory. This ALSO overwrites the
    // vdiff_buf / U_buf scratch regions within ns_head with state data.
    // --------------------------------------------------------------------
    for (uint i = tid; i < D_D; i += N_THREADS) {
        ns_head[i] = float(state_tg[i]);
    }
"""


@lru_cache(maxsize=16)
def _build_cross_chunk_scalar_kernel(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template at call time
    return mx.fast.metal_kernel(
        name="flash_kda_phase3a_cross_chunk_scalar",
        input_names=[
            "state_slice",
            "k_decayed",
            "q_decayed",
            "k_restored",
            "Mqk",
            "INV_bf",
            "vc",
            "beta_bf16",
            "g_total_exp",
            "n_chunks_ptr",
        ],
        output_names=["out_h", "new_state"],
        header=_HEADER,
        source=_CROSS_CHUNK_SCALAR_SOURCE,
        ensure_row_contiguous=True,
    )


def metal_recurrence_cross_chunk_scalar(
    state_in: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
) -> tuple[mx.array, mx.array]:
    """Phase 3a cross-chunk scalar Metal kernel.

    One launch per forward; the chunk loop runs inside the shader with
    state held in threadgroup memory (bf16, 32 KB per head).

    Shapes & dtypes:
      state_in     : [H, D, D]                   fp32 (bf16-valued)
      k_decayed    : [n_chunks, H, CHUNK, D]     fp32
      q_decayed    : [n_chunks, H, CHUNK, D]     fp32
      k_restored   : [n_chunks, H, CHUNK, D]     fp32
      Mqk          : [n_chunks, H, CHUNK, CHUNK] fp32
      INV_bf       : [n_chunks, H, CHUNK, CHUNK] fp32
      vc           : [n_chunks, H, CHUNK, D]     fp32
      beta_bf16    : [n_chunks, H, CHUNK, 1]     fp32
      g_total_exp  : [n_chunks, H, D, 1]         fp32

    Returns:
      out_h        : [n_chunks, H, CHUNK, D]     fp32 (bf16-valued)
      new_state    : [H, D, D]                   fp32 (bf16-valued)

    Only CHUNK=16, D=128 are supported (Phase 3 ships the bench-scale
    configuration; other shapes assert). M3+ only; gate upstream with
    ``HAS_METAL_KERNEL``.
    """
    assert HAS_METAL_KERNEL, (
        "metal_recurrence_cross_chunk_scalar called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    assert state_in.dtype == mx.float32
    assert k_decayed.dtype == mx.float32
    assert q_decayed.dtype == mx.float32
    assert k_restored.dtype == mx.float32
    assert Mqk.dtype == mx.float32
    assert INV_bf.dtype == mx.float32
    # vc passthrough may be bf16 under fused/fused2 (PR H follow-on 3 Phase B).
    assert vc.dtype in (mx.float32, mx.bfloat16)
    assert beta_bf16.dtype == mx.float32
    assert g_total_exp.dtype == mx.float32

    H = state_in.shape[0]
    D = state_in.shape[1]
    n_chunks = k_decayed.shape[0]
    CHUNK = k_decayed.shape[2]

    assert state_in.shape == (H, D, D)
    assert k_decayed.shape == (n_chunks, H, CHUNK, D)
    assert q_decayed.shape == (n_chunks, H, CHUNK, D)
    assert k_restored.shape == (n_chunks, H, CHUNK, D)
    assert vc.shape == (n_chunks, H, CHUNK, D)
    assert Mqk.shape == (n_chunks, H, CHUNK, CHUNK)
    assert INV_bf.shape == (n_chunks, H, CHUNK, CHUNK)
    assert beta_bf16.shape == (n_chunks, H, CHUNK, 1)
    assert g_total_exp.shape == (n_chunks, H, D, 1)
    # Phase 3 is specialized to CHUNK=16 (the simdgroup tile geometry in
    # Phase 3b depends on it). Callers route CHUNK≠16 through mx.compile.
    assert CHUNK == 16, f"Phase 3 requires CHUNK=16, got {CHUNK}"
    assert D == 128, f"Phase 3 requires D=128, got {D}"

    n_chunks_arr = mx.array([n_chunks], dtype=mx.int32)

    kernel = _build_cross_chunk_scalar_kernel(H=H, D=D, CHUNK=CHUNK)
    out_h, new_state = kernel(
        inputs=[
            state_in, k_decayed, q_decayed, k_restored,
            Mqk, INV_bf, vc, beta_bf16, g_total_exp,
            n_chunks_arr,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(_CROSS_CHUNK_SCALAR_N_THREADS, H, 1),
        threadgroup=(_CROSS_CHUNK_SCALAR_N_THREADS, 1, 1),
        output_shapes=[(n_chunks * CHUNK, H, D), (H, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out_h, new_state


def warmup_cross_chunk_scalar_shapes(
    shapes: list[tuple[int, int, int, int]],
) -> None:
    """Pre-compile the Phase 3a kernel for each ``(H, CHUNK, D, n_chunks)``.

    First dispatch of a unique ``(H, CHUNK, D)`` triple compiles the kernel
    (template-parameterized); ``n_chunks`` is runtime and doesn't need
    warming. We only need one ``n_chunks`` per ``(H, CHUNK, D)`` to cover
    the compile. Pass a representative n_chunks≥1 for each unique triple.
    """
    if not HAS_METAL_KERNEL:
        return
    for H, CHUNK, D, n_chunks in shapes:
        state = mx.zeros((H, D, D), dtype=mx.float32)
        k_dec = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        q_dec = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        k_rest = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        Mqk = mx.zeros((n_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        INV = mx.zeros((n_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        vc = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        beta = mx.zeros((n_chunks, H, CHUNK, 1), dtype=mx.float32)
        gte = mx.zeros((n_chunks, H, D, 1), dtype=mx.float32)
        out_h, ns = metal_recurrence_cross_chunk_scalar(
            state, k_dec, q_dec, k_rest, Mqk, INV, vc, beta, gte,
        )
        mx.eval(out_h, ns)


# ---------------------------------------------------------------------------
# Phase 3b: cross-chunk simdgroup kernel
# ---------------------------------------------------------------------------
#
# Combines Phase 2's simdgroup_matrix MMA efficiency with Phase 3a's
# architectural change (chunk loop lives inside the kernel). State stays
# in DEVICE memory — the 32 KB threadgroup budget is too tight to hold
# state AND the three Phase-2 TG staging buffers (32+8+8+8 = 56 KB > 32).
#
# Key trick: after ``new_state = state_slice`` seed, the same output buffer
# ``new_state`` doubles as the per-chunk state source for subsequent chunks.
# matmul #5's delta_s_T is routed to a separate ``delta_scratch`` device
# output so step 6 can read the pre-update state from ``new_state`` while
# delta_s_T is staged elsewhere. ``delta_scratch`` is unused after the
# kernel returns.
#
# Dispatch geometry (same as Phase 2):
#   grid        = (1024, H, 1)
#   threadgroup = (1024, 1, 1)         # 32 simdgroups per head
#
# TG memory (same as Phase 2):
#   vdiff_sm, U_sm, scratch_sm         # 3 × 8 KB = 24 KB
#
# What this wins over Phase 2: one kernel launch per forward instead of
# n_chunks per-head dispatches. Saves the MLX graph-construction +
# scheduler overhead per chunk. State still round-trips to device but
# stays hot in L2 across chunks within the same kernel.

_CROSS_CHUNK_SIMDGROUP_SOURCE = """
    // Template params: H, CHUNK, D are int constants.
    constexpr uint CHUNK_TILES = CHUNK / 8;    // 2 at CHUNK=16
    constexpr uint D_TILES     = D / 8;        // 16 at D=128
    constexpr uint K_D_TILES   = D / 8;
    constexpr uint K_C_TILES   = CHUNK / 8;
    constexpr uint D_D         = D * D;
    constexpr uint N_THREADS   = 1024;

    const uint n_chunks = uint(n_chunks_ptr[0]);

    // TG staging identical to Phase 2.
    threadgroup float vdiff_sm[CHUNK * D];     // 8 KB
    threadgroup float U_sm[CHUNK * D];         // 8 KB
    threadgroup float scratch_sm[CHUNK * D];   // 8 KB

    uint head_id = threadgroup_position_in_grid.y;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint tid     = thread_position_in_threadgroup.x;
    uint tpg     = threads_per_threadgroup.x;

    const device float* state_head_in = state_slice + head_id * D_D;
    device float*       ns_head       = new_state   + head_id * D_D;
    device float*       ds_head       = delta_scratch + head_id * D_D;

    // -----------------------------------------------------------------
    // Seed new_state with state_slice (one-time). Subsequent chunks will
    // read from new_state and update it in place.
    // -----------------------------------------------------------------
    for (uint i = tid; i < D_D; i += N_THREADS) {
        ns_head[i] = state_head_in[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // -----------------------------------------------------------------
    // Main chunk loop.
    // -----------------------------------------------------------------
    for (uint c = 0; c < n_chunks; ++c) {
        const uint chunkD_base  = c * H * CHUNK * D      + head_id * CHUNK * D;
        const uint chunkC_base  = c * H * CHUNK * CHUNK  + head_id * CHUNK * CHUNK;
        const uint betaC_base   = c * H * CHUNK          + head_id * CHUNK;
        const uint gteD_base    = c * H * D              + head_id * D;

        const device float* k_dec_cb  = k_decayed   + chunkD_base;
        const device float* q_dec_cb  = q_decayed   + chunkD_base;
        const device float* k_rest_cb = k_restored  + chunkD_base;
        const device float* Mqk_cb    = Mqk         + chunkC_base;
        const device float* INV_cb    = INV_bf      + chunkC_base;
        // ``vc`` may be fp32 or bf16 (PR H follow-on 3 Phase B); index inline.
        const uint          vc_off    = chunkD_base;
        const device float* beta_cb   = beta_bf16   + betaC_base;
        const device float* gte_cb    = g_total_exp + gteD_base;
        // ``out_h`` layout is [T_total, H, D] (= [n_chunks*CHUNK, H, D])
        // — written directly by the kernel epilogue so the caller can
        // skip a transpose+reshape. Per-chunk-row stride is H*D.
        device float*       out_cb    =
            out_h + c * CHUNK * H * D + head_id * D;

        // state_src == ns_head, updated in place by step 6. Reads in steps
        // 1/3/6 happen before step 6's writes (all guarded by barriers).
        const device float* state_src = ns_head;

        // ================================================================
        // Step 1: vdiff_sm = q_bf(q_bf(vc - q_bf(k_decayed @ state_T)) * beta)
        // Identical simdgroup tiling as Phase 2.
        // ================================================================
        {
            uint tile_r = simd_id / D_TILES;
            uint tile_c = simd_id % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> C_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
                const device float* A_src =
                    k_dec_cb + (tile_r * 8) * D + (k_tile * 8);
                simdgroup_load(A_tile, A_src, D);

                const device float* B_src =
                    state_src + (tile_c * 8) * D + (k_tile * 8);
                simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
            }

            threadgroup float* C_dst =
                &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(C_tile, C_dst, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;
            float p = vdiff_sm[idx];
            float v = float(vc[vc_off + idx]);  // bf16-or-fp32 read
            float t = float(bfloat(v - float(bfloat(p))));
            float b = beta_cb[i];
            vdiff_sm[idx] = float(bfloat(t * b));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 2: U_sm = q_bf(INV_bf @ vdiff). Scalar over K=CHUNK=16.
        // ================================================================
        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;
            uint j = idx % D;
            float acc = 0.0f;
            for (uint k = 0; k < CHUNK; ++k) {
                float iv = INV_cb[i * CHUNK + k];
                float vv = vdiff_sm[k * D + j];
                acc += iv * vv;
            }
            U_sm[idx] = float(bfloat(acc));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 3+4: out_h = q_bf(q_bf(q_decayed @ state_T) + q_bf(Mqk @ U))
        // ================================================================
        {
            uint tile_r = simd_id / D_TILES;
            uint tile_c = simd_id % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> qs_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
                const device float* A_src =
                    q_dec_cb + (tile_r * 8) * D + (k_tile * 8);
                simdgroup_load(A_tile, A_src, D);
                const device float* B_src =
                    state_src + (tile_c * 8) * D + (k_tile * 8);
                simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(qs_tile, A_tile, B_tile, qs_tile);
            }

            simdgroup_matrix<float, 8, 8> mu_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                const device float* A_src =
                    Mqk_cb + (tile_r * 8) * CHUNK + (k_tile * 8);
                simdgroup_load(A_tile, A_src, CHUNK);
                const threadgroup float* B_src =
                    &U_sm[(k_tile * 8) * D + (tile_c * 8)];
                simdgroup_load(B_tile, B_src, D);
                simdgroup_multiply_accumulate(mu_tile, A_tile, B_tile, mu_tile);
            }

            threadgroup float* qs_dst =
                &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(qs_tile, qs_dst, D);
            threadgroup float* mu_dst =
                &scratch_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(mu_tile, mu_dst, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;                             // chunk-local row
            uint j = idx % D;                             // d-col
            float q_val = float(bfloat(vdiff_sm[idx]));
            float m_val = float(bfloat(scratch_sm[idx]));
            // Direct-write to [T_total, H, D] layout: stride H*D per row.
            out_cb[i * H * D + j] = float(bfloat(q_val + m_val));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 5: matmul #5 → delta_s_T stored to SEPARATE scratch buffer
        // (ds_head, device), so step 6 can still read state from ns_head.
        // ================================================================
        {
            for (uint tile_batch = 0; tile_batch < 8; ++tile_batch) {
                uint tile_idx = simd_id * 8 + tile_batch;
                uint tile_a = tile_idx / D_TILES;
                uint tile_b = tile_idx % D_TILES;

                simdgroup_matrix<float, 8, 8> A_tile;
                simdgroup_matrix<float, 8, 8> B_tile;
                simdgroup_matrix<float, 8, 8> C_tile =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

                for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                    const threadgroup float* A_src =
                        &U_sm[(k_tile * 8) * D + (tile_a * 8)];
                    simdgroup_load(A_tile, A_src, D, ulong2(0, 0), true);

                    const device float* B_src =
                        k_rest_cb + (k_tile * 8) * D + (tile_b * 8);
                    simdgroup_load(B_tile, B_src, D);

                    simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
                }

                device float* C_dst =
                    ds_head + (tile_a * 8) * D + (tile_b * 8);
                simdgroup_store(C_tile, C_dst, D);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ================================================================
        // Step 6: fuse delta_s_T (scratch) with state (ns_head), write
        // final new_state back to ns_head. Each thread reads/writes one
        // cell — no cross-thread conflicts.
        //
        // We read state from ns_head BEFORE overwriting, same index. Since
        // only this thread writes idx, no race.
        // ================================================================
        for (uint idx = tid; idx < D_D; idx += tpg) {
            uint a = idx / D;
            uint b = idx % D;
            float pre = ds_head[idx];                           // delta_s_T[a, b]
            float stv = float(bfloat(ns_head[a * D + b]));      // q_bf(state[a, b])
            float gte = gte_cb[b];
            ns_head[idx] = float(bfloat(pre + stv * gte));
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""


@lru_cache(maxsize=16)
def _build_cross_chunk_simdgroup_kernel(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template at call time
    return mx.fast.metal_kernel(
        name="flash_kda_phase3b_cross_chunk_simdgroup",
        input_names=[
            "state_slice",
            "k_decayed",
            "q_decayed",
            "k_restored",
            "Mqk",
            "INV_bf",
            "vc",
            "beta_bf16",
            "g_total_exp",
            "n_chunks_ptr",
        ],
        output_names=["out_h", "new_state", "delta_scratch"],
        header=_HEADER,
        source=_CROSS_CHUNK_SIMDGROUP_SOURCE,
        ensure_row_contiguous=True,
    )


def metal_recurrence_cross_chunk_simdgroup(
    state_in: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
) -> tuple[mx.array, mx.array]:
    """Phase 3b cross-chunk simdgroup Metal kernel.

    Same external contract as ``metal_recurrence_cross_chunk_scalar``; the
    ``delta_scratch`` output tensor is allocated by MLX and discarded by
    this wrapper. Only ``out_h`` and ``new_state`` are returned.
    """
    assert HAS_METAL_KERNEL, (
        "metal_recurrence_cross_chunk_simdgroup called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    assert state_in.dtype == mx.float32
    assert k_decayed.dtype == mx.float32
    assert q_decayed.dtype == mx.float32
    assert k_restored.dtype == mx.float32
    assert Mqk.dtype == mx.float32
    assert INV_bf.dtype == mx.float32
    # vc is the passthrough of caller-provided v; under fused/fused2 prepare
    # modes the entry-side fp32 promotion is skipped (PR H follow-on 3
    # Phase B), so vc may be bf16 here. The kernel reads via implicit
    # bfloat→float conversion so both dtypes are equivalent.
    assert vc.dtype in (mx.float32, mx.bfloat16)
    assert beta_bf16.dtype == mx.float32
    assert g_total_exp.dtype == mx.float32

    H = state_in.shape[0]
    D = state_in.shape[1]
    n_chunks = k_decayed.shape[0]
    CHUNK = k_decayed.shape[2]

    assert state_in.shape == (H, D, D)
    assert k_decayed.shape == (n_chunks, H, CHUNK, D)
    assert q_decayed.shape == (n_chunks, H, CHUNK, D)
    assert k_restored.shape == (n_chunks, H, CHUNK, D)
    assert vc.shape == (n_chunks, H, CHUNK, D)
    assert Mqk.shape == (n_chunks, H, CHUNK, CHUNK)
    assert INV_bf.shape == (n_chunks, H, CHUNK, CHUNK)
    assert beta_bf16.shape == (n_chunks, H, CHUNK, 1)
    assert g_total_exp.shape == (n_chunks, H, D, 1)
    assert CHUNK == 16, f"Phase 3 requires CHUNK=16, got {CHUNK}"
    assert D == 128, f"Phase 3 requires D=128, got {D}"

    n_chunks_arr = mx.array([n_chunks], dtype=mx.int32)

    kernel = _build_cross_chunk_simdgroup_kernel(H=H, D=D, CHUNK=CHUNK)
    out_h, new_state, _scratch = kernel(
        inputs=[
            state_in, k_decayed, q_decayed, k_restored,
            Mqk, INV_bf, vc, beta_bf16, g_total_exp,
            n_chunks_arr,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(1024, H, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (n_chunks * CHUNK, H, D),  # [T_total, H, D] — direct layout
            (H, D, D),
            (H, D, D),  # delta_scratch, discarded
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return out_h, new_state


def warmup_cross_chunk_simdgroup_shapes(
    shapes: list[tuple[int, int, int, int]],
) -> None:
    """Pre-compile the Phase 3b kernel for each ``(H, CHUNK, D, n_chunks)``."""
    if not HAS_METAL_KERNEL:
        return
    for H, CHUNK, D, n_chunks in shapes:
        state = mx.zeros((H, D, D), dtype=mx.float32)
        k_dec = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        q_dec = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        k_rest = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        Mqk = mx.zeros((n_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        INV = mx.zeros((n_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        vc = mx.zeros((n_chunks, H, CHUNK, D), dtype=mx.float32)
        beta = mx.zeros((n_chunks, H, CHUNK, 1), dtype=mx.float32)
        gte = mx.zeros((n_chunks, H, D, 1), dtype=mx.float32)
        out_h, ns = metal_recurrence_cross_chunk_simdgroup(
            state, k_dec, q_dec, k_rest, Mqk, INV, vc, beta, gte,
        )
        mx.eval(out_h, ns)


# ---------------------------------------------------------------------------
# Phase 4: packed cross-chunk simdgroup kernel
# ---------------------------------------------------------------------------
#
# Multi-sequence variant of Phase 3b. Extends the dispatch grid with an
# N-axis so each threadgroup owns one (seq, head) pair. State-freeze
# semantics match ``_recurrence_body_packed``: for chunk index c and
# sequence n, if c >= n_chunks_per_seq[n] the per-seq state update is
# skipped (state remains at the value from chunk ``n_chunks_per_seq[n]-1``
# for that sequence). out_h is written unconditionally — caller trims per
# seq_lens upstream.
#
# Dispatch geometry:
#   grid        = (1024, H, N)
#   threadgroup = (1024, 1, 1)
#
# TG memory: identical to Phase 3b (24 KB). Device scratch
# ``delta_scratch`` grows to ``[N, H, D, D]`` since each (seq, head) TG
# needs its own delta_s_T slot.

_CROSS_CHUNK_PACKED_SIMDGROUP_SOURCE = """
    // Template params: H, CHUNK, D.
    constexpr uint CHUNK_TILES = CHUNK / 8;
    constexpr uint D_TILES     = D / 8;
    constexpr uint K_D_TILES   = D / 8;
    constexpr uint K_C_TILES   = CHUNK / 8;
    constexpr uint D_D         = D * D;
    constexpr uint N_THREADS   = 1024;

    // Runtime scalars.
    const uint max_chunks = uint(n_chunks_ptr[0]);

    threadgroup float vdiff_sm[CHUNK * D];
    threadgroup float U_sm[CHUNK * D];
    threadgroup float scratch_sm[CHUNK * D];

    uint head_id = threadgroup_position_in_grid.y;
    uint seq_id  = threadgroup_position_in_grid.z;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint tid     = thread_position_in_threadgroup.x;
    uint tpg     = threads_per_threadgroup.x;

    // Per-(seq, head) device bases. state_slice / new_state / delta_scratch
    // each have leading layout [N, H, D, D].
    const uint nh_base = (seq_id * H + head_id) * D_D;
    const device float* state_head_in = state_slice   + nh_base;
    device float*       ns_head       = new_state     + nh_base;
    device float*       ds_head       = delta_scratch + nh_base;

    // How many chunks are real for this sequence? Past this count, step 6
    // skips the state writeback so the per-seq state remains at the last
    // active chunk's value.
    const uint my_n_chunks = uint(n_chunks_per_seq_ptr[seq_id]);

    // Seed ns_head from state_slice[seq_id, head_id].
    for (uint i = tid; i < D_D; i += N_THREADS) {
        ns_head[i] = state_head_in[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint c = 0; c < max_chunks; ++c) {
        // Per (seq, chunk, head) input base. Input layout
        // [N, max_chunks, H, ...]:
        //   (((seq_id * max_chunks) + c) * H + head_id) * stride_tail
        const uint chunkD_base =
            (((seq_id * max_chunks) + c) * H + head_id) * CHUNK * D;
        const uint chunkC_base =
            (((seq_id * max_chunks) + c) * H + head_id) * CHUNK * CHUNK;
        const uint betaC_base  =
            (((seq_id * max_chunks) + c) * H + head_id) * CHUNK;
        const uint gteD_base   =
            (((seq_id * max_chunks) + c) * H + head_id) * D;

        const device float* k_dec_cb  = k_decayed   + chunkD_base;
        const device float* q_dec_cb  = q_decayed   + chunkD_base;
        const device float* k_rest_cb = k_restored  + chunkD_base;
        const device float* Mqk_cb    = Mqk         + chunkC_base;
        const device float* INV_cb    = INV_bf      + chunkC_base;
        // ``vc`` may be fp32 or bf16 (PR H follow-on 3 Phase B); index inline.
        const uint          vc_off    = chunkD_base;
        const device float* beta_cb   = beta_bf16   + betaC_base;
        const device float* gte_cb    = g_total_exp + gteD_base;
        // ``out_h`` layout is [N, padded_T, H, D] (= [N, max_chunks*CHUNK, H, D])
        // — written directly so the caller can skip a transpose+reshape.
        // Per-chunk-row stride is H*D.
        device float*       out_cb    =
            out_h + (seq_id * max_chunks + c) * CHUNK * H * D + head_id * D;

        const device float* state_src = ns_head;
        const bool active = (c < my_n_chunks);

        // ================================================================
        // Step 1: vdiff_sm = q_bf(q_bf(vc - q_bf(k_decayed @ state_T)) * beta)
        // ================================================================
        {
            uint tile_r = simd_id / D_TILES;
            uint tile_c = simd_id % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> C_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
                const device float* A_src =
                    k_dec_cb + (tile_r * 8) * D + (k_tile * 8);
                simdgroup_load(A_tile, A_src, D);

                const device float* B_src =
                    state_src + (tile_c * 8) * D + (k_tile * 8);
                simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
            }

            threadgroup float* C_dst =
                &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(C_tile, C_dst, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;
            float p = vdiff_sm[idx];
            float v = float(vc[vc_off + idx]);  // bf16-or-fp32 read
            float t = float(bfloat(v - float(bfloat(p))));
            float b = beta_cb[i];
            vdiff_sm[idx] = float(bfloat(t * b));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 2: U_sm = q_bf(INV_bf @ vdiff)
        // ================================================================
        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;
            uint j = idx % D;
            float acc = 0.0f;
            for (uint k = 0; k < CHUNK; ++k) {
                float iv = INV_cb[i * CHUNK + k];
                float vv = vdiff_sm[k * D + j];
                acc += iv * vv;
            }
            U_sm[idx] = float(bfloat(acc));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 3+4: out_h = q_bf(q_bf(q_decayed @ state_T) + q_bf(Mqk @ U))
        // ================================================================
        {
            uint tile_r = simd_id / D_TILES;
            uint tile_c = simd_id % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> qs_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
                const device float* A_src =
                    q_dec_cb + (tile_r * 8) * D + (k_tile * 8);
                simdgroup_load(A_tile, A_src, D);
                const device float* B_src =
                    state_src + (tile_c * 8) * D + (k_tile * 8);
                simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(qs_tile, A_tile, B_tile, qs_tile);
            }

            simdgroup_matrix<float, 8, 8> mu_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                const device float* A_src =
                    Mqk_cb + (tile_r * 8) * CHUNK + (k_tile * 8);
                simdgroup_load(A_tile, A_src, CHUNK);
                const threadgroup float* B_src =
                    &U_sm[(k_tile * 8) * D + (tile_c * 8)];
                simdgroup_load(B_tile, B_src, D);
                simdgroup_multiply_accumulate(mu_tile, A_tile, B_tile, mu_tile);
            }

            threadgroup float* qs_dst =
                &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(qs_tile, qs_dst, D);
            threadgroup float* mu_dst =
                &scratch_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(mu_tile, mu_dst, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;                             // chunk-local row
            uint j = idx % D;                             // d-col
            float q_val = float(bfloat(vdiff_sm[idx]));
            float m_val = float(bfloat(scratch_sm[idx]));
            // Direct-write to [N, padded_T, H, D] layout: stride H*D per row.
            out_cb[i * H * D + j] = float(bfloat(q_val + m_val));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 5: matmul #5 → delta_s_T in ds_head (per (seq, head) slot)
        // ================================================================
        {
            for (uint tile_batch = 0; tile_batch < 8; ++tile_batch) {
                uint tile_idx = simd_id * 8 + tile_batch;
                uint tile_a = tile_idx / D_TILES;
                uint tile_b = tile_idx % D_TILES;

                simdgroup_matrix<float, 8, 8> A_tile;
                simdgroup_matrix<float, 8, 8> B_tile;
                simdgroup_matrix<float, 8, 8> C_tile =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

                for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                    const threadgroup float* A_src =
                        &U_sm[(k_tile * 8) * D + (tile_a * 8)];
                    simdgroup_load(A_tile, A_src, D, ulong2(0, 0), true);

                    const device float* B_src =
                        k_rest_cb + (k_tile * 8) * D + (tile_b * 8);
                    simdgroup_load(B_tile, B_src, D);

                    simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
                }

                device float* C_dst =
                    ds_head + (tile_a * 8) * D + (tile_b * 8);
                simdgroup_store(C_tile, C_dst, D);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ================================================================
        // Step 6 (gated): state update, skipped for frozen seqs.
        // The branch is uniform across the threadgroup (``active`` depends
        // only on ``c`` and ``my_n_chunks`` which are loop-invariant within
        // the TG), so no divergence.
        // ================================================================
        if (active) {
            for (uint idx = tid; idx < D_D; idx += tpg) {
                uint a = idx / D;
                uint b = idx % D;
                float pre = ds_head[idx];
                float stv = float(bfloat(ns_head[a * D + b]));
                float gte = gte_cb[b];
                ns_head[idx] = float(bfloat(pre + stv * gte));
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
"""


@lru_cache(maxsize=16)
def _build_cross_chunk_packed_kernel(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template at call time
    return mx.fast.metal_kernel(
        name="flash_kda_phase4_cross_chunk_packed",
        input_names=[
            "state_slice",
            "k_decayed",
            "q_decayed",
            "k_restored",
            "Mqk",
            "INV_bf",
            "vc",
            "beta_bf16",
            "g_total_exp",
            "n_chunks_per_seq_ptr",
            "n_chunks_ptr",
        ],
        output_names=["out_h", "new_state", "delta_scratch"],
        header=_HEADER,
        source=_CROSS_CHUNK_PACKED_SIMDGROUP_SOURCE,
        ensure_row_contiguous=True,
    )


def metal_recurrence_cross_chunk_packed(
    state_in: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
    n_chunks_per_seq: mx.array,
) -> tuple[mx.array, mx.array]:
    """Phase 4 packed cross-chunk simdgroup Metal kernel.

    Shapes & dtypes:
      state_in          : [N, H, D, D]                         fp32 (bf16-valued)
      k_decayed         : [N, max_chunks, H, CHUNK, D]         fp32
      q_decayed         : [N, max_chunks, H, CHUNK, D]         fp32
      k_restored        : [N, max_chunks, H, CHUNK, D]         fp32
      Mqk               : [N, max_chunks, H, CHUNK, CHUNK]     fp32
      INV_bf            : [N, max_chunks, H, CHUNK, CHUNK]     fp32
      vc                : [N, max_chunks, H, CHUNK, D]         fp32
      beta_bf16         : [N, max_chunks, H, CHUNK, 1]         fp32
      g_total_exp       : [N, max_chunks, H, D, 1]             fp32
      n_chunks_per_seq  : [N] int32

    Returns:
      out_h     : [N, padded_T, H, D] fp32 (bf16-valued) — padded-tail
                  rows past per-seq seq_lens hold garbage; caller trims.
                  ``padded_T == max_chunks * CHUNK``.
      new_state : [N, H, D, D] fp32 (bf16-valued) — per-seq state frozen
                  at the ``n_chunks_per_seq[n]``-th chunk boundary.
    """
    assert HAS_METAL_KERNEL, (
        "metal_recurrence_cross_chunk_packed called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    assert state_in.dtype == mx.float32
    assert k_decayed.dtype == mx.float32
    assert q_decayed.dtype == mx.float32
    assert k_restored.dtype == mx.float32
    assert Mqk.dtype == mx.float32
    assert INV_bf.dtype == mx.float32
    # vc passthrough may be bf16 under fused/fused2 (PR H follow-on 3 Phase B).
    assert vc.dtype in (mx.float32, mx.bfloat16)
    assert beta_bf16.dtype == mx.float32
    assert g_total_exp.dtype == mx.float32
    assert n_chunks_per_seq.dtype == mx.int32

    N = state_in.shape[0]
    H = state_in.shape[1]
    D = state_in.shape[2]
    max_chunks = k_decayed.shape[1]
    CHUNK = k_decayed.shape[3]

    assert state_in.shape == (N, H, D, D)
    assert k_decayed.shape == (N, max_chunks, H, CHUNK, D)
    assert q_decayed.shape == (N, max_chunks, H, CHUNK, D)
    assert k_restored.shape == (N, max_chunks, H, CHUNK, D)
    assert vc.shape == (N, max_chunks, H, CHUNK, D)
    assert Mqk.shape == (N, max_chunks, H, CHUNK, CHUNK)
    assert INV_bf.shape == (N, max_chunks, H, CHUNK, CHUNK)
    assert beta_bf16.shape == (N, max_chunks, H, CHUNK, 1)
    assert g_total_exp.shape == (N, max_chunks, H, D, 1)
    assert n_chunks_per_seq.shape == (N,)
    assert CHUNK == 16, f"Phase 4 requires CHUNK=16, got {CHUNK}"
    assert D == 128, f"Phase 4 requires D=128, got {D}"

    n_chunks_arr = mx.array([max_chunks], dtype=mx.int32)

    kernel = _build_cross_chunk_packed_kernel(H=H, D=D, CHUNK=CHUNK)
    out_h, new_state, _scratch = kernel(
        inputs=[
            state_in, k_decayed, q_decayed, k_restored,
            Mqk, INV_bf, vc, beta_bf16, g_total_exp,
            n_chunks_per_seq, n_chunks_arr,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(1024, H, N),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (N, max_chunks * CHUNK, H, D),  # [N, padded_T, H, D] — direct
            (N, H, D, D),
            (N, H, D, D),  # delta_scratch, discarded
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return out_h, new_state


def warmup_cross_chunk_packed_shapes(
    shapes: list[tuple[int, int, int, int, int]],
) -> None:
    """Pre-compile for each ``(N, H, CHUNK, D, max_chunks)``."""
    if not HAS_METAL_KERNEL:
        return
    for N, H, CHUNK, D, max_chunks in shapes:
        state = mx.zeros((N, H, D, D), dtype=mx.float32)
        k_dec = mx.zeros((N, max_chunks, H, CHUNK, D), dtype=mx.float32)
        q_dec = mx.zeros((N, max_chunks, H, CHUNK, D), dtype=mx.float32)
        k_rest = mx.zeros((N, max_chunks, H, CHUNK, D), dtype=mx.float32)
        Mqk = mx.zeros((N, max_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        INV = mx.zeros((N, max_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        vc = mx.zeros((N, max_chunks, H, CHUNK, D), dtype=mx.float32)
        beta = mx.zeros((N, max_chunks, H, CHUNK, 1), dtype=mx.float32)
        gte = mx.zeros((N, max_chunks, H, D, 1), dtype=mx.float32)
        ncs = mx.array([max_chunks] * N, dtype=mx.int32)
        out_h, ns = metal_recurrence_cross_chunk_packed(
            state, k_dec, q_dec, k_rest, Mqk, INV, vc, beta, gte, ncs,
        )
        mx.eval(out_h, ns)


# ---------------------------------------------------------------------------
# PR M Option B: flat-ragged cross-chunk simdgroup kernel
# ---------------------------------------------------------------------------
#
# Flat-ragged variant of the Phase 3b simdgroup kernel. Consumes the
# already-flat-ragged prepare buffers produced by ``metal_prepare_chunk_fused_v4``
# (no inter-sequence padding chunks) and direct-writes outputs into a flat
# ``[T_total, H, D]`` tensor — eliminating the per-seq Python recurrence loop
# AND the trailing ``mx.concatenate(seq_outs)``.
#
# Dispatch geometry:
#   grid        = (1024, H, N)
#   threadgroup = (1024, 1, 1)
#
# Each TG owns one (seq_id, head_id) pair, walks chunks
# ``[seq_chunk_start[seq_id] .. seq_chunk_start[seq_id+1])`` (global indices
# into the flat-ragged prepare buffers), and writes output rows directly to
# ``out_h[seq_token_start[seq_id] + c_in_seq*CHUNK + slot, head_id, *]`` with
# tail-row masking against ``seq_len = seq_token_start[seq_id+1] - seq_token_start[seq_id]``.
#
# State and per-(seq, head) device scratch are sized ``[N, H, D, D]`` so each
# TG has its own slot — identical layout to the packed variant.

_CROSS_CHUNK_FLAT_RAGGED_SIMDGROUP_SOURCE = """
    // Template params: H, CHUNK, D are int constants.
    constexpr uint D_TILES     = D / 8;        // 16 at D=128
    constexpr uint K_D_TILES   = D / 8;
    constexpr uint K_C_TILES   = CHUNK / 8;
    constexpr uint D_D         = D * D;
    constexpr uint N_THREADS   = 1024;

    // TG staging — identical to Phase 3b/4 (24 KB total).
    threadgroup float vdiff_sm[CHUNK * D];     // 8 KB
    threadgroup float U_sm[CHUNK * D];         // 8 KB
    threadgroup float scratch_sm[CHUNK * D];   // 8 KB

    uint head_id = threadgroup_position_in_grid.y;
    uint seq_id  = threadgroup_position_in_grid.z;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint tid     = thread_position_in_threadgroup.x;
    uint tpg     = threads_per_threadgroup.x;

    // Per-(seq, head) device bases. state_slice / new_state / delta_scratch
    // each have leading layout [N, H, D, D].
    const uint nh_base = (seq_id * H + head_id) * D_D;
    const device float* state_head_in = state_slice   + nh_base;
    device float*       ns_head       = new_state     + nh_base;
    device float*       ds_head       = delta_scratch + nh_base;

    // Chunk-range and token-range bounds for this sequence.
    const uint c_start  = uint(seq_chunk_start[seq_id]);
    const uint c_end    = uint(seq_chunk_start[seq_id + 1]);
    const uint n_chunks_seq = c_end - c_start;
    const uint tok_start = uint(seq_token_start[seq_id]);
    const uint seq_len   = uint(seq_token_start[seq_id + 1]) - tok_start;

    // -----------------------------------------------------------------
    // Seed new_state with state_slice (one-time). Subsequent chunks will
    // read from new_state and update it in place.
    // -----------------------------------------------------------------
    for (uint i = tid; i < D_D; i += N_THREADS) {
        ns_head[i] = state_head_in[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // -----------------------------------------------------------------
    // Main chunk loop. ``c_in_seq`` is the per-seq chunk index;
    // ``c_global`` is the index into the flat-ragged prepare buffers.
    // -----------------------------------------------------------------
    for (uint c_in_seq = 0; c_in_seq < n_chunks_seq; ++c_in_seq) {
        const uint c_global    = c_start + c_in_seq;
        const uint chunkD_base = (c_global * H + head_id) * CHUNK * D;
        const uint chunkC_base = (c_global * H + head_id) * CHUNK * CHUNK;
        const uint betaC_base  = (c_global * H + head_id) * CHUNK;
        const uint gteD_base   = (c_global * H + head_id) * D;

        const device float* k_dec_cb  = k_decayed   + chunkD_base;
        const device float* q_dec_cb  = q_decayed   + chunkD_base;
        const device float* k_rest_cb = k_restored  + chunkD_base;
        const device float* Mqk_cb    = Mqk         + chunkC_base;
        const device float* INV_cb    = INV_bf      + chunkC_base;
        const uint          vc_off    = chunkD_base;
        const device float* beta_cb   = beta_bf16   + betaC_base;
        const device float* gte_cb    = g_total_exp + gteD_base;
        // out_h is flat [T_total, H, D]. The chunk's first output row is
        // at global token index (tok_start + c_in_seq*CHUNK).
        const uint tok_chunk_start = tok_start + c_in_seq * CHUNK;
        device float*       out_cb    =
            out_h + tok_chunk_start * H * D + head_id * D;

        const device float* state_src = ns_head;

        // ================================================================
        // Step 1: vdiff_sm = q_bf(q_bf(vc - q_bf(k_decayed @ state_T)) * beta)
        // ================================================================
        {
            uint tile_r = simd_id / D_TILES;
            uint tile_c = simd_id % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> C_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
                const device float* A_src =
                    k_dec_cb + (tile_r * 8) * D + (k_tile * 8);
                simdgroup_load(A_tile, A_src, D);

                const device float* B_src =
                    state_src + (tile_c * 8) * D + (k_tile * 8);
                simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
            }

            threadgroup float* C_dst =
                &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(C_tile, C_dst, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;
            float p = vdiff_sm[idx];
            float v = float(vc[vc_off + idx]);
            float t = float(bfloat(v - float(bfloat(p))));
            float b = beta_cb[i];
            vdiff_sm[idx] = float(bfloat(t * b));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 2: U_sm = q_bf(INV_bf @ vdiff). Scalar over K=CHUNK=16.
        // ================================================================
        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;
            uint j = idx % D;
            float acc = 0.0f;
            for (uint k = 0; k < CHUNK; ++k) {
                float iv = INV_cb[i * CHUNK + k];
                float vv = vdiff_sm[k * D + j];
                acc += iv * vv;
            }
            U_sm[idx] = float(bfloat(acc));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 3+4: out_h = q_bf(q_bf(q_decayed @ state_T) + q_bf(Mqk @ U))
        // ================================================================
        {
            uint tile_r = simd_id / D_TILES;
            uint tile_c = simd_id % D_TILES;

            simdgroup_matrix<float, 8, 8> A_tile;
            simdgroup_matrix<float, 8, 8> B_tile;
            simdgroup_matrix<float, 8, 8> qs_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint k_tile = 0; k_tile < K_D_TILES; ++k_tile) {
                const device float* A_src =
                    q_dec_cb + (tile_r * 8) * D + (k_tile * 8);
                simdgroup_load(A_tile, A_src, D);
                const device float* B_src =
                    state_src + (tile_c * 8) * D + (k_tile * 8);
                simdgroup_load(B_tile, B_src, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(qs_tile, A_tile, B_tile, qs_tile);
            }

            simdgroup_matrix<float, 8, 8> mu_tile =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                const device float* A_src =
                    Mqk_cb + (tile_r * 8) * CHUNK + (k_tile * 8);
                simdgroup_load(A_tile, A_src, CHUNK);
                const threadgroup float* B_src =
                    &U_sm[(k_tile * 8) * D + (tile_c * 8)];
                simdgroup_load(B_tile, B_src, D);
                simdgroup_multiply_accumulate(mu_tile, A_tile, B_tile, mu_tile);
            }

            threadgroup float* qs_dst =
                &vdiff_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(qs_tile, qs_dst, D);
            threadgroup float* mu_dst =
                &scratch_sm[(tile_r * 8) * D + (tile_c * 8)];
            simdgroup_store(mu_tile, mu_dst, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Output epilogue: tail-row mask. Rows where
        // (c_in_seq * CHUNK + i) >= seq_len would write past this seq's
        // allocated [seq_len, H, D] band into the next sequence's rows
        // (or past the buffer entirely on the last seq), so skip them.
        for (uint idx = tid; idx < CHUNK * D; idx += tpg) {
            uint i = idx / D;                             // chunk-local row
            uint j = idx % D;                             // d-col
            // Tail-mask: only write rows that fall within this seq's range.
            if (tok_chunk_start + i >= tok_start + seq_len) continue;
            float q_val = float(bfloat(vdiff_sm[idx]));
            float m_val = float(bfloat(scratch_sm[idx]));
            out_cb[i * H * D + j] = float(bfloat(q_val + m_val));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Step 5: matmul #5 → delta_s_T → ds_head (device scratch).
        // ================================================================
        {
            for (uint tile_batch = 0; tile_batch < 8; ++tile_batch) {
                uint tile_idx = simd_id * 8 + tile_batch;
                uint tile_a = tile_idx / D_TILES;
                uint tile_b = tile_idx % D_TILES;

                simdgroup_matrix<float, 8, 8> A_tile;
                simdgroup_matrix<float, 8, 8> B_tile;
                simdgroup_matrix<float, 8, 8> C_tile =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

                for (uint k_tile = 0; k_tile < K_C_TILES; ++k_tile) {
                    const threadgroup float* A_src =
                        &U_sm[(k_tile * 8) * D + (tile_a * 8)];
                    simdgroup_load(A_tile, A_src, D, ulong2(0, 0), true);

                    const device float* B_src =
                        k_rest_cb + (k_tile * 8) * D + (tile_b * 8);
                    simdgroup_load(B_tile, B_src, D);

                    simdgroup_multiply_accumulate(C_tile, A_tile, B_tile, C_tile);
                }

                device float* C_dst =
                    ds_head + (tile_a * 8) * D + (tile_b * 8);
                simdgroup_store(C_tile, C_dst, D);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ================================================================
        // Step 6: fuse delta_s_T (scratch) with state, write back to ns_head.
        // ================================================================
        for (uint idx = tid; idx < D_D; idx += tpg) {
            uint a = idx / D;
            uint b = idx % D;
            float pre = ds_head[idx];                           // delta_s_T[a, b]
            float stv = float(bfloat(ns_head[a * D + b]));      // q_bf(state[a, b])
            float gte = gte_cb[b];
            ns_head[idx] = float(bfloat(pre + stv * gte));
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""


@lru_cache(maxsize=16)
def _build_cross_chunk_flat_ragged_kernel(H: int, D: int, CHUNK: int):
    del H, D, CHUNK  # baked via template at call time
    return mx.fast.metal_kernel(
        name="flash_kda_pr_m_optb_cross_chunk_flat_ragged",
        input_names=[
            "state_slice",
            "k_decayed",
            "q_decayed",
            "k_restored",
            "Mqk",
            "INV_bf",
            "vc",
            "beta_bf16",
            "g_total_exp",
            "seq_chunk_start",
            "seq_token_start",
        ],
        output_names=["out_h", "new_state", "delta_scratch"],
        header=_HEADER,
        source=_CROSS_CHUNK_FLAT_RAGGED_SIMDGROUP_SOURCE,
        ensure_row_contiguous=True,
    )


def metal_recurrence_cross_chunk_flat_ragged(
    state_in: mx.array,
    k_decayed: mx.array,
    q_decayed: mx.array,
    k_restored: mx.array,
    Mqk: mx.array,
    INV_bf: mx.array,
    vc: mx.array,
    beta_bf16: mx.array,
    g_total_exp: mx.array,
    seq_chunk_start: mx.array,
    seq_token_start: mx.array,
    T_total: int,
) -> tuple[mx.array, mx.array]:
    """PR M Option B flat-ragged cross-chunk simdgroup Metal kernel.

    Consumes flat-ragged prepare buffers (no inter-seq padding) and
    direct-writes outputs into ``[T_total, H, D]`` at the per-seq token
    offsets.

    Shapes & dtypes:
      state_in        : [N, H, D, D]                       fp32 (bf16-valued)
      k_decayed       : [total_chunks, H, CHUNK, D]        fp32
      q_decayed       : [total_chunks, H, CHUNK, D]        fp32
      k_restored      : [total_chunks, H, CHUNK, D]        fp32
      Mqk             : [total_chunks, H, CHUNK, CHUNK]    fp32
      INV_bf          : [total_chunks, H, CHUNK, CHUNK]    fp32
      vc              : [total_chunks, H, CHUNK, D]        fp32 or bf16
      beta_bf16       : [total_chunks, H, CHUNK, 1]        fp32
      g_total_exp     : [total_chunks, H, D, 1]            fp32
      seq_chunk_start : [N+1] int32 — chunk-range bounds per seq
      seq_token_start : [N+1] int32 — token-range bounds per seq (= cu_seqlens)
      T_total         : int — output buffer length (= seq_token_start[N])

    Returns:
      out_h     : [T_total, H, D] fp32 (bf16-valued) — direct-written
      new_state : [N, H, D, D]    fp32 (bf16-valued) — per-seq final states
    """
    assert HAS_METAL_KERNEL, (
        "metal_recurrence_cross_chunk_flat_ragged called on non-M3+ hardware; "
        "gate with HAS_METAL_KERNEL upstream."
    )
    assert state_in.dtype == mx.float32
    assert k_decayed.dtype == mx.float32
    assert q_decayed.dtype == mx.float32
    assert k_restored.dtype == mx.float32
    assert Mqk.dtype == mx.float32
    assert INV_bf.dtype == mx.float32
    assert vc.dtype in (mx.float32, mx.bfloat16)
    assert beta_bf16.dtype == mx.float32
    assert g_total_exp.dtype == mx.float32
    assert seq_chunk_start.dtype == mx.int32
    assert seq_token_start.dtype == mx.int32

    N = state_in.shape[0]
    H = state_in.shape[1]
    D = state_in.shape[2]
    total_chunks = k_decayed.shape[0]
    CHUNK = k_decayed.shape[2]

    assert state_in.shape == (N, H, D, D)
    assert k_decayed.shape == (total_chunks, H, CHUNK, D)
    assert q_decayed.shape == (total_chunks, H, CHUNK, D)
    assert k_restored.shape == (total_chunks, H, CHUNK, D)
    assert vc.shape == (total_chunks, H, CHUNK, D)
    assert Mqk.shape == (total_chunks, H, CHUNK, CHUNK)
    assert INV_bf.shape == (total_chunks, H, CHUNK, CHUNK)
    assert beta_bf16.shape == (total_chunks, H, CHUNK, 1)
    assert g_total_exp.shape == (total_chunks, H, D, 1)
    assert seq_chunk_start.shape == (N + 1,)
    assert seq_token_start.shape == (N + 1,)
    assert CHUNK == 16, f"PR M Option B requires CHUNK=16, got {CHUNK}"
    assert D == 128, f"PR M Option B requires D=128, got {D}"

    kernel = _build_cross_chunk_flat_ragged_kernel(H=H, D=D, CHUNK=CHUNK)
    out_h, new_state, _scratch = kernel(
        inputs=[
            state_in, k_decayed, q_decayed, k_restored,
            Mqk, INV_bf, vc, beta_bf16, g_total_exp,
            seq_chunk_start, seq_token_start,
        ],
        template=[("H", H), ("CHUNK", CHUNK), ("D", D)],
        grid=(1024, H, N),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (T_total, H, D),  # flat direct-write
            (N, H, D, D),
            (N, H, D, D),     # delta_scratch, discarded
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return out_h, new_state


def warmup_cross_chunk_flat_ragged_shapes(
    shapes: list[tuple[int, int, int, int, int, int]],
) -> None:
    """Pre-compile for each ``(N, H, CHUNK, D, total_chunks, T_total)``.

    Provided shapes must satisfy ``T_total <= total_chunks * CHUNK`` (the
    upper bound of valid tokens — partial-last-chunks contribute fewer).
    """
    if not HAS_METAL_KERNEL:
        return
    for N, H, CHUNK, D, total_chunks, T_total in shapes:
        state = mx.zeros((N, H, D, D), dtype=mx.float32)
        k_dec = mx.zeros((total_chunks, H, CHUNK, D), dtype=mx.float32)
        q_dec = mx.zeros((total_chunks, H, CHUNK, D), dtype=mx.float32)
        k_rest = mx.zeros((total_chunks, H, CHUNK, D), dtype=mx.float32)
        Mqk = mx.zeros((total_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        INV = mx.zeros((total_chunks, H, CHUNK, CHUNK), dtype=mx.float32)
        vc = mx.zeros((total_chunks, H, CHUNK, D), dtype=mx.float32)
        beta = mx.zeros((total_chunks, H, CHUNK, 1), dtype=mx.float32)
        gte = mx.zeros((total_chunks, H, D, 1), dtype=mx.float32)
        # Build trivial uniform metadata: each seq holds total_chunks/N chunks.
        per_seq = total_chunks // N
        scs = mx.array(
            [i * per_seq for i in range(N)] + [total_chunks], dtype=mx.int32,
        )
        # Distribute T_total tokens evenly; last seq absorbs the remainder.
        per_tok = T_total // N
        tok_starts = [i * per_tok for i in range(N)] + [T_total]
        sts = mx.array(tok_starts, dtype=mx.int32)
        out_h, ns = metal_recurrence_cross_chunk_flat_ragged(
            state, k_dec, q_dec, k_rest, Mqk, INV, vc, beta, gte,
            scs, sts, T_total,
        )
        mx.eval(out_h, ns)


__all__ = [
    "HAS_METAL_KERNEL",
    "metal_matmul_A_by_B",
    "warmup_matmul_shapes",
    "metal_recurrence_body_single",
    "warmup_recurrence_shapes",
    "metal_recurrence_cross_chunk_scalar",
    "warmup_cross_chunk_scalar_shapes",
    "metal_recurrence_cross_chunk_simdgroup",
    "warmup_cross_chunk_simdgroup_shapes",
    "metal_recurrence_cross_chunk_packed",
    "warmup_cross_chunk_packed_shapes",
    "metal_recurrence_cross_chunk_flat_ragged",
    "warmup_cross_chunk_flat_ragged_shapes",
]
