"""Section-by-section wall-clock profiler for the MLX FlashKDA optimized path.

Purpose
-------
Answer the go/no-go question for a custom Metal Neumann-inverse kernel
(plan.md §Phase 8 "Custom Metal rule"). We need hard numbers for:

* dispatch accounting: what fraction of time is in the pre-compute phase
  vs the sequential chunk recurrence;
* which pre-compute sub-section dominates (Neumann chain vs. casts vs.
  L construction vs. Mqk);
* for varlen: whether the host-side per-sequence Python loop is a real
  overhead for realistic workloads.

Methodology
-----------
MLX is lazy: ``mx.matmul`` etc. queue graph nodes, ``mx.eval`` flushes.
We therefore cannot trivially time "section (e) in isolation" by wrapping
those lines in ``time.perf_counter()`` — MLX happily fuses and defers
them until the next ``mx.eval``.

Instead we use **cumulative timing**:

1. We re-implement the forward path inside this file (copied from
   ``flash_kda_mlx/optimized.py``) so we can insert ``mx.eval`` barriers at
   named section boundaries.
2. Each measurement runs the forward *from scratch* up to a named
   barrier B_k, calls ``mx.eval`` on every tensor live at that point,
   and records the elapsed wall-clock time ``T_k``.
3. Section times are the differences ``T_k - T_{k-1}``.

This recipe is the MLX-idiomatic analog of CUDA events: you always
measure cumulative work, never an isolated range. It is correct even
when the graph fuses operations internally.

Caveats
-------
* MLX fusion may fold two adjacent "sections" into a single GPU kernel.
  When ``T_k - T_{k-1}`` is within noise of zero we report it that way
  explicitly rather than pretending the sub-section was independently
  measurable.
* We enforce a "no free lunch" invariant: the cumulative time at the
  final barrier must match the end-to-end time of an un-instrumented
  call, within noise. The driver checks and reports the residual.
* p90 > 2× median is flagged. We run 3 warmups + 10 measured iters.

Usage
-----
    uv run --no-config python -m benchmarks.section_timings
    uv run --no-config python -m benchmarks.section_timings --cases fixed_T4096_H8
    uv run --no-config python -m benchmarks.section_timings --json \\
        benchmarks/results/section_timings.json

Output
------
* stdout: a human-readable markdown table per case;
* ``--json`` path: machine-readable rows keyed by (case, section).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx

# Share the bench-harness case defs + input construction so we never
# diverge from the real benchmark numbers.
from ._harness import (
    BENCH_CASES,
    DEFAULT_CASES,
    STRESS_CASES,
    Case,
    build_call_kwargs,
)

# Pull the same primitives the optimized path uses so quantization /
# cast boundaries are bit-for-bit identical.
from flash_kda_mlx.reference import (
    CHUNK,
    LOG2E,
    _ex2_ftz,
    _l2_normalize,
    _q_bf16,
    _validate,
)


# ---------------------------------------------------------------------------
# Section identifiers. Kept as constants so JSON output and report tables
# share a single vocabulary.
# ---------------------------------------------------------------------------

# Pre-compute sub-sections (run inside _precompute_chunk_tensors)
SEC_L2_GATE      = "a_l2_gate_decay"     # L2-norm q/k + gate activation (pre-seq-loop)
SEC_GCUMSUM      = "b_gcumsum_exp"       # g_cumsum + ex2 of ±g_cumsum, g_total
SEC_KQ_CASTS     = "c_k_q_decayed_casts" # k_decayed, q_decayed, k_inv, k_restored
SEC_L_CONSTRUCT  = "d_L_construct_tril"  # L = k_decayed @ k_inv.T, tril, *beta
SEC_NEUMANN      = "e_neumann_inv_chain" # 6 fp16 matmuls building INV
SEC_MQK          = "f_mqk_matmul_tril"   # Mqk = q_decayed @ k_inv.T, tril, bf16
SEC_BETA_CAST    = "g_beta_casts_misc"   # beta_bf16 materialization + g_total_exp
SEC_RECURRENCE   = "h_chunk_recurrence"  # sequential per-chunk state update
SEC_CONCAT_FINAL = "i_output_concat"     # trailing concat + final_state cast

ALL_SECTIONS = [
    SEC_L2_GATE, SEC_GCUMSUM, SEC_KQ_CASTS, SEC_L_CONSTRUCT,
    SEC_NEUMANN, SEC_MQK, SEC_BETA_CAST, SEC_RECURRENCE, SEC_CONCAT_FINAL,
]


# ---------------------------------------------------------------------------
# Instrumented forward. This is a mirror of flash_kda_mlx.optimized.fwd_optimized
# that accepts a ``stop_at`` barrier name. When the barrier is hit, we
# ``mx.eval`` every tensor currently live at that semantic boundary and
# return (so downstream sections are never executed).
#
# Keeping this local (rather than modifying optimized.py) isolates the
# instrumentation from parallel edits to the production path.
# ---------------------------------------------------------------------------

def _fp16_mm(a: mx.array, b: mx.array) -> mx.array:
    return mx.matmul(a.astype(mx.float16), b.astype(mx.float16)).astype(mx.float32)


def _pad_to_multiple(x: mx.array, multiple: int, axis: int = 0) -> tuple[mx.array, int]:
    n = x.shape[axis]
    rem = n % multiple
    if rem == 0:
        return x, 0
    pad_len = multiple - rem
    pad_shape = list(x.shape)
    pad_shape[axis] = pad_len
    pad = mx.zeros(tuple(pad_shape), dtype=x.dtype)
    return mx.concatenate([x, pad], axis=axis), pad_len


def _forward_up_to(
    *,
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
    scale: float, out_like: mx.array,
    A_log: mx.array, dt_bias: mx.array, lower_bound: float,
    initial_state: Optional[mx.array],
    final_state_like: Optional[mx.array],
    cu_seqlens: Optional[mx.array],
    stop_at: Optional[str],
) -> None:
    """Run the optimized forward, evaluating all live tensors at ``stop_at``.

    The function returns ``None`` because the instrumentation contract is
    "do work, then force eval" — the actual arrays are not needed by the
    caller.

    ``stop_at=None`` means "run the whole forward and eval the final
    outputs", which gives the end-to-end baseline.
    """
    _validate(q, k, v, g, beta, out_like, A_log, dt_bias,
              initial_state, final_state_like, cu_seqlens)

    B, T_seq, H, D = q.shape
    T_total = B * T_seq

    q = q.reshape(T_total, H, D).astype(mx.float32)
    k = k.reshape(T_total, H, D).astype(mx.float32)
    v = v.reshape(T_total, H, D).astype(mx.float32)
    g = g.reshape(T_total, H, D).astype(mx.float32)
    beta = beta.reshape(T_total, H).astype(mx.float32)

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

    want_final = final_state_like is not None
    state_fp32 = False
    if initial_state is not None and initial_state.dtype == mx.float32:
        state_fp32 = True
    if final_state_like is not None and final_state_like.dtype == mx.float32:
        state_fp32 = True

    # --- Section a: L2 + gate/decay setup -----------------------------------
    q = _q_bf16(_l2_normalize(q))
    k = _q_bf16(_l2_normalize(k))

    g = g + dt_bias[None, :, :].astype(mx.float32)
    a_log_exp = _ex2_ftz(A_log.astype(mx.float32) * LOG2E)
    a_log_exp = a_log_exp[None, :, None]
    g = (lower_bound * LOG2E) * mx.sigmoid(a_log_exp * g)

    if initial_state is not None:
        work_state = initial_state.astype(mx.bfloat16).astype(mx.float32)
    else:
        work_state = mx.zeros((N, H, D, D), dtype=mx.float32)

    if stop_at == SEC_L2_GATE:
        mx.eval(q, k, g, work_state)
        return

    # Force work_state to materialize so the sequential loop doesn't re-run
    # it each chunk. Matches the production path's mx.eval(work_state) call.
    mx.eval(work_state)

    scale_bf16_rt = _q_bf16(mx.array([scale], dtype=mx.float32))[0]

    all_out_chunks: list[mx.array] = []

    # Per-sequence totals we accumulate across barriers. We need these
    # because some "sections" (e.g. the Neumann chain) live inside the
    # per-sequence loop and must be summed across sequences.
    last_live: list[mx.array] = []

    for seq_idx in range(N):
        bos = cu_list[seq_idx]
        eos = cu_list[seq_idx + 1]
        seq_len = eos - bos

        g_seq = g[bos:eos]
        q_seq = q[bos:eos]
        k_seq = k[bos:eos]
        v_seq = v[bos:eos]
        beta_seq = beta[bos:eos]

        g_seq, _ = _pad_to_multiple(g_seq, CHUNK, axis=0)
        q_seq, _ = _pad_to_multiple(q_seq, CHUNK, axis=0)
        k_seq, _ = _pad_to_multiple(k_seq, CHUNK, axis=0)
        v_seq, _ = _pad_to_multiple(v_seq, CHUNK, axis=0)
        beta_seq, _ = _pad_to_multiple(beta_seq, CHUNK, axis=0)

        padded_T = g_seq.shape[0]
        assert padded_T % CHUNK == 0
        n_chunks = padded_T // CHUNK

        def _to_chunks_hd(x: mx.array) -> mx.array:
            return x.reshape(n_chunks, CHUNK, H, D).transpose(0, 2, 1, 3)

        gc = _to_chunks_hd(g_seq)
        qc = _to_chunks_hd(q_seq)
        kc = _to_chunks_hd(k_seq)
        vc = _to_chunks_hd(v_seq)
        bc = beta_seq.reshape(n_chunks, CHUNK, H).transpose(0, 2, 1)

        # --- Section b: g_cumsum + ex2 -------------------------------------
        g_cumsum = mx.cumsum(gc, axis=2)
        g_total = g_cumsum[:, :, -1:, :]
        ex_pos = _q_bf16(_ex2_ftz(g_cumsum))
        ex_neg = _q_bf16(_ex2_ftz(-g_cumsum))
        ex_gtot = _q_bf16(_ex2_ftz(g_total))

        if stop_at == SEC_GCUMSUM:
            last_live = [ex_pos, ex_neg, ex_gtot, qc, kc, vc, bc,
                         work_state, *last_live]
            # Keep accumulating per-seq live tensors; eval only at end.
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

        # --- Section c: k_decayed / q_decayed / k_inv / k_restored casts ---
        k_decayed = _q_bf16(kc * ex_pos)
        q_decayed = _q_bf16(_q_bf16(qc * ex_pos) * scale_bf16_rt)
        k_inv = _q_bf16(kc * ex_neg)
        k_restored = _q_bf16(k_inv * ex_gtot)

        if stop_at == SEC_KQ_CASTS:
            last_live = [k_decayed, q_decayed, k_inv, k_restored, vc, bc,
                         g_total, work_state, *last_live]
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

        # --- Section d: L construction + tril + beta scale ------------------
        k_inv_T = k_inv.transpose(0, 1, 3, 2)
        L = mx.matmul(k_decayed, k_inv_T)
        L = L.astype(mx.float16).astype(mx.float32)

        beta_act = mx.sigmoid(bc)
        beta_bf16 = _q_bf16(beta_act)[:, :, :, None]
        beta_fp16 = (beta_act.astype(mx.float16).astype(mx.float32))[:, :, :, None]

        L = mx.tril(L, k=-1) * beta_fp16
        L = L.astype(mx.float16).astype(mx.float32)

        if stop_at == SEC_L_CONSTRUCT:
            last_live = [L, k_decayed, q_decayed, k_inv, k_restored, vc,
                         beta_bf16, g_total, work_state, *last_live]
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

        # --- Section e: Neumann series INV chain (6 fp16 matmuls) ----------
        eye_chunk = mx.eye(CHUNK, dtype=mx.float32).reshape(1, 1, CHUNK, CHUNK)
        INV = (eye_chunk - L).astype(mx.float16).astype(mx.float32)

        L2 = _fp16_mm(L, L)
        INV = INV + _fp16_mm(INV, L2)
        L4 = _fp16_mm(L2, L2)
        INV = INV + _fp16_mm(INV, L4)
        L8 = _fp16_mm(L4, L4)
        INV = INV + _fp16_mm(INV, L8)

        INV_bf = _q_bf16(INV)

        if stop_at == SEC_NEUMANN:
            last_live = [INV_bf, k_decayed, q_decayed, k_restored, vc,
                         beta_bf16, g_total, work_state, *last_live]
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

        # --- Section f: Mqk ------------------------------------------------
        Mqk = _q_bf16(mx.matmul(q_decayed, k_inv_T))
        Mqk = _q_bf16(mx.tril(Mqk))

        if stop_at == SEC_MQK:
            last_live = [Mqk, INV_bf, k_decayed, q_decayed, k_restored, vc,
                         beta_bf16, g_total, work_state, *last_live]
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

        # --- Section g: beta/g_total casts wrapped up ----------------------
        g_total_exp = _ex2_ftz(g_total).reshape(n_chunks, H, D, 1)

        if stop_at == SEC_BETA_CAST:
            last_live = [Mqk, INV_bf, k_decayed, q_decayed, k_restored, vc,
                         beta_bf16, g_total_exp, work_state, *last_live]
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

        # --- Section h: sequential chunk recurrence ------------------------
        state_slice = work_state[seq_idx]
        chunk_outs: list[mx.array] = []
        for chunk_idx in range(n_chunks):
            k_dec_c = k_decayed[chunk_idx]
            q_dec_c = q_decayed[chunk_idx]
            k_res_c = k_restored[chunk_idx]
            Mqk_c = Mqk[chunk_idx]
            INV_c = INV_bf[chunk_idx]
            vc_c = vc[chunk_idx]
            beta_c = beta_bf16[chunk_idx]
            gte_c = g_total_exp[chunk_idx]

            state_bf = _q_bf16(state_slice)

            vdiff = vc_c - _q_bf16(mx.matmul(k_dec_c, state_bf.transpose(0, 2, 1)))
            vdiff = _q_bf16(vdiff)
            vdiff = _q_bf16(vdiff * beta_c)

            U = _q_bf16(mx.matmul(INV_c, vdiff))
            out_h = _q_bf16(mx.matmul(q_dec_c, state_bf.transpose(0, 2, 1)))
            out_h = _q_bf16(out_h + _q_bf16(mx.matmul(Mqk_c, U)))

            delta_s = mx.matmul(k_res_c.transpose(0, 2, 1), U)
            new_state_fp32 = delta_s + state_bf.transpose(0, 2, 1) * gte_c
            state_slice = _q_bf16(new_state_fp32.transpose(0, 2, 1))

            chunk_outs.append(out_h.transpose(1, 0, 2))

        seq_out = mx.concatenate(chunk_outs, axis=0)
        if seq_out.shape[0] > seq_len:
            seq_out = seq_out[:seq_len]

        all_out_chunks.append(seq_out)
        work_state[seq_idx] = state_slice

        if stop_at == SEC_RECURRENCE:
            last_live = [seq_out, work_state, *last_live]
            if seq_idx == N - 1:
                mx.eval(*last_live)
                return
            continue

    # --- Section i: final concat + optional state cast --------------------
    if all_out_chunks:
        out_flat = mx.concatenate(all_out_chunks, axis=0)
    else:
        out_flat = mx.zeros((T_total, H, D), dtype=mx.float32)
    out = out_flat.reshape(B, T_seq, H, D)

    final: Optional[mx.array] = None
    if want_final:
        if state_fp32:
            final = work_state.astype(mx.float32)
        else:
            final = _q_bf16(work_state).astype(mx.bfloat16)

    if final is not None:
        mx.eval(out, final)
    else:
        mx.eval(out)


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    n_iters: int
    noisy: bool  # p90 > 2x median

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "p90_ms": self.p90_ms,
            "min_ms": self.min_ms,
            "n_iters": self.n_iters,
            "noisy": self.noisy,
        }


def _time_call(
    fn: Callable[[], None],
    *,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> TimingStats:
    for _ in range(n_warmup):
        fn()

    samples: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    median = statistics.median(samples)
    p90 = samples[int(math.ceil(0.9 * len(samples))) - 1]
    return TimingStats(
        median_ms=median,
        mean_ms=statistics.fmean(samples),
        p90_ms=p90,
        min_ms=samples[0],
        n_iters=n_iters,
        noisy=p90 > 2.0 * median,
    )


# ---------------------------------------------------------------------------
# Per-case driver
# ---------------------------------------------------------------------------

def _run_case(case: Case, *, n_warmup: int, n_iters: int) -> dict[str, Any]:
    """Measure cumulative section timings for a single case.

    Returns:
        {
            "case": name,
            "cumulative": {section: TimingStats as dict, ...},
            "section": {section: {..., "delta_ms": ...}, ...},
            "end_to_end": TimingStats as dict,
            "residual_ms": float  # end_to_end - last cumulative
        }
    """
    kwargs = build_call_kwargs(case)

    def _call(stop_at: Optional[str]) -> Callable[[], None]:
        def go() -> None:
            _forward_up_to(
                q=kwargs["q"], k=kwargs["k"], v=kwargs["v"],
                g=kwargs["g"], beta=kwargs["beta"],
                scale=kwargs["scale"], out_like=kwargs["out"],
                A_log=kwargs["A_log"], dt_bias=kwargs["dt_bias"],
                lower_bound=kwargs["lower_bound"],
                initial_state=kwargs.get("initial_state"),
                final_state_like=kwargs.get("final_state"),
                cu_seqlens=kwargs.get("cu_seqlens"),
                stop_at=stop_at,
            )
        return go

    print(f"\n## {case.name}")
    print(f"  measuring cumulative times across {len(ALL_SECTIONS)} barriers + end-to-end...")

    cumulative: dict[str, TimingStats] = {}
    for sec in ALL_SECTIONS:
        stats = _time_call(_call(sec), n_warmup=n_warmup, n_iters=n_iters)
        cumulative[sec] = stats
        flag = "  [NOISY]" if stats.noisy else ""
        print(f"    cum @ {sec:<22s}: median {stats.median_ms:7.3f} ms "
              f"(p90 {stats.p90_ms:7.3f}){flag}")

    e2e = _time_call(_call(None), n_warmup=n_warmup, n_iters=n_iters)
    print(f"    end-to-end              : median {e2e.median_ms:7.3f} ms "
          f"(p90 {e2e.p90_ms:7.3f})")

    # Section deltas
    section_times: dict[str, dict[str, float | int | bool]] = {}
    prev_median = 0.0
    for sec in ALL_SECTIONS:
        cur = cumulative[sec].median_ms
        delta = cur - prev_median
        row = cumulative[sec].as_dict()
        row["delta_ms"] = delta
        section_times[sec] = row
        prev_median = cur

    # Residual: how much of e2e is unaccounted for?
    residual = e2e.median_ms - cumulative[ALL_SECTIONS[-1]].median_ms

    return {
        "case": case.name,
        "cumulative": {s: cumulative[s].as_dict() for s in ALL_SECTIONS},
        "section": section_times,
        "end_to_end": e2e.as_dict(),
        "residual_ms": residual,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _format_case_table(result: dict[str, Any]) -> str:
    lines: list[str] = []
    case = result["case"]
    e2e = result["end_to_end"]["median_ms"]
    lines.append(f"\n### {case}  (end-to-end median: {e2e:.3f} ms)\n")
    lines.append(
        "| Section                    | Cum median (ms) | Delta (ms) | % of E2E | Noisy? |"
    )
    lines.append(
        "|----------------------------|-----------------|------------|----------|--------|"
    )
    for sec in ALL_SECTIONS:
        row = result["section"][sec]
        pct = 100.0 * row["delta_ms"] / e2e if e2e > 0 else 0.0
        noisy = "yes" if row["noisy"] else "no"
        lines.append(
            f"| {sec:<26s} | {row['median_ms']:>15.3f} | "
            f"{row['delta_ms']:>10.3f} | {pct:>7.1f}% | {noisy:>6s} |"
        )
    lines.append(
        f"| residual (e2e - last cum)  |               - | "
        f"{result['residual_ms']:>10.3f} | "
        f"{100.0 * result['residual_ms'] / e2e if e2e > 0 else 0.0:>7.1f}% |      - |"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_cases(names: list[str] | None) -> list[Case]:
    universe = {c.name: c for c in (*DEFAULT_CASES, *STRESS_CASES, *BENCH_CASES)}
    if not names:
        # Default: the three cases the report requires.
        default = ["fixed_T1024_H4", "fixed_T4096_H8", "varlen_mixed_H4"]
        return [universe[n] for n in default]
    missing = [n for n in names if n not in universe]
    if missing:
        raise SystemExit(
            f"Unknown case(s): {missing}. "
            f"Available: {sorted(universe)}"
        )
    return [universe[n] for n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Case names (default: fixed_T1024_H4 fixed_T4096_H8 varlen_mixed_H4)")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--json", type=str,
                        default=str(Path(__file__).parent / "results" / "section_timings.json"),
                        help="Output JSON path.")
    args = parser.parse_args(argv)

    cases = _find_cases(args.cases)
    print(f"# MLX section timings — device={mx.default_device()}  "
          f"n_warmup={args.n_warmup}  n_iters={args.n_iters}")

    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(_run_case(case, n_warmup=args.n_warmup, n_iters=args.n_iters))

    # Pretty markdown summary
    print("\n\n# Summary tables\n")
    for r in results:
        print(_format_case_table(r))

    # JSON dump
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Flatten to one row per (case, section) plus end-to-end rows for easy grep.
    flat_rows: list[dict[str, Any]] = []
    for r in results:
        for sec in ALL_SECTIONS:
            row = {"case": r["case"], "section": sec, **r["section"][sec]}
            flat_rows.append(row)
        flat_rows.append({
            "case": r["case"],
            "section": "_end_to_end",
            **r["end_to_end"],
            "delta_ms": r["end_to_end"]["median_ms"],
        })
        flat_rows.append({
            "case": r["case"],
            "section": "_residual",
            "median_ms": r["residual_ms"],
            "mean_ms": r["residual_ms"],
            "p90_ms": r["residual_ms"],
            "min_ms": r["residual_ms"],
            "n_iters": r["end_to_end"]["n_iters"],
            "noisy": False,
            "delta_ms": r["residual_ms"],
        })
    out.write_text(json.dumps(flat_rows, indent=2))
    print(f"\nWrote JSON to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
