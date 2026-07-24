"""Validate the predictor on hard-coded shapes it never measured.

One --bench per grid, matching run.py (gemm_bf16/fp16, attn_bf16, moe_bf16/mxfp4, allreduce). Each
bench walks a fixed list of test shapes — real projection / head / expert / collective dimensions
AND their tensor/expert-parallel per-rank variants, with the parallelism setting baked straight
into the dims (TP shards a weight dim or the attention heads; EP shards MoE experts and their
tokens). The grid is **workload-agnostic**, so we test raw shapes, not any one model: the labels
name a model + parallelism only for provenance.

For each shape we measure the kernel and ask predict.py for the latency both WITH the measured
efficiency grid (roofline × efficiency) and WITHOUT it (bare roofline) — the relative error shows
how much the benchmark data improves the analytic roofline.

    python3 validate_predict.py --bench gemm_bf16
    python3 validate_predict.py --bench attn_bf16 --csv fidelity.csv   # + append tidy per-shape CSV
    python3 validate_predict.py --bench moe_mxfp4
    python3 validate_predict.py --bench allreduce

--csv writes two files: the per-shape table (the given path) and a per-bench summary
(<stem>_summary.csv) with mean/median/min/max + latency-weighted error for the roofline-only
baseline and for roofline+efficiency (the grid). Reuse one --csv across benches to accumulate both.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ops.gemm import run_gemm_sweep
from predict import Predictor

# ── Model registry ───────────────────────────────────────────────────────────────────────────────
# hidden, n_q heads, n_kv heads, head_dim; dense models carry `ffn`, MoE models carry E/top_k/moe_ffn.
# attn="mla" -> skip attention (out of the GQA model). Each bench DERIVES its per-rank shapes from
# these and sweeps every parallelism setting (TP shards a weight dim / heads / MoE intermediate; EP
# shards MoE experts + tokens), so each (TP, EP) config is its own validation shape. Workload-agnostic
# in spirit: many models, but we test the raw shapes they produce, not any one workload.
MODELS: dict[str, dict] = {
    "llama3.1-8b":  dict(h=4096, nq=32,  nkv=8,   D=128, ffn=14336, vocab=128256),
    "llama3.3-70b": dict(h=8192, nq=64,  nkv=8,   D=128, ffn=28672, vocab=128256),
    "qwen2.5-72b":  dict(h=8192, nq=64,  nkv=8,   D=128, ffn=29568, vocab=152064),
    "mixtral-8x7b": dict(h=4096, nq=32,  nkv=8,   D=128, E=8,   tk=2, moe_ffn=14336, vocab=32000),
    "qwen3-235b":   dict(h=4096, nq=64,  nkv=4,   D=128, E=128, tk=8, moe_ffn=1536,  vocab=151936),
    "deepseek-v3":  dict(h=7168, nq=128, nkv=128, D=128, E=256, tk=8, moe_ffn=2048,  vocab=129280, attn="mla"),
    "gpt-oss-20b":  dict(h=2880, nq=64,  nkv=8,   D=64,  E=32,  tk=4, moe_ffn=2880,  vocab=201088),
    "qwen3-30b":    dict(h=2048, nq=32,  nkv=4,   D=128, E=128, tk=8, moe_ffn=768,   vocab=151936),
}
DECODE_M, PREFILL_M = 12, 2048          # batch tokens (not sharded) — decode / prefill regimes
TP_DEGREES = [1, 2, 4, 8]               # tensor-parallel degrees swept per shape


def _is_moe(m: dict) -> bool:
    return "E" in m


# ── GEMM: per model, derive the projections; TP shards N (col-parallel qkv/gate_up/lm_head) or K
#    (row-parallel o/down). Swept over TP × {decode, prefill}. ─────────────────────────────────────
def _gemm_cases():
    for name, m in MODELS.items():
        h, nq, nkv, D = m["h"], m["nq"], m["nkv"], m["D"]
        projs = [("qkv", h, (nq + 2 * nkv) * D, "N"), ("o", nq * D, h, "K"),
                 ("lm_head", h, m["vocab"], "N")]
        if not _is_moe(m):
            projs += [("gate_up", h, 2 * m["ffn"], "N"), ("down", m["ffn"], h, "K")]
        for proj, K, N, shard in projs:
            for tp in TP_DEGREES:
                Kr, Nr = (K, N // tp) if shard == "N" else (K // tp, N)
                for M in (DECODE_M, PREFILL_M):
                    yield (M, Kr, Nr, tp, f"{name} {proj} TP{tp} {'dec' if M <= DECODE_M else 'pre'}")


# ── Attention: per GQA model, TP shards the heads (n_heads/TP, n_kv_heads/TP); swept over TP × case.
ATTN_CASE = [   # (kind, n_req, q_len, kv_len)
    ("decode",  1,    1, 4096),
    ("prefill", 1, 2048, 2048),
    ("chunked", 1, 1024, 8192),
]


def _attn_cases():
    for name, m in MODELS.items():
        if m.get("attn") == "mla":                          # MLA is out of the GQA model
            continue
        for tp in TP_DEGREES:
            H, Hkv, D = max(1, m["nq"] // tp), max(1, m["nkv"] // tp), m["D"]
            for kind, nreq, ql, kvl in ATTN_CASE:
                yield (nreq, ql, kvl, H, Hkv, D, tp, f"{name} {kind} TP{tp}")


# ── MoE: per MoE model, TP shards the intermediate I; EP shards experts E + (uniform) tokens M
#    (per-rank M = tokens/EP); H stays full. Swept over (TP, EP) × {decode, prefill}. When EP shrinks
#    the per-rank expert count below top_k (e.g. mixtral E=8 at EP=8 -> 1 expert), top_k is capped at
#    the local expert count and the token count rebalanced so the routed-token work (M·top_k) is
#    preserved — top_k > E is an invalid routing config (out-of-bounds -> illegal memory access). ────
MOE_PAR = [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (4, 2)]   # (tp, ep)


def _moe_cases():
    for name, m in MODELS.items():
        if not _is_moe(m):
            continue
        E, tk, H, I = m["E"], m["tk"], m["h"], m["moe_ffn"]
        for tp, ep in MOE_PAR:
            E_rank = max(1, E // ep)
            tk_rank = min(tk, E_rank)                     # EP can't route to more experts than a rank holds
            for M in (1, PREFILL_M):
                T_rank = max(1, M // ep) * tk             # routed tokens on the rank (work to preserve)
                M_rank = max(1, T_rank // tk_rank)
                yield (M_rank, E_rank, tk_rank, H, I // tp, tp, ep,
                       f"{name} TP{tp}EP{ep} {'dec' if M == 1 else 'pre'}")


# ── All-reduce: the TP collective over [tokens, hidden]; bytes = tokens·hidden·2. Per model,
#    world size (= TP degree) × {decode, prefill} tokens; only W ≤ node GPU count is measured. ──────
AR_ELEM = 2   # bf16 activations (all-reduce cost is byte-driven)
AR_WORLD = [2, 4, 8]


def _allreduce_cases():
    for name, m in MODELS.items():
        for W in AR_WORLD:
            for tok in (1, PREFILL_M):
                yield (W, tok, m["h"], f"{name} TP{W} {'dec' if tok == 1 else 'pre'}")


GEMM_CASES = list(_gemm_cases())            # (M, K, N, tp, label)
ATTN_CASES = list(_attn_cases())            # (n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim, tp, label)
MOE_CASES = list(_moe_cases())              # (M, E, top_k, H, I, tp, ep, label)
ALLREDUCE_CASES = list(_allreduce_cases())  # (world_size, tokens, hidden, label)


# ── CSV: one tidy row per validation shape, appended across benches (reuse one --csv to accumulate;
#    header written once). Concat across GPUs and pivot by (bench, gpu). ───────────────────────────
CSV_COLUMNS = ["bench", "gpu", "label", "tp", "ep", "shape", "measured_ms", "pred_ms", "roof_ms",
               "pred_rel_err", "roof_rel_err"]


def _gpu_name(args) -> str:
    import torch
    try:
        return torch.cuda.get_device_name(args.device)
    except Exception:
        return "unknown"


def _row(bench: str, gpu: str, label: str, tp: int, ep: int, shape: str,
         meas: float, pred_ms: float, roof_ms: float) -> dict:
    """One CSV row (tp/ep = the tensor/expert-parallel degree baked into the shape); relative
    errors derived from the latencies so they always agree with them."""
    return {"bench": bench, "gpu": gpu, "label": label.strip(), "tp": tp, "ep": ep, "shape": shape,
            "measured_ms": f"{meas:.6f}", "pred_ms": f"{pred_ms:.6f}", "roof_ms": f"{roof_ms:.6f}",
            "pred_rel_err": f"{abs(pred_ms - meas) / meas:.6f}",
            "roof_rel_err": f"{abs(roof_ms - meas) / meas:.6f}"}


def _append_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    """Append rows to `path`, writing the header only if the file is new/empty."""
    p = Path(path)
    new = not p.exists() or p.stat().st_size == 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def _emit_csv(args, rows: list[dict]) -> None:
    if getattr(args, "csv", None) and rows:
        _append_csv(args.csv, rows, CSV_COLUMNS)


# One summary row per bench (companion CSV): mean/median/min/max (+ latency-weighted) for the
# roofline-only baseline and for roofline+efficiency (the grid).
SUMMARY_COLUMNS = ["bench", "gpu", "n_shapes",
                   "grid_mean", "grid_median", "grid_min", "grid_max", "grid_lat_wt",
                   "roof_mean", "roof_median", "roof_min", "roof_max", "roof_lat_wt"]


def _stats(errs, meas) -> dict:
    """mean/median/min/max/p90 of the relative errors + the latency-weighted error."""
    a, w = np.asarray(errs), np.asarray(meas)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "p90": float(np.percentile(a, 90)),
            "lat_wt": float((a * w).sum() / w.sum())}


def _summary(pred_all, roof_all, meas_all) -> tuple[dict, dict]:
    """Print the error summary and return (grid_stats, roofline_stats)."""
    grid, roof = _stats(pred_all, meas_all), _stats(roof_all, meas_all)
    print(f"\n  relative latency error over {len(pred_all)} shapes:")
    for name, s in [("roofline only (no grid)", roof), ("with efficiency grid  ", grid)]:
        print(f"    {name}:  mean {s['mean']*100:.1f}%  median {s['median']*100:.1f}%  "
              f"min {s['min']*100:.1f}%  max {s['max']*100:.1f}%  p90 {s['p90']*100:.1f}%")
    # latency-weighted: heavy (high-latency) shapes dominate, reflecting real serving cost.
    print(f"    latency-weighted (Σ|pred-meas| / Σ meas):  "
          f"grid {grid['lat_wt']*100:.1f}%   roofline {roof['lat_wt']*100:.1f}%")
    return grid, roof


def _emit_summary(args, bench: str, gpu: str, n: int, grid: dict, roof: dict) -> None:
    """Append one summary row for this bench to a companion CSV (<stem>_summary.csv)."""
    if not getattr(args, "csv", None):
        return
    p = Path(args.csv)
    row = {"bench": bench, "gpu": gpu, "n_shapes": n}
    for prefix, s in [("grid", grid), ("roof", roof)]:
        for k in ("mean", "median", "min", "max", "lat_wt"):
            row[f"{prefix}_{k}"] = f"{s[k]:.6f}"
    _append_csv(str(p.with_name(f"{p.stem}_summary{p.suffix}")), [row], SUMMARY_COLUMNS)


def validate_gemm(args, dtype: str) -> None:
    path = args.results or f"results/gemm_{dtype}.json"
    pred = Predictor.from_json(path)
    gpu = _gpu_name(args)

    # measure every distinct (K,N) over the distinct M's, then look each shape up
    shapes = {f"k{K}_n{N}": (K, N) for _, K, N, _, _ in GEMM_CASES}
    recs = run_gemm_sweep(shapes, sorted({M for M, *_ in GEMM_CASES}), [dtype],
                          device=args.device, iters=args.iters, warmup=args.warmup)
    meas_by = {(r.M, r.K, r.N): r.median_ms for r in recs}

    print(f"grid: {path}   gemm[{dtype}]   {len(GEMM_CASES)} shapes\n")
    print(f"  {'shape':30} {'M':>5} {'K':>6} {'N':>7} | {'pred':>5} {'roof':>5} | {'meas':>9}")
    pred_all, roof_all, meas_all, rows = [], [], [], []
    for M, K, N, tp, label in GEMM_CASES:
        meas = meas_by.get((M, K, N))
        if meas is None:
            continue
        pms, rms = pred.latency_ms(M, K, N, dtype), pred.roofline_ms(M, K, N, dtype)
        pred_all.append(abs(pms - meas) / meas); roof_all.append(abs(rms - meas) / meas)
        meas_all.append(meas)
        rows.append(_row(f"gemm_{dtype}", gpu, label, tp, 1, f"M{M}_K{K}_N{N}", meas, pms, rms))
        print(f"  {label:30} {M:>5} {K:>6} {N:>7} | {pred_all[-1]*100:>4.0f}% "
              f"{roof_all[-1]*100:>4.0f}% | {meas:8.3f} ms")
    grid, roof = _summary(pred_all, roof_all, meas_all)
    _emit_csv(args, rows)
    _emit_summary(args, f"gemm_{dtype}", gpu, len(pred_all), grid, roof)


def validate_attn(args) -> None:
    import ops.attn as attn
    path = args.results or "results/attn_bf16.json"
    pred = Predictor.from_json(path)
    gpu = _gpu_name(args)

    print(f"grid: {path}   attention   {len(ATTN_CASES)} shapes\n")
    print(f"  {'shape':28} {'n_req':>5} {'q_len':>5} {'kv_len':>6} {'heads':>9} | "
          f"{'pred':>5} {'roof':>5} | {'meas':>9}")
    pred_all, roof_all, meas_all, rows = [], [], [], []
    for n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim, tp, label in ATTN_CASES:
        try:
            meas = attn.measure_attn_ms(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim,
                                        device=args.device, iters=args.iters, warmup=args.warmup)
        except Exception as e:          # no FlashInfer kernel could run this shape -> skip
            print(f"  {label:28} {n_req:>5} {q_len:>5} {kv_len:>6} "
                  f"{f'{n_heads}/{n_kv_heads}/{head_dim}':>9} |    -     -  | "
                  f"skip ({str(e).splitlines()[0][:26]})")
            continue
        pms = pred.attn_latency_ms(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim)
        rms = pred.attn_roofline_ms(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim)
        pred_all.append(abs(pms - meas) / meas); roof_all.append(abs(rms - meas) / meas)
        meas_all.append(meas)
        shape = f"nreq{n_req}_qlen{q_len}_kvlen{kv_len}_nheads{n_heads}_nkvheads{n_kv_heads}_headdim{head_dim}"
        rows.append(_row("attn_bf16", gpu, label, tp, 1, shape, meas, pms, rms))
        print(f"  {label:28} {n_req:>5} {q_len:>5} {kv_len:>6} {f'{n_heads}/{n_kv_heads}/{head_dim}':>9} | "
              f"{pred_all[-1]*100:>4.0f}% {roof_all[-1]*100:>4.0f}% | {meas:8.4f} ms")
    grid, roof = _summary(pred_all, roof_all, meas_all)
    _emit_csv(args, rows)
    _emit_summary(args, "attn_bf16", gpu, len(pred_all), grid, roof)


def validate_moe(args, quant: str) -> None:
    import ops.moe as moe
    path = args.results or f"results/moe_{quant}.json"
    pred = Predictor.from_json(path)
    gpu = _gpu_name(args)

    # mxfp4/Marlin needs 128-aligned dims; pad H,I up to 128 like production does (no-op when aligned).
    pad = (lambda x: -(-x // 128) * 128) if quant == "mxfp4" else (lambda x: x)
    note = "  (mxfp4 H,I padded to 128)" if quant == "mxfp4" else ""
    print(f"grid: {path}   moe[{quant}]   {len(MOE_CASES)} shapes{note}\n")
    print(f"  {'shape':32} {'M':>5} {'E':>4} {'tk':>3} {'H':>5} {'I':>6} | {'pred':>5} {'roof':>5} | {'meas':>9}")
    pred_all, roof_all, meas_all, rows = [], [], [], []
    for M, E, tk, H0, I0, tp, ep, label in MOE_CASES:
        H, I = pad(H0), pad(I0)
        try:
            meas = moe.measure_moe_ms(M, E, tk, H, I, quant=quant, device=args.device,
                                      iters=args.iters, warmup=args.warmup)
        except Exception as e:          # unsupported / OOM shape -> skip, don't crash
            msg = str(e).splitlines()[0]
            # A device-side fault (illegal access / assert) kills the CUDA context: every later
            # shape would fail the same way, so abort the bench instead of cascading fake skips.
            if "illegal memory access" in msg or "device-side assert" in msg:
                print(f"  {label:32} {M:>5} {E:>4} {tk:>3} {H:>5} {I:>6} |    -     -  | FATAL")
                print(f"\n  ** device-side CUDA fault on this shape corrupts the context — aborting "
                      f"moe[{quant}] ({len(pred_all)}/{len(MOE_CASES)} measured). **")
                break
            print(f"  {label:32} {M:>5} {E:>4} {tk:>3} {H:>5} {I:>6} |    -     -  | skip ({msg[:30]})")
            continue
        pms, rms = pred.moe_latency_ms(M, E, tk, H, I), pred.moe_roofline_ms(M, E, tk, H, I)
        pred_all.append(abs(pms - meas) / meas); roof_all.append(abs(rms - meas) / meas)
        meas_all.append(meas)
        rows.append(_row(f"moe_{quant}", gpu, label, tp, ep, f"M{M}_E{E}_tk{tk}_H{H}_I{I}", meas, pms, rms))
        print(f"  {label:32} {M:>5} {E:>4} {tk:>3} {H:>5} {I:>6} | {pred_all[-1]*100:>4.0f}% "
              f"{roof_all[-1]*100:>4.0f}% | {meas:8.3f} ms")
    if pred_all:
        grid, roof = _summary(pred_all, roof_all, meas_all)
        _emit_csv(args, rows)
        _emit_summary(args, f"moe_{quant}", gpu, len(pred_all), grid, roof)
    else:
        print("\n  (no measurable shapes — all rejected by the kernel)")


def validate_allreduce(args) -> None:
    import torch
    from ops.allreduce import measure_allreduce_ms

    path = args.results or "results/allreduce.json"
    pred = Predictor.from_json(path)
    if pred.op != "allreduce":
        raise SystemExit(f"{path} is not an all-reduce grid (op={pred.op}).")
    gpu = _gpu_name(args)
    n_gpus = torch.cuda.device_count()

    by_w: dict[int, list] = {}                          # world_size -> [(tokens, hidden, bytes, label)]
    for W, tok, hid, label in ALLREDUCE_CASES:
        by_w.setdefault(W, []).append((tok, hid, tok * hid * AR_ELEM, label))

    print(f"grid: {path}   all-reduce   {len(ALLREDUCE_CASES)} shapes  ({n_gpus} GPUs on node)")
    pred_all, roof_all, meas_all, rows = [], [], [], []
    for W in sorted(by_w):
        if W not in pred.ar_curves or W > n_gpus:
            print(f"  [skip W={W}] not measurable (grid world sizes {sorted(pred.ar_curves)}, "
                  f"{n_gpus} GPUs)")
            continue
        cases = by_w[W]
        measured = measure_allreduce_ms(W, sorted({b for *_, b, _ in cases}),
                                        iters=args.iters, warmup=args.warmup)
        print(f"\n  == world_size {W} ==")
        print(f"  {'shape':24} {'tokens':>6} {'hidden':>6} {'bytes':>10} | {'pred':>5} {'roof':>5} | {'meas':>10}")
        for tok, hid, b, label in cases:
            meas = measured[b]
            pms, rms = pred.allreduce_latency_ms(b, W), pred.allreduce_roofline_ms(b, W)
            pred_all.append(abs(pms - meas) / meas); roof_all.append(abs(rms - meas) / meas)
            meas_all.append(meas)
            rows.append(_row("allreduce", gpu, label, W, 1, f"W{W}_tok{tok}_hid{hid}_bytes{b}", meas, pms, rms))
            print(f"  {label:24} {tok:>6} {hid:>6} {b:>10} | {pred_all[-1]*100:>4.0f}% "
                  f"{roof_all[-1]*100:>4.0f}% | {meas:8.4f} ms")
    if pred_all:
        grid, roof = _summary(pred_all, roof_all, meas_all)
        _emit_csv(args, rows)
        _emit_summary(args, "allreduce", gpu, len(pred_all), grid, roof)
    else:
        print("\n  (no measurable world sizes on this node)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="gemm_bf16",
                    choices=["gemm_bf16", "gemm_fp16", "attn_bf16", "moe_bf16", "moe_mxfp4", "allreduce"],
                    help="which grid to validate (matches run.py).")
    ap.add_argument("--results", default=None, help="grid JSON (default: results/<bench>.json).")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--csv", default=None,
                    help="append per-shape results to this CSV (one tidy table across benches/GPUs).")
    args = ap.parse_args()

    op, _, dtype = args.bench.partition("_")
    if op == "allreduce":
        validate_allreduce(args)
    elif op == "attn":
        validate_attn(args)
    elif op == "moe":
        validate_moe(args, dtype)
    else:
        validate_gemm(args, dtype)


if __name__ == "__main__":
    main()
