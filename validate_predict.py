"""Validate the predictor on REAL model workloads it never measured.

Measures real workloads, then asks predict.py for the latency both *with* the
model-agnostic grid (roofline × measured efficiency) and *without* it (bare
roofline, no benchmark data) — so the relative error shows how much the benchmark
data improves the analytic roofline.

    python3 validate_predict.py                       # gemm_bf16 (real projections)
    python3 validate_predict.py --bench gemm_mxfp4    # mxfp4 w4a16
    python3 validate_predict.py --bench attn_bf16     # decode attention, real head configs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gemm import run_gemm_sweep
from predict import Predictor

# GEMM: real projections — the grid (gemm.SHAPES) is keyed on none of these.
MODEL_SHAPES: dict[str, tuple[int, int]] = {
    # gpt-oss-20b  (hidden 2880, head_dim 64, GQA 64/8)
    "gptoss_qkv":     (2880, 5120),     # 64*(64 + 2*8)
    "gptoss_o":       (4096, 2880),     # 64*64 -> 2880
    "gptoss_lmhead":  (2880, 201088),
    "gptoss_moe_up":  (2880, 5760),     # one expert, up+gate (2*2880)
    "gptoss_moe_dn":  (2880, 2880),     # one expert, down
    # Qwen3-Coder-30B-A3B  (hidden 2048, head_dim 128, GQA 32/4)
    "qwen_qkv":       (2048, 5120),     # 128*(32 + 2*4)
    "qwen_o":         (4096, 2048),     # 128*32 -> 2048
    "qwen_lmhead":    (2048, 151936),
    "qwen_moe_up":    (2048, 1536),     # one expert, up+gate (2*768)
    "qwen_moe_dn":    (768, 2048),      # one expert, down
}
VAL_MS = [1, 4, 16, 64, 256, 1024, 4096]

# Attention: real head configs (H, H_kv, D) × cases covering decode / full prefill /
# chunked prefill. S_q, S_kv are mostly off-grid to exercise interpolation.
ATTN_CONFIGS: dict[str, tuple[int, int, int]] = {
    "gptoss": (64, 8, 64),
    "qwen":   (32, 4, 128),
}
ATTN_CASES = [   # (kind, R, S_q, S_kv)
    ("decode",   1,    1,  2048),
    ("decode",  32,    1,  8192),
    ("prefill",  1,  512,   512),
    ("prefill",  2, 2048,  2048),
    ("prefill",  1, 8192,  8192),
    ("chunked",  1,  512,  4096),
    ("chunked",  4, 2048,  8192),
    ("chunked",  1,  256, 16384),
]


def _summary(pred_all, roof_all) -> None:
    pe, re = np.array(pred_all), np.array(roof_all)
    print(f"\n  relative latency error over {len(pe)} points:")
    print(f"    roofline only (no grid):  mean {re.mean()*100:.1f}%  median {np.median(re)*100:.1f}%  "
          f"p90 {np.percentile(re,90)*100:.1f}%  max {re.max()*100:.1f}%")
    print(f"    with efficiency grid:     mean {pe.mean()*100:.1f}%  median {np.median(pe)*100:.1f}%  "
          f"p90 {np.percentile(pe,90)*100:.1f}%  max {pe.max()*100:.1f}%")


def validate_gemm(args, dtype: str) -> None:
    if dtype == "mxfp4":
        from marlin import run_marlin_sweep
        glob = "marlin_mxfp4_*.json"
        recs = run_marlin_sweep(MODEL_SHAPES, VAL_MS, device=args.device,
                                iters=args.iters, warmup=args.warmup)
    else:
        glob = "gemm_*.json"
        recs = run_gemm_sweep(MODEL_SHAPES, VAL_MS, [dtype], device=args.device,
                              iters=args.iters, warmup=args.warmup)
    path = args.results or str(sorted(Path("results").glob(glob))[-1])
    pred = Predictor.from_json(path)

    print(f"grid: {path}   {dtype}   validating {len(MODEL_SHAPES)} real projections\n")
    print(f"  {'shape':14} {'K':>6} {'N':>7} | "
          + " ".join(f"M={m:<5}" for m in VAL_MS) + "  | pred  roof")
    pred_all, roof_all = [], []
    for name, (K, N) in MODEL_SHAPES.items():
        cells, pe, re = [], [], []
        for m in VAL_MS:
            r = next((x for x in recs if x.shape == name and x.M == m), None)
            if r is None:                       # shape Marlin couldn't run
                cells.append("   -  "); continue
            meas = r.median_ms
            ep = abs(pred.latency_ms(m, K, N, dtype) - meas) / meas
            er = abs(pred.roofline_ms(m, K, N, dtype) - meas) / meas
            cells.append(f"{ep*100:5.0f}%")
            pe.append(ep); re.append(er); pred_all.append(ep); roof_all.append(er)
        tail = (f"{np.mean(pe)*100:4.0f}%  {np.mean(re)*100:4.0f}%" if pe
                else "n/a (unsupported)")
        print(f"  {name:14} {K:>6} {N:>7} | " + " ".join(f"{c:>7}" for c in cells)
              + f"  | {tail}")
    _summary(pred_all, roof_all)


def validate_attn(args) -> None:
    import attn
    path = args.results or str(sorted(Path("results").glob("attn_*.json"))[-1])
    pred = Predictor.from_json(path)

    print(f"grid: {path}   attention   {len(ATTN_CONFIGS)} head configs x "
          f"{len(ATTN_CASES)} cases (decode / prefill / chunked)\n")
    print(f"  {'config':8} {'kind':8} {'R':>3} {'S_q':>5} {'S_kv':>6} | {'pred':>5} {'roof':>5}")
    pred_all, roof_all = [], []
    for name, (H, H_kv, D) in ATTN_CONFIGS.items():
        for kind, R, Sq, Sk in ATTN_CASES:
            meas = attn.measure_attn_ms(R, Sq, Sk, H, H_kv, D, device=args.device,
                                        iters=args.iters, warmup=args.warmup)
            ep = abs(pred.attn_latency_ms(R, Sq, Sk, H, H_kv, D) - meas) / meas
            er = abs(pred.attn_roofline_ms(R, Sq, Sk, H, H_kv, D) - meas) / meas
            pred_all.append(ep); roof_all.append(er)
            print(f"  {name:8} {kind:8} {R:>3} {Sq:>5} {Sk:>6} | "
                  f"{ep*100:>4.0f}% {er*100:>4.0f}%")
    _summary(pred_all, roof_all)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="gemm_bf16",
                    choices=["gemm_bf16", "gemm_fp16", "gemm_mxfp4", "attn_bf16"])
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    op, _, dtype = args.bench.partition("_")
    if op == "attn":
        validate_attn(args)
    else:
        validate_gemm(args, dtype)


if __name__ == "__main__":
    main()
