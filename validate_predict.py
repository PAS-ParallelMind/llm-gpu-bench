"""Validate the grid predictor on REAL model projections it never measured.

Measures each MODEL_SHAPES projection across a decode->prefill M sweep, then asks
predict.py for the latency using only the model-agnostic grid. Reports
|predicted - measured| / measured. A grid keyed on no model should still predict
every model's shapes.

    python3 validate_predict.py                  # bf16 (torch F.linear) grid
    python3 validate_predict.py --dtype mxfp4    # mxfp4 w4a16 (Marlin) grid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gemm import run_gemm_sweep
from predict import Predictor

# Real projections used only for validation — the grid (gemm.SHAPES) is keyed on
# none of these. (K = input dim, N = output dim of y = x @ W^T.)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "mxfp4"])
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    if args.dtype == "mxfp4":
        from marlin import run_marlin_sweep
        glob = "marlin_mxfp4_*.json"
        recs = run_marlin_sweep(MODEL_SHAPES, VAL_MS, device=args.device,
                                iters=args.iters, warmup=args.warmup)
    else:
        glob = "gemm_*.json"
        recs = run_gemm_sweep(MODEL_SHAPES, VAL_MS, [args.dtype], device=args.device,
                              iters=args.iters, warmup=args.warmup)

    path = args.results or str(sorted(Path("results").glob(glob))[-1])
    pred = Predictor.from_json(path)

    print(f"grid: {path}   dtype {args.dtype}   validating {len(MODEL_SHAPES)} real projections\n")
    print(f"  {'shape':14} {'K':>6} {'N':>7} | "
          + " ".join(f"M={m:<5}" for m in VAL_MS) + "  | mean")
    all_err = []
    for name, (K, N) in MODEL_SHAPES.items():
        cells, errs = [], []
        for m in VAL_MS:
            r = next((x for x in recs if x.shape == name and x.M == m), None)
            if r is None:                       # shape Marlin couldn't run
                cells.append("   -  "); continue
            pm = pred.latency_ms(m, K, N, args.dtype)
            e = abs(pm - r.median_ms) / r.median_ms
            errs.append(e); all_err.append(e)
            cells.append(f"{e*100:5.0f}%")
        tail = f"{np.mean(errs)*100:4.0f}%" if errs else "n/a (unsupported)"
        print(f"  {name:14} {K:>6} {N:>7} | " + " ".join(f"{c:>7}" for c in cells)
              + f"  | {tail}")
    a = np.array(all_err)
    print(f"\n  latency error over {len(a)} (shape,M): "
          f"mean {a.mean()*100:.1f}%  median {np.median(a)*100:.1f}%  "
          f"p90 {np.percentile(a,90)*100:.1f}%  max {a.max()*100:.1f}%")


if __name__ == "__main__":
    main()
