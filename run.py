"""Run the GEMM grid sweep, derive C_peak / B_peak, dump JSON.

    python3 run.py --dtypes bf16             # whole grid, bf16 (cuBLAS)
    python3 run.py --dtypes mxfp4            # w4a16 mxfp4 (vLLM Marlin)
    python3 run.py --shapes k2048_n4096 --dtypes bf16

bf16/fp16 go through cuBLASLt; mxfp4 goes through the vLLM Marlin w4a16 kernel
(needs vLLM) with its own byte model. Run mxfp4 separately from bf16/fp16 — they
have different kernels, ceilings, and byte models, so different output files.
Needs torch + a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from gemm import (
    DEFAULT_MS,
    SHAPES,
    derive_c_peak,
    roofline_residual,
    run_gemm_sweep,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"],
                    help="bf16/fp16 (cuBLAS) or mxfp4 (Marlin w4a16); not mixed.")
    ap.add_argument("--shapes", nargs="+", default=None,
                    help="Subset of SHAPES keys (default: all).")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this benchmark needs a GPU.")

    dev = args.device
    props = torch.cuda.get_device_properties(dev)
    l2_mb = getattr(props, "L2_cache_size", 0) / 1e6
    print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB | "
          f"L2 {l2_mb:.0f} MB | torch {torch.__version__}")

    shapes = SHAPES if not args.shapes else {k: SHAPES[k] for k in args.shapes}

    cublas = [d for d in args.dtypes if d in ("bf16", "fp16")]
    quant = [d for d in args.dtypes if d == "mxfp4"]
    if cublas and quant:
        raise SystemExit("run mxfp4 separately from bf16/fp16 — different kernel, "
                         "ceilings, and byte model (different output files).")
    if not cublas and not quant:
        raise SystemExit(f"unknown --dtypes {args.dtypes}; use bf16, fp16, or mxfp4.")

    if quant:
        # mxfp4 w4a16 via vLLM Marlin (imported lazily — pulls in vLLM).
        import vllm
        from marlin import BYTES_MODEL, marlin_roofline_residual, run_marlin_sweep
        print("\n== mxfp4 gemm (Marlin w4a16) ==")
        recs = run_marlin_sweep(shapes, DEFAULT_MS,
                                device=dev, iters=args.iters, warmup=args.warmup)
        c_peak = derive_c_peak(recs)
        b_peak = max(r.gbps for r in recs)
        marlin_roofline_residual(recs, c_peak["mxfp4"][0], b_peak)
        scheme, bytes_model = "mxfp4_w4a16", BYTES_MODEL
        lib = {"vllm": vllm.__version__}          # the Marlin kernel ships in vLLM
        default_out = f"results/marlin_mxfp4_{props.name.replace(' ', '_')}.json"
    else:
        print("\n== gemm (cuBLAS) ==")
        recs = run_gemm_sweep(shapes, DEFAULT_MS, cublas,
                              device=dev, iters=args.iters, warmup=args.warmup)
        c_peak = derive_c_peak(recs)
        # B_peak: best memory throughput the sweep itself reached (small-M, large-
        # weight GEMMs are HBM-read-bound), so no separate bandwidth probe needed.
        b_peak = max(r.gbps for r in recs)
        roofline_residual(recs, c_peak, b_peak)
        scheme, bytes_model = "+".join(cublas), {"w": 2.0, "a": 2.0, "o": 2.0}
        lib = {"torch": torch.__version__}        # cuBLASLt ships in torch
        default_out = f"results/gemm_{props.name.replace(' ', '_')}.json"

    for dt, (tf, where) in c_peak.items():
        print(f"  C_peak[{dt}] (achieved): {tf:7.0f} TFLOP/s  @ {where}")
    print(f"  B_peak (achieved): {b_peak:7.0f} GB/s")

    # Worst roofline residuals — where measured most exceeds the model.
    worst = sorted(recs, key=lambda r: r.residual, reverse=True)[:8]
    print("  largest roofline residuals (measured / predicted):")
    for r in worst:
        print(f"    {r.dtype} {r.shape:16} M={r.M:<5} "
              f"resid {r.residual:4.1f}x  ({r.median_ms:.3f} ms vs {r.predicted_ms:.3f})")

    out = Path(args.out or default_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gpu": props.name,
        "lib": lib,
        "scheme": scheme,
        "b_peak_gbps": b_peak,
        "c_peak": {dt: tf for dt, (tf, _) in c_peak.items()},
        "bytes_model": bytes_model,
        "gemm": [asdict(r) for r in recs],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
