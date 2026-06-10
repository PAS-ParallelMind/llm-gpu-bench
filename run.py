"""Run the GEMM grid sweep, score efficiency vs the theoretical roofline, dump JSON.

    python3 run.py --dtypes bf16  --c-peak 165 --b-peak 1008    # RTX 4090, bf16
    python3 run.py --dtypes mxfp4 --c-peak 165 --b-peak 1008    # w4a16 (vLLM Marlin)
    python3 run.py --shapes k2048_n4096 --dtypes bf16 --c-peak 165 --b-peak 1008

bf16/fp16 go through torch's F.linear (torch picks the GEMM backend —
cuBLAS / cuBLASLt / CUTLASS — per shape); mxfp4 goes through the vLLM Marlin w4a16
kernel (needs vLLM) with its own byte model. Run mxfp4 separately from bf16/fp16 —
different kernels and byte models, so different output files.

The roofline ceiling is the GPU's theoretical peak, passed via --c-peak (tensor-core
TFLOP/s) and --b-peak (memory GB/s). Its absolute scale cancels in the predictor (it
only normalizes the efficiency factor), so any consistent value works. Needs torch +
a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from gemm import DEFAULT_MS, SHAPES, roofline_residual, run_gemm_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"],
                    help="bf16/fp16 (torch F.linear) or mxfp4 (Marlin w4a16); not mixed.")
    ap.add_argument("--shapes", nargs="+", default=None,
                    help="Subset of SHAPES keys (default: all).")
    ap.add_argument("--c-peak", type=float, required=True,
                    help="Theoretical compute peak TFLOP/s (e.g. RTX 4090: 165).")
    ap.add_argument("--b-peak", type=float, required=True,
                    help="Theoretical memory bandwidth GB/s (e.g. RTX 4090: 1008).")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this benchmark needs a GPU.")

    dev = args.device
    props = torch.cuda.get_device_properties(dev)
    l2_mb = getattr(props, "L2_cache_size", 0) / 1024**2
    print(f"GPU: {props.name} | {props.total_memory / 1024**3:.1f} GB | "
          f"L2 {l2_mb:.0f} MB | torch {torch.__version__}")

    shapes = SHAPES if not args.shapes else {k: SHAPES[k] for k in args.shapes}

    fp_dtypes = [d for d in args.dtypes if d in ("bf16", "fp16")]
    quant = [d for d in args.dtypes if d == "mxfp4"]
    if fp_dtypes and quant:
        raise SystemExit("run mxfp4 separately from bf16/fp16 — different kernel "
                         "and byte model (different output files).")
    if not fp_dtypes and not quant:
        raise SystemExit(f"unknown --dtypes {args.dtypes}; use bf16, fp16, or mxfp4.")

    c_peak, b_peak = args.c_peak, args.b_peak

    if quant:
        # mxfp4 w4a16 via vLLM Marlin (imported lazily — pulls in vLLM).
        import vllm
        from marlin import BYTES_MODEL, marlin_roofline_residual, run_marlin_sweep
        print("\n== mxfp4 gemm (Marlin w4a16) ==")
        recs = run_marlin_sweep(shapes, DEFAULT_MS,
                                device=dev, iters=args.iters, warmup=args.warmup)
        marlin_roofline_residual(recs, c_peak, b_peak)
        c_peaks = {"mxfp4": c_peak}
        scheme, bytes_model = "mxfp4_w4a16", BYTES_MODEL
        lib = {"vllm": vllm.__version__}                  # the Marlin kernel ships in vLLM
        default_out = f"results/marlin_mxfp4_{props.name.replace(' ', '_')}.json"
    else:
        print("\n== gemm (torch F.linear) ==")
        recs = run_gemm_sweep(shapes, DEFAULT_MS, fp_dtypes,
                              device=dev, iters=args.iters, warmup=args.warmup)
        roofline_residual(recs, {dt: (c_peak, "input") for dt in fp_dtypes}, b_peak)
        c_peaks = {dt: c_peak for dt in fp_dtypes}
        scheme, bytes_model = "+".join(fp_dtypes), {"w": 2.0, "a": 2.0, "o": 2.0}
        lib = {"torch": torch.__version__}                # the F.linear GEMM ships in torch
        default_out = f"results/gemm_{props.name.replace(' ', '_')}.json"

    # achieved throughput, as a sanity check against the input ceiling
    achieved_c = max(r.tflops for r in recs)
    achieved_b = max(r.gbps for r in recs)
    print(f"  roofline ceiling (input):  C_peak {c_peak:7.0f} TFLOP/s   B_peak {b_peak:6.0f} GB/s")
    print(f"  achieved (this sweep):     C_peak {achieved_c:7.0f} TFLOP/s   B_peak {achieved_b:6.0f} GB/s")

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
        "c_peak": c_peaks,
        "achieved": {"c_tflops": round(achieved_c, 1), "b_gbps": round(achieved_b, 1)},
        "bytes_model": bytes_model,
        "gemm": [asdict(r) for r in recs],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
