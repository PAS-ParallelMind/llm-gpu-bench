"""Run a kernel benchmark, score efficiency vs the roofline, dump JSON.

    python3 run.py --bench gemm_bf16   --c-peak 165 --b-peak 1008   # RTX 4090
    python3 run.py --bench gemm_mxfp4  --c-peak 165 --b-peak 1008   # w4a16 (vLLM Marlin)
    python3 run.py --bench attn_bf16   --c-peak 165 --b-peak 1008   # flash-attn (decode+prefill)
    python3 run.py --bench gemm_bf16 --shapes k2048_n4096 --c-peak 165 --b-peak 1008

--bench is <op>_<dtype>: gemm_bf16 / gemm_fp16 go through torch's F.linear (cuBLAS /
cuBLASLt / CUTLASS per shape); gemm_mxfp4 goes through the vLLM Marlin w4a16 kernel;
attn_bf16 sweeps flash-attention (vLLM FlashAttention, paged KV) — a decode KV-byte
curve plus a prefill (S_q, S_kv, R·H) × D grid. Each writes its own results file.

GEMM and prefill attention need the theoretical roofline ceiling via --c-peak (TFLOP/s)
and --b-peak (GB/s); decode attention is always memory-bound, so --b-peak carries it. The
ceiling's absolute scale cancels in the predictor (it only normalizes the efficiency
factor). Needs torch + a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from gemm import DEFAULT_MS, SHAPES, roofline_residual, run_gemm_sweep


def _attn_module(backend: str):
    """Select the attention backend implementation (same hybrid API, different kernels)."""
    if backend == "flashinfer":
        import attn_flashinfer as m
    else:
        import attn as m
    return m


def _run_attn(args, props, dtype: str) -> None:
    """Flash-attention: decode KV-byte curve + prefill (S_q, S_kv, R*H) x D grid."""
    import vllm
    mod = _attn_module(args.attn_backend)
    label = {"flash_attn": "FlashAttention", "flashinfer": "FlashInfer"}[args.attn_backend]
    print(f"\n== attn ({label}, paged KV) ==")
    decode, grid = mod.run_full_attn_sweep(c_peak=args.c_peak, b_peak=args.b_peak, dtype=dtype,
                                           device=args.device, iters=args.iters, warmup=args.warmup)
    print(f"  ceiling: C_peak {args.c_peak:.0f} TFLOP/s   B_peak {args.b_peak:.0f} GB/s")
    print(f"  decode curve {len(decode)} pts ({decode[0].efficiency:.2f}..{decode[-1].efficiency:.2f})"
          f"   prefill grid {len(grid)} pts")
    out = Path(args.out or f"results/attn_{args.attn_backend}_{props.name.replace(' ', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gpu": props.name,
        "lib": {"vllm": vllm.__version__},
        "op": "attn",
        "backend": args.attn_backend,
        "c_peak_tflops": args.c_peak,
        "b_peak_gbps": args.b_peak,
        "decode": [asdict(r) for r in decode],
        "grid": [asdict(r) for r in grid],
    }, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="gemm_bf16",
                    choices=["gemm_bf16", "gemm_fp16", "gemm_mxfp4", "attn_bf16"],
                    help="which benchmark to run (<op>_<dtype>).")
    ap.add_argument("--attn-backend", default="flash_attn",
                    choices=["flash_attn", "flashinfer"],
                    help="attention backend (attn only): vLLM FA varlen or FlashInfer wrappers.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shapes", nargs="+", default=None,
                    help="Subset of SHAPES keys (gemm only; default: all).")
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

    op, _, dtype = args.bench.partition("_")   # op in {gemm, attn}; dtype in {bf16, fp16, mxfp4}

    if op == "attn":
        _run_attn(args, props, dtype)
        return

    shapes = SHAPES if not args.shapes else {k: SHAPES[k] for k in args.shapes}
    c_peak, b_peak = args.c_peak, args.b_peak

    if dtype == "mxfp4":
        # mxfp4 w4a16 via vLLM Marlin (imported lazily — pulls in vLLM).
        import vllm
        from marlin import BYTES_MODEL, marlin_roofline_residual, run_marlin_sweep
        print("\n== gemm_mxfp4 (Marlin w4a16) ==")
        recs = run_marlin_sweep(shapes, DEFAULT_MS,
                                device=dev, iters=args.iters, warmup=args.warmup)
        marlin_roofline_residual(recs, c_peak, b_peak)
        c_peaks, bytes_model = {"mxfp4": c_peak}, BYTES_MODEL
        lib = {"vllm": vllm.__version__}                  # the Marlin kernel ships in vLLM
        default_out = f"results/marlin_mxfp4_{props.name.replace(' ', '_')}.json"
    else:
        print("\n== gemm (torch F.linear) ==")
        recs = run_gemm_sweep(shapes, DEFAULT_MS, [dtype],
                              device=dev, iters=args.iters, warmup=args.warmup)
        roofline_residual(recs, {dtype: (c_peak, "input")}, b_peak)
        c_peaks, bytes_model = {dtype: c_peak}, {"w": 2.0, "a": 2.0, "o": 2.0}
        lib = {"torch": torch.__version__}                # the F.linear GEMM ships in torch
        default_out = f"results/gemm_{props.name.replace(' ', '_')}.json"

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
        "bench": args.bench,
        "op": op,
        "b_peak_gbps": b_peak,
        "c_peak": c_peaks,
        "achieved": {"c_tflops": round(achieved_c, 1), "b_gbps": round(achieved_b, 1)},
        "bytes_model": bytes_model,
        "gemm": [asdict(r) for r in recs],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
