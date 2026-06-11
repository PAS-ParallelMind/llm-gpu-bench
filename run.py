"""Run a kernel benchmark, score efficiency vs the roofline, dump JSON.

    python3 run.py --bench gemm_bf16   --c-peak 165 --b-peak 1008   # RTX 4090
    python3 run.py --bench gemm_mxfp4  --c-peak 165 --b-peak 1008   # w4a16 (vLLM Marlin)
    python3 run.py --bench attn_bf16   --c-peak 165 --b-peak 1008   # flash-attn (decode+prefill)
    python3 run.py --bench gemm_bf16 --shapes k2048_n4096 --c-peak 165 --b-peak 1008

--bench is <op>_<dtype>: gemm_bf16 / gemm_fp16 go through torch's F.linear (cuBLAS /
cuBLASLt / CUTLASS per shape); gemm_mxfp4 goes through the vLLM Marlin w4a16 kernel;
attn_bf16 sweeps flash-attention (FlashInfer, paged KV; best of its fa2/fa3/cutlass/
trtllm-gen kernels per shape) — a decode KV-byte curve plus a prefill (S_q, S_kv, R·H)
× D grid. Each writes its own results file.

GEMM and prefill attention need the theoretical roofline ceiling via --c-peak (TFLOP/s)
and --b-peak (GB/s); decode attention is always memory-bound, so --b-peak carries it. The
ceiling's absolute scale cancels in the predictor (it only normalizes the efficiency
factor). Needs torch + a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gemm import BYTES_MODEL, DEFAULT_MS, SHAPES, roofline_residual, run_gemm_sweep


def _dump(out, gpu, c_peak, b_peak, bench, impl, bytes_model, results) -> None:
    """Write the unified result JSON: hardware / operation / per-case results."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "hardware": {"gpu": gpu, "c_peak_tflops": c_peak, "b_peak_gbps": b_peak},
        "operation": {"bench": bench, "impl": impl, "bytes_model": bytes_model},
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")


def _run_attn(args, props, dtype: str) -> None:
    """Flash-attention (FlashInfer, best of kernels): decode KV-byte curve + prefill grid."""
    import flashinfer
    from attn import run_full_attn_sweep
    print("\n== attn (FlashInfer, paged KV, best of backends) ==")
    decode, grid = run_full_attn_sweep(c_peak=args.c_peak, b_peak=args.b_peak, dtype=dtype,
                                       device=args.device, iters=args.iters, warmup=args.warmup)
    print(f"  ceiling: C_peak {args.c_peak:.0f} TFLOP/s   B_peak {args.b_peak:.0f} GB/s")
    print(f"  decode curve {len(decode)} pts   prefill grid {len(grid)} pts")
    won: dict[str, int] = {}
    for r in decode + grid:
        won[r.backend] = won.get(r.backend, 0) + 1
    print("  winning kernels: " + ", ".join(f"{k}×{v}" for k, v in sorted(won.items())))
    results = [r.result() for r in decode] + [r.result() for r in grid]
    out = args.out or f"results/{args.bench}.json"
    _dump(out, props.name, args.c_peak, args.b_peak, args.bench,
          {"flashinfer": flashinfer.__version__}, {"elem": 2}, results)


def _run_moe(args, props, dtype: str) -> None:
    """MoE: token-routed grouped GEMM, two-grouped-GEMM roofline. dtype selects the scheme:
    bf16 -> fused_experts (Triton); mxfp4 -> fused_marlin_moe (w4a16 Marlin)."""
    import vllm
    from moe import WEIGHT_BYTES, run_full_moe_sweep
    quant = dtype                                      # "bf16" or "mxfp4"
    label = {"bf16": "fused_experts Triton", "mxfp4": "fused_marlin_moe w4a16"}[quant]
    print(f"\n== moe[{quant}] (vLLM {label}, token-routed grouped GEMM) ==")
    recs = run_full_moe_sweep(c_peak=args.c_peak, b_peak=args.b_peak, quant=quant, dtype="bf16",
                              device=args.device, iters=args.iters, warmup=args.warmup)
    print(f"  ceiling: C_peak {args.c_peak:.0f} TFLOP/s   B_peak {args.b_peak:.0f} GB/s")
    print(f"  {len(recs)} points; efficiency "
          f"{min(r.efficiency for r in recs):.2f} .. {max(r.efficiency for r in recs):.2f}")
    out = args.out or f"results/{args.bench}.json"
    _dump(out, props.name, args.c_peak, args.b_peak, args.bench,
          {"vllm": vllm.__version__}, {"w": WEIGHT_BYTES[quant], "a": 2.0},
          [r.result() for r in recs])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="gemm_bf16",
                    choices=["gemm_bf16", "gemm_fp16", "gemm_mxfp4", "attn_bf16", "moe_bf16", "moe_mxfp4"],
                    help="which benchmark to run (<op>_<dtype>).")
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

    if op == "moe":
        _run_moe(args, props, dtype)
        return

    shapes = SHAPES if not args.shapes else {k: SHAPES[k] for k in args.shapes}
    c_peak, b_peak = args.c_peak, args.b_peak

    if dtype == "mxfp4":
        import vllm                                        # Marlin kernel ships in vLLM
        print("\n== gemm_mxfp4 (Marlin w4a16) ==")
        lib = {"vllm": vllm.__version__}
    else:
        print("\n== gemm (torch F.linear) ==")
        lib = {"torch": torch.__version__}                # the F.linear GEMM ships in torch
    default_out = f"results/{args.bench}.json"
    recs = run_gemm_sweep(shapes, DEFAULT_MS, [dtype],
                          device=dev, iters=args.iters, warmup=args.warmup)
    roofline_residual(recs, {dtype: (c_peak, "input")}, b_peak)

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

    _dump(args.out or default_out, props.name, c_peak, b_peak, args.bench,
          lib, BYTES_MODEL[dtype], [r.result() for r in recs])


if __name__ == "__main__":
    main()
