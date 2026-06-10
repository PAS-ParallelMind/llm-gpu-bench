"""Predict kernel latency from a measured grid (op auto-detected from the JSON).

GEMM (op=gemm):  t = roofline(C_peak, B_peak) / efficiency(M, K, N), with efficiency
by trilinear interpolation in (log M, log K, log N) over the model-agnostic grid.

Decode attention (op=attn_decode):  t = (KV_bytes / B_peak) / f(KV_bytes), a single
1-D curve in total (block-padded) KV bytes — decode is always memory-bound and the
curve is model-agnostic. Holds for per-request context >~ 128 tokens; it over-predicts
efficiency for large-batch x very-short-context (see attn.py).

Pure stdlib — prediction needs no GPU or torch. Measurement lives in run.py.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def _kv_bytes(kv_tokens: int, H_kv: int, D: int, elem: int = 2) -> float:
    """Total KV-cache bytes read in a decode step (K+V over all context tokens)."""
    return 2 * elem * kv_tokens * H_kv * D


def _interp1d(curve: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation of a sorted [(x, y)] curve, clamped at the ends."""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for i in range(1, len(curve)):
        if x <= curve[i][0]:
            (x0, y0), (x1, y1) = curve[i - 1], curve[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


@dataclass
class Predictor:
    b_peak: float                                                       # GB/s
    op: str = "gemm"
    c_peak: dict[str, float] = field(default_factory=dict)              # TFLOP/s per dtype (gemm)
    axes: dict = field(default_factory=dict)                            # dtype -> (Ms,Ks,Ns) (gemm)
    eff: dict = field(default_factory=dict)                             # dtype -> {(M,K,N): eff} (gemm)
    bytes_model: dict[str, float] = field(default_factory=lambda: {"w": 2.0, "a": 2.0, "o": 2.0})
    attn_curve: list[tuple[float, float]] = field(default_factory=list)  # sorted [(log KV_bytes, eff)]

    @classmethod
    def from_json(cls, path: str | Path) -> "Predictor":
        d = json.loads(Path(path).read_text())
        op = d.get("op", "gemm")
        b_peak = float(d["b_peak_gbps"])
        if op == "attn_decode":
            curve = sorted((math.log(_kv_bytes(r["kv_tokens"], r["H_kv"], r["D"])),
                            r["efficiency"]) for r in d["attn"])
            return cls(b_peak=b_peak, op=op, attn_curve=curve)
        c_peak = {k: float(v) for k, v in d["c_peak"].items()}
        # bf16/fp16 read & write 2 bytes/elem; quant schemes override via the JSON.
        bytes_model = d.get("bytes_model", {"w": 2.0, "a": 2.0, "o": 2.0})
        eff: dict = {}
        for r in d["gemm"]:
            res = r.get("residual", 0.0)
            e = (1.0 / res) if res else float("nan")
            eff.setdefault(r["dtype"], {})[(r["M"], r["K"], r["N"])] = e
        axes = {dt: (sorted({k[0] for k in t}), sorted({k[1] for k in t}),
                     sorted({k[2] for k in t})) for dt, t in eff.items()}
        return cls(b_peak=b_peak, op=op, c_peak=c_peak, axes=axes, eff=eff,
                   bytes_model=bytes_model)

    # --- GEMM: roofline + trilinear efficiency --------------------------
    def _ideal_compute_s(self, M, K, N, dtype):
        return 2 * M * N * K / (self.c_peak[dtype] * 1e12)

    def _ideal_mem_s(self, M, K, N, dtype):
        bm = self.bytes_model
        nbytes = bm["w"] * N * K + bm["a"] * M * K + bm["o"] * M * N
        return nbytes / (self.b_peak * 1e9)

    def roofline_ms(self, M, K, N, dtype="bf16"):
        return max(self._ideal_compute_s(M, K, N, dtype),
                   self._ideal_mem_s(M, K, N, dtype)) * 1e3

    @staticmethod
    def _bracket(vals: list[int], x: int) -> list[tuple[int, float]]:
        """Two (index, weight) pairs bracketing log(x) in log(vals); clamped at ends."""
        lx = math.log(x)
        if lx <= math.log(vals[0]):
            return [(0, 1.0), (0, 0.0)]
        if lx >= math.log(vals[-1]):
            return [(len(vals) - 1, 1.0), (len(vals) - 1, 0.0)]
        for i in range(1, len(vals)):
            if lx <= math.log(vals[i]):
                t = (lx - math.log(vals[i - 1])) / (math.log(vals[i]) - math.log(vals[i - 1]))
                return [(i - 1, 1.0 - t), (i, t)]
        return [(len(vals) - 1, 1.0), (len(vals) - 1, 0.0)]

    def efficiency(self, M, K, N, dtype="bf16"):
        Ms, Ks, Ns = self.axes[dtype]
        tbl = self.eff[dtype]
        tot = wsum = 0.0
        for mi, wm in self._bracket(Ms, M):
            for ki, wk in self._bracket(Ks, K):
                for ni, wn in self._bracket(Ns, N):
                    e = tbl.get((Ms[mi], Ks[ki], Ns[ni]))
                    if e is None or e != e:
                        continue
                    w = wm * wk * wn
                    tot += w * e
                    wsum += w
        return tot / wsum if wsum > 0 else float("nan")

    def latency_ms(self, M, K, N, dtype="bf16"):
        return self.roofline_ms(M, K, N, dtype) / self.efficiency(M, K, N, dtype)

    # --- decode attention: 1-D curve in total (block-padded) KV bytes ---
    #     kv_tokens = Σ_i ceil(L_i/block)·block  (what the kernel reads; ≈ Σ L_i)
    def decode_roofline_ms(self, kv_tokens, H_kv, D, dtype="bf16"):
        return _kv_bytes(kv_tokens, H_kv, D) / (self.b_peak * 1e9) * 1e3

    def decode_efficiency(self, kv_tokens, H_kv, D):
        return _interp1d(self.attn_curve, math.log(_kv_bytes(kv_tokens, H_kv, D)))

    def decode_latency_ms(self, kv_tokens, H_kv, D, dtype="bf16"):
        return self.decode_roofline_ms(kv_tokens, H_kv, D) / self.decode_efficiency(kv_tokens, H_kv, D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="results JSON (default: newest results/gemm_*.json)")
    ap.add_argument("--dtype", default="bf16")
    # gemm
    ap.add_argument("--shape", nargs=2, type=int, metavar=("K", "N"), default=None)
    ap.add_argument("--M", nargs="+", type=int, default=[1, 8, 64, 512, 4096])
    # decode attention
    ap.add_argument("--kv", nargs="+", type=int, default=None,
                    help="decode attn: total KV tokens (Σ L_i) to predict.")
    ap.add_argument("--head", nargs=2, type=int, metavar=("H_kv", "D"), default=[8, 128])
    args = ap.parse_args()

    path = args.results
    if path is None:
        cands = sorted(Path("results").glob("gemm_*.json"))
        if not cands:
            raise SystemExit("no results/gemm_*.json — pass --results or run run.py")
        path = str(cands[-1])
    p = Predictor.from_json(path)

    if p.op == "attn_decode":
        H_kv, D = args.head
        kvs = args.kv or [16384, 131072, 1048576]
        print(f"{path}  |  decode attention  |  B_peak {p.b_peak:.0f} GB/s")
        print(f"predict decode attn, head H_kv={H_kv} D={D}\n")
        print(f"  {'KV tokens':>10} {'KV(MB)':>8} {'eff':>6} {'roofline':>10} {'predicted':>10}")
        for kv in kvs:
            e = p.decode_efficiency(kv, H_kv, D)
            rl = p.decode_roofline_ms(kv, H_kv, D)
            print(f"  {kv:>10} {_kv_bytes(kv, H_kv, D)/1e6:>8.1f} {e:>6.2f} "
                  f"{rl:>8.3f}ms {rl/e:>8.3f}ms")
        return

    if args.shape is None:
        raise SystemExit("--shape K N is required for gemm prediction")
    K, N = args.shape
    print(f"{path}  |  C_peak[{args.dtype}] {p.c_peak[args.dtype]:.0f} TFLOP/s  "
          f"B_peak {p.b_peak:.0f} GB/s")
    print(f"predict K={K} N={N} ({args.dtype}), footprint {K*N/1e6:.1f}M elem\n")
    print(f"  {'M':>6} {'regime':>8} {'eff':>6} {'roofline':>10} {'predicted':>10}")
    for M in args.M:
        tc, tm = p._ideal_compute_s(M, K, N, args.dtype), p._ideal_mem_s(M, K, N, args.dtype)
        reg = "compute" if tc > 2 * tm else "memory" if tm > 2 * tc else "transit"
        eff = p.efficiency(M, K, N, args.dtype)
        rl = p.roofline_ms(M, K, N, args.dtype)
        print(f"  {M:>6} {reg:>8} {eff:>6.2f} {rl:>8.3f}ms {rl/eff:>8.3f}ms")


if __name__ == "__main__":
    main()
