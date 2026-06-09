"""Predict GEMM latency from the measured grid — trilinear interpolation.

    t_pred = roofline(C_peak, B_peak) / efficiency(M, K, N)

The efficiency factor is read from the dense, model-agnostic grid
(results/gemm_*.json) by trilinear interpolation in (log M, log K, log N), with
clamping at the grid edges. The grid is dense enough in both K and N that this
plain interpolation needs no regime model or aspect-ratio metric — it just reads
off the measured surface. (Earlier sparse grids needed a footprint<->K distance
metric; see README.)

Pure stdlib — prediction needs no GPU or torch. Measurement lives in run.py.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Predictor:
    c_peak: dict[str, float]                                  # achieved TFLOP/s per dtype
    b_peak: float                                             # achieved GB/s
    axes: dict[str, tuple[list[int], list[int], list[int]]]   # dtype -> (Ms, Ks, Ns) sorted
    eff: dict[str, dict[tuple[int, int, int], float]]         # dtype -> {(M,K,N): efficiency}
    bytes_model: dict[str, float]                            # {w,a,o} bytes/elem (memory roofline)

    @classmethod
    def from_json(cls, path: str | Path) -> "Predictor":
        d = json.loads(Path(path).read_text())
        c_peak = {k: float(v) for k, v in d["c_peak"].items()}
        # bf16/fp16 read & write 2 bytes/elem; quant schemes override via the JSON.
        bytes_model = d.get("bytes_model", {"w": 2.0, "a": 2.0, "o": 2.0})
        eff: dict = {}
        for r in d["gemm"]:
            res = r.get("residual", 0.0)
            e = (1.0 / res) if res else float("nan")
            eff.setdefault(r["dtype"], {})[(r["M"], r["K"], r["N"])] = e
        axes = {}
        for dt, tbl in eff.items():
            axes[dt] = (sorted({k[0] for k in tbl}),
                        sorted({k[1] for k in tbl}),
                        sorted({k[2] for k in tbl}))
        return cls(c_peak=c_peak, b_peak=float(d["b_peak_gbps"]), axes=axes, eff=eff,
                   bytes_model=bytes_model)

    # --- roofline -------------------------------------------------------
    def _ideal_compute_s(self, M, K, N, dtype):
        return 2 * M * N * K / (self.c_peak[dtype] * 1e12)

    def _ideal_mem_s(self, M, K, N, dtype):
        bm = self.bytes_model
        nbytes = bm["w"] * N * K + bm["a"] * M * K + bm["o"] * M * N
        return nbytes / (self.b_peak * 1e9)

    def roofline_ms(self, M, K, N, dtype="bf16"):
        return max(self._ideal_compute_s(M, K, N, dtype),
                   self._ideal_mem_s(M, K, N, dtype)) * 1e3

    # --- efficiency (trilinear in log space) ----------------------------
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="grid JSON (default: newest results/gemm_*.json)")
    ap.add_argument("--shape", nargs=2, type=int, metavar=("K", "N"), required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--M", nargs="+", type=int, default=[1, 8, 64, 512, 4096])
    args = ap.parse_args()

    path = args.results
    if path is None:
        cands = sorted(Path("results").glob("gemm_*.json"))
        if not cands:
            raise SystemExit("no results/gemm_*.json — run run.py first")
        path = cands[-1]
    p = Predictor.from_json(path)
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
