"""Predict kernel latency from a measured grid (op auto-detected from the JSON).

GEMM (op=gemm):  t = roofline(C_peak, B_peak) / efficiency(M, K, N), efficiency by
trilinear interpolation in (log M, log K, log N) over the model-agnostic grid.

Attention (op=attn): hybrid — decode (S_q=1) and prefill (S_q>1) have different physics,
so they use different efficiency descriptors but route through one attn_latency_ms:
  * decode  — memory-bound; eff is a 1-D curve in (block-padded) total KV bytes.
  * prefill — batched causal GEMM; eff interpolated over (log S_q, log S_kv, log R·H, log D),
    roofline over the causal trapezoid:
        FLOPs = 4·H·D·R·(S_q·S_kv − S_q(S_q−1)/2);  bytes = 2·elem·R·(S_q·H·D + S_kv·H_kv·D)

MoE (op=moe): two grouped GEMMs (gate+up, down) under uniform routing; eff interpolated over
(log T, log E, log H, log I) with T=M·top_k routed tokens, E_act=min(E,T) active experts:
    FLOPs = 6·T·H·I;  bytes = E_act·3·H·I·elem + 2·M·H·elem

Pure stdlib — prediction needs no GPU or torch. Measurement lives in run.py.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def _kv_bytes(kv_tokens: int, H_kv: int, D: int, elem: int = 2) -> float:
    return 2 * elem * kv_tokens * H_kv * D


def _interp1d(curve: list[tuple[float, float]], x: float) -> float:
    """Linear interp of a sorted [(x, y)] curve, clamped at the ends."""
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
    b_peak: float                                                # GB/s
    op: str = "gemm"
    c_peak: dict[str, float] = field(default_factory=dict)       # TFLOP/s per dtype (gemm)
    axes: dict = field(default_factory=dict)                     # dtype -> (Ms,Ks,Ns) (gemm)
    eff: dict = field(default_factory=dict)                      # dtype -> {(M,K,N): eff} (gemm)
    bytes_model: dict[str, float] = field(default_factory=lambda: {"w": 2.0, "a": 2.0})
    attn_c: float = 0.0                                          # TFLOP/s (attn compute ceiling)
    attn_eff: dict = field(default_factory=dict)                 # {(Sq,Sk,RH,D): eff} (prefill grid)
    attn_axes: tuple = field(default_factory=tuple)              # (Sqs, Sks, RHs, Ds)
    attn_decode_curve: list = field(default_factory=list)        # sorted [(log KV_bytes, eff)]
    attn_backend: str = "flashinfer"                             # library the grid was measured on
    moe_c: float = 0.0                                           # TFLOP/s (moe compute ceiling)
    moe_eff: dict = field(default_factory=dict)                  # {(T,E,H,I): eff}  T=M*top_k
    moe_axes: tuple = field(default_factory=tuple)               # (Ts, Es, Hs, Is)
    moe_bytes_model: dict[str, float] = field(default_factory=lambda: {"w": 2.0, "a": 2.0})

    @classmethod
    def from_json(cls, path: str | Path) -> "Predictor":
        """Build a predictor from a results JSON. Reads the unified schema
        (hardware / operation / results); falls back to the legacy per-op schema."""
        d = json.loads(Path(path).read_text())
        if "hardware" not in d:
            return cls._from_legacy(d)
        hw, opn = d["hardware"], d["operation"]
        b_peak, c_peak = float(hw["b_peak_gbps"]), float(hw["c_peak_tflops"])
        op, _, dtype = opn["bench"].partition("_")          # gemm/attn/moe ; bf16/fp16/mxfp4
        results = d["results"]

        def sh(r):
            return r["shape"]

        if op == "attn":
            dec = [r for r in results if sh(r).get("kind") == "decode"]
            pre = [r for r in results if sh(r).get("kind") == "prefill"]
            dcurve = sorted((math.log(_kv_bytes(sh(r)["kv_tokens"], sh(r)["H_kv"], sh(r)["D"])),
                             r["efficiency"]) for r in dec)
            geff = {(sh(r)["Sq"], sh(r)["Sk"], sh(r)["RH"], sh(r)["D"]): r["efficiency"] for r in pre}
            gaxes = tuple(sorted({k[i] for k in geff}) for i in range(4))
            return cls(b_peak=b_peak, op="attn", attn_c=c_peak, attn_eff=geff, attn_axes=gaxes,
                       attn_decode_curve=dcurve, attn_backend=next(iter(opn.get("impl", {})), "flashinfer"))
        if op == "moe":
            meff = {(sh(r)["M"] * sh(r)["top_k"], sh(r)["E"], sh(r)["H"], sh(r)["I"]): r["efficiency"]
                    for r in results}
            maxes = tuple(sorted({k[i] for k in meff}) for i in range(4))   # (Ts, Es, Hs, Is)
            return cls(b_peak=b_peak, op="moe", moe_c=c_peak, moe_eff=meff, moe_axes=maxes,
                       moe_bytes_model=opn.get("bytes_model", {"w": 2.0, "a": 2.0}))
        eff = {dtype: {(sh(r)["M"], sh(r)["K"], sh(r)["N"]):
                       (r["efficiency"] if r["efficiency"] else float("nan")) for r in results}}
        axes = {dt: (sorted({k[0] for k in t}), sorted({k[1] for k in t}), sorted({k[2] for k in t}))
                for dt, t in eff.items()}
        return cls(b_peak=b_peak, op="gemm", c_peak={dtype: c_peak}, axes=axes, eff=eff,
                   bytes_model=opn.get("bytes_model", {"w": 2.0, "a": 2.0}))

    @classmethod
    def _from_legacy(cls, d: dict) -> "Predictor":
        """Reader for the pre-unification per-op JSON schema (so old result files still load)."""
        op = d.get("op", "gemm")
        b_peak = float(d["b_peak_gbps"])
        if op == "attn":
            dcurve = sorted((math.log(_kv_bytes(r["kv_tokens"], r["H_kv"], r["D"])), r["efficiency"])
                            for r in d["decode"])
            geff = {(r["Sq"], r["Sk"], r["RH"], r["D"]): r["efficiency"] for r in d["grid"]}
            gaxes = tuple(sorted({k[i] for k in geff}) for i in range(4))
            return cls(b_peak=b_peak, op="attn", attn_c=float(d["c_peak_tflops"]),
                       attn_eff=geff, attn_axes=gaxes, attn_decode_curve=dcurve,
                       attn_backend=d.get("backend", "flashinfer"))
        if op == "moe":
            meff = {(r["M"] * r["top_k"], r["E"], r["H"], r["I"]): r["efficiency"] for r in d["moe"]}
            maxes = tuple(sorted({k[i] for k in meff}) for i in range(4))
            return cls(b_peak=b_peak, op="moe", moe_c=float(d["c_peak_tflops"]),
                       moe_eff=meff, moe_axes=maxes,
                       moe_bytes_model=d.get("moe_bytes_model", {"w": 2.0, "a": 2.0}))
        c_peak = {k: float(v) for k, v in d["c_peak"].items()}
        bytes_model = d.get("bytes_model", {"w": 2.0, "a": 2.0})   # legacy files may carry "o"; ignored
        eff: dict = {}
        for r in d["gemm"]:
            res = r.get("residual", 0.0)
            eff.setdefault(r["dtype"], {})[(r["M"], r["K"], r["N"])] = (1.0 / res) if res else float("nan")
        axes = {dt: (sorted({k[0] for k in t}), sorted({k[1] for k in t}), sorted({k[2] for k in t}))
                for dt, t in eff.items()}
        return cls(b_peak=b_peak, op=op, c_peak=c_peak, axes=axes, eff=eff, bytes_model=bytes_model)

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

    # --- GEMM: roofline + trilinear efficiency --------------------------
    def _ideal_compute_s(self, M, K, N, dtype):
        return 2 * M * N * K / (self.c_peak[dtype] * 1e12)

    def _ideal_mem_s(self, M, K, N, dtype):
        bm = self.bytes_model
        return (bm["w"] * N * K + bm["a"] * (M * K + M * N)) / (self.b_peak * 1e9)

    def roofline_ms(self, M, K, N, dtype="bf16"):
        return max(self._ideal_compute_s(M, K, N, dtype),
                   self._ideal_mem_s(M, K, N, dtype)) * 1e3

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

    # --- attention: decode (S_q=1) / prefill (S_q=S_kv) / chunked (interior) ---
    def attn_roofline_ms(self, R, Sq, Sk, H, H_kv, D, elem=2):
        if Sq == 1:                                    # decode: memory roofline (KV bytes)
            pad = ((Sk + 15) // 16) * 16
            return _kv_bytes(R * pad, H_kv, D, elem) / (self.b_peak * 1e9) * 1e3
        pairs = Sq * Sk - Sq * (Sq - 1) // 2           # prefill: causal-trapezoid roofline
        flops = 4 * H * D * R * pairs
        nbytes = 2 * elem * R * (Sq * H * D + Sk * H_kv * D)
        return max(flops / (self.attn_c * 1e12), nbytes / (self.b_peak * 1e9)) * 1e3

    def _prefill_efficiency(self, Sq, Sk, RH, D):
        """4-D interpolation over the prefill grid (log S_q, log S_kv, log R·H, log D),
        skipping missing (S_kv < S_q) corners and renormalising by present weight."""
        Sqs, Sks, RHs, Ds = self.attn_axes
        tot = wsum = 0.0
        for qi, wq in self._bracket(Sqs, Sq):
            for ki, wk in self._bracket(Sks, Sk):
                for ri, wr in self._bracket(RHs, RH):
                    for di, wd in self._bracket(Ds, D):
                        e = self.attn_eff.get((Sqs[qi], Sks[ki], RHs[ri], Ds[di]))
                        if e is None or e != e:
                            continue
                        w = wq * wk * wr * wd
                        tot += w * e
                        wsum += w
        return tot / wsum if wsum > 0 else float("nan")

    def attn_efficiency(self, R, Sq, Sk, H, H_kv, D):
        if Sq == 1:                                    # decode: 1-D KV-byte curve
            pad = ((Sk + 15) // 16) * 16
            return _interp1d(self.attn_decode_curve, math.log(_kv_bytes(R * pad, H_kv, D)))
        return self._prefill_efficiency(Sq, Sk, R * H, D)

    def attn_latency_ms(self, R, Sq, Sk, H, H_kv, D):
        return self.attn_roofline_ms(R, Sq, Sk, H, H_kv, D) / self.attn_efficiency(R, Sq, Sk, H, H_kv, D)

    # --- MoE: two grouped GEMMs (uniform routing); efficiency over (log T, log E, log H, log I) ---
    def moe_roofline_ms(self, M, E, top_k, H, I):
        bm = self.moe_bytes_model                      # {"w": weight B/elem, "a": act B/elem}
        T = M * top_k
        E_act = min(E, T)                              # only top_k experts fire at small M
        flops = 6 * T * H * I                          # gate+up (4THI) + down (2THI)
        nbytes = E_act * 3 * H * I * bm["w"] + 2 * M * H * bm["a"]
        return max(flops / (self.moe_c * 1e12), nbytes / (self.b_peak * 1e9)) * 1e3

    def moe_efficiency(self, M, E, top_k, H, I):
        T = M * top_k
        Ts, Es, Hs, Is = self.moe_axes
        tot = wsum = 0.0
        for ti, wt in self._bracket(Ts, T):
            for ei, we in self._bracket(Es, E):
                for hi, wh in self._bracket(Hs, H):
                    for ii, wi in self._bracket(Is, I):
                        e = self.moe_eff.get((Ts[ti], Es[ei], Hs[hi], Is[ii]))
                        if e is None or e != e:
                            continue
                        w = wt * we * wh * wi
                        tot += w * e
                        wsum += w
        return tot / wsum if wsum > 0 else float("nan")

    def moe_latency_ms(self, M, E, top_k, H, I):
        return self.moe_roofline_ms(M, E, top_k, H, I) / self.moe_efficiency(M, E, top_k, H, I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="results JSON (default: results/gemm_bf16.json)")
    ap.add_argument("--dtype", default="bf16")
    # gemm
    ap.add_argument("--shape", nargs=2, type=int, metavar=("K", "N"), default=None)
    ap.add_argument("--M", nargs="+", type=int, default=[1, 8, 64, 512, 4096])
    # attention: one (R, S_q, S_kv) case + head config (H, H_kv, D)
    ap.add_argument("--attn", nargs=3, type=int, metavar=("R", "S_q", "S_kv"), default=None)
    ap.add_argument("--head", nargs=3, type=int, metavar=("H", "H_kv", "D"), default=[32, 8, 128])
    # moe: expert config (E, top_k, H, I); sweeps --M
    ap.add_argument("--moe", nargs=4, type=int, metavar=("E", "top_k", "H", "I"), default=None)
    args = ap.parse_args()

    path = args.results or "results/gemm_bf16.json"
    if not Path(path).exists():
        raise SystemExit(f"no {path} — pass --results or run run.py")
    p = Predictor.from_json(path)

    if p.op == "attn":
        H, H_kv, D = args.head
        cases = [tuple(args.attn)] if args.attn else [
            (1, 1, 16384), (1, 1, 65536), (4, 2048, 2048), (16, 512, 8192)]
        print(f"{path}  |  attention  |  C_peak {p.attn_c:.0f} TFLOP/s  B_peak {p.b_peak:.0f} GB/s")
        print(f"predict attn, head H={H} H_kv={H_kv} D={D}\n")
        print(f"  {'R':>4} {'S_q':>6} {'S_kv':>7} {'eff':>6} {'roofline':>10} {'predicted':>10}")
        for R, Sq, Sk in cases:
            e = p.attn_efficiency(R, Sq, Sk, H, H_kv, D)
            rl = p.attn_roofline_ms(R, Sq, Sk, H, H_kv, D)
            print(f"  {R:>4} {Sq:>6} {Sk:>7} {e:>6.2f} {rl:>8.3f}ms {rl/e:>8.3f}ms")
        return

    if p.op == "moe":
        E, top_k, H, I = args.moe if args.moe else [128, 8, 2048, 768]
        print(f"{path}  |  MoE  |  C_peak {p.moe_c:.0f} TFLOP/s  B_peak {p.b_peak:.0f} GB/s")
        print(f"predict MoE E={E} top_k={top_k} H={H} I={I}\n")
        print(f"  {'M':>6} {'T':>8} {'regime':>7} {'eff':>6} {'roofline':>10} {'predicted':>10}")
        for M in args.M:
            T, E_act = M * top_k, min(E, M * top_k)
            tc = 6 * T * H * I / (p.moe_c * 1e12)
            tm = (E_act * 3 * H * I * 2 + 2 * M * H * 2) / (p.b_peak * 1e9)
            reg = "compute" if tc > tm else "memory"
            e = p.moe_efficiency(M, E, top_k, H, I)
            rl = p.moe_roofline_ms(M, E, top_k, H, I)
            print(f"  {M:>6} {T:>8} {reg:>7} {e:>6.2f} {rl:>8.3f}ms {rl/e:>8.3f}ms")
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
