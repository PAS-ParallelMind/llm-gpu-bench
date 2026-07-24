"""Predict kernel latency from a measured grid (op auto-detected from the JSON).

GEMM (op=gemm):  t = roofline(C_peak, B_peak) / efficiency(M, K, N), efficiency by
trilinear interpolation in (log M, log K, log N) over the model-agnostic grid.

Attention (op=attn): hybrid — decode (q_len=1) and prefill (q_len>1) have different physics,
so they use different efficiency descriptors but route through one attn_latency_ms.
  * decode  — memory-bound; eff is a 1-D curve in (block-padded) total KV bytes.
  * prefill — batched causal GEMM; eff interpolated over (log q_len, log kv_len, log total_heads, log head_dim),
    roofline over the causal trapezoid:
        FLOPs = 4·n_heads·head_dim·n_req·(q_len·kv_len − q_len(q_len−1)/2)

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


def _kv_bytes(kv_tokens: int, n_kv_heads: int, head_dim: int, elem: int = 2) -> float:
    return 2 * elem * kv_tokens * n_kv_heads * head_dim


def _nearest_eff(eff: dict, query: tuple) -> float:
    """Nearest measured grid point (min sum-of-squared log-distance) — fills a hole where the
    bracket interpolation neighborhood is empty (a shape whose bracketing grid points all failed
    to measure during the sweep). Graceful extrapolation instead of NaN."""
    best_e, best_d = float("nan"), float("inf")
    ql = [math.log(v) for v in query]
    for key, e in eff.items():
        if e != e:                          # skip NaN entries
            continue
        d = sum((math.log(k) - q) ** 2 for k, q in zip(key, ql))
        if d < best_d:
            best_d, best_e = d, e
    return best_e


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
    attn_eff: dict = field(default_factory=dict)                 # {(q_len,kv_len,total_heads,head_dim): eff} (prefill grid)
    attn_axes: tuple = field(default_factory=tuple)              # (q_lens, kv_lens, total_heads, head_dims)
    attn_decode_curve: list = field(default_factory=list)        # sorted [(log KV_bytes, eff)]
    attn_backend: str = "flashinfer"                             # library the grid was measured on
    moe_c: float = 0.0                                           # TFLOP/s (moe compute ceiling)
    moe_eff: dict = field(default_factory=dict)                  # {(T,E,H,I): eff}  T=M*top_k
    moe_axes: tuple = field(default_factory=tuple)               # (Ts, Es, Hs, Is)
    moe_bytes_model: dict[str, float] = field(default_factory=lambda: {"w": 2.0, "a": 2.0})
    ar_curves: dict = field(default_factory=dict)               # {W: sorted [(log bytes, latency_ms)]} (allreduce)

    @classmethod
    def from_json(cls, path: str | Path) -> "Predictor":
        """Build a predictor from a results JSON (unified schema: hardware / operation / results)."""
        d = json.loads(Path(path).read_text())
        hw, opn = d["hardware"], d["operation"]
        b_peak = float(hw.get("b_peak_gbps", 0) or 0)       # absent for allreduce (no roofline)
        c_peak = float(hw.get("c_peak_tflops", 0) or 0)
        op, _, dtype = opn["bench"].partition("_")          # gemm/attn/moe/allreduce ; bf16/fp16/mxfp4
        results = d["results"]

        def sh(r):
            return r["shape"]

        if op == "allreduce":
            # Curves are (bytes, latency_ms) and interpolate LINEARLY IN BYTES -- not log-bytes.
            # Unlike the other ops (which interpolate a bounded efficiency and divide a roofline
            # that already carries the ∝size scaling), all-reduce interpolates latency directly,
            # so the interpolator itself must carry latency ∝ bytes. Measured on the grid by
            # hold-out: linear-in-bytes 1.1% median vs 19.4% for linear-in-log-bytes.
            curves: dict[int, list] = {}
            for r in results:
                s = sh(r)
                curves.setdefault(s["world_size"], []).append((float(s["bytes"]), r["latency_ms"]))
            return cls(b_peak=b_peak, op="allreduce",
                       ar_curves={w: sorted(c) for w, c in curves.items()})
        if op == "attn":
            dec = [r for r in results if sh(r).get("kind") == "decode"]
            pre = [r for r in results if sh(r).get("kind") == "prefill"]
            dcurve = sorted((math.log(_kv_bytes(sh(r)["kv_tokens"], sh(r)["n_kv_heads"], sh(r)["head_dim"])),
                             r["efficiency"]) for r in dec)
            geff = {(sh(r)["q_len"], sh(r)["kv_len"], sh(r)["total_heads"], sh(r)["head_dim"]):
                    r["efficiency"] for r in pre}
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
        return tot / wsum if wsum > 0 else _nearest_eff(tbl, (M, K, N))

    def latency_ms(self, M, K, N, dtype="bf16"):
        return self.roofline_ms(M, K, N, dtype) / self.efficiency(M, K, N, dtype)

    # --- attention: decode (q_len=1) / prefill (q_len=kv_len) / chunked (interior) ---
    def attn_roofline_ms(self, n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim, elem=2):
        if q_len == 1:                                 # decode: memory roofline (KV bytes)
            pad = ((kv_len + 15) // 16) * 16
            return _kv_bytes(n_req * pad, n_kv_heads, head_dim, elem) / (self.b_peak * 1e9) * 1e3
        pairs = q_len * kv_len - q_len * (q_len - 1) // 2   # prefill: causal-trapezoid roofline
        flops = 4 * n_heads * head_dim * n_req * pairs
        nbytes = 2 * elem * n_req * (q_len * n_heads * head_dim + kv_len * n_kv_heads * head_dim)
        return max(flops / (self.attn_c * 1e12), nbytes / (self.b_peak * 1e9)) * 1e3

    def _prefill_efficiency(self, q_len, kv_len, total_heads, head_dim):
        """4-D interpolation over the prefill grid (log q_len, log kv_len, log total_heads, log head_dim),
        skipping missing (kv_len < q_len) corners and renormalising by present weight."""
        q_ax, kv_ax, th_ax, hd_ax = self.attn_axes
        tot = wsum = 0.0
        for qi, wq in self._bracket(q_ax, q_len):
            for ki, wk in self._bracket(kv_ax, kv_len):
                for ri, wr in self._bracket(th_ax, total_heads):
                    for di, wd in self._bracket(hd_ax, head_dim):
                        e = self.attn_eff.get((q_ax[qi], kv_ax[ki], th_ax[ri], hd_ax[di]))
                        if e is None or e != e:
                            continue
                        w = wq * wk * wr * wd
                        tot += w * e
                        wsum += w
        return tot / wsum if wsum > 0 else _nearest_eff(self.attn_eff, (q_len, kv_len, total_heads, head_dim))

    def attn_efficiency(self, n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim):
        if q_len == 1:                                 # decode: 1-D KV-byte curve
            pad = ((kv_len + 15) // 16) * 16
            return _interp1d(self.attn_decode_curve, math.log(_kv_bytes(n_req * pad, n_kv_heads, head_dim)))
        return self._prefill_efficiency(q_len, kv_len, n_req * n_heads, head_dim)

    def attn_latency_ms(self, n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim):
        return (self.attn_roofline_ms(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim)
                / self.attn_efficiency(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim))

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
        return tot / wsum if wsum > 0 else _nearest_eff(self.moe_eff, (T, E, H, I))

    def moe_latency_ms(self, M, E, top_k, H, I):
        return self.moe_roofline_ms(M, E, top_k, H, I) / self.moe_efficiency(M, E, top_k, H, I)

    # --- all-reduce: interpolate measured latency directly (log bytes per world size) ---
    def allreduce_latency_ms(self, nbytes: int, world_size: int) -> float:
        """TP all-reduce latency for `nbytes` over `world_size` ranks: interpolate the measured
        latency curve linearly in bytes for that W (no roofline -- latency is measured directly;
        latency ∝ bytes once bandwidth-bound, so bytes is the right coordinate). W between measured
        world sizes interpolates linearly in W; W<=1 has no collective (0 cost)."""
        if world_size <= 1:
            return 0.0                          # single rank -> no all-reduce (W=1 is not measured)
        curves = self.ar_curves
        if not curves:
            raise ValueError("no all-reduce curves loaded")
        lx = float(nbytes)
        if world_size in curves:
            return _interp1d(curves[world_size], lx)
        ws = sorted(curves)
        if world_size <= ws[0]:
            return _interp1d(curves[ws[0]], lx)
        if world_size >= ws[-1]:
            return _interp1d(curves[ws[-1]], lx)
        for i in range(1, len(ws)):
            if world_size <= ws[i]:
                w0, w1 = ws[i - 1], ws[i]
                l0, l1 = _interp1d(curves[w0], lx), _interp1d(curves[w1], lx)
                return l0 + (l1 - l0) * (world_size - w0) / (w1 - w0)
        return _interp1d(curves[ws[-1]], lx)

    def allreduce_roofline_ms(self, nbytes: int, world_size: int) -> float:
        """Bandwidth-roofline baseline for the collective (no per-shape grid): bytes / peak achieved
        algorithm bandwidth for that world size, the peak taken from the largest measured message
        (the interconnect BW ceiling). The all-reduce analog of the GEMM/attn roofline -- one BW
        scalar, no efficiency table. Small latency-bound messages fall far below it (roofline
        under-predicts toward 0); that gap is exactly what the measured log-bytes curve recovers."""
        if world_size <= 1:
            return 0.0
        curves = self.ar_curves
        if not curves:
            raise ValueError("no all-reduce curves loaded")
        W = world_size if world_size in curves else min(curves, key=lambda w: abs(w - world_size))
        algbw_peak = max(b / (lat * 1e-3) for b, lat in curves[W])               # bytes/s (plateau)
        return nbytes / algbw_peak * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="results JSON (default: results/gemm_bf16.json)")
    ap.add_argument("--dtype", default="bf16")
    # gemm
    ap.add_argument("--shape", nargs=2, type=int, metavar=("K", "N"), default=None)
    ap.add_argument("--M", nargs="+", type=int, default=[1, 8, 64, 512, 4096])
    # attention: one (n_req, q_len, kv_len) case + head config (n_heads, n_kv_heads, head_dim)
    ap.add_argument("--attn", nargs=3, type=int, metavar=("n_req", "q_len", "kv_len"), default=None)
    ap.add_argument("--head", nargs=3, type=int, metavar=("n_heads", "n_kv_heads", "head_dim"),
                    default=[32, 8, 128])
    # moe: expert config (E, top_k, H, I); sweeps --M
    ap.add_argument("--moe", nargs=4, type=int, metavar=("E", "top_k", "H", "I"), default=None)
    # all-reduce: world size; sweeps --bytes
    ap.add_argument("--allreduce", type=int, metavar="WORLD_SIZE", default=None)
    ap.add_argument("--bytes", nargs="+", type=int,
                    default=[1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26])
    args = ap.parse_args()

    path = args.results or "results/gemm_bf16.json"
    if not Path(path).exists():
        raise SystemExit(f"no {path} — pass --results or run run.py")
    p = Predictor.from_json(path)

    if p.op == "allreduce":
        W = args.allreduce or max(p.ar_curves)
        print(f"{path}  |  all-reduce  |  world sizes {sorted(p.ar_curves)}")
        print(f"predict NCCL all-reduce, world_size={W}\n")
        print(f"  {'bytes':>12} {'predicted':>11}")
        for b in args.bytes:
            print(f"  {b:>12} {p.allreduce_latency_ms(b, W):>9.4f}ms")
        return

    if p.op == "attn":
        n_heads, n_kv_heads, head_dim = args.head
        cases = [tuple(args.attn)] if args.attn else [
            (1, 1, 16384), (1, 1, 65536), (4, 2048, 2048), (16, 512, 8192)]
        print(f"{path}  |  attention  |  C_peak {p.attn_c:.0f} TFLOP/s  B_peak {p.b_peak:.0f} GB/s")
        print(f"predict attn, head n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim}\n")
        print(f"  {'n_req':>5} {'q_len':>6} {'kv_len':>7} {'eff':>6} {'roofline':>10} {'predicted':>10}")
        for n_req, q_len, kv_len in cases:
            e = p.attn_efficiency(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim)
            rl = p.attn_roofline_ms(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim)
            print(f"  {n_req:>5} {q_len:>6} {kv_len:>7} {e:>6.2f} {rl:>8.3f}ms {rl/e:>8.3f}ms")
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
