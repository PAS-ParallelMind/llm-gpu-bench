"""Compare two attention backends' measured efficiency, point-for-point.

Both backends are swept on the *same* grid (attn.py drives both via a pluggable kernel
call), so decode-curve and prefill-grid points line up exactly and the only difference is
the kernel. Prints where the kernels diverge.

    python compare_backends.py                       # newest flash_attn vs flashinfer
    python compare_backends.py A.json B.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _load(path):
    d = json.loads(Path(path).read_text())
    dec = {r["kv_tokens"]: r["efficiency"] for r in d["decode"]}
    grid = {(r["Sq"], r["Sk"], r["RH"], r["D"]): r["efficiency"] for r in d["grid"]}
    return d.get("backend", path), dec, grid


def main() -> None:
    if len(sys.argv) >= 3:
        pa, pb = sys.argv[1], sys.argv[2]
    else:
        pa = str(sorted(Path("results").glob("attn_flash_attn_*.json"))[-1])
        pb = str(sorted(Path("results").glob("attn_flashinfer_*.json"))[-1])
    na, deca, grida = _load(pa)
    nb, decb, gridb = _load(pb)
    print(f"A = {na:12} ({pa})\nB = {nb:12} ({pb})\n")

    print("== decode curve: efficiency vs total KV tokens ==")
    print(f"  {'kv_tokens':>10} {'A':>6} {'B':>6}  {'B/A':>6}")
    for kv in sorted(set(deca) & set(decb)):
        a, b = deca[kv], decb[kv]
        print(f"  {kv:>10} {a:6.2f} {b:6.2f}  {b/a:6.2f}x")

    keys = sorted(set(grida) & set(gridb))
    ra = np.array([grida[k] for k in keys])
    rb = np.array([gridb[k] for k in keys])
    ratio = rb / ra
    print(f"\n== prefill grid: {len(keys)} shared points, eff(B)/eff(A) ==")
    print(f"  median {np.median(ratio):.2f}x   mean {ratio.mean():.2f}x   "
          f"range {ratio.min():.2f}..{ratio.max():.2f}x")
    print(f"  A mean eff {ra.mean():.2f}   B mean eff {rb.mean():.2f}")
    order = np.argsort(ratio)
    print("\n  most B-favoured (B faster):")
    for i in order[::-1][:5]:
        sq, sk, rh, d = keys[i]
        print(f"    Sq={sq:<5} Sk={sk:<5} RH={rh:<4} D={d:<4}  A={ra[i]:.2f} B={rb[i]:.2f}  {ratio[i]:.2f}x")
    print("  most A-favoured (A faster):")
    for i in order[:5]:
        sq, sk, rh, d = keys[i]
        print(f"    Sq={sq:<5} Sk={sk:<5} RH={rh:<4} D={d:<4}  A={ra[i]:.2f} B={rb[i]:.2f}  {ratio[i]:.2f}x")


if __name__ == "__main__":
    main()
