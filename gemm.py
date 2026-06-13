"""GEMM sweep — achievable compute (C_peak), achievable bandwidth, roofline residual.

For ``y = x @ W^T`` with ``x:[M,K]``, ``W:[N,K]`` (vLLM's ``F.linear`` layout):

    FLOPs = 2 * M * N * K
    bytes = w*N*K + a*(M*K + M*N)            # read W, read x, write y  (w/a bytes/elem per dtype)
    arithmetic intensity = FLOPs / bytes ~= M   (for M << N, K)

So sweeping ``M`` walks each projection from memory-bound (decode, small M) up
through the ridge point to compute-bound (prefill, large M). One sweep yields the
compute ceiling (large-M plateau), the achievable bandwidth (small-M slope), and
— against measured C_peak / B_peak — the roofline residual (how tight the model
is on the kernel it is *supposed* to fit).

bf16 / fp16, via torch ``F.linear`` (the call vLLM's unquantized linear makes; torch picks
cuBLAS / cuBLASLt / CUTLASS / triton per shape). Weights 2 bytes/elem. (mxfp4 *dense* GEMM was
dropped — no model uses 4-bit weights for the dense projections; mxfp4 lives in the MoE
experts, see moe.py.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from timing import measure, progress

_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# byte model as {weight, activation} bytes/elem per dtype — shared with the predictor.
# "a" covers both the activation read and the output write (same dtype).
BYTES_MODEL: dict[str, dict[str, float]] = {
    "bf16": {"w": 2.0, "a": 2.0},
    "fp16": {"w": 2.0, "a": 2.0},
}

# ---------------------------------------------------------------------------
# Model-agnostic (K, N) grid — a dense octave grid in BOTH dims. A one-time,
# per-GPU sweep; the predictor (predict.py) is plain trilinear interpolation in
# (log M, log K, log N), so the grid only has to bracket every real projection
# closely in both dimensions. Octaves from 128 (sub-tile dims are rare and very
# inefficient; powers-of-two are tile-aligned and measure cleanly) up through the
# largest hidden / MoE / vocab sizes. N reaches 131072 so lmhead (N ~150-200k) is
# bracketed, not extrapolated. Real projections sit inside the hull: K in
# [768 .. 8192], N in [1536 .. ~201k] (see validate_predict.py).
# ---------------------------------------------------------------------------
GRID_K: list[int] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
GRID_N: list[int] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def make_grid(
    ks: list[int] = GRID_K, ns: list[int] = GRID_N
) -> dict[str, tuple[int, int]]:
    """A model-agnostic {name: (K, N)} grid, name keyed as ``k{K}_n{N}``."""
    return {f"k{k}_n{n}": (k, n) for k in ks for n in ns}


SHAPES: dict[str, tuple[int, int]] = make_grid()

DEFAULT_MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@dataclass
class GemmRecord:
    shape: str
    M: int
    K: int
    N: int
    dtype: str
    median_ms: float
    min_ms: float
    tflops: float          # achieved: 2MNK / median
    gbps: float            # achieved: bytes / median
    ai: float              # arithmetic intensity (FLOPs / bytes)
    predicted_ms: float = 0.0   # roofline w/ measured peaks (filled in later)
    residual: float = 0.0       # median / predicted (1.0 == roofline is tight)

    def result(self) -> dict:
        eff = (self.predicted_ms / self.median_ms) if self.median_ms > 0 else 0.0  # roofline/measured
        return {"shape": {"M": self.M, "K": self.K, "N": self.N},
                "latency_ms": self.median_ms, "tflops": self.tflops, "gbps": self.gbps,
                "efficiency": eff}


def gemm_bytes(dtype: str, M: int, K: int, N: int) -> float:
    bm = BYTES_MODEL[dtype]
    return bm["w"] * N * K + bm["a"] * (M * K + M * N)   # weight read + (activation read + output write)


def run_gemm_sweep(
    shapes: dict[str, tuple[int, int]],
    Ms: list[int],
    dtypes: list[str],
    *,
    device: int | torch.device = 0,
    iters: int = 100,
    warmup: int = 25,
) -> list[GemmRecord]:
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    recs: list[GemmRecord] = []
    pbar = progress(len(dtypes) * len(shapes) * len(Ms), "gemm")
    for dtype_name in dtypes:
        dt = _DTYPES[dtype_name]
        for shape_name, (K, N) in shapes.items():
            W = torch.randn(N, K, device=dev, dtype=dt)
            for M in Ms:
                x = torch.randn(M, K, device=dev, dtype=dt)
                t = measure(lambda: F.linear(x, W), device=dev, iters=iters, warmup=warmup)
                flops = 2 * M * N * K
                nbytes = gemm_bytes(dtype_name, M, K, N)
                sec = t.median_ms * 1e-3
                recs.append(GemmRecord(
                    shape=shape_name, M=M, K=K, N=N, dtype=dtype_name,
                    median_ms=t.median_ms, min_ms=t.min_ms,
                    tflops=flops / sec / 1e12,
                    gbps=nbytes / sec / 1e9,
                    ai=flops / nbytes,
                ))
                pbar.set_postfix_str(f"{dtype_name} {shape_name} M={M}")
                pbar.update(1)
                del x
            del W
            torch.cuda.empty_cache()
    pbar.close()
    return recs


def roofline_residual(
    recs: list[GemmRecord],
    c_peak: dict[str, tuple[float, str]],
    b_peak_gbps: float,
) -> None:
    """Fill in predicted_ms / residual using the C_peak / B_peak ceiling (in place)."""
    b = b_peak_gbps * 1e9
    for r in recs:
        c = c_peak[r.dtype][0] * 1e12
        flops = 2 * r.M * r.N * r.K
        nbytes = gemm_bytes(r.dtype, r.M, r.K, r.N)
        pred_s = max(flops / c, nbytes / b)
        r.predicted_ms = pred_s * 1e3
        r.residual = (r.median_ms / r.predicted_ms) if r.predicted_ms > 0 else 0.0
