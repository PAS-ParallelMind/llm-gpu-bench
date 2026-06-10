"""Decode flash-attention sweep via vLLM's flash_attn_varlen_func (paged KV cache).

Decode attention (1 query token per request, attending over the whole KV cache) is
*always* memory-bound: arithmetic intensity = 2·(H/H_kv)/elem (the GQA ratio), far
below the roofline ridge regardless of batch or context length. Measurement shows
the efficiency depends ONLY on the total KV-cache bytes streamed — not on the head
config (H, H_kv, D), the request count R, or how the bytes split across requests'
context lengths. So a single 1-D curve  eff = f(KV_bytes)  predicts decode attention
for any model and any continuous batch:

    t_decode = (KV_bytes / B_peak) / f(KV_bytes)
    KV_bytes = 2 · elem · Σ_i ceil(L_i / block) · block · H_kv · D   # block-padded K+V

KV_bytes is block-padded because the kernel reads whole paged blocks; for L_i >> block
it is just 2·elem·(Σ_i L_i)·H_kv·D. Stress-tested and verified to hold across head
config, paged block size, request count, and batch composition — *for per-request
context L_i >~ 128 tokens* (covers realistic decode). In the large-batch ×
very-short-context corner (many requests each with L_i < ~128), per-request overhead
pulls efficiency below the curve, so the 1-D model over-predicts efficiency there;
documented limitation, not modeled.

The kernel is vLLM's own flash_attn_varlen_func with a paged KV cache + block_table
(verified against a naive reference). We add the timing harness and the byte model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from timing import measure, progress
from vllm.vllm_flash_attn import flash_attn_varlen_func

BLOCK_SIZE = 16   # vLLM's default paged-KV block size
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Shape variables (one decode step of a continuous batch):
#   R     requests in the batch (decode batch size) — 1 query token per request
#   L     KV-cache / context length per request (uniform here; Σ_i L_i in general)
#   H     query heads
#   H_kv  key/value heads — GQA: H_kv <= H, and H/H_kv is the group size. Only H_kv
#         heads are stored in the KV cache (the GQA memory saving)
#   D     head dimension (per-head query/key/value vector size)
#
# KV bytes = 2·elem·(Σ_i L_i)·H_kv·D  (read K + V);  FLOPs ∝ R·H·L·D.  Decode
# arithmetic intensity ≈ 2·(H/H_kv)/elem — set by the GQA ratio, always memory-bound.

# Canonical sweep to trace eff = f(KV bytes). The head config is only a vehicle to
# generate KV-byte points — the curve is config-independent, so it predicts any
# model's decode attention. L spans ~0.5 MB to ~2 GB of KV cache at R=8.
DECODE_L_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DECODE_CONFIG = {"H": 32, "H_kv": 8, "D": 128, "R": 8}


def kv_bytes(kv_tokens: int, H_kv: int, D: int, elem: int = 2) -> float:
    """Total KV-cache bytes read in one decode step (K+V). `kv_tokens` should be the
    block-padded token total (see padded_kv_tokens) — what the kernel actually reads."""
    return 2 * elem * kv_tokens * H_kv * D


def padded_kv_tokens(Ls, block_size: int = BLOCK_SIZE) -> int:
    """KV tokens the kernel actually reads: each request's context rounded up to a
    whole paged block. Equals Σ L_i once L_i >> block_size."""
    return sum(((L + block_size - 1) // block_size) * block_size for L in Ls)


@dataclass
class AttnRecord:
    kv_tokens: int          # total context tokens in the batch (Σ L_i = R·L here)
    R: int                  # requests in the batch (decode batch size)
    L: int                  # context / KV-cache length per request
    H: int                  # query heads
    H_kv: int               # key/value heads (GQA: H_kv <= H)
    D: int                  # head dimension
    dtype: str
    kv_mb: float            # total KV bytes / 1e6
    median_ms: float
    min_ms: float
    gbps: float             # achieved KV-read GB/s
    predicted_ms: float = 0.0
    efficiency: float = 0.0


def _decode_call(R, L, H, H_kv, D, dt, dev):
    nblk = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    nb = R * nblk
    q = torch.randn(R, H, D, device=dev, dtype=dt)
    kc = torch.randn(nb, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    vc = torch.randn(nb, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    bt = torch.arange(nb, device=dev, dtype=torch.int32).view(R, nblk)
    cu = torch.arange(R + 1, device=dev, dtype=torch.int32)
    su = torch.full((R,), L, device=dev, dtype=torch.int32)
    scale = 1.0 / math.sqrt(D)
    fn = lambda: flash_attn_varlen_func(
        q=q, k=kc, v=vc, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=L,
        seqused_k=su, softmax_scale=scale, causal=True, block_table=bt)
    return fn, (q, kc, vc)


def run_decode_sweep(
    Ls: list[int],
    *,
    H: int = 32, H_kv: int = 8, D: int = 128, R: int = 8,
    dtype: str = "bf16",
    device: int | torch.device = 0,
    iters: int = 30, warmup: int = 10,
) -> list[AttnRecord]:
    """Sweep context length L (at fixed R / head config) to trace eff = f(KV bytes).

    The head config is just a vehicle to generate KV-byte points — the curve is
    config-independent, so it transfers to any model.
    """
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    recs: list[AttnRecord] = []
    pbar = progress(len(Ls), "decode-attn")
    for L in Ls:
        fn, bufs = _decode_call(R, L, H, H_kv, D, dt, dev)
        t = measure(fn, device=dev, iters=iters, warmup=warmup)
        nbytes = kv_bytes(R * L, H_kv, D, elem)
        sec = t.median_ms * 1e-3
        recs.append(AttnRecord(
            kv_tokens=R * L, R=R, L=L, H=H, H_kv=H_kv, D=D, dtype=dtype,
            kv_mb=nbytes / 1e6, median_ms=t.median_ms, min_ms=t.min_ms,
            gbps=nbytes / sec / 1e9))
        pbar.set_postfix_str(f"L={L} {nbytes/1e6:.0f}MB")
        pbar.update(1)
        del fn, bufs
        torch.cuda.empty_cache()
    pbar.close()
    return recs


def measure_decode_ms(R, L, H, H_kv, D, *, dtype="bf16",
                      device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """Median ms for one decode-attention step at (R, L, H, H_kv, D)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    fn, bufs = _decode_call(R, L, H, H_kv, D, _DTYPES[dtype], dev)
    t = measure(fn, device=dev, iters=iters, warmup=warmup)
    del fn, bufs
    torch.cuda.empty_cache()
    return t.median_ms


def decode_roofline_residual(recs: list[AttnRecord], b_peak_gbps: float) -> None:
    """Fill predicted_ms / efficiency using the memory roofline (decode is always
    memory-bound, so the ceiling is just B_peak). In place."""
    b = b_peak_gbps * 1e9
    for r in recs:
        elem = _DTYPES[r.dtype].itemsize
        nbytes = kv_bytes(r.kv_tokens, r.H_kv, r.D, elem)
        r.predicted_ms = nbytes / b * 1e3
        r.efficiency = r.predicted_ms / r.median_ms if r.median_ms > 0 else 0.0
