"""Flash-attention sweep — a hybrid model over the sequence plane (vLLM, paged KV).

vLLM runs decode + prefill through one unified flash_attn_varlen_func (paged KV, varlen
cu_seqlens), so the op spans the whole (S_q, S_kv) plane. But the efficiency has two
physics regimes that want *different* scale variables, so we model them separately:

  * DECODE (S_q = 1): always memory-bound; split-KV provides the parallelism. Efficiency
    collapses to a 1-D curve in total (block-padded) KV bytes — model-agnostic across
    head config, request count, and batch composition. (Validated ~2%.)
        KV_bytes = 2·elem·Σ_i ceil(L_i/block)·block · H_kv·D
        t = (KV_bytes / B_peak) / f_decode(KV_bytes)

  * PREFILL / CHUNKED (S_q > 1): a batched causal GEMM (per head QK^T then PV) over R·H
    heads. Efficiency is a 3-D surface over (S_q, S_kv, R·H) per head-dim D (H_kv washes
    out in this compute regime). The roofline spans both regimes:
        FLOPs = 4·H·D·R·(S_q·S_kv − S_q(S_q−1)/2);  bytes = 2·elem·R·(S_q·H·D + S_kv·H_kv·D)
        t = max(FLOPs/C_peak, bytes/B_peak) / f_prefill(S_q, S_kv, R·H, D)

Why hybrid: decode scales with R·S_kv·H_kv·D (KV bytes), prefill with R·H (parallelism) —
different functions of R and H, so they don't share one grid's axes (measured). The
kernel is vLLM's flash_attn_varlen_func + paged KV (checked vs a bottom-right-causal ref);
we add the timing harness and roofline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from timing import measure, progress
from vllm.vllm_flash_attn import flash_attn_varlen_func

BLOCK_SIZE = 16   # vLLM's default paged-KV block size
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Shape variables: R requests, H query heads, H_kv KV heads (GQA: H_kv <= H), D head dim,
# Sq query/chunk length (decode: 1), Sk KV/context length (>= Sq).

# Decode curve: S_q=1 sweep over context L (= S_kv) tracing f_decode(KV bytes). The
# head config is a vehicle — the curve is in KV bytes, so it is model-agnostic.
DECODE_L_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DECODE_CONFIG = {"H": 32, "H_kv": 8, "D": 128, "R": 8}

# Prefill grid: (S_q <= S_kv) plane x parallelism R*H, per head-dim D (S_q >= 16; S_q=1
# is the decode curve). H_kv washes out in this compute regime, so a vehicle value is used.
ATTN_SQ_GRID = [16, 64, 256, 1024, 4096]
ATTN_SK_GRID = [16, 64, 256, 1024, 4096, 16384]
ATTN_RH_GRID = [32, 128, 512]
ATTN_D_GRID = [64, 128, 256]


def kv_bytes(kv_tokens: int, H_kv: int, D: int, elem: int = 2) -> float:
    """Decode KV-cache bytes read (K+V). `kv_tokens` = block-padded Σ_i L_i."""
    return 2 * elem * kv_tokens * H_kv * D


def padded_kv_tokens(Ls, block_size: int = BLOCK_SIZE) -> int:
    """KV tokens the kernel reads: each request's context rounded up to a whole block."""
    return sum(((L + block_size - 1) // block_size) * block_size for L in Ls)


def attn_flops(R: int, Sq: int, Sk: int, H: int, D: int) -> int:
    pairs = Sq * Sk - Sq * (Sq - 1) // 2          # causal (query, key) pairs per head
    return 4 * H * D * R * pairs


def attn_bytes(R: int, Sq: int, Sk: int, H: int, H_kv: int, D: int, elem: int = 2) -> int:
    return 2 * elem * R * (Sq * H * D + Sk * H_kv * D)   # Q,O over S_q + K,V over S_kv


def _attn_call(R, Sq, Sk, H, H_kv, D, dt, dev):
    nblk = (Sk + BLOCK_SIZE - 1) // BLOCK_SIZE
    nb = R * nblk
    q = torch.randn(R * Sq, H, D, device=dev, dtype=dt)
    kc = torch.randn(nb, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    vc = torch.randn(nb, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    bt = torch.arange(nb, device=dev, dtype=torch.int32).view(R, nblk)
    cu = torch.arange(0, (R + 1) * Sq, Sq, device=dev, dtype=torch.int32)
    su = torch.full((R,), Sk, device=dev, dtype=torch.int32)
    fn = lambda: flash_attn_varlen_func(
        q=q, k=kc, v=vc, max_seqlen_q=Sq, cu_seqlens_q=cu, max_seqlen_k=Sk,
        seqused_k=su, softmax_scale=1.0 / math.sqrt(D), causal=True, block_table=bt)
    return fn, (q, kc, vc)


def measure_attn_ms(R, Sq, Sk, H, H_kv, D, *, dtype="bf16",
                    device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """Median ms for one attention call (decode S_q=1 or prefill S_q>1)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    fn, bufs = _attn_call(R, Sq, Sk, H, H_kv, D, _DTYPES[dtype], dev)
    t = measure(fn, device=dev, iters=iters, warmup=warmup)
    del fn, bufs
    torch.cuda.empty_cache()
    return t.median_ms


@dataclass
class DecodeRecord:
    kv_tokens: int          # block-padded total KV tokens (R·L here)
    H_kv: int
    D: int
    dtype: str
    median_ms: float
    efficiency: float = 0.0


@dataclass
class AttnRecord:
    Sq: int
    Sk: int
    RH: int                 # total heads R*H (the parallelism axis)
    D: int
    dtype: str
    median_ms: float
    regime: str             # "C" compute-bound, "M" memory-bound
    efficiency: float = 0.0


def run_decode_sweep(Ls, *, b_peak, H=32, H_kv=8, D=128, R=8, dtype="bf16",
                     device: int | torch.device = 0, iters=30, warmup=10):
    """S_q=1 sweep over context L -> the 1-D f_decode(KV bytes) curve (memory-bound)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    B = b_peak * 1e9
    recs: list[DecodeRecord] = []
    pbar = progress(len(Ls), "decode")
    for L in Ls:
        fn, bufs = _attn_call(R, 1, L, H, H_kv, D, dt, dev)
        t = measure(fn, device=dev, iters=iters, warmup=warmup)
        nbytes = kv_bytes(R * L, H_kv, D, elem)
        sec = t.median_ms * 1e-3
        recs.append(DecodeRecord(kv_tokens=R * L, H_kv=H_kv, D=D, dtype=dtype,
                    median_ms=t.median_ms, efficiency=(nbytes / B / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"L={L}")
        pbar.update(1)
        del fn, bufs
        torch.cuda.empty_cache()
    pbar.close()
    return recs


def run_attn_sweep(Sqs, Sks, RHs, *, D, c_peak, b_peak, H=32, H_kv=8, dtype="bf16",
                   device: int | torch.device = 0, iters=30, warmup=10):
    """Prefill grid: the (S_q <= S_kv) plane x R*H for one head-dim D (S_q > 1)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    C, B = c_peak * 1e12, b_peak * 1e9
    work = [(sq, sk, rh) for rh in RHs for sq in Sqs for sk in Sks if sk >= sq]
    recs: list[AttnRecord] = []
    pbar = progress(len(work), f"prefill D={D}")
    for sq, sk, rh in work:
        R = max(rh // H, 1)
        fn, bufs = _attn_call(R, sq, sk, H, H_kv, D, dt, dev)
        t = measure(fn, device=dev, iters=iters, warmup=warmup)
        tc = attn_flops(R, sq, sk, H, D) / C
        tm = attn_bytes(R, sq, sk, H, H_kv, D, elem) / B
        sec = t.median_ms * 1e-3
        recs.append(AttnRecord(Sq=sq, Sk=sk, RH=R * H, D=D, dtype=dtype, median_ms=t.median_ms,
                    regime="C" if tc > tm else "M",
                    efficiency=(max(tc, tm) / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"Sq={sq} Sk={sk} RH={R*H}")
        pbar.update(1)
        del fn, bufs
        torch.cuda.empty_cache()
    pbar.close()
    return recs


def run_full_attn_sweep(*, c_peak, b_peak, dtype="bf16", device=0, iters=30, warmup=10):
    """Hybrid sweep: the decode KV-byte curve + the prefill grid over all head dims D."""
    decode = run_decode_sweep(DECODE_L_GRID, b_peak=b_peak, dtype=dtype,
                              device=device, iters=iters, warmup=warmup, **DECODE_CONFIG)
    grid: list[AttnRecord] = []
    for D in ATTN_D_GRID:
        grid += run_attn_sweep(ATTN_SQ_GRID, ATTN_SK_GRID, ATTN_RH_GRID, D=D,
                               c_peak=c_peak, b_peak=b_peak, dtype=dtype,
                               device=device, iters=iters, warmup=warmup)
    return decode, grid
