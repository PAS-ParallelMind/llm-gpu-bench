"""Flash-attention sweep — a hybrid model over the sequence plane (FlashInfer, paged KV).

vLLM's targeted attention library is FlashInfer, which dispatches a paged-KV call to one of
several underlying kernels. Which ones exist depends on the GPU: prefill can be fa2 (the
Ampere+ baseline), fa3 (Hopper SM90), cutlass, or trtllm-gen (Hopper/Blackwell); decode is
fa2 (CUDA-core or tensor-core) or trtllm-gen. So for each sweep point we try every candidate,
skip the ones that don't support this GPU/shape, and keep the FASTEST — the efficiency factor
is then the *best achievable* across FlashInfer's kernels, and we record which backend won.
(On Ada / RTX 4090 only fa2 runs; on Hopper/Blackwell the faster kernels are picked up.)

The op spans the whole (S_q, S_kv) plane, but the efficiency has two physics regimes that
want *different* scale variables, so we model them separately (one `attn_latency_ms` routes
on S_q):

  * DECODE (S_q = 1): always memory-bound; the decode wrapper's split-KV provides the
    parallelism. Efficiency collapses to a 1-D curve in total (block-padded) KV bytes —
    model-agnostic across head config, request count, and batch composition.
        KV_bytes = 2·elem·Σ_i ceil(L_i/block)·block · H_kv·D
        t = (KV_bytes / B_peak) / f_decode(KV_bytes)

  * PREFILL / CHUNKED (S_q > 1): a batched causal GEMM (per head QK^T then PV) over R·H
    heads. Efficiency is a 3-D surface over (S_q, S_kv, R·H) per head-dim D (H_kv washes
    out in this compute regime). The roofline spans both regimes:
        FLOPs = 4·H·D·R·(S_q·S_kv − S_q(S_q−1)/2);  bytes = 2·elem·R·(S_q·H·D + S_kv·H_kv·D)
        t = max(FLOPs/C_peak, bytes/B_peak) / f_prefill(S_q, S_kv, R·H, D)

Why hybrid: decode scales with R·S_kv·H_kv·D (KV bytes), prefill with R·H (parallelism) —
different functions of R and H, so they don't share one grid's axes (measured). vLLM splits
a continuous-batching step into a BatchDecode + a BatchPrefill wrapper, so this hybrid is
exactly how a real step decomposes, and mixed steps compose additively (t_prefill + t_decode).
"""
from __future__ import annotations

from dataclasses import dataclass

import flashinfer
import torch

from timing import measure, progress

BLOCK_SIZE = 16   # paged-KV page size (vLLM's default block size)
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Shape variables: R requests, H query heads, H_kv KV heads (GQA: H_kv <= H), D head dim,
# Sq query/chunk length (decode: 1), Sk KV/context length (>= Sq), L = Sk for decode.

# Decode curve: S_q=1 sweep over (R requests, context L) tracing f_decode(KV bytes). The
# head config is a vehicle — the curve is in KV bytes, so it is model-agnostic. Sweeping R
# too extends the KV-byte range to the saturation plateau (a big GPU needs far more in-flight
# bytes — e.g. B200's 8 TB/s) and, where different R·L land on the same KV bytes, checks the
# curve really collapses on total bytes (distribution-independent).
DECODE_L_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DECODE_R_GRID = [1, 8, 64]
DECODE_CONFIG = {"H": 32, "H_kv": 8, "D": 128}

# Prefill grid: (S_q <= S_kv) plane x parallelism R*H, per head-dim D (S_q >= 16; S_q=1
# is the decode curve). H_kv washes out in this compute regime, so a vehicle value is used.
ATTN_SQ_GRID = [16, 64, 256, 1024, 4096]
ATTN_SK_GRID = [16, 64, 256, 1024, 4096, 16384]
ATTN_RH_GRID = [32, 64, 128, 256, 512]
ATTN_D_GRID = [64, 128, 256]

# FlashInfer paged-KV kernel candidates tried per point (best wins; unsupported skipped).
# Decode is (backend, use_tensor_cores); prefill is just the backend name. Hard-coded to the
# full viable set; the try/skip in _best_call discovers which run on the current GPU:
#   decode : fa2, trtllm-gen (Blackwell SM100), cudnn       (all confirmed selectable)
#   prefill: fa2, fa3 (Hopper SM90), trtllm-gen (Blackwell)
# Excluded (can't run via these paged wrappers, any GPU): prefill cudnn (needs the block-table
# plan form, not the indptr/indices one we use), cute-dsl ("not yet supported for paged KV"),
# cutlass (ragged wrapper only).
# use_tensor_cores only toggles the fa2 decode path (CUDA-core kernel vs the tensor-core
# seqlen_q=1 "prefill" path — faster at large GQA group); trtllm-gen/cudnn pick their own
# kernels, so the flag is a don't-care for them (each listed once).
DECODE_CANDIDATES = [("fa2", False), ("fa2", True), ("trtllm-gen", False), ("cudnn", False)]
PREFILL_CANDIDATES = ["fa2", "fa3", "trtllm-gen"]


def _name(cand) -> str:
    return f"{cand[0]}{'+tc' if cand[1] else ''}" if isinstance(cand, tuple) else cand


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


# --- FlashInfer paged-KV wrappers (one per candidate; lazily built, re-planned per point) --
_WS_BYTES = 256 * 1024 * 1024
_wrappers: dict = {}
_disabled: set = set()   # (stage, D, cand) that failed -> not retried at this head-dim


def _wrapper(stage, cand, dev):
    key = (stage, cand)
    w = _wrappers.get(key)
    if w is None:
        ws = torch.empty(_WS_BYTES, dtype=torch.uint8, device=dev)
        if stage == "dec":
            be, tc = cand
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc, backend=be)
        else:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=cand)
        _wrappers[key] = w
    return w


def _paged(sks, H_kv, D, dt, dev):
    """Paged KV cache (NHD) + index arrays for requests with contexts `sks`; contiguous pages."""
    pages = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in sks]
    tot = sum(pages)
    indptr = torch.zeros(len(sks) + 1, device=dev, dtype=torch.int32)
    indptr[1:] = torch.tensor(pages, device=dev, dtype=torch.int32).cumsum(0)
    indices = torch.arange(tot, device=dev, dtype=torch.int32)
    last = torch.tensor([((s - 1) % BLOCK_SIZE) + 1 for s in sks], device=dev, dtype=torch.int32)
    kc = torch.randn(tot, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    vc = torch.randn(tot, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    return indptr, indices, last, (kc, vc)


def _decode_call(cand, sks, H, H_kv, D, dt, dev):
    w = _wrapper("dec", cand, dev)
    indptr, indices, last, kv = _paged(sks, H_kv, D, dt, dev)
    w.plan(indptr, indices, last, H, H_kv, D, BLOCK_SIZE, q_data_type=dt, kv_data_type=dt)
    q = torch.randn(len(sks), H, D, device=dev, dtype=dt)
    return (lambda: w.run(q, kv)), (q, kv)


def _prefill_call(cand, reqs, H, H_kv, D, dt, dev):
    w = _wrapper("pre", cand, dev)
    sqs = [sq for sq, _ in reqs]
    indptr, indices, last, kv = _paged([sk for _, sk in reqs], H_kv, D, dt, dev)
    qo = torch.zeros(len(reqs) + 1, device=dev, dtype=torch.int32)
    qo[1:] = torch.tensor(sqs, device=dev, dtype=torch.int32).cumsum(0)
    w.plan(qo, indptr, indices, last, H, H_kv, D, BLOCK_SIZE, causal=True,
           q_data_type=dt, kv_data_type=dt)
    q = torch.randn(sum(sqs), H, D, device=dev, dtype=dt)
    return (lambda: w.run(q, kv)), (q, kv)


def _best_call(stage, D, candidates, build, args, dev, iters, warmup):
    """Try each candidate kernel for one point, skipping unsupported ones; return the
    fastest as (median_ms, cand, fn, bufs) with the winner's call still live."""
    best = None
    for cand in candidates:
        if (stage, D, cand) in _disabled:
            continue
        try:
            fn, bufs = build(cand, *args)
            ms = measure(fn, device=dev, iters=iters, warmup=warmup).median_ms
        except Exception as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():    # shape too big for this candidate here —
                continue                             #   not unsupported, keep it for other points
            _disabled.add((stage, D, cand))          # architectural: unsupported on this GPU / D
            _wrappers.pop((stage, cand), None)       # free its workspace
            print(f"  [skip] {stage} {_name(cand)} D={D}: {str(e).splitlines()[0][:70]}")
            continue
        if best is None or ms < best[0]:
            prev, best = best, (ms, cand, fn, bufs)
            del prev                                 # drop previous winner's fn/bufs for GC
        else:
            del fn, bufs
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError(f"no FlashInfer {stage} backend supports D={D}, args={args[0]}")
    return best


@dataclass
class DecodeRecord:
    kv_tokens: int          # block-padded total KV tokens (R·L here)
    H_kv: int
    D: int
    dtype: str
    median_ms: float
    backend: str            # winning FlashInfer kernel
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
    backend: str            # winning FlashInfer kernel
    efficiency: float = 0.0


def measure_attn_ms(R, Sq, Sk, H, H_kv, D, *, dtype="bf16",
                    device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """Best-of-backends median ms for one homogeneous call (decode S_q=1 or prefill S_q>1)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    if Sq == 1:
        ms, _, fn, bufs = _best_call("dec", D, DECODE_CANDIDATES, _decode_call,
                                     ([Sk] * R, H, H_kv, D, dt, dev), dev, iters, warmup)
    else:
        ms, _, fn, bufs = _best_call("pre", D, PREFILL_CANDIDATES, _prefill_call,
                                     ([(Sq, Sk)] * R, H, H_kv, D, dt, dev), dev, iters, warmup)
    del fn, bufs
    torch.cuda.empty_cache()
    return ms


def measure_mixed_ms(reqs, H, H_kv, D, *, dtype="bf16",
                     device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """vLLM's FlashInfer mixed step: best prefill kernel + best decode kernel, back-to-back
    on one stream (separate kernels — so latency is additive, decode keeps its split-KV)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    pre_reqs = [(sq, sk) for sq, sk in reqs if sq > 1]
    dec_sks = [sk for sq, sk in reqs if sq == 1]
    fns, keep = [], []
    if pre_reqs:
        _, _, fn, bufs = _best_call("pre", D, PREFILL_CANDIDATES, _prefill_call,
                                    (pre_reqs, H, H_kv, D, dt, dev), dev, iters, warmup)
        fns.append(fn); keep += [fn, bufs]
    if dec_sks:
        _, _, fn, bufs = _best_call("dec", D, DECODE_CANDIDATES, _decode_call,
                                    (dec_sks, H, H_kv, D, dt, dev), dev, iters, warmup)
        fns.append(fn); keep += [fn, bufs]

    def step():
        for f in fns:
            f()

    t = measure(step, device=dev, iters=iters, warmup=warmup)
    del fns, keep
    torch.cuda.empty_cache()
    return t.median_ms


def run_decode_sweep(Ls, Rs, *, b_peak, H=32, H_kv=8, D=128, dtype="bf16",
                     device: int | torch.device = 0, iters=30, warmup=10):
    """S_q=1 sweep over (R requests, context L) -> the 1-D f_decode(KV bytes) curve
    (memory-bound). Multiple R extend the range and probe the KV-byte collapse."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    B = b_peak * 1e9
    recs: list[DecodeRecord] = []
    pbar = progress(len(Rs) * len(Ls), "decode")
    for R in Rs:
        for L in Ls:
            try:
                ms, cand, fn, bufs = _best_call("dec", D, DECODE_CANDIDATES, _decode_call,
                                                ([L] * R, H, H_kv, D, dt, dev), dev, iters, warmup)
            except RuntimeError as e:                 # no backend could run this point (e.g. OOM)
                print(f"  [skip pt] decode R={R} L={L}: {str(e).splitlines()[0][:50]}")
                pbar.update(1)
                torch.cuda.empty_cache()
                continue
            nbytes = kv_bytes(R * L, H_kv, D, elem)
            sec = ms * 1e-3
            recs.append(DecodeRecord(kv_tokens=R * L, H_kv=H_kv, D=D, dtype=dtype, median_ms=ms,
                        backend=_name(cand), efficiency=(nbytes / B / sec) if sec > 0 else 0.0))
            pbar.set_postfix_str(f"R={R} L={L} [{_name(cand)}]")
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
        try:
            ms, cand, fn, bufs = _best_call("pre", D, PREFILL_CANDIDATES, _prefill_call,
                                            ([(sq, sk)] * R, H, H_kv, D, dt, dev), dev, iters, warmup)
        except RuntimeError as e:                     # no backend could run this point (e.g. OOM)
            print(f"  [skip pt] prefill Sq={sq} Sk={sk} RH={R*H} D={D}: {str(e).splitlines()[0][:50]}")
            pbar.update(1)
            torch.cuda.empty_cache()
            continue
        tc = attn_flops(R, sq, sk, H, D) / C
        tm = attn_bytes(R, sq, sk, H, H_kv, D, elem) / B
        sec = ms * 1e-3
        recs.append(AttnRecord(Sq=sq, Sk=sk, RH=R * H, D=D, dtype=dtype, median_ms=ms,
                    regime="C" if tc > tm else "M", backend=_name(cand),
                    efficiency=(max(tc, tm) / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"Sq={sq} Sk={sk} RH={R*H} [{_name(cand)}]")
        pbar.update(1)
        del fn, bufs
        torch.cuda.empty_cache()
    pbar.close()
    return recs


def run_full_attn_sweep(*, c_peak, b_peak, dtype="bf16", device=0, iters=30, warmup=10):
    """Hybrid sweep: the decode KV-byte curve + the prefill grid over all head dims D."""
    decode = run_decode_sweep(DECODE_L_GRID, DECODE_R_GRID, b_peak=b_peak, dtype=dtype,
                              device=device, iters=iters, warmup=warmup, **DECODE_CONFIG)
    grid: list[AttnRecord] = []
    for D in ATTN_D_GRID:
        grid += run_attn_sweep(ATTN_SQ_GRID, ATTN_SK_GRID, ATTN_RH_GRID, D=D,
                               c_peak=c_peak, b_peak=b_peak, dtype=dtype, device=device,
                               iters=iters, warmup=warmup)
    return decode, grid
