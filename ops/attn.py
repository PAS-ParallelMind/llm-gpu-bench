"""Flash-attention sweep — a hybrid model over the sequence plane (FlashInfer, paged KV).

vLLM's targeted attention library is FlashInfer, which dispatches a paged-KV call to one of
several underlying kernels. Which ones exist depends on the GPU: prefill can be fa2 (the
Ampere+ baseline), fa3 (Hopper SM90), cutlass, or trtllm-gen (Hopper/Blackwell); decode is
fa2 (CUDA-core or tensor-core) or trtllm-gen. So for each sweep point we try every candidate,
skip the ones that don't support this GPU/shape, and keep the FASTEST — the efficiency factor
is then the *best achievable* across FlashInfer's kernels, and we record which backend won.
(On Ada / RTX 4090 only fa2 runs; on Hopper/Blackwell the faster kernels are picked up.)

The op spans the whole (q_len, kv_len) plane, but the efficiency has two physics regimes that
want *different* scale variables, so we model them separately (one `attn_latency_ms` routes
on q_len):

  * DECODE (q_len = 1): always memory-bound; the decode wrapper's split-KV provides the
    parallelism. Efficiency collapses to a 1-D curve in total (block-padded) KV bytes —
    model-agnostic across head config, request count, and batch composition.
        KV_bytes = 2·elem·Σ_i ceil(L_i/block)·block · n_kv_heads·head_dim
        t = (KV_bytes / B_peak) / f_decode(KV_bytes)

  * PREFILL / CHUNKED (q_len > 1): a batched causal GEMM (per head QK^T then PV) over
    total_heads = n_req·n_heads. Efficiency is a 3-D surface over (q_len, kv_len, total_heads) per
    head-dim (n_kv_heads washes out in this compute regime). The roofline spans both regimes:
        FLOPs = 4·n_heads·head_dim·n_req·(q_len·kv_len − q_len(q_len−1)/2)
        bytes = 2·elem·n_req·(q_len·n_heads·head_dim + kv_len·n_kv_heads·head_dim)
        t = max(FLOPs/C_peak, bytes/B_peak) / f_prefill(q_len, kv_len, total_heads, head_dim)

Why hybrid: decode scales with n_req·kv_len·n_kv_heads·head_dim (KV bytes), prefill with total_heads
(parallelism) — different functions of n_req and n_heads, so they don't share one grid's axes (measured).
vLLM splits a continuous-batching step into a BatchDecode + a BatchPrefill wrapper, so this
hybrid is exactly how a real step decomposes, and mixed steps compose additively (t_pf + t_dec).
"""
from __future__ import annotations

from dataclasses import dataclass

import flashinfer
import torch

from timing import measure, progress

BLOCK_SIZE = 16   # paged-KV page size (vLLM's default block size)
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Decode curve: q_len=1 sweep over (n_req requests, context kv_len) tracing f_decode(KV bytes). The
# head config is a vehicle — the curve is in KV bytes, so it is model-agnostic. Sweeping n_req
# too extends the KV-byte range to the saturation plateau (a big GPU needs far more in-flight
# bytes — e.g. B200's 8 TB/s) and, where different n_req·kv_len land on the same KV bytes, checks
# the curve really collapses on total bytes (distribution-independent).
DECODE_L_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DECODE_R_GRID = [1, 8, 64]
DECODE_CONFIG = {"n_heads": 32, "n_kv_heads": 8, "head_dim": 128}

# Prefill grid: (q_len <= kv_len) plane x parallelism total_heads, per head-dim (q_len >= 16; q_len=1
# is the decode curve). n_kv_heads washes out in this compute regime, so a vehicle value is used.
# total_heads reaches down to 4 (below the 32-head vehicle) so tensor-parallel single-request
# prefill is covered: TP shards heads, so a per-rank config has total_heads = model_heads/TP (e.g. a
# 32-head model at TP=8 prefills 4 heads on a rank). Those small-total_heads points are realized as
# one request with n_heads=total_heads (see _rh_vehicle), matching how they run, not clamped to 32.
ATTN_SQ_GRID = [16, 64, 256, 1024, 4096]
ATTN_SK_GRID = [16, 64, 256, 1024, 4096, 16384]
ATTN_RH_GRID = [4, 8, 16, 32, 64, 128, 256, 512]
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


def kv_bytes(kv_tokens: int, n_kv_heads: int, head_dim: int, elem: int = 2) -> float:
    """Decode KV-cache bytes read (K+V). `kv_tokens` = block-padded Σ_i kv_len_i."""
    return 2 * elem * kv_tokens * n_kv_heads * head_dim


def padded_kv_tokens(kv_lens, block_size: int = BLOCK_SIZE) -> int:
    """KV tokens the kernel reads: each request's context rounded up to a whole block."""
    return sum(((kv_len + block_size - 1) // block_size) * block_size for kv_len in kv_lens)


def attn_flops(n_req: int, q_len: int, kv_len: int, n_heads: int, head_dim: int) -> int:
    pairs = q_len * kv_len - q_len * (q_len - 1) // 2     # causal (query, key) pairs per head
    return 4 * n_heads * head_dim * n_req * pairs


def attn_bytes(n_req: int, q_len: int, kv_len: int, n_heads: int, n_kv_heads: int,
               head_dim: int, elem: int = 2) -> int:
    # Q,O over q_len + K,V over kv_len
    return 2 * elem * n_req * (q_len * n_heads * head_dim + kv_len * n_kv_heads * head_dim)


# --- FlashInfer paged-KV wrappers (one per candidate; lazily built, re-planned per point) --
_WS_BYTES = 256 * 1024 * 1024
_wrappers: dict = {}

# Backend viability bookkeeping, per (stage, head_dim, cand). A candidate can fail for two very
# different reasons and they must not be conflated:
#   * ARCHITECTURAL -- the kernel doesn't exist/run on this GPU at this head-dim (fa3 off
#     Hopper, trtllm-gen off Blackwell). Worth remembering: stop retrying it.
#   * SHAPE-SPECIFIC -- the kernel runs here, but rejects THIS point (sequence too long,
#     unsupported GQA group, layout/workspace constraint). Must NOT be remembered, or one odd
#     shape strips the candidate from every later point and a slower runner-up gets recorded
#     as "best" -- silently corrupting the best-of-backends result.
# We can't read the failure's intent, so we infer it: a candidate that has EVER succeeded at
# this (stage, head_dim) is architecturally fine, so later failures are shape-specific. One that
# has never succeeded is only condemned after it fails _FAIL_LIMIT *distinct* points -- so a single
# pathological first shape can't disable a working kernel, while a genuinely absent one costs
# just a few cheap probes instead of being retried across the whole grid.
_FAIL_LIMIT = 3
_disabled: set = set()   # (stage, head_dim, cand) proven unsupported here -> not retried
_ok: set = set()         # (stage, head_dim, cand) that has succeeded at least once
_fails: dict = {}        # (stage, head_dim, cand) -> failed-point count (never-succeeded candidates)


def reset_backend_cache() -> None:
    """Forget which backends are viable/disabled (and drop their workspaces).

    The bookkeeping above is process-global and deliberately sticky, which is right for a grid
    sweep but wrong for ad-hoc probing, where an early exotic shape can bias what later,
    unrelated measurements are even allowed to try. Call this between independent measurements
    to give every candidate a clean shot."""
    _disabled.clear()
    _ok.clear()
    _fails.clear()
    _wrappers.clear()


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


def _paged(kv_lens, n_kv_heads, head_dim, dt, dev):
    """Paged KV cache (NHD) + index arrays for requests with contexts `kv_lens`; contiguous pages."""
    pages = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in kv_lens]
    tot = sum(pages)
    indptr = torch.zeros(len(kv_lens) + 1, device=dev, dtype=torch.int32)
    indptr[1:] = torch.tensor(pages, device=dev, dtype=torch.int32).cumsum(0)
    indices = torch.arange(tot, device=dev, dtype=torch.int32)
    last = torch.tensor([((s - 1) % BLOCK_SIZE) + 1 for s in kv_lens], device=dev, dtype=torch.int32)
    kc = torch.randn(tot, BLOCK_SIZE, n_kv_heads, head_dim, device=dev, dtype=dt)
    vc = torch.randn(tot, BLOCK_SIZE, n_kv_heads, head_dim, device=dev, dtype=dt)
    return indptr, indices, last, (kc, vc)


def _decode_call(cand, kv_lens, n_heads, n_kv_heads, head_dim, dt, dev):
    w = _wrapper("dec", cand, dev)
    indptr, indices, last, kv = _paged(kv_lens, n_kv_heads, head_dim, dt, dev)
    w.plan(indptr, indices, last, n_heads, n_kv_heads, head_dim, BLOCK_SIZE,
           q_data_type=dt, kv_data_type=dt)
    q = torch.randn(len(kv_lens), n_heads, head_dim, device=dev, dtype=dt)
    return (lambda: w.run(q, kv)), (q, kv)


def _prefill_call(cand, reqs, n_heads, n_kv_heads, head_dim, dt, dev):
    w = _wrapper("pre", cand, dev)
    q_lens = [q_len for q_len, _ in reqs]
    indptr, indices, last, kv = _paged([kv_len for _, kv_len in reqs], n_kv_heads, head_dim, dt, dev)
    qo = torch.zeros(len(reqs) + 1, device=dev, dtype=torch.int32)
    qo[1:] = torch.tensor(q_lens, device=dev, dtype=torch.int32).cumsum(0)
    w.plan(qo, indptr, indices, last, n_heads, n_kv_heads, head_dim, BLOCK_SIZE, causal=True,
           q_data_type=dt, kv_data_type=dt)
    q = torch.randn(sum(q_lens), n_heads, head_dim, device=dev, dtype=dt)
    return (lambda: w.run(q, kv)), (q, kv)


def _best_call(stage, head_dim, candidates, build, args, dev, iters, warmup):
    """Try each candidate kernel for one point, skipping unsupported ones; return the
    fastest as (median_ms, cand, fn, bufs) with the winner's call still live."""
    best = None
    for cand in candidates:
        key = (stage, head_dim, cand)
        if key in _disabled:
            continue
        try:
            fn, bufs = build(cand, *args)
            ms = measure(fn, device=dev, iters=iters, warmup=warmup).median_ms
        except Exception as e:
            torch.cuda.empty_cache()
            msg = str(e).splitlines()[0][:70]
            if "out of memory" in str(e).lower():    # shape too big for this candidate here —
                continue                             #   not unsupported, keep it for other points
            if key in _ok:
                # Already proven to run at this head-dim -> this is a shape-specific reject.
                # Drop it for THIS point only; the candidate stays live for the rest of the grid.
                print(f"  [shape-skip] {stage} {_name(cand)} head_dim={head_dim}: {msg}")
                continue
            n = _fails[key] = _fails.get(key, 0) + 1
            if n >= _FAIL_LIMIT:                     # failed enough distinct points -> unsupported
                _disabled.add(key)
                _wrappers.pop((stage, cand), None)   # free its workspace
                print(f"  [disable] {stage} {_name(cand)} head_dim={head_dim} after {n} points: {msg}")
            else:
                print(f"  [skip {n}/{_FAIL_LIMIT}] {stage} {_name(cand)} head_dim={head_dim}: {msg}")
            continue
        _ok.add(key)                                 # proven viable at this (stage, head-dim)
        _fails.pop(key, None)
        if best is None or ms < best[0]:
            prev, best = best, (ms, cand, fn, bufs)
            del prev                                 # drop previous winner's fn/bufs for GC
        else:
            del fn, bufs
        torch.cuda.empty_cache()
    if best is None:
        live = [_name(c) for c in candidates if (stage, head_dim, c) not in _disabled]
        raise RuntimeError(f"no FlashInfer {stage} backend ran head_dim={head_dim}, args={args[0]} "
                           f"(tried {live or 'none — all disabled on this GPU'})")
    return best


@dataclass
class DecodeRecord:
    kv_tokens: int          # block-padded total KV tokens (n_req·kv_len here)
    n_kv_heads: int
    head_dim: int
    dtype: str
    median_ms: float
    backend: str            # winning FlashInfer kernel
    tflops: float = 0.0     # achieved compute throughput
    gbps: float = 0.0       # achieved memory throughput
    efficiency: float = 0.0

    def result(self) -> dict:
        return {"shape": {"kind": "decode", "kv_tokens": self.kv_tokens,
                          "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
                          "backend": self.backend},
                "latency_ms": self.median_ms, "tflops": self.tflops, "gbps": self.gbps,
                "efficiency": self.efficiency}


@dataclass
class AttnRecord:
    q_len: int
    kv_len: int
    total_heads: int        # n_req·n_heads (the parallelism axis)
    head_dim: int
    dtype: str
    median_ms: float
    regime: str             # "C" compute-bound, "M" memory-bound
    backend: str            # winning FlashInfer kernel
    tflops: float = 0.0     # achieved compute throughput
    gbps: float = 0.0       # achieved memory throughput
    efficiency: float = 0.0

    def result(self) -> dict:
        return {"shape": {"kind": "prefill", "q_len": self.q_len, "kv_len": self.kv_len,
                          "total_heads": self.total_heads, "head_dim": self.head_dim,
                          "backend": self.backend},
                "latency_ms": self.median_ms, "tflops": self.tflops, "gbps": self.gbps,
                "efficiency": self.efficiency}


def measure_attn_ms(n_req, q_len, kv_len, n_heads, n_kv_heads, head_dim, *, dtype="bf16",
                    device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """Best-of-backends median ms for one homogeneous call (decode q_len=1 or prefill q_len>1)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    if q_len == 1:
        ms, _, fn, bufs = _best_call("dec", head_dim, DECODE_CANDIDATES, _decode_call,
                                     ([kv_len] * n_req, n_heads, n_kv_heads, head_dim, dt, dev),
                                     dev, iters, warmup)
    else:
        ms, _, fn, bufs = _best_call("pre", head_dim, PREFILL_CANDIDATES, _prefill_call,
                                     ([(q_len, kv_len)] * n_req, n_heads, n_kv_heads, head_dim, dt, dev),
                                     dev, iters, warmup)
    del fn, bufs
    torch.cuda.empty_cache()
    return ms


def measure_mixed_ms(reqs, n_heads, n_kv_heads, head_dim, *, dtype="bf16",
                     device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """vLLM's FlashInfer mixed step: best prefill kernel + best decode kernel, back-to-back
    on one stream (separate kernels — so latency is additive, decode keeps its split-KV).
    `reqs` is a list of (q_len, kv_len): q_len>1 rows go to prefill, q_len==1 rows to decode."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    pre_reqs = [(q_len, kv_len) for q_len, kv_len in reqs if q_len > 1]
    dec_kv_lens = [kv_len for q_len, kv_len in reqs if q_len == 1]
    fns, keep = [], []
    if pre_reqs:
        _, _, fn, bufs = _best_call("pre", head_dim, PREFILL_CANDIDATES, _prefill_call,
                                    (pre_reqs, n_heads, n_kv_heads, head_dim, dt, dev), dev, iters, warmup)
        fns.append(fn); keep += [fn, bufs]
    if dec_kv_lens:
        _, _, fn, bufs = _best_call("dec", head_dim, DECODE_CANDIDATES, _decode_call,
                                    (dec_kv_lens, n_heads, n_kv_heads, head_dim, dt, dev), dev, iters, warmup)
        fns.append(fn); keep += [fn, bufs]

    def step():
        for f in fns:
            f()

    t = measure(step, device=dev, iters=iters, warmup=warmup)
    del fns, keep
    torch.cuda.empty_cache()
    return t.median_ms


def run_decode_sweep(kv_lens_grid, n_reqs_grid, *, b_peak, n_heads=32, n_kv_heads=8, head_dim=128,
                     dtype="bf16", device: int | torch.device = 0, iters=30, warmup=10):
    """q_len=1 sweep over (n_req requests, context kv_len) -> the 1-D f_decode(KV bytes) curve
    (memory-bound). Multiple n_req extend the range and probe the KV-byte collapse."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    B = b_peak * 1e9
    recs: list[DecodeRecord] = []
    pbar = progress(len(n_reqs_grid) * len(kv_lens_grid), "decode")
    for n_req in n_reqs_grid:
        for kv_len in kv_lens_grid:
            try:
                ms, cand, fn, bufs = _best_call("dec", head_dim, DECODE_CANDIDATES, _decode_call,
                                                ([kv_len] * n_req, n_heads, n_kv_heads, head_dim, dt, dev),
                                                dev, iters, warmup)
            except RuntimeError as e:                 # no backend could run this point (e.g. OOM)
                print(f"  [skip pt] decode n_req={n_req} kv_len={kv_len}: {str(e).splitlines()[0][:50]}")
                pbar.update(1)
                torch.cuda.empty_cache()
                continue
            nbytes = kv_bytes(n_req * kv_len, n_kv_heads, head_dim, elem)
            flops = attn_flops(n_req, 1, kv_len, n_heads, head_dim)
            sec = ms * 1e-3
            recs.append(DecodeRecord(kv_tokens=n_req * kv_len, n_kv_heads=n_kv_heads, head_dim=head_dim,
                        dtype=dtype, median_ms=ms, backend=_name(cand),
                        tflops=flops / sec / 1e12 if sec > 0 else 0.0,
                        gbps=nbytes / sec / 1e9 if sec > 0 else 0.0,
                        efficiency=(nbytes / B / sec) if sec > 0 else 0.0))
            pbar.set_postfix_str(f"n_req={n_req} kv_len={kv_len} [{_name(cand)}]")
            pbar.update(1)
            del fn, bufs
            torch.cuda.empty_cache()
    pbar.close()
    return recs


def _rh_vehicle(total_heads: int, n_heads_base: int = 32, n_kv_heads_base: int = 8) -> tuple[int, int, int]:
    """Realize a target total_heads (= n_req·n_heads) as (n_req, n_heads, n_kv_heads). total_heads >= base:
    n_req requests of n_heads_base heads (the batched-prefill vehicle). total_heads < base: ONE
    request with n_heads=total_heads (n_kv_heads=1) -- this is how tensor-parallel single-request
    prefill actually runs (few heads per rank), so the small-total_heads grid point matches the real
    shape instead of being faked from many 32-head requests. n_kv_heads washes out in the prefill
    compute regime, so the vehicle value is a don't-care for efficiency."""
    if total_heads >= n_heads_base:
        return max(total_heads // n_heads_base, 1), n_heads_base, n_kv_heads_base
    return 1, total_heads, 1


def run_attn_sweep(q_lens_grid, kv_lens_grid, total_heads_grid, *, head_dim, c_peak, b_peak,
                   n_heads=32, n_kv_heads=8, dtype="bf16",
                   device: int | torch.device = 0, iters=30, warmup=10):
    """Prefill grid: the (q_len <= kv_len) plane x total_heads for one head-dim (q_len > 1)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    C, B = c_peak * 1e12, b_peak * 1e9
    work = [(q_len, kv_len, th) for th in total_heads_grid
            for q_len in q_lens_grid for kv_len in kv_lens_grid if kv_len >= q_len]
    recs: list[AttnRecord] = []
    pbar = progress(len(work), f"prefill head_dim={head_dim}")
    for q_len, kv_len, total_heads in work:
        # small total_heads -> single request, n_heads = total_heads
        n_req, n_heads_pt, n_kv_heads_pt = _rh_vehicle(total_heads, n_heads, n_kv_heads)
        try:
            ms, cand, fn, bufs = _best_call("pre", head_dim, PREFILL_CANDIDATES, _prefill_call,
                                            ([(q_len, kv_len)] * n_req, n_heads_pt, n_kv_heads_pt,
                                             head_dim, dt, dev), dev, iters, warmup)
        except RuntimeError as e:                     # no backend could run this point (e.g. OOM)
            print(f"  [skip pt] prefill q_len={q_len} kv_len={kv_len} total_heads={n_req*n_heads_pt} "
                  f"head_dim={head_dim}: {str(e).splitlines()[0][:50]}")
            pbar.update(1)
            torch.cuda.empty_cache()
            continue
        flops = attn_flops(n_req, q_len, kv_len, n_heads_pt, head_dim)
        nbytes = attn_bytes(n_req, q_len, kv_len, n_heads_pt, n_kv_heads_pt, head_dim, elem)
        tc, tm = flops / C, nbytes / B
        sec = ms * 1e-3
        recs.append(AttnRecord(q_len=q_len, kv_len=kv_len, total_heads=n_req * n_heads_pt,
                    head_dim=head_dim, dtype=dtype, median_ms=ms,
                    regime="C" if tc > tm else "M", backend=_name(cand),
                    tflops=flops / sec / 1e12 if sec > 0 else 0.0,
                    gbps=nbytes / sec / 1e9 if sec > 0 else 0.0,
                    efficiency=(max(tc, tm) / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"q_len={q_len} kv_len={kv_len} total_heads={n_req*n_heads_pt} [{_name(cand)}]")
        pbar.update(1)
        del fn, bufs
        torch.cuda.empty_cache()
    pbar.close()
    return recs


def run_full_attn_sweep(*, c_peak, b_peak, dtype="bf16", device=0, iters=30, warmup=10):
    """Hybrid sweep: the decode KV-byte curve + the prefill grid over all head dims."""
    decode = run_decode_sweep(DECODE_L_GRID, DECODE_R_GRID, b_peak=b_peak, dtype=dtype,
                              device=device, iters=iters, warmup=warmup, **DECODE_CONFIG)
    grid: list[AttnRecord] = []
    for head_dim in ATTN_D_GRID:
        grid += run_attn_sweep(ATTN_SQ_GRID, ATTN_SK_GRID, ATTN_RH_GRID, head_dim=head_dim,
                               c_peak=c_peak, b_peak=b_peak, dtype=dtype, device=device,
                               iters=iters, warmup=warmup)
    return decode, grid
