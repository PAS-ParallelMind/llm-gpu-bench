"""FlashInfer attention backend — same hybrid sweep as attn.py, different kernels.

vLLM's FlashInfer backend (vllm/v1/attention/backends/flashinfer.py) splits a step into a
BatchDecodeWithPagedKVCacheWrapper (S_q=1 requests) and a BatchPrefillWithPagedKVCacheWrapper
(S_q>1), launched separately. We mirror that: decode uses the decode wrapper (its own
split-KV, so it stays memory-bound and fast even when prefill is co-batched), prefill uses
the prefill wrapper.

The decode-curve and prefill-grid *definitions* (roofline, grids, efficiency, dataclasses)
are shared with attn.py — only the kernel call differs — so the two backends are measured
identically and compare directly. attn.py's sweeps take a pluggable `call=`; here it routes
to the right wrapper, letting the shared drivers run FlashInfer unchanged.

Paged KV: NHD layout, (k_cache, v_cache) each (num_pages, page_size, H_kv, D); each request
owns a contiguous page range. plan() is host-side scheduling (done once, outside the timed
region); only run() is measured.
"""
from __future__ import annotations

import flashinfer
import torch

import attn
from attn import _DTYPES, BLOCK_SIZE
from timing import measure

_WS_BYTES = 256 * 1024 * 1024            # FlashInfer scratch / scheduling workspace
_dec_wrap = _pre_wrap = _ws_dec = _ws_pre = None


def _wrappers(dev):
    """Lazily build the decode + prefill wrappers (separate workspaces so a mixed step can
    plan both before running either)."""
    global _dec_wrap, _pre_wrap, _ws_dec, _ws_pre
    if _dec_wrap is None:
        _ws_dec = torch.empty(_WS_BYTES, dtype=torch.uint8, device=dev)
        _ws_pre = torch.empty(_WS_BYTES, dtype=torch.uint8, device=dev)
        _dec_wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_ws_dec, "NHD")
        _pre_wrap = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_ws_pre, "NHD")
    return _dec_wrap, _pre_wrap


def _paged(sks, H_kv, D, dt, dev):
    """Paged KV cache (NHD) + index arrays for requests with contexts `sks`; pages are
    laid out contiguously per request."""
    pages = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in sks]
    tot = sum(pages)
    indptr = torch.zeros(len(sks) + 1, device=dev, dtype=torch.int32)
    indptr[1:] = torch.tensor(pages, device=dev, dtype=torch.int32).cumsum(0)
    indices = torch.arange(tot, device=dev, dtype=torch.int32)
    last = torch.tensor([((s - 1) % BLOCK_SIZE) + 1 for s in sks], device=dev, dtype=torch.int32)
    kc = torch.randn(tot, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    vc = torch.randn(tot, BLOCK_SIZE, H_kv, D, device=dev, dtype=dt)
    return indptr, indices, last, (kc, vc)


def _decode_run(R, L, H, H_kv, D, dt, dev):
    dec, _ = _wrappers(dev)
    indptr, indices, last, kv = _paged([L] * R, H_kv, D, dt, dev)
    dec.plan(indptr, indices, last, H, H_kv, D, BLOCK_SIZE, q_data_type=dt, kv_data_type=dt)
    q = torch.randn(R, H, D, device=dev, dtype=dt)
    return (lambda: dec.run(q, kv)), (q, kv)


def _prefill_run(R, Sq, Sk, H, H_kv, D, dt, dev):
    _, pre = _wrappers(dev)
    indptr, indices, last, kv = _paged([Sk] * R, H_kv, D, dt, dev)
    qo = torch.arange(0, (R + 1) * Sq, Sq, device=dev, dtype=torch.int32)
    pre.plan(qo, indptr, indices, last, H, H_kv, D, BLOCK_SIZE, causal=True,
             q_data_type=dt, kv_data_type=dt)
    q = torch.randn(R * Sq, H, D, device=dev, dtype=dt)
    return (lambda: pre.run(q, kv)), (q, kv)


def _attn_call(R, Sq, Sk, H, H_kv, D, dt, dev):
    """Match attn._attn_call's signature so attn.py's shared sweeps drive FlashInfer:
    decode wrapper for S_q=1, prefill wrapper otherwise."""
    if Sq == 1:
        return _decode_run(R, Sk, H, H_kv, D, dt, dev)
    return _prefill_run(R, Sq, Sk, H, H_kv, D, dt, dev)


def measure_attn_ms(R, Sq, Sk, H, H_kv, D, **kw):
    return attn.measure_attn_ms(R, Sq, Sk, H, H_kv, D, call=_attn_call, **kw)


def run_full_attn_sweep(**kw):
    return attn.run_full_attn_sweep(call=_attn_call, **kw)


def measure_mixed_ms(reqs, H, H_kv, D, *, dtype="bf16",
                     device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """vLLM's FlashInfer mixed step: prefill wrapper then decode wrapper, back-to-back on
    one stream (separate kernels — so latency is additive, decode keeps its split-KV)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    dec, pre = _wrappers(dev)
    pre_reqs = [(sq, sk) for sq, sk in reqs if sq > 1]
    dec_sks = [sk for sq, sk in reqs if sq == 1]
    fns, bufs = [], []
    if pre_reqs:
        sqs = [sq for sq, _ in pre_reqs]
        indptr, indices, last, kv = _paged([sk for _, sk in pre_reqs], H_kv, D, dt, dev)
        qo = torch.zeros(len(pre_reqs) + 1, device=dev, dtype=torch.int32)
        qo[1:] = torch.tensor(sqs, device=dev, dtype=torch.int32).cumsum(0)
        pre.plan(qo, indptr, indices, last, H, H_kv, D, BLOCK_SIZE, causal=True,
                 q_data_type=dt, kv_data_type=dt)
        q = torch.randn(sum(sqs), H, D, device=dev, dtype=dt)
        fns.append(lambda q=q, kv=kv: pre.run(q, kv)); bufs += [q, kv]
    if dec_sks:
        indptr, indices, last, kv = _paged(dec_sks, H_kv, D, dt, dev)
        dec.plan(indptr, indices, last, H, H_kv, D, BLOCK_SIZE, q_data_type=dt, kv_data_type=dt)
        q = torch.randn(len(dec_sks), H, D, device=dev, dtype=dt)
        fns.append(lambda q=q, kv=kv: dec.run(q, kv)); bufs += [q, kv]

    def step():
        for f in fns:
            f()

    t = measure(step, device=dev, iters=iters, warmup=warmup)
    del fns, bufs
    torch.cuda.empty_cache()
    return t.median_ms
