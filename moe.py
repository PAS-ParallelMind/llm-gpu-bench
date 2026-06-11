"""MoE sweep — token-routed grouped GEMM (vLLM fused_experts, Triton, bf16).

A fused-MoE layer routes each of M tokens to top_k of E experts, then runs two grouped
GEMMs per expert: gate+up (H -> 2I), SiLU, down (I -> H). We model it as those two grouped
GEMMs under *uniform* routing (the analytic roofline) but measure the real fused_experts
kernel, so fusion + routing land in the efficiency factor -- same roofline / measured split
as gemm/attn. (Triton path only; vLLM picks its own Triton config, tuned or default, which
is exactly what serving runs -- tuned-vs-untuned cancels in roofline / efficiency.)

  routed tokens  T    = M * top_k
  active experts E_act = min(E, T)           (only top_k experts fire at small M)
  FLOPs = 6 * T * H * I            (gate+up 4*T*H*I  +  down 2*T*H*I)
  bytes = E_act * 3 * H * I * elem (active w1+w2)  +  2 * M * H * elem (in/out acts)
  t = max(FLOPs/C_peak, bytes/B_peak) / efficiency(T, E, H, I)

Efficiency is keyed on T = M*top_k (the grouped-GEMM work), so a model's own top_k folds in
and the grid is swept at one benchmark top_k. Needs torch + vLLM (the fused_experts kernel).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from timing import measure, progress
from vllm.model_executor.layers.fused_moe import fused_experts

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Variables: M tokens, E experts, top_k experts/token, H hidden (GEMM K), I intermediate.
# Model-agnostic grid: routed tokens via M (decode->prefill) x experts E x hidden H x intermediate I.
MOE_M_GRID = [1, 8, 64, 512, 4096]          # T = M*top_k spans 8 .. 32768 at top_k=8
MOE_E_GRID = [8, 32, 128]
MOE_H_GRID = [2048, 4096, 8192]             # brackets real hidden (2048, 2880)
MOE_I_GRID = [512, 1024, 2048, 4096]        # brackets real intermediate (768, 2880)
MOE_TOPK = 8                                # benchmark top_k; efficiency is keyed on T=M*top_k


def moe_flops(M: int, top_k: int, H: int, I: int) -> int:
    return 6 * M * top_k * H * I            # gate+up (4*T*H*I) + down (2*T*H*I), T=M*top_k


def moe_bytes(M: int, E: int, top_k: int, H: int, I: int, elem: int = 2) -> int:
    E_act = min(E, M * top_k)               # only the fired experts' weights are read
    return E_act * 3 * H * I * elem + 2 * M * H * elem


def _uniform_routing(M, E, top_k, dev):
    """Balanced round-robin assignment: each expert gets ~M*top_k/E tokens (the uniform
    model), so the measured kernel matches the uniform-routing roofline."""
    ids = (torch.arange(M * top_k, device=dev) % E).to(torch.int32).view(M, top_k)
    weights = torch.full((M, top_k), 1.0 / top_k, device=dev, dtype=torch.float32)
    return weights, ids


def _moe_call(M, E, top_k, H, I, dt, dev):
    x = torch.randn(M, H, device=dev, dtype=dt)
    w1 = torch.randn(E, 2 * I, H, device=dev, dtype=dt) * 0.02    # [E, 2I, H] gate+up
    w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02        # [E, H, I]  down
    tw, tid = _uniform_routing(M, E, top_k, dev)
    fn = lambda: fused_experts(x, w1, w2, tw, tid)
    return fn, (x, w1, w2, tw, tid)


def measure_moe_ms(M, E, top_k, H, I, *, dtype="bf16",
                   device: int | torch.device = 0, iters=30, warmup=10) -> float:
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    fn, bufs = _moe_call(M, E, top_k, H, I, _DTYPES[dtype], dev)
    t = measure(fn, device=dev, iters=iters, warmup=warmup)
    del fn, bufs
    torch.cuda.empty_cache()
    return t.median_ms


@dataclass
class MoERecord:
    M: int
    E: int
    top_k: int
    H: int
    I: int
    dtype: str
    median_ms: float
    regime: str             # "C" compute-bound, "M" memory-bound (weight-read)
    efficiency: float = 0.0


def run_moe_sweep(Ms, Es, Hs, Is, top_k, *, c_peak, b_peak, dtype="bf16",
                  device: int | torch.device = 0, iters=30, warmup=10):
    """Sweep fused_experts over (M, E, H, I) at one top_k; efficiency = roofline / measured."""
    dt = _DTYPES[dtype]
    elem = dt.itemsize
    C, B = c_peak * 1e12, b_peak * 1e9
    work = [(M, E, H, I) for E in Es for H in Hs for I in Is for M in Ms]
    recs: list[MoERecord] = []
    pbar = progress(len(work), "moe")
    for M, E, H, I in work:
        try:
            ms = measure_moe_ms(M, E, top_k, H, I, dtype=dtype, device=device,
                                iters=iters, warmup=warmup)
        except Exception as e:                        # OOM (huge E*H*I weights) on smaller GPUs
            torch.cuda.empty_cache()
            print(f"  [skip pt] moe M={M} E={E} H={H} I={I}: {str(e).splitlines()[0][:50]}")
            pbar.update(1)
            continue
        tc = moe_flops(M, top_k, H, I) / C
        tm = moe_bytes(M, E, top_k, H, I, elem) / B
        sec = ms * 1e-3
        recs.append(MoERecord(M=M, E=E, top_k=top_k, H=H, I=I, dtype=dtype, median_ms=ms,
                    regime="C" if tc > tm else "M",
                    efficiency=(max(tc, tm) / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"M={M} E={E} H={H} I={I}")
        pbar.update(1)
    pbar.close()
    return recs


def run_full_moe_sweep(*, c_peak, b_peak, dtype="bf16", device=0, iters=30, warmup=10):
    return run_moe_sweep(MOE_M_GRID, MOE_E_GRID, MOE_H_GRID, MOE_I_GRID, MOE_TOPK,
                         c_peak=c_peak, b_peak=b_peak, dtype=dtype, device=device,
                         iters=iters, warmup=warmup)
