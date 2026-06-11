"""MoE sweep — token-routed grouped GEMM (vLLM): Triton bf16 + Marlin mxfp4 w4a16.

A fused-MoE layer routes each of M tokens to top_k of E experts, then runs two grouped
GEMMs per expert: gate+up (`H→2I`), SiLU, down (`I→H`). We model it as those two grouped
GEMMs under *uniform* routing (the analytic roofline) but measure the real kernel, so fusion
+ routing land in the efficiency factor -- same roofline / measured split as gemm/attn.

Two quant schemes, differing only in the **weight byte model** (FLOPs are identical -- Marlin
dequants 4-bit weights to bf16 and runs the same bf16 tensor cores on Ada):
  * bf16  : fused_experts (Triton);          weights 2 bytes/elem.
  * mxfp4 : fused_marlin_moe (w4a16 Marlin);  weights ~0.53 bytes/elem (4-bit + E8M0 scale/32),
            bf16 activations. Per-expert Marlin weights built via marlin.make_mxfp4_weight.

  routed tokens  T    = M * top_k ;  active experts E_act = min(E, T)
  FLOPs = 6 * T * H * I            (gate+up 4*T*H*I + down 2*T*H*I)
  bytes = E_act * 3 * H * I * w_bytes  +  2 * M * H * elem
  t = max(FLOPs/C_peak, bytes/B_peak) / efficiency(T, E, H, I)

Efficiency is keyed on T = M*top_k, so a model's own top_k folds in. Needs torch + vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from timing import measure, progress
from vllm.model_executor.layers.fused_moe import fused_experts

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Variables: M tokens, E experts, top_k experts/token, H hidden (GEMM K), I intermediate.
# Model-agnostic grid: routed tokens via M (decode->prefill) x experts E x hidden H x intermediate I.
MOE_M_GRID = [1, 4, 16, 64, 256, 1024, 4096]   # x4 steps: T = M*top_k spans 8 .. 32768 at top_k=8
MOE_E_GRID = [8, 32, 128]
MOE_H_GRID = [2048, 4096, 8192]             # brackets real hidden (2048, 2880)
MOE_I_GRID = [512, 1024, 2048, 4096]        # brackets real intermediate (768, 2880)
MOE_TOPK = 8                                # benchmark top_k; efficiency is keyed on T=M*top_k

# Weight bytes/elem per scheme (mxfp4: 4-bit weight + 1-byte E8M0 scale per 32 elems = 0.53125).
WEIGHT_BYTES = {"bf16": 2.0, "mxfp4": 0.53125}


def moe_flops(M: int, top_k: int, H: int, I: int) -> int:
    return 6 * M * top_k * H * I            # gate+up (4*T*H*I) + down (2*T*H*I), T=M*top_k


def moe_bytes(M: int, E: int, top_k: int, H: int, I: int, quant: str = "bf16", elem: int = 2) -> float:
    E_act = min(E, M * top_k)               # only the fired experts' weights are read
    return E_act * 3 * H * I * WEIGHT_BYTES[quant] + 2 * M * H * elem


def _uniform_routing(M, E, top_k, dev):
    """Balanced round-robin assignment: each expert gets ~M*top_k/E tokens (the uniform
    model), so the measured kernel matches the uniform-routing roofline."""
    ids = (torch.arange(M * top_k, device=dev) % E).to(torch.int32).view(M, top_k)
    weights = torch.full((M, top_k), 1.0 / top_k, device=dev, dtype=torch.float32)
    return weights, ids


def _marlin_moe_weights(E, H, I, dev):
    """Per-expert mxfp4 Marlin weights, stacked: w1 [E, H//16, 4I], w2 [E, I//16, 2H]."""
    from marlin import make_mxfp4_weight

    def stack(N, K):
        qs, ss = zip(*[make_mxfp4_weight(N, K, dev) for _ in range(E)])
        return torch.stack(qs).contiguous(), torch.stack(ss).contiguous()

    return stack(2 * I, H), stack(H, I)     # (w1q, w1s), (w2q, w2s)


def _moe_call(M, E, top_k, H, I, dt, dev, quant="bf16"):
    x = torch.randn(M, H, device=dev, dtype=dt)
    tw, tid = _uniform_routing(M, E, top_k, dev)
    if quant == "mxfp4":
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.scalar_type import scalar_types
        (w1q, w1s), (w2q, w2s) = _marlin_moe_weights(E, H, I, dev)
        qid = scalar_types.float4_e2m1f.id          # mxfp4 weights, bf16 activations
        fn = lambda: fused_marlin_moe(x, w1q, w2q, None, None, w1s, w2s, tw, tid, qid)
        return fn, (x, w1q, w1s, w2q, w2s, tw, tid)
    w1 = torch.randn(E, 2 * I, H, device=dev, dtype=dt) * 0.02   # [E, 2I, H] gate+up
    w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02       # [E, H, I]  down
    fn = lambda: fused_experts(x, w1, w2, tw, tid)
    return fn, (x, w1, w2, tw, tid)


def measure_moe_ms(M, E, top_k, H, I, *, quant="bf16", dtype="bf16",
                   device: int | torch.device = 0, iters=30, warmup=10) -> float:
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    fn, bufs = _moe_call(M, E, top_k, H, I, _DTYPES[dtype], dev, quant)
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
    quant: str              # "bf16" or "mxfp4" (weight scheme)
    median_ms: float
    regime: str             # "C" compute-bound, "M" memory-bound (weight-read)
    efficiency: float = 0.0


def run_moe_sweep(Ms, Es, Hs, Is, top_k, *, c_peak, b_peak, quant="bf16", dtype="bf16",
                  device: int | torch.device = 0, iters=30, warmup=10):
    """Sweep the MoE kernel over (M, E, H, I) at one top_k; efficiency = roofline / measured.
    quant selects the scheme (bf16 fused_experts / mxfp4 fused_marlin_moe) and its byte model."""
    C, B = c_peak * 1e12, b_peak * 1e9
    work = [(M, E, H, I) for E in Es for H in Hs for I in Is for M in Ms]
    recs: list[MoERecord] = []
    pbar = progress(len(work), f"moe[{quant}]")
    for M, E, H, I in work:
        try:
            ms = measure_moe_ms(M, E, top_k, H, I, quant=quant, dtype=dtype, device=device,
                                iters=iters, warmup=warmup)
        except Exception as e:                        # OOM (huge weights) / Marlin shape reject
            torch.cuda.empty_cache()
            print(f"  [skip pt] moe[{quant}] M={M} E={E} H={H} I={I}: {str(e).splitlines()[0][:50]}")
            pbar.update(1)
            continue
        tc = moe_flops(M, top_k, H, I) / C
        tm = moe_bytes(M, E, top_k, H, I, quant) / B
        sec = ms * 1e-3
        recs.append(MoERecord(M=M, E=E, top_k=top_k, H=H, I=I, quant=quant, median_ms=ms,
                    regime="C" if tc > tm else "M",
                    efficiency=(max(tc, tm) / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"M={M} E={E} H={H} I={I}")
        pbar.update(1)
    pbar.close()
    return recs


def run_full_moe_sweep(*, c_peak, b_peak, quant="bf16", dtype="bf16", device=0, iters=30, warmup=10):
    return run_moe_sweep(MOE_M_GRID, MOE_E_GRID, MOE_H_GRID, MOE_I_GRID, MOE_TOPK,
                         c_peak=c_peak, b_peak=b_peak, quant=quant, dtype=dtype,
                         device=device, iters=iters, warmup=warmup)
