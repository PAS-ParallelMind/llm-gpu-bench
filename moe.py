"""MoE sweep — token-routed grouped GEMM (vLLM): Triton bf16 + best-of mxfp4 w4a16.

A fused-MoE layer routes each of M tokens to top_k of E experts, then runs two grouped
GEMMs per expert: gate+up (`H→2I`), SiLU, down (`I→H`). We model it as those two grouped
GEMMs under *uniform* routing (the analytic roofline) but measure the real kernel, so fusion
+ routing land in the efficiency factor -- same roofline / measured split as gemm/attn.

Two quant schemes, differing only in the **weight byte model** (FLOPs are identical -- all
the 4-bit kernels dequant to bf16 and run the same bf16 tensor cores):
  * bf16  : fused_experts (Triton);  weights 2 bytes/elem.
  * mxfp4 : w4a16 (4-bit weights, ~0.53 bytes/elem incl. E8M0 scale/32; bf16 activations).
            Best-of-backends like attention: per shape we try every vLLM mxfp4 MoE kernel
            that runs on this GPU -- Marlin (fused_marlin_moe) and the Triton kernel
            (triton_kernel_moe_forward; OpenAI's triton_kernels package, currently adopted by
            gpt-oss) -- skip the unsupported, keep the FASTEST, and record which won. On Ada
            (RTX 4090) only Marlin runs; the Triton kernel needs SM90-100 + the triton_kernels
            package, so it is picked up on Hopper/Blackwell.

  NOT every vLLM mxfp4 MoE backend is covered (see README): the FlashInfer TRTLLM/CUTLASS
  modular kernels (SM90/SM100) and DeepGEMM have no functional entry (class-based modular
  kernels), and the activation-quantized w4a8 variants (MXFP8 / FP8) are a *different
  precision* with a different byte model -- both out of scope for this w4a16 best-of.

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
MXFP4_GROUP = 32                            # E8M0 scale group size

# mxfp4 w4a16 MoE kernels tried per shape (best wins; unsupported skipped). Marlin runs on any
# CUDA GPU; the Triton kernel needs SM90-100 + the triton_kernels package, so on
# Ada (RTX 4090) only Marlin runs. (Other vLLM mxfp4 backends -- FlashInfer TRTLLM/CUTLASS,
# DeepGEMM, and the w4a8 MXFP8/FP8 variants -- are out of scope; see the module docstring.)
MXFP4_BACKENDS = ["marlin", "triton"]
_moe_disabled: set = set()                  # backends unsupported on this GPU/install -> not retried


def make_mxfp4_weight(N: int, K: int, device: torch.device, group_size: int = MXFP4_GROUP):
    """Random Marlin-format mxfp4 weight + E8M0 scales (no bf16 reference) for one [N, K]
    matrix. Mirrors vLLM's rand_marlin_weight_mxfp4_like (random fp4 -> repack ->
    permute/process-scales); we only time the kernel, so the dequantized reference is dropped.
    Used per-expert to build the Marlin MoE weights."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
    )
    scales = torch.randint(110, 120, (N, K // group_size), dtype=torch.uint8,
                           device=device).view(torch.float8_e8m0fnu)
    fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    fp4 = fp4.view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = ops.gptq_marlin_repack(b_q_weight=fp4, perm=perm, size_k=K, size_n=N,
                                     num_bits=4, is_a_8bit=False)
    s = marlin_permute_scales(s=scales.T.to(torch.bfloat16), size_k=K, size_n=N,
                              group_size=group_size, is_a_8bit=False)
    s = mxfp4_marlin_process_scales(s, input_dtype=None)
    return qweight, s.to(torch.float8_e8m0fnu)


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
    def stack(N, K):
        qs, ss = zip(*[make_mxfp4_weight(N, K, dev) for _ in range(E)])
        return torch.stack(qs).contiguous(), torch.stack(ss).contiguous()

    return stack(2 * I, H), stack(H, I)     # (w1q, w1s), (w2q, w2s)


def _bf16_call(M, E, top_k, H, I, dt, dev):
    """bf16 grouped GEMM via Triton fused_experts."""
    x = torch.randn(M, H, device=dev, dtype=dt)
    tw, tid = _uniform_routing(M, E, top_k, dev)
    w1 = torch.randn(E, 2 * I, H, device=dev, dtype=dt) * 0.02   # [E, 2I, H] gate+up
    w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02       # [E, H, I]  down
    return (lambda: fused_experts(x, w1, w2, tw, tid)), (x, w1, w2, tw, tid)


def _marlin_call(M, E, top_k, H, I, dt, dev):
    """w4a16 Marlin via fused_marlin_moe (per-expert Marlin weights + bf16 activations)."""
    try:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    except ModuleNotFoundError:   # vLLM <= 0.16: flat layout, no experts/ subpackage
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types
    x = torch.randn(M, H, device=dev, dtype=dt)
    tw, tid = _uniform_routing(M, E, top_k, dev)
    (w1q, w1s), (w2q, w2s) = _marlin_moe_weights(E, H, I, dev)
    qid = scalar_types.float4_e2m1f.id              # mxfp4 weights, bf16 activations
    fn = lambda: fused_marlin_moe(x, w1q, w2q, None, None, w1s, w2s, tw, tid, qid)
    return fn, (x, w1q, w1s, w2q, w2s, tw, tid)


def _triton_moe_weights(E, H, I, dev):
    """Per-expert Triton mxfp4 weights (swizzled) + PrecisionConfigs, from random
    checkpoint-layout fp4 weights + E8M0 scales (uint8). Mirrors vLLM's _setup_kernel TRITON
    branch exactly (same shapes + the same _swizzle_mxfp4), so it runs wherever vLLM's does."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    def build(N, K):                                # logical fp4 [E, N, K] -> packed + E8M0 scale
        w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        s = torch.randint(118, 128, (E, N, K // MXFP4_GROUP), dtype=torch.uint8, device=dev)
        wt, flex, st = _swizzle_mxfp4(w, s, num_warps=8)   # 8 = vLLM's non-batched default;
        return wt, PrecisionConfig(weight_scale=st, flex_ctx=FlexCtx(rhs_data=flex))  # required positional on 0.16

    w13, w13_pc = build(2 * I, H)                   # gate+up: out 2I, in H
    w2, w2_pc = build(H, I)                         # down:    out H,  in I
    return w13, w2, w13_pc, w2_pc


def _triton_call(M, E, top_k, H, I, dt, dev):
    """w4a16 Triton kernel via triton_kernel_moe_forward (routes from logits). The quant config
    comes straight from config.mxfp4_w4a16_moe_quant_config (present in both vLLM 0.16 and
    0.22+, and what the oracle's TRITON branch forwards to); the swiglu clamp params aren't
    passed -- the functional kernel uses its own defaults -- so the call is version-portable."""
    try:                          # vLLM 0.18+: experts/ subpackage
        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
            triton_kernel_moe_forward,
        )
    except ModuleNotFoundError:   # vLLM <= 0.16: flat layout, no experts/ subpackage
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            triton_kernel_moe_forward,
        )
    from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
    x = torch.randn(M, H, device=dev, dtype=dt)
    gating = torch.randn(M, E, device=dev, dtype=dt)            # router logits (routed internally)
    w13, w2, w13_pc, w2_pc = _triton_moe_weights(E, H, I, dev)
    qcfg = mxfp4_w4a16_moe_quant_config(w1_scale=w13_pc, w2_scale=w2_pc)
    fn = lambda: triton_kernel_moe_forward(x, w13, w2, gating, topk=top_k, renormalize=True,
                                           quant_config=qcfg, global_num_experts=E)
    return fn, (x, gating, w13, w2, w13_pc, w2_pc)


_MOE_BUILDERS = {"fused_experts": _bf16_call, "marlin": _marlin_call, "triton": _triton_call}


def _backend_supported(backend: str, dev: torch.device) -> tuple[bool, str]:
    """Cheap pre-flight: is this backend even runnable on this GPU/install? (bf16 + Marlin run
    on any CUDA GPU; the Triton kernel needs SM90-100 + the triton_kernels package.)"""
    if backend == "triton":
        cap = torch.cuda.get_device_capability(dev)
        if not ((9, 0) <= cap < (11, 0)):           # vLLM gates the triton_kernels to SM90..SM100
            return False, f"Triton needs SM90-100, have sm_{cap[0]}{cap[1]}"
        import importlib.util
        if importlib.util.find_spec("triton_kernels") is None:
            return False, "triton_kernels not installed"
    return True, ""


def _best_moe_call(M, E, top_k, H, I, dt, dev, *, quant, iters, warmup):
    """Best kernel for one MoE point: bf16 has the lone fused_experts; mxfp4 tries each of
    MXFP4_BACKENDS, skips the unsupported (caching architectural failures in _moe_disabled),
    and keeps the fastest. Returns (median_ms, backend, fn, bufs) with the winner's call live."""
    cands = (["fused_experts"] if quant == "bf16"
             else [b for b in MXFP4_BACKENDS if b not in _moe_disabled])
    best = None
    for backend in cands:
        ok, why = _backend_supported(backend, dev)
        if not ok:                                  # architectural: never runs here -> disable once
            _moe_disabled.add(backend)
            print(f"  [skip] moe {backend}: {why}")
            continue
        try:
            fn, bufs = _MOE_BUILDERS[backend](M, E, top_k, H, I, dt, dev)
            ms = measure(fn, device=dev, iters=iters, warmup=warmup).median_ms
        except Exception as e:
            torch.cuda.empty_cache()
            if isinstance(e, ImportError):          # package missing/broke -> never retry
                _moe_disabled.add(backend)
            # else per-shape reject (alignment) or OOM -> skip this point only, keep the backend
            print(f"  [skip pt] moe {backend} M={M} E={E} H={H} I={I}: {str(e).splitlines()[0][:50]}")
            continue
        if best is None or ms < best[0]:
            prev, best = best, (ms, backend, fn, bufs)
            del prev                                 # drop previous winner's fn/bufs for GC
        else:
            del fn, bufs
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError(f"no MoE backend ran for M={M} E={E} H={H} I={I}")
    return best


def measure_moe_ms(M, E, top_k, H, I, *, quant="bf16", dtype="bf16",
                   device: int | torch.device = 0, iters=30, warmup=10) -> float:
    """Best-of-backends median ms for one MoE point (bf16: the lone fused_experts kernel)."""
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    ms, _, fn, bufs = _best_moe_call(M, E, top_k, H, I, _DTYPES[dtype], dev,
                                     quant=quant, iters=iters, warmup=warmup)
    del fn, bufs
    torch.cuda.empty_cache()
    return ms


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
    backend: str = ""       # winning kernel ("fused_experts" / "marlin" / "triton")
    tflops: float = 0.0     # achieved compute throughput
    gbps: float = 0.0       # achieved memory throughput
    efficiency: float = 0.0

    def result(self) -> dict:
        return {"shape": {"M": self.M, "E": self.E, "top_k": self.top_k, "H": self.H,
                          "I": self.I, "backend": self.backend},
                "latency_ms": self.median_ms, "tflops": self.tflops, "gbps": self.gbps,
                "efficiency": self.efficiency}


def run_moe_sweep(Ms, Es, Hs, Is, top_k, *, c_peak, b_peak, quant="bf16", dtype="bf16",
                  device: int | torch.device = 0, iters=30, warmup=10):
    """Sweep the MoE kernel over (M, E, H, I) at one top_k; efficiency = roofline / measured.
    quant selects the scheme (bf16 fused_experts / mxfp4 fused_marlin_moe) and its byte model."""
    C, B = c_peak * 1e12, b_peak * 1e9
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    dt = _DTYPES[dtype]
    work = [(M, E, H, I) for E in Es for H in Hs for I in Is for M in Ms]
    recs: list[MoERecord] = []
    pbar = progress(len(work), f"moe[{quant}]")
    for M, E, H, I in work:
        try:
            ms, backend, fn, bufs = _best_moe_call(M, E, top_k, H, I, dt, dev,
                                                   quant=quant, iters=iters, warmup=warmup)
        except RuntimeError as e:                     # no backend ran (OOM / all shapes rejected)
            torch.cuda.empty_cache()
            print(f"  [skip pt] moe[{quant}] M={M} E={E} H={H} I={I}: {str(e).splitlines()[0][:50]}")
            pbar.update(1)
            continue
        del fn, bufs
        torch.cuda.empty_cache()
        flops = moe_flops(M, top_k, H, I)
        nbytes = moe_bytes(M, E, top_k, H, I, quant)
        tc, tm = flops / C, nbytes / B
        sec = ms * 1e-3
        recs.append(MoERecord(M=M, E=E, top_k=top_k, H=H, I=I, quant=quant, median_ms=ms,
                    regime="C" if tc > tm else "M", backend=backend,
                    tflops=flops / sec / 1e12 if sec > 0 else 0.0,
                    gbps=nbytes / sec / 1e9 if sec > 0 else 0.0,
                    efficiency=(max(tc, tm) / sec) if sec > 0 else 0.0))
        pbar.set_postfix_str(f"M={M} E={E} H={H} I={I} [{backend}]")
        pbar.update(1)
    pbar.close()
    return recs


def run_full_moe_sweep(*, c_peak, b_peak, quant="bf16", dtype="bf16", device=0, iters=30, warmup=10):
    return run_moe_sweep(MOE_M_GRID, MOE_E_GRID, MOE_H_GRID, MOE_I_GRID, MOE_TOPK,
                         c_peak=c_peak, b_peak=b_peak, quant=quant, dtype=dtype,
                         device=device, iters=iters, warmup=warmup)
