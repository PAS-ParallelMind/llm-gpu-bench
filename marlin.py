"""mxfp4 w4a16 GEMM sweep via vLLM Marlin — weight-only 4-bit, bf16 activation.

On Ada (no FP4 tensor cores) Marlin dequantizes the 4-bit weight to bf16 and runs
a bf16 matmul. So the FLOPs match bf16 but the weight read is ~3.8x lighter:

    FLOPs = 2*M*N*K                          (same matmul as bf16)
    bytes = 0.53125 * N*K                     (4-bit weight + 1-byte E8M0 scale/32)
          + 2 * (M*K + M*N)                   (bf16 activation read + output write)

That lighter weight read moves the roofline ridge to a much smaller M, so decode
is far more memory-efficient than bf16 — the whole point of w4a16. Compute is still
bf16 tensor cores but with dequant overhead, so achievable C_peak < cuBLAS.

Reuses gemm.py's GemmRecord; dtype label is "mxfp4". The kernel call is vLLM's own
tested path (rand_marlin_weight_mxfp4_like + apply_fp4_marlin_linear); we add only
our timing harness (so it is apples-to-apples with the bf16 sweep) and the
weight-only byte model (marlin_bytes), which must match the predictor's roofline.
"""
from __future__ import annotations

import torch

from gemm import GemmRecord
from timing import measure
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    mxfp4_marlin_process_scales,
)

MXFP4_GROUP = 32
# weight bytes/elem: 4 bits (0.5) + one E8M0 (1 byte) scale per 32-elem group.
W_BYTES_PER_ELEM = 0.5 + 1.0 / MXFP4_GROUP          # = 0.53125
A_BYTES = 2                                          # bf16 activation / output

# byte model as {weight, activation, output} bytes/elem — shared with the predictor.
BYTES_MODEL = {"w": W_BYTES_PER_ELEM, "a": A_BYTES, "o": A_BYTES}


def marlin_bytes(M: int, K: int, N: int) -> float:
    return W_BYTES_PER_ELEM * N * K + A_BYTES * (M * K + M * N)


def make_mxfp4_weight(N: int, K: int, device: torch.device,
                      group_size: int = MXFP4_GROUP):
    """Random Marlin-format mxfp4 weight + E8M0 scales, with NO bf16 reference.

    Mirrors vLLM's ``rand_marlin_weight_mxfp4_like`` (same random fp4 -> repack ->
    permute/process-scales path) but drops the dequantized reference tensor — we
    only time the kernel, and that reference costs several GB of bf16 transients at
    large N. Correctness of this exact path is verified against vLLM's reference
    separately (see README), so timing-only weights are safe here.
    """
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


def run_marlin_sweep(
    shapes: dict[str, tuple[int, int]],
    Ms: list[int],
    *,
    device: int | torch.device = 0,
    iters: int = 100,
    warmup: int = 25,
) -> list[GemmRecord]:
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    ws = marlin_make_workspace_new(dev)
    recs: list[GemmRecord] = []
    for name, (K, N) in shapes.items():
        qw, scales = make_mxfp4_weight(N, K, dev)
        warned = False
        for M in Ms:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            try:
                t = measure(
                    lambda: apply_fp4_marlin_linear(x, qw, scales, None, ws,
                                                    size_n=N, size_k=K),
                    device=dev, iters=iters, warmup=warmup,
                )
            except RuntimeError as e:
                # Marlin rejects shapes where K and N are both only 64-aligned
                # (no valid tile config). Real models pad to marlin-friendly dims;
                # we skip such shapes here rather than crash the sweep.
                if not warned:
                    print(f"  [skip] Marlin unsupported: {name} K={K} N={N} "
                          f"({str(e).splitlines()[0][:48]})")
                    warned = True
                del x
                continue
            flops = 2 * M * N * K
            nbytes = marlin_bytes(M, K, N)
            sec = t.median_ms * 1e-3
            recs.append(GemmRecord(
                shape=name, M=M, K=K, N=N, dtype="mxfp4",
                median_ms=t.median_ms, min_ms=t.min_ms,
                tflops=flops / sec / 1e12,
                gbps=nbytes / sec / 1e9,
                ai=flops / nbytes,
            ))
            del x
        del qw, scales
        torch.cuda.empty_cache()
    return recs


def marlin_roofline_residual(
    recs: list[GemmRecord], c_peak_tflops: float, b_peak_gbps: float
) -> None:
    """Fill predicted_ms / residual using the w4a16 byte model (in place)."""
    c = c_peak_tflops * 1e12
    b = b_peak_gbps * 1e9
    for r in recs:
        flops = 2 * r.M * r.N * r.K
        nbytes = marlin_bytes(r.M, r.K, r.N)
        pred_s = max(flops / c, nbytes / b)
        r.predicted_ms = pred_s * 1e3
        r.residual = (r.median_ms / r.predicted_ms) if r.predicted_ms > 0 else 0.0
