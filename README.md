# llm-gpu-bench

A benchmark suite that characterises a GPU's **achievable** compute/memory
throughput and turns it into **LLM inference latency predictions**. It measures the
*efficiency factor* of real kernel shapes — the gap between the GPU's theoretical
roofline and what it actually reaches — so the model can predict latency for shapes
it never measured.

## Idea

Latency of a GEMM (and, later, attention / MoE) is split into two measured pieces:

    t = roofline(C_peak, B_peak) / efficiency(shape)

1. **Ceilings** — the GPU's **theoretical** peaks `C_peak[dtype]` (tensor-core
   TFLOP/s) and `B_peak` (memory GB/s), passed via `--c-peak` / `--b-peak`, setting
   the floor `t_roof = max(FLOPs / C_peak, Bytes / B_peak)`. Their absolute scale
   *cancels* in the predictor (it only normalizes the efficiency factor), so any
   consistent value works.
2. **Efficiency factor** `efficiency = t_roof / t_measured ∈ (0,1]` — how close the
   real kernel gets to the floor. This is what varies with shape, so it is what we
   sample and interpolate.

The challenge: (model, op, batch) shapes are unbounded — we can't measure them all.
So we measure one **model-agnostic grid** per GPU and predict any shape by
interpolating its efficiency factor to the nearest measured neighbourhood.

## What the sweep found (bf16 GEMM, RTX 4090)

For `y = x @ Wᵀ`, sweeping M (tokens) × K (contraction) × N (output), efficiency is
low-dimensional and lives on physics axes, not raw (K,N):

- **Decode (small M):** efficiency tracks the **weight footprint N·K** — a small
  weight can't saturate HBM (eff ≈ 0.2), a big one does (eff ≈ 1.0).
- **Prefill (large M):** efficiency tracks **K** (mainloop length): 0.88 (K=512) →
  0.97 (K≥4096).
- **Transition (M ≈ 64–256):** a ~10% efficiency sag where neither ceiling is
  saturated. Jagged (wave quantization) — the irreducible hard region.

Two approaches ruled out **by measurement**, not opinion:

- An **analytic tile/wave-quant correction** (`ceil(M/Tm)·ceil(N/Tn)` vs SM count)
  does *not* help — split-K / stream-K kernels redistribute work and defeat it.
  Measure the transition, don't model it.
- A **sparse grid + footprint↔K distance metric** worked but was fragile on
  narrow-N shapes. A **dense grid + trilinear interpolation** is simpler *and* more
  accurate, so that's what shipped.

## The predictor

- **Grid** (`gemm.py`) — a dense octave grid, K ∈ [128 … 16384], N ∈ [128 …
  131072] (88 pairs) × an M-sweep [1 … 4096]. Model-agnostic: keyed on no model's
  shapes. Real projections fall inside the hull (K∈[768,8192], N∈[1536,~201k]); N
  reaches 131072 so lmhead is bracketed, not extrapolated.
- **Predict** (`predict.py`) — `t = roofline / efficiency`, efficiency by trilinear
  interpolation in (log M, log K, log N), clamped at the edges. Pure stdlib: no GPU
  needed to predict.

### Accuracy

Validated on 10 real projections (gpt-oss-20b, Qwen3-Coder-30B) the grid never
saw, across M = 1…4096 (`validate_predict.py`):

    latency error:  median 2.7%   mean 4.8%

- Most shapes (lmhead, qkv, o, most MoE) are **1–6%** across all M.
- **Known floor — power-of-2 cliffs.** Grid points sit on tile-aligned ("lucky")
  sizes; torch's GEMM backend has kernel-selection dips at non-pow2 dims *between*
  them. e.g. at
  K=2048, M=512: N=1024→0.54, **N=1536→0.37**, N=2048→0.64 — a V-notch the grid
  interpolates straight over. A non-pow2 dim landing in a transition-band
  (M≈256–1024) dip can carry **~50% error** (~1 in 6 non-pow2 shapes). This is
  sub-octave and intrinsic to the kernel library — not fixable by grid density; documented as
  the accuracy floor.

## mxfp4 w4a16 (vLLM Marlin)

Same framework, second scheme: 4-bit weights (mxfp4 — 32-elem groups + E8M0 scale),
bf16 activations. On Ada (no FP4 cores) Marlin dequantizes W→bf16 and runs a bf16
matmul, so only the **byte model** changes — the weight read is ~3.8× lighter:

    bytes = 0.53125·N·K  (4-bit weight + 1-byte scale/32)  +  2·(M·K + M·N)

The predictor is unchanged; each scheme just carries its `bytes_model` in the
results JSON (`predict.py` reads it; bf16 defaults to 2/2/2). The theoretical
ceiling is **shared with bf16** — mxfp4 dequants to the same bf16 tensor cores and
the memory ceiling is the same GDDR6X — so all the kernel-specific behaviour lives
in the efficiency factor.

What the sweep found (RTX 4090), as *achieved* throughput against that shared
165 TFLOP/s · 1008 GB/s ceiling:

- **Compute ≈ 171 TFLOP/s achieved** — matches bf16; dequant is fully hidden at large M.
- **Memory ≈ 904 GB/s achieved** weight-read — the biggest weights approach the 1008
  ceiling, but moderate ones run ~600 GB/s (dequant-limited); the efficiency factor
  absorbs the shape dependence.
- **~3× decode speedup vs bf16** (M ≲ 16), vanishing by M ≈ 1024 where both are
  compute-bound on the same tensor cores. The lighter weight read moves the roofline
  ridge down to small M — the whole point of w4a16.

Accuracy on the 9 runnable real projections (`validate_predict.py --bench gemm_mxfp4`):

    latency error:  median 5.9%   mean 7.5%   p90 15%

- **Marlin shape constraint:** it rejects dims where K and N are *both* only
  64-aligned (e.g. gpt-oss moe_dn 2880×2880) — no valid tile config. The power-of-2
  grid is unaffected; production pads such weights to marlin-friendly dims, and the
  sweep skips unrunnable shapes rather than crash.

## decode attention (vLLM FlashAttention, paged KV)

A third op, and the simplest. Decode attention (1 query token per request, reading
the whole KV cache) is *always* memory-bound — arithmetic intensity = 2·(H/H_kv)/elem
(the GQA ratio), far below the ridge. Measurement shows efficiency depends only on
the **total KV bytes** streamed — not on head config (H, H_kv, D), request count, or
how the bytes split across requests' context lengths (a skewed mixed batch matches a
uniform one with the same total KV bytes). So a single **1-D curve `eff = f(KV bytes)`**
predicts decode attention for any model and any continuous batch:

    t = (KV_bytes / B_peak) / f(KV_bytes),   KV_bytes = 2·elem·Σ_i ⌈L_i/16⌉·16·H_kv·D

vLLM's FlashAttention backend runs prefill+decode through one unified
`flash_attn_varlen_func` (paged KV + cu_seqlens), so decode is the S_q=1 slice. The
curve runs **0.03 (1 MB KV) → 0.92 (2 GB KV)** — paged-KV reads saturate HBM more
slowly than GEMM's contiguous weight read, so it's its own curve.

Accuracy on real head configs (gpt-oss 64/8/64, Qwen 32/4/128) × (batch, context),
`validate_predict.py --bench attn_bf16`:

    latency error:  median 1.7%   mean 3.7%   (roofline-only baseline: median 31%)

- **Verified range.** Stress-tested across head config, paged block size, request
  count, and batch composition — the 1-D collapse holds for per-request context
  **L_i ≳ 128 tokens** (covers realistic decode). In the large-batch × very-short-context
  corner (many requests each < ~128 tokens), per-request overhead pulls efficiency below
  the curve, so it over-predicts there; documented, not modeled.

## Files

    timing.py             CUDA-event timing, L2 flush, robust stats      (torch)
    gemm.py               GEMM sweep, model-agnostic grid, roofline       (torch)
    marlin.py             mxfp4 w4a16 sweep via vLLM Marlin + byte model (torch+vLLM)
    attn.py               decode flash-attn sweep + KV-byte curve        (torch+vLLM)
    run.py                run a benchmark (--bench <op>_<dtype>), dump JSON (torch)
    predict.py            latency predictor (gemm trilinear / attn curve) (stdlib)
    validate_predict.py   predicted vs measured on real workloads         (torch)
    results/              gemm_<gpu>.json, marlin_mxfp4_<gpu>.json, attn_decode_<gpu>.json

## Run

Activate the env (torch + CUDA), then:

    python run.py --bench gemm_bf16  --c-peak 165 --b-peak 1008   # bf16 GEMM grid
    python run.py --bench gemm_mxfp4 --c-peak 165 --b-peak 1008   # mxfp4 w4a16 (Marlin)
    python run.py --bench attn_bf16               --b-peak 1008   # decode flash-attn curve
    python predict.py --shape 2880 5120                           # gemm: latency vs M
    python predict.py --results results/attn_decode_*.json --head 4 128 --kv 131072  # attn
    python validate_predict.py --bench gemm_bf16                  # gemm accuracy
    python validate_predict.py --bench attn_bf16                  # decode-attn accuracy

The sweep needs torch + a CUDA GPU (mxfp4 / attn also need vLLM); prediction does not.
GEMM needs `--c-peak`/`--b-peak`; decode attention is memory-bound, so only `--b-peak`.

## Scope / next

- GPU: **RTX 4090** (Ada, SM89). No FP4 tensor cores, so mxfp4 is weight-only
  dequant→bf16 (memory win, not compute).
- Done: **bf16 GEMM** + **mxfp4 w4a16 (Marlin)** + **decode flash-attention** —
  benchmark, predictor, and validation, all via `run.py --bench <op>_<dtype>`.
- Next: **prefill attention** (compute-bound regime), then **fusedMoE** — each a new
  op with its own descriptor feeding the same roofline ÷ efficiency split.
