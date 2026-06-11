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

## attention (FlashInfer, paged KV) — hybrid, best of kernels

A third op, targeting vLLM's **FlashInfer** backend. FlashInfer dispatches a paged-KV call
to one of several underlying kernels — prefill: `fa2` / `fa3` / `cutlass` / `trtllm-gen`;
decode: `fa2` (CUDA-core or tensor-core) / `trtllm-gen` — and which exist depends on the GPU
(`fa3` is Hopper SM90, `cutlass`/`trtllm-gen` lean Hopper/Blackwell, `fa2` is the Ampere+
baseline). So the sweep **tries every candidate per shape, skips the unsupported ones, and
keeps the fastest** — the efficiency factor is the *best achievable* on the GPU, and each
grid point records which kernel won. On the **RTX 4090 (Ada) only `fa2` runs** (188/193
points; tensor-core `fa2` wins the 5 smallest decode points); on Hopper/Blackwell the faster
kernels are picked up automatically, no code change.

The roofline (FLOPs, bytes) is physics and kernel-independent; only the efficiency depends on
the kernel. The **efficiency has two physics regimes that want different scale variables**, so
we model them separately and route on `S_q` inside one `attn_latency_ms`:

**Decode (`S_q = 1`)** — 1 query token per request reading the whole KV cache, *always*
memory-bound (arithmetic intensity = 2·(H/H_kv)/elem, the GQA ratio, far below the ridge).
Efficiency depends only on the **total KV bytes** streamed — not on head config
(H, H_kv, D), request count, or how the bytes split across requests' contexts (a skewed
mixed batch matches a uniform one with the same total KV bytes). One **1-D curve
`eff = f(KV bytes)`** predicts decode for any model and any continuous batch:

    t = (KV_bytes / B_peak) / f(KV_bytes),   KV_bytes = 2·elem·Σ_i ⌈L_i/16⌉·16·H_kv·D

The curve is swept over (R requests, context L) so it spans KV bytes up to the saturation
plateau — **0.01 (small) → 0.95 (multi-GB KV)** — and the overlapping R·L points double as a
check that efficiency really collapses on total KV bytes. Paged-KV reads saturate HBM more
slowly than GEMM's contiguous weight read, so it's its own curve.

**Prefill / chunked (`S_q > 1`)** — a batched causal GEMM (per-head QKᵀ then PV) over
R·H heads. Efficiency is a **3-D surface over (S_q, S_kv, R·H) per head-dim D** (H_kv
washes out in this compute regime; R·H is the parallelism axis and collapses on the
product). The roofline is the causal trapezoid:

    FLOPs = 4·H·D·R·(S_q·S_kv − S_q(S_q−1)/2);  bytes = 2·elem·R·(S_q·H·D + S_kv·H_kv·D)
    t = max(FLOPs/C_peak, bytes/B_peak) / f(S_q, S_kv, R·H, D)

**Why hybrid, not one grid.** Decode efficiency scales with `R·S_kv·H_kv·D` (KV bytes);
prefill with `R·H` (parallelism) — *different* functions of R and H. A single shared
grid was tried and forced to drop one or the other: it pulled gpt-oss decode (H=64) to
~30% error because decode doesn't scale with R·H. Measured, not assumed — so decode keeps
its KV-byte curve and prefill keeps its (S_q, S_kv, R·H, D) grid.

Accuracy on real head configs (gpt-oss 64/8/64, Qwen 32/4/128) × 8 cases spanning
decode / full prefill / chunked prefill, `validate_predict.py --bench attn_bf16`:

    latency error:  median 2.6%   mean 6.0%   p90 12%   (roofline-only baseline: median 20%)

- **Decode** lands ~0–11% (mixed-batch decode ~0%); model-agnostic across head config.
- **Single-request transition** is the floor: full prefill at S_q=S_kv≈512 (R=1) hits
  ~30–44% — the compute-ramp where the kernel crosses from memory- to compute-bound, steep
  between octave grid points, analogous to GEMM's transition sag. Batched and longer cases
  sit at 1–7%.
- **Decode verified range.** Stress-tested across head config, paged block size, request
  count, and batch composition — the 1-D collapse holds for per-request context
  **L_i ≳ 128 tokens**. In the large-batch × very-short-context corner (many requests each
  < ~128 tokens), per-request overhead pulls efficiency below the curve; documented, not modeled.

### Continuous batching (mixed prefill + decode)

vLLM's FlashInfer backend **splits a step** into a `BatchDecode` and a `BatchPrefill` kernel
(`split_decodes_and_prefills`, threshold `S_q=1`), launched back-to-back on one stream — so
decode keeps its dedicated split-KV kernel and a mixed step is exactly `t_prefill + t_decode`.
Measured on real continuous-batching steps, `validate_predict.py --bench attn_mixed`:

    step latency vs  t_prefill + t_decode:   median 0.8%   mean 1.8%   max 7.1%

So continuous batching is predicted by **adding the two homogeneous predictions** — the decode
curve and prefill grid characterize real serving directly, no separate mixed-batch model. This
additivity is a property of the split launch: a *fused* kernel that routes decode rows through
the prefill path (e.g. vLLM's FlashAttention/FA2 on Ada, where `num_splits>1` is FA3-only)
loses decode's split-KV — it is **1.1–5.8× slower on mixed steps and not composable** (sum
mispredicts by ~50%). That is the concrete reason FlashInfer is the targeted backend here.

## MoE (vLLM fused_experts, Triton)

A fourth op. A fused-MoE layer routes each of M tokens to top_k of E experts, then runs two
grouped GEMMs per expert: gate+up (`H→2I`), SiLU, down (`I→H`). We **model it as those two
grouped GEMMs under uniform routing** (the analytic roofline) but **measure the real
`fused_experts` kernel**, so fusion + routing land in the efficiency factor — same roofline ÷
efficiency split as the other ops.

    routed tokens  T = M·top_k ;  active experts E_act = min(E, T) ;  per-expert tokens T/E_act
    FLOPs = 6·T·H·I  (gate+up 4·T·H·I + down 2·T·H·I)
    bytes = E_act·3·H·I·elem  (active w1+w2)  +  2·M·H·elem  (in/out acts)
    t = max(FLOPs/C_peak, bytes/B_peak) / f(T, E, H, I)

Two MoE-specific points the probe pinned down:
- **Active-expert correction.** At small M only `top_k` experts fire (not all E) — `E_act =
  min(E, M·top_k)`. Without it, decode over-predicts 6–11× (it assumes all E experts' weights
  are read). With it, decode lands at ~1%.
- **Efficiency keyed on `T = M·top_k`**, the grouped-GEMM work — so a model's own top_k folds
  in and one grid (swept at top_k=8) predicts any top_k. (Validated: gpt-oss top-4 predicted
  from the top-8 grid.)

Routing is **uniform** (the chosen simplification): balanced round-robin assignment, so the
measured kernel matches the uniform roofline. Real serving routing is imbalanced — a
load-imbalance factor is future work.

Accuracy on real expert configs (gpt-oss-20b: 32 experts top-4, H=I=2880; Qwen3-30B-A3B: 128
experts top-8, H=2048 I=768) × M = 1…4096, `validate_predict.py --bench moe_bf16`:

    latency error:  median 5.2%  mean 4.8%  p90 8.3%   (latency-weighted 4.6%, roofline-only 16%)

The Triton path uses whatever config vLLM selects (default heuristic where no tuned JSON
exists — exactly what serving runs); tuned-vs-untuned cancels in roofline ÷ efficiency, so no
autotuning is needed. mxfp4 MoE via Marlin is the planned next variant (like `marlin.py`).

## Files

    timing.py             CUDA-event timing, L2 flush, robust stats      (torch)
    gemm.py               GEMM sweep, model-agnostic grid, roofline       (torch)
    marlin.py             mxfp4 w4a16 sweep via vLLM Marlin + byte model (torch+vLLM)
    attn.py               FlashInfer attn sweep (best of fa2/fa3/cutlass/trtllm-gen per shape) (torch+flashinfer)
    moe.py                MoE sweep: fused_experts (Triton), two-grouped-GEMM roofline (torch+vLLM)
    run.py                run a benchmark (--bench <op>_<dtype>), dump JSON (torch)
    predict.py            latency predictor (gemm trilinear / attn hybrid / moe grouped-GEMM) (stdlib)
    validate_predict.py   predicted vs measured on real workloads         (torch)
    results/              gemm_<gpu>.json, marlin_mxfp4_<gpu>.json, attn_<gpu>.json, moe_<gpu>.json

## Run

Activate the env (torch + CUDA), then:

    python run.py --bench gemm_bf16  --c-peak 165 --b-peak 1008   # bf16 GEMM grid
    python run.py --bench gemm_mxfp4 --c-peak 165 --b-peak 1008   # mxfp4 w4a16 (Marlin)
    python run.py --bench attn_bf16  --c-peak 165 --b-peak 1008   # FlashInfer attn (best of kernels)
    python run.py --bench moe_bf16   --c-peak 165 --b-peak 1008   # MoE fused_experts (Triton)
    python predict.py --shape 2880 5120                           # gemm: latency vs M
    python predict.py --results results/attn_*.json --attn 4 1 8192 --head 32 4 128  # attn (R Sq Skv)
    python predict.py --results results/moe_*.json --moe 128 8 2048 768  # moe (E top_k H I) vs M
    python validate_predict.py --bench gemm_bf16                  # gemm accuracy
    python validate_predict.py --bench attn_bf16                  # attention accuracy
    python validate_predict.py --bench attn_mixed                 # mixed-step composition (t_pf+t_dec)
    python validate_predict.py --bench moe_bf16                   # moe accuracy

The sweep needs torch + a CUDA GPU (mxfp4 needs vLLM/Marlin, attn needs FlashInfer);
prediction does not. GEMM and prefill attention need `--c-peak`/`--b-peak`; decode is
memory-bound (B_peak only).

## Scope / next

- GPU: **RTX 4090** (Ada, SM89). No FP4 tensor cores, so mxfp4 is weight-only
  dequant→bf16 (memory win, not compute).
- Done: **bf16 GEMM** + **mxfp4 w4a16 (Marlin)** + **flash-attention (FlashInfer, hybrid:
  decode KV-byte curve + prefill grid)** + **fused MoE (Triton, two-grouped-GEMM)** —
  benchmark, predictor, and validation, all via `run.py --bench <op>_<dtype>`.
- Attention targets **FlashInfer** and per shape keeps the **best of its kernels**
  (fa2/fa3/cutlass/trtllm-gen, skipping unsupported) — best-achievable efficiency, portable
  to Hopper/Blackwell. Because FlashInfer splits decode/prefill, continuous batching composes
  additively (mixed step = `t_prefill + t_decode`, ~1.8%).
- Next: **mxfp4 MoE via Marlin** (like `marlin.py`); a **MoE load-imbalance factor** beyond
  uniform routing; re-measure on Hopper/Blackwell to pick up `fa3`/`trtllm` kernels.
