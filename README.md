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

(mxfp4 w4a16 is benchmarked only where models actually use it — the **MoE experts**, see
below. Dense linear projections run bf16 in every model we target, so there's no dense
`gemm_mxfp4` sweep.)

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

## MoE (vLLM — Triton bf16 + Marlin mxfp4)

A fourth op. A fused-MoE layer routes each of M tokens to top_k of E experts, then runs two
grouped GEMMs per expert: gate+up (`H→2I`), SiLU, down (`I→H`). We **model it as those two
grouped GEMMs under uniform routing** (the analytic roofline) but **measure the real kernel**,
so fusion + routing land in the efficiency factor — same roofline ÷ efficiency split as the
other ops. Two schemes (`--bench moe_bf16` / `moe_mxfp4`) differing only in the weight byte
model — `fused_experts` (Triton, bf16, 2 B/elem) and `fused_marlin_moe` (w4a16 Marlin, mxfp4
4-bit + E8M0 scale ≈ 0.53 B/elem; FLOPs identical since Marlin dequants to bf16 tensor cores).
Only the weight byte model differs between the two schemes.

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
experts top-8, H=2048 I=768) × M = 1…4096:

    bf16  (4090) :  median 4.1%  mean 3.7%   (latency-weighted 2.9%, roofline-only 16%)
    mxfp4 (GB200):  median 1.8%  latency-weighted 2.5%   (roofline-only 91% — 4-bit weights
                    put the theoretical roofline far above achieved, so the grid does the work)

For mxfp4, gpt-oss's 2880 dims are padded to 2944 (128-aligned) since Marlin can't tile
non-128 dims — that's the shape production runs. The lone outlier is gpt-oss M=1 (~57%, single-
token decode, 0.18 ms) — the steep low-efficiency corner that's the hard floor across every op
and contributes nothing to real latency (hence the 2.5% latency-weighted error).

The Triton path uses whatever config vLLM selects (default heuristic where no tuned JSON
exists — exactly what serving runs); tuned-vs-untuned cancels in roofline ÷ efficiency, so no
autotuning is needed. The **mxfp4 (Marlin)** scheme reuses the same grid/roofline with the
weight byte model swapped (per-expert mxfp4 Marlin weights via `marlin.make_mxfp4_weight`);
the predictor reads `moe_bytes_model` from the JSON.

## Files

    timing.py             CUDA-event timing, L2 flush, robust stats      (torch)
    gemm.py               GEMM sweep (bf16/fp16 via torch F.linear), model-agnostic grid, roofline (torch)
    attn.py               FlashInfer attn sweep (best of fa2/fa3/cutlass/trtllm-gen per shape) (torch+flashinfer)
    moe.py                MoE sweep: Triton bf16 + Marlin mxfp4, two-grouped-GEMM roofline (torch+vLLM)
    run.py                run a benchmark (--bench <op>_<dtype>), dump JSON (torch)
    run_all.sh            run every benchmark for one GPU (--c-peak/--b-peak)         (bash)
    predict.py            latency predictor (gemm trilinear / attn hybrid / moe grouped-GEMM) (stdlib)
    validate_predict.py   predicted vs measured on real workloads         (torch)
    results/              <op>_<dtype>.json — gemm_bf16, attn_bf16, moe_bf16, moe_mxfp4

## Run

Activate the env (torch + CUDA), then run all benchmarks for a GPU at once:

    ./run_all.sh --c-peak 165  --b-peak 1008                      # RTX 4090
    ./run_all.sh --c-peak 2250 --b-peak 8000                      # GB200 / B200

or one at a time:

    python run.py --bench gemm_bf16  --c-peak 165 --b-peak 1008   # bf16 GEMM grid
    python run.py --bench attn_bf16  --c-peak 165 --b-peak 1008   # FlashInfer attn (best of kernels)
    python run.py --bench moe_bf16   --c-peak 165 --b-peak 1008   # MoE fused_experts (Triton bf16)
    python run.py --bench moe_mxfp4  --c-peak 165 --b-peak 1008   # MoE w4a16 (Marlin mxfp4)
    python predict.py --shape 2880 5120                           # gemm: latency vs M
    python predict.py --results results/attn_bf16.json --attn 4 1 8192 --head 32 4 128  # attn (R Sq Skv)
    python predict.py --results results/moe_bf16.json --moe 128 8 2048 768  # moe (E top_k H I) vs M
    python validate_predict.py --bench gemm_bf16                  # gemm accuracy
    python validate_predict.py --bench attn_bf16                  # attention accuracy
    python validate_predict.py --bench attn_mixed                 # mixed-step composition (t_pf+t_dec)
    python validate_predict.py --bench moe_bf16                   # moe accuracy (Triton bf16)
    python validate_predict.py --bench moe_mxfp4                  # moe accuracy (Marlin mxfp4)

The sweep needs torch + a CUDA GPU (mxfp4 needs vLLM/Marlin, attn needs FlashInfer);
prediction does not. GEMM and prefill attention need `--c-peak`/`--b-peak`; decode is
memory-bound (B_peak only).

## Output format

Every benchmark writes the **same JSON schema** (`run.py`'s `_dump`; `predict.py` reads it,
and still loads pre-unification files):

    {
      "hardware":  { "gpu", "c_peak_tflops", "b_peak_gbps" },
      "operation": { "bench",                 # e.g. "gemm_bf16", "attn_bf16", "moe_mxfp4"
                     "impl",                   # {"torch"/"vllm"/"flashinfer": version}
                     "bytes_model" },          # {"w","a"} bytes/elem (weight, activation); attn w=0
      "results": [ { "shape":      { … op-specific input dims … },
                     "latency_ms",             # average latency
                     "tflops",                 # achieved compute throughput
                     "gbps",                   # achieved memory throughput
                     "efficiency" }, … ]       # roofline / measured
    }

`shape` carries each op's input dimensions: gemm `{M,K,N}`; moe `{M,E,top_k,H,I}`; attention
`{kind:"decode", kv_tokens,H_kv,D}` or `{kind:"prefill", Sq,Sk,RH,D}` (one `results` list holds
both). The efficiency factor is what the predictor interpolates; tflops/gbps/latency are the
measured throughputs for inspection.

## Scope / next

- GPU: **RTX 4090** (Ada, SM89). No FP4 tensor cores, so mxfp4 is weight-only
  dequant→bf16 (memory win, not compute).
- Done: **bf16 GEMM** + **flash-attention (FlashInfer, hybrid: decode KV-byte curve +
  prefill grid)** + **fused MoE (Triton bf16 + Marlin mxfp4 w4a16, two-grouped-GEMM)** —
  benchmark, predictor, and validation, all via `run.py --bench <op>_<dtype>`. (mxfp4 is in
  the MoE experts only — no model uses 4-bit dense projections.)
- Attention targets **FlashInfer** and per shape keeps the **best of its kernels**
  (fa2/fa3/cutlass/trtllm-gen, skipping unsupported) — best-achievable efficiency, portable
  to Hopper/Blackwell. Because FlashInfer splits decode/prefill, continuous batching composes
  additively (mixed step = `t_prefill + t_decode`, ~1.8%).
- MoE has two schemes: **bf16 (Triton `fused_experts`)** and **mxfp4 w4a16 (Marlin
  `fused_marlin_moe`)** — only the weight byte model differs. Both validated (~2–4%
  latency-weighted); gpt-oss mxfp4 measured at 2944 (Marlin needs 128-aligned dims, as prod pads).
- Next: a **MoE load-imbalance factor** beyond uniform routing; **best-of MoE backends**
  (trtllm/flashinfer) for Blackwell; re-measure attention on Hopper to pick up `fa3`/`trtllm`.
