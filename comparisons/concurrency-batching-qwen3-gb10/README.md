# Concurrency and native batching on one GB10

Measured 2026-10-26–27 UTC through CapyCTL v0.1.8 and the explicitly identified experimental admission builds. **Experimental comparison, not a deployable production recipe.**

Inspired by [Cerebras’s explanation of frontier-model serving](https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras). Its wafer/pipeline results are not GPU performance evidence.

[All 313 measurement summaries](bench/results.json) · [Dense details](dense.md) · [MoE details](moe.md) · [Reproduction](reproduce.md)

### TL;DR — question, results and useful knobs

**The ask:** inspired by Cerebras’s concurrency/microbatching discussion, test whether more concurrent requests and native-engine batching/prefill tuning improve throughput through CapyCTL **without breaking latency, correctness or memory limits**. This was a single-GPU experiment, not an implementation of Cerebras-style pipeline parallelism or a new CapyCTL scheduler.

**The result:** higher concurrency produced the largest demonstrated scaling gains. Batch/prefill tuning was workload-specific: sometimes it improved throughput, sometimes it mainly reduced stalls, and sometimes it did not help.

**Dense 4B: initial baseline → best three-repeat-confirmed point.** Both endpoints below are three-run medians and passed the service gates. The initial sweep used 24 GiB total / 4 GiB KV and offered concurrency no higher than 8; the selected profiles use 40 GiB total / 16 GiB KV, different engine-slot limits, and experimental admission binaries. Long/mixed profiles also use a 32,768-token scheduling budget. These are **combined configuration gains**, not isolated batching gains or gains over the best possible stock-release configuration.

| Engine | Workload | Offered C, before → after | Output tok/s, before → after | Gain |
|---|---|---:|---:|---:|
| sglang | short | 8 → 104 | 201.73 → 1476.83 | +632.1% |
| sglang | long | 4 → 8 | 26.39 → 33.69 | +27.6% |
| sglang | mixed | 8 → 20 | 95.76 → 138.42 | +44.5% |
| vllm | short | 8 → 124 | 198.68 → 1653.15 | +732.1% |
| vllm | long | 2 → 8 | 22.23 → 33.81 | +52.1% |
| vllm | mixed | 4 → 14 | 66.09 → 114.38 | +73.1% |

**Which knobs helped?**

- **Concurrency: strongest isolated scaling evidence.** With the same dense-short baseline configuration—including memory, engine slots and batching settings—raising only offered concurrency from 4 to 8 increased SGLang from **95.37 to 202.21 tok/s (+112%)**, and vLLM from **97.79 to 197.67 (+102%)**. These are single screening comparisons, not repeated gain estimates. Higher concurrency eventually violated latency gates.
- **KV capacity: an enabling constraint, not an unlimited speed knob.** With the original 4-GiB KV budget, dense long-prompt throughput levelled around 26–27 tok/s as concurrency rose, while latency worsened. Larger-memory profiles supported better operating points, but their slot/build changes prevent assigning a clean percentage gain to KV alone.
- **Batch/prefill budget: workload-specific, including latency rather than throughput wins.** At fixed dense-mixed C=16, 48-GiB KV and 128 engine slots, changing only vLLM’s token budget from 2,048 to 32,768 changed throughput from **124.73 to 121.52 tok/s (−2.6%)**, but reduced p95 content-chunk gaps from **343 to 86 ms**, turning that screening cell from a latency failure into a pass. This was a single comparison; the final three-repeat-confirmed mixed point is C=14, not C=16. The original fixed-budget repeated comparisons did not establish a broad batching-default win.

**The MoE check mattered:** its confirmed short-prompt concurrency was only **24 for SGLang / 16 for vLLM**, versus **104 / 124** for the dense 4B model. That is a model-specific serving result, not a controlled architecture comparison. One concurrency profile does not fit both.

**Action:** fix overload/error propagation first, then separately validate model/workload-specific profiles. Keep production defaults unchanged and batching inside the engines. We found useful operating points—not the GPU’s absolute ceiling.

### Detailed results

**Retain existing production defaults.** Native concurrency and engine scheduling materially change the usable serving envelope, but the best settings depend on model and workload. This experiment does not justify adding token scheduling to CapyCTL or promoting the largest measured concurrency to a default.

Completed **313 indexed attempts: 219 dense 4B and 94 MoE**, on one authorized GB10, through CapyCTL, with unchanged approved SGLang 0.5.22 and vLLM 0.32.0 installations. The 12 selected model/engine/workload points each passed **three completed confirmation repeats**. All rejected and interrupted attempts remain published; 302 cohorts have valid aggregate token accounting and 11 do not. Valid accounting alone does not mean a latency pass.

### Best confirmed points within the tested matrix

Each engine cell is **offered concurrency → median aggregate output tok/s (three-run range)**. These are actual output tokens over the full cohort, not stream speed multiplied by concurrency. All selected points use the explicitly recorded experimental admission builds, not unchanged-release qualification.

| Model | Workload | SGLang | vLLM |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | short | 104 → 1476.83 (1475.78–1481.85) | 124 → 1653.15 (1651.62–1657.22) |
| Qwen3-4B-Instruct-2507 | long | 8 → 33.69 (33.66–34.92) | 8 → 33.81 (33.67–33.93) |
| Qwen3-4B-Instruct-2507 | mixed | 20 → 138.42 (138.29–138.89) | 14 → 114.38 (112.41–114.46) |
| Qwen3-30B-A3B | short | 24 → 180.69 (177.06–181.15) | 16 → 141.95 (141.43–143.55) |
| Qwen3-30B-A3B | long | 6 → 27.11 (25.99–27.27) | 5 → 26.12 (25.65–26.27) |
| Qwen3-30B-A3B | mixed | 12 → 65.28 (65.00–67.86) | 10 → 67.33 (66.34–68.02) |

The unchanged gates were p95 TTFT ≤2s short / ≤10s long and mixed, pooled p95 content-chunk gap ≤200ms, zero request/contract failures, and protected memory headroom. Some passing points are very close to those boundaries. These finite closed-loop cohorts include ramp/drain and are not a production SLA or a stationary/open-loop capacity guarantee. Both model reports include exact configurations, stream-speed tradeoffs and worst-repeat latency.

### What CapyCTL should take from this

1. **Fix overload/error propagation before increasing production admission.** The released 32-request server bound, separate 64-request host ingress bound and experiment host’s 1,024-soft-FD limit were distinct constraints. Controlled experimental changes removed the corresponding boundaries. Rejected forwards/resource failures surfaced as empty HTTP-200 streams in some cohorts; those were counted as errors, never successful throughput. The remaining four empty streams in one dense SGLang C=1,024 repeat have no established cause.
2. **Validate model/workload-specific native-engine profiles separately.** The original fixed-memory official-release comparison did not establish a broad batching-default win. Larger concurrency, KV capacity and prefill budgets can improve particular workloads, while worsening individual-stream latency or isolated stalls. Admission changes need coordinated ingress/descriptor capacity; batch formation stays in the engine.
3. **Do not call the largest result a hardware ceiling.** Dense high-concurrency results levelled near 2.5k output tok/s in some runs but had substantial variation and latency failures. At MoE C=1,024, vLLM nearly flattened with a 97.3%-occupied KV cache, while SGLang still increased with 91% occupancy. Known cache/slot/admission limits remain, and all these diagnostic endpoints fail service gates. Maximum physical throughput remains unresolved beyond this bounded recipe/envelope.
4. **Keep attribution and measurement limits explicit.** More admitted engine concurrency and engine-specific settings change results; reduced controller overhead was not established. There is no matched direct-engine baseline or internal token-batch trace. SSE gaps are not exact token latencies. Accelerator-memory allocation is unavailable independently on this GB10, and no whole-system energy claim is made. Available-memory headroom did not mean zero memory pressure; the MoE report documents the observed pressure and warmup-history caveat.

### Reproducible evidence index

- [Dense 4B: configurations, repeat ranges, stream tradeoffs and all 219 attempts](https://github.com/edurdias/capyctl/issues/211#issuecomment-6431303620)
- [MoE: configurations, repeat ranges, stream tradeoffs and all 94 attempts](https://github.com/edurdias/capyctl/issues/211#issuecomment-6434346764)
- [Occupancy, KV, system memory, pressure, FD limits and clock-event evidence](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443586946)
- [TTFT, chunk-gap tails, per-request maximum gaps and stream rates](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443587549)
- [Request/token accounting, output lengths, refusals, failures, cancellations and cohort boundaries](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443588115)
- [Executable driver and observer source](https://github.com/edurdias/capyctl/issues/211#issuecomment-6435986028)
- [Executable orchestration and report-generation source](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443042524)
- [Setup, binary provenance, manifests and reproduction commands](https://github.com/edurdias/capyctl/issues/211#issuecomment-6436349184)

### Verified teardown

The management deployment list was empty before shutdown. Ownership-verified host/server stop helpers completed, and a fresh independent check confirmed both recorded role PIDs absent. Host MemAvailable was **117.71 GiB**, with memory-pressure avg10/avg60/avg300 all zero. The process-local FD-limit adjustment ended with the host role. No production defaults, engine environments, drivers or system packages changed; host B was untouched. No native experiment remains running.

This completes the bounded experiment and its evidence publication. Admission/error-handling fixes or recipe/default changes are separate work; none is silently included here.

## Data and method

`bench/results.json` contains all 313 cohort summaries, including invalid attempts, checkpoint identities, build labels, request/token accounting, latency distributions and sampled occupancy/memory. `evidence_id` joins the tables published on the source issue. The local run-directory path is omitted; IDs retain their original values. Missing measurements remain null, not zero. These are measured summaries, **not the full raw SSE, request or telemetry logs**; those remain in the original local evidence archive. This is a custom bounded-cohort experiment, not a capyctl-bench report.

The corpus is 40 deterministic prompts, seed 211. Short prompts average about 145 actual input tokens with at most 256 output tokens; long prompts about 8,785 with at most 64; mixed admissions alternate short/long equally. Temperature is zero and EOS is respected. Two serial warmup requests (four for mixed) are excluded. Minimum admission duration is 60 seconds, with at least 40 completions; high-concurrency diagnostics require at least twice offered concurrency. The full deadline, memory, cache and build protocol is in [reproduce.md](reproduce.md).

Aggregate rate is actual output-token usage divided by the interval from first measured dispatch to final drain. The dataset's `steady_*` fields describe the admission interval, which includes ramp-up; they do not establish statistical steady state. The per-stream estimate is `(completion_tokens - 1) / (DONE arrival - first content arrival)`. Correctness checks HTTP/SSE/usage and numbered-list structure, not general model quality. Thinking is explicitly disabled for the MoE. Runtime dtype is BF16, context is 16,384, and graphs, speculation and reusable prefix/radix caching are disabled.

No deployment manifest is promoted as a recipe. Reproduction includes experimental manifests and requires reviewing the isolated-build and host-safety instructions first. Existing production defaults are unchanged.
