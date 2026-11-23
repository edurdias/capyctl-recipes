---
license: apache-2.0
base_model: openai/gpt-oss-120b
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - gpt-oss
  - mxfp4
  - fp8
  - quark
  - quantized
---

# capyctl/gpt-oss-120b-MXFP4-FP8

I converted [OpenAI GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) at revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. This Apache-2.0 derivative retains the original model architecture and tokenizer. Weights revision: `628e2ca15177aaea61dfe8cb4fbe1be9737129dc`.

## What changed

I preserve OpenAI's MXFP4 experts, E8M0 scales and expert biases in a reversible per-expert Quark layout. Only the 144 attention Q/K/V/O matrices change to E4M3 FP8 with one FP32 multiplicative scale per output channel. Scale is `max(abs(weight))/448`, clamped to positive FP32 tiny; weights are divided by scale, clamped to [-448,448], then cast to E4M3. The BF16 output head, embeddings, routers, norms, sinks and other biases remain unchanged. There is no calibration or fine-tuning.

The exhaustive CPU audit covers 687 source tensors and 28,263 output tensors. All 216 expert packed-weight/scale/bias tensors reconstruct byte-identically, all retained tensors are unchanged, and attention bytes/scales match the conversion rule. Tensor payload is 64,294,452,864 bytes.

[INFERENCE] Shape-derived serialized selected-weight reads fall from **5.003 to 4.049 GB/token** (5,002,907,904 to 4,048,545,024 bytes/token), a 19.076% reduction. This counts one read of selected experts, dense weights and head plus one embedding row; it excludes KV/activation traffic, padding, reuse and caching. It is not measured bandwidth or a speed prediction.
## Measured results

All speeds are tokens/s. Short c1 is post-first-token decode; c2/c4/c8 are total completion tokens divided by round wall time. Long columns are one-stream post-first-token decode, not total request throughput. Each timing is the median of three measured rounds after warm-up, fixed chat/code prompts, 512 completion-token cap and normal EOS.

| Setup | Short c1 decode | c2 aggregate | c4 aggregate | c8 aggregate | Decode 8k / 32k / 64k / 110k |
|---|---:|---:|---:|---:|---|
| CapyCTL · vLLM 0.32 | 45.345 | 66.911 | 88.191 | 112.054 | 43.848 / 40.259 / 35.882 / 31.735 |
| OpenAI · vLLM 0.32 | 34.796 | 57.685 | 79.736 | 104.849 | 33.874 / 31.721 / 29.242 / 26.123 |
| Published OpenAI · vLLM 0.30 | 34.591 | 58.534 | 80.504 | 105.733 | 33.153 / 29.393 / 25.493 / 21.472 |

The matched vLLM 0.32 rows use expandable segments, lazy loading, 131072 context, eight sequences, an 8 GiB FP8 E4M3 KV pool, Marlin experts, no drafter, no prefix caching and LMCache off. Final timings come from the shipped-file runs; full quality comes from their completed full rows. Candidate gains are 30.3% short c1 decode, 6.9% c8 aggregate and 21.5% 110k decode. The 28-cell length/concurrency sweep is retained in `bench/final-matrix.json`; every paired candidate aggregate cell is higher, by 5.3–29.6%. These are finite observations, not uncertainty bounds or a universal speed claim.

Both matched checkpoints score GSM8K 239/250, tools 30/30 and HumanEval 157/164, with zero HTTP errors and truncations. Greedy generation uses low reasoning effort, eight concurrent requests, a 32768 total-output-token cap and pinned task records. Length-limited outputs count as wrong. The stated same-engine Wilson/truncation/tool gates pass; this is not statistical equivalence or broad capability qualification.

**One retrieval regression remains:** CapyCTL scores 47/48 observations and 143/144 codes versus OpenAI 48/48 and 144/144. The miss is one of six 111k one-line requests (5/6, 17/18 codes); both pass all six original-format 111k documents. Both checkpoints pass 100/100 strict copy cases. No retrieval/copy request has an HTTP error or output-cap hit.

The published vLLM 0.30 recipe is a serving-setup comparison, not an engine-version-only experiment: it has prefix caching enabled and auto KV, not confirmed FP8 KV. Its quality is only a 50/30/50 screen: GSM8K 48/50, tools 30/30, HumanEval 49/50. Repeated long-document TTFT includes cache hits and must not be compared with uncached prefill on 0.32.

## Loading and allocator finding

Default lazy loading and eager-per-file loading both failed during candidate shard reads before KV allocation or graph capture, hitting the 3 GiB available-memory guard. The retained minima are 2,867,150,848 and 1,277,296,640 bytes. Eager loading did not fix this. Explicit lazy loading plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reached readiness without changing checkpoint bytes or installed packages; available memory stayed roughly 49–51 GB during the 37 shard reads, with a 41,606,955,008-byte minimum through readiness.

[INFERENCE] This intervention strongly supports CUDA caching-allocator fragmentation across the 28,263 output tensors, rather than a necessary extra live model copy. Process-local allocated/reserved counters were unavailable; fragmentation was not directly measured. The recipe therefore keeps expandable segments **and lazy loading**. Register the engine with `--approve-env PYTORCH_CUDA_ALLOC_CONF`; do not remove the variable merely because the checkpoint is small enough on disk.

Final shipped-file readiness was 480.64 seconds for CapyCTL versus 529.67 for matched OpenAI, with whole-run minimum MemAvailable 39,686,987,776 versus 25,940,623,360 bytes. These are sampled unified-memory system observations, not model-only GPU allocations. Local-path execution passed; Hub provisioning at `628e2ca15177aaea61dfe8cb4fbe1be9737129dc` has not been exercised.

## Usage and limitations

Use vLLM 0.32.0 with Quark FP8 dense linears, Marlin MXFP4 experts, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and lazy safetensors loading. The tested recipe uses 131072 context, eight sequences, 8 GiB FP8 E4M3 KV, no prefix caching, no drafter and LMCache off. Use the original Harmony template and OpenAI GPT-OSS reasoning/tool parsers; tests explicitly use low reasoning effort.

The [recipe and evidence](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/gpt-oss-120b-capyctl-gb10) include the converter, exhaustive audit, runtime settings and comparison. Paths named `bench/` above refer to the [comparison evidence](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/gpt-oss-120b-builds-gb10). Local-path recipe execution was measured, not Hub provisioning of the placeholder revision.

These finite tests on one GB10 do not establish general capability equivalence, safety, reliability or support for other engines/hardware. In particular, the one 111k retrieval miss is a retained regression. CPU byte checks do not establish engine qualification. Generated answers can be wrong; assess your own task and retain OpenAI's model limitations and usage policy.

## License and attribution

Apache-2.0, derived from OpenAI's GPT-OSS-120B; retain the source LICENSE, NOTICE where supplied, and usage policy. This is my independent conversion and is not an OpenAI release or endorsement.
