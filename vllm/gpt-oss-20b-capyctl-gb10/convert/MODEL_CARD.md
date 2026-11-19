---
license: apache-2.0
base_model: openai/gpt-oss-20b
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - gpt-oss
  - mxfp4
  - fp8
  - quark
  - capyctl
---

# gpt-oss-20b-MXFP4-FP8

CapyCTL's conversion of [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
at `6cee5e81ee83917806bbde320786a8fb61efebee`: **original MXFP4 experts,
per-output-channel FP8 E4M3 attention, BF16 output head, embedding and routers**.
The Quark layout is qualified on vLLM 0.32.0 on one NVIDIA GB10, without a drafter.
The measurements used local shards with the same SHA-256 as this repository's files.

## Layout

| Tensors | This checkpoint | OpenAI's released checkpoint |
|---|---|---|
| Expert weights and local scales | Original MXFP4, group 32, E8M0 scales; reversible per-expert Quark serialization | MXFP4, combined expert banks |
| Expert biases | Original BF16, byte-identical | BF16 |
| Attention Q/K/V/O: 96 matrices | FP8 E4M3, FP32 per-output-channel multiplicative scales; dynamic per-token FP8 input quantization at runtime | BF16 |
| Output head and input embedding | Original untied BF16 | Untied BF16 |
| Routers, norms, attention sinks and other biases | Original BF16 | BF16 |
| Tensor payload | 13,125,022,848 bytes | 13,761,264,768 bytes |
| Estimated batch-one weight reads | 3.072 GB/token | 3.708 GB/token |

[INFERENCE] The last row counts one read of attention, the head and router, four
selected experts per layer, their weight scales and small retained state. It excludes
KV/activation traffic, scratch space and batch/cache reuse. The 17.2% reduction is
payload arithmetic, not observed bandwidth. GB is decimal. The head remains BF16:
the installed GPT-OSS model does not forward quantization configuration to its head.
No checkpoint-only declaration enables an FP8 head in that implementation.

The [complete audit](https://github.com/edurdias/capyctl-recipes/blob/main/vllm/gpt-oss-20b-capyctl-gb10/convert/checkpoint-audit.json)
covers 459 source and 5,019 candidate tensors. All 144 source expert packed-weight,
scale and bias tensors reconstruct byte-identically; all 96 attention weight/scale
pairs match the conversion rule. Every other tensor is unchanged.

## Measured on one GB10

Through [CapyCTL](https://github.com/edurdias/capyctl) 0.1.12, TP1, 131,072-token
context, FP8 E4M3 KV, LMCache off, temperature zero and reasoning effort low.
Speed uses eight fixed chat/code prompts, one warm-up and three measured rounds,
normal EOS and a 512-total-completion-token cap including reasoning. One stream is
post-first-token decode; eight streams is aggregate completion tokens / round wall time.

| Checkpoint and setup | 1 stream, tokens/s | 8 streams, aggregate tokens/s | 24k TTFT, 1 / 4 streams, s | GSM8K | Tools |
|---|---:|---:|---|---:|---:|
| **This checkpoint, vLLM 0.32.0, no drafter** | **57.77** | **196.71** | 3.543 / 9.601 | 237/250 | 30/30 |
| OpenAI, same vLLM 0.32.0 | 45.81 | 184.07 | 3.760 / 10.141 | 236/250 | 30/30 |
| OpenAI's checkpoint on SGLang 0.5.22 with a 131k-context production configuration | 44.85 | 178.67 | 3.180 / 12.236 | 232/250 | 30/30 |
| LMSYS `gpt-oss-20b-bf16`, same vLLM, BF16 loaded MoE | 28.22 | 70.14 | Not measured | 237/250 | 30/30 |
| ggml-org MXFP4/Q8_0, llama.cpp b12242, no drafter | 76.16 | 145.87 | 6.092 / 18.688 | 237/250, one cap hit | 30/30 |

This checkpoint is 26.1% faster at one stream and 6.9% at eight than OpenAI's
checkpoint on the same vLLM setup. Against production SGLang, the differences
combine engine and checkpoint. llama.cpp leads single-stream decode, but its
retained GSM8K cap hit repeats Unicode whitespace; it is not a clean output-review
pass. A one-answer GSM8K difference does not establish an accuracy advantage.
Native accuracy uses an 8,192-total-token cap; this checkpoint has zero HTTP errors
and zero accuracy truncations and passes output review.

### The public builds

| Checkpoint | Revision | Observed result under vLLM 0.32.0 |
|---|---|---|
| [LMSYS](https://huggingface.co/lmsys/gpt-oss-20b-bf16) | `8de8d3dfa4da071323322a4a3049612203a3b1eb` | Experts stored E5M2 FP8, but config requests no quantization; loaded BF16 `TritonExperts`, not native FP8 MoE. Speed/tasks above; four coherence prompts pass. |
| [shanjiaz NVFP4](https://huggingface.co/shanjiaz/gpt-oss-20b-nvfp4-modelopt) | `fb9848e169d5b38cbc00ecf3383283ea1fc33a21` | Fails before readiness: `KeyError: layers.0.mlp.experts.w2_bias`. No speed/task/agreement score. |
| [mmangkad NVFP4](https://huggingface.co/mmangkad/gpt-oss-20b-nvfp4) | `86940c3e09c00f291dca96580770e4ba81542c9c` | Same expert-bias loader failure; no speed/task/agreement score. |
| [RedHatAI](https://huggingface.co/RedHatAI/gpt-oss-20b) | `9fff8613d547ba9b79bf4604fa9820877d0f6aff` | All three root safetensor shards match OpenAI's bytes and hashes exactly; no duplicate benchmark. |

Both NVFP4 checkpoints contain all 48 expert bias tensors. Their selected quantized
path does not allocate the requested parameter; this is a loader failure, not a zero
score or evidence that their files omit biases. Transformers also rejects their packed
expert shapes in the unchanged evaluation environment, so offline agreement is not measured.

### Long-context decode and retrieval

| Exact input tokens | This checkpoint, tokens/s | OpenAI, same vLLM | Production SGLang |
|---:|---:|---:|---:|
| 8,000 | 56.35 | 44.85 | 43.67 |
| 32,000 | 52.23 | 42.18 | 40.32 |
| 64,000 | 47.57 | 39.17 | 36.71 |
| 110,000 | 42.54 | 35.50 | 32.48 |

Three single-stream rounds per fixed document, 512-total-token cap; decode excludes
TTFT. vLLM prefix caching is disabled. These are not uncached-prefill measurements.
Retrieval passes **48/48 observations**: six documents, original/one-line prompts,
32k/64k/96k/111k exact original input lengths (one-line adds 14 tokens), all three
codes present and ordered exactly. Zero HTTP errors or 2,048-token cap hits.
OpenAI on the same vLLM misses one 111k one-line observation at the output cap.
The finite corpus does not guarantee arbitrary 131k retrieval.

The separate shipped-file sweep completes all 28 exact-token input/concurrency
cells from 500 to 32,000 input tokens at 1/2/4/8 streams. At 500 tokens it gives
57.49 one-stream decode and 258.27 aggregate at eight; at 32k, 52.03 / 65.07.
This shared-document workload differs from the ranking corpus and must not replace
its 196.71 eight-stream result. Local ready time is 104.19 s, warm restart 103.30 s;
sampled host/GPU peaks are 67.545 / 58.730 GB. OS/file/JIT caches were not cleared.
Low/medium reasoning separation and required-argument/empty-schema tool round trips pass.

### Why no drafter

On this checkpoint with Triton attention, DFlash8 reaches 81.26 / 264.44 tokens/s
at 1 / 8 short-prompt streams, but decodes only 49.08 / 22.01 at 8k / 32k input,
against no-draft 56.25 / 52.30. Nebius EAGLE3 depth 3 reaches 82.59 / 265.55
short-prompt speed, but loses at 32k and scores 29/30 tools. Depth 5 scores 29/30
tools and repeats a phrase through a short guard's cap. None meets both task and
long-context requirements; the recipe keeps auto/FlashInfer attention without drafts.

### Offline next-token agreement

The same 200 Harmony prompts and 52,550 answer positions, weights reconstructed
into BF16, eager attention, torch 2.13.0+cu130 / Transformers 5.14.1:

| Weights | Against actual teacher-forced OpenAI reconstruction | Against cached generated answer IDs |
|---|---:|---:|
| OpenAI reconstruction | 100% by definition | 98.39201% self-consistency |
| This checkpoint | 98.14843% (51,577/52,550) | 97.77545% (51,381/52,550) |
| LMSYS | 100% (52,550/52,550) | 98.39201% |

**98.15% agreement and OpenAI's 98.39% self-consistency have different targets.**
On the common generated-ID basis the gap is about 0.62 percentage points. The
initial 99.5% offline gate failed; it was judged too strict before native vLLM
qualification passed and this checkpoint was selected for publication. All positions
remain scored, including 64 capped records (136 completed). This weight-only test
excludes dynamic activation quantization, FP8 KV and native engine kernels.

## Run it with CapyCTL

[Install CapyCTL](https://edurdias.github.io/capyctl/docs/install/) and vLLM 0.32.0.
The [recipe](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/gpt-oss-20b-capyctl-gb10)
contains `deployment.yaml`, its explicit ten-file runtime list and first-request commands.
Prepare the offline Harmony vocabulary as described there, then register the engine:

```bash
CUDA_HOME=/usr/local/cuda-13.0 capyctl engine add /home/me/vllm-0.32.0-venv \
  --env TIKTOKEN_ENCODINGS_BASE=/home/me/models/gpt-oss-tokenizer
capyctl start standalone \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB
```

In another terminal, from the recipe directory, after replacing its publication
revision placeholder with the uploaded weights commit:

```bash
capyctl deploy model --file deployment.yaml
capyctl start deployment gpt-oss-20b-capyctl-vllm --wait
```

The 40 GiB KV grant is not the total footprint or a minimum-memory recommendation.
The measured file was locally validated with only its source replaced by the audited
checkpoint. Hub provisioning remains untested until a real weights commit is assigned.

## Plain vLLM command

This exposes the recipe flags; the measurements above went through CapyCTL, not
this command example. Download the pinned checkpoint after replacing the placeholder:

```bash
export MODEL_REVISION='66d5d29bb56ebef0d24c0333b7646d058b581170'
hf download capyctl/gpt-oss-20b-MXFP4-FP8 --revision "$MODEL_REVISION" \
  --local-dir /home/me/models/gpt-oss-20b-MXFP4-FP8
TIKTOKEN_ENCODINGS_BASE=/home/me/models/gpt-oss-tokenizer \
  /home/me/vllm-0.32.0-venv/bin/vllm serve /home/me/models/gpt-oss-20b-MXFP4-FP8 \
  --tensor-parallel-size 1 --max-model-len 131072 --max-num-seqs 16 \
  --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 42949672960 \
  --max-cudagraph-capture-size 16 --no-enable-prefix-caching \
  --reasoning-parser openai_gptoss --enable-auto-tool-choice --tool-call-parser openai
```

Use an environment with the measured CUDA/FlashInfer setup. Do not add a KV
connector or speculative configuration if reproducing the no-drafter recipe.

## How it was made

[`mixed_attention.py`](https://github.com/edurdias/capyctl-recipes/blob/main/vllm/gpt-oss-20b-capyctl-gb10/convert/mixed_attention.py)
runs CPU-only in the existing ModelOpt 0.47.0 environment using torch/safetensors;
no ModelOpt or Quark exporter is invoked. For each attention output channel,
`scale = max(abs(weight)) / 448`, clamped to positive FP32 tiny; weights are
clamped to E4M3's range and cast to FP8, with multiplicative FP32 scales stored
separately. There is no activation-calibration dataset: runtime inputs use dynamic
FP8 quantization. Expert packed nibbles, E8M0 scales, gate/up interleave and biases
are split by expert without dequantization or requantization. The audit reverses
that layout and compares source bytes. Tokenizer, Harmony template, generation
configuration and retained BF16 tensors are copied unchanged.

```bash
CUDA_VISIBLE_DEVICES='' /home/me/modelopt-0.47.0-venv/bin/python mixed_attention.py \
  /home/me/models/gpt-oss-20b /home/me/models/gpt-oss-20b-MXFP4-FP8 --scale channel
```

The output directory must not already exist. Conversion/audit scripts and file hashes
are in the [conversion directory](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/gpt-oss-20b-capyctl-gb10/convert).
The [comparison](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/gpt-oss-20b-builds-gb10)
contains per-request evidence, exact methods and all failed or unmeasured controls.

## Attribution and license

Base model: **GPT-OSS-20B, OpenAI**, from the pinned source above. Distributed
under **Apache License 2.0**, as the source. Retain the
[source license](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/LICENSE),
applicable copyright, patent, trademark and attribution notices and any applicable
NOTICE contents when redistributing. Retain OpenAI's
[USAGE_POLICY](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/USAGE_POLICY),
which requires compliance with applicable law. The original
[model card](https://huggingface.co/openai/gpt-oss-20b) remains the reference for
OpenAI's intended uses, safety findings and limitations.

**Modifications by CapyCTL:** attention weights quantized to per-channel FP8 E4M3;
FP32 attention weight scales added; original MXFP4 experts repacked into reversible
per-expert Quark tensors; quantization configuration, index, conversion manifests
and serving recipe added. Expert numerical values and retained BF16 tensors are
unchanged. This is an independent conversion, not an official OpenAI release;
OpenAI, AMD and NVIDIA do not endorse it. Apache-2.0 grants no trademark endorsement.

## Limitations

- **vLLM only:** qualified on vLLM 0.32.0 and one GB10. The inspected SGLang model
  does not pass quantization configuration to attention; this is not a drop-in
  SGLang checkpoint. Other hardware and vLLM versions are not qualified here.
- **Head stays BF16.** No FP8-head speed claim. Model-loading memory is not a
  separately measured attention allocation or total serving footprint.
- **Agreement is 98.15%, not lossless.** OpenAI's 98.39% self-consistency uses
  generated IDs, not the candidate's teacher-forced comparison target. Native
  task results on small samples do not prove capability equivalence.
- FP8 KV falls back to unit scales where calibration tensors are absent. The
  conversion does not calibrate KV or static activations.
- Retrieval passes the tested 48 observations, not every application or arbitrary
  131k prompt. Copy-100 on this checkpoint is **not measured**; OpenAI's SGLang
  controls score 50/100 strict and 100/100 code presence because half add `Code: `.
- Drafters can improve short-input speed while regressing long-input decode or
  tool accuracy. No drafter ships. Parking/waking is not measured.
- Publication revision is pending; local inference is qualified, Hub provisioning
  is not. No prequantization OpenAI BF16 teacher was available.
