---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - fp8
  - modelopt
  - gemma4
  - capyctl
---

# Gemma-4-26B-A4B-NVFP4

[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) at
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, quantized with NVIDIA Model Optimizer
0.47.0. This is CapyCTL's r3: **NVFP4 routed experts, FP8 dense MLP, attention and untied
output head**, with BF16 embedding, router, norms, layer scalars and vision left unchanged.
The measurements used local shards byte-identical to this repository's files (SHA-256 values
in the [build notes](https://github.com/edurdias/capyctl-recipes/blob/main/vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/convert/NOTES.md)).

## Layout

| Tensors | This checkpoint | NVIDIA NVFP4 (`a19cfe00`) |
|---|---|---|
| Routed experts: 11,520 matrices, 30 layers × 128 experts × 3 projections | NVFP4 W4A4, group 16, searched block scales | NVFP4 W4A4, group 16 |
| Dense MLP: 90 projections | Per-tensor FP8 E4M3 W8A8 | BF16 |
| Attention: 115 projections | Per-tensor FP8 E4M3 W8A8 | BF16 |
| Output head | Separate FP8 E4M3 W8A8; `tie_word_embeddings=false` | Tied BF16 |
| Input embedding, router, norms, layer scalars, vision | Original BF16 | BF16 |
| FP8 scales | Scalar FP32 weight scale and calibrated static input scale | Not applicable to its BF16 projections |
| KV cache | FP8 metadata declaration; no calibrated K/V scale tensors | FP8 metadata declaration; no text-attention K/V scale tensors |
| Tensor payload | 17.875 GB | 18.782 GB |
| Weight-read estimate, including head | 3.208 GB/token | 5.592 GB/token |

GB is decimal. [INFERENCE] The final row counts one read of attention, dense MLP, router,
eight selected experts per layer and the output head, including stored weight scales.
It excludes activations, KV traffic, temporary buffers, other layer state and cache/batch
reuse. It is not measured bandwidth or a complete engine-traffic model. The untied head
adds stored weights but halves its weight payload relative to a BF16 output-head read.

The layout was chosen from a [family ablation](https://github.com/edurdias/capyctl-recipes/blob/main/comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/tensors/ablation/results/summary.md).
With the same searched NVFP4 experts, BF16 dense MLP/attention gives 97.397% greedy
next-token agreement. NVFP4 dense MLP alone lowers it to 96.455%; NVFP4 attention alone
to 95.512%. FP8 for both gives 97.151%. A simulated NVFP4 head lowers that to 96.861%;
a simulated FP8 head gives 97.134%. The stored r3 uses a per-tensor FP8 head, not that
simulation's per-channel head, and measures 97.177%.

## Measured on one GB10

Through [CapyCTL](https://github.com/edurdias/capyctl) 0.1.12, TP1, `temperature: 0`, normal
EOS, up to 512 output tokens, three recorded rounds after a warm-up. One stream is
per-request decode after the first token; 8 streams is aggregate completion tokens /
round wall time. Declared context is 131,072; long-prompt TTFT is measured at exactly
24,000 input tokens. These are recipe comparisons, not checkpoint-only speedups:

| Checkpoint and recipe | 1 stream, tokens/s | 8 streams, total tokens/s | 24k TTFT, 1 / 4 streams | GSM8K | Tools |
|---|---:|---:|---|---|---|
| **This checkpoint, vLLM 0.32.0 + Google MTP depth 3** | **79.56** | **302.29** | 7.277 / 18.857 s | 244/250 | 30/30 |
| This checkpoint, SGLang 0.5.22 + DFlash width 8 | 72.04 | 254.05 | 5.348 / 13.836 s | 244/250 | 30/30 |
| This checkpoint, SGLang + Google MTP depth 3 | 62.24 | 278.11 | 5.227 / 13.760 s | 242/250 | 30/30 |
| This checkpoint, vLLM, no drafts | 58.53 | 216.26 | 6.540 / 16.373 s | 244/250 | 30/30 |
| This checkpoint, SGLang, no drafts | 56.99 | 210.78 | 5.414 / 13.402 s | 244/250 | 30/30 |
| NVIDIA's build on SGLang 0.5.22 with a 131k-context production configuration, no drafts | 29.47 | 167.67 | 5.738 / 14.221 s | 243/250 | 30/30 |
| NVIDIA NVFP4, vLLM, no drafts | 29.73 | 164.89 | 7.129 / 17.779 s | 244/250 | 30/30 |
| Unsloth NVFP4, vLLM, no drafts | 45.74 | 202.29 | 6.602 / 16.463 s | 244/250 | 30/30 |

GSM8K uses the same pinned 250 test questions, 2,048-token cap; tools use 30 automatic
function-call cases through the Gemma4 parser. The main recipe and NVIDIA row have zero
cuts and zero HTTP errors. One GSM8K answer is 0.4 percentage points; these results do
not establish an accuracy advantage. The selected drafters also pass same-engine native
task and output-review gates. Exact greedy wording is not a losslessness test here:
no-draft outputs themselves change with execution shape and, on vLLM, across processes.

These are the selected search-run results. The best vLLM MTP3 result is 2.7× the
production baseline's one-stream decode and 1.8× its eight-stream throughput; engine
and drafter settings differ. Separate full sweeps of the shipped deployment files gave:

| Shipped recipe | 1 stream, decode tokens/s | 8 streams, aggregate tokens/s | 32k-target TTFT | Cold / warm ready |
|---|---:|---:|---:|---|
| vLLM MTP3 | 76.36 | 297.10 | 13.118 s | 261.79 / 184.99 s |
| SGLang DFlash8 | 71.69 | 255.15 | 7.836 s | 148.99 / 120.67 s |

The full sweeps use five rounds per concurrency and three per context target; their
32k-target prompts span 32,300–32,451 actual tokens. They are separate from the
fixed-24k measurements above. See the [final matrix](https://github.com/edurdias/capyctl-recipes/blob/main/comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/bench/final-matrix.json)
for every context and concurrency point.

### Under other publishers' recipes

Each publisher's recipe and workload, run on the same GB10 with its own checkpoint first and
then with this checkpoint in its place, nothing else changed:

| Recipe and workload | Its own checkpoint | This checkpoint | Published |
|---|---:|---:|---:|
| NVIDIA's official DGX Spark recipe (vLLM, Google MTP depth 4), SPEED-Bench coding, mean decode tokens/s | 81.46 (NVIDIA NVFP4) | **98.79** | 78.10 |
| A published DSpark repository's MTP depth 3 recipe, code / essay / chat tokens/s | 92.51 / 65.81 / 78.67 (RedHat NVFP4) | **95.94 / 69.34 / 82.39** | 98.18 / 68.26 / 80.91 |
| The same repository's DSpark, 7 draft tokens, code / essay / chat | 158.77 / 61.50 / 88.32 (RedHat NVFP4) | 157.80 / 58.83 / 94.82 | 159.18 / 65.86 / 93.26 |

NVIDIA's workload is reconstructed from its published counts on the pinned SPEED-Bench rows;
the row selection and timer are not NVIDIA's unpublished harness. Coding correctness was not
scored in these runs. Details are in the comparison.

### Closer to BF16

Every build reads the same BF16-generated answers to 200 prompts (chat, coding, math,
multilingual, tool use), and chooses the next token at all **62,658 answer positions**.
Weights are reconstructed into BF16 and evaluated with torch/Transformers, BF16
activations and SDPA, not native engine quantized kernels:

| Build | Revision | Agrees with BF16 | Coding |
|---|---|---:|---:|
| BF16 source | `4d7ae498` | 100% by construction, not an independent repeat | 100% |
| **This checkpoint, r3** | `241dbe07` | **97.177%** | **97.742%** |
| CapyCTL r3e, tied BF16 head | Local control | 97.191% | 97.819% |
| [`nvidia/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | `a19cfe00` | 97.132% | 97.701% |
| [`unsloth/gemma-4-26B-A4B-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4) | `20df0542` | 96.018% | 96.900% |
| CapyCTL r2, all-linear NVFP4, searched scales | Local control | 94.805% | 95.771% |
| CapyCTL r1, all-linear NVFP4 | Local control | 94.515% | 95.453% |
| [`RedHatAI/gemma-4-26B-A4B-it-NVFP4`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4) | `5557756b` | 94.476% | 95.489% |

Agreement counts synonyms and formatting changes as disagreements. It is neither a task
score nor an evaluation of activation quantization, FP8 KV, engine loaders or kernel
numerics. Full category results, ablations, scripts and per-request engine results are
in the [comparison](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10).

## Run it with CapyCTL

[Install CapyCTL](https://edurdias.github.io/capyctl/docs/install/) and register the
engine version. The [vLLM recipe](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10)
and [SGLang recipe](https://github.com/edurdias/capyctl-recipes/tree/main/sglang/gemma-4-26b-a4b-capyctl-nvfp4-gb10)
hold the deployment files. Download the chosen drafter; CapyCTL does not download it.

vLLM 0.32.0, Google MTP depth 3:

```bash
hf download google/gemma-4-26B-A4B-it-assistant \
  --revision 6e5aaaf4c42b98394530b8fda2e95cadd65c151c \
  --local-dir ~/drafters/gemma-4-26B-A4B-it-assistant
capyctl engine add ~/vllm-0.32.0-venv \
  --approve-option=--speculative-config --approve-path ~/drafters
```

SGLang 0.5.22, DFlash width 8:

```bash
hf download z-lab/gemma-4-26B-A4B-it-DFlash \
  --revision 77d4202772dfe50b2396ec7bac9cfffc7b9e7057 \
  --local-dir ~/drafters/gemma-4-26B-A4B-it-DFlash
capyctl engine add ~/sglang-0.5.22-venv \
  --approve-option=--speculative-draft-model-path --approve-path ~/drafters
```

Set the absolute drafter path in the chosen `deployment.yaml`; it already pins this
repository's weights commit, `241dbe0757d0fca815530e9216de6affb43565ff`. In one terminal:

```bash
capyctl start standalone \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB
```

From the chosen recipe directory in another terminal:

```bash
capyctl deploy model --file deployment.yaml
# vLLM recipe:
capyctl start deployment gemma4-26b-a4b-capyctl-nvfp4-vllm --wait
# Or the SGLang recipe:
capyctl start deployment gemma4-26b-a4b-capyctl-nvfp4-sglang --wait
```

The files retain the comparison's 40 GiB vLLM KV grant or 80 GiB SGLang request. Those
are not minimum-memory recommendations or equal cache capacities. See each recipe for
startup and system-wide peak-memory measurements. Parking/waking was not measured.

## Plain engine commands

These expose the recipe's serving flags without CapyCTL. Measurements above were made
through CapyCTL, not by benchmarking these command examples. The weights commit:

```bash
export MODEL_REVISION='241dbe0757d0fca815530e9216de6affb43565ff'
hf download capyctl/Gemma-4-26B-A4B-NVFP4 --revision "$MODEL_REVISION" \
  --local-dir ~/models/Gemma-4-26B-A4B-NVFP4
```

vLLM 0.32.0, with the Google assistant downloaded above:

```bash
~/vllm-0.32.0-venv/bin/vllm serve ~/models/Gemma-4-26B-A4B-NVFP4 \
  --tensor-parallel-size 1 --max-model-len 131072 \
  --max-num-batched-tokens 4096 --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 42949672960 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --speculative-config "{\"method\":\"mtp\",\"model\":\"$HOME/drafters/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":3}"
```

SGLang 0.5.22, with the DFlash checkpoint downloaded above:

```bash
~/sglang-0.5.22-venv/bin/python -m sglang.launch_server \
  --model-path ~/models/Gemma-4-26B-A4B-NVFP4 --tp-size 1 \
  --context-length 131072 --chunked-prefill-size 4096 --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --moe-runner-backend flashinfer_cutlass \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path ~/drafters/gemma-4-26B-A4B-it-DFlash \
  --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 8
```

The SGLang example leaves memory sizing to the engine; it does not reproduce CapyCTL's
80 GiB admission request. Neither example adds a KV connector or patches an engine.

## How it was made

[`convert/`](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/convert)
contains the driver, recipe, calibration builder, weight reader and audits. ModelOpt
0.47.0 calibrates 512 conversations of up to 1,024 tokens, 256 each from
[`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
and [`ise-uiuc/Magicoder-OSS-Instruct-75K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)
(both MIT), with the original Gemma chat template. Calibration SHA-256:
`a3e005c39c3c71918ffff53504091ed32bd02baf3d37241072e5a616b0dc31eb`.

The driver unties a copy of the source head before quantization, restricts FP8 body
selection to text layers, exports ModelOpt mixed precision, then re-rounds expert blocks.
The search compares the original scale plus 21 FP8-rounded candidates from 0.60× to
1.10× of the max-derived scale, in 0.025 steps, minimizing BF16 weight squared error over
that finite set. Expert relative L2 error is 8.58999%, versus 9.48309% before search.
No FP8 or retained BF16 tensor changes during the expert search.

The calibration gate checks all 60 shared expert-input and 7,680 fused expert-weight
quantizers and every exported activation/global scale. Calibration selected 3,833 of
3,840 layer/expert pairs; seven unselected pairs retain the genuinely calibrated shared
input scale, not invented per-expert observations. The final audit checks 47,446 tensors,
source assets, tying, exact dtype/shape and finite positive scales. A 542-route static
installed-source audit checks dispatch on both engines, including SGLang's explicit
expert-bank recipe aliases; native qualification is separate. Manifests retain source
revision, tools, recipe/calibration/script hashes, shard digests and layout results.

## Attribution and license

Base model: **Gemma 4 26B-A4B instruction-tuned model, Google**, from the pinned repository
above. Distributed under **Apache License 2.0**, as the source. Retain the source license,
applicable attribution and any NOTICE file when redistributing; the original model card
remains the reference for the source model's intended uses and limitations.

**Modifications by CapyCTL:** post-training NVFP4/FP8 quantization; expert block-scale
search; separate quantized output head with a BF16 input embedding; mixed-precision and
FP8-KV metadata, engine recipe aliases, conversion manifests and audits. Tokenizer,
processor, chat template and retained BF16 tensors are copied unchanged. This independent
conversion is not an official Google or NVIDIA release and implies no endorsement by
Google, NVIDIA or the publishers of comparison checkpoints.

## Limitations

- Text-only calibration. Vision weights are unchanged BF16; one image request answered
  sensibly on each shipped recipe, but image quality was not benchmarked.
- One GB10, one prompt corpus and small task sets; no hardware-wide speed claim or broad
  capability equivalence. The 131,072-token setting is serving capacity, not reliable retrieval.
- Offline agreement reconstructs weights into BF16. It excludes native W4A4/W8A8
  activation error, FP8 KV, engine loading and quantized-kernel numerics.
- Missing K/V scale tensors fall back to 1.0; FP8 metadata is not KV-scale calibration.
- Drafters may change wording. Statistical/task gates are not proof of mathematical
  speculative losslessness. DFlash width 4 was rejected for repetitive output despite
  passing score thresholds; width 8 is separately qualified.
- vLLM logs retain FP8 output-head fallback-tactic warnings for some captured shapes.
  Installed source commits are unknown; exact engine package versions are recorded.
- Long retrieval scores exact presence of all three codes in any order separately from
  ordered exact answers. At 111k input tokens on twelve documents, SGLang 0.5.22 without
  drafts gives NVIDIA 8/12 / 6/12 any-order and 4/12 / 4/12 ordered; r3 and the tied-BF16-head
  r3e control both give 6/12 / 4/12 any-order and 4/12 / 2/12 ordered (original / one-line).
  Both candidates versus NVIDIA have 0 wins, 4 losses, 8 ties; document-paired two-sided
  sign-test p=0.125. This is not equivalence or a significant gap. r3e matches r3 on every
  document/variant, so these results do not isolate an FP8-head cause.
- On six documents, shipped DFlash8 falls from 6/6 / 5/6 any-order at 32k to 1/6 / 0/6
  at 111k; ordered exact falls from 5/6 / 3/6 to 0/6 / 0/6. At 96k, removing DFlash
  improves any-order from 2/6 / 1/6 to 3/6 / 4/6, but still fails the full set.
  Neither shipped drafter recipe passes all six documents with both prompts at any
  tested 32k/64k/96k/111k point. NVIDIA's production configuration passes that finite
  any-order criterion at 32k, but ordered exact is only 3/6 / 3/6.
- Short exact-copy tests use 100 identical documents of exactly 2,000 input tokens,
  each containing one code. On vLLM, r3 with MTP3, r3 without drafts, NVIDIA without
  drafts and r3e with MTP3 each score 100/100 (95% Wilson: 96.30–100%).
  NVIDIA with MTP3 scores 99/100 (94.55–99.82%); its error drops a hex digit.
  Exact-code-presence counts equal strict-copy counts. These samples establish neither
  perfect copying nor a significant difference, and do not qualify long retrieval.

Full answers, retrieval scoring and paired statistics are linked from the
[comparison](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10).
