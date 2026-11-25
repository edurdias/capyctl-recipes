# CapyCTL recipes

Deployment files for [CapyCTL](https://github.com/edurdias/capyctl), one per
engine, model and GPU, each measured through CapyCTL on the hardware it names.

A recipe is a directory with two files:

- `deployment.yaml`, which `capyctl deploy model --file` accepts as it is;
- `README.md`, which states the hardware, the engine and its exact version, the
  model and its pinned revision, the drafter if there is one, the commands from
  `capyctl engine add` to a first answer, and the numbers measured through
  CapyCTL, with the date and the CapyCTL version they were measured on.

Recipes hold configuration only: no container images or engine patches, and
no scripts except one that rebuilds a converted checkpoint a recipe serves (in
the recipe's `convert/`). CapyCTL runs the engine installation you already
have; see its [engine guide](https://github.com/edurdias/capyctl/blob/main/docs/guide/engines.md).

## Use one

1. Install CapyCTL and start it on the machine
   ([Run on one machine](https://github.com/edurdias/capyctl/blob/main/docs/guide/one-machine.md)).
2. Install the engine version the recipe names in a virtual environment, then
   register it with `capyctl engine add <venv>`.
3. `capyctl deploy model --file <recipe>/deployment.yaml`, then
   `capyctl start deployment <name> --wait`.

The recipe's README lists the exact commands and what they printed.

## Recipes

| Recipe | Engine | Model | Hardware | Decode, one stream | Measured |
|---|---|---|---|---|---|
| [sglang/nemotron-3.5-super-vl-120b-a12b-mtp-gb10](sglang/nemotron-3.5-super-vl-120b-a12b-mtp-gb10/) | SGLang 0.5.22; native launch ([fixed-memory-share limitation](https://github.com/edurdias/capyctl/issues/264)) | NVIDIA NVFP4 `9838c915`, native MTP, measured bounded pools | 1x GB10 | Direct engine: 36.24 one-stream decode; 51.44 c2 aggregate; full task gate passes (242/250, 30/30, 161/164); retrieval 11/12 through 64k, 96k/111k not completed after hard freeze | 2026-11-23 |
| [sglang/clef-flash-capyctl-mixed-gb10](sglang/clef-flash-capyctl-mixed-gb10/) | SGLang 0.5.22 | Clef-Flash CapyCTL mixed r1, NVFP4/FP8 body and BF16 heads | 1x GB10 | Decisions: 49.35 / 50.75 ms p50 / p95, one question at nominal 200 input tokens; held-out BF16 top agreement 5200/5294 (98.224%), below 99% | 2026-11-22 |
| [sglang/inkling-2-mini-capyctl-fp8-gb10](sglang/inkling-2-mini-capyctl-fp8-gb10/) | SGLang PR #51926 `92e894ef` + loader patch; native launch | CapyCTL Inkling 2 Mini: original NVFP4/MXFP8 experts and BF16 head, FP8 attention; DSpark gamma 7 / block 8 | GB10, static memory share 0.80 | 48.08 one-stream decode; 47.31 / 134.13 aggregate at 1 / 8 streams; GSM8K 240/250, tools 28/30, HumanEval 149/164; retrieval 56/60 (strict fail) | 2026-11-21 |
| [vllm/qwen3.6-35b-a3b-capyctl-nvfp4-gb10](vllm/qwen3.6-35b-a3b-capyctl-nvfp4-gb10/) | vLLM 0.32.0 | CapyCTL DeltaNet NVFP4, NVIDIA NVFP4 experts/head and FP8 full attention; DFlash2 block 8 | 1x GB10, 24 GiB KV grant | Shipped-file run: 131.2 one-stream decode, 336.3 aggregate at 8 streams; ranking corpus 129.7 / 328.4 aggregate at 1 / 8; GSM8K 234/250, tools 30/30, HumanEval 159/164; retrieval 60/60 | 2026-11-20–21 |
| [vllm/gpt-oss-20b-capyctl-gb10](vllm/gpt-oss-20b-capyctl-gb10/) | vLLM 0.32.0 | CapyCTL GPT-OSS-20B: original MXFP4 experts, FP8 attention, BF16 head; no drafter, 131k context | 1x GB10, 40 GiB KV grant | Fixed chat/code corpus: 57.77 tokens/s (196.71 at 8 streams); 42.54 at 110k input; GSM8K 237/250, tools 30/30, retrieval 48/48 observations | 2026-11-19 |
| [vllm/gpt-oss-120b-capyctl-gb10](vllm/gpt-oss-120b-capyctl-gb10/) | vLLM 0.32.0 | GPT-OSS-120B: MXFP4 experts, FP8 attention; no drafter, expandable allocator and lazy loading | 1x GB10, 8 GiB FP8 KV | 45.345 one-stream decode, 112.054 aggregate at 8 streams; 31.735 decode at 110k; GSM8K 239/250, tools 30/30, HumanEval 157/164; retrieval 47/48 (one 111k miss) | 2026-11-22 |
| [vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10](vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/) | vLLM 0.32.0 | Gemma 4 26B-A4B CapyCTL r3: NVFP4 experts, FP8 body/untied head; Google MTP3; default across contexts | 1x GB10, 40 GiB KV grant | Final shipped sweep: 76.36 tokens/s (297.10 at 8 streams); GSM8K 244/250, tools 30/30; long-context retrieval remains unqualified | 2026-11-16–17 |
| [sglang/gemma-4-26b-a4b-capyctl-nvfp4-gb10](sglang/gemma-4-26b-a4b-capyctl-nvfp4-gb10/) | SGLang 0.5.22 | Gemma 4 26B-A4B CapyCTL r3, DFlash8 for short/medium prompts; use vLLM or no drafter for long prompts | 1x GB10, 80 GiB request | Final shipped sweep: 71.69 tokens/s (255.15 at 8 streams); no-draft 38.74 / 37.24 at 96k / 111k; GSM8K 244/250, tools 30/30 | 2026-11-16–17 |
| [vllm/qwen3.8-27b-capyctl-nvfp4-gb10](vllm/qwen3.8-27b-capyctl-nvfp4-gb10/) | vLLM 0.32.0 | Qwen3.8-27B CapyCTL NVFP4 (every large Linear NVFP4, second revision: block scales searched), DFlash2 drafts | 1x GB10 | 33.9 tokens/s (177 tokens/s at 8 streams) | 2026-11-06 |
| [sglang/qwen3.8-27b-capyctl-nvfp4-gb10](sglang/qwen3.8-27b-capyctl-nvfp4-gb10/) | SGLang 0.5.22 | Qwen3.8-27B CapyCTL NVFP4 (every large Linear NVFP4), DFlash2 drafts | 1x GB10, managed limit above 65 GiB | 36.3 tokens/s (166 tokens/s at 8 streams) | 2026-11-04 |
| [custom/basal-2.0-4.5b-gb10](custom/basal-2.0-4.5b-gb10/) | basal-serve 2.2.0 (custom engine) | basal-2.0-4.5B (decision model), bf16 | 1x GB10, 38 GiB | Decisions: 90 ms for one question about a 200-token text; 1.2 decisions/s; ready in about 12 minutes | 2026-10-21 |
| [llamacpp/clef-flash-gb10](llamacpp/clef-flash-gb10/) | llama.cpp v0.7.0 | Clef-Flash (decision model), Q8_0 GGUF | 1x GB10, 18 GiB | Decisions: 203 ms for one question about a 200-token text; 1.3 decisions/s | 2026-10-18 |
| [llamacpp/clef-gb10](llamacpp/clef-gb10/) | llama.cpp v0.7.0 | Clef (decision model), Q8_0 GGUF | 1x GB10, 38 GiB | Decisions: 560 ms; 0.43 decisions/s | 2026-10-18 |
| [llamacpp/kev-4b-gb10](llamacpp/kev-4b-gb10/) | llama.cpp v0.7.0 | Kev-4B (decision model), Q8_0 GGUF | 1x GB10, 28 GiB | Decisions: 88 ms; 1.5 decisions/s | 2026-10-18 |
| [llamacpp/julia-1-gb10](llamacpp/julia-1-gb10/) | llama.cpp v0.7.0 | Julia-1 (decision model), Q8_0 GGUF | 1x GB10, 2 GiB | Decisions: 20 ms; 4.4 decisions/s | 2026-10-18 |
| [sglang/clef-flash-gb10](sglang/clef-flash-gb10/) | SGLang 0.5.22 | Clef-Flash (decision model), BF16 | 1x GB10, 52 GiB | Decisions: 155 ms; 1.6 decisions/s | 2026-10-18 |
| [sglang/clef-gb10](sglang/clef-gb10/) | SGLang 0.5.22 | Clef (decision model), BF16 | 1x GB10, managed limit 108 GiB | Decisions: 395 ms; 0.58 decisions/s | 2026-10-18 |
| [vllm/nemotron-3.5-super-vl-120b-a12b-nvfp4-gb10](vllm/nemotron-3.5-super-vl-120b-a12b-nvfp4-gb10/) | vLLM 0.31.0 | Nemotron 3.5 Super VL 120B-A12B NVFP4, LiLiCorr drafts, 4 running requests | 1x GB10, managed limit 108 GiB | Historical 42.7 tokens/s (70.7 at 4 streams); [needs memory-safety review](comparisons/nemotron-3.5-super-vl-drafters-gb10/#other-rows-and-safety) before reuse | 2026-10-16 |
| [llamacpp/nemotron-3.5-bolt-74b-a9b-24gb-gb10](llamacpp/nemotron-3.5-bolt-74b-a9b-24gb-gb10/) | llama.cpp v0.7.0 | Nemotron 3.5 Bolt 74B-A9B 24GB (2-bit GGUF), LiLiCorr drafts, 262,144-token context | 1x GB10, 35 GiB | 78.8 tokens/s (one request at a time) | 2026-10-16 |
| [llamacpp/nemotron-3.5-bolt-74b-a9b-24gb-8x32k-gb10](llamacpp/nemotron-3.5-bolt-74b-a9b-24gb-8x32k-gb10/) | llama.cpp v0.7.0 | Nemotron 3.5 Bolt 74B-A9B 24GB (2-bit GGUF), 8 slots of 32,768 tokens, no drafts | 1x GB10, 30 GiB | 37.8 tokens/s (97.2 tokens/s at 8 streams) | 2026-10-16 |
| [sglang/qwen3.8-27b-nvfp4-dflash2-mix-gb10](sglang/qwen3.8-27b-nvfp4-dflash2-mix-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4, DFlash2 drafts, mixed chunked prefill, 12 running requests | 1x GB10, managed limit above 75 GiB | 28.6 tokens/s (155 tokens/s at 22 streams) | 2026-10-15 |
| [sglang/qwen3.8-27b-nvfp4-mtp-eagle-gb10](sglang/qwen3.8-27b-nvfp4-mtp-eagle-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4, MTP drafts (EAGLE), 22 running requests | 1x GB10, managed limit 100 GiB | 24.9 tokens/s (337 tokens/s at 22 streams) | 2026-10-15 |
| [sglang/qwen3.8-flash-next-nvfp4-gb10](sglang/qwen3.8-flash-next-nvfp4-gb10/) | SGLang 0.5.21 | Qwen3.8-Flash-Next NVFP4, MTP drafts, n-gram table on disk | 1x GB10, managed limit 110 GiB | 21.3 tokens/s (77.8 tokens/s at 8 streams) | 2026-10-09 |
| [tensorfold/qwen3.8-flash-next-mlx-4bit-gb10](tensorfold/qwen3.8-flash-next-mlx-4bit-gb10/) | TensorFold 0.6.5 | Qwen3.8-Flash-Next MLX 4-bit, MTP drafts, n-gram table on disk | 1x GB10, managed limit 110 GiB | 54.7 tokens/s (122 tokens/s at 8 streams) | 2026-10-09 |
| [vllm/gpt-oss-120b-gb10](vllm/gpt-oss-120b-gb10/) | vLLM 0.30.0 | gpt-oss-120b, MXFP4 experts | 1x GB10, managed limit 96 GiB | 34.2 tokens/s (107 tokens/s at 8 streams) | 2026-10-09 |
| [tensorfold/nemotron-3.5-lightning-30b-a3b-mlx-4bit-32gib-gb10](tensorfold/nemotron-3.5-lightning-30b-a3b-mlx-4bit-32gib-gb10/) | TensorFold 0.6.5 | Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, MTP drafts | 1x GB10, sized for 32 GiB | 137 tokens/s (one request at a time) | 2026-10-07 |
| [vllm/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10](vllm/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Nemotron 3.5 Lightning 30B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 69.9 tokens/s (164 tokens/s at 4 streams) | 2026-10-07 |
| [sglang/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10](sglang/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Nemotron 3.5 Lightning 30B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 69.9 tokens/s (163 tokens/s at 4 streams) | 2026-10-07 |
| [tensorfold/qwen3.6-35b-a3b-mlx-4bit-32gib-gb10](tensorfold/qwen3.6-35b-a3b-mlx-4bit-32gib-gb10/) | TensorFold 0.6.5 | Qwen3.6-35B-A3B MLX 4-bit, MTP drafts | 1x GB10, sized for 32 GiB | 164 tokens/s (368 tokens/s at 8 streams) | 2026-10-07 |
| [vllm/qwen3.6-35b-a3b-nvfp4-32gib-gb10](vllm/qwen3.6-35b-a3b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Qwen3.6-35B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 76.7 tokens/s (180 tokens/s at 4 streams) | 2026-10-07 |
| [sglang/qwen3.6-35b-a3b-nvfp4-32gib-gb10](sglang/qwen3.6-35b-a3b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Qwen3.6-35B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 83.5 tokens/s (186 tokens/s at 4 streams) | 2026-10-07 |
| [vllm/gpt-oss-20b-32gib-gb10](vllm/gpt-oss-20b-32gib-gb10/) | vLLM 0.30.0 | gpt-oss-20b, MXFP4 experts | 1x GB10, sized for 32 GiB | 46.7 tokens/s (195 tokens/s at 8 streams) | 2026-10-07 |
| [sglang/gpt-oss-20b-32gib-gb10](sglang/gpt-oss-20b-32gib-gb10/) | SGLang 0.5.21 | gpt-oss-20b, MXFP4 experts | 1x GB10, sized for 32 GiB | 45.8 tokens/s (191 tokens/s at 8 streams) | 2026-10-07 |
| [vllm/gemma-4-26b-a4b-nvfp4-32gib-gb10](vllm/gemma-4-26b-a4b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Gemma 4 26B-A4B NVFP4 | 1x GB10, sized for 32 GiB | 30.3 tokens/s (105 tokens/s at 4 streams) | 2026-10-07 |
| [sglang/gemma-4-26b-a4b-nvfp4-32gib-gb10](sglang/gemma-4-26b-a4b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Gemma 4 26B-A4B NVFP4 | 1x GB10, sized for 32 GiB | 30.0 tokens/s (103 tokens/s at 4 streams) | 2026-10-07 |
| [vllm/gemma-4-31b-w4a16-32gib-gb10](vllm/gemma-4-31b-w4a16-32gib-gb10/) | vLLM 0.30.0 | Gemma 4 31B QAT INT4 W4A16 | 1x GB10, sized for 32 GiB | 10.8 tokens/s (22.3 tokens/s at 2 streams) | 2026-10-07 |
| [tensorfold/qwen3.8-27b-nvfp4-32gib-gb10](tensorfold/qwen3.8-27b-nvfp4-32gib-gb10/) | TensorFold 0.6.5 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10, sized for 32 GiB | 41.5 tokens/s (73.5 tokens/s at 2 streams) | 2026-10-06 |
| [vllm/qwen3.8-27b-nvfp4-32gib-gb10](vllm/qwen3.8-27b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Qwen3.8-27B NVFP4 | 1x GB10, sized for 32 GiB | 12.5 tokens/s (46.2 tokens/s at 4 streams) | 2026-10-06 |
| [sglang/qwen3.8-27b-nvfp4-32gib-gb10](sglang/qwen3.8-27b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4 | 1x GB10, sized for 32 GiB | 13.0 tokens/s (one request at a time) | 2026-10-06 |
| [sglang/frognano-4b-rtx4090](sglang/frognano-4b-rtx4090/) | SGLang 0.5.21 | FrogNano-4B-2609, bf16 | 1x RTX 4090 Laptop GPU 16 GB | 62.5 tokens/s (381 tokens/s at 7 streams) | 2026-10-04 |
| [tensorfold/nemotron-3.5-lightning-30b-a3b-4bit-gb10](tensorfold/nemotron-3.5-lightning-30b-a3b-4bit-gb10/) | TensorFold 0.6.5 | Nemotron 3.5 Lightning 30B-A3B 4-bit, MTP drafts | 1x GB10 | 135 tokens/s (one request at a time on CUDA) | 2026-10-04 |
| [tensorfold/qwen3.8-27b-nvfp4-gb10](tensorfold/qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.5 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 38.9 tokens/s (195 tokens/s at 8 streams) | 2026-10-04 |
| [vllm/qwen3-4b-gb10](vllm/qwen3-4b-gb10/) | vLLM 0.30.0 | Qwen3-4B, bf16 | 1x GB10 | 22.1 tokens/s (205 tokens/s at 8 streams) | 2026-10-04 |
| [sglang/qwen3-4b-gb10](sglang/qwen3-4b-gb10/) | SGLang 0.5.20 | Qwen3-4B, bf16 | 1x GB10 | 22.4 tokens/s | 2026-10-02 |
| [vllm/qwen3.8-27b-nvfp4-gb10](vllm/qwen3.8-27b-nvfp4-gb10/) | vLLM 0.30.0 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 23.7 tokens/s (141 tokens/s at 8 streams) | 2026-10-04 |
| [sglang/qwen3.8-27b-nvfp4-gb10](sglang/qwen3.8-27b-nvfp4-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 28.2 tokens/s (120 tokens/s at 8 streams) | 2026-10-04 |
| [tensorfold/frognano-4b-mlx-4bit-rtx4090](tensorfold/frognano-4b-mlx-4bit-rtx4090/) | TensorFold 0.6.3 | FrogNano-4B-2609, MLX 4-bit | 1x RTX 4090 Laptop GPU 16 GB | 52.4 tokens/s (319 tokens/s at 8 streams) | 2026-10-03 |
| [vllm/frognano-4b-rtx4090](vllm/frognano-4b-rtx4090/) | vLLM 0.30.0 | FrogNano-4B-2609, bf16 | 1x RTX 4090 Laptop GPU 16 GB | 59.9 tokens/s | 2026-10-03 |

### Qwen3.8-27B NVFP4 on three engines

The same checkpoint (`nvidia/Qwen3.8-27B-NVFP4` at `482ca0f3`), the same
DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c`), the same GB10, the
same CapyCTL commit (`1f2cfc7`) and the same three requests:

| Engine | Decode, DFlash2 drafts | Decode, no drafts | Time to first token, drafts | Peak memory, drafts | Ready, warm |
|---|---|---|---|---|---|
| TensorFold 0.6.5 | 38.9 tokens/s | 11.7 tokens/s | 0.12 s | 30.3 GiB | 14 s |
| vLLM 0.30.0 | 23.7 tokens/s | 11.9 tokens/s | 0.24 s | 51.5 GiB | 102 s |
| SGLang 0.5.21 | 28.2 tokens/s | 12.3 tokens/s | 0.21 s | 68.9 GiB | 253 s |

Without drafts the three decode within 6% of each other. With drafts and
sampling on, decode varies from request to request with the drafts accepted;
each recipe's README has the ranges, the settings, what each engine needed and
a capyctl-bench report with context and concurrency sweeps.

## Comparisons

The same model and machine on two engine versions, on several engines, through
CapyCTL against the same engine launched directly, or across serving configurations.
Each directory states its method and includes measured evidence in `bench/`;
most include deployment files and a capyctl-bench report.

| Comparison | Engine versions | Model | Hardware | Result | Measured |
|---|---|---|---|---|---|
| [comparisons/nemotron-3.5-super-vl-inproj-nvfp4-gb10](comparisons/nemotron-3.5-super-vl-inproj-nvfp4-gb10/) | SGLang 0.5.22 | NVIDIA Nemotron Super VL; selective Mamba in_proj W4A4 study | 1x GB10 + CPU check | Stopped after activation check: about 7.5–13.5 percentage points extra zeros; no conversion or speedup claim | 2026-11-25 |
| [comparisons/qwen3.5-9b-builds-gb10](comparisons/qwen3.5-9b-builds-gb10/) | SGLang 0.5.22 | CapyCTL G32 INT4 / INT8-head C versus cyankiwi/Qwen3.5-9B-AWQ-4bit @ e4d2d39c | GB10; RTX 4090 Laptop GPU 16 GB | No checkpoint published: C is 16–18% faster on the laptop, but HumanEval is 147/164 versus 154/164 and code truncations 10 versus 3 fail the gate; AWQ stays | 2026-11-22–23 |
| [comparisons/nemotron-3.5-super-vl-drafters-gb10](comparisons/nemotron-3.5-super-vl-drafters-gb10/) | SGLang 0.5.22, vLLM 0.32.0/0.31.0 | NVIDIA NVFP4 `9838c915`, no drafter/native MTP/LiLiCorr; historical recipe `579ffa0b` | 1x GB10 | Elected native SGLang MTP: 36.24 vs 20.71 one-stream decode; full task gate passes; retrieval 32k 5/6 vs 6/6, 64k 6/6 vs 5/6; MTP 96k/111k stopped by hard freeze with no recovered log; vLLM not viable within tested memory envelope; separate automatic-KV OOM retained | 2026-11-23 |
| [comparisons/inkling-2-mini-builds-gb10](comparisons/inkling-2-mini-builds-gb10/) | SGLang PR #51926 + FP8 loader patch; pinned vLLM Inkling image | Inkling 2 Mini: CapyCTL FP8 attention versus original weights and @tech2wild's published community standard/max-FP8 recipes | GB10 | Elected FP8 DSpark: 47.31 / 134.13 aggregate tokens/s at 1 / 8 streams; wins 2/4/8 streams and 8k/64k/110k, loses 1 stream and 32k; HumanEval 154 community / 151 original SGLang / 149 FP8; both recipes fail strict retrieval (56/60); one cold-JIT start 882 s, minimum 4.36 GiB available | 2026-11-21 |
| [comparisons/qwen3.6-35b-a3b-builds-gb10](comparisons/qwen3.6-35b-a3b-builds-gb10/) | vLLM 0.32.0/0.33.0, SGLang 0.5.22, llama.cpp v0.7.0/b12242 | Qwen3.6-35B-A3B: CapyCTL DeltaNet/combined, NVIDIA, RedHatAI, Unsloth, Qwen FP8, Intel mixed, GGUF; published GB10 recipes | 1x GB10 | Elected DeltaNet DFlash2: 129.7 / 328.4 aggregate tokens/s at 1 / 8 streams; published NVIDIA MTP4 recipe reproduced at 103.7 / 304.3; faster than published recipes tested at all measured loads/contexts, but loses short c8 to NVIDIA with our DFlash2 settings; retrieval 60/60 | 2026-11-20–21 |
| [comparisons/gpt-oss-20b-builds-gb10](comparisons/gpt-oss-20b-builds-gb10/) | SGLang 0.5.22, vLLM 0.32.0/0.33.0, llama.cpp b12242 | GPT-OSS-20B: CapyCTL MXFP4-FP8, OpenAI, LMSYS, RedHatAI, public NVFP4 and GGUF; DFlash/EAGLE3 alternatives | 1x GB10 | CapyCTL no-drafter vLLM: 57.77 / 196.71 tokens/s at 1 / 8 streams; OpenAI's checkpoint on SGLang 0.5.22 with 131k-context serving settings, no drafter 44.85 / 178.67; llama.cpp leads one-stream decode at 76.16 but has one retained repetition/cap failure; no drafter passes both task and long-context requirements | 2026-11-18–19 |
| [comparisons/gpt-oss-120b-builds-gb10](comparisons/gpt-oss-120b-builds-gb10/) | vLLM 0.32.0 and 0.30.0 | CapyCTL MXFP4-FP8 versus matched OpenAI and published OpenAI recipe | 1x GB10 | Matched 0.32: 45.345 vs 34.796 one-stream decode; 112.054 vs 104.849 c8 aggregate; identical full task scores, retrieval 47/48 vs 48/48; allocator loading diagnosis retained | 2026-11-22 |
| [comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10](comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/) | SGLang 0.5.22, vLLM 0.32.0 | Gemma 4 26B-A4B: CapyCTL r1/r2/r3/r3e, NVIDIA, RedHat, Unsloth and BF16; MTP, DFlash, DSpark, EAGLE3 | 1x GB10 | Final shipped r3 vLLM MTP3: 76.36 / 297.10 tokens/s at 1 / 8 streams; NVIDIA's NVFP4 build on SGLang 0.5.22 with its 131k-context serving settings, no drafter: 29.47 / 167.67; offline BF16 agreement 97.177% / 97.132%; neither build passes all-document 111k retrieval | 2026-11-16–17 |
| [comparisons/qwen3.8-27b-fanout-vllm-rtx-pro-6000](comparisons/qwen3.8-27b-fanout-vllm-rtx-pro-6000/) | vLLM 0.32.0 | Qwen3.8-27B CapyCTL NVFP4, DFlash2 drafts; one coding task fanned out to 8 lanes against one stream | 1x RTX PRO 6000 Blackwell (Confidential Computing) | 8-lane fan-out 849 tokens/s combined with the memory a 32 GB card fits, 1,566 with a 16 GiB KV cache, 310 for one stream; lanes left to finish wrote 2 to 3 times the tokens, so the finished page came sooner only with 16 GiB (18 to 20 s against 31 s); no page ran in either mode; vLLM needs UVA off on a CC guest | 2026-11-11 |
| [comparisons/qwen3.8-27b-nvfp4-builds-gb10](comparisons/qwen3.8-27b-nvfp4-builds-gb10/) | SGLang 0.5.22, vLLM 0.32.0, exllamav3 (MiaAI-Lab fork) | Qwen3.8-27B: CapyCTL NVFP4 against NVIDIA, gittensor, QUASAR, unsloth NVFP4, RedHatAI INT4, MiaAI-Lab EXL3 and the BF16 source; with and without DFlash2 | 1x GB10 | CapyCTL NVFP4 (every large Linear NVFP4) fastest without drafts (14.9 against NVIDIA's 12.5 tokens/s on one stream) and with DFlash2 on vLLM (36.1 and 181 tokens/s at 1 and 8 streams, +23% and +13% against NVIDIA's on the same settings); every build within noise on GSM8K and tool calls; vLLM's bf16 DeltaNet state cache and MiaAI-Lab's SGLang flags the settings that moved; past 8 streams vLLM peaks at 390 tokens/s (32 streams, 32 GiB KV cache) and SGLang at 329, where CapyCTL's 32-in-flight bound ends the curve; the checkpoint's second revision (searched block scales) agrees with BF16 more often (93.00% against 92.63% of greedy next tokens) at the same accuracy, 1% to 5% slower on vLLM and up to 13% on SGLang | 2026-11-03–06 |
| [comparisons/concurrency-batching-qwen3-gb10](comparisons/concurrency-batching-qwen3-gb10/) | SGLang 0.5.22, vLLM 0.32.0 | Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B BF16 | 1x GB10 | 313 attempts; doubling short-prompt concurrency roughly doubled throughput in screening; 12 operating points confirmed three times; experimental admission builds, no default change | 2026-10-26–27 UTC |
| [comparisons/native-tp2-direct-vs-capyctl-qwen3-30b-a3b-a6000](comparisons/native-tp2-direct-vs-capyctl-qwen3-30b-a3b-a6000/) | vLLM 0.32.0, SGLang 0.5.22; direct vs CapyCTL | Qwen3-30B-A3B, native TP2 | 2x RTX A6000 48 GB | Matched concurrency 1/4/8: CapyCTL throughput changes -5.76% to +3.22% versus direct launch in this sampled run; separate deep/host-backed parking evidence | 2026-10-27 UTC |
| [comparisons/tensorfold-0.6.1-vs-0.6.2-nemotron-3.5-lightning-gb10](comparisons/tensorfold-0.6.1-vs-0.6.2-nemotron-3.5-lightning-gb10/) | TensorFold 0.6.1, 0.6.2 | Nemotron 3.5 Lightning 30B-A3B 4-bit, MTP drafts | 1x GB10 | Identical outputs; one-stream speed within 1% (one stream only: neither version batches this model on CUDA) | 2026-10-02 |
| [comparisons/tensorfold-0.6.1-vs-0.6.2-qwen3.8-27b-nvfp4-gb10](comparisons/tensorfold-0.6.1-vs-0.6.2-qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.1, 0.6.2 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | Identical outputs; 0.6.2 0.5% to 2.0% higher aggregate at 1 to 8 streams; context sweep to 128k within about 1% | 2026-10-02 |
| [comparisons/tensorfold-0.6.2-vs-0.6.3-qwen3.8-27b-nvfp4-gb10](comparisons/tensorfold-0.6.2-vs-0.6.3-qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.2, 0.6.3 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | Quick check (1, 4, 8 streams; 2k, 32k, 128k): identical outputs; throughput and decode within 1% at every point | 2026-10-02 |
| [comparisons/tensorfold-0.6.3-vs-0.6.5-qwen3.8-27b-nvfp4-gb10](comparisons/tensorfold-0.6.3-vs-0.6.5-qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.3, 0.6.5 | Qwen3.8-27B NVFP4, DFlash2 drafts; Nemotron 3.5 Lightning one stream | 1x GB10 | Quick check (1, 4, 8 streams; 2k, 32k, 128k): throughput and prefill within 0.4% except one slower one-stream round; identical outputs except the 32k and 128k replies (0.6.4 attention sums); Nemotron one stream 3.4% faster | 2026-10-04 |
| [comparisons/qwen3.8-27b-nvfp4-three-engines-gb10](comparisons/qwen3.8-27b-nvfp4-three-engines-gb10/) | TensorFold 0.6.3, vLLM 0.30.0, SGLang 0.5.21 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 8 running requests on each: TensorFold highest aggregate at 1 to 8 streams (194 against 136 and 122 tokens/s at 8) and fastest prefill; SGLang lowest time to first token at 5, 6 and 8 streams; vLLM fastest one-stream decode at 2k and 256k and flat ~51 GiB memory | 2026-10-03 |
| [comparisons/frognano-4b-nvfp4-three-engines-gb10](comparisons/frognano-4b-nvfp4-three-engines-gb10/) | TensorFold 0.6.3, vLLM 0.30.0, SGLang 0.5.21 | FrogNano-4B-2609 NVFP4, no drafts | 1x GB10 | 8 running requests on each: vLLM and SGLang 10% to 18% higher decode at 1 to 8 streams (414 and 416 against 352 tokens/s at 8); TensorFold fastest prefill, lowest time to first token at 1 to 5 streams, least memory (10 GiB at 8 streams) and a 5 s warm start | 2026-10-04 |
| [comparisons/qwen3.6-35b-a3b-three-engines-gb10](comparisons/qwen3.6-35b-a3b-three-engines-gb10/) | TensorFold 0.6.5, vLLM 0.30.0, SGLang 0.5.21 | Qwen3.6-35B-A3B: TensorFold MLX 4-bit with MTP drafts, vLLM and SGLang NVFP4 without drafts | 1x GB10 | The three 32 GiB recipes as served (vLLM and SGLang run 4 requests, TensorFold 8): TensorFold about twice the one-stream decode (162 against 77 and 84 tokens/s) and 368 against 180 and 187 tokens/s at 8 streams; vLLM fastest prefill and time to first token up to 2k; TensorFold least memory | 2026-10-07 |
| [comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10](comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4: DFlash2 drafts against MTP/EAGLE drafts; BF16 and NVFP4 LM heads; five variants between them | 1x GB10 | No single sweet spot: DFlash2 for 1 to 4 users (28 against 24 tokens/s at one stream), MTP/EAGLE at 12 running for about 8 to 12 (163 tokens/s at 8), at 22 running for 16 to 22 (344 tokens/s at 22); mixed chunks cut DFlash2's long-prompt stall from 56 s to 9 s; CapyCTL matches SGLang launched directly; the NVFP4 LM head 41% faster at one stream with GSM8K 93.2% against 93.6%; every variant parks and wakes | 2026-10-15 |
| [comparisons/nemotron-3.5-super-vl-engines-gb10](comparisons/nemotron-3.5-super-vl-engines-gb10/) | vLLM 0.31.0, llama.cpp v0.7.0, SGLang 0.5.21 | Nemotron 3.5 Super VL 120B-A12B: NVFP4 on vLLM, NVFP4 and Q4_0 GGUF on llama.cpp; LiLiCorr, MTP and no drafts | 1x GB10 | vLLM wins: 34 to 45 tokens/s for one user (1k to 32k prompts) and about 3,000 tokens/s of prompt processing, against 26 to 37 and about 800 on llama.cpp (a 32k prompt: 11 s against 40 s to the first token); LiLiCorr beats MTP on vLLM; 4 running requests cost one user nothing; SGLang 0.5.21 does not start it; deep parking fails CapyCTL's wake canary, so the recipe restarts instead | 2026-10-16 |
| [comparisons/decision-models-gb10](comparisons/decision-models-gb10/) | llama.cpp v0.7.0, SGLang 0.5.22 | Six decision models: Clef, Clef-Flash, Kev-4B, lev, Julia-1, Laya | 1x GB10 | Clef-Flash the pick: 15 GiB and 203 ms on llama.cpp, within 3 points of Clef 27B on every test; published scores reproduced (Clef 91.9% API-Bank, Julia-1 73.0% Typed Decisions); Julia-1 20 ms but narrow and overconfident; SGLang reads 8k inputs twice as fast for 3x the memory; more users add no throughput; CapyCTL adds 12 to 26 ms per request | 2026-10-18 |

## Contribute one

Open a pull request that adds `<engine>/<model>-<gpu>/` with a
`deployment.yaml`, a `README.md` and a capyctl-bench report, and a row in the
Recipes table. A recipe is accepted only if:

- it was run through CapyCTL, not against the engine directly;
- it includes a [capyctl-bench](tools/capyctl-bench/) report in `bench/`: the
  results JSON from `capyctl_bench.py run` and the `report.html`, `summary.md`
  and `data.csv` that `capyctl_bench.py report` renders from it (the PNG charts
  are optional);
- its README has the numbers below, measured on the hardware it names;
- it names the exact CapyCTL version (or commit), engine version, model
  revision and drafter revision it was measured with, and the date.

The numbers each recipe reports:

| Number | How |
|---|---|
| Ready time, cold | `capyctl start deployment <name> --wait` after `deploy model`, weights already in the model store |
| Ready time, warm | the same command again after `capyctl stop deployment <name>` has finished |
| Time to first token | streaming chat completion through the CapyCTL inference endpoint, request sent to first content or reasoning token |
| Decode tokens/s, one stream | the same request, (completion tokens - 1) / (last token time - first token time) |
| Peak memory | the peak CapyCTL measured (`capyctl status deployment <name> --json`, `startup.measured`), next to the reservation the deployment holds |

Use three different prompts, `max_tokens: 512`, one request at a time, and
report the median. The capyctl-bench report adds the context sweep and the
concurrency curves; its README explains each metric and the rules that keep
two reports comparable. State the thinking setting and sampling parameters. Commands
and outputs in the README are copied from the run; shorten paths to
`/home/me/...` and replace the machine's name.

A decision model (`api: decisions`, answering `POST /v1/systemone`) generates
no tokens, so its recipe reports these numbers instead, from
`capyctl_bench.py run --decisions --accuracy` through CapyCTL with the bench
defaults (100 timed runs per point; `--plan` prints the request count and an
estimated time first):

| Number | How |
|---|---|
| Ready time, cold and warm | as above |
| Latency p50 and p95, per question | one request at a time, client side, request sent to the whole answer read, at 1, 3 and 5 questions and about 200, 2k and 8k state tokens; per question is the amortized cost |
| Cache | a repeated state against new states, p50 and p95 |
| Requests/s and questions/s | closed loop with 1, 2, 4 and 8 users, each request its own state, with latency and error rate |
| Peak memory | as above, CapyCTL's `startup.measured` next to the reservation |
| Quality | Typed Decisions (all 400 test cases: accuracy, KL, Brier, ECE), the Decision Index components API-Bank, GPQA Diamond and Home appliances (score, coverage, skill; single components, not an index), and AG News and BoolQ (500 rows each, seed 0: accuracy, macro-F1, Brier, NLL, ECE) |
| Revisions | the model revision, and the dataset revisions and row hashes the report lists |

State the engine's decision settings and any `--extra-body`.

A custom engine (one CapyCTL starts from a manifest, registered with
`capyctl engine add --custom`; CapyCTL 0.1.6 or later) goes in
`custom/<model>-<gpu>/`, and its recipe also includes:

- the manifest, as `manifest.yaml`, exactly as it was registered, and the
  `capyctl engine add --custom <executable> --name <profile> --manifest
  manifest.yaml` command with what it printed; the deployment's `engine:` is
  that profile name;
- the engine's exact version (the manifest's `version` and the package
  version installed), and how it was installed;
- the root CapyCTL fingerprinted and its digest, as `engine add` printed them
  (`pinned root` and `installation`), so a reader can tell whether their
  installation matches.

Every pull request runs `capyctl validate config --file` on each
`deployment.yaml` with the latest CapyCTL release. That check is offline: it
does not prove the recipe runs.

## License

Apache-2.0, as CapyCTL. See [LICENSE](LICENSE).
