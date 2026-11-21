---
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-35B-A3B
  - nvidia/Qwen3.6-35B-A3B-NVFP4
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - qwen3_6
  - nvfp4
  - capyctl
---

# Qwen3.6-35B-A3B-NVFP4

CapyCTL's DeltaNet-only conversion of [NVIDIA's NVFP4 build](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), derived from [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). **NVIDIA's NVFP4 experts and head are retained; DeltaNet's large projections become searched W4A16 NVFP4; full attention stays FP8.** Elected on one GB10 with vLLM 0.32.0, DFlash2 block 8 and CapyCTL 0.1.12.

Published weights revision: `cb64dfd952ed32865b3a7420d02d730522f498d8`. NVIDIA source: `1355db6a052410cfd62085d94b58866fd0f2c3c5`; Qwen BF16 source: `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.

## What changed and why

| Family | This checkpoint |
|---|---|
| Routed/shared experts and output head | NVIDIA NVFP4 codes and scales, unchanged |
| DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj` | BF16 source weights converted to W4A16 NVFP4 with searched block scales |
| Full-attention projections | NVIDIA FP8, unchanged |
| MTP, routers, small projections, convolution, norms, embedding and vision | Retained from NVIDIA's build, byte-identical |

The conversion targets weight traffic without requantizing the already-NVFP4 experts. Each block of 16 searches scale multipliers from 0.60 to 1.10 in steps of 0.025, FP8-rounding scales before minimizing squared weight error. Fused qkv/z modules share a compatible global scale. ModelOpt supplies the NVFP4 packing/global-scale convention. Obsolete FP8 input scales are removed only for changed W4A16 matrices. There is no new activation calibration. DeltaNet's recurrent arithmetic is **not** claimed to run in NVFP4.

| Layout | [INFERENCE] logical bytes/token | Offline BF16-answer-ID agreement |
|---|---:|---:|
| NVIDIA | 2,254,641,880 | 19,506/20,102 (97.0351%) |
| **This checkpoint** | **1,814,239,600** | **19,398/20,102 (96.4979%)** |
| Combined alternative, full attention also NVFP4 | 1,694,963,920 | 19,356/20,102 (96.2889%) |

[INFERENCE] Logical traffic sums stored active tensor payloads, routing 8/256 experts and fetching one embedding row. It excludes inactive MTP/vision, KV/state/activation traffic and cache reuse. This is payload arithmetic, **not measured DRAM bandwidth**. [Traffic and agreement evidence](https://github.com/edurdias/capyctl-recipes/blob/main/comparisons/qwen3.6-35b-a3b-builds-gb10/tensors/results/family-ablation.json).

Offline agreement reconstructs weights into BF16 on the same 200 prompts and 20,102 teacher-forced BF16 answer positions, with thinking disabled. Of 200 continuations, 124 reach the diagnostic 128-token cap. This is not task accuracy or an emulation of engine activation/KV quantization. Saved-file scores match the virtual ablations. **CPU format checks and offline agreement do not qualify an engine.**

## Measured on one GB10

Default thinking, temperature zero, normal EOS, one warm-up and three measured rounds. Short throughput uses a 512-total-token cap including reasoning; all short columns below are aggregate completion tokens / round wall time, even one stream. Long context is post-first-token decode, three rounds per exact input.

| Setup | Short 1 stream | Short 8 streams | Decode 32k / 110k | GSM8K | Tools | HumanEval |
|---|---:|---:|---|---:|---:|---:|
| **This checkpoint, vLLM 0.32 DFlash2** | **129.7** | **328.4** | **137.6 / 130.6** | **234/250** | **30/30** | **159/164** |
| NVIDIA, same DFlash2 setup | 119.0 | 336.5 | 135.0 / 114.3 | 229/250 | 30/30 | 156/164 |
| NVIDIA published GB10 vLLM MTP4 recipe, reproduced on 0.32 | 103.7 | 304.3 | 101.8 / 90.0 | 231/250 | 30/30 | 159/164 |
| Combined alternative, same DFlash2 setup | 127.1 | 338.7 | 133.0 / 114.0 | 231/250 | 30/30 | 154/164 |

Tokens/s throughout. This checkpoint beats the published recipes tested here, but loses eight-stream short throughput to NVIDIA on our DFlash2 settings. The combined checkpoint is a measured alternative, not the election: lower HumanEval and losses to NVIDIA DFlash2 at 32k/110k. Small sample differences do not establish an accuracy advantage.

Full gates use 32768 total output tokens, unchanged thinking, truncations wrong, zero HTTP errors and reviewed outputs. Accuracy must meet the baseline's ordinary two-sided 95% Wilson lower endpoint; truncation rate must not exceed its upper endpoint. Tools require 30/30. This checkpoint has GSM8K/tools/HumanEval truncations 10/0/1 and passes. Screens elsewhere are not full qualification.

Retrieval passes 60/60 all-three-code observations and 180/180 codes across six seeded documents, original/one-line variants, at 32k/64k/96k/111k/220k original input tokens; ordered exact extraction also passes. Strict copy is 100/100 on 2000-token documents, Wilson95 96.30–100%. The baseline misses one 96k observation at the cap. These finite tests do not guarantee arbitrary 262k accuracy. DFlash2 accepted lengths at 8k/32k/64k/110k are 3.513/3.770/3.784/4.231, with draft-token acceptance 35.89%/39.57%/39.77%/46.16%.

The separate shipped-file run gives 128.6/336.3 short aggregate tokens/s at 1/8 streams and 131.2 one-stream decode. Its single 110k request gives 140.3 decode tokens/s; do not substitute that observation for the three-round ranking medians.

## Run with CapyCTL

[Install CapyCTL](https://github.com/edurdias/capyctl) 0.1.12 and vLLM 0.32.0 in a CUDA-capable environment, then use the [pinned deployment](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/qwen3.6-35b-a3b-capyctl-nvfp4-gb10):

```bash
capyctl engine add /home/me/vllm-0.32.0-venv
capyctl start standalone
# In another terminal, from the recipe directory:
capyctl deploy model --file deployment.yaml
capyctl start deployment qwen3.6-35b-a3b-capyctl-vllm --wait
curl -s "$CAPYCTL_INFERENCE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-capyctl-vllm","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":512}'
```

Set `CAPYCTL_INFERENCE_URL` from your controller's inference endpoint and use its configured authentication. Commands are examples, not captured terminal output.

The deployment pins the DFlash2 revision, requests 24 GiB FP8 KV (not total memory), 262144 context, eight running requests, LMCache off and restart-only residency. Configure host admission for available unified memory; the KV grant alone is not a minimum-memory recommendation. The shipped file was run with only model sources replaced by audited local copies. Offline schema validation is not host resolution, Hub provisioning or engine qualification. Hub materialization was not tested; the weights repository remains private until the recipe PR merges, and links use its final public URL.

## Plain vLLM

This exposes equivalent recipe flags; the elected measurements above went through CapyCTL, not this example. Download both pinned repositories first:

```bash
hf download capyctl/Qwen3.6-35B-A3B-NVFP4 \
  --revision cb64dfd952ed32865b3a7420d02d730522f498d8 \
  --local-dir /home/me/models/qwen36-capyctl
hf download incoai/Qwen3.6-35B-A3B-DFlash2 \
  --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 \
  --local-dir /home/me/models/qwen36-dflash2
/home/me/vllm-0.32.0-venv/bin/vllm serve /home/me/models/qwen36-capyctl \
  --tensor-parallel-size 1 --max-model-len 262144 --max-num-seqs 8 \
  --language-model-only --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 25769803776 --max-num-batched-tokens 8192 \
  --attention-backend FLASHINFER --moe-backend marlin \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --enable-prefix-caching --async-scheduling \
  --speculative-config '{"method":"dflash","model":"/home/me/models/qwen36-dflash2","num_speculative_tokens":7}'
```

## Rebuild and audit

[Conversion scripts, manifests and native load receipts](https://github.com/edurdias/capyctl-recipes/tree/main/vllm/qwen3.6-35b-a3b-capyctl-nvfp4-gb10/convert) retain the packing rule, preserved-tensor hashes and native dense-kernel compatibility checks. The [comparison](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/qwen3.6-35b-a3b-builds-gb10) contains every measured published build/recipe, failed gates and full methods. vLLM load receipts record native Marlin W4A16 dense execution and actual FP8 E4M3 KV, not BF16 dequantization as the serving path.

## License and attribution

Apache-2.0 derivative of **Qwen's Qwen3.6-35B-A3B** and **NVIDIA's NVFP4 build**, not an official Qwen or NVIDIA release. Retain their licenses, copyright/attribution notices and any applicable NOTICE contents when redistributing. See the [Qwen model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [NVIDIA model card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) for upstream intended uses and limitations. No endorsement or trademark rights are implied.

**Modifications by CapyCTL:** DeltaNet large projections quantized to searched NVFP4 W4A16; corresponding obsolete FP8 input scales removed; quantization metadata, shard index, manifests and serving recipe updated. Untargeted tensors are unchanged.

## Limitations

- Only GB10 performance and native task quality were measured. Other GPUs, engine versions, vision/multimodal accuracy and arbitrary long-context applications are unqualified.
- Short compatibility probes on vLLM 0.33 and SGLang 0.5.22 are not full quality qualification. SGLang cold FlashInfer JIT needed `MAX_JOBS=2` after an OOM.
- Quantization is lossy. Weight-only offline agreement does not measure native activation quantization, KV behavior, speculative decoding or capability equivalence.
- Retrieval/copy and task gates cover finite corpora, not safety, factuality or universal tool correctness. Truncated reasoning remains a failure.
- CapyCTL left an owned engine behind after the shipped-file run; confirmed owned processes were separately terminated and cleanup recorded. Parking/waking is not qualified.
