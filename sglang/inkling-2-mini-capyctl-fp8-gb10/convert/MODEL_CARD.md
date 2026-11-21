---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
library_name: sglang
pipeline_tag: image-text-to-text
tags:
  - inkling2_loop
  - nvfp4
  - fp8
  - capyctl
---

# Inkling-2-Mini-NVFP4-FP8

Selective FP8-attention derivative of [Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). **Native NVFP4/MXFP8 experts, the BF16 output head, DSpark draft and tokenizer remain byte-identical. Exactly 144 attention projections become E4M3 FP8 with per-output-channel FP32 scales.** This is not a full-model FP8 conversion.

Weights revision: **`f3a7d6f69777e2713adaffbda1460581b1dabf60`** in [capyctl/Inkling-2-Mini-NVFP4-FP8](https://huggingface.co/capyctl/Inkling-2-Mini-NVFP4-FP8). Source revision: **`ed40e7da625f594fc9c4990f5f66be24873c22d7`**. The pin includes the corrected tensor index; later card-only commits do not replace it. The repository is private until the recipe PR merges; links use the final public URL.

## What changed and why

The model has 24 physical layers and 48 executed visits: the central 12 layers are visited three times. Sharing stored weights does not remove those repeated weight reads. Selective attention quantization targets that repeated dense traffic without requantizing the model's already-low-bit experts or changing the head shared with DSpark.

| Family | This checkpoint |
|---|---|
| `wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `gate_attn`, `wo_ud` in all 24 attention layers | 144 E4M3FN matrices, with 144 new FP32 `[output_channels, 1]` dequantization scales |
| Experts in layers 0–19 / 20–23 | Original NVFP4 / MXFP8 codes and scales, byte-identical |
| Output head, embeddings, vision, routers, loop controllers, sinks, convolutions, relative-logit basis and norms | Untargeted tensor payloads preserved |
| Bundled `mtp.safetensors` and tokenizer files | Unchanged |

The converter uses per-row maximum absolute value scaling and round-to-nearest E4M3, not activation calibration or a searched quality objective. The engine uses the existing compressed-tensors FP8 linear method with dynamic per-token FP8 activations and BF16 outputs into unchanged attention/loop kernels. A small sink projection is split into its own BF16 linear. The loader patch retains exact dtype, unknown/duplicate tensor and completeness checks. The obsolete top-level quantization stub is removed so it cannot shadow the complete hybrid groups in `hf_quant_config.json`.

The main shard is **18,882,995,072 bytes**, versus **19,526,019,168** for the source. All **1,100 untargeted tensor payloads** were copied and independently read back against source hashes. The repaired index maps all **1,388 tensors**, including the new scales, and declares **18,882,827,424 tensor payload bytes**; this is not the header-inclusive file size.

| File | SHA-256 |
|---|---|
| Main shard | `b7fbb2591af534e7124254af7f4c5e92a9909ec555a48208b4b75b11969b2b5d` |
| Unchanged draft | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Repaired index | `299eec2ea8fd903cd2cafe74c3c09e5baa6161b5c85359b5f911712a32da7f71` |

### Logical bytes per token — [INFERENCE]

| Scope | Source bytes/output token | This checkpoint bytes/output token |
|---|---:|---:|
| Selected attention projections, counting 48 visits and FP32 channel scales | 2,576,351,232 | 1,290,264,576 |
| Selected attention + top-8/128 experts + output head subtotal | 5,342,039,136 | 4,055,952,480 |

**[INFERENCE]** Saving is **1,286,086,656 bytes/output token** for batch-one plain decode. These are shape-derived logical weight streams, **not measured DRAM traffic, achieved bandwidth or a speed prediction**. The scoped subtotal excludes untargeted dense/small tensors; both rows exclude KV/activation traffic, padding, kernel rereads, cache reuse and DSpark amortization. The looped core is counted three times, not once. [Converter and per-tensor manifest](https://github.com/edurdias/capyctl-recipes/tree/main/sglang/inkling-2-mini-capyctl-fp8-gb10/convert).

## Measured on GB10

The elected setup is native SGLang PR #51926 at `92e894ef904bfdb5c0cf2f2a541cbd4eec023573` plus the supplied loader patch, DSpark **gamma 7 / block 8**, BF16 KV, memory share **0.80**, `MAX_JOBS=2`. It has no context override: the model's default limit is **1,048,576 tokens**, not a measured million-token capability. Retrieval was checked through 220k and decode through 110k; nothing beyond 220k was tested on this setup.

The baseline is [@tech2wild's published community recipe](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), standard tier `tv-std`, using original weights and its pinned vLLM image. It caps context at **262,144**, uses BF16 KV, DSpark seven speculative tokens and probabilistic draft sampling. Its max-FP8 tier `tv-fp8` uses FP8 KV and a 1,048,576 context limit. These are native recipe measurements, not CapyCTL-managed measurements. Operational pins/deviations and every failed row are in the [comparison](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/inkling-2-mini-builds-gb10).

Default high effort (0.9), temperature zero, normal EOS, fixed chat/code prompts, one warm-up and three measured rounds. Short columns are **aggregate native completion tokens / round wall span**, including one stream. Long decode excludes TTFT. Both use a 512-total-output-token cap including reasoning.

| Setup | Short aggregate tokens/s, 1 / 2 / 4 / 8 streams | Decode tokens/s, 8k / 32k / 64k / 110k | GSM8K | Tools | HumanEval |
|---|---|---|---:|---:|---:|
| **This checkpoint, patched SGLang DSpark** | **47.31 / 66.59 / 89.06 / 134.13** | **42.67 / 40.57 / 40.99 / 32.72** | **240/250** | **28/30** | **149/164** |
| Original weights, SGLang DSpark | 46.42 / 62.35 / 83.42 / 128.53 | 38.21 / 36.45 / 34.89 / 33.12 | 49/50 screen only | 30/30 screen only | 151/164 |
| Published community standard, reproduced (`tv-std`) | 48.58 / 62.51 / 84.34 / 129.86 | 41.42 / 40.79 / 34.75 / 31.92 | 241/250 | 28/30 | 154/164 |
| Published community max-FP8, reproduced (`tv-fp8`) | 45.69 / 59.29 / 82.06 / 128.82 | 40.55 / 39.24 / 34.72 / 39.93 | 48/50 screen only | 29/30 screen only | 46/50 screen only |

The election equally weights four short aggregate and four exact-context decode ratios against `tv-std`, using their geometric mean: **1.0431** for this checkpoint with DSpark. It wins 2/4/8 streams and 8k/64k/110k decode, but loses one stream (**−2.61%, about −3%**) and 32k (**−0.55%, about −1%**). It also loses short eight-stream TTFT (0.447 versus 0.384 s). Small point-estimate differences are not significance or equivalence claims.

Full gates use a common **32,768-total-token cap**, default high effort, concurrency eight, truncations counted wrong, zero HTTP errors and reviewed output. Every task must meet the baseline's ordinary two-sided 95% Wilson lower accuracy endpoint and upper truncation endpoint; tools additionally allow no new failure type. This checkpoint passes those declared gates with task truncations **2/0/12** versus baseline **2/0/7**. Passing this relative gate does not mean equal quality.

### HumanEval attribution

| Setup, full 164 cases | Passed | Truncations |
|---|---:|---:|
| Published community standard, original weights | 154 | 7 |
| SGLang DSpark, original weights | 151 | 10 |
| SGLang DSpark, FP8 attention | 149 | 12 |

**[INFERENCE] Both the execution stack and the FP8 path contribute to the observed gap.** Original-weight SGLang already loses three passes versus the community recipe; FP8 loses two more under the same SGLang settings. The entire five-pass gap cannot be assigned to FP8. This is one greedy cohort, not a statistically proven causal split. The engine comparison also retains its earlier machine/runtime assignment. [Attribution evidence](https://github.com/edurdias/capyctl-recipes/blob/main/comparisons/inkling-2-mini-builds-gb10/bench/humaneval-attribution.json).

Both SGLang full-code results retain all 50 original screen cases, including failures, and request only the remaining 114; per-observation `launch_run` and continuation receipts preserve provenance. No code failure was rerun. Full FP8 math likewise retains its screen and adds only the remaining 200 cases; tools are not repeated.

### Retrieval and cold start

Both the elected setup and `tv-std` score **56/60 all-three-code retrieval and 56/60 ordered exact extraction**, so **both fail the strict all-observation completeness gate**. All 30 original variants pass; four one-line variants reach the output cap. Original lengths are 32k/64k/96k/111k/220k, six documents and two variants each. Strict copy is **100/100** for both on 2000-token documents. These finite tests do not establish arbitrary long-context accuracy.

One elected fresh-JIT start at 0.80 and `MAX_JOBS=2` reached readiness in **881.58 s (about 882 s)**, with **4.36 GiB minimum MemAvailable** and 22.50 GiB available at ready. The complete isolated JIT cache was moved aside then restored; filesystem page cache was not dropped. This is not a cold-machine/download result or qualification of plain-mode cold startup. An earlier original-weight plain start at 0.80 was safety-stopped below 3 GiB.

## How to run

Use the [native recipe](https://github.com/edurdias/capyctl-recipes/tree/main/sglang/inkling-2-mini-capyctl-fp8-gb10), including its exact measured installer, pinned source and **required three-file patch (+81/−13)**. An unpatched release or unpatched PR is not supported by this checkpoint. The recipe documents applying the patch to both the isolated source and installed package, then downloading the explicit runtime file list at the weights pin.

```bash
# After following the native recipe's installer and patch steps:
hf download capyctl/Inkling-2-Mini-NVFP4-FP8 \
  --revision f3a7d6f69777e2713adaffbda1460581b1dabf60 \
  --local-dir /home/me/models/Inkling-2-Mini-capyctl-fp8attn
# From the recipe directory, set INKLING_BIND to your loopback bind value:
bash launch.sh
```

The launch preserves FA4, BF16 compute/KV, page size 128, xgrammar, multimodal enabled, reasoning parser `inkling`, tool parser `inkling2`, all four DSpark optimization variables, static share 0.80 and `MAX_JOBS=2`. Keep the default high reasoning effort when reproducing numbers. The model's template does not use `enable_thinking:false`.

This is **not a managed CapyCTL deployment**. CapyCTL 0.1.12's candidate passed schema validation only; its reserved memory-fraction policy and missing loop-aware context fitting do not express this measured native setup. Selecting the PR interpreter via `local_engine.sglang` does not qualify the controller path. No parking/waking qualification is claimed.

## Rebuild and evidence

The recipe's `convert/convert-fp8-attention.py` and `convert/fp8-attention-manifest.json` are byte-identical copies of the measured converter and audit. [Comparison evidence](https://github.com/edurdias/capyctl-recipes/tree/main/comparisons/inkling-2-mini-builds-gb10) retains the original copies, tensor-index regression, kernel traces, launch receipts, every gate failure and publication audit. Real GB10 traces show SM120-family CUTLASS dense E4M3×E4M3 GEMM with FP32 accumulation and BF16 output on SM121; this is not a BF16-dequantized serving path.

## License and attribution

**Apache-2.0 derivative of Thinking Machines' Inkling 2 Mini**, not an official Thinking Machines release. Retain upstream license, copyright/attribution and applicable NOTICE contents when redistributing. Consult the [upstream model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for intended uses, limitations and its linked acceptable-use policy. No endorsement or trademark rights are implied.

Modifications by CapyCTL: selective attention E4M3 quantization and FP32 scales, hybrid precision metadata, repaired tensor index, conversion manifest and serving recipe. Untargeted tensor payloads remain unchanged.

## Limits

- **Only GB10 was measured.** Other hardware, engine versions, vision/multimodal quality and general million-token operation are unqualified.
- **DSpark gamma 7 / block 8 only.** Gamma 2 fails the model's exact-size guard; gamma 4 is [INFERENCE] excluded by that guard and was not redundantly launched.
- HumanEval is lower than the community baseline and original-weight SGLang; the baseline-relative gate is not proof of quality equivalence.
- Both recipes fail strict retrieval completeness; the default context limit is not a quality claim.
- CPU format checks, teacher-forced/offline agreement and load-only checks do not qualify engine quality or performance.
- Native lifecycle only; no managed CapyCTL, parking/waking or controller/native equivalence claim.
