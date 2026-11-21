# Inkling 2 Mini CapyCTL FP8 attention on GB10

**Use the CapyCTL checkpoint with the patched SGLang PR build and DSpark gamma 7 / block 8. This is a native SGLang recipe, not a CapyCTL-managed deployment.** It wins six of eight measured throughput points against [@tech2wild's published community recipe](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), but loses one-stream throughput and 32k decode. [Full comparison](../../comparisons/inkling-2-mini-builds-gb10/) · [Model card](convert/MODEL_CARD.md).

| Item | Pinned setup |
|---|---|
| Hardware | NVIDIA GB10, unified memory; other hardware unmeasured |
| Weights | [capyctl/Inkling-2-Mini-NVFP4-FP8](https://huggingface.co/capyctl/Inkling-2-Mini-NVFP4-FP8) at `f3a7d6f69777e2713adaffbda1460581b1dabf60` |
| Engine | [SGLang PR #51926](https://github.com/sgl-project/sglang/pull/51926) at `92e894ef904bfdb5c0cf2f2a541cbd4eec023573`, plus the supplied three-file loader patch (+81/−13) |
| Draft | Bundled, unchanged `mtp.safetensors`; DSpark gamma 7 / eight tokens per verify block |
| Memory | Static share 0.80, BF16 KV, `MAX_JOBS=2` |
| Context | No override: the model's default limit is 1,048,576 tokens; elected setup checked only through 220k retrieval and 110k decode |
| Lifecycle | Stop/restart only; parking/waking unqualified |

Exactly 144 attention projection matrices become E4M3 FP8 with per-output-channel FP32 scales. Thinking Machines' native NVFP4/MXFP8 experts, BF16 head, DSpark draft and tokenizer remain byte-identical. This is not a full-model FP8 conversion.

## Why native, not a deployment YAML

The candidate was checked with **CapyCTL 0.1.12** using `capyctl validate config --file <candidate>`. It passed document/schema validation, which explicitly did not check host resolution or engine qualification. CapyCTL reserves `--mem-fraction-static` and derives it from a byte grant and free memory at startup, so a deployment cannot preserve this measured fixed 0.80 fraction as written. Its generic context fitter also requires `max_position_embeddings`; Inkling instead declares `model_max_length` and needs loop-aware local/global cache and convolution-state accounting. Renaming a config field would not fix that accounting.

The PR interpreter can be selected with `local_engine.sglang: /home/me/sglang-inkling-pr51926-venv/bin/python3` in a CapyCTL role document, but selecting an installation does not resolve those gaps. No managed deployment or controller/native equivalence is claimed here. The comparison records native engine measurements, not CPU/offline qualification.

## Install and apply the required patch

Prerequisites are Linux aarch64, Python 3.12 and the existing CUDA 13.0 toolkit. The installer creates an isolated environment and does not change system packages, drivers or the toolkit. **`install.sh` and `patches/sglang-inkling-fp8-attention.diff` are byte-for-byte copies of the exact files used for the measurements**, not a new installation recipe. Their SHA-256 identities are:

```text
3ae7f48baae032d85c65c6f09b645642746f259ae47f7e90251e7b50b476249f  install.sh
46ec5d560bd8507c2a5f935092af98c1d803bb10d2c2cf872b6c68a1c0bf02d8  patches/sglang-inkling-fp8-attention.diff
```

From this directory, install once, then apply the patch to both the pinned source and installed package:

```bash
export PYTHONDONTWRITEBYTECODE=1
bash install.sh
PATCH="$PWD/patches/sglang-inkling-fp8-attention.diff"
source /home/me/sglang-inkling-pr51926-venv/bin/inkling-env
patch --directory "$VIRTUAL_ENV/.cache/src/sglang-92e894ef904bfdb5c0cf2f2a541cbd4eec023573" -p1 < "$PATCH"
patch --directory "$VIRTUAL_ENV/lib/python3.12/site-packages" -p2 < "$PATCH"
```

Do not apply it to another SGLang version or environment. The unpatched PR does not load these FP8 attention weights. The patch reuses the existing compressed-tensors dynamic-per-token FP8 linear method, returns BF16 activations to unchanged attention kernels and retains strict tensor/dtype/completeness checks.

## Download the pinned runtime files

The Hugging Face repository is private until the recipe PR merges; links use its final public URL. This Hub download path was not reproduced while private. The measured local payloads and repaired index were audited separately.

```bash
hf download capyctl/Inkling-2-Mini-NVFP4-FP8 \
  --revision f3a7d6f69777e2713adaffbda1460581b1dabf60 \
  --local-dir /home/me/models/Inkling-2-Mini-capyctl-fp8attn \
  --include model-00001-of-00001.safetensors model.safetensors.index.json \
  mtp.safetensors config.json generation_config.json hf_quant_config.json \
  chat_template.jinja processor_config.json special_tokens_map.json \
  tiktoken/tokenizer.model tokenizer.json tokenizer_config.json
```

| File | SHA-256 |
|---|---|
| Main shard | `b7fbb2591af534e7124254af7f4c5e92a9909ec555a48208b4b75b11969b2b5d` |
| Bundled draft | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Repaired index | `299eec2ea8fd903cd2cafe74c3c09e5baa6161b5c85359b5f911712a32da7f71` |

The pin includes the index repair: 1,388 tensor names, including all 144 new scales, mapped to the actual main shard. Later model-card-only commits do not replace this weights pin.

## Start it

Run with the GPU otherwise idle. Set `INKLING_BIND` to your local loopback bind value before starting; no machine-specific bind value is published. `launch.sh` preserves the measured SGLang arguments and four DSpark optimization variables, with only the local checkpoint/interpreter paths and bind value supplied by the operator. It deliberately adds no context override.

```bash
bash launch.sh
# In another terminal, set INKLING_API_URL to the engine's local API endpoint.
curl -s "$INKLING_API_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":512}'
```

Keep the model's default high effort (0.9) when reproducing the measurements. `enable_thinking:false` is not an Inkling template switch. Gamma 2 failed before readiness; gamma 4 is **[INFERENCE]** excluded by the same exact-size guard and was not redundantly launched. Only gamma 7 is supported by this implementation.

## Measured results

Short values are aggregate completion tokens / round wall span, median of three measured rounds after one warm-up. Context values are single-stream post-first-token decode, also three rounds. All use default high effort, temperature zero and a 512-total-output-token cap including reasoning.

| Measurement | Elected FP8 DSpark | Community standard (`tv-std`) |
|---|---:|---:|
| Aggregate tokens/s, 1 / 2 / 4 / 8 streams | 47.31 / 66.59 / 89.06 / 134.13 | 48.58 / 62.51 / 84.34 / 129.86 |
| Decode tokens/s, 8k / 32k / 64k / 110k | 42.67 / 40.57 / 40.99 / 32.72 | 41.42 / 40.79 / 34.75 / 31.92 |
| Short TTFT s, 1 / 2 / 4 / 8 streams | 0.195 / 0.245 / 0.302 / 0.447 | 0.231 / 0.274 / 0.303 / 0.384 |
| GSM8K / tools / HumanEval | 240/250 / 28/30 / 149/164 | 241/250 / 28/30 / 154/164 |
| Truncations, same task order | 2 / 0 / 12 | 2 / 0 / 7 |
| Retrieval, all-three-code / ordered exact | 56/60 / 56/60 — strict fail | 56/60 / 56/60 — strict fail |
| Strict copy | 100/100 | 100/100 |

The elected setup passes the declared baseline-relative Wilson/no-new-tool-type gates, not a claim of equal quality. Full HumanEval on original-weight SGLang is 151/164 with 10 truncations. **[INFERENCE] Both the engine stack and FP8 contribute to the observed 154 → 151 → 149 gap**; this single greedy cohort does not prove a causal split or statistical significance.

## Cold start and limits

One fresh-JIT start at 0.80 with `MAX_JOBS=2` reached readiness in **881.58 s (about 882 s)**, with minimum observed **4.36 GiB MemAvailable** across 444 two-second samples and 22.50 GiB at ready. The isolated JIT cache was moved aside and restored; the filesystem page cache was not dropped. This is cold-JIT, not cold-machine/download qualification. Measurements used a 3 GiB memory watcher; `launch.sh` is the engine command, not that benchmark supervisor. Monitor MemAvailable and keep the machine otherwise idle during first compilation.

An earlier original-checkpoint plain cold start at 0.80 was stopped below the safety floor. The elected success does not qualify plain-mode cold startup. Warm elected speed-start readiness was 247.13 s, with 20.48 GiB available at ready and 19.83 GiB minimum; these are not total model allocations.

Only GB10 was measured. The model's default limit remains 1,048,576 tokens, but nothing beyond 220k was tested on the elected setup. Both recipes fail strict retrieval completeness; no arbitrary long-context accuracy claim is made. Vision/multimodal quality, other hardware, other engine versions and parking/waking remain unqualified. CPU/offline checks are not engine qualification.

Apache-2.0 derivative of Thinking Machines' model. Retain upstream licenses and notices; this independent conversion implies no upstream endorsement. See the [model card](convert/MODEL_CARD.md) for conversion details and attribution.
