# GPT-OSS-20B CapyCTL MXFP4 + FP8 attention on vLLM 0.32.0, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10, compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Kernel 6.17.0-1031-nvidia, NVIDIA driver 580.173.02, CUDA 13.0 |
| Engine | vLLM 0.32.0; torch 2.13.0+cu130, flashinfer-python 0.7.0.post1, transformers 5.19.0; installed source commit unknown |
| Model | [`capyctl/gpt-oss-20b-MXFP4-FP8`](https://huggingface.co/capyctl/gpt-oss-20b-MXFP4-FP8) at `66d5d29b`; unchanged OpenAI MXFP4 experts, per-channel FP8 E4M3 attention, BF16 head/embedding/router; Quark format |
| Drafter | None |
| CapyCTL | 0.1.12, standalone; explicit `lmcache: off` |
| Measured | 2026-11-19; full profile, matched long-context probes and shipped-file sweep in the [comparison evidence](../../comparisons/gpt-oss-20b-builds-gb10/bench/fp8-attention/) |

**Use this recipe without a drafter across context lengths.** On the fixed chat/code
corpus it measures **57.77 tokens/s at one stream and 196.71 aggregate at eight**,
with **237/250 GSM8K and 30/30 tools**. OpenAI's checkpoint on the same vLLM version
measures 45.81 / 184.07; OpenAI's checkpoint on SGLang 0.5.22 with 131k-context
serving settings, no drafter, measures 44.85 / 178.67. The latter comparison changes
both checkpoint and engine, not just weight precision.

The [comparison](../../comparisons/gpt-oss-20b-builds-gb10/) retains every measured
alternative. llama.cpp leads single-stream short-prompt decode at 76.16 tokens/s,
but gives 145.87 at eight streams and has one retained repetition/cap failure.
This recipe is not the fastest setup at every concurrency.

### The checkpoint

Made from [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) at
`6cee5e81ee83917806bbde320786a8fb61efebee` with [`convert/`](convert/):

| Tensors | Stored format |
|---|---|
| Routed expert weights and E8M0 scales | Original MXFP4, group 32; reversible per-expert Quark layout |
| Expert biases | Original BF16, byte-identical |
| Attention: 96 Q/K/V/O projection matrices | FP8 E4M3, FP32 multiplicative scale per output channel; dynamic FP8 inputs at runtime |
| Output head, input embedding, routers, norms, sinks and other biases | Unchanged BF16 |
| KV cache | Selected by serving flags: FP8 E4M3; not calibrated by this conversion |

**Why FP8 attention:** it reduces attention weight storage without requantizing
OpenAI's experts. [INFERENCE] Batch-one weight-read arithmetic falls from
3.708 to 3.072 GB/token, a 17.2% reduction; this excludes KV/activation traffic and
reuse and is not a memory-traffic measurement. Tensor payload is 13,125,022,848 bytes.
The [audit](convert/checkpoint-audit.json) covers all 459 source and 5,019 output
tensors: all 144 source expert tensors reconstruct byte-identically, all 96
attention matrices/scales match the conversion rule, and every retained tensor is unchanged.
The output head stays BF16 because this installed GPT-OSS implementation does not
pass quantization configuration to its head.

Offline next-token agreement is **98.15%** against the actual teacher-forced
BF16 reconstruction of OpenAI's released checkpoint. OpenAI's own generated-ID
self-consistency is **98.39%**, a different target: on that common generated-ID basis,
this checkpoint scores **97.78%**. The original 99.5% offline gate failed; it was
judged too strict before subsequent native qualification and selection for publication.
This is not an accuracy-equivalence claim. See [the agreement method](../../comparisons/gpt-oss-20b-builds-gb10/tensors/NOTES.md).

### The settings

- **No drafter.** On this checkpoint with Triton attention, DFlash8 improves short
  decode to 81.26 tokens/s but falls to 49.08 at 8k input and 22.01 at 32k, versus
  no-draft 56.25 / 52.30. Nebius EAGLE3 depth 3 reaches 82.59 on short prompts,
  but loses at 32k (44.41 versus 52.30) and scores only 29/30 tools. Depth 5 also
  loses at long input, scores 29/30 tools and repeats a phrase through a guard's
  output cap. None passes both task quality and the no-long-context-loss rule.
- **131,072 context, max 16 requests, 40 GiB KV pool, FP8 E4M3 KV.** The cache
  grant is the comparison budget, not a minimum memory requirement or total footprint.
  Prefix caching is disabled. The 131k serving limit is not a throughput guarantee.
- **Auto/FlashInfer attention; Marlin MXFP4 MoE.** Native logs select
  `CutlassFP8ScaledMMLinearKernel` for `QuarkW8A8Fp8`. The tensor audit and scheme
  mapping establish that those FP8 linears are attention; the head is not FP8.
  Native model-loading memory is 13.13 GiB, not a separate attention allocation.
- **`lmcache: off` and `residency: restart_only`.** No parking/waking measurement.
  KV scales fall back to 1.0 where the checkpoint supplies none; FP8 KV is not calibrated.
- **Harmony parsers:** `openai_gptoss` reasoning and `openai` tools. Measurements
  explicitly use `reasoning_effort: low`; completion counts include reasoning.

## Run it

[Install CapyCTL](https://edurdias.github.io/capyctl/docs/install/) and vLLM 0.32.0.
The measured environment uses CUDA 13.0 and the existing Harmony vocabulary.
For an offline launch, place `o200k_base.tiktoken` in
`/home/me/models/gpt-oss-tokenizer`; its SHA-256 is
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.
The vocabulary is available from
[OpenAI's tokenizer assets](https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken).

```bash
CUDA_HOME=/usr/local/cuda-13.0 capyctl engine add /home/me/vllm-0.32.0-venv \
  --env TIKTOKEN_ENCODINGS_BASE=/home/me/models/gpt-oss-tokenizer
```

In one terminal, with the comparison's resource policy:

```bash
capyctl start standalone \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB
```

From this recipe directory in another terminal:

```bash
capyctl deploy model --file deployment.yaml
capyctl start deployment gpt-oss-20b-capyctl-vllm --wait
```

[`deployment.yaml`](deployment.yaml) selects only the three safetensor shards,
index, config, generation config, tokenizer JSON/config, special-token map and
chat template. It pins `capyctl/gpt-oss-20b-MXFP4-FP8` at the weights commit
`66d5d29bb56ebef0d24c0333b7646d058b581170` using CapyCTL's tagged Hugging Face source
form; later commits to the repository change only the model card. The shards at that
commit have the same SHA-256 as the measured local checkpoint. The validated local run substituted only the audited local
checkpoint for that source; it did not change engine settings.

Set `CAPYCTL_INFERENCE_URL` to your local inference base URL and `CAPYCTL_API_KEY`
from CapyCTL's credentials file, as described in the
[client guide](https://github.com/edurdias/capyctl/blob/main/docs/guide/one-machine.md).
Neither value belongs in the deployment file. Send a first request:

```bash
curl -s "$CAPYCTL_INFERENCE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $CAPYCTL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b-capyctl-vllm","messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"low","temperature":0,"max_tokens":512}'
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, process-cold | 104.19 s, local weights; OS/file/JIT caches not cleared |
| Ready, warm stop/start | 103.30 s |
| Peak system-used memory | 67.545 GB absolute, sampled across startup, sweep and restart |
| Peak GPU compute allocation | 58.730 GB sampled |
| Model-loading memory, native log | 13.13 GiB total; not the whole serving footprint |
| Exactly-24k TTFT, selection profile | 3.543 / 9.601 s at 1 / 4 streams |
| Accuracy, separate native run | GSM8K 237/250, tools 30/30; zero HTTP errors or accuracy truncations |

System-used memory is `MemTotal - MemAvailable`, sampled every 0.5 s, not a model
allocation or CapyCTL `startup.measured` peak. GB is decimal; GiB is binary.
The [shipped summary](../../comparisons/gpt-oss-20b-builds-gb10/bench/fp8-attention/shipped/summary.json)
records the validated file digest and the one local-source substitution.

## Benchmark

The fixed chat/code ranking corpus gives:

| Streams | Together, tokens/s | Each, decode tokens/s | TTFT, s |
|---|---:|---:|---:|
| 1 | 57.08 | 57.77 | 0.124 |
| 2 | 93.37 | 47.41 | 0.182 |
| 4 | 134.68 | 34.08 | 0.207 |
| 8 | 196.71 | 24.84 | 0.252 |

The separate exact-file sweep uses a shared document, three rounds in each of
28 input/concurrency cells. c1 below is post-first-token decode; c2/c4/c8 are
aggregate completion tokens / round wall time, including prefill. These are not
substitutes for the ranking corpus above.

| Input tokens | c1 decode | c2 aggregate | c4 aggregate | c8 aggregate | c1 TTFT, s |
|---:|---:|---:|---:|---:|---:|
| 500 | 57.49 | 99.16 | 165.38 | 258.27 | 0.152 |
| 1,000 | 57.23 | 99.97 | 152.96 | 239.86 | 0.200 |
| 2,000 | 56.35 | 95.28 | 146.56 | 225.47 | 0.298 |
| 4,000 | 56.65 | 90.27 | 135.57 | 200.27 | 0.540 |
| 8,000 | 55.88 | 80.80 | 115.43 | 159.76 | 1.038 |
| 16,000 | 54.53 | 66.70 | 88.52 | 112.62 | 2.179 |
| 32,000 | 52.03 | 46.30 | 56.09 | 65.07 | 5.067 |

Native FP8 attention/KV and LMCache off were confirmed. Matched single-stream
long-input decode, three rounds and a 512-total-token cap:

| Input tokens | This recipe, tokens/s | OpenAI, same vLLM | Reference · OpenAI, SGLang |
|---:|---:|---:|---:|
| 8,000 | 56.35 | 44.85 | 43.67 |
| 32,000 | 52.23 | 42.18 | 40.32 |
| 64,000 | 47.57 | 39.17 | 36.71 |
| 110,000 | 42.54 | 35.50 | 32.48 |

### Functional gates and context limits

Low/medium reasoning separation, required-argument tool round trip and forced
empty-schema tool round trip pass. Retrieval passes **48/48 document/variant
observations**: six documents, two prompt variants, four lengths (32k/64k/96k/111k),
all three codes present and ordered exactly, with zero cap hits or HTTP errors.
This finite corpus does not guarantee retrieval on arbitrary prompts.

Copy-100 was measured only on OpenAI's the reference configuration, with and
without DFlash8: both score 50/100 strict exact copy and 100/100 exact-code presence.
The strict failures add `Code: `; they remain failures. **Copy-100 on this checkpoint
is not measured.**

## License

Apache-2.0, as OpenAI's source; retain the source license, applicable notices and
[USAGE_POLICY](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/USAGE_POLICY).
CapyCTL changed attention weights/scales, expert serialization and serving metadata,
not the expert values or retained BF16 tensors. This is an independent conversion,
not an OpenAI release or endorsement. See the [model card](convert/MODEL_CARD.md).
