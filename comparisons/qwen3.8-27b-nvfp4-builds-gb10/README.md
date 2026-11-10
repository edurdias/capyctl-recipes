# Qwen3.8-27B builds on one GB10: CapyCTL NVFP4 against the published ones

`capyctl/Qwen3.8-27B-NVFP4` is Qwen3.8-27B with every large Linear in NVFP4, made with
[`convert/`](../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/convert/). This comparison measures it
through CapyCTL on one GB10 against the BF16 source and five published 4-bit builds, on
SGLang 0.5.22 and vLLM 0.32.0, with and without the DFlash2 drafter, and against
MiaAI-Lab's EXL3 build on their exllamav3 server. It then searches the engine settings for
the recipe. The results:

- **Fastest without drafts:** CapyCTL NVFP4 at 14.9 tokens/s on one stream, 19% above
  NVIDIA's NVFP4 (12.5) on SGLang. Its layout matches gittensor's build (14.8), which
  removed the MTP head. Decode reads 14.45 GB a token against NVIDIA's 17.61 GB.
- **Fastest with DFlash2:** vLLM 0.32.0 with the DeltaNet state cache in bf16, at 36.1
  tokens/s on one stream and 181 at 8 streams, 23% and 13% above NVIDIA's checkpoint on
  the same settings. It is the
  [vLLM recipe](../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/); the best SGLang settings are
  the [SGLang recipe](../../sglang/qwen3.8-27b-capyctl-nvfp4-gb10/).
- **Past 8 streams** (run 6, cap raised to 48): vLLM peaks at 390 tokens/s at 32 streams with
  a 32 GiB KV cache (300 with the recipe's 16 GiB, which holds 22 requests at most), SGLang at
  329. CapyCTL 0.1.10 keeps at most 32 requests in flight per deployment, so 32 is as far as
  this goes through CapyCTL.
- **No measurable cost in accuracy:** every build scores 97.9% to 98.4% on the GSM8K
  answers it finished, against 98.3% for BF16, and 29 or 30 of 30 tool calls.
  Quantization changes the exact wording sooner than run-to-run noise does (see
  [Accuracy](#accuracy)).
- **Closer to BF16** ([tensors](#closer-to-bf16-the-tensors-and-greedy-agreement), weights
  only): NVIDIA searched its 4-bit block scales and ModelOpt's default (ours, gittensor's,
  RadixArk's) does not, so NVIDIA's NVFP4 weights carry 8.6% error against 9.5%. Re-rounding
  ours with searched scales, the checkpoint's second revision, raises greedy next-token
  agreement with BF16 from 92.63% to 93.00%; NVIDIA's build agrees 94.33%, RadixArk's 93.73%.
  The rest of the gap is their FP8 attention and DeltaNet, which cost 3.16 GB a token. Through
  the recipes ([second revision](#the-second-revision-through-the-recipes)) it scores the same
  on GSM8K and tool calls and is 1% to 5% slower on vLLM and up to 13% on SGLang, where DFlash2
  accepts fewer of its drafts; the vLLM recipe pins it, the SGLang recipe the first revision.
- **Against Shardio's recipe** ([r13](#shardios-recipe-r13): RadixArk's checkpoint, EAGLE
  drafts from its MTP head, SGLang 0.5.22): the vLLM recipe is 32%, 30% and 15% faster at 1, 4
  and 8 streams for the same memory; r13's own settings on our checkpoint are 11% to 21%
  faster than on RadixArk's.
- **Ternary Bonsai 2** ([below](#ternary-bonsai-2)): 90.93% teacher-forced agreement
  with BF16, against 92.63–94.33% for NVFP4, with a different backend; GSM8K
  235/240 finished (97.92%) and 30/30 tool calls. Speed and memory remain to be measured.

![Summary: builds without drafts](bench/builds/summary/summary-wide.png)

![Summary: DFlash2 settings](bench/dflash2/summary/summary-wide.png)

The two grids compare each row with the outlined one, NVIDIA's NVFP4 build on the same engine
and settings (the DFlash2 settings variants with the SGLang recipe): blue is better, red worse,
darker a bigger gap up to 25%, and "level" a gap under 3% or one the paired rounds disagree on.
Rows ahead of the reference are above it, rows behind below.

![Summary: the recipes past 8 streams](bench/concurrency/summary/summary-wide.png)

The full reports are [bench/builds/report.html](bench/builds/report.html),
[bench/dflash2/report.html](bench/dflash2/report.html) and
[bench/concurrency/report.html](bench/concurrency/report.html): download them and open them
locally (GitHub shows the HTML source). [bench/results/](bench/results/) has every
capyctl-bench results file, [bench/accuracy/](bench/accuracy/) every accuracy answer, and
[bench/runs/](bench/runs/) each run's log, start times and versions. Run 5's results, each
recipe as it ships, are in the recipes' own `bench/`.

## Setup

| | |
|---|---|
| Hardware | 1x NVIDIA GB10, unified memory (121.7 GiB), aarch64 |
| System | Ubuntu 24.04.5, kernel 6.17.0-1031-nvidia, NVIDIA driver 580.173.02, CUDA 13.0 (`nvcc` 13.0.88) |
| CapyCTL | 0.1.10 release (linux-aarch64, SHA-256 checked), `capyctl start standalone` |
| SGLang | 0.5.22 (torch 2.14.1, triton 3.8.0, flashinfer-python 0.7.1, sglang-kernel 0.4.9, transformers 5.19.0), Python 3.12.14 |
| vLLM | 0.32.0 (torch 2.13.0+cu130, triton 3.7.1, flashinfer-python 0.7.0.post1, transformers 5.19.0), Python 3.12.14 |
| exllamav3 | MiaAI-Lab's fork at `63b32f00` (torch 2.13.0+cu130, triton 3.7.1) with their deployment kit at `12421873`, registered as a CapyCTL custom engine ([`manifests/`](manifests/), [`mia-setup.sh`](mia-setup.sh)) |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c`; MiaAI-Lab's EXL3 build uses [`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) at `4f043626` |
| Benchmark | capyctl-bench from this repository (results written by 0.3.0, reports by 0.3.1) |
| Measured | 2026-11-03 and 2026-11-04 |

CapyCTL ran standalone with its own state directory for each run, the managed limit
raised to 100 GiB and the free reserve to 10 GiB:

```bash
capyctl start standalone --listen 127.0.0.1:18543 --management-listen 127.0.0.1:18544 \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB \
  --set host.resource_policy.memory.system.parked_limit=45GiB
```

### The checkpoints

| Checkpoint | Revision | Format | Read a token (GB) |
|---|---|---|---|
| **[`capyctl/Qwen3.8-27B-NVFP4`](https://huggingface.co/capyctl/Qwen3.8-27B-NVFP4)** | first revision `4a47ea5e` (measured as `8b03c818`, the same weights before the model card); second revision `0ce7e69f`, block scales searched | NVFP4 W4A4 on every large Linear; MTP head BF16 | **14.45** |
| [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | `482ca0f3` | NVFP4 MLP and head; attention and DeltaNet projections FP8 | 17.61 |
| [`RadixArk/Qwen3.8-27B-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) (Shardio's r13) | `319f741c` | NVIDIA's layout (NVFP4 MLP and head, FP8 attention and DeltaNet), default block scales | 17.61 |
| [`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) | `5b7a687f` | NVFP4 on every large Linear; MTP head removed | 14.45 |
| [`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | `4a22293a` | NVFP4 on the large Linears, BF16 head | 16.25 |
| [`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | `91bd022d` | INT4 weights (compressed-tensors), BF16 head | 15.14 |
| [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | `f0b7c9e7` | NVFP4 MLP; FP8 attention, DeltaNet and head | 19.10 |
| [`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) | `19441ac8` | EXL3 3.5 bits a weight | not computed (15.4 GB on disk) |
| [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2` | BF16, the source | about 51 |

"Read a token" is the bytes of the tensors decode reads for every token (the language
model's Linears and head, not the embedding, vision tower or MTP head), from each
checkpoint's safetensors headers. One-stream decode is bound by memory bandwidth, so it
predicts the order without drafts: CapyCTL NVFP4 at 14.9 tokens/s reads 215 GB/s,
NVIDIA's at 12.5 reads 220 GB/s, about 80% of the GB10's 273 GB/s.

## Method

- Every request went through the CapyCTL inference endpoint: streaming chat completions,
  `temperature: 0`, thinking on (the model's default). Thinking and answer tokens both count.
- Each deployment: `context_length: 32768`, `max_concurrent_requests: 8`, an FP8 KV cache
  of 16 GiB, `residency: restart_only` (this SGLang 0.5.22 environment has no memory saver,
  and the comparison measures serving, not parking). SGLang NVFP4 builds state
  `quantization: modelopt`; vLLM ones `language_model_only: true`. One deployment at a
  time, each deleted before the next.
- **Speed**: capyctl-bench, 1, 4 and 8 streams of its eight prompts at `max_tokens: 512`
  (3 rounds in runs 1 and 2, 8 in run 3), and one stream at 2k and 32k prompt tokens
  (2 runs, `max_tokens: 128`). Medians.
- **DFlash2**: SGLang `--speculative-algorithm DFLASH`, 8 draft tokens, the drafter
  unquantized; vLLM `{"method":"dflash","num_speculative_tokens":7}`.
- **Accuracy** ([`../qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/b18.py`](../qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/b18.py)),
  non-streaming, `temperature: 0`, 8 at a time:
  - agreement: 200 prompts (chat, coding, math, multilingual, tool use), 256 tokens, each
    reply compared with the BF16 source's reply to the same prompt;
  - GSM8K: 250 test questions ([selection](../qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy/sources.json)),
    `max_tokens: 2048`, the final `\boxed{}` number;
  - tool calls: 30 cases, right function with valid JSON arguments and the required keys.
- **MiaAI-Lab's exllamav3 server** streams without a token count, so
  [`mia_speed.py`](mia_speed.py) times its streamed replies through CapyCTL and counts
  their tokens with the checkpoint's tokenizer.
- Runs ([`run-host.sh`](run-host.sh)): 1, the two checkpoints on both engines with and
  without DFlash2, and BF16; 2, a second BF16 launch, the published builds and MiaAI-Lab's;
  3 ([`run-3.sh`](run-3.sh)), the DFlash2 settings sweep; 4 ([`run-4.sh`](run-4.sh)),
  MiaAI-Lab's speed and the two recipes' accuracy as they ship; 5 ([`run-5.sh`](run-5.sh)),
  each recipe's `deployment.yaml` with capyctl-bench's full sweeps; 6 ([`run-6.sh`](run-6.sh)),
  the same files with the concurrency cap raised to 48, at 8 to 48 streams; 7
  ([`run-7.sh`](run-7.sh)), Shardio's recipe r13 on RadixArk's checkpoint and on ours; 8
  ([`run-8.sh`](run-8.sh)), the second revision with run 3's vLLM settings and run 4's
  accuracy set; 8b ([`run-8b.sh`](run-8b.sh)), the first revision again straight after, the
  same-day control; 9 ([`run-9.sh`](run-9.sh)), run 5 again on the second revision, and the
  SGLang recipe's accuracy set on it.
- **Tensors and greedy agreement** ([`tensors/`](tensors/)), without activations:
  [`read_tensors.py`](../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/convert/read_tensors.py) rebuilds
  every quantized tensor from its stored format and compares it with the BF16 source;
  [`next_token_agree.py`](tensors/next_token_agree.py) has the BF16 model answer the 200
  agreement prompts greedily (512 tokens), then each build read those answers and pick its next
  token at every position (vLLM 0.32.0 offline, the same settings for every build: no drafter,
  BF16 KV cache, no prefix caching). [`run-tensors.sh`](tensors/run-tensors.sh) runs both;
  [`summarize_tensors.py`](tensors/summarize_tensors.py) prints the tables below.
- **Bonsai follow-up:** weights read on 2026-11-06; GPU agreement and CapyCTL
  accuracy on host A on 2026-11-10, llama.cpp b12242 (`3b4d6ab3`), CapyCTL 0.1.11.
  [`bench/bonsai/RESULTS.json`](bench/bonsai/RESULTS.json) indexes the evidence;
  [`method.json`](bench/bonsai/method.json) records settings and backend differences.
  These are separate from the speed runs above.

## Results

### Without drafts

| Checkpoint (SGLang 0.5.22) | 1 stream | 4 streams | 8 streams | 2k: decode, time to first token | 32k: decode, time to first token |
|---|---|---|---|---|---|
| **CapyCTL NVFP4** | **14.9** | **55.2** | **100.9** | **14.9, 0.69 s** | **14.0, 18.1 s** |
| gittensor NVFP4 | 14.8 | 54.8 | 100.3 | 14.8, 0.71 s | 13.9, 18.3 s |
| RedHatAI INT4 | 13.7 | 52.9 | 99.3 | 13.7, 1.69 s | 12.9, 30.6 s |
| NVIDIA NVFP4 | 12.5 | 46.2 | 85.9 | 12.5, 0.77 s | 11.8, 18.6 s |
| QUASAR NVFP4 | 12.4 | 48.8 | 90.9 | 12.2, 0.76 s | 11.6, 22.0 s |
| unsloth NVFP4 | 11.1 | 42.0 | 79.0 | 11.1, 0.86 s | 10.6, 27.9 s |
| Qwen BF16 | 4.5 | 17.2 | 33.4 | 4.4, 1.53 s | 4.4, 26.8 s |

Tokens/s; 4 and 8 streams are the total. On vLLM 0.32.0: CapyCTL NVFP4 14.6 / 53.6 / 100.1,
NVIDIA NVFP4 12.1 / 45.1 / 85.5. MiaAI-Lab's EXL3 build on its exllamav3 server: 15.4
tokens/s at 1, 4 and 8 streams, since the server decodes one request at a time.

### With DFlash2: the settings sweep

| Engine, checkpoint, settings | 1 stream | 4 streams | 8 streams | 2k: decode | Drafted tokens accepted per round |
|---|---|---|---|---|---|
| **vLLM, CapyCTL NVFP4, DeltaNet state cache bf16 (recipe)** | **36.1** | 106.4 | **181.4** | 50.6 | 3.32 |
| vLLM, NVIDIA NVFP4, DeltaNet state cache bf16 | 29.3 | 96.5 | 160.3 | 47.7 | 3.26 |
| SGLang, CapyCTL NVFP4, MiaAI-Lab flags (recipe) | 35.6 | **108.7** | 165.8 | 45.9 | 3.28 |
| SGLang, NVIDIA NVFP4, MiaAI-Lab flags | 30.6 | 93.3 | 150.1 | 47.9 | 3.32 |
| SGLang, CapyCTL NVFP4, 3 continuous decode steps | 35.5 | 105.2 | 168.0 | 45.7 | 3.30 |
| SGLang, CapyCTL NVFP4, MiaAI-Lab flags + torch.compile | 33.4 | 98.7 | 169.3 | 46.3 | 3.25 |
| SGLang, CapyCTL NVFP4, MiaAI-Lab flags, 6 draft tokens | 34.8 | 102.7 | 138.8 | 39.5 | 3.14 |
| SGLang, CapyCTL NVFP4, MiaAI-Lab flags, 12 draft tokens | 35.3 | 73.8 | 154.7 | 56.4 | 3.39 |
| exllamav3, MiaAI-Lab EXL3 3.5bpw, its DFlash2 EXL3 drafter | 36.2 | 38.1 | 36.0 | | |

Tokens/s, run 3 (8 rounds a point); exllamav3 from run 4. MiaAI-Lab's flags are
`--mamba-ssm-dtype bfloat16 --num-continuous-decode-steps 2`, from their
[SGLang GB10 recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark). Drafts
accepted per round come from the engine logs.

Stock settings (run 1, 3 rounds): vLLM CapyCTL NVFP4 35.6 / 98.7 / 150.4, NVIDIA
29.3 / 91.1 / 139.8; SGLang CapyCTL NVFP4 34.5 / 93.2 / 140.3, NVIDIA 34.3 / 85.3 / 127.2.

- **The weights, not the drafter:** on the same settings both checkpoints accept the
  same number of drafted tokens per round (3.26 to 3.37), so the 10% to 23% gap is the
  cheaper verify step of a checkpoint that reads fewer bytes.
- **What moved:** vLLM's DeltaNet state cache in bf16 (`--mamba-ssm-cache-dtype bfloat16`)
  raised 8 streams from 150 to 181 tokens/s. MiaAI-Lab's SGLang flags raised it from 140 to
  166.
- **What did not:** a third continuous decode step and torch.compile stayed within the
  spread (torch.compile also took 522 s to start against 306 s); 6 and 12 draft tokens
  lost at 4 and 8 streams.
- **Spread:** the SGLang recipe settings gave 38.1 tokens/s on one stream in run 2 and
  35.6 in run 3, with 4 and 8 streams within 3%. One-stream differences of 2 tokens/s are
  inside that spread.
- **exllamav3** is level with the vLLM recipe on one stream, but serves one request at a
  time, so 8 streams total 36 tokens/s against 181.

### Past 8 streams

Run 6: each recipe file with `max_concurrent_requests: 48` instead of 8, at 8 to 48 streams,
three rounds each, `max_tokens: 512`. CapyCTL refused SGLang at the file's 65 GiB request and
at 85 GiB: SGLang keeps 49.7 GB of DeltaNet and draft-token state for 48 running requests, so
it needs 93.5 GiB; it ran with 94 GiB. vLLM ran once with the file's 16 GiB KV cache and once
with 32 GiB. Total tokens/s:

| Recipe, change | 8 | 12 | 16 | 24 | 32 | 40 | 48 | Startup peak |
|---|---|---|---|---|---|---|---|---|
| vLLM, cap 48, 16 GiB KV cache | 183 | 240 | 267 | 254 | 300 | | 292 | 49.6 GiB |
| vLLM, cap 48, 32 GiB KV cache | | | 280 | 354 | **390** | 338 | 359 | 63.3 GiB |
| SGLang, cap 48, 94 GiB request | 168 | 207 | 250 | 303 | **329** | | 307 | 94.3 GiB |

- **vLLM's 16 GiB cache holds 22 requests at most.** It holds the DeltaNet state blocks too;
  the log shows 20 to 22 running and up to 12 waiting, so total throughput levels off near
  300. With 32 GiB all 32 run, and the total still grows 10% from 24 to 32 streams.
- **SGLang is nearly saturated at 32**: 8% more from 24 to 32 streams, for twice vLLM's
  memory.
- **CapyCTL's bound ends the curve at 32.** CapyCTL 0.1.10 keeps at most 32 requests in
  flight per deployment (`MAX_REQUESTS_PER_DEPLOYMENT` in `capyctl-domain`), whatever
  `max_concurrent_requests` says; at 40 and 48 streams the rest wait in its queue (worst
  first token 37 to 48 s), while the engines' logs show at most 32 running on vLLM (32 GiB)
  and 33 on SGLang, neither queuing. The 40- and 48-stream totals are not engine
  throughput.
- The run 6c results file's label was changed to `rcp-vl-hi-kv32` when it was copied here,
  to tell it from run 6's `rcp-vl-hi`.

### Closer to BF16: the tensors and greedy agreement

Every quantized tensor rebuilt and compared with BF16 (relative error, size-weighted):

| Tensors | CapyCTL | gittensor | RadixArk | NVIDIA | Searched block scales |
|---|---|---|---|---|---|
| MLP gate/up/down (NVFP4 in all four) | 9.49–9.50% | 9.49–9.50% | 9.49–9.50% | **8.63–8.65%** | 8.59–8.60% |
| `lm_head` (NVFP4 in all four) | 9.48% | 9.48% | 9.48% | **8.48%** | 8.60% |
| Attention and DeltaNet | NVFP4 9.40–9.49% | NVFP4 9.40–9.49% | FP8 2.67% | FP8 2.67% | NVFP4 8.57–8.59% |
| Block scales exactly as ModelOpt's max calibration sets them | 99.94% | 99.94% | 99.94% | 28.07% | – |

- **The error is format noise, the same everywhere:** 9.4–9.5% in every one of the 401 NVFP4
  tensors, every type and layer, no scale overflow or underflow, and no more of it on each
  tensor's 32 strongest input directions than uniform noise would put there (1.01–1.08×; attention
  k 1.18×). Weighting each input channel by its norm gain changes nothing.
- **NVIDIA searched its block scales; ModelOpt's default does not.** Only 28% of NVIDIA's scales
  are the default's, and its NVFP4 tensors carry about 9% less error. gittensor's and RadixArk's
  NVFP4 weights carry exactly ours: max calibration of weights does not depend on calibration
  data.
- **Searching each block's scale** (21 from 0.60× to 1.10× of the default, FP8-rounded, the one
  closest to BF16) brings every tensor to 8.6%, at no cost in size:
  [`requant_scales.py`](../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/convert/requant_scales.py),
  the checkpoint's second revision. Its speed through the recipes is
  [below](#the-second-revision-through-the-recipes).

Greedy next-token agreement with BF16 on 46,706 answer tokens:

| Build | Agrees | Coding | Math | Chat | Multilingual | Tool use |
|---|---|---|---|---|---|---|
| BF16 itself (the engine's floor: reading against writing) | 99.54% | 99.51% | 99.82% | 99.44% | 99.45% | 99.76% |
| NVIDIA (FP8 attention and DeltaNet, searched scales) | **94.33%** | 94.58% | 97.36% | 92.49% | 93.28% | 97.20% |
| RadixArk (FP8 attention and DeltaNet, default scales) | 93.73% | 93.47% | 96.84% | 92.09% | 92.73% | 97.18% |
| **CapyCTL, second revision (all NVFP4, searched scales)** | **93.00%** | 92.98% | 96.34% | 91.00% | 92.05% | 96.39% |
| gittensor (all NVFP4, default scales) | 92.69% | 92.30% | 96.36% | 90.83% | 91.68% | 96.45% |
| CapyCTL, first revision (all NVFP4, default scales) | 92.63% | 92.29% | 96.30% | 90.51% | 91.91% | 96.38% |
| Ternary Bonsai 2 Q2_0 (llama.cpp, cross-backend) | 90.93% | 91.76% | 97.51% | 88.70% | 86.77% | 96.23% |

Four builds separate the two causes: FP8 attention and DeltaNet are worth about 1.1 points of
agreement (CapyCTL to RadixArk), searched scales 0.4 to 0.6 (CapyCTL to its second revision,
RadixArk to NVIDIA). Agreement counts every changed token, a synonym as much as an error; it did
not show in GSM8K or tool calls ([Accuracy](#accuracy) and the
[second revision](#the-second-revision-through-the-recipes)), which RadixArk's was not
measured on.

### Ternary Bonsai 2

PrismML's Ternary Bonsai 2 27B stores weights as {-1, 0, +1}, with an FP16 scale
per 128 weights, Hadamard-rotated in 1,024-wide blocks. Its files are 5.95 GB in
PTQ1_0 and 7.63 GB in Q2_0 (file sizes, not serving memory). This run uses
`Ternary-Bonsai-2-27B-Q2_0.gguf`.

**Weights only, 2026-11-06.** [`read_bonsai.py`](tensors/read_bonsai.py) decodes
Q2_0, restores the rotation, signs and head order, and compares 402 tensors with
BF16. [`read-bonsai.csv`](bench/bonsai/read-bonsai.csv) has every tensor;
[`summarize_bonsai.py`](tensors/summarize_bonsai.py) pools the errors: the square
root of the weight-count-weighted mean squared per-tensor relative Frobenius
error, not an unweighted average of tensor percentages.

| Weights | Pooled relative error against BF16 |
|---|---|
| Bonsai | 47.39% |
| Best plain ternary rounding, same rotated basis, scale per 128 | 43.25% |
| Best plain ternary rounding, without rotation | 43.79% |
| NVFP4 builds | 7.45–9.49% |

Bonsai is about 4.1 percentage points further from BF16 than error-optimal
rounding in every tensor family. Its trits agree 85.5% with that rounding.
[`bonsai_rules.py`](tensors/bonsai_rules.py) compares sampled tensors with
BitNet's absmean rounding in the rotated basis: agreement is 86–95%, falling
from 94.8% in layer 3 to 85.9% in layer 55. The zero fraction is 32.8% in every
tensor (absmean: 30.9%); adjacent Q2_0 scale pairs are equal (groups of 128);
no weight uses code 3.

This suggests values optimised against outputs, through training or output-aware
rounding starting from an absmean-like rule. That is an inference from weights,
not evidence of PrismML's training procedure. Lower weight error alone does not
establish better answers.

**Agreement, 2026-11-10.** Bonsai agrees at 42,472/46,706 BF16 answer positions
(90.93%); category scores sit beside the NVFP4 builds above.
[`bonsai-agree.cpp`](bench/bonsai/bonsai-agree.cpp) takes the raw-logit argmax
at each answer-predicting position using the exact saved BF16 prompt and answer
token IDs. Every prefix uses BF16's preceding tokens, not Bonsai's generated
tokens. All 248,077 live tokenizer entries match the GGUF vocabulary. Settings:
BF16 K/V, fresh state per record, no drafts or prefix reuse, context 4,096,
batch and microbatch 512.

This is **cross-backend**: native llama.cpp CUDA b12242 (`3b4d6ab3`) for Bonsai
versus vLLM 0.32.0 for BF16 and NVFP4. The GGUF loader, Hadamard rotation,
batching, recurrent state and numerical kernels differ. The agreement definition
is the same, but this is not a within-engine quantization-only comparison.
BF16's 99.54% floor above is a vLLM floor, not a measured llama.cpp floor.

**Accuracy through CapyCTL 0.1.11:** thinking on, temperature 0, eight requests
at a time, the same b18.py prompts and scorers, 2,048-token answer limit:

| Build and run | GSM8K, all 250 | Cut at limit | GSM8K, finished | Tools |
|---|---|---|---|---|
| CapyCTL NVFP4, first revision, vLLM run 4 | 237/250 (94.8%) | 9 | 237/241 (98.3%) | 30/30 |
| CapyCTL NVFP4, second revision, vLLM run 8 | 235/250 (94.0%) | 11 | 235/239 (98.3%) | 30/30 |
| Ternary Bonsai 2 Q2_0, llama.cpp | 235/250 (94.0%) | 10 | 235/240 (97.92%) | 30/30 |

BF16 finished at 98.3% ([Accuracy](#accuracy)). Bonsai had no HTTP errors.
The small sets do not establish equivalence; agreement counts changed choices,
not whether they are errors. Answers are in
[`bonsai-gsm8k.json`](bench/bonsai/bonsai-gsm8k.json) and
[`bonsai-tools.json`](bench/bonsai/bonsai-tools.json).
Speed, throughput, startup time and serving memory **remain to be measured**:
host A was shared with other work, so no such numbers are reported here.

### The second revision through the recipes

Revision `0ce7e69f` of `capyctl/Qwen3.8-27B-NVFP4`, from a local copy with the same shard
SHA-256s (`1525b2d1…`, `93fda33b…`). Total tokens/s through each recipe's `deployment.yaml`,
capyctl-bench's full sweeps, five rounds a point:

| Recipe, revision | 1 stream | 4 streams | 8 streams |
|---|---|---|---|
| vLLM, first (run 5) | 34.9 | 102.5 | 186.6 |
| vLLM, second (run 9) | 33.5 | 101.0 | 177.4 |
| SGLang, first (run 5) | 35.9 | 105.9 | 165.6 |
| SGLang, second (run 9) | 31.3 | 103.3 | 161.8 |

Runs 5 and 9 are two days apart. Runs 8 and 8b put the two revisions through run 3's vLLM
settings (`vl-full-df-mbf16`, eight rounds a point) on the same afternoon:

| vLLM `vl-full-df-mbf16` | 1 stream | 4 streams | 8 streams |
|---|---|---|---|
| First revision, 2026-11-04 (run 3) | 35.7 | 106.4 | 181.4 |
| Second revision, 2026-11-06 (run 8) | 34.6 | 109.6 | 166.2 |
| First revision, 2026-11-06 (run 8b) | 35.3 | 103.4 | 178.4 |
| Second against first, same prompts round by round (median) | 0.96× | 1.05× | 0.96× |

- **The cost is the drafter, not the engine.** Both revisions have the same layout, sizes and
  kernels; DFlash2 accepts fewer tokens on the second's text: 3.24 against 3.28 a step in
  vLLM's speed rounds, 3.27 against 3.47 in SGLang's at one stream and 4.20 against 4.59 in
  SGLang's accuracy set. Why weights closer to BF16 draft worse was not investigated.
- **The same day accounts for about 2 of vLLM's 5 points at 8 streams** (run 3 to 8b: 181.4 to
  178.4). All eight rounds at 1 stream and the first five at 4 streams of run 8 ran beside the
  checkpoint's upload to Hugging Face (network and disk only); the 8-stream rounds did not.
- vLLM's time to first token is unchanged at every prompt length (0.5k to 32k); SGLang's was
  16% to 17% lower from 8k to 32k in run 9 (18.3 s at 32k against 21.7 s), not investigated.
  The startup peak moved by at most 1.1 GiB and no request failed in either run.

| Accuracy | vLLM, first (run 4) | vLLM, second (run 8) | SGLang, first (run 4) | SGLang, second (run 9) |
|---|---|---|---|---|
| GSM8K, answers that finished in 2,048 tokens | 237/241 (98.3%) | 235/239 (98.3%) | 240/245 (98.0%) | 234/238 (98.3%) |
| GSM8K answers cut at 2,048 tokens | 9 | 11 | 5 | 12 |
| Tool calls | 30/30 | 30/30 | 29/30 | 29/30 |

The vLLM recipe now pins the second revision, for its closer agreement with BF16 at 1% to 5%
of speed; the SGLang recipe keeps the first, where the second cost up to 13% on one stream.

### Shardio's recipe r13

Shardio serves Qwen3.8-27B with r13, `bld_qwen3.8-27b-sglang-xrargf`: RadixArk's checkpoint at
`319f741c`, SGLang 0.5.22, EAGLE drafts from the checkpoint's MTP head (3 steps, top-k 1, 4 draft
tokens, `--enable-linear-replayssm-spec`) and `--prefill-decode-interval 1`. Run 7 ran those flags
through CapyCTL at this comparison's context (32,768) and 8 streams, without its 262k context,
`--max-mamba-cache-size 110` and LMCache, and with the KV cache at the engine's default as r13 has
it. Total tokens/s:

| Setup | 1 stream | 4 streams | 8 streams | Startup peak | Drafted tokens accepted a round |
|---|---|---|---|---|---|
| r13 (RadixArk's checkpoint) | 27.0 | 81.9 | 157.4 | 48.7 GiB | 2.66 |
| r13's flags on CapyCTL's first revision | 30.0 | 99.4 | 175.5 | 51.8 GiB | 2.65 |
| **The vLLM recipe** (CapyCTL, DFlash2; run 3) | **35.7** | 106.4 | **181.4** | 48.1 GiB | 3.3 |
| The SGLang recipe (CapyCTL, DFlash2; run 3) | 35.2 | **108.7** | 165.8 | 58.8 GiB | 3.3 |

The checkpoint alone is worth 11% to 21% under r13's own flags, and DFlash2's 3.3 accepted tokens a
round against MTP drafting's 2.66 the rest. r13 was built for Shardio's gateway: its
prefill-decode interval keeps running streams from stalling behind a long prefill, which these
short-prompt runs do not test, and it was measured on an RTX PRO 6000, not a GB10.

### Accuracy

| Checkpoint (no drafts unless noted) | GSM8K | Cut at 2048 tokens | GSM8K, finished answers | Tool calls | Same reply as BF16 (of 200) |
|---|---|---|---|---|---|
| Qwen BF16 | 234/250 (93.6%) | 12 | 234/238 (98.3%) | 29/30 | 139, a second BF16 launch |
| NVIDIA NVFP4 | 240/250 (96.0%) | 6 | 240/244 (98.4%) | 30/30 | 17 |
| **CapyCTL NVFP4** | 237/250 (94.8%) | 9 | 237/241 (98.3%) | 30/30 | 9 |
| gittensor NVFP4 | 238/250 (95.2%) | 8 | 238/242 (98.3%) | 29/30 | 5 |
| QUASAR NVFP4 | 237/250 (94.8%) | 10 | 236/240 (98.3%) | 29/30 | 17 |
| RedHatAI INT4 | 238/250 (95.2%) | 7 | 238/243 (97.9%) | 29/30 | 12 |
| unsloth NVFP4 | 235/250 (94.0%) | 11 | 235/239 (98.3%) | 30/30 | 14 |
| MiaAI-Lab EXL3 3.5bpw (exllamav3, DFlash2) | 238/250 (95.2%) | 8 | 238/242 (98.3%) | 30/30 | 1 |
| **vLLM recipe** (CapyCTL NVFP4, DFlash2, bf16 state cache) | 238/250 (95.2%) | 9 | 237/241 (98.3%) | 30/30 | 1 |
| **SGLang recipe** (CapyCTL NVFP4, DFlash2, MiaAI-Lab flags) | 240/250 (96.0%) | 5 | 240/245 (98.0%) | 29/30 | 4 |

All on SGLang 0.5.22 unless the row names another engine; BF16 is the SGLang reference.

- On 250 questions one answer is 0.4 points, so the GSM8K rows are level; their headline
  differences come from answers cut at the 2,048-token limit.
- Two launches of the BF16 source give the same 256-token reply to 139 of 200 prompts.
  The 4-bit builds give BF16's exact reply to 5 to 17, and builds that keep more layers in
  FP8 or BF16 (NVIDIA, QUASAR) stay identical a few tokens longer (first difference at
  token 16 to 19, median, against 10 to 11 for the all-NVFP4 builds). The task scores
  above do not move with it.
- A reply from another engine (vLLM, exllamav3) is rarely identical to SGLang's even with
  the same weights; for those rows the task scores are the measure.

### Memory and starts

| | Peak memory CapyCTL measured | Ready |
|---|---|---|
| vLLM, CapyCTL NVFP4, DFlash2, bf16 state cache (recipe) | **48.1 GiB** | 326 s |
| vLLM, NVIDIA NVFP4, DFlash2, bf16 state cache | 50.2 GiB | 316 s |
| SGLang, CapyCTL NVFP4, DFlash2, MiaAI-Lab flags (recipe) | 58.8 GiB | **306 s** |
| SGLang, NVIDIA NVFP4, DFlash2, MiaAI-Lab flags | 60.8 GiB | 323 s |
| SGLang, CapyCTL NVFP4, no drafts | 50.9 GiB | |
| SGLang, NVIDIA NVFP4, no drafts | 54.1 GiB | |
| SGLang, Qwen BF16, no drafts | 81.9 GiB | |
| exllamav3, MiaAI-Lab EXL3 3.5bpw, its DFlash2 | 20.3 GiB (`MemAvailable` drop; status, read 10 s after ready, held no measured peak yet) | 10 s |

The four DFlash2 rows come from run 3, back to back on warm kernel caches; the rows without
drafts from run 1. The recipes' own cold and warm starts are in their READMEs.

## Caveats

- One GB10, one request shape (512 tokens out of eight short prompts), `temperature: 0`.
- The accuracy sets are small: 250 GSM8K questions and 30 tool calls measure a large loss,
  not a small one.
- The EXL3 build was measured with another tool ([`mia_speed.py`](mia_speed.py)): its token
  counts come from encoding the replies again, within about 1% of the generated count.
- Parking was not measured: every deployment ran `restart_only`.
- CapyCTL NVFP4 keeps its MTP head, but no MTP recipe was measured here.

## Reproduce

[`mia-setup.sh`](mia-setup.sh) builds the exllamav3 environment; the run scripts expect
the checkpoints and drafters under `~/models`, `~/drafters` and `~/q38-requant/ckpts`
(their headers name every path and variable), and [`speed.py`](speed.py) and
[`summarize.py`](summarize.py) turn a results directory into the tables above.
