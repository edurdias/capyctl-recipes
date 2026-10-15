# SGLang 0.5.21, Qwen3.8-27B: DFlash2 drafter against the MTP/EAGLE recipe, one GB10

Two SGLang 0.5.21 recipes for Qwen3.8-27B on the same machine, through CapyCTL,
measured with capyctl-bench and interleaved pass by pass:

- **DFlash2** (this repository's catalog recipe, [sglang/qwen3.8-27b-nvfp4-gb10](../../sglang/qwen3.8-27b-nvfp4-gb10/)):
  `nvidia/Qwen3.8-27B-NVFP4`, the DFlash2 drafter with 8 draft tokens, FP8 KV cache
  of 16 GiB, 8 running requests, a 65 GiB memory request.
- **MTP/EAGLE**: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, an NVFP4 checkpoint with a BF16 LM head,
  drafting with its own MTP layer (EAGLE, 3 steps, top-1, 4 draft tokens) and SGLang's
  ReplaySSM speculative verify; a 110-slot state cache for 22 running requests, mixed
  chunked prefill, a 262144-token context and a 79 GiB memory request with a 29 GiB FP8 KV pool.

![Summary](bench/summary/summary-wide.png)

The full report is [bench/report.html](bench/report.html): download it and open
it locally (GitHub shows the HTML source). [bench/summary.md](bench/summary.md)
has every table, [bench/data.csv](bench/data.csv) every point, and
[bench/results/](bench/results/) the capyctl-bench results files with every
request. [results/](results/) holds the lifecycle, check and stall-test records of the run.

## Setup

| | |
|---|---|
| Hardware | 1x NVIDIA GB10, unified memory, aarch64 |
| System | NVIDIA driver 580.173.02, CUDA 13.0, kernel 6.17.0-1031-nvidia |
| CapyCTL | `main` at `e5219a7` plus the ReplaySSM state sizing fix (`d4b8684`, prints `capyctl 0.1.2`), release build, `capyctl start standalone` |
| SGLang | 0.5.21 (torch 2.13.0, flashinfer-python 0.6.18, sglang-kernel 0.4.7, transformers 5.12.1, torch-memory-saver 0.0.10), Python 3.12.14 |
| DFlash2 model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at `482ca0f3832238542f8f5295dde86b5f22711d80` |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| MTP/EAGLE model | [`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead) at `009632fef96dd349150baa780c984e62e70e91fe`, its own MTP layer, no separate drafter |
| Benchmark | capyctl-bench from this repository at `b68d77d` |
| Measured | 2026-10-14 |

CapyCTL ran standalone with its own state directory, the managed memory limit
raised to 100GiB, the free reserve lowered to
10GiB (the limit and the reserve together must fit the
machine's memory) and the parked limit raised to 45GiB:

```bash
capyctl start standalone --listen 127.0.0.1:18543 \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB \
  --set host.resource_policy.memory.system.parked_limit=45GiB
```

Each recipe ran under its own profile of the same SGLang venv, so that each
had its own, initially empty, kernel caches (`SGLANG_CACHE_DIR` and the
FlashInfer, Triton, TVM-FFI, Inductor and CUDA cache variables in the profile's
`env`); the DFlash2 profile approves the drafter path:

```bash
capyctl engine add ~/sglang-0.5.21-venv --name b18a \
  --approve-option=--speculative-draft-model-path --approve-path ~/drafters --env SGLANG_CACHE_DIR=<dir a> ...
capyctl engine add ~/sglang-0.5.21-venv --name b18b --env SGLANG_CACHE_DIR=<dir b> ...
```

## Deployments

- [deployment-dflash2.yaml](deployment-dflash2.yaml): the catalog recipe, short-prompt runs
  (`context_length: 32768`); [deployment-dflash2-ctx.yaml](deployment-dflash2-ctx.yaml) the same with
  `context_length: 262144`, for the 128k point and the stall test.
- [deployment-mtp-eagle.yaml](deployment-mtp-eagle.yaml): the MTP/EAGLE recipe, every run;
  [deployment-mtp-eagle-nomix.yaml](deployment-mtp-eagle-nomix.yaml) the same without
  `--enable-mixed-chunk`, for the stall test only.

Only one deployment ran at a time.

| | DFlash2 | MTP/EAGLE |
|---|---|---|
| Speculative decoding | `DFLASH`, separate drafter, 8 draft tokens, drafter unquantized | `EAGLE` on the checkpoint's MTP layer, 3 steps, top-1, 4 draft tokens, `--enable-linear-replayssm-spec` |
| Running requests | `max_concurrent_requests: 8`; engine `max_running_requests=8` | `max_concurrent_requests: 22`, `--max-mamba-cache-size 110`; engine `max_running_requests=22` |
| KV cache | `fp8_e4m3`, 16 GiB; engine 399457 tokens | `fp8_e4m3` (the checkpoint's calibrated default, stated), 29 GiB; engine 776670 tokens |
| Chunked prefill, max prefill | 8192, 16384 | 8192, 16384 |
| State cache | sized by CapyCTL for 8 requests | 110 slots |
| Memory request | 65 GiB | 79 GiB |
| Parsers | `qwen3_coder`, `qwen3` (CapyCTL's choice for the family) | `qwen3_coder`, `qwen3` (declared) |
| Residency | CapyCTL default (`deep`, weights kept resident while parked) | the same |

## Method

- Every request went through the CapyCTL inference endpoint: OpenAI streaming
  chat completions, `temperature: 0`, thinking on (the model's default).
- **Short prompts**: 1, 4, 8, 16, 22 streams, `max_tokens: 512`, capyctl-bench's
  `prompts.json`, 5 measured rounds after one warm-up round per stream count, in
  2 passes interleaved DFlash2, MTP/EAGLE, DFlash2, MTP/EAGLE; each pass a fresh warm
  start. The DFlash2 recipe runs 8 requests at once, so at 16 and 22 streams the rest wait.
- **Context**: one stream, 2k, 32k, 128k prompt tokens, 3 runs each, `max_tokens: 128`.
- **Long-prompt stall**: 4 streams decode (`max_tokens: 3072`, `ignore_eos`), then one cold
  prompt of about 80000 tokens arrives; the figure is the longest gap between token
  chunks on the running streams while that prompt is prefilled, 2 trials per variant.
- **Lifecycle**: a first start with empty kernel caches (cold), warm starts after `stop`,
  the ready footprint (drop in `MemAvailable` from before the start to Ready),
  one park and wake (a request wakes it) with CapyCTL's parked charge, and stop.
- Engine figures come from the engine's own log, which CapyCTL keeps redacted
  (`<state dir>/logs/<deployment>/<launch>.log`), and CapyCTL's load read.

## Results

### Lifecycle and memory

| | DFlash2 | MTP/EAGLE |
|---|---|---|
| First start, empty kernel caches | 726 s | 604 s |
| `MAX_JOBS` CapyCTL derived for that start | 6 | 4 |
| Kernel caches after the first start | 344M | 313M |
| Warm start (`start` after `stop`) | 260 to 279 s | 263 to 295 s |
| Ready footprint (`MemAvailable` drop) | 67.8 GiB | 81.5 GiB |
| CapyCTL measured startup peak | 70.3 GiB | 81.9 GiB |
| Park, parked charge, parked footprint | n/a (the park was refused) | n/a (the park was refused) |
| Wake by request, to the answer | n/a | n/a |
| Engine KV tokens / running requests | 399457 / 8 | 776670 / 22 |
| CapyCTL load read `max_running` | 8 (engine) | 22 (engine) |
| Mean accept length (engine log, short-prompt passes) | 3.29 | 2.60 |

Neither park happened. Both were refused at once as "the engine is not quiescent":
the run's state directory put SGLang's saver observation socket over the 107 bytes a
Unix socket address holds, so the engine served without the saver observation every
park needs. That is a CapyCTL bug, fixed on `main` in
[capyctl#119](https://github.com/edurdias/capyctl/pull/119); the park and wake
figures are therefore n/a here.

### Short prompts

| Streams | DFlash2 aggregate, tok/s | MTP/EAGLE aggregate | DFlash2 per stream p50 | MTP/EAGLE per stream p50 | DFlash2 TTFT p50 | MTP/EAGLE TTFT p50 | Errors |
|---|---|---|---|---|---|---|---|
| 1 | 32.4 | 17.7 (-45%) | 33.1 | 17.9 | 0.21 s | 0.27 s | 0 / 0 |
| 4 | 81.4 | 66.9 (-18%) | 22.4 | 17.9 | 0.23 s | 0.27 s | 0 / 0 |
| 8 | 123.4 | 119.4 (-3%) | 17.6 | 17.2 | 0.27 s | 0.31 s | 0 / 0 |
| 16 | 124.1 | 214.7 (+73%) | 17.7 | 15.3 | 9.76 s | 0.42 s | 0 / 0 |
| 22 | 122.0 | 280.7 (+130%) | 17.4 | 14.4 | 29.29 s | 0.51 s | 0 / 0 |

### Context

| Prompt | DFlash2 TTFT | MTP/EAGLE TTFT | DFlash2 decode, tok/s | MTP/EAGLE decode | DFlash2 prompt, tok/s | MTP/EAGLE prompt |
|---|---|---|---|---|---|---|
| 2k | 0.86 s | 0.94 s | 56.9 | 19.3 | 2349 | 2134 |
| 32k | 18.67 s | 21.41 s | 28.8 | 18.5 | 1734 | 1509 |
| 128k | 122.39 s | 135.95 s | 30.4 | 16.9 | 1051 | 946 |

### Long-prompt stall

Longest gap between token chunks on the running streams while the cold prompt
is prefilled, per trial (the gap before it arrives in brackets):

| | DFlash2 | MTP/EAGLE | MTP/EAGLE without mixed chunks |
|---|---|---|---|
| Longest gap, s | 56.5 / 57.1 | 9.0 / 9.0 | 60.8 / 59.1 |
| Before the prompt, s | (0.15 / 0.15) | (0.16 / 0.15) | (0.15 / 0.15) |
| Long prompt: tokens, time to first token | 79045 / 80024, 61.1 / 61.7 s | 78607 / 80783, 67.7 / 69.5 s | 81124 / 79590, 65.6 / 63.8 s |

### Checks

| | DFlash2 | MTP/EAGLE |
|---|---|---|
| Tool call (`get_weather`, structured `tool_calls`) | pass: HTTP 200, finish tool_calls, calls ['get_weather'], reasoning 108 chars, markup in content none | pass: HTTP 200, finish tool_calls, calls ['get_weather'], reasoning 89 chars, markup in content none |
| Reasoning stream (trace in `reasoning_content`, none in `content`) | pass: 17 reasoning chunks, 2 content chunks, markup in content none, answer 391, usage yes, metrics comment yes | pass: 17 reasoning chunks, 2 content chunks, markup in content none, answer 391, usage yes, metrics comment yes |

## Caveats

- One machine, 2 passes for the short prompts and one for the context sweep.
  Thinking was on, so most of each 512-token reply is reasoning.
- The two recipes differ in checkpoint (the MTP/EAGLE checkpoint keeps a BF16 LM head
  and carries the MTP layer), drafter, KV pool, running requests and context: this
  compares recipes, not one setting.
- The kernel caches started empty for each recipe's first start; later starts reused them.
- CapyCTL passed the MTP/EAGLE recipe's 29 GiB KV cache to SGLang as 950,272 tokens
  (`--max-total-tokens`, 32 KiB a token in FP8); the engine held 776,670.
- The MTP/EAGLE configuration has been reported at about 26 to 27 tokens/s per stream at
  one stream, about 387 tokens/s at 22, with an accept length of 2.8 to 2.9. Here, through
  CapyCTL, it decoded 17.9 tokens/s per stream at one stream and 280.7 tokens/s at 22,
  with an accept length of 2.60. The cause is not established yet; `run.sh --variants`
  compares CapyCTL's default `deep` residency (SGLang's memory saver on) with
  `restart_only`, and a KV cache dtype stated against one left to SGLang.
