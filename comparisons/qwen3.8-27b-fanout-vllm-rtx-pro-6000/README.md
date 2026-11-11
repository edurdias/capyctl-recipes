# Qwen3.8-27B NVFP4, one coding task fanned out to 8 lanes, vLLM 0.32.0 on an RTX PRO 6000

## Summary

On 2026-11-10 Ash Hart [posted](https://x.com/ashxhart/status/2120292846431347076) about 700
tokens/s from Qwen3.8-27B NVFP4 on one RTX 5090 with TensorFold: one prompt ("build me a html
three.js website for harry potter") split into 8 parallel lanes, merged into one file, DFlash2
drafts, thinking off. His stat line: 8,859 tokens in 13.32 s, 764 tokens/s decode, 665.3
tokens/s end to end.

This page runs the same shape through CapyCTL with this repository's vLLM recipe
([vllm/qwen3.8-27b-capyctl-nvfp4-gb10](../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/)) on one
RTX PRO 6000 Blackwell (no 5090 could be rented that day), at two memory sizes:

- **A, sized for a 32 GB card**: 5 GiB KV cache; the engine used 29.7 GiB of the GPU.
- **B, the recipe's memory**: 16 GiB KV cache; 41.4 GiB.

| Combined tokens/s, end to end (median of 3 rounds, range) | A: 32 GB-sized | B: recipe memory |
|---|---:|---:|
| 8-lane fan-out, lanes capped at 2,048 tokens (run 1) | **849** (841 to 860) | **1,566** (1,472 to 1,581) |
| 8-lane fan-out, lanes end on their own (run 2) | 712 (670 to 767) | 1,102 (1,101 to 1,272) |
| The same task as one stream (run 2) | 311 | 309 |
| 8 unrelated coding prompts at once, 1,024 tokens (run 1) | 1,020 (978 to 1,073) | 1,424 (1,402 to 1,426) |
| 1 coding prompt, 1,024 tokens (run 1) | 323 | 327 |
| *Ash: 5090, TensorFold, 8 lanes* | *665* | |

- **The fan-out number depends on memory.** At 32 GB, vLLM holds 4 to 6 lanes at once
  (its log: 4 to 6 running, up to 3 waiting) and queues the rest, so the slowest lane ends 7 to
  15 s after the median one. With 16 GiB of KV cache all 8 run together.
- **Each lane decodes at about 230 to 240 tokens/s** in both, against 310 for one stream;
  DFlash2 accepted 5.5 to 6.1 tokens a step on this code.
- **Fan-out wrote more, not faster, in this test.** Left to finish, the 8 lanes wrote 20,452
  to 23,485 tokens in all, the single stream 7,972 to 9,657 for the same task. Time to the
  last token of the page:

  | Wall time to a finished page (run 2) | A | B |
  |---|---:|---:|
  | 8-lane fan-out | 26.7 to 33.0 s | 17.7 to 19.9 s |
  | One stream | 25.7 s | 31.3 s |

- **No run produced a working page.** Both modes renamed the shared objects the contract
  provides (`camera` became `b1_camera`; the single stream wrote `b1_setup`), so every page
  stops on a `ReferenceError` in bucket 1. The prompt's naming rule was misread the same way
  in both modes; this is not evidence for or against fan-out, and it is not a measured
  quality comparison.
- **An RTX PRO 6000 is an upper bound for a 5090.** Same GB202 chip and the same 1,792 GB/s
  of GDDR7, but 188 SMs against 170, a higher boost clock and 4,000 against 3,352 FP4 TOPS.
  One stream is bandwidth-bound and should carry over closely; the 8-lane totals lean on
  compute and likely read high for a 5090.

## Setup

| | |
|---|---|
| Machine | Verda `1RTXPRO6000.30V.CC`, FIN-03: one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB, compute capability 12.0), 30 vCPU, 90 GB RAM, Ubuntu 24.04.5, driver 595.91.07 (CUDA 13.2), Confidential Computing on |
| Engine | vLLM 0.32.0 in a fresh venv on uv's Python 3.12.16: torch 2.13.0+cu130, triton 3.7.1, flashinfer-python 0.7.0.post1, transformers 5.19.0 (the recipe's pins) |
| CapyCTL | 0.1.12 release binary, `capyctl start standalone` |
| Model | [`capyctl/Qwen3.8-27B-NVFP4`](https://huggingface.co/capyctl/Qwen3.8-27B-NVFP4) at `0ce7e69f126b6b6f6fd2058373d9f81ec360adee`, both shards checked against the Hub's SHA-256 |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, 7 speculative tokens |
| Deployment | [deployment.yaml](deployment.yaml): the recipe's settings (fp8 KV cache, bf16 DeltaNet state cache, `language_model_only`, 32,768-token context, 8 concurrent requests) with `kv_cache: 5GiB` for A; B is the same file with `kv_cache: 16GiB` |
| Requests | through the CapyCTL endpoint on the machine, streamed, `temperature: 0`, `chat_template_kwargs: {enable_thinking: false}` |
| Measured | 2026-11-11 |

[fanout.py](fanout.py) is the client:

- **fanout**: a system prompt with the task, a shared contract and the 8 buckets
  (scene setup and lights, castle, terrain and lake, sky, Quidditch pitch, particles,
  Hogwarts Express, HTML overlay), then 8 concurrent requests, "Write bucket #N only". The
  replies are concatenated in order into one `<script type="module">`
  (`fanout-rN.html`, `fanout-rN.js`).
- **whole**: the same system prompt and one request for all 8 buckets.
- **batch**: 1 or 8 unrelated coding prompts.

Per round it records total completion tokens (the engine's usage), wall time from the first
request sent to the last stream's end (`e2e_tok_s`), the window from the first streamed token
to the last stream's end (`decode_tok_s`), each lane's own rate, and `tail_s`, the time the
last lane ran past the median lane's end.

## Runs

- **Run 1** (`results/A-32g-v1`, `results/B-recipe-16g`): lanes capped at 2,048 tokens, as
  close as this client gets to Ash's run (his lanes averaged about 1,100 tokens). Half the
  lanes reached the cap (`finish: length`), so these pages are cut off. Bucket 1 was asked
  to create the scene, camera and renderer.
- **Run 2** (`results/A-32g-v2`, `results/B-16g-v2`): the page template creates `scene`,
  `camera`, `renderer`, `controls`, `clock`, `updaters` and the render loop, and bucket 1
  sets up lights, fog and the camera; lanes capped at 6,144 tokens (one lane in one round
  reached it), the single stream at 24,000. [fanout.py](fanout.py) and [serve.sh](serve.sh)
  are run 2's.

Every `*.jsonl` holds each round's figures, including per-lane tokens, end times and finish
reasons.

## Confidential Computing

The only RTX PRO 6000 available was Verda's CC image. Two things were needed there, neither
from CapyCTL:

1. The GPU starts "not ready" (`cudaGetDeviceCount` error 802) until the guest runs
   `nvidia-smi conf-compute -srs 1`.
2. vLLM produced garbage for every model, including Qwen3-0.6B BF16 (for example
   "ductductduct…"), with every NVFP4 kernel, NVFP4 emulation, eager mode and Triton
   attention, while Hugging Face transformers answered correctly on the same GPU. vLLM's
   V1 GPU worker passes inputs through pinned host memory mapped into the GPU (UVA, `UvaBuffer`
   in `vllm/v1/worker/gpu/buffer_utils.py`), and a CC guest's GPU cannot read that memory
   [INFERENCE from the fix: no CC documentation was read]. Reporting UVA as unavailable
   (`vllm.utils.platform_utils.is_uva_available`, [nouva/sitecustomize.py](nouva/sitecustomize.py))
   makes vLLM use its non-UVA buffers, and both models then answer correctly. CapyCTL does not
   let a deployment set `PYTHONPATH`, so the hook was a one-line `.pth` file in that machine's
   vLLM environment ([setup.sh](setup.sh)). [bisect.sh](bisect.sh) is the hand-launch check
   used to find it.

Before the fix the engine also stopped on `AssertionError: request.num_output_placeholders >= 0`
(`async_scheduler.py`) under KV pressure; with it, no run hit the assertion.

## Checks

- CPU and Fake-engine tests are not qualification, and neither is this page: one rented
  machine, one day, a CC-specific hook outside CapyCTL.
- Not compared: TensorFold's own fan-out (not released when this ran), DSpine, a real
  RTX 5090, and GB10. For scale, the recipe on one GB10 decodes 33.9 tokens/s on one stream
  and 177 at 8 (its [README](../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/)).
- Temperature 0 throughout; Ash's sampling settings are not known.
