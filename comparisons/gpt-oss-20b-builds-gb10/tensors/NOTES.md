# FP8-attention checkpoint: agreement result and native qualification

**2026-11-18–19 history:** the initial offline 99.5% gate failed at **98.14843%** agreement. That gate was judged too strict, then native vLLM 0.32.0 qualification passed: **237/250 GSM8K, 30/30 tools, 48/48 retrieval observations and 26.1% faster one-stream decode** than OpenAI on the same engine (57.77 versus 45.81 tokens/s). The checkpoint was then selected for publication. The failed offline threshold remains recorded; native qualification does not make it lossless. OpenAI's own **98.39201%** self-consistency in this harness uses generated IDs, a different target; the common-basis result below preserves that distinction. See [the final comparison](../README.md) and [`bench/fp8-attention/`](../bench/fp8-attention/).

## Layout and audit

Source: OpenAI revision `6cee5e81ee83917806bbde320786a8fb61efebee`. vLLM-only Quark layout: byte-identical MXFP4 experts, including packed bits, E8M0 scales, gate/up interleave and expert biases; per-expert storage. Only 96 attention Q/K/V/O weights become E4M3 FP8, with FP32 per-output-channel multiplicative scales. The head, embedding, routers, norms, sinks and other biases remain unchanged. Native Quark config declares dynamic FP8 input quantization, which this weight-only test does not exercise.

Candidate tensor payload: **13,125,022,848 bytes**. The complete CPU audit covers 459 source and 5,019 candidate tensors, with all 144 source expert tensors byte-identical and all 96 attention weights/scales matching the conversion rule. File bytes/hashes and tensor hashes are in `../../../vllm/gpt-oss-20b-capyctl-gb10/convert/checkpoint-audit.json`; `quark-config-check.json` records successful installed vLLM 0.32 CPU config parsing, not an engine load.

[INFERENCE: batch-one weight-traffic arithmetic] The candidate saves **636,241,920 bytes/token (about 636 MB, 17.16%, rounded 17.2%)**, reducing estimated weight traffic from 3,708,089,088 to 3,071,847,168 bytes/token. This arithmetic is not a speed measurement; the separate native results above establish the measured improvement.

## Method

`agreement-prompts.json` is the unchanged Gemma corpus: 200 records, 40 each coding/math/chat/multilingual/tool-use, preserving IDs, messages and tools. SHA-256: `69615834a09006f82c47daa8b6723ed905c372a422456f714962cfcceccc97bc`. GPT-OSS's pinned Harmony template renders it with effort low. Generate greedily, batch 8, at most 512 tokens; EOS IDs are 199999, 200002 and 200012.

The teacher is Transformers' BF16 reconstruction of OpenAI's released MXFP4 checkpoint, not an unavailable prequantization checkpoint. After reference generation, recompute the teacher's predictions at every answer position using `prompt_ids + answer_ids[:-1]`. Then replace only attention weights in that same ordinary BF16 graph with `candidate_fp8.float() * channel_scale`, cast once to BF16, and recompute predictions. Score candidate against the actual teacher-forced teacher, and report both models against the same generated IDs as a separate comparison.

Installed Transformers 5.14.1 rejects GPT-OSS SDPA before model loading; attempt 001 records that failure. The successful run uses its supported eager-attention implementation identically for both sides, with torch 2.13.0+cu130, all 411 parameter tensors BF16, expert alpha 1.702 and clamp 7 intact. No engine package was patched or installed. This isolates weight error; it does not qualify Quark kernels or FP8 activation quantization.

The integration owner explicitly cleared the exclusive GPU slot before execution. Exact command: stage this directory's harness/corpus and the existing Gemma `memory_guard.py` under `/home/me/gptoss-work`, then execute `/home/me/gptoss-work/run-agreement.sh` on the measured GB10. The harness refuses an occupied GPU and preserves an 8-GiB host-memory margin. The successful run exited zero in 5,708.41 seconds; GPU compute processes were empty afterward.

## Results

| Candidate vs teacher-forced teacher | Agree / positions | Agreement |
|---|---:|---:|
| Overall | 51,577 / 52,550 | **98.14843%** |
| Chat | 14,833 / 15,190 | 97.64977% |
| Coding | 17,866 / 18,186 | 98.24040% |
| Math | 6,241 / 6,306 | 98.96924% |
| Multilingual | 10,664 / 10,887 | 97.95169% |
| Tool use | 1,973 / 1,981 | 99.59616% |

On the **same generated-ID basis**, the actual teacher-forcing baseline is **51,705 / 52,550 = 98.39201%**, while the candidate is **51,381 / 52,550 = 97.77545%**. Do not assume the teacher has 100% agreement with its cached batched generation. This common-basis difference is about 0.62 percentage points.

Against the actual teacher-forced teacher, per-channel FP8 attention costs GPT-OSS **1.85 percentage points (about 1.9)** of next-token agreement, versus about **0.25 points for Gemma 4** in the comparison precedent. The initial 99.5% offline gate failed. It was later judged too strict and superseded for selection by the separately completed native quality/performance qualification described above; the measured agreement is unchanged.

All 200 records and every reference position were scored. **64 records hit the cap; only 136 completed.** Capped positions remain in the denominator, and cap-hit IDs are listed separately in `score-fp8attn.json`; they are not counted as successful completed answers. Median first disagreement is 29.5 tokens. Full answer IDs, picked IDs, agreement masks and logit margins are retained in `bf16-answers.json`, `score-bf16.json` and `score-fp8attn.json`.

## NVFP4 finding

The CPU diagnostic finds **100% of 597,196,800 blocks** (all 48 scale tensors) losslessly representable under ModelOpt's sufficient closed-form scale window; see `../../../vllm/gpt-oss-20b-capyctl-gb10/convert/cast-share.json`. This is a representability calculation, not an exported-weight comparison, and no NVFP4 checkpoint was built. NVFP4 still increases expert weight/local-scale storage by **5.88%**. Installed-source inspection finds no bias-carrying GPT-OSS NVFP4 MoE path in **SGLang 0.5.22 or vLLM 0.32**. The lossless-block result therefore does not make NVFP4 a correct or bandwidth-saving candidate.
