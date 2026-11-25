# Nemotron 3.5 Super VL NVFP4 with native MTP on GB10

I elect NVIDIA's unchanged checkpoint on SGLang 0.5.22 with native MTP and the measured bounded pools. **Retrieval is measured through 64k input tokens, not 96k/111k.** The corrected longer MTP retrieval run caused a hard freeze; no log was recovered for that attempt. Earlier the same day, a separate vLLM automatic-KV launch caused a confirmed OOM. I retain both findings and both seeded retrieval misses in the [comparison](../../comparisons/nemotron-3.5-super-vl-drafters-gb10/).

| Item | Value |
|---|---|
| Hardware | One GB10, 128 GB unified memory |
| Checkpoint | [NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tree/9838c91537e6be21dec2f5276b784211a06fbeea) |
| Revision | `9838c91537e6be21dec2f5276b784211a06fbeea` |
| Engine | Installed SGLang 0.5.22; no engine patch |
| Drafter | Checkpoint's native MTP; EAGLE, 3 steps, top-k 1, 4 draft-tree tokens |
| Pools | Static fraction 0.78; 262,144 total tokens; 32 Mamba states |
| Attention / state | FlashInfer; FP8 E4M3 KV; BF16 Mamba state |
| Scheduling | 2 running requests; 1,024-token prefill chunks; 8,192 maximum prefill tokens |
| Parsers | `nemotron_3` reasoning; `qwen3_coder` tool calls |
| Context setting | 73,728 total tokens: the 65,536-token qualified input plus 8,192 output tokens (measured runs used 114,688; lowered so the engine rejects much longer inputs) |
| Measured | 2026-11-23, direct engine, greedy with default thinking |

## Why this is a native launch, not a deployment.yaml

CapyCTL 0.1.12 manages `--mem-fraction-static` from its resource grant and reserves that option against extra-argument overrides. A memory request is not an exact fixed share of available memory. I cannot honestly reproduce the measured **0.78** share and bounded pools with a deployment document; [CapyCTL issue #264](https://github.com/edurdias/capyctl/issues/264) tracks recipes that fix a memory share.

I ran `capyctl validate config --file` from the requested 0.1.12 release against an exact-profile deployment candidate. It returned `valid: true` as a **document-only** check, with `resolved_against: null` and host/profile/resource checks still required. That is not permission to pass the reserved option at launch, nor evidence of matching pool allocation. The candidate and raw validator result are archived privately; the [publication audit](../../comparisons/nemotron-3.5-super-vl-drafters-gb10/publication-audit.json) records the result. I ship [launch.json](launch.json) and [launch.py](launch.py) instead of an invalid launch promise.

These measurements are not through CapyCTL. Hub provisioning, this packaged launcher, cold/warm readiness, parking/wake, and a CapyCTL-served first answer have not been exercised. Offline validation does not qualify an engine.

## Download and launch

Use an installed SGLang 0.5.22 environment with its compatible CUDA/FlashInfer stack. `huggingface_hub` is needed for download. The manifest pins the full Hub revision and lists every runtime file explicitly: all 88 weight shards, their index, tokenizer/chat files, processor configurations, and publisher Python modules. NVIDIA's checkpoint is used as is, including its pinned processor correction; there is no conversion or checkpoint upload. The [weight audit](../../comparisons/nemotron-3.5-super-vl-drafters-gb10/checkpoint-audit.json) matched all 88 weight hashes (75,010,188,240 bytes).

From this directory:

```sh
/home/me/sglang-0.5.22-venv/bin/python -B launch.py download \
  --model-dir /home/me/models/Nemotron-3.5-Super-VL-120B-A12B-NVFP4

python3 -B launch.py serve \
  --model-dir /home/me/models/Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --python /home/me/sglang-0.5.22-venv/bin/python
```

The launcher preserves the measured model arguments, served model name `nemotron-super`, port 8350, and loopback binding via SGLang's default. It sets `MAX_JOBS=2`, puts the engine environment on `PATH` for kernel compilation, and disables Python bytecode. It refuses an occupied compute device and requires a working user systemd manager. Each invocation gets its own scope with `MemoryMax=95G` and swap disabled. A 90 GiB planned footprint must leave at least 20 GiB available before launch; a separate 4 GiB MemAvailable guard stops only that invocation's scope. Ctrl-C also stops that owned scope, not other services.

**Those stricter operator safeguards have not been engine-measured.** The completed measurements used a 100 GiB cap, and the later long-probe attempt still froze despite its cap and guard. [INFERENCE] The 90 GiB plan combines observed weights, bounded pools, and runtime allowance; it is not a measured maximum. Do not rely on polling or cgroup accounting alone to prevent another hard freeze. Run only one engine at a time. **Keep the complete rendered input at or below 64,000 tokens; do not launch 96k/111k probes from this recipe.** This recipe sets `--context-length 73728` (the measured runs used 114,688) so the engine itself rejects longer inputs; the memory pools are set separately and unchanged, and the lower limit was not re-measured.

Once the engine reports readiness, set `ENDPOINT` to its loopback API endpoint and send a short request, for example:

```sh
curl "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-super","messages":[{"role":"user","content":"What is 6 times 7?"}],"temperature":0,"max_tokens":512}'
```

This is a reproduction command, not an observed output. Thinking remains at the model default. For request budgeting, MTP reserves four slots: `max_tokens <= min(32768, 73728 - input_tokens - 4)`.

## Retained measurements

| Metric | NVIDIA NVFP4 · SGLang 0.5.22 native MTP |
|---|---:|
| c1 aggregate / post-first-token decode | 35.50 / 36.24 tokens/s |
| c1 time to first token | 0.26 s |
| c2 aggregate / per-user decode | 51.44 / 26.49 tokens/s |
| c4 / c8 aggregate, two engine slots | 49.75 / 50.80 tokens/s |
| 8,192 / 32,768 / 65,536 input-token decode | 33.74 / 35.42 / 36.43 tokens/s |
| Sampled peak system use / minimum available | 88.08 / 33.61 GiB, completed phases only |
| GSM8K / tools / HumanEval | 242/250 / 30/30 / 161/164 |
| Retrieval 32k / 64k | 5/6 / 6/6 |
| Copy, 100 deterministic 2k-token cases | 100/100 |

Timings are medians of three measured rounds after warmup, with a 512-output-token cap; aggregate includes time to first token, decode excludes it. c4/c8 include queuing at the two-request engine limit. Memory is sampled system-wide unified memory, not a GPU-only allocation or the later frozen phase's peak.

The full task gate passes against **the same NVIDIA checkpoint on SGLang 0.5.22 without a drafter** using the established Wilson95 accuracy/truncation bounds, complete matching methods and case sets, zero HTTP errors, tools 30/30, and explicit output review. All full suites have zero HTTP errors and truncations. b has two paired GSM8K losses and one HumanEval gain; both rows share helper-function HumanEval limitations. The [comparison](../../comparisons/nemotron-3.5-super-vl-drafters-gb10/) gives the cases, bounds, and retained scores without rescoring them.

The 32k MTP retrieval miss is real; the no-drafter row misses the analogous seeded document at 64k. Neither is erased. A successful 110k decode-speed probe is retained in the comparison, but it does not qualify 96k/111k retrieval or safe operation. Image/video, endurance, maximum context/concurrency, and parking remain unqualified. The existing `vllm/nemotron-3.5-super-vl-120b-a12b-nvfp4-gb10` recipe needs a memory-safety review; its earlier success is not evidence that it meets this reserve.
