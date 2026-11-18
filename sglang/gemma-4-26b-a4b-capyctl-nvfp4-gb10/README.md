# Gemma 4 26B-A4B CapyCTL NVFP4 on SGLang 0.5.22 with DFlash drafts, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10, compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Kernel 6.17.0-1031-nvidia, NVIDIA driver 580.173.02, CUDA 13.0 |
| Engine | SGLang 0.5.22 in a venv (torch 2.14.1, flashinfer-python 0.7.1, transformers 5.19.0); installed source commit unknown |
| Model | [`capyctl/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/capyctl/Gemma-4-26B-A4B-NVFP4) at `241dbe07`, r3: searched NVFP4 experts, FP8 dense MLP/attention/untied output head; 17,875,114,764-byte tensor payload |
| Drafter | [`z-lab/gemma-4-26B-A4B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) at `77d4202772dfe50b2396ec7bac9cfffc7b9e7057`, 0.86 GB safetensors; verify width 8 |
| CapyCTL | 0.1.12 release; `capyctl start standalone`; LMCache explicitly off |
| Measured | 2026-11-16–17; `sg-r3-shipped-full-001`, with separate speed and native-quality runs in the comparison |

**Use DFlash width 8 for short/medium prompts; use [vLLM](../../vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/)
or SGLang without a drafter for long prompts.** vLLM MTP3 is the default across contexts.
The final shipped DFlash8 sweep measures **71.69 tokens/s on one stream and 255.15
aggregate at 8**, versus vLLM's **76.36 / 297.10**. Separate three-round selection
sweeps measured 72.04 / 254.05 and 79.56 / 302.29 respectively.
Both shipped settings score **244/250 GSM8K and 30/30 tools**.

On retrieval fixtures, SGLang r3 without drafts measures **38.74 / 37.24 tokens/s
at 96k / 111k**, versus DFlash8's **20.67 / 15.56** (router-reported decode medians).
Removing DFlash improves the 96k retrieval draw but does not qualify all-document
retrieval. The declared 131,072 context is a runtime limit, not a quality guarantee.

NVIDIA's build on SGLang 0.5.22 with a 131k-context production configuration measured
29.47 / 167.67 tokens/s, 243/250 GSM8K and 30/30 tools. That comparison changes
checkpoint and drafter; it is not a checkpoint-only speedup or an accuracy advantage.
The [comparison](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/) contains the controls.

### The checkpoint

The same r3 as the vLLM recipe: NVFP4 W4A4 group-16 routed experts with searched block
scales; per-tensor FP8 E4M3 dense MLP, attention and untied output head; original BF16
embedding, router, norms, layer scalars and vision. The tensor payload is
17,875,114,764 bytes. The [build notes](../../vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/convert/NOTES.md)
record calibration, hashes and format audits. The
[family ablation](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/tensors/ablation/results/summary.md)
is why the always-active projections stay FP8 instead of NVFP4. Actual stored r3 agrees
with BF16 on 97.177% of 62,658 reference next tokens, against NVIDIA's 97.132%.
This weights-only result does not measure native activation/KV/kernel error.

### The settings

- **DFlash verify width 8**, with an unquantized drafter. Stock SGLang 0.5.22 loaded the
  pinned checkpoint without engine patches; target and draft attention use Triton. This
  is not the DFlash card's patched-branch/FA4 setup.
- **Widths 4, 8, 12 and 16 were measured.** Width 8 leads one-stream decode. Width 4's
  258.65 tokens/s at 8 streams does not qualify it: its native GSM8K output repeats a
  phrase through the 2,048-token cap. Width 8 passes its own output review and task gate.
  The narrower-than-trained-width warning is retained, not hidden or patched away.
- **131,072 context, 4,096 chunked prefill, FP8 E4M3 KV**, `flashinfer_cutlass` MoE and
  Gemma4 tool/reasoning parsers. No additional scheduler or graph tuning is claimed.
  The 80 GiB memory request reproduces the comparison grant; it is not a measured minimum.
- **`lmcache: off`.** LMCache 0.5.6 rejects `@` in CapyCTL's pinned Hub model paths,
  causing failed stores/lookups and 300-second lookup timeouts on measured vLLM
  ([issue #259](https://github.com/edurdias/capyctl/issues/259)). SGLang with LMCache
  was not qualified; both recipes keep the explicit off policy.
- **`residency: restart_only`** measures process startup, not parking or waking.
- The mixed loader reports FP8 and NVFP4 components and FP8 GEMM autotuning. It does not
  name every head/projection quant method. The checkpoint recipes and resolver audit
  establish those routes; r3 versus r3e on SGLang supplies a runtime head ablation:
  56.99 versus 43.67 tokens/s without drafts. vLLM separately logs its FP8 head kernel.

## Run it

[Install CapyCTL](https://edurdias.github.io/capyctl/docs/install/) and SGLang 0.5.22.
CapyCTL does not download drafters:

```bash
hf download z-lab/gemma-4-26B-A4B-it-DFlash \
  --revision 77d4202772dfe50b2396ec7bac9cfffc7b9e7057 \
  --local-dir ~/drafters/gemma-4-26B-A4B-it-DFlash

capyctl engine add ~/sglang-0.5.22-venv \
  --approve-option=--speculative-draft-model-path --approve-path ~/drafters
```

Set the drafter path in [`deployment.yaml`](deployment.yaml) to your absolute path.
Start the standalone in one terminal with the comparison's memory grant:

```bash
capyctl start standalone \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB
```

From this recipe directory in another terminal:

```bash
capyctl deploy model --file deployment.yaml
capyctl start deployment gemma4-26b-a4b-capyctl-nvfp4-sglang --wait
```

The file pins the weights commit `241dbe0757d0fca815530e9216de6affb43565ff`; later commits
to the repository change only the model card. The measured run used the same shards from a
hash-verified local copy, with only the local model and drafter paths substituted.
The same-shape SGLang no-draft c1 and c8 diagnostic captures each repeated exactly.
Cross-concurrency and drafter outputs still differ: reproducible inputs/settings do not
promise identical wording across all execution shapes.

## Measured through CapyCTL

| | |
|---|---|
| Ready, process-cold | 148.99 s; OS page caches and compiled kernels were not cleared |
| Ready, warm stop/start | 120.67 s |
| Time to first token, exactly 24k input tokens | 5.348 s at 1 stream; 13.836 s at 4 (selection sweep) |
| Decode, one stream | 71.69 tokens/s median (final shipped sweep) |
| Peak system-used memory | 93.31 GB absolute; 88.42 GB above prelaunch baseline |
| Peak GPU compute allocation | 81.51 GB sampled through startup, serving and warm restart |
| Context | 131,072 declared; shipped sweep targets 500–32,000 input tokens; retrieval probes reach 111,000 |

System-used memory is `MemTotal - MemAvailable`, sampled every 0.5 s, not a model
allocation or CapyCTL `startup.measured` peak. GPU compute allocation is measured
separately. GB is decimal and GiB binary; the memory request is not an allocation.

### Accuracy

| | This recipe | r3, SGLang, no drafts | NVIDIA NVFP4, SGLang, no drafts |
|---|---|---|---|
| GSM8K | 244/250 (97.6%) | 244/250 (97.6%) | 243/250 (97.2%) |
| Answers cut at 2,048 tokens | 0 | 0 | 0 |
| Tool calls | 30/30 | 30/30 | 30/30 |

Width 8 passes the same-engine no-draft 95% Wilson interval gate
(94.864%–98.896%), zero HTTP errors, matching method/data, unchanged tool score and
explicit output review. One question is 0.4 percentage points; no accuracy advantage is
established. This finite gate is not proof of exact speculative equivalence.

## Benchmark

| Streams | Together, tokens/s | Each, decode tokens/s |
|---|---:|---:|
| 1 | 71.08 | 71.69 |
| 2 | 124.62 | 64.45 |
| 3 | 131.49 | 51.17 |
| 4 | 163.38 | 47.80 |
| 5 | 186.72 | 49.17 |
| 6 | 212.32 | 42.85 |
| 7 | 235.98 | 45.48 |
| 8 | 255.15 | 42.86 |

The [matrix](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/bench/final-matrix.json)
and [shipped run](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/bench/runs/sg-r3-shipped-full-001/)
retain requests and native evidence. Five rounds follow one warm-up at each concurrency;
output cap is 512. Together is completion tokens / round wall time; each is decode
after the first token. Context targets have three requests after a warm-up, output cap
128; actual token counts are recorded rather than assumed exact.

| Input target | TTFT s | Decode tokens/s |
|---|---:|---:|
| 500 | 0.109 | 63.67 |
| 1,000 | 0.173 | 58.54 |
| 2,000 | 0.336 | 52.22 |
| 4,000 | 0.675 | 56.74 |
| 8,000 | 1.378 | 52.05 |
| 16,000 | 3.117 | 48.17 |
| 32,000 | 7.836 | 39.37 |

The final sweep has zero benchmark errors and confirms native FP8 KV and LMCache off.
The separate selection sweep's periodic native acceptance medians are 2.51 / 2.93
at 1 / 8 streams (10 / 17 observations), not token-weighted means.

### Functional gates and context limits

Tool-result round trip and vision pass. The initial reasoning probe separates reasoning
but reaches its 2,048-token cap without final content; that cap alone is not a
reasoning-capability verdict. vLLM's separate 8,192-token rerun passes.

DFlash8 retrieves all three codes on 2/6 / 1/6 documents at 96k and 1/6 / 0/6 at
111k (original / one-line). SGLang r3 no-draft improves the 96k draw to 3/6 / 4/6.
On the twelve-document 111k no-draft comparison, r3 and tied-BF16-head r3e both score
6/12 / 4/12; NVIDIA scores 8/12 / 6/12. Paired document tests give p=0.125 for
each candidate versus NVIDIA: neither equivalence nor a significant gap is established.
Both NVIDIA and r3 fail the all-document 111k gate on both engines. No tested length
passes all six documents with both prompts for the shipped DFlash8 recipe.
Ordering and exact-copy failures are separate from missing-code retrieval.
No isolated FP8-head or drafter cause is established.

For the no-drafter long-context alternative, copy the deployment and remove
`--speculative-algorithm`, `--speculative-draft-model-path`,
`--speculative-draft-model-quantization` and `--speculative-num-draft-tokens`,
including each option's value. Keep `--moe-runner-backend flashinfer_cutlass`,
FP8 KV and the other serving settings unchanged.

### More load: Google MTP depth 3

With the same r3 and base settings, Google assistant at
`6e5aaaf4c42b98394530b8fda2e95cadd65c151c` and
`NEXTN`, 3 steps, 4 draft tokens, top-k 1, unquantized draft model:

| | DFlash width 8 (this file) | Google MTP depth 3 |
|---|---|---|
| One stream | 72.04 tokens/s | 62.24 tokens/s |
| 8 streams, total | 254.05 tokens/s | 278.11 tokens/s |
| 24k TTFT at 1 / 4 streams | 5.348 / 13.836 s | 5.227 / 13.760 s |
| GSM8K / tools | 244/250 / 30/30 | 242/250 / 30/30 |

Download the assistant to `~/drafters/gemma-4-26B-A4B-it-assistant`, change
`--speculative-draft-model-path`, replace `DFLASH` with `NEXTN`, change draft tokens to
`4` and add `--speculative-num-steps 3 --speculative-eagle-topk 1` in a copy of the file.
The [measured deployment](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/bench/runs/sg-r3-mtp-3-speed-001/deployment.yaml)
is the exact reference. NEXTN resolves to FROZEN_KV_MTP; SGLang resets max-running requests
to 48. Depth 3 has its own native task-quality approval. Depth 2 is another qualified
MTP point (68.71 / 258.91 tokens/s), not the file shipped here.

## License

Apache-2.0, as the Google source. This independent conversion quantizes weights, searches
expert scales, unties the head and adds serving metadata; it implies no Google or NVIDIA
endorsement. Attribution, modifications and limitations are in the
[model card](../../vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/convert/MODEL_CARD.md#attribution-and-license).
