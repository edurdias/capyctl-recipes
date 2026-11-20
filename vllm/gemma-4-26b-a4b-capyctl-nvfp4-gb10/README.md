# Gemma 4 26B-A4B CapyCTL NVFP4 on vLLM 0.32.0 with Google MTP drafts, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10, compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Kernel 6.17.0-1031-nvidia, NVIDIA driver 580.173.02, CUDA 13.0 |
| Engine | vLLM 0.32.0 in a venv (torch 2.13.0+cu130, flashinfer-python 0.7.0.post1, transformers 5.19.0); installed source commit unknown |
| Model | [`capyctl/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/capyctl/Gemma-4-26B-A4B-NVFP4) at `241dbe07`, r3: searched NVFP4 routed experts, FP8 dense MLP/attention/output head; 17,875,114,764-byte tensor payload |
| Drafter | [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) at `6e5aaaf4c42b98394530b8fda2e95cadd65c151c`, 0.84 GB safetensors; MTP depth 3 |
| CapyCTL | 0.1.12 release; `capyctl start standalone`; LMCache explicitly off |
| Measured | 2026-11-16–17; `vl-r3-shipped-full-001`, with separate speed and native-quality runs in the comparison |

**Use vLLM MTP depth 3 across context lengths.** The final shipped sweep measures
**76.36 tokens/s on one stream and 297.10 tokens/s aggregate at 8 streams**.
The separate three-round selection sweep measured 79.56 / 302.29.
Native task quality is **244/250 GSM8K and 30/30 tool calls**.
NVIDIA's NVFP4 build on SGLang 0.5.22 with its 131k-context serving settings, no drafter measured
29.47 / 167.67 tokens/s, 243/250 GSM8K and 30/30 tools. This changes checkpoint,
engine and drafter; it is not a checkpoint-only speedup or an accuracy advantage.

The [comparison](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/) includes
no-draft controls. The [SGLang DFlash8 recipe](../../sglang/gemma-4-26b-a4b-capyctl-nvfp4-gb10/)
is an alternative for short/medium prompts. For long prompts, use vLLM or SGLang
without a drafter: SGLang r3 no-draft measured **38.74 / 37.24 tokens/s at 96k / 111k**,
versus DFlash8's **20.67 / 15.56** on the retrieval fixtures. These are router-reported
decode medians, not an all-document retrieval qualification.

### The checkpoint

r3 is made from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` by [`convert/`](convert/):

| Tensors | Stored format |
|---|---|
| Routed experts: 11,520 gate/up/down matrices | NVFP4 W4A4, group 16, searched block scales |
| Dense MLP: 90 projections; attention: 115 projections | FP8 E4M3 W8A8, scalar FP32 weight and calibrated static input scales |
| Separate, untied output head | The same per-tensor FP8 format |
| Input embedding, router, norms, learned layer scalars, vision | Unchanged BF16 |
| KV cache | FP8 declared in metadata; no calibrated K/V scale tensors are added |

The [family ablation](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/tensors/ablation/results/summary.md)
explains the FP8 body. Keeping the searched NVFP4 experts and BF16 dense MLP/attention
agrees with BF16 on 97.397% of reference next tokens. Quantizing only the dense MLP to
NVFP4 lowers that to 96.455%; quantizing only attention lowers it to 95.512%.
Per-tensor FP8 for both reaches 97.151%. The exported r3, including its untied FP8 head,
agrees on **97.177%**, against NVIDIA's 97.132%, Unsloth's 96.018% and RedHat's 94.476%.
These are weights reconstructed into BF16, not native quantized-kernel accuracy scores.

r3e keeps the same FP8 body and a tied BF16 head. On SGLang without drafts, r3 decodes
56.99 tokens/s against r3e's 43.67; their offline agreement is 97.177% and 97.191%.
The head/tie change has a measured decode effect, while its agreement difference is small.

[INFERENCE] Counting one weight read, eight selected experts per layer and the output
head gives 3.208 GB/token for r3 against NVIDIA's 5.592 GB/token. This is payload
arithmetic, not measured bandwidth; it excludes activations, KV traffic and cache reuse.
The checkpoint contains 17,875,114,764 bytes of tensor payload and 17,924,353,176 bytes
of regular files in the completed local build. [`convert/NOTES.md`](convert/NOTES.md)
records the sizes, shard SHA-256s, calibration audit and build commands;
[`convert/MODEL_CARD.md`](convert/MODEL_CARD.md) is its Hugging Face card.

### The settings

- **Google assistant MTP depth 3.** vLLM repeats the assistant step; these are not three
  separate prediction heads. Depths 1–4 were measured. Depth 3 led one-stream decode and
  nominal 8-stream throughput; its 302.29 at 8 is close to depth 2's 299.78 from three rounds.
  The selected depth has its own native task-quality approval.
- **131,072 context, 4,096 batched tokens, FP8 E4M3 KV.** These match the controlled
  comparison. The 40 GiB KV allocation is a measured comparison budget, not a minimum
  memory requirement. The engine reports 2,460,946 cache tokens; that is capacity, not a
  measured request limit. No scheduler or CUDA-graph override was added.
- **`lmcache: off`.** LMCache 0.5.6 rejects `@` in CapyCTL's pinned Hub model paths,
  causing failed stores/lookups and 300-second lookup timeouts in the measured vLLM
  environment ([issue #259](https://github.com/edurdias/capyctl/issues/259)).
  Automatic policy enables it when installed; omission is not equivalent to off.
- **Gemma4 tool and reasoning parsers.** Tool accuracy goes through the real automatic
  tool parser, not forced-call grammar. Benchmarks use the source chat template's default
  thinking policy; they do not force thinking off.
- The native r3 log names **`FlashInferFP8ScaledMMLinearKernel`** for the output head,
  attention and dense projections, **FLASHINFER_CUTLASS** for NVFP4 MoE and **TRITON_ATTN**.
  Missing KV-scale warnings remain in the evidence: the fallback scale is 1.0, not calibrated.
- **`residency: restart_only`.** These runs measure serving after a new process starts,
  not parking or wake behavior.

## Run it

[Install CapyCTL](https://edurdias.github.io/capyctl/docs/install/) and the engine version
above. Download the drafter yourself; CapyCTL does not download drafters:

```bash
hf download google/gemma-4-26B-A4B-it-assistant \
  --revision 6e5aaaf4c42b98394530b8fda2e95cadd65c151c \
  --local-dir ~/drafters/gemma-4-26B-A4B-it-assistant

capyctl engine add ~/vllm-0.32.0-venv \
  --approve-option=--speculative-config --approve-path ~/drafters
```

Set the drafter path in [`deployment.yaml`](deployment.yaml) to your absolute path. Run
CapyCTL in one terminal; the memory grant below is the comparison's, not a general default:

```bash
capyctl start standalone \
  --set host.resource_policy.memory.system.managed_limit=100GiB \
  --set host.resource_policy.memory.system.free_reserve=10GiB
```

In another terminal, from this recipe directory:

```bash
capyctl deploy model --file deployment.yaml
capyctl start deployment gemma4-26b-a4b-capyctl-nvfp4-vllm --wait
```

The file pins the weights commit `241dbe0757d0fca815530e9216de6affb43565ff`; later commits
to the repository change only the model card. The measured run used the same shards from a
hash-verified local copy, with only the local model and drafter paths substituted.

## Measured through CapyCTL

| | |
|---|---|
| Ready, process-cold | 261.79 s; compiled kernels and OS page caches were not cleared |
| Ready, warm stop/start | 184.99 s |
| Time to first token, exactly 24k input tokens | 7.277 s at 1 stream; 18.857 s at 4 (selection sweep) |
| Decode, one stream | 76.36 tokens/s median (final shipped sweep) |
| Peak system-used memory | 79.14 GB absolute; 75.19 GB above prelaunch baseline |
| Peak GPU compute allocation | 67.71 GB sampled across startup, serving and warm restart |
| Context | 131,072 tokens declared; shipped sweep targets 500–32,000 input tokens; separate retrieval probes reach 111,000 |

System-used memory is `MemTotal - MemAvailable`, sampled every 0.5 s. It is not a
model allocation or CapyCTL `startup.measured` peak. GPU compute allocation is a
separate sampled measurement. The 40 GiB KV cache is not the whole footprint.
GB is decimal; GiB is binary.

### Accuracy

| | This recipe | r3, vLLM, no drafts | NVIDIA NVFP4, SGLang, no drafts |
|---|---|---|---|
| GSM8K | 244/250 (97.6%) | 244/250 (97.6%) | 243/250 (97.2%) |
| Answers cut at 2,048 tokens | 0 | 0 | 0 |
| Tool calls | 30/30 | 30/30 | 30/30 |

One GSM8K answer is 0.4 percentage points; this sample does not establish an accuracy
advantage. The no-draft reference's 95% Wilson interval is 94.864%–98.896%. Drafter
qualification also requires coherent reviewed outputs, matching method/data, zero HTTP
errors and unchanged tool score. It is not proof of exact speculative equivalence.
Fresh-process vLLM no-draft replies differed on all eight diagnostic prompts; exact greedy
wording is therefore not used as the decisive quality gate.

## Benchmark

The [matrix](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/bench/final-matrix.json)
and [final shipped run](../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/bench/runs/vl-r3-shipped-full-001/)
retain the numeric sources, requests, deployment and native evidence.

| Streams | Together, tokens/s | Each, decode tokens/s |
|---|---:|---:|
| 1 | 75.30 | 76.36 |
| 2 | 129.88 | 68.25 |
| 3 | 146.54 | 53.15 |
| 4 | 184.42 | 50.21 |
| 5 | 199.54 | 48.19 |
| 6 | 246.11 | 47.37 |
| 7 | 263.77 | 43.95 |
| 8 | 297.10 | 44.54 |

Five recorded rounds follow one warm-up per concurrency, with a 512-token output cap.
Together is completion tokens / round wall time; each is per-request decode after the
first token. Context targets use three requests per point, one warm-up and a 128-token
output cap; actual token counts are recorded rather than assumed exact.

| Input target | TTFT s | Decode tokens/s |
|---|---:|---:|
| 500 | 0.129 | 70.00 |
| 1,000 | 0.166 | 67.01 |
| 2,000 | 0.267 | 61.97 |
| 4,000 | 0.507 | 53.87 |
| 8,000 | 1.215 | 52.94 |
| 16,000 | 3.518 | 41.64 |
| 32,000 | 13.118 | 29.45 |

Native FP8 KV and LMCache off were confirmed; the shipped sweep had zero benchmark errors.
The separate selection sweep's native acceptance medians were 2.68 / 2.76 at 1 / 8
streams from one / three periodic observations, not token-weighted means.

### Functional gates and context limits

Tool-result round trip, vision, and reasoning with an 8,192-token budget pass.
The earlier 2,048-token reasoning cap produced no final answer; the larger-budget
rerun supersedes it. Short exact-copy testing passes 100/100 at 2,000 input tokens.
Neither that result nor the 131,072 runtime limit qualifies long-context retrieval.
Across six documents and two prompt variants, MTP3 retrieves all three codes on
4/6 / 3/6 at 111k (original / one-line). Both NVIDIA and r3 fail the all-document
111k gate on both engines. Ordering and exact-copy errors are scored separately.
No tested context length passes every document with both prompts for this shipped
drafter recipe; choose context limits for the application's own retrieval checks.

## License

Apache-2.0, as the Google source. CapyCTL quantized the weights, searched expert block
scales, untied and quantized the output head, and added serving metadata. This is an
independent conversion; Google and NVIDIA do not endorse it. See the
[model card](convert/MODEL_CARD.md#attribution-and-license) for attribution and limitations.
