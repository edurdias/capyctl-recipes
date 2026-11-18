# Closer to BF16: greedy next-token agreement (README section draft)

## Method

This comparison runs offline on host A in the existing `~/modelopt-0.47.0-venv`, using torch and Transformers, not an inference engine. It follows the Qwen comparison's greedy-reference/teacher-forcing method, with the **same 200 prompts**: 40 each of chat, coding, math, multilingual and tool-use, including the original tool schemas. `agreement-prompts.json` is copied unchanged from `capyctl-recipes/comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy/agreement-prompts.json`.

The pinned Gemma BF16 tokenizer renders every prompt using its canonical chat template, with `enable_thinking=False`. BF16 generates up to 512 tokens greedily, with BF16 KV cache, source EOS tokens and eight left-padded requests per generation batch. The reference JSON retains the unpadded prompt IDs and generated answer IDs, including the first EOS when present. BF16 is generated once; its agreement is **100% by construction**, not a second teacher-forced execution or a measurement of numerical self-consistency.

Each quantized build is streamed into the same source architecture, reconstructing stored weights into BF16. The existing Gemma `convert/read_tensors.py` reader decodes E2M1 low/high nibbles, FP8 block scales of group size 16 and global scales; compressed-tensors global scales are reciprocated. Per-expert gate/up/down matrices are restored to Transformers' fused banks, persistent buffers are preserved, and embedding/head tying follows the checkpoint's text config. FP8 weights use either scalar FP32 scales (per-tensor, as in r3) or per-output-channel scales `[out_features,1]` (as in Unsloth); they are **not** decoded as NVFP4. All forward execution uses BF16 weights and activations (with Transformers' usual higher-precision reductions), SDPA attention and torch grouped matrix multiplication for experts.

For every build, each saved BF16 prompt plus answer prefix is teacher-forced as a single sequence. Logits at prompt position `n_prompt - 1` predict answer token zero. The denominator and exact `(prompt, answer position)` pairs are identical across all builds. Agreement is pooled by answer-token count, both overall and within each category, rather than averaged across prompts. `first_disagreement` is zero-based; a fully agreeing answer receives its answer length, so the median is censored by those lengths. The result JSON retains picked token IDs and position-by-position agreement masks.

**Limit:** this is a weights-only comparison. Activation quantization, quantized KV caches, inference engines' loaders, quantized kernels and kernel-specific numerical effects are not modelled. Token disagreement can be a synonym or formatting choice, not necessarily a wrong answer. This is not a task-accuracy benchmark, an engine qualification, or a throughput comparison.

## Reproduce

Run these commands **on host A** from this `tensors/` directory. The existing converter helpers must be available in the recipe's `convert/` directory or in `~/gemma4-work/convert/`. Nothing installs packages or changes an environment.

```bash
bash run-tensors.sh bf16
bash run-tensors.sh r1
bash run-tensors.sh r2
bash run-tensors.sh nvidia
bash run-tensors.sh redhat
bash run-tensors.sh unsloth
bash run-tensors.sh r3
bash run-tensors.sh r3e
```

With no arguments, `bash run-tensors.sh` runs BF16, r1, r2, NVIDIA, RedHat, Unsloth, r3 and r3e in that order. Each command writes `results/<build>.log` and `results/score-<build>.json`; BF16 also writes `results/bf16-answers.json`. `summarize.py` checks reference-file digests and per-prompt position counts before writing `results/summary.md` and `results/summary.json`. Its partial table after each command contains only completed builds.

Every GPU command runs inside the existing `convert/memory_guard.py` 8-GiB host-`MemAvailable` guard. It only signals its own process if needed. Another workload's engine remains resident and is never signalled. CUDA allocated/reserved peaks, peak process RSS, minimum observed host memory and load/evaluation/wall times are saved per build. `/usr/bin/time -v` records whole-command timings and exit status in each log.

## Model pins

| Build | Host-A directory under `~/models/` | Revision |
|---|---|---|
| BF16 | `gemma-4-26b-a4b-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| CapyCTL r1 | `Gemma-4-26B-A4B-NVFP4-capyctl-r1` | Local conversion; manifest in checkpoint |
| CapyCTL r2 | `Gemma-4-26B-A4B-NVFP4-capyctl-r2` | Local searched-scale revision; manifest in checkpoint |
| CapyCTL r3 | `Gemma-4-26B-A4B-NVFP4-capyctl-r3` | Local mixed-FP8/searched-NVFP4 revision with untied FP8 head; manifest in checkpoint |
| CapyCTL r3e | `Gemma-4-26B-A4B-NVFP4-capyctl-r3e` | Local mixed-FP8/searched-NVFP4 revision with tied BF16 head; manifest in checkpoint |
| NVIDIA | `gemma-4-26b-a4b-nvidia-nvfp4` | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` |
| RedHat | `gemma-4-26b-a4b-redhat-nvfp4` | `5557756b8dce33ac72f2bd702b11729fdba3b839` |
| Unsloth | `unsloth--gemma-4-26B-A4B-it-NVFP4@20df0542b1a86ce19f495ac2eca2c7c12bce82f9` | `20df0542b1a86ce19f495ac2eca2c7c12bce82f9` |

## Measured results

All eight stored-build commands completed on 2026-11-16, using torch `2.13.0+cu130` and Transformers `5.14.1`. The reference contains **62,658 answer-token positions across all 200 prompts**. Every report records the same reference SHA-256, `ac75c312aa1342efc482bbc1db2e772d3a7eb6f344a049692850c1522a96ee3c`, and the summarizer checked every prompt ID and answer length.

| Build | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (by construction) | 100.000 | 331.5 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 |
| CapyCTL r1 | 94.515 | 19 | 92.274 | 95.453 | 98.381 | 92.778 | 99.801 |
| CapyCTL r2 | 94.805 | 20.5 | 92.817 | 95.771 | 98.207 | 93.084 | 99.734 |
| CapyCTL r3 (FP8 dense/attention/head) | 97.177 | 28 | 96.122 | 97.742 | 99.180 | 96.052 | 99.934 |
| CapyCTL r3e (FP8 dense/attention; BF16 head) | 97.191 | 30 | 96.104 | 97.819 | 99.160 | 96.045 | 99.934 |
| NVIDIA | 97.132 | 30 | 96.075 | 97.701 | 99.201 | 95.952 | 100.000 |
| RedHat | 94.476 | 19 | 92.297 | 95.489 | 98.299 | 92.598 | 99.801 |
| Unsloth (FP8 attention) | 96.018 | 22.5 | 94.384 | 96.900 | 98.903 | 94.451 | 99.867 |

Among the initial six builds, NVIDIA was closest to BF16; the stored mixed-FP8 builds now reach **97.177% (r3)** and **97.191% (r3e)**. CapyCTL's searched scales improve overall agreement by **0.290 percentage points** from r1 to r2, although math and tool-use agreement are slightly lower. Unsloth's mixed FP8-attention/NVFP4 build falls between NVIDIA and the all-NVFP4 builds. These observations do not isolate quantization as the sole difference between publishers' checkpoints.

The loader reconstructed 11,725 NVFP4 matrices each for r1, r2 and RedHat; 11,520 for NVIDIA; and 11,610 NVFP4 plus **115 FP8 channel-scaled attention matrices** for Unsloth. BF16 loaded all 1,013 source state tensors.

| Build | Load s | Evaluation s | Wall s | Peak CUDA allocated GiB | Peak CUDA reserved GiB | Peak RSS GiB | Minimum host available GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (by construction) | 72.57 | 3186.19 | 3258.75 | 50.068 | 50.295 | 5.687 | 26.761 |
| CapyCTL r1 | 23.23 | 64.83 | 88.06 | 49.987 | 50.438 | 3.951 | 29.403 |
| CapyCTL r2 | 47.12 | 66.10 | 113.23 | 49.987 | 50.438 | 3.902 | 29.599 |
| CapyCTL r3 (FP8 dense/attention/head) | 22.45 | 65.55 | 88.00 | 53.579 | 54.590 | 4.915 | 24.187 |
| CapyCTL r3e (FP8 dense/attention; BF16 head) | 22.59 | 64.75 | 87.34 | 49.964 | 50.312 | 3.917 | 29.556 |
| NVIDIA | 41.57 | 65.05 | 106.62 | 50.068 | 50.160 | 3.932 | 29.634 |
| RedHat | 45.49 | 64.85 | 110.34 | 49.987 | 50.438 | 3.916 | 29.756 |
| Unsloth (FP8 attention) | 48.87 | 65.08 | 113.95 | 50.000 | 50.328 | 3.946 | 29.642 |

These Python wall times include loading and evaluation, but not interpreter import/shutdown. Whole-command `/usr/bin/time` measurements were BF16 **54:22.99**, r1 **1:33.57**, r2 **1:57.10**, NVIDIA **1:51.85**, RedHat **1:55.10**, Unsloth **1:59.21**. BF16 performs autoregressive generation; the other five perform parallel teacher forcing, so these times must not be compared as build speed.

All six logs report **exit status 0** and all six guards report no memory stop. `results/exit-audit.json` confirms zero agreement processes after completion, host `MemAvailable` of **89,865,986,048 bytes**, and only the original **170-MiB and 33,190-MiB** allocations belonging to another workload's engine and its parent. No environment was modified, no inference engine was started, and no resident workload was signalled. Unsloth was already present at the requested revision; its existing download metadata, like the three pinned references, matched the requested commit. See `results/checkpoint-provenance.json`, the six `score-*.json` files, `bf16-answers.json`, logs, and generated `summary.md`/`summary.json`.

## Family ablation

`ablation/ablate.py` uses the same shared streaming loader, score alignment and summary functions as `next_token_agree.py`. It reuses the exact BF16 reference above: no regeneration, no new conversion and no checkpoint weight files are written. All new variants reconstruct **r2 experts** and mix in either BF16 source weights, reconstructed r2 projections, or FP8 weights simulated directly from the original BF16 weights. All unchanged state, including the router, is BF16 source state. All forward execution remains BF16 and uses the same SDPA/grouped-MM settings.

| Variant | Routed experts | Dense MLP | Attention |
|---|---|---|---|
| A | r2 NVFP4 | BF16 | BF16 |
| B | r2 NVFP4 | r2 NVFP4 | BF16 |
| C | r2 NVFP4 | BF16 | r2 NVFP4 |
| D | r2 NVFP4 | r2 NVFP4 | Simulated per-tensor FP8 |
| D-channel | r2 NVFP4 | r2 NVFP4 | Simulated per-channel FP8 |
| E | r2 NVFP4 | Simulated per-tensor FP8 | Simulated per-tensor FP8 |
| F | NVIDIA NVFP4 | BF16 | BF16 |

F reuses NVIDIA's original measured score; full-r2 results are also reused as a comparison row. A has NVIDIA's family layout but r2's searched-scale experts; it is not a claim that their quantized expert tensors are otherwise identical.

The per-tensor simulation uses `scale = amax(abs(BF16 weight))/448` in FP32, divides by the scale, clamps to `[-448,448]`, rounds to `torch.float8_e4m3fn`, then reconstructs and casts to BF16. Unsloth instead stores **per-output-channel BF16 scales** with shape `[out_features,1]`; D-channel simulates that format, computing row-wise `amax/448`, rounding the scale to BF16 before FP8 rounding and reconstruction. It is a source-weight scaling ablation, not a reuse of Unsloth's published attention tensors. Zero-amax tensors/channels use scale 1 and remain zero. Every simulated module's scale shape, dtype, extrema and payload-byte counts are recorded in the result JSON.

On host A, from this directory:

```bash
bash ablation/run-ablation.sh A
bash ablation/run-ablation.sh B
bash ablation/run-ablation.sh C
bash ablation/run-ablation.sh D
bash ablation/run-ablation.sh D-channel
bash ablation/run-ablation.sh E
```

With no arguments, the runner executes A, B, C, D, D-channel, E, G, H and I in that order. Every GPU job retains the 8-GiB memory guard; each writes its full position-level JSON, resource metrics and `/usr/bin/time` log under `ablation/results/`. The CPU-only summarizer reuses the F/full-r2 scores and writes the combined tables.

**[INFERENCE] Byte arithmetic, not measured bandwidth or speed:** for each decode token, count one read of every attention, dense-MLP and router tensor, plus eight experts per layer. BF16 takes two bytes per weight; NVFP4 counts stored packed weights plus FP8 block scales and FP32 global weight scales; simulated FP8 takes one byte per weight plus its FP32 per-tensor or BF16 per-channel scales. Attention Q/K norms and all router state are included. The expert term is the complete bank's stored weight/weight-scale payload multiplied by `8/128`, since every expert has the same projection shapes. Activation scales are excluded: this is weight payload arithmetic, not complete engine traffic.

The requested family sum excludes the common BF16 output head/tied embedding, other layer norms/state, activations, KV-cache reads/writes, temporary buffers and batching/cache reuse. Those omissions matter when estimating speed; in particular, the output head must also be read for a complete decode step. The offline ablations run entirely reconstructed BF16 weights, so their actual torch weight traffic is not the packed-format arithmetic.

Family selection and byte sums are restricted to `model.language_model.layers.*`. Vision/audio weights also contain attention/MLP module names but remain unchanged BF16 and do not enter this text-only per-token estimate.

### Family-ablation results

The six new variants completed on 2026-11-16; F and full r2 below reuse their original runs. All rows use the same 62,658 reference positions, with per-prompt IDs/lengths checked by the summarizer.

| Variant | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: r2 experts; BF16 dense/attention | 97.397 | 31.5 | 96.347 | 98.009 | 99.252 | 96.319 | 100.000 |
| B: r2 experts+dense; BF16 attention | 96.455 | 27 | 94.940 | 97.275 | 98.924 | 95.152 | 99.867 |
| C: r2 experts+attention; BF16 dense | 95.512 | 22 | 93.793 | 96.259 | 98.401 | 94.158 | 99.934 |
| D: r2 experts+dense; FP8 tensor attention | 96.344 | 24 | 94.727 | 97.203 | 98.801 | 95.098 | 99.867 |
| D-channel: r2 experts+dense; FP8 channel attention | 96.393 | 24.5 | 94.804 | 97.311 | 98.852 | 95.045 | 99.867 |
| E: r2 experts; FP8 tensor dense/attention | 97.151 | 29 | 96.051 | 97.834 | 99.170 | 95.912 | 99.934 |
| F: NVIDIA experts; BF16 rest (reused) | 97.132 | 30 | 96.075 | 97.701 | 99.201 | 95.952 | 100.000 |
| Full r2 NVFP4 baseline (reused) | 94.805 | 20.5 | 92.817 | 95.771 | 98.207 | 93.084 | 99.734 |

Controlled changes within the r2-expert ablations: quantizing only the dense MLP from A to B costs **0.942 percentage points**; quantizing only attention from A to C costs **1.885 points**. With r2 dense MLP held fixed, FP8 attention in D is **1.539 points above** full-r2 NVFP4 attention and **0.111 points below** BF16 attention in B. Channel-scaled D is **0.049 points above** per-tensor D. E, with per-tensor FP8 dense MLP and attention, reaches **97.151%**, versus **97.397%** for BF16 dense MLP and attention in A.

**[INFERENCE] Arithmetic bytes per decode token for the requested text-language families:**

| Variant | Attention B/token | Dense MLP B/token | Eight experts B/token | Router B/token | Total B/token | Total GiB/token |
|---|---:|---:|---:|---:|---:|---:|
| A | 2,220,395,520 | 1,070,530,560 | 802,900,800 | 21,803,520 | 4,115,630,400 | 3.832980 |
| B | 2,220,395,520 | 301,087,080 | 802,900,800 | 21,803,520 | 3,346,186,920 | 3.116379 |
| C | 624,512,460 | 1,070,530,560 | 802,900,800 | 21,803,520 | 2,519,747,340 | 2.346698 |
| D | 1,110,216,140 | 301,087,080 | 802,900,800 | 21,803,520 | 2,236,007,540 | 2.082444 |
| D-channel | 1,110,886,400 | 301,087,080 | 802,900,800 | 21,803,520 | 2,236,677,800 | 2.083069 |
| E | 1,110,216,140 | 535,265,640 | 802,900,800 | 21,803,520 | 2,470,186,100 | 2.300540 |
| F | 2,220,395,520 | 1,070,530,560 | 802,900,800 | 21,803,520 | 4,115,630,400 | 3.832980 |
| Full r2 | 624,512,460 | 301,087,080 | 802,900,800 | 21,803,520 | 1,750,303,860 | 1.630097 |

Add the common **1,476,395,008-byte BF16 output-head weight** for a more complete weight-read estimate; this still excludes the other traffic listed above. No tokens-per-second measurement or kernel-speed claim follows from this table.

The final runs loaded 11,520/11,610/11,635 NVFP4 matrices for A/B/C respectively; D and D-channel loaded 11,610 NVFP4 plus exactly **115 simulated FP8 attention matrices**; E loaded 11,520 NVFP4 plus exactly **205 simulated FP8 attention/dense matrices**. The byte-count metadata contains 175 attention state tensors (115 projections plus 60 Q/K norms), 90 dense projections and 90 router tensors.

All six final jobs exited **0**, all guards remained clear, and minimum host available memory was **29.356 GiB (rounded)**. Python wall times were A **81.90 s**, B **82.12 s**, C **82.71 s**, D **82.90 s**, D-channel **83.12 s**, E **83.25 s**; complete load/evaluation and peak-memory tables are in `ablation/results/summary.md`. The final `ablation/results/exit-audit.json` confirms **zero agreement processes**, only the original **170/33,190-MiB** resident allocations and **89,152,774,144 bytes** host `MemAvailable`. The resident workload was never signalled; no environment or checkpoint weight file was modified.

## Untied output-head ablation

G and H extend E; I extends A. They leave the input embedding byte-for-byte BF16 and load a **separate** reconstructed output-head tensor with `tie_word_embeddings=false` on both the outer and text configs. The loader asserts separate storage, BF16 input embedding and complete parameter population. Transformers' configured final logit softcap remains **30**. The same original BF16 reference and scoring settings are reused.

| Variant | Body | Separate output head |
|---|---|---|
| G | E: r2 experts, per-tensor FP8 dense/attention | Per-output-channel FP8 E4M3 |
| H | E: r2 experts, per-tensor FP8 dense/attention | NVFP4 group 16, searched block scales |
| I | A: r2 experts, BF16 dense/attention | Per-output-channel FP8 E4M3 |

G/I use the same row-wise FP8 simulation as D-channel: FP32 row `amax/448` is rounded to BF16 for the scale, weights are rounded to E4M3, then reconstructed into BF16. H uses the existing converter's **exact `requant_scales.requant`** function, in 1,024-row chunks, with a single FP32 head global scale `amax/(6*448)`, a max-derived FP8 block-scale baseline, and the r2 search's 21 FP8-rounded multipliers from 0.60 to 1.10. It records the scale range, payload sizes and actual reconstructed relative weight error. No checkpoint weight files or conversion artifacts are written.

```bash
bash ablation/run-ablation.sh G
bash ablation/run-ablation.sh H
bash ablation/run-ablation.sh I
```

The combined `ablation/results/summary.md` now includes the output head in the **[INFERENCE]** weight-read arithmetic for every variant, retaining the separate family subtotal. BF16 head: **1,476,395,008 bytes**; per-channel FP8 head: **738,721,792 bytes** (738,197,504 one-byte weights plus 524,288 BF16 scale bytes); searched NVFP4 head: **415,236,100 bytes** (369,098,752 packed bytes, 46,137,344 block-scale bytes and one four-byte global scale). The input embedding lookup row, other layer state, activations and KV traffic remain excluded. These are payload arithmetic, not observed engine memory traffic or speed.

### Installed engine source: untied quantized heads

This is **read-only static source support, not runtime qualification**. No engines were started. Citation roots below are **SG** = `~/sglang-0.5.22-venv/lib/python3.12/site-packages/sglang/` and **VL** = `~/vllm-0.32.0-venv/lib/python3.12/site-packages/vllm/`. Eight key SGLang model/loader/logits files were also SHA-256 compared against `~/sglang-0.5.22-lmcache-venv/`; all were byte-identical. Complete format contracts and citations are saved in `ablation/results/engine-head-support.json`.

| Installed path | Separate NVFP4 head, BF16 embedding | Separate per-channel FP8 head, BF16 embedding | Final softcap |
|---|---|---|---|
| SGLang ModelOpt / ModelOptMixed | Conditional yes, including E's mixed body | **No ModelOpt dispatch** for the per-channel algorithm; compressed-tensors is a separate supported schema | Applied after quantized head logits |
| vLLM ModelOptMixed | Conditional yes, including E's mixed body | Conditional yes with `FP8_PER_CHANNEL_PER_TOKEN`, correct `[vocab]` scales and mixed per-layer recipe | Applied after quantized head logits |
| Either engine's compressed-tensors path | Requires matching compressed-tensors scheme/body serialization | Conditional yes with explicitly targeted head and `[vocab,1]` channel scales | Applied after quantized head logits |

**Actual Gemma4 entrypoints and tying:** SGLang registers `Gemma4ForConditionalGeneration` in `SG/srt/models/gemma4_mm.py:138,1172`; its `config.text_config.tie_word_embeddings` branch creates a separate quantization-aware `ParallelLMHead` at `:254-268`. Its embedding remains an ordinary scaled embedding (`SG/srt/models/gemma4_causal.py:885-891`, `gemma3_causal.py:532-548`). Head quantization uses prefix `model.lm_head`, while registered parameter names are `lm_head.*`; the loader strips a leading `model.` and loads the head parameters (`gemma4_mm.py:197,263-267,924-932,1061-1077`). vLLM's registry maps the same architecture to `gemma4_mm` (`VL/model_executor/models/registry.py:413-417`), whose wrapper passes the nested text config to the causal model (`gemma4_mm.py:1134-1141`). That model creates a quantized `ParallelLMHead` and ties it only when the text flag is true (`gemma4.py:1447-1458`); the wrapper maps `lm_head.*` to `language_model.lm_head.*` (`gemma4_mm.py:1053-1054`). vLLM's weight loader skips actual aliases, not every head name (`VL/model_executor/models/utils.py:202-213,255-260,291-292,480-489`).

**NVFP4:** both ModelOpt dispatchers explicitly include `ParallelLMHead` (SG `srt/layers/quantization/modelopt_quant.py:338-345`; VL `model_executor/layers/quantization/modelopt.py:224-260,1847-1856`). Packed weights, E4M3 block scales and global scales are allocated for the head (SG `modelopt_quant.py:1798-1860`; VL `modelopt.py:1992-2034`). W4A4 additionally requires the appropriate valid activation scale (VL `modelopt.py:2067-2083`); the offline H experiment does not calibrate or model that activation quantizer. Remove every head-matching `ignore`/`exclude_modules` pattern, and for mixed precision add an explicit `lm_head` recipe. Merely untying a checkpoint or adding a `quantize_lm_head` boolean is not sufficient. Hardware/backend compatibility remains untested.

**FP8 and the important engine difference:** plain ModelOpt `"FP8"` is per-tensor, not per-channel (SG `srt/layers/quantization/modelopt_quant.py:599-608`; VL `model_executor/layers/quantization/modelopt.py:2677-2684`). SGLang's mixed dispatcher only selects `"FP8"`, block-FP8, `"MXFP8"`, `"NVFP4"` and `"W4A16_NVFP4"`; an unrecognized `FP8_PER_CHANNEL_PER_TOKEN` recipe falls back to unquantized (`SG/srt/layers/quantization/modelopt_quant.py:1047-1062`). Its demonstrated channel-scaled head path is compressed-tensors with named/regex head targeting (`SG/srt/layers/quantization/compressed_tensors/compressed_tensors.py:199-207,1081-1129`; `SG/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:91-110`). That requires a compatible body/schema; no automatic composition of a ModelOpt body with a compressed-tensors head was established.

vLLM **does** include `FP8_PER_CHANNEL_PER_TOKEN` in mixed ModelOpt `LINEAR_ALGOS` alongside FP8 and NVFP4 (`VL/model_executor/layers/quantization/modelopt.py:139-149,1847-1856,2685-2691`). Its head uses E4M3 `[vocab,hidden]` weights and **FP32 `[vocab]` scales**, with dynamic per-token FP8 activations (`:2183-2201`). G/I's BF16-rounded `[vocab,1]` scale values would need to be squeezed and cast to FP32 for that serialization; the vocabulary loader does not reshape them (`VL/model_executor/layers/vocab_parallel_embedding.py:515-520`). Compressed-tensors instead allocates FP32 `[vocab,1]` scales (`VL/model_executor/layers/quantization/utils/fp8_utils.py:1384-1411`) and supports explicitly targeted heads (`compressed_tensors/compressed_tensors.py:186-211`). A broad `"Linear"` target is insufficient for `ParallelLMHead` in either engine. Generic `quant_method="fp8"` also does not dispatch a quantized head in these installations (SG `srt/layers/quantization/fp8.py:390-401`; VL `model_executor/layers/quantization/fp8.py:177-208`).

**[INFERENCE] Scale-promotion adjustment:** the G/I payload arithmetic uses the simulated BF16 channel-scale representation. The inspected compatible engines allocate those scales in FP32, so their loaded head weight/scale estimate is **739,246,080 bytes**, **524,288 bytes** above that raw payload. Accordingly, the loaded-FP32-scale totals are G **3,209,432,180 bytes/token** and I **4,854,876,480 bytes/token**. H's estimate is unchanged. These are still arithmetic assumptions, not measured kernel traffic.

**Softcapping remains:** SGLang passes the text config to its logits processor (`SG/srt/models/gemma4_mm.py:272-273`), applies the head quantizer (`SG/srt/layers/logits_processor.py:976-981,1412-1476`), then applies configured softcapping (`:948-966`). vLLM similarly passes `final_logit_softcapping` (`VL/model_executor/models/gemma4.py:1447-1458`), calls the head quantizer (`VL/model_executor/layers/logits_processor.py:143-146`), then computes `cap*tanh(logits/cap)` (`:112-116`). Untying or quantizing the head does not bypass the source model's cap of 30.

These findings do **not** make G/H/I deployable checkpoints: none was exported or engine-tested. Native head activation quantization and kernel numerics remain outside the weights-only agreement measurement.

### Output-head results

All three new runs completed on the same **62,658 positions / 200 prompts**:

| Variant | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| G: E + per-channel FP8 head | 97.134 | 29 | 96.010 | 97.839 | 99.108 | 95.912 | 100.000 |
| H: E + searched NVFP4 head | 96.861 | 28 | 95.673 | 97.619 | 99.139 | 95.425 | 99.934 |
| I: A + per-channel FP8 head | 97.355 | 31 | 96.323 | 97.998 | 99.201 | 96.219 | 100.000 |

Relative to E's 97.151%, the FP8 head in G costs **0.017 percentage points**, and the searched NVFP4 head in H costs **0.290 points**. I costs **0.042 points** versus A's 97.397%. G reaches essentially NVIDIA/F's measured agreement (97.132%) with a much smaller arithmetic weight payload; this is not an engine-speed or task-accuracy claim. H's reconstructed head relative weight error is **8.59426%**.

**[INFERENCE] Including the output head:** the family term remains attention + dense MLP + eight experts per layer + router. The final column accounts for FP32 channel-scale storage in the compatible engine loaders; raw G/I payloads use the simulation's BF16 scale representation.

| Variant | Family subtotal B/token | Head raw payload B/token | Total raw payload B/token | Total with FP32 head-channel scales B/token |
|---|---:|---:|---:|---:|
| E | 2,470,186,100 | 1,476,395,008 | 3,946,581,108 | 3,946,581,108 |
| G | 2,470,186,100 | 738,721,792 | 3,208,907,892 | 3,209,432,180 |
| H | 2,470,186,100 | 415,236,100 | 2,885,422,200 | 2,885,422,200 |
| A / F | 4,115,630,400 | 1,476,395,008 | 5,592,025,408 | 5,592,025,408 |
| I | 4,115,630,400 | 738,721,792 | 4,854,352,192 | 4,854,876,480 |
| Full r2 | 1,750,303,860 | 1,476,395,008 | 3,226,698,868 | 3,226,698,868 |

| Variant | Load s | Evaluation s | Wall s | Peak CUDA allocated GiB | Peak CUDA reserved GiB | Peak RSS GiB | Minimum host available GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| G | 19.93 | 64.07 | 84.00 | 57.056 | 58.160 | 4.006 | 22.006 |
| H | 43.38 | 65.23 | 108.61 | 52.929 | 53.420 | 4.003 | 26.617 |
| I | 25.13 | 64.42 | 89.55 | 57.123 | 57.879 | 3.996 | 21.755 |

Each head was actually untied, the input embedding remained `torch.bfloat16`, final logit softcapping remained **30**, and every GPU command exited **0** without triggering its 8-GiB guard. `ablation/results/head-exit-audit.json` records **zero comparison processes**, unchanged original **170/33,190-MiB** resident allocations and **89,174,876,160 bytes** host `MemAvailable` after completion. No engine was started, resident process signalled, checkpoint exported, or environment modified.

## Stored FP8 scale compatibility

The shared agreement loader accepts ModelOpt per-tensor FP8 scales (scalar/singleton FP32) as well as compressed-tensors per-output-channel scales `[out_features,1]`. Both paths now call the converter's `Reader.dequantize` rather than maintaining a second decoder. Coverage reports distinguish `fp8_tensor_matrices` from `fp8_channel_matrices`.

CPU-only characterization passed for scalar FP32, singleton FP32 and channel-BF16 safetensors fixtures. Three actual Unsloth attention matrices (layer 0 Q/O and layer 5 K) reconstructed bit-exactly against the previous explicit channel decoder after casting to BF16. CUDA remained uninitialized; no GPU or model-forward work was performed. Evidence: `results/fp8-decoder-cpu-check.json`. This establishes decoder compatibility, not a new agreement measurement; the original full Unsloth result remains 96.018%.

## Stored mixed-FP8 revision 3

`bash run-tensors.sh r3` scores the actual exported `~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r3` checkpoint. It restores searched-NVFP4 experts, per-tensor FP8 dense MLP/attention and a **per-tensor FP8 untied output head**. Unlike ablation G's per-channel head simulation, r3 loads the head and scalar FP32 weight scale actually stored by ModelOpt. Static input activation scales remain unused in this weights-only comparison.

The agreement command reads `text_config.tie_word_embeddings` from each checkpoint to select the shared loader's `untie_output_head` setting. Thus r3's head is loaded separately rather than skipped or tied back to the BF16 embedding; its report records the tie setting, embedding dtype and final logit softcap.

r3 completed successfully on all **62,658 positions / 200 prompts**:

| Build | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actual stored r3 | 97.177 | 28 | 96.122 | 97.742 | 99.180 | 96.052 | 99.934 |
| Ablation G simulation | 97.134 | 29 | 96.010 | 97.839 | 99.108 | 95.912 | 100.000 |

r3 agrees at **60,889** positions versus G's **60,862**: **27 more positions**, or **+0.043091 percentage points** overall. The output-head formats differ (stored per-tensor FP8 versus simulated per-channel FP8), so this is a comparison with G, not a controlled claim that head scaling alone caused the difference.

The loader reconstructed **11,520 NVFP4 expert matrices and 206 per-tensor FP8 matrices**, including the separate head, while preserving 748 unquantized state tensors. The report confirms an untied head, BF16 embedding and softcap **30**. Load/evaluation/Python wall times were **22.45 / 65.55 / 88.00 seconds**, with peak CUDA allocation/reservation **53.579 / 54.590 GiB**, peak RSS **4.915 GiB**, and minimum host available memory **24.187 GiB**. The guarded process exited **0**; no guard fired, and `results/r3-vs-G.json` records zero active comparison processes. GPU ownership was released immediately after exit for the next conversion. No further GPU job was started.

Full evidence is in `results/score-r3.json`, `results/r3.log`, `results/r3-vs-G.json` and the updated `results/summary.md`/`summary.json`.

## Stored mixed-FP8 revision 3e

`bash run-tensors.sh r3e` scores the actual exported `~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r3e` checkpoint: searched-NVFP4 experts and scalar-scale per-tensor FP8 dense MLP/attention, with the **original tied BF16 embedding/output head**. It uses the checkpoint's true text-config tie flag, so `untie_output_head=False` and no separate quantized head is loaded. This is the stored-checkpoint counterpart of ablation E, rather than r3/G's untied-head layout. Activations remain BF16 and stored activation scales are not modelled.

r3e completed on all **62,658 positions / 200 prompts**:

| Build | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actual stored r3e | 97.191 | 30 | 96.104 | 97.819 | 99.160 | 96.045 | 99.934 |
| Ablation E simulation | 97.151 | 29 | 96.051 | 97.834 | 99.170 | 95.912 | 99.934 |

r3e agrees at **60,898** positions versus E's **60,873**: **25 more positions**, or **+0.039899 percentage points** overall. The stored exporter result is therefore close to, but not numerically identical with, E's direct FP8 simulation. r3e also agrees at nine more positions than stored r3 (97.177%).

The loader reconstructed **11,520 NVFP4 expert matrices and 205 per-tensor FP8 body matrices**, preserving 748 unquantized state tensors. Its head remained tied, the embedding remained BF16, and final softcapping remained **30**. Load/evaluation/Python wall times were **22.59 / 64.75 / 87.34 seconds**; peak CUDA allocation/reservation were **49.964 / 50.312 GiB**, peak RSS **3.917 GiB**, and minimum host available memory **29.556 GiB**. The guarded command exited **0** without a memory stop.

Evidence: `results/score-r3e.json`, `results/r3e.log`, `results/r3e-vs-E.json`, and the updated combined `results/summary.md`/`summary.json`.

## Full Unsloth FP8-loader regression

After generalizing scalar/channel FP8 loading to `Reader.dequantize`, Unsloth was rerun fully on the same **200 prompts / 62,658 positions**, with the original result preserved. It again agrees at **60,163 positions: 96.018066%**, with median first disagreement **22.5**. **Every picked-token ID and the complete summary, including all per-category figures, exactly match the original run.** This closes the decoder regression check, beyond the earlier CPU matrix characterization.

The independent rerun is saved under `results/unsloth-regression/` (`score-unsloth.json`, `unsloth.log`, `comparison.json`). Load/evaluation/Python wall times were **44.28 / 64.88 / 109.16 seconds**. Its peak CUDA allocation was **53,686,553,088 bytes** and minimum host available memory **31,815,553,024 bytes**; no guard fired.

Both r3e and the full Unsloth regression exited **0**. The final `comparison.json` records **zero comparison processes**, only the unchanged original **170/33,190-MiB** resident allocations, and **90,357,157,888 bytes** host `MemAvailable`. GPU release and both results were reported immediately after collecting the regression's exit evidence. No further GPU work was started.
