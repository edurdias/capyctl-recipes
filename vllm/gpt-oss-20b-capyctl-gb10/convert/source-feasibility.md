# GPT-OSS-20B checkpoint feasibility on one GB10

Installed-source feasibility with completed CPU conversion/audit, GPU agreement and native vLLM qualification. The Quark checkpoint has **13,125,022,848 tensor bytes**; `checkpoint-audit.json` records complete byte/hash coverage. **2026-11-18–19 history:** the initial offline 99.5% gate failed at **98.14843%** agreement (OpenAI's own generated-ID self-consistency is **98.39201%**, a different comparison target). That gate was judged too strict. Subsequent native vLLM 0.32.0 qualification passed: **237/250 GSM8K, 30/30 tools, 48/48 retrieval observations and 26.1% faster one-stream decode** than OpenAI on the same engine (57.77 versus 45.81 tokens/s). The checkpoint was then selected for publication. See the [recipe](../README.md) and [final comparison](../../../comparisons/gpt-oss-20b-builds-gb10/README.md); the offline shortfall remains a limitation.

Reference: OpenAI's checkpoint on SGLang 0.5.22 with a 131k-context production configuration. Source checkpoint revision: `6cee5e81ee83917806bbde320786a8fb61efebee`.

## Offline result and subsequent qualification

The vLLM Quark layout preserves OpenAI's MXFP4 expert packed bits, E8M0 scales and biases exactly, splitting them into per-expert tensors. Only the 96 Q/K/V/O attention matrices change: E4M3 FP8 weights with FP32 per-output-channel multiplicative scales, dynamic FP8 inputs declared for native Quark execution. Head, embeddings, routers, norms, attention sinks and other biases remain unchanged. The complete CPU audit covers 459 source and 5,019 candidate tensors, including all 144 source expert tensors.

[INFERENCE: weight-traffic arithmetic, not measured bandwidth] The layout reduces batch-one weight traffic from 3,708,089,088 to 3,071,847,168 bytes/token: **636,241,920 bytes/token saved (about 636 MB, 17.16%, rounded 17.2%)**. That saving does not establish a performance win.

After explicit exclusive-GPU clearance, the adapted Gemma corpus's 200 records were rendered with GPT-OSS Harmony at effort low. Installed Transformers 5.14.1 does not support GPT-OSS SDPA; both sides therefore used the same supported eager-attention BF16 graph. The teacher reconstructs OpenAI's released MXFP4 weights; the candidate changes only attention weights to its actual FP8 bytes times scales, cast once to BF16. This is not a test of native FP8 activation quantization or Quark kernels.

| Comparison/category | Agree / positions | Agreement |
|---|---:|---:|
| Candidate vs teacher-forced teacher, overall | 51,577 / 52,550 | **98.14843%** |
| Chat | 14,833 / 15,190 | 97.64977% |
| Coding | 17,866 / 18,186 | 98.24040% |
| Math | 6,241 / 6,306 | 98.96924% |
| Multilingual | 10,664 / 10,887 | 97.95169% |
| Tool use | 1,973 / 1,981 | 99.59616% |
| Teacher-forced teacher vs generated teacher IDs | 51,705 / 52,550 | **98.39201%** |
| Candidate vs the same generated teacher IDs | 51,381 / 52,550 | **97.77545%** |

All reference answer positions were scored, including those in the **64 capped records**; 136 records completed. Cap hits are separately identified, never counted as completed answers. Cached batched generation and full-sequence teacher forcing are not numerically identical: the teacher's generated-ID self-consistency is 98.39%, not an assumed 100%. On the common generated-ID basis, candidate agreement drops by about 0.62 percentage points. Against the actual teacher-forced teacher, per-channel FP8 attention costs GPT-OSS **1.85 points (about 1.9)** of next-token agreement, versus about **0.25 points for Gemma 4** in the comparison precedent. The initial 99.5% offline gate failed; it was later judged too strict and superseded for selection by separate native qualification, without changing this measured result.

The GPU run completed successfully in 5,708.41 seconds, with all 411 reconstructed parameter tensors BF16, expert alpha 1.702/clamp 7 intact, no memory-guard stop and no resident GPU processes before or after. Full evidence is under `comparisons/gpt-oss-20b-builds-gb10/tensors/`: `bf16-answers.json`, `score-bf16.json`, `score-fp8attn.json`; the exact command is `run-agreement.sh`. Native launch evidence later selected `CutlassFP8ScaledMMLinearKernel` for Quark attention and Marlin MXFP4 experts, with FP8 KV and LMCache off. Native tasks/output review and the exact-file acceptance sweep passed; no drafter met both task and long-context requirements.

The CPU NVFP4 cast diagnostic also completed: **100% of all 597,196,800 blocks** fall within ModelOpt's sufficient lossless scale window (all 48 scale tensors). No NVFP4 checkpoint was built. Even this lossless representability does not make NVFP4 useful here: expert weight/local-scale bytes increase **5.88%**, and neither inspected SGLang 0.5.22 nor vLLM 0.32 supplies a bias-carrying GPT-OSS NVFP4 MoE path. See the installed-source citations below and `cast-share.json`.


## Verdicts

| Engine | Unchanged MXFP4 experts + FP8 attention | FP8 output head | NVFP4 experts, correct GPT-OSS semantics |
|---|---|---|---|
| Installed SGLang 0.5.22 | **Not feasible without an engine change.** The decoder does not forward quant_config to attention. | **Not feasible without an engine change.** | **Not feasible through the inspected ModelOpt route:** expert biases are not allocated or passed. |
| Installed vLLM 0.32.0 | **Feasible with conditions:** Quark config, reversible per-expert repack, FP8 E4M3 attention weights and dynamic FP8 inputs; qualify actual backend and agreement before publication. | **Not feasible without an engine change.** | **Not feasible through inspected ModelOpt, Quark or compressed-tensors NVFP4 routes:** expert bias state/propagation is missing. |
| Installed vLLM 0.33.0 | **Feasible with the same conditions and Quark layout.** | **Not feasible without an engine change.** | **Not feasible through the inspected NVFP4 paths:** same bias obstacle. |
| llama.cpp b12242 | **Feasible with conditions:** keep native MXFP4 experts and use Q8_0/Q5_0/Q4_0 for attention, head and embeddings. These are integer block formats, not FP8 GEMMs. | Quantizable in GGUF. | Not a recommended checkpoint candidate: no demonstrated speed advantage; b12242 can already use native W4A4 for MXFP4. |

**Source alone cannot establish that any candidate is faster.** It establishes reachable execution paths and potential bandwidth savings. Neither a byte reduction nor a native W4A4 kernel is a measured speedup.

### Installed source citation roots

All engine citations below are installed files read on the measured GB10, not assumptions from upstream version tags. Let:

- `S/` = `/home/me/sglang-0.5.22-venv/lib/python3.12/site-packages/sglang/srt/`
- `V32/` = `/home/me/vllm-0.32.0-venv/lib/python3.12/site-packages/vllm/`
- `V33/` = `/home/me/vllm-0.33.0-venv/lib/python3.12/site-packages/vllm/`
- `L/` = `/home/me/llamacpp-b12242/`
- `M/` = `/home/me/modelopt-0.47.0-venv/lib/python3.12/site-packages/modelopt/`

## 1. Preserve MXFP4; reduce attention traffic

### SGLang: the decisive missing call argument

`S/models/gpt_oss.py:544-559` constructs `GptOssAttention` without `quant_config=quant_config`. Its constructor defaults quant_config to None at `:380`; the inner QKV and O linears accept it at `:419` and `:436`, but never receive the model's config. Loading FP8 bytes into those unquantized BF16 parameters would not activate FP8 GEMMs and would lose their multiplicative scale semantics.

This blocks **every checkpoint-only config shape**, even though the generic quantizers look promising in isolation:

- `S/layers/quantization/compressed_tensors/compressed_tensors.py:228-241` really does route MXFP4 MoE to `Mxfp4MoEMethod`.
- `:675-694` recognizes float4/group32 experts, and `:372-399` handles per-group mixed formats.
- `schemes/compressed_tensors_w8a8_fp8.py:91-110,228-250` allocates E4M3 weights and calls FP8 execution.
- `S/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:992-1013` requires AITER; `quark_w4a8_mxfp4_moe.py:398-423` also requires AITER. Those Quark expert routes are not GB10 CUDA routes.
- `S/layers/quantization/modelopt_quant.py:1078-1091` has no MXFP4 expert branch in ModelOpt MIXED_PRECISION.

**No SGLang mixed-attention conversion script is supplied**, because a checkpoint emitted for that proposed route would not execute the requested behavior. Fixing the missing attention call argument and qualifying the fused expert loader would be engine work, outside this checkpoint-only assignment.

The original dedicated MXFP4 route does preserve biases and OpenAI SwiGLU: `S/models/gpt_oss.py:240-252` requests bias/alpha/clamp, and `S/layers/quantization/mxfp4.py:1531-1549` sends both biases plus alpha/beta/limit to the SM12x CUTLASS payload. Attention sinks are separate parameters and passed to attention at `S/models/gpt_oss.py:424-429,498`.

### vLLM: Quark is a reachable layout in both versions

Use `mixed_attention.py` with default per-output-channel scaling; `--scale tensor` is also supported. It changes **only the 96 attention matrices**, adds FP32 multiplicative scales, and keeps output head, embeddings, router, norms, sinks and all biases unchanged. It never dequantizes or requantizes experts. Native expert tensors are split into 32 per-expert tensors per projection, with interleaved gate/up rows and packed nibbles unchanged.

The config shape is:

- `quant_method: quark`
- global expert weight: `dtype=fp4`, `qscheme=per_group`, `group_size=32`, `scale_format=e8m0`, static, symmetric; `input_tensors: null` selects W4A16 expert dispatch.
- attention `layer_quant_config["*.self_attn.{q,k,v,o}_proj"]`: `weight.dtype=fp8_e4m3`, static per_channel or per_tensor; dynamic FP8 per_channel input scales mean per-token activation quantization.
- `export: {pack_method: reorder, kv_cache_group: []}`; head/embedding/router excluded. KV precision is independently selected by launch flags, not altered by this checkpoint.

Evidence:

- Attention receives quant_config: `V32/model_executor/models/gpt_oss.py:138-153`; `V33/.../gpt_oss.py:137-153`.
- Quark supports both tensor/channel FP8 weights plus dynamic activations: `V32/model_executor/layers/quantization/quark/quark.py:430-478`; the FP8 parameter and execution code is `quark/schemes/quark_w8a8_fp8.py:150-210`. `V33/.../schemes/quark_w8a8_fp8.py:50-77,195-213` retains the same path.
- Weight-only OCP MX expert config: `V32/.../quark/quark.py:653-678`; same `V33/.../quark/quark.py:653-678`. Native expert backend selection: `V32/.../quark/quark_moe.py:1607-1610`.
- GPT-OSS Quark loader explicitly supports per-expert names, remaps routed-expert state and copies packed MXFP4 weights/scales: `V32/model_executor/models/gpt_oss.py:830-858,871-953`; `V33/.../gpt_oss.py:818-840,860-941`. Per-expert export avoids the combined-tensor branch's `param.data[expert_id]` indexing with expert_id=None.
- Expert biases remain present: `V32/.../quark/quark_moe.py:1761-1780`, `V33/.../quark/quark_moe.py:1784-1803`. The GPT-OSS-specific format conversion carries both biases: `V32/.../quark/quark_moe.py:1802-1842`, `V33/.../quark/quark_moe.py:1826-1865`.
- Custom alpha/beta/clamp are passed: `V32/.../quark/quark_moe.py:1873-1884`, `V33/.../quark/quark_moe.py:1896-1907`; model activation is `swigluoai` with has_bias=True at `V33/model_executor/models/gpt_oss.py:390-403` (V32 `:390-404`).
- Sinks remain separate and attached to attention: `V32/model_executor/models/gpt_oss.py:130-132,187`, loader `:1085-1089`; V33 `:131,186`.

This is real FP8 weight/input GEMM dispatch, not permanently dequantized BF16 attention: Quark calls `init_fp8_linear_kernel`, stores E4M3 weights, and invokes its `apply_weights`. The CUTLASS FP8 implementation is `V32/model_executor/kernels/linear/scaled_mm/cutlass.py:158-175,245-255`. Actual kernel selection still needs a launch-log gate, including ruling out any unwanted fallback.

Native `quant_method=mxfp4` cannot do this: `V32/model_executor/layers/quantization/mxfp4.py:91-113` explicitly returns unquantized linears. ModelOpt MIXED_PRECISION lacks an MXFP4 expert branch (`V32/.../quantization/modelopt.py:1858-1884`, V33 `:1889-1915`). Quark is the supported candidate, not a claim that every mixed config works.

### Output head: no checkpoint-only route

SGLang comments out quant_config in `ParallelLMHead` construction at `S/models/gpt_oss.py:767-773`. vLLM omits it at `V32/model_executor/models/gpt_oss.py:1459-1463` and `V33/.../gpt_oss.py:1417-1421`. Generic head quantizer support cannot override this model construction. No compressed-tensors, ModelOpt or Quark JSON declaration can activate FP8 GPT-OSS lm_head GEMMs in these installed implementations.

## 2. NVFP4 experts: more bytes, no correct Python-engine cutover yet

[INFERENCE: exact unpadded batch-one weight arithmetic] Four selected experts per layer contain 2,388,787,200 weight values:

| Expert format | Weight + local scale bytes/token | Difference |
|---|---:|---:|
| MXFP4: 16 packed bytes + 1 E8M0 byte per 32 weights | 1,269,043,200 | Reference |
| NVFP4: 8 packed bytes + 1 E4M3 byte per 16 weights | 1,343,692,800 | **+74,649,600 (+5.88%)**, plus small global-scale metadata |

Full checkpoint expert storage rises from 10,152,345,600 to 10,749,542,400 bytes, before global scales. Keeping dense tensors BF16 gives about **3.783 GB/token**, versus **3.708 GB/token** for OpenAI's layout. This is not a bandwidth optimization.

### Closed-form cast and CPU diagnostic

ModelOpt's cast is conditionally lossless, not unconditionally lossless. Installed `M/torch/quantization/utils/numeric_utils.py:61-101` computes `k = E8M0_byte - 127`, maximum k over nonzero scales, and `m=k_max-8`. A block is counted lossless if `k >= k_max-17`, or the scale byte is zero (the implementation treats that as an all-zero block). The per-NVFP4-block scale is the exactly representable power of two `2^(k-m)`; global scale_2 is `2^m`. One MXFP4 block becomes two NVFP4 blocks. Out-of-window blocks use a data-derived fallback, so their correctness cannot be certified by exponent counts alone.

The upstream 0.47.0 cast implementation is [cast_mxfp4_to_nvfp4.py](https://github.com/NVIDIA/Model-Optimizer/blob/c10fb3bf061229798b9e5fc53e5d9dd09c1bc7a5/examples/hf_ptq/cast_mxfp4_to_nvfp4.py), `build_amax_map` and `apply_to_model`; the latter repeats scales for the two halves and reports the lossless share.

`cast_share.py` implements that histogram calculation using only numpy/safetensors on CPU. It reads **597,196,800 scale bytes**, not the 9.56 GB of packed weights, and emits per-tensor exponent histograms and source config/index hashes. This share is a sufficient representability diagnostic, not an actual exported-weight comparison.

**Completed CPU diagnostic:** after source download verification, all **597,196,800 / 597,196,800 blocks (100%)** and all 48 scale tensors satisfy the sufficient lossless scale window. Full exponent histograms and source config/index hashes are in `cast-share.json`. The earlier missing-shard prerequisite failure is resolved. No NVFP4 checkpoint was built; this share does not prove an exported checkpoint or native engine correct.

### Why no correct NVFP4 engine candidate is recommended

- **SGLang:** `S/layers/quantization/modelopt_quant.py:2423-2585` allocates weights and scales, not expert biases, despite GPT-OSS passing with_bias=True. Its NVFP4 CUTLASS payload at `:3251-3269` carries custom SwiGLU values but no biases. The original MXFP4 payload explicitly has biases (`mxfp4.py:1537-1547`). Correct activation parameters alone do not fix missing biases.
- **vLLM 0.32:** `V32/.../quantization/modelopt.py:892-1006` likewise allocates no expert biases. Its NVFP4 config maker at `:1097-1110` passes alpha/beta/clamp but no biases. Quark NVFP4 `quark/quark_moe.py:2008-2111` and compressed-tensors NVFP4 `compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py:73-186` have the same allocation gap.
- **vLLM 0.33:** the ModelOpt allocator is `V33/.../quantization/modelopt.py:912-1026`, and maker `:1128-1142`; Quark NVFP4 allocation is `quark/quark_moe.py:2215-2318`, also without bias buffers.
- **Both vLLMs:** `model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:263-325` initializes both bias arguments to None; the NVFP4 branch never sets them, whereas MXFP4 sets them at `:321-322`. The native b12x MoE also explicitly rejects expert biases (`V32/model_executor/layers/fused_moe/b12x.py:432-441`).

[INFERENCE] Native NVFP4 W4A4 could in principle beat MXFP8×MXFP4 on a compute-limited workload despite 5.88% higher expert traffic. That is a kernel hypothesis, **not a currently correct GPT-OSS checkpoint path in these inspected Python engines**. llama.cpp b12242 already selects native Q4 activations for both MXFP4 and NVFP4 on Blackwell (`L/ggml/src/ggml-cuda/mmq.cu:73-86,125-131`), so NVFP4 is not uniquely able to obtain W4A4 there either. No source-only evidence establishes an NVFP4 speed win.

## 3. GGUF dense quantization, including K-quant limits

`L/conversion/gpt_oss.py:19-22,49-81` bypasses expert dequantization and losslessly repacks OpenAI's MXFP4 blocks/scales; gate/up splitting and biases are handled at `:106-121`. `L/conversion/base.py:1125-1158,1533-1538` keeps routers, norms, biases and sinks F32. The GPT-OSS graph separately loads all expert biases (`L/src/models/openai-moe.cpp:44-54`), passes them to the OpenAI SwiGLU MoE (`:119-130`) and keeps attention sinks (`:99-101`).

The supplied `gguf_dense.py` uses b12242's own converter with an in-process dense-type policy, without modifying that install. Python Q8_0/Q5_0/Q4_0 quantizers exist at `L/gguf-py/gguf/quants.py:222-240,293-313,380-395`; actual conversion invokes them at `L/conversion/base.py:1580-1600`. CUDA matrix/vector and matrix/matrix paths support these types (`L/ggml/src/ggml-cuda/mmvq.cu:43-49`, `mmq.cu:16-29`). This is feasible without new installs, though conversion and quality are untested.

### K-quants cannot cover all GPT-OSS dense rows as requested

Q6_K/Q5_K/Q4_K use 256-value superblocks (`L/ggml/src/ggml-common.h:89,323-368`). Q/K/V projections, output head and embedding rows have contracting width **2880**, not divisible by 256. Only O projection rows contract over **4096**, which is divisible. Without changing model geometry, a literal all-attention-and-head K-quant layout is not available.

`L/src/llama-quant.cpp:450-479` defines the effective fallbacks:

- Q6_K request → Q8_0 for 2880-wide rows; O projection remains Q6_K.
- Q5_K request → Q5_1 for 2880-wide rows; O remains Q5_K.
- Q4_K request → Q5_0 for 2880-wide rows; O remains Q4_K.

Use regex `--tensor-type` overrides, not `--output-tensor-type`/`--token-embedding-type`: those special options return before the generic fallback (`L/src/llama-quant.cpp:790-838`). Experts must explicitly stay MXFP4; same-type tensors are copied without requantization (`:1302-1307`). CLI override syntax is at `L/tools/quantize/quantize.cpp:142-153,431-437`.

**Tool prerequisite:** build/bin in all approved `/home/me/llamacpp-b12242`, `/home/me/llamacpp-v0.6.0` and `/home/me/llamacpp-v0.7.0` contains llama-server and shared libraries, but no llama-quantize executable. No build or install was attempted. `gguf_kquants.py` is a complete conditional command recipe taking an already-approved quantizer path; it cannot currently run. The owner must provide an existing approved quantizer or authorize a separate build. A quantizer from another version must support MXFP4_MOE and these override/fallback rules, not merely the four dense types.

### Expected bytes per token

[INFERENCE] Batch one; four selected experts/layer; each weight streamed once; no KV, activation, padding, cache-reuse or speculation traffic. The source estimate includes expert biases, router, norms and sinks. GGUF estimates include F32 small/router state. Read **one embedding row**, not the full embedding table. Converted attention has 637,009,920 weights; head has 579,133,440; one embedding row has 2,880. Baseline is 3,708,089,088 bytes/token.

| Feasible or conditional layout | Weight bytes/token | Saving vs source |
|---|---:|---:|
| vLLM MXFP4 + FP8 channel attention, BF16 head/rest | 3,071,847,168 | **636,241,920 (17.16%)** |
| vLLM MXFP4 + FP8 tensor attention, BF16 head/rest | about 3,071,079,552 | **about 637,009,536 (17.18%)** |
| GGUF MXFP4 + Q8_0 attention/head/embedding control | 2,574,705,396 | **1,133,383,692 (30.57%)** |
| GGUF MXFP4 + Q5_0 attention/head/embedding | 2,118,650,556 | **1,589,438,532 (42.86%)** |
| GGUF MXFP4 + Q4_0 attention/head/embedding | 1,966,632,276 | **1,741,456,812 (46.96%)** |
| Conditional Q6_K request: O=Q6_K, Q/K/V/head/embedding=Q8_0 | 2,506,138,356 | **1,201,950,732 (32.41%)** |
| Conditional Q5_K request: O=Q5_K, Q/K/V/head/embedding=Q5_1 | 2,176,964,976 | **1,531,124,112 (41.29%)** |
| Conditional Q4_K request: O=Q4_K, Q/K/V/head/embedding=Q5_0 | 2,083,261,116 | **1,624,827,972 (43.82%)** |

Q8_0/Q5_0/Q4_0 use 34/22/18 bytes per 32 values; Q5_1 uses 24. Q6_K/Q5_K/Q4_K use 210/176/144 bytes per 256 values. The Q6_K-request layout saves only 68,567,040 bytes/token versus the Q8_0 control because the large head and Q/K/V remain Q8_0. These savings are not performance predictions.

## Conversion commands and status

Run locally on the intended GB10 only after confirming source revision/completeness. The channel-attention conversion and scale diagnostic below have completed; the tensor-scale and GGUF alternatives remain unrun and are not being built. All use an existing approved environment. No environment installation or modification is required for Python quantizers.

```sh
PY=/home/me/modelopt-0.47.0-venv/bin/python
CONVERT=/home/me/capyctl-recipes/vllm/gpt-oss-20b-capyctl-gb10/convert

# CPU scale diagnostic; the initial missing-source-shard attempt was resolved.
"$PY" "$CONVERT/cast_share.py" /home/me/models/gpt-oss-20b

# vLLM-only mixed attention candidates. Never use these for SGLang 0.5.22.
"$PY" "$CONVERT/mixed_attention.py" /home/me/models/gpt-oss-20b /home/me/models/gpt-oss-20b-capyctl-fp8attn --scale channel
"$PY" "$CONVERT/mixed_attention.py" /home/me/models/gpt-oss-20b /home/me/models/gpt-oss-20b-capyctl-fp8-tensor --scale tensor

# Install-free GGUF candidates. Expert bits remain OpenAI's MXFP4 values.
"$PY" "$CONVERT/gguf_dense.py" /home/me/models/gpt-oss-20b /home/me/models/gpt-oss-20b-capyctl-MXFP4-Q8_0.gguf --dense-type q8_0
"$PY" "$CONVERT/gguf_dense.py" /home/me/models/gpt-oss-20b /home/me/models/gpt-oss-20b-capyctl-MXFP4-Q5_0.gguf --dense-type q5_0
"$PY" "$CONVERT/gguf_dense.py" /home/me/models/gpt-oss-20b /home/me/models/gpt-oss-20b-capyctl-MXFP4-Q4_0.gguf --dense-type q4_0
```

K-quant commands are generated by `gguf_kquants.py`; its required `--quantizer` argument intentionally has no fabricated default. None of the approved installs currently supplies that executable.

## Agreement-test protocol

The channel-attention run above completed this protocol. Schedule any new GPU run only after confirming the GPU is free of foreign jobs. Every future CapyCTL deployment must set `lmcache: off`, and launch logs must confirm KV dtype/cache policy.

Reuse `comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/tensors/next_token_agree.py` and its `agreement-prompts.json` in this worktree. Keep that corpus's IDs, categories, messages and tools exactly. Reuse its `score_record`/`summary` calculations (`:112-126`), reference-answer record schema (`:149-150`), teacher-forcing alignment (`:195-205`), and hashes/coverage reporting (`:188-223`). **Do not run its Gemma model constructor unchanged**: `:22,51-59` is Gemma-specific. Its FP8 reconstruction pattern is at `:69-81`.

1. **Pin inputs and preserve experts.** Record the source revision and SHA256 of source config, safetensor index, tokenizer, canonical chat template and corpus. Verify every source tensor is present. For the Quark derivative, concatenate per-expert tensors, reshape `_blocks` back to their source shape, and compare exact bytes for every expert packed weight, E8M0 scale and bias against the source hashes in `conversion.json`. All other non-attention tensors must be bit-identical. For GGUF, invert the converter's nibble shuffle and gate/up split, or compare independently dequantized expert tensors exactly; verify all 72 expert matrices and every bias. Assert recorded types, shapes, finite values and full parameter coverage; never silently skip missing keys.
2. **Build the correct teacher.** Instantiate the pinned GPT-OSS Transformers architecture, not Gemma. Remove quantization_config for an ordinary BF16 graph; materialize native MXFP4 experts as BF16 dequantizations of OpenAI's released weights (e.g. the approved environment's Transformers `Mxfp4Config(dequantize=True)` loader). Keep head untied, sinks, expert bias, interleaved gate/up layout, alpha=1.702 and clamp=7 intact. This is not an unavailable prequantization BF16 teacher.
3. **Freeze GPT-OSS answer IDs once.** Render the same 200 prompt records with OpenAI's pinned Harmony template and one explicit reasoning effort, identical for every candidate. Do not reuse Gemma answer IDs or Gemma's enable_thinking=False flag. Match the Gemma protocol's greedy settings, maximum 512 generated tokens and reference record schema. Preserve the original tools. Record EOS/stop IDs, whether each answer hit the cap, exact rendered prompt token IDs and generation batch size. Report truncated records separately, not as successful completed answers.
4. **Teacher-force the teacher too.** Recompute teacher predictions from `prompt_ids + answer_ids[:-1]`, selecting logits starting at n_prompt-1, for all answer positions. Unlike the old harness's generation-only 100% baseline (`:216`), record this actual teacher-forcing baseline: full-sequence arithmetic can differ from cached generation. Candidate-vs-teacher-forced argmax is the fidelity comparison; candidate-vs-generated IDs is also reported using the existing scorer.
5. **Weight-only reconstruction first.** Instantiate that same BF16 GPT-OSS graph for each candidate. Reconstruct FP8 attention as `fp8.float() * scale`, broadcasting per-output-channel scales correctly, then cast once to BF16. Use unchanged expert values and all other tensors. For GGUF, dequantize its actual dense/head/embedding types, restore tensor names/orientation, and use exactly the same BF16 graph. This isolates checkpoint error from activation quantization and engine kernels. Per-tensor QKV uses one exported common scale, avoiding load-time requantization drift.
6. **Score every reference position.** Reuse the Gemma harness's strict-length agreement mask, total agreement, by-category agreement and first-disagreement statistics. Keep per-record picked IDs and masks. Record logit top-1/top-2 margins for disagreements, full coverage counts, tensor/type inventory and software versions. Do not treat generation divergence or perplexity alone as teacher-forced agreement. Compare Q8_0 as a real quantized control, not as OpenAI's numerical layout.
7. **Then qualify native engines.** Weight-only reconstruction does not test FP8 input quantization, MoE activation precision, sinks in the engine, or quantizer/backend loading. In a separately scheduled run, require launch-log proof of Quark FP8 attention GEMMs and native MXFP4 expert backend, no ignored tensor/scale warnings, preserved bias/SwiGLU behavior, matched KV settings, and the full existing task-quality/acceptance gates. GGUF candidates similarly need native-kernel quality checks. Only publish a CapyCTL checkpoint if quality gates pass and the existing matched performance harness measures a win over the reference; otherwise publish the winning recipe using OpenAI's checkpoint.
