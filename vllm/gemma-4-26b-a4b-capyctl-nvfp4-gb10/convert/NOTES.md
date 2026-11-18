# Gemma 4 NVFP4 and mixed FP8 conversion

## Current result

All four checkpoints exist on host A: `~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r1`, `-r2`, `-r3` and `-r3e`, with passing layout audits, conversion manifests and offline explicit-decode sanity loads. r3/G is mixed NVFP4-expert/FP8-body with a separate FP8 head; r3e/E keeps the head tied BF16. Sizes and shard hashes are below. The complete source passed pinned Hugging Face file-size, shard-SHA-256, tensor-index and payload checks. Every build uses the same byte-identical 512-conversation calibration set. No inference engine was started by conversion and no environment was modified.

**Approved calibration gate:** all 60 actual shared input quantizer amax values must be finite and positive; all 7,680 per-expert fused weight quantizer amax values must be present, finite and positive; every exported `input_scale` and `weight_scale_2` must be finite and positive. Per-layer selected-expert counts and never-routed expert IDs are information, not a failure. The gate and full routing coverage are recorded in `expert-calibration-audit.json` and `conversion-manifest.json`. No synthetic routing or fabricated per-expert input observations are used. The earlier attempt below stopped on a stricter all-experts-selected condition that is not part of the approved gate.

Revision 1 completed successfully (`r1-approved-gate.log`) with a passing 47,648-tensor layout audit, all 11,725 requested quantized matrices and 748 unchanged BF16 tensors. The successful run measured:

| Observation | Measured value |
|---|---:|
| Calibration batches / samples / maximum length | 128 / 512 / 1,024 tokens |
| Calibration forward loop | 7 min 56 s |
| ModelOpt export | 111.576 seconds |
| Calibration/export stage | 644.819 seconds |
| Converter elapsed before final manifest hashing | 759.886 seconds |
| Timed whole command | 12 min 48.45 s |
| Peak CUDA allocation | 55,976,323,072 bytes |
| Peak CUDA reservation | 56,115,593,216 bytes |
| Peak process RSS | 11,988,434,944 bytes |
| Minimum host MemAvailable | 26,256,871,424 bytes |
| Whole-host peak non-available memory, including another workload's engine | 104,406,298,624 bytes |

The 8-GiB guard did not fire. Source assets were copied byte for byte, including processor/tokenizer/chat-template/generation configuration, despite ModelOpt's optional processor auto-save warning. The output audit verified those copies.

All 60 shared input amax values and 7,680 per-expert fused weight amax values passed. Of 3,840 layer/expert pairs, 3,833 were selected during calibration:

| Layer(s) | Unique experts selected per layer | Never-routed expert IDs |
|---|---:|---|
| 0 | 126 / 128 | 42, 82 |
| 1 | 125 / 128 | 17, 19, 126 |
| 2 | 127 / 128 | 89 |
| 5 | 127 / 128 | 12 |
| 3, 4, 6–29 | 128 / 128 | None |

The seven unselected experts retain quantized weights and the corresponding genuinely calibrated shared input scale. They are not claimed to have independent input observations. The full per-layer report is in both calibration audit and manifest.

An earlier complete calibration (`r1-complete-state.log`, no export) stopped on the superseded all-experts-selected gate: 8 min 03 s calibration, 10 min 51.04 s whole command, 6,589,452,288-byte peak RSS and 26,764,877,824-byte minimum MemAvailable. Its CUDA peaks were the same as the successful run; no memory guard fired.

### Revision 2 measured conversion

All 11,725 matrices were re-rounded and checked against BF16. Every final packed tensor was finite and no worse than its revision-1 tensor in squared error. The output layout audit passed on all 47,648 tensors (`r2.log`):

| Observation | Measured value |
|---|---:|
| Requantization/audit elapsed before final manifest hashing | 452.904 seconds |
| Timed whole command | 7 min 44.20 s |
| Peak CUDA allocation | 752,615,424 bytes |
| Peak CUDA reservation | 1,176,502,272 bytes |
| Peak process RSS | 11,561,324,544 bytes |
| Minimum host MemAvailable | 76,992,778,240 bytes |
| Whole-host peak non-available memory | 53,670,391,808 bytes |

The same co-resident engine remained allocated. Reference-shard SHA-256 hashing overlapped this run for 24.60 seconds; these are operational conversion timings, not an isolated throughput benchmark. The 8-GiB guard did not fire.


## Pinned inputs

| Build | Repository | Revision | Published tensor payload, bytes |
|---|---|---|---:|
| BF16 source | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 51,611,872,412 |
| NVIDIA reference | `nvidia/Gemma-4-26B-A4B-NVFP4` | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` | 18,782,360,732 |
| RedHat reference | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | `5557756b8dce33ac72f2bd702b11729fdba3b839` | 16,417,035,012 |

These payload sizes are read from the pinned Hugging Face indexes, not measured local files. Source SHA-256 values published by Hugging Face:

- `model-00001-of-00002.safetensors`: `1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3`.
- `model-00002-of-00002.safetensors`: `aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7`.

Observed environment: ModelOpt 0.47.0, torch 2.13.0+cu130, transformers 5.14.1, datasets 5.1.0, safetensors 0.8.0, Python 3.12.3, CUDA runtime 13.0, driver 580.173.02, aarch64, kernel 6.17.0-1031-nvidia, compute capability 12.1. ModelOpt checkout: clean tag 0.47.0 / commit `c10fb3bf061229798b9e5fc53e5d9dd09c1bc7a5`; `examples/hf_ptq/hf_ptq.py` SHA-256 `a6ee271925202f11e07f8c4b10d8a09acdcd34b11ee2ca74c8ac5f952f504189`, `example_utils.py` SHA-256 `ccae9a0cff8e255bd4ce25f34558c34b475aa20a4cdef04bb1686384c4e37bc5`. Only the existing `~/modelopt-0.47.0-venv` and checkout were used. Scripts live under `~/gemma4-work/convert`, calibration data and logs under `~/gemma4-work`, and weights under `~/models`.

The streaming loader used accelerate 1.15.0. Temporary files were directed to the existing `~/gemma4-work/temp` directory with `TMPDIR`; create that directory during input preparation when reproducing on a fresh host.

## Revision 3 mixed layouts

The same driver accepts `--layout G` (r3) and `--layout E` (r3e); its default `all-nvfp4` preserves the r1 command, and the separate r2 command remains unchanged. Both mixed layouts use **11,520 routed-expert NVFP4 W4A4 group-16 matrices**, with r2's original-plus-21-candidate block-scale search applied only to those expert tensors. All **90 dense-MLP and 115 attention projections** are **per-tensor FP8 E4M3 W8A8**, with FP32 scalar `weight_scale` and calibrated static `input_scale`. Global-attention layers still have no invented V projection.

G clones the original tied BF16 head into a separate in-memory `lm_head` parameter **before ModelOpt quantization**, sets both outer/text `tie_word_embeddings=false`, and exports that head in the same per-tensor FP8 format. E keeps the output head tied to the original BF16 embedding. The input embedding, router, norms, learned layer scalars and vision remain byte-identical BF16. The audit checks every tensor's name, dtype and shape, scalar-scale shape, positive finite activation/global scales, finite FP8 weights, preserved payloads, exact mixed per-layer metadata and copied assets. Scale search additionally hashes all non-expert tensors before/after to prove FP8 and retained tensors are unchanged.

Both use ModelOpt 0.47.0 and the same SHA-256-locked 512-conversation calibration set and approved shared-input/expert-weight gate as r1. The body FP8 recipe is a text-layer-only driver override of the existing recipe; the manifest records layout, recipe hash and converter hash. Revision-3 export automatically invokes the expert-only search and then `sanity_load.py` in separate processes after calibration/export has released its model. All GPU phases retain the 8-GiB MemAvailable guard; no engine is started.

### FP8 format decision and installed-source citations

**Choose plain ModelOpt `FP8`, not `FP8_PER_CHANNEL_PER_TOKEN`, including the untied head.** The HF config uses `quant_algo: MIXED_PRECISION` and `quantized_layers` entries `{quant_algo: NVFP4, group_size: 16}` for experts and `{quant_algo: FP8}` for each body/head FP8 module. The matching converted config is produced by ModelOpt, not a compressed-tensors replacement. This is conditional source-level loader compatibility in **both SGLang 0.5.22 and vLLM 0.32.0**, not native-engine qualification.

Citation roots: **SG** = `~/sglang-0.5.22-venv/lib/python3.12/site-packages/sglang/`; **VL** = `~/vllm-0.32.0-venv/lib/python3.12/site-packages/vllm/`; **MO** = `~/modelopt-0.47.0-venv/lib/python3.12/site-packages/modelopt/`. These files were read without modifying either engine environment.

| Contract | SGLang installed file:line | vLLM installed file:line |
|---|---|---|
| Attention passes quant config to fused QKV/output projections | `srt/models/gemma4_causal.py:357-373` | `model_executor/models/gemma4.py:423-445` |
| Dense gate/up/down passes quant config to fused gate/up and down | `srt/models/gemma4_causal.py:71,87,592-599`; `srt/models/gemma3_causal.py:83-109` | `model_executor/models/gemma4.py:266-280` |
| Untied quantization-aware head | `srt/models/gemma4_mm.py:254-268` | `model_executor/models/gemma4.py:1447-1454` |
| Mixed FP8 dispatch includes linears **and ParallelLMHead** | `srt/layers/quantization/modelopt_quant.py:1047-1062` | `model_executor/layers/quantization/modelopt.py:139-149,1847-1856,2677-2684` |
| FP8 weight `[out,in]`, E4M3; FP32 per-tensor weight/input scales | `srt/layers/quantization/modelopt_quant.py:568-608` | `model_executor/layers/quantization/modelopt.py:2104-2140` |
| Resolve source unfused names/fused projections and head aliases | `srt/layers/quantization/modelopt_quant.py:977-1030` | `model_executor/layers/quantization/modelopt.py:1769-1811` |

SGLang's mixed dispatcher has no `FP8_PER_CHANNEL_PER_TOKEN` case and otherwise returns unquantized (`SG/modelopt_quant.py:1047-1062` under the quantization directory above). vLLM supports that per-channel/dynamic-token format (`VL/modelopt.py:139-143,2685-2691` under its quantization directory), but it is not a shared serialization. **A per-tensor FP8 head is dispatched by both**, so no vLLM-only r3 exception is needed. MO `torch/export/unified_export_hf.py:631-658` writes the scalar FP32 weight/static input scales; `torch/export/convert_hf_config.py:236-259` preserves the mixed per-layer map.

The G ablation used a per-channel BF16-rounded head scale; r3 deliberately uses the shared-engine **per-tensor** format. Body E already used per-tensor scales. Agreement on the real exported tensors, rather than the simulation, is therefore the acceptance measurement; all offline checks reconstruct stored weights and retain BF16 activations, and do not measure native W8A8/NVFP4 activation/kernel error.

Both mixed exports additionally set **`hf_quant_config.json.quantization.kv_cache_quant_algo: "FP8"`**, matching NVIDIA's declaration, and regenerate `config.json.quantization_config` with the equivalent FP8 KV scheme. This is metadata enabling the default FP8 KV policy, **not a claim of calibrated KV scales**: no K/V scale tensors are added. SGLang reads the nested field at `SG/srt/layers/quantization/modelopt_quant.py:875-879`, stores it in the mixed config and dispatches KV loading at `:1075-1076`; its automatic dtype resolver consumes `model.quant_config.kv_cache_quant_algo` and chooses E4M3 at `SG/srt/mem_cache/kv_cache_dtype.py:38-46`. vLLM reads nested `kv_cache_quant_algo` or converted `kv_cache_scheme` at `VL/model_executor/layers/quantization/modelopt.py:330-350` and dispatches its KV method at `:1817-1820`. Absent scales fall back to **1.0** in SG `srt/layers/quantization/kv_cache.py:59-63` and VL `model_executor/layers/quantization/kv_cache.py:123-127`; vLLM warns about that uncalibrated E4M3 default at `:162-166`. The offline reconstruction/agreement keeps BF16 KV and therefore does not evaluate this FP8 KV policy.

**SGLang-0.5.22-only expert dispatch aliases:** ModelOpt exports one canonical NVFP4 recipe per fused expert bank (30 entries), not per individual stored expert projection. SGLang Gemma4 constructs those modules under `.moe.experts` (`SG/srt/models/gemma4_causal.py:287-294,646-651`), and its tensor loader remaps `.experts.` into `.moe.experts.` (`SG/srt/models/gemma4_mm.py:957-961`), but its mixed recipe resolver only checks exact/descendant keys plus outer-prefix aliases (`SG/srt/layers/quantization/modelopt_quant.py:977-1030`). Therefore the export adds **30 explicit `.moe.experts` recipe aliases** beside the 30 canonical `.experts` entries; no duplicate tensors are stored. These aliases exist only for that SGLang resolver mismatch and can be dropped when a later engine normalizes the segment itself.

vLLM retains its actual `.experts` module prefix (`VL/model_executor/models/gemma4.py:646-653`). Its mixed parser reads and retains the full recipe map (`VL/model_executor/layers/quantization/modelopt.py:1647-1658,1707-1710,1563-1565`); its constructor validates specialized block-codec contracts, not module-name existence (`:1566-1578`). Actual modules look up their canonical entries first (`:1729-1732`) before descendant fallbacks, so the extra SGLang-only names neither select a wrong algorithm nor cause unknown-name errors. `engine-format-audit.json` records an **AST-only execution of the installed resolver methods**, without importing or starting engines: **542 routes** across both engines (all 30 expert banks, fused attention/MLP projections, retained router/norm paths and the head) passed, and removing the 30 SGLang aliases left every tested vLLM route unchanged. This is dispatch evidence, not an engine-load test.


### One-command builds on host A

From the directory containing `convert/`, with the pinned source, calibration and existing environment already present:

```bash
env TMPDIR=$HOME/gemma4-work/temp \
  ~/modelopt-0.47.0-venv/bin/python convert/convert_gemma4_26b_a4b_nvfp4.py \
  --layout G --source ~/models/gemma-4-26b-a4b-it \
  --out ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r3 \
  --hf-ptq-dir ~/Model-Optimizer-0.47.0/examples/hf_ptq \
  --calib ~/gemma4-work/calib-chat-code-512.jsonl --gpu-gib 100

env TMPDIR=$HOME/gemma4-work/temp \
  ~/modelopt-0.47.0-venv/bin/python convert/convert_gemma4_26b_a4b_nvfp4.py \
  --layout E --source ~/models/gemma-4-26b-a4b-it \
  --out ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r3e \
  --hf-ptq-dir ~/Model-Optimizer-0.47.0/examples/hf_ptq \
  --calib ~/gemma4-work/calib-chat-code-512.jsonl --gpu-gib 100
```

An initial G attempt was rejected by the tensor-inventory audit because broad post-recipe FP8 overrides also selected vision projections. No rejected checkpoint was accepted or evaluated: its generated weights were removed, the overrides were restricted to `language_model.layers`, and G was rebuilt from the pinned BF16 source and calibration set. The driver also checks enabled text/head linear coverage before its calibration forward loop; the final tensor audit independently enforces unchanged BF16 vision.

The corrected G export passed all tensor checks, then stopped at an overly strict metadata assertion that had expected 11,520 expert recipe entries rather than ModelOpt's actual 30 fused banks. Metadata finalization added the documented SGLang aliases and accepted the bank-level schema without changing tensors or recalibrating. G calibration/export used converter SHA-256 `f3bf42b56c9a8c752e2a0150c13c74cc4b748fe9194bbdc07db053d6dc47a624`; its final manifest records the later finalizer/audit implementation.

The first expert-search invocation then exposed a pre-existing r2 assumption: weight, block scale and global scale were in the same shard. The larger mixed layout straddles a 5-GB boundary for one expert block scale. `requant_scales.py` now reads scales by the original index and defers only boundary-scale writes, preserving every tensor's shard assignment. G's complete restarted search exercised **one** cross-shard block-scale update and checked all 11,520 final expert tensors. Its legacy `r1_squared_error` fields describe each tensor's state at the beginning of that invocation, including already-searched tensors from the interrupted attempt; the final, independently BF16-referenced expert relative error is **8.5899864%**. No FP8/non-expert tensor changed during search.

### Measured mixed build results

r3/G passes its **47,446-tensor** final layout audit: 11,520 NVFP4 matrices, 206 FP8 matrices and 748 preserved BF16 tensors. All 60 shared expert input quantizers and 7,680 expert weight quantizers passed. Routing coverage was **3,833/3,840**, with exactly the same seven never-routed layer/expert pairs listed for r1 above. Its offline reconstruction populated **1,014 state tensors**, retained separate head/embedding storage, produced finite logits and answered **`4`**.

r3e/E passes its **47,443-tensor** final audit: the same 11,520 NVFP4 experts and 748 unchanged BF16 tensors, **205** FP8 body matrices and a tied BF16 head. Its approved 60-input/7,680-weight gate and 3,833/3,840 routing coverage match G and r1. The clean expert search reduced relative weight error from **9.4830936% to 8.5899864%**, exercised one cross-shard scale update and left every non-expert tensor unchanged. Offline reconstruction loaded **1,013 state tensors**, verified the tie and finite logits, and answered **`4`**. Both checkpoints also pass the 542-route installed-source dispatch audit.

| Build / stage | Elapsed seconds | Peak CUDA allocated bytes | Peak CUDA reserved bytes | Peak process RSS bytes | Minimum host MemAvailable bytes |
|---|---:|---:|---:|---:|---:|
| r3 calibration/export subprocess | 690.611 | 57,452,719,104 | 57,583,599,616 | 11,716,521,984 | 25,626,431,488 |
| r3 complete expert search/audit, before manifest hashing | 505.738 | 66,512,896 | 100,663,296 | 13,180,731,392 | 79,470,014,464 |
| r3 offline sanity reconstruction | 28.454 | 57,530,131,456 | 58,615,398,400 | 5,594,988,544 | 26,172,444,672 |
| r3e calibration/export subprocess | 605.661 | 55,976,323,072 | 56,115,593,216 | 12,053,798,912 | 26,084,405,248 |
| r3e expert search/audit, before manifest hashing | 442.908 | 66,512,896 | 100,663,296 | 10,878,328,832 | 78,481,645,568 |
| r3e offline sanity reconstruction | 27.169 | 53,648,083,456 | 54,022,635,520 | 4,526,522,368 | 31,409,209,344 |

G's corrected calibration/export command through the metadata-assertion stop took **13:29.60**; ModelOpt's calibration loop took **7:58** and export **97.453 seconds**. The complete restarted search command took **8:35.72**, and sanity **32.49 seconds**. Metadata finalization and the interrupted search are separately logged in `r3-search-sanity.log`; the completed search/sanity is `r3-search-sanity-final.log`. These are observed split-stage timings, not a claimed uninterrupted whole-command timing. All completed GPU stages exited 0 and their 8-GiB guards remained clear. The original co-resident allocations were left untouched.

r3e's full documented command completed uninterrupted in **20:07.66**, exit **0** (`r3e.log`), including its initial/final audits, calibration/export, scale search, shard hashing and sanity reconstruction. Its calibration loop took **7:57**, ModelOpt export **110.270 seconds**, and the converter's pre-search stage before manifest hashing **715.419 seconds**. No memory guard fired.

### Stored-format next-token agreement

`tensors/next_token_agree.py` evaluates these **real stored formats**, not simulated FP8, on the unchanged 200 prompts / 62,658 BF16-reference answer positions. It explicitly reconstructs weights into BF16, so its agreement does not qualify native activation quantization, FP8 KV or engine kernels. Full commands, category rows and resource reports are in [the tensor comparison notes](../../../comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10/tensors/NOTES.md).

| Exported build | Agreement % | Matching ablation | Ablation agreement % | Difference, percentage points |
|---|---:|---|---:|---:|
| r3 / G, per-tensor FP8 head | 97.177 | G, per-channel FP8 head | 97.134 | +0.043 |
| r3e / E, tied BF16 head | 97.191 | E, tied BF16 head | 97.151 | +0.040 |

Both exported builds are within **0.1 percentage points** of their matching ablations. r3e's median first-disagreement position is **30**; the full per-category and first-disagreement records are retained in the linked comparison results. The format change from G's simulated per-channel head to r3's shared-engine per-tensor head did not create a larger-than-allowed agreement gap.

**[INFERENCE] Weight-read arithmetic, not measured bandwidth:** with the comparison's one-read-per-weight assumption, eight selected experts per layer and static input scales omitted, the FP8 body/expert/router family subtotal is 2,470,186,100 bytes/token. G's stored per-tensor FP8 head adds 738,197,508 weight/weight-scale bytes, for **3,208,383,608 bytes/token**, versus **5,592,025,408** for NVIDIA's BF16 dense/attention/head layout. E's tied BF16 head gives **3,946,581,108 bytes/token**. Embedding lookup rows, other state, activations, KV traffic, buffers and cache/batching reuse are outside that arithmetic; it is not a tokens/second claim.

## Reproduce

First let the existing pinned source download finish. The converter refuses a missing/truncated shard, inventory/payload mismatch, or mismatch with pinned HF file sizes. It records every source shard's SHA-256. It uses anonymous HF metadata requests; it does not need or print a token.

### Loader and memory safety

The driver caps exported shards at **5 GB**, rather than ModelOpt's 10-GB default, to bound save-time CPU staging beside the resident engine. This serialization setting is recorded in the manifest and does not change tensor coverage or quantization.

Another workload's SGLang engine occupied 33,190 MiB (about 33 GiB), plus a 170-MiB parent, during conversion. Those processes were left alone. The first loading attempt failed because ModelOpt's `get_model` used the fixed budget for preliminary inference but omitted it from its final `from_pretrained` call; that call auto-offloaded and calibration encountered meta tensors (200.69 seconds, log `r1.log`). A sequential-map attempt retained the budget but the HF loader's staging transient was OOM-killed (278.70 seconds, log `r1-fixed-placement.log`).

The final driver instead constructs a meta model and streams exactly one source BF16 state tensor at a time directly onto GPU 0 using existing `accelerate`/torch APIs and safetensors 0.8's **CUDA `backend="pread"` path**. Persistent buffers, including all 30 learned `layer_scalar` values, are loaded explicitly alongside parameters. The CPU safetensors path was not sufficient: it stopped cleanly at the memory guard after 832 parameters (305.96 seconds; `r1-streaming.log`, retained resource record `memory-guard-pageable.json`). A bounded 512-parameter diagnostic held identical BF16 GPU weights, 14,146,565,984 bytes, but CUDA pread increased host MemAvailable from 58,341,580,800 to 70,364,491,776 bytes and reduced process RSS from 14,727,816 to 4,711,008 KiB. CUDA cache release and `malloc_trim` did not recover the CPU-path overhead. These diagnostics measured loading, not inference. The final path uses one bounded pinned staging tensor, checks the parameter payload against `--gpu-gib`, verifies complete tensor coverage/shapes, retains the tied head, and never selects disk offload. The documented command remains one command.

`memory_guard.py` samples host `MemAvailable` every 100 ms during calibration/export, revision-2 requantization and offline sanity loading. If it falls below 8 GiB, the guard sends SIGINT **only to its own process**, unwinding the conversion instead of risking the resident engine. Manifests distinguish the minimum host MemAvailable and whole-host peak non-available memory from process RSS and CUDA peak allocations/reservations; the whole-host measure includes another workload's engine. It is not a model-only footprint. Resource reports record anonymous co-resident GPU memory amounts, not process identities or paths.

Reuse the exact Qwen precedent's `calib-chat-code-512.jsonl`, or rebuild it from the same UltraChat test-SFT parquet and Magicoder JSONL:

```bash
~/modelopt-0.47.0-venv/bin/python convert/make_calib.py \
  --chat test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet \
  --code data-oss_instruct-decontaminated.jsonl \
  --out ~/gemma4-work/calib-chat-code-512.jsonl
```

Both builder and converter require SHA-256 `a3e005c39c3c71918ffff53504091ed32bd02baf3d37241072e5a616b0dc31eb`: 256 UltraChat plus 256 Magicoder conversations, unchanged message selection and JSON serialization. ModelOpt applies **Gemma's source chat template**, with at most 1,024 tokens per sample. The manifest records the calibration and chat-template hashes. This is text-only calibration; the vision weights remain byte-identical BF16.

Revision 1, one command from the directory containing `convert/`:

```bash
env TMPDIR=$HOME/gemma4-work/temp \
  ~/modelopt-0.47.0-venv/bin/python convert/convert_gemma4_26b_a4b_nvfp4.py \
  --source ~/models/gemma-4-26b-a4b-it \
  --out ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r1 \
  --hf-ptq-dir ~/Model-Optimizer-0.47.0/examples/hf_ptq \
  --calib ~/gemma4-work/calib-chat-code-512.jsonl --gpu-gib 100
```

Revision 2, one command:

```bash
env TMPDIR=$HOME/gemma4-work/temp \
  ~/modelopt-0.47.0-venv/bin/python convert/requant_scales.py \
  --source ~/models/gemma-4-26b-a4b-it \
  --ckpt ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r1 \
  --out ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r2
```

The output directories must not already contain safetensors shards. Revision 2 keeps every global weight scale and static input scale. It re-rounds every quantized matrix, including all expert gate/up/down matrices, using the original scale plus 21 FP8-rounded block-scale candidates, 0.60 through 1.10 times the max-derived scale in steps of 0.025. This minimizes squared BF16 weight error over that finite candidate set, not over all possible scales. Every decoded result must be finite and no worse in aggregate than its revision-1 tensor. Neither lower weight MSE nor a passing layout audit establishes task accuracy.

The scripts write `layout-audit.json`, `conversion-manifest.json`, per-shard SHA-256 values and measured timing/memory metadata. Revision 1 also writes `expert-calibration-audit.json`; revision 2 writes `requant-scales.json` and `requant-summary.json`. `files_bytes` and `allocated_bytes` include the completed manifest and all other regular files in the output directory; tensor payload is recorded separately. Peak RSS is process RSS, not whole-machine unified-memory usage. CUDA peak allocation/reservation is reported separately.

### Offline sanity load

```bash
env TMPDIR=$HOME/gemma4-work/temp \
  ~/modelopt-0.47.0-venv/bin/python convert/sanity_load.py \
  --ckpt ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r1
env TMPDIR=$HOME/gemma4-work/temp \
  ~/modelopt-0.47.0-venv/bin/python convert/sanity_load.py \
  --ckpt ~/models/Gemma-4-26B-A4B-NVFP4-capyctl-r2
```

This explicitly decodes the packed checkpoint into a BF16 Transformers model, restores fused expert tensors and persistent buffers, checks tied embedding/head identity and finite logits, and generates up to 16 tokens for a simple arithmetic prompt. It records `offline-sanity-load.json` and updates the manifest. This is an offline serialization/execution sanity check, **not** native NVFP4 kernel/engine qualification or an activation-quantization accuracy evaluation. It uses the same 8-GiB host-memory guard.

Revision 1 passed on all 1,013 source state tensors and generated **`4`** for the arithmetic prompt. Its explicit-decode sanity load took 25.334 seconds (29.33 seconds timed command), with 53,672,987,136-byte peak CUDA allocation, 4,547,809,280-byte peak RSS and 31,906,484,224-byte minimum host MemAvailable. Tied embedding/head identity and finite logits passed; no memory guard fired (`r1-sanity-final.log`).

Revision 2 also passed on all 1,013 state tensors and generated **`4`**. Its explicit-decode sanity load took 33.791 seconds, with 53,672,987,136-byte peak CUDA allocation, 4,497,448,960-byte peak RSS and 31,525,261,312-byte minimum MemAvailable. Its checkpoint's `offline-sanity-load.json` records tied embedding/head identity, finite logits and no guard stop.

## Layout and calibration guard

NVFP4 W4A4, group 16, on exactly 11,725 language projections:

- 11,520 routed-expert gate/up/down matrices: 30 layers × 128 experts × three projections.
- 90 always-active/shared MLP projections.
- 115 self-attention projections. Full-attention layers 5, 11, 17, 23 and 29 have no separate V projection: `attention_k_eq_v` shares K/V. The script derives expected names and shapes from the source, rather than inventing V tensors.

Packed values are `U8 [out, in/2]`, block scales `F8_E4M3 [out, in/16]`, global weight scale and static input scale `F32` scalars. Preserve tied embedding/output head without materializing `lm_head.weight`; preserve router projections/scales, norms, layer scalars, vision tower and `embed_vision` byte for byte. The audit rejects missing and unexpected tensors, wrong dtypes/shapes, nonfinite/uncalibrated scales, changed BF16 tensor payloads, changed source assets, untied embeddings, and KV-cache scales.

ModelOpt's generic fused-expert wrapper intercepts eager per-expert `F.linear`; grouped/batched matmul bypasses it. The driver installs ModelOpt's `force_eager_experts_impl_on_the_fly` helper before quantization. ModelOpt 0.47 uses **two shared activation quantizers per layer**, gate/up and down, not independent input quantizer objects per expert. The guard requires eager calls to all 60 actual input quantizers, valid positive amax on all 60, and valid positive amax on every one of the 7,680 fused expert weight quantizers. The output audit independently requires valid positive global input/weight scales on all 11,725 exported matrices. Each expert inherits the appropriate genuinely calibrated shared input scale; unselected experts are explicitly listed rather than assigned invented local observations. Per-expert call counts and observed input maxima remain in the calibration audit, and per-layer selected counts/never-routed IDs are also in the manifest.

Both revisions' audited tensor payload is **16,417,035,012 bytes**, matching RedHat's module coverage. On-disk files and shard hashes are recorded independently in each manifest; they include serialization headers and sidecars, not just tensor data.

## Engine configuration findings

These are **source-level findings, not engine-load qualification**:

- Keep ModelOpt's unified HF export: `hf_quant_config.json` declares `quant_algo: NVFP4`, `group_size: 16`, `kv_cache_quant_algo: null`, plus exclusions; `config.json.quantization_config` uses ModelOpt's converted format. SGLang 0.5.22's `ModelOptFp4Config.from_config` accepts both formats with explicit `--quantization modelopt_fp4`; vLLM 0.32 recognizes the ModelOpt NVFP4 algorithm and selects `modelopt_fp4` automatically. Do not replace the quantization config with RedHat's compressed-tensors config.
- NVIDIA uses ModelOpt names (`weight`, `weight_scale`, `weight_scale_2`, `input_scale`) and quantizes 11,520 expert matrices. RedHat quantizes 11,725 matrices, but uses compressed-tensors names (`weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`). Its global scales are **reciprocals** of ModelOpt's decode scales. The tensor comparison reader handles both conventions explicitly.
- The pinned index name sets agree on the 11,520 expert modules; RedHat additionally quantizes the same 90 MLP and 115 attention modules requested here. No global-layer V projection should be manufactured.
- **Checkpoint KV scales are not required for FP8 KV operation.** Both versions' `BaseKVCacheMethod` use scale 1.0 when checkpoint K/V scales are absent. vLLM warns for uncalibrated E4M3 scale 1.0; this is a quality warning, not a load requirement. SGLang's ModelOpt FP4 parser normalizes absent/null KV algo to `auto`, so an explicit serving `--kv-cache-dtype fp8_e4m3` remains important. Revision 1 therefore adds no K/V scales and makes no FP8-KV accuracy claim.
- The pinned NVIDIA index itself contains **zero** `k_scale`, `v_scale` or other text-attention scale tensors, despite declaring `kv_cache_quant_algo: FP8`. Its declaration therefore does not establish that KV scales were calibrated.

Sources: [ModelOpt expert wrapper/helper](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/modelopt/torch/quantization/plugins/huggingface.py), [export handlers](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/modelopt/torch/export/hf_export_handlers.py), [SGLang ModelOpt parser](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/layers/quantization/modelopt_quant.py), [SGLang KV fallback](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/layers/quantization/kv_cache.py), [vLLM ModelOpt parser](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/quantization/modelopt.py), [vLLM KV fallback](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/quantization/kv_cache.py), [compressed-tensors scale handling](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py).

## Tensor comparison

RedHat was downloaded on host A at the pinned revision into `~/models/gemma-4-26b-a4b-redhat-nvfp4`; its safetensors file is 16,423,500,596 bytes. Reproduce the complete four-build comparison with:

```bash
~/modelopt-0.47.0-venv/bin/python convert/read_tensors.py \
  --source ~/models/gemma-4-26b-a4b-it \
  --ckpt r1=$HOME/models/Gemma-4-26B-A4B-NVFP4-capyctl-r1 \
  --ckpt r2=$HOME/models/Gemma-4-26B-A4B-NVFP4-capyctl-r2 \
  --ckpt NVIDIA=$HOME/models/gemma-4-26b-a4b-nvidia-nvfp4 \
  --ckpt RedHat=$HOME/models/gemma-4-26b-a4b-redhat-nvfp4 \
  --out ~/gemma4-work/tensor-errors.csv
```

The reference-only measurement was actually run with the same command but without the two CapyCTL `--ckpt` arguments, writing `~/gemma4-work/reference-errors.csv` and `reference-errors.summary.json`; full output is `reference-errors-bounded.log`. It covered all 11,520 NVIDIA expert matrices and all 11,725 RedHat matrices in 377.139 seconds (timed command wall 6 min 19.34 s), with peak CUDA allocation 738,197,504 bytes and minimum host MemAvailable 87,857,512,448 bytes.

The CapyCTL r1/final-packed-r2 energies were measured during revision-2 conversion, not in a duplicate GPU comparison pass. Its `requant-scales.json` contains all 11,725 paired tensor records. Those records were converted to `capy-errors.csv` and pooled with the reference CSV into `~/gemma4-work/tensor-errors.csv` and `tensor-errors.summary.json`: **46,695 build/tensor rows**, with the common-module flag recomputed across all four builds. The summary preserves both actual measurement stages and their separate timing/memory reports. Local copies are under `/var/tmp/capyctl-webui/gemma/`.

The reader avoids safetensors 0.8's `pread get_slice` implementation, which reads the whole fused expert bank before slicing. It computes each contiguous BF16 expert/half offset from the checked safetensors header and reads only that matrix into a bounded pinned staging buffer. Gate, up and down for layer 0 expert 127 were verified bit-exact against the normal safetensors reader on the real source/GPU (`reader-check.log`). The earlier whole-bank comparison was deliberately interrupted after 6 min 13.71 s; its partial CSV was replaced by the complete bounded-read measurement. No partial-family result is used below.

The metric is `sqrt(sum((Wq-Wbf16)^2) / sum(Wbf16^2))`, pooled by tensor family, not an unweighted mean of per-tensor errors. The CSV contains every tensor; the adjacent `.summary.json` records family sums, counts and the exact intersection of modules quantized by all builds (11,520 expert matrices). NVIDIA's BF16 MLP/attention are not misreported as quantized measurements.

| Family | CapyCTL r1 | CapyCTL r2 | NVIDIA | RedHat |
|---|---|---|---|---|
| Routed experts | 9.48309% | 8.58999% | 9.48309% | 9.48422% |
| Shared MLP | 9.40486% | 8.56611% | BF16, not quantized | 9.40581% |
| Attention | 9.45257% | 8.58036% | BF16, not quantized | 9.45295% |

Revision 2 reduces relative L2 weight error by 9.4179% for experts, 8.9183% for shared MLP and 9.2273% for attention versus revision 1, with unchanged static input/global weight scales. This is not a task-accuracy or serving-speed claim.

## Completed checkpoint sizes

Sizes include the completed manifest and offline sanity report. Allocated bytes count filesystem blocks; total file bytes include tokenizer and other sidecars. Weight files additionally include safetensors headers.

| Build | Tensor payload, bytes | Safetensors files, bytes | All regular files, bytes | Allocated bytes |
|---|---:|---:|---:|---:|
| CapyCTL r1 | 16,417,035,012 | 16,423,196,524 | 16,463,421,999 | 16,463,482,880 |
| CapyCTL r2 | 16,417,035,012 | 16,423,196,524 | 16,466,321,759 | 16,466,382,848 |
| CapyCTL r3 / G | 17,875,114,764 | 17,881,255,628 | 17,924,353,176 | 17,924,423,680 |
| CapyCTL r3e / E | 17,136,917,252 | 17,143,054,556 | 17,186,151,607 | 17,186,222,080 |

## Safetensors SHA-256

Every output shard's size and digest is also in its conversion manifest. Reference digests were measured from the pinned host-A files (`reference-shards.sha256`, `reference-shards.stat`); NVIDIA's two weight files total 18,788,485,588 bytes, versus 18,782,360,732 bytes of tensor payload.

| Build | Shard | SHA-256 |
|---|---|---|
| CapyCTL r1 | `model-00001-of-00004.safetensors` | `416f2e6ee780d76e46e3d942e69b2b627417f9ad552ba006e306706364ba5a5f` |
| CapyCTL r1 | `model-00002-of-00004.safetensors` | `223ee5823d653db97399256001da3380ecb3035f26608240a4ad77ee87fc4cac` |
| CapyCTL r1 | `model-00003-of-00004.safetensors` | `2caaeaa3ffaf221f6f3d6609b3fe91c138af4f7c1acaa3add397e79ca7742d09` |
| CapyCTL r1 | `model-00004-of-00004.safetensors` | `d8cc1508a975bf5534e6b423fb011b72076ab71d0325bd0053c996dbee8f25d0` |
| CapyCTL r2 | `model-00001-of-00004.safetensors` | `ca5e1b8a5c66e6fe13bc1271e1062770d3b7812e16247a30d934dd4079d92f08` |
| CapyCTL r2 | `model-00002-of-00004.safetensors` | `6db49c3a17bd8b0b58149739a53757bd5ec6c3fc9321dbb8b752723190357f84` |
| CapyCTL r2 | `model-00003-of-00004.safetensors` | `058e8081f4a57f0fc3d53f423caa7d73db8cc5c0becce1ed37a1d58caf494a1b` |
| CapyCTL r2 | `model-00004-of-00004.safetensors` | `464ae1f2350bfc859eb25a92d73fbbae3f5d6c7d834c4d7d4f07f71d642479e7` |
| CapyCTL r3 | `model-00001-of-00004.safetensors` | `ccaa874b94103a9e490bddab10db66e2cfd9ef029a00b22aacfa38262e075d71` |
| CapyCTL r3 | `model-00002-of-00004.safetensors` | `7adff41c85b5bf7959595e4d67b05a512c8e18ad5365f3b5455046f8eea79962` |
| CapyCTL r3 | `model-00003-of-00004.safetensors` | `4abc79d45bb7fa8e399cfe519133177db4b374d3d332c28b11dcc86d67252773` |
| CapyCTL r3 | `model-00004-of-00004.safetensors` | `79e8092df19568754718cf4c5ad40eb41c01974a1170a6f9399b2896551e8ec8` |
| CapyCTL r3e | `model-00001-of-00004.safetensors` | `ccaa874b94103a9e490bddab10db66e2cfd9ef029a00b22aacfa38262e075d71` |
| CapyCTL r3e | `model-00002-of-00004.safetensors` | `7adff41c85b5bf7959595e4d67b05a512c8e18ad5365f3b5455046f8eea79962` |
| CapyCTL r3e | `model-00003-of-00004.safetensors` | `4abc79d45bb7fa8e399cfe519133177db4b374d3d332c28b11dcc86d67252773` |
| CapyCTL r3e | `model-00004-of-00004.safetensors` | `d5bfdf3d59fd6700ba0e37c4b8f334ebd91515074313cd4d097ec93d0f771d8d` |
| NVIDIA | `model-00001-of-00002.safetensors` | `b5df31122600666617b05f9be2015552cd2edff401e86b1d99b9127efdc6d819` |
| NVIDIA | `model-00002-of-00002.safetensors` | `ff11061ebf57327af4f1993ff758b0859d7746b0c03ca1b17ded7dec30410962` |
| RedHat | `model.safetensors` | `62fd36b05775b9462b7914af3f911998d372d8d5a200a34b9c5ae88666dce56f` |

No inference engine was started, no other workload was signaled, and no package/environment/driver change was made. Native-engine execution is intentionally outside this offline assignment.

After the original r1/r2 offline jobs exited, the only GPU allocations reported were the original 170-MiB parent and 33,190-MiB engine; host MemAvailable was 89,691,254,784 bytes. Both mixed conversion pipelines have also exited; their per-stage guards and the comparison's final exit audit distinguish them from the untouched resident workload.
