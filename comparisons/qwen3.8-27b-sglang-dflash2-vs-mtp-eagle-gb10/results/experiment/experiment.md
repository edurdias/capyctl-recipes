# Qwen3.8-27B on SGLang 0.5.21: checkpoint, drafter and depth variants, direct and through CapyCTL

The CapyCTL variants ask for `memory.request: 84GiB` (the committed recipe's 79 GiB is
refused; see the check below).

| Variant | C1 aggregate / per stream, tok/s | C8 aggregate / per stream, tok/s | C22 aggregate / per stream, tok/s | TTFT p50 C1 | Accept length | Memory saver | Saver preload | `mem_fraction_static` | KV dtype | KV tokens passed / engine, running | CUDA graph decode bs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 17.5 / 17.8 | 122.5 / 16.5 | 278.6 / 14.0 | 0.36 s | 2.65 | False | no | 0.65 | auto | n/a / 884688 / 22 | 256 |
| bf16-deep | 17.3 / 17.7 | 127.0 / 18.0 | 291.5 / 15.2 | 0.35 s | 2.73 | True | yes | 0.6502 | fp8_e4m3 | 894373 / 894373 / 22 | 256 |
| bf16-restart | 17.5 / 17.7 | 120.1 / 17.2 | 273.9 / 14.3 | 0.27 s | 2.62 | False | no | 0.6501 | fp8_e4m3 | 894373 / 894373 / 22 | 256 |
| nvfp4-restart | 24.7 / 24.9 | 156.5 / 20.9 | 336.6 / 17.2 | 0.21 s | 2.60 | False | no | 0.6468 | fp8_e4m3 | 894373 / 894373 / 22 | 256 |

## Sizing

| Variant | Memory request | Static pool (CapyCTL) | `--max-total-tokens` rendered | `mem_fraction_static` | Engine `max_total_num_tokens` | Engine `max_running_requests` | Ready footprint (`MemAvailable` drop) | CapyCTL Ready charge | Start |
|---|---|---|---|---|---|---|---|---|---|
| raw | none (raw) | n/a | n/a | 0.65 | 884688 | 22 | 83.9 GiB | n/a | 614 s |
| bf16-deep | 84.0 GiB | 76.0 GiB | 894373 | 0.6502 | 894373 | 22 | 82.9 GiB | 85.2 GiB | 370 s |
| bf16-restart | 84.0 GiB | 76.0 GiB | 894373 | 0.6501 | 894373 | 22 | 79.4 GiB | 85.2 GiB | 256 s |
| nvfp4-restart | 84.0 GiB | 76.0 GiB | 894373 | 0.6468 | 894373 | 22 | 78.9 GiB | 85.2 GiB | 261 s |

## The committed 79 GiB file

- exit code n/a, request named: n/a
- message: n/a

## CapyCTL launch settings that differ between the CapyCTL variants

- `memory_saver`: bf16-deep `True`, bf16-restart `False`, nvfp4-restart `False`
- `weight_restore`: bf16-deep `resident`, bf16-restart `disk_reload`, nvfp4-restart `disk_reload`

## SGLang resolved server arguments that differ

- `<redacted>`: raw `None`, bf16-deep `False`, bf16-restart `False`, nvfp4-restart `False`
- `enable_cache_report`: raw `False`, bf16-deep `True`, bf16-restart `True`, nvfp4-restart `True`
- `enable_memory_saver`: raw `False`, bf16-deep `True`, bf16-restart `False`, nvfp4-restart `False`
- `kv_cache_dtype`: raw `auto`, bf16-deep `fp8_e4m3`, bf16-restart `fp8_e4m3`, nvfp4-restart `fp8_e4m3`
- `log_level_http`: raw `None`, bf16-deep `info`, bf16-restart `info`, nvfp4-restart `info`
- `max_running_requests`: raw `48`, bf16-deep `22`, bf16-restart `22`, nvfp4-restart `22`
- `max_total_tokens`: raw `None`, bf16-deep `894373`, bf16-restart `894373`, nvfp4-restart `894373`
- `mem_fraction_static`: raw `0.65`, bf16-deep `0.6502`, bf16-restart `0.6501`, nvfp4-restart `0.6468`
- `model_path`: raw `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead/snapshots/009632fef96dd349150baa780c984e62e70e91fe`, bf16-deep `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead/snapshots/009632fef96dd349150baa780c984e62e70e91fe`, bf16-restart `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead/snapshots/009632fef96dd349150baa780c984e62e70e91fe`, nvfp4-restart `~/models/sources/huggingface/nvidia--Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80`
- `quantization`: raw `None`, bf16-deep `None`, bf16-restart `None`, nvfp4-restart `modelopt`
- `random_seed`: raw `62002716`, bf16-deep `321828509`, bf16-restart `338548731`, nvfp4-restart `830940562`
- `remote_instance_weight_loader_start_seed_via_transfer_engine`: raw `False`, bf16-deep `None`, bf16-restart `None`, nvfp4-restart `None`
- `served_model_name`: raw `qwen3.8-27b`, bf16-deep `qwen38-e-bf16-deep`, bf16-restart `qwen38-e-bf16`, nvfp4-restart `qwen38-e-nvfp4`
- `skip_server_warmup`: raw `False`, bf16-deep `True`, bf16-restart `True`, nvfp4-restart `True`
- `speculative_draft_model_quantization`: raw `None`, bf16-deep `None`, bf16-restart `None`, nvfp4-restart `modelopt`
- `speculative_dspark_align_verify_tokens_to_graph_tier`: raw `False`, bf16-deep `None`, bf16-restart `None`, nvfp4-restart `None`
- `tokenizer_path`: raw `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead/snapshots/009632fef96dd349150baa780c984e62e70e91fe`, bf16-deep `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead/snapshots/009632fef96dd349150baa780c984e62e70e91fe`, bf16-restart `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead/snapshots/009632fef96dd349150baa780c984e62e70e91fe`, nvfp4-restart `~/models/sources/huggingface/nvidia--Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80`

## Long-prompt stall

| Variant | Longest gap while the cold prompt prefills, s | Before it, s | Long prompt tokens | Its time to first token, s |
|---|---|---|---|---|
| nvfp4-restart-t1 | 8.469 | 0.118 | 80292 | 64.58 |

## Accuracy: BF16 LM head against NVFP4 LM head

Both through CapyCTL with `residency: restart_only` and the same EAGLE MTP flags, greedy (temperature 0),
thinking as the model's default. The checkpoints may differ beyond the head (the BF16-head checkpoint also
carries calibrated FP8 KV scales; the NVFP4-head one has none and runs FP8 KV unscaled), so a difference is
the checkpoints', not the head's alone. SGLang's greedy decoding is not batch-invariant: the BF16 repeat row
is the same checkpoint against itself.

| Token agreement (max_tokens 256) | Prompts | Exact match | First divergence, tokens (p25 / median / p75) | First divergence, characters (median) | By category (exact) |
|---|---|---|---|---|---|
| BF16 head vs NVFP4 head | 200 | 7 (3.5%) | 3 / 16 / 25 | 69 | chat 0/40, coding 0/40, math 2/40, multilingual 0/40, tool-use 5/40 |
| BF16 head vs itself (repeat of the first prompts) | 50 | 9 (18.0%) | 28 / 66 / 92 | 286 | coding 3/40, math 6/10 |

| Check | BF16 head | NVFP4 head |
|---|---|---|
| GSM8K test subset, final number in `\boxed{}` (max_tokens 2048) | 234/250 (93.6%), 9 cut at max_tokens, 0 errors | 233/250 (93.2%), 13 cut at max_tokens, 0 errors |
| Tool calls: all three checks | 29/30 | 30/30 |
| Tool calls: arguments are a JSON object | 29/30 | 30/30 |
| Tool calls: the expected function | 30/30 | 30/30 |
| Tool calls: required arguments present | 29/30 | 30/30 |
