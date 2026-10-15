# Qwen3.8-27B on SGLang 0.5.21: direct SGLang, BF16 and NVFP4 heads, DFlash2 mix

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with MTP/EAGLE raw SGLang (SGLang 0.5.21).

| Series | Model | Measured | Meta |
|---|---|---|---|
| MTP/EAGLE raw SGLang (SGLang 0.5.21) | `qwen3.8-27b` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=none, model=RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@009632fef96dd349150baa780c984e62e70e91fe, drafter=none (the checkpoint's MTP layer) |
| MTP/EAGLE bf16-deep (SGLang 0.5.21) | `qwen38-e-bf16-deep` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=e6e5c92ae6b18fd4fdbe353709cc0c2795f93305, model=RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@009632fef96dd349150baa780c984e62e70e91fe, drafter=none (the checkpoint's MTP layer) |
| MTP/EAGLE bf16-restart (SGLang 0.5.21) | `qwen38-e-bf16` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=e6e5c92ae6b18fd4fdbe353709cc0c2795f93305, model=RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@009632fef96dd349150baa780c984e62e70e91fe, drafter=none (the checkpoint's MTP layer) |
| MTP/EAGLE nvfp4-restart (SGLang 0.5.21) | `qwen38-e-nvfp4` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=e6e5c92ae6b18fd4fdbe353709cc0c2795f93305, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=none (the checkpoint's MTP layer) |

## Concurrency

### Aggregate (tok/s)

| Streams | MTP/EAGLE raw SGLang (SGLang 0.5.21) | MTP/EAGLE bf16-deep (SGLang 0.5.21) | MTP/EAGLE bf16-restart (SGLang 0.5.21) | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|---|---|---|
| 1 | 17.5 | 17.3 (-0.8%) | 17.5 (+0.3%) | 24.7 (+41.1%) |
| 8 | 123 | 127 (+3.7%) | 120 (-2.0%) | 157 (+27.8%) |
| 22 | 279 | 292 (+4.6%) | 274 (-1.7%) | 337 (+20.8%) |

### Per stream (tok/s)

| Streams | MTP/EAGLE raw SGLang (SGLang 0.5.21) | MTP/EAGLE bf16-deep (SGLang 0.5.21) | MTP/EAGLE bf16-restart (SGLang 0.5.21) | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|---|---|---|
| 1 | 17.8 | 17.7 (-0.8%) | 17.7 (-0.9%) | 24.9 (+39.6%) |
| 8 | 16.5 | 18.0 (+9.1%) | 17.2 (+4.2%) | 20.9 (+26.8%) |
| 22 | 14.0 | 15.2 (+8.7%) | 14.3 (+2.4%) | 17.2 (+23.1%) |

### TTFT p50 (s)

| Streams | MTP/EAGLE raw SGLang (SGLang 0.5.21) | MTP/EAGLE bf16-deep (SGLang 0.5.21) | MTP/EAGLE bf16-restart (SGLang 0.5.21) | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|---|---|---|
| 1 | 0.362 | 0.348 (-3.8%) | 0.273 (-24.7%) | 0.207 (-42.7%) |
| 8 | 0.332 | 0.323 (-2.8%) | 0.326 (-1.9%) | 0.285 (-14.2%) |
| 22 | 0.521 | 0.509 (-2.3%) | 0.507 (-2.7%) | 0.567 (+8.9%) |

### TTFT p95 (s)

| Streams | MTP/EAGLE raw SGLang (SGLang 0.5.21) | MTP/EAGLE bf16-deep (SGLang 0.5.21) | MTP/EAGLE bf16-restart (SGLang 0.5.21) | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|---|---|---|
| 1 | 0.597 | 0.598 (+0.3%) | 0.276 (-53.8%) | 0.213 (-64.4%) |
| 8 | 0.375 | 0.502 (+33.7%) | 0.368 (-2.0%) | 0.390 (+3.8%) |
| 22 | 0.524 | 0.529 (+0.9%) | 0.532 (+1.5%) | 0.589 (+12.4%) |

### TPOT (ms)

| Streams | MTP/EAGLE raw SGLang (SGLang 0.5.21) | MTP/EAGLE bf16-deep (SGLang 0.5.21) | MTP/EAGLE bf16-restart (SGLang 0.5.21) | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|---|---|---|
| 1 | 56.1 | 56.5 (+0.8%) | 56.5 (+0.9%) | 40.1 (-28.4%) |
| 8 | 60.6 | 55.6 (-8.3%) | 58.2 (-4.0%) | 47.8 (-21.2%) |
| 22 | 71.5 | 65.8 (-8.0%) | 69.8 (-2.3%) | 58.1 (-18.8%) |

### Peak memory (GiB)

| Streams | MTP/EAGLE raw SGLang (SGLang 0.5.21) | MTP/EAGLE bf16-deep (SGLang 0.5.21) | MTP/EAGLE bf16-restart (SGLang 0.5.21) | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|---|---|---|
| 1 | 86.7 | 87.3 (+0.7%) | 83.7 (-3.5%) | 82.6 (-4.8%) |
| 8 | 88.2 | 89.5 (+1.4%) | 85.2 (-3.4%) | 83.9 (-4.9%) |
| 22 | 91.3 | 92.7 (+1.4%) | 88.4 (-3.2%) | 87.3 (-4.4%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
