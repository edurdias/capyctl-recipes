# Qwen3.8-27B NVFP4, EAGLE MTP, SGLang 0.5.21, GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with MTP/EAGLE nvfp4-restart (SGLang 0.5.21).

| Series | Model | Measured | Meta |
|---|---|---|---|
| MTP/EAGLE nvfp4-restart (SGLang 0.5.21) | `qwen38-e-nvfp4` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=e6e5c92ae6b18fd4fdbe353709cc0c2795f93305, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=none (the checkpoint's MTP layer) |

## Concurrency

### Aggregate (tok/s)

| Streams | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|
| 1 | 24.7 |
| 8 | 157 |
| 22 | 337 |

### Per stream (tok/s)

| Streams | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|
| 1 | 24.9 |
| 8 | 20.9 |
| 22 | 17.2 |

### TTFT p50 (s)

| Streams | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|
| 1 | 0.207 |
| 8 | 0.285 |
| 22 | 0.567 |

### TTFT p95 (s)

| Streams | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|
| 1 | 0.213 |
| 8 | 0.390 |
| 22 | 0.589 |

### TPOT (ms)

| Streams | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|
| 1 | 40.1 |
| 8 | 47.8 |
| 22 | 58.1 |

### Peak memory (GiB)

| Streams | MTP/EAGLE nvfp4-restart (SGLang 0.5.21) |
|---|---|
| 1 | 82.6 |
| 8 | 83.9 |
| 22 | 87.3 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
