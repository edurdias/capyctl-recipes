# Qwen3.8-27B NVFP4, DFlash2 + mixed chunks, SGLang 0.5.21, GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with DFlash2 mixed chunks, 12 running (SGLang 0.5.21).

| Series | Model | Measured | Meta |
|---|---|---|---|
| DFlash2 mixed chunks, 12 running (SGLang 0.5.21) | `qwen38-w-dm12` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=52e264cba3b3c9dacb728713be3beed1399294ca, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 48.0 |
| 32k | 31.6 |
| 128k | 25.2 |

### Prompt processing (tok/s, higher is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 2,350 |
| 32k | 1,558 |
| 128k | 991 |

### Time to first token (s, lower is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 0.861 |
| 32k | 20.7 |
| 128k | 130 |

### Time per output token (ms, lower is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 23.0 |
| 32k | 31.8 |
| 128k | 39.9 |

### Total time (s, lower is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 3.80 |
| 32k | 24.8 |
| 128k | 135 |

### Peak memory (GiB, lower is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 83.9 |
| 32k | 83.9 |
| 128k | 84.0 |

### Output bytes per second (B/s, higher is better)

| Context | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 2k | 248 |
| 32k | 135 |
| 128k | 115 |

## Concurrency

### Aggregate (tok/s)

| Streams | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 1 | 28.3 |
| 4 | 85.5 |
| 8 | 118 |
| 16 | 133 |
| 22 | 155 |

### Per stream (tok/s)

| Streams | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 1 | 28.6 |
| 4 | 23.1 |
| 8 | 16.9 |
| 16 | 13.9 |
| 22 | 15.5 |

### TTFT p50 (s)

| Streams | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 1 | 0.214 |
| 4 | 0.246 |
| 8 | 0.274 |
| 16 | 0.324 |
| 22 | 0.331 |

### TTFT p95 (s)

| Streams | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 1 | 0.223 |
| 4 | 0.290 |
| 8 | 0.282 |
| 16 | 38.3 |
| 22 | 41.2 |

### TPOT (ms)

| Streams | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 1 | 35.0 |
| 4 | 43.2 |
| 8 | 59.2 |
| 16 | 72.1 |
| 22 | 64.3 |

### Peak memory (GiB)

| Streams | DFlash2 mixed chunks, 12 running (SGLang 0.5.21) |
|---|---|
| 1 | 80.1 |
| 4 | 81.0 |
| 8 | 82.2 |
| 16 | 83.9 |
| 22 | 83.9 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
