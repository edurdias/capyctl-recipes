# Qwen3.8-27B NVFP4 on SGLang 0.5.21, within 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `qwen38-27b-sglang` | 2026-10-06 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=a6b2560, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80 resharded to 2 GiB files, drafter=none, managed_limit=32GiB |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 13.1 |
| 1k | 13.1 |
| 2k | 13.1 |
| 4k | 13.0 |
| 8k | 12.9 |
| 16k | 12.7 |
| 32k | 12.4 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 1,921 |
| 1k | 2,349 |
| 2k | 2,492 |
| 4k | 2,544 |
| 8k | 2,482 |
| 16k | 2,358 |
| 32k | 2,064 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.255 |
| 1k | 0.425 |
| 2k | 0.816 |
| 4k | 1.59 |
| 8k | 3.21 |
| 16k | 6.79 |
| 32k | 15.7 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 76.4 |
| 1k | 76.4 |
| 2k | 76.6 |
| 4k | 76.9 |
| 8k | 77.5 |
| 16k | 78.5 |
| 32k | 80.7 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 9.97 |
| 1k | 10.1 |
| 2k | 10.6 |
| 4k | 11.4 |
| 8k | 13.1 |
| 16k | 16.8 |
| 32k | 25.9 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 33.0 |
| 1k | 33.0 |
| 2k | 33.0 |
| 4k | 33.0 |
| 8k | 33.0 |
| 16k | 33.0 |
| 32k | 33.0 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 68.1 |
| 1k | 67.7 |
| 2k | 64.2 |
| 4k | 63.8 |
| 8k | 60.6 |
| 16k | 62.3 |
| 32k | 57.1 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 13.1 |
| 2 | 13.1 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 13.1 |
| 2 | 13.1 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.170 |
| 2 | 19.8 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.172 |
| 2 | 39.7 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 76.3 |
| 2 | 76.4 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 33.2 |
| 2 | 33.1 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
