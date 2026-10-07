# Qwen3.8-27B NVFP4 on SGLang 0.5.21, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `qwen38-27b-sglang` | 2026-10-06 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=0c4ccb4, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 12.9 |
| 1k | 12.9 |
| 2k | 12.9 |
| 4k | 12.8 |
| 8k | 12.7 |
| 16k | 12.6 |
| 32k | 12.2 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 1,941 |
| 1k | 2,338 |
| 2k | 2,434 |
| 4k | 2,514 |
| 8k | 2,457 |
| 16k | 2,347 |
| 32k | 2,053 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.258 |
| 1k | 0.426 |
| 2k | 0.830 |
| 4k | 1.60 |
| 8k | 3.25 |
| 16k | 6.82 |
| 32k | 15.7 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 77.6 |
| 1k | 77.6 |
| 2k | 77.8 |
| 4k | 78.1 |
| 8k | 78.6 |
| 16k | 79.6 |
| 32k | 81.8 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 10.1 |
| 1k | 10.3 |
| 2k | 10.7 |
| 4k | 11.5 |
| 8k | 13.2 |
| 16k | 16.9 |
| 32k | 26.1 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 36.7 |
| 1k | 37.1 |
| 2k | 37.6 |
| 4k | 38.0 |
| 8k | 38.3 |
| 16k | 38.9 |
| 32k | 39.8 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 67.1 |
| 1k | 66.7 |
| 2k | 63.3 |
| 4k | 62.8 |
| 8k | 59.7 |
| 16k | 61.5 |
| 32k | 56.1 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 12.9 |
| 2 | 12.9 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 12.9 |
| 2 | 12.9 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.172 |
| 2 | 20.0 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.176 |
| 2 | 40.1 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 77.6 |
| 2 | 77.5 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 34.9 |
| 2 | 36.4 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
