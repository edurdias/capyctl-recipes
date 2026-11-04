# Qwen3.8-27B CapyCTL NVFP4 on SGLang 0.5.22 with DFlash2

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.22, DFlash2.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.22, DFlash2 | `q38-rcp-sg` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=rcp-sg, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 61.1 |
| 1k | 44.8 |
| 2k | 44.6 |
| 4k | 35.1 |
| 8k | 37.9 |
| 16k | 37.4 |
| 32k | 33.6 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 1,893 |
| 1k | 2,828 |
| 2k | 2,566 |
| 4k | 2,434 |
| 8k | 1,686 |
| 16k | 1,629 |
| 32k | 1,491 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 0.263 |
| 1k | 0.354 |
| 2k | 0.783 |
| 4k | 1.67 |
| 8k | 4.73 |
| 16k | 9.88 |
| 32k | 21.7 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 16.4 |
| 1k | 22.3 |
| 2k | 22.4 |
| 4k | 28.5 |
| 8k | 26.4 |
| 16k | 26.8 |
| 32k | 29.8 |

### Total time (s, lower is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 2.35 |
| 1k | 3.26 |
| 2k | 3.65 |
| 4k | 5.31 |
| 8k | 8.17 |
| 16k | 13.2 |
| 32k | 25.5 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 61.4 |
| 1k | 61.4 |
| 2k | 61.4 |
| 4k | 61.8 |
| 8k | 62.8 |
| 16k | 62.8 |
| 32k | 62.9 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.22, DFlash2 |
|---|---|
| 0.5k | 324 |
| 1k | 200 |
| 2k | 210 |
| 4k | 153 |
| 8k | 166 |
| 16k | 169 |
| 32k | 144 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.22, DFlash2 |
|---|---|
| 1 | 35.9 |
| 2 | 64.7 |
| 3 | 78.2 |
| 4 | 106 |
| 5 | 94.7 |
| 6 | 110 |
| 7 | 149 |
| 8 | 166 |

### Per stream (tok/s)

| Streams | SGLang 0.5.22, DFlash2 |
|---|---|
| 1 | 36.3 |
| 2 | 32.9 |
| 3 | 28.5 |
| 4 | 29.7 |
| 5 | 20.2 |
| 6 | 19.9 |
| 7 | 24.2 |
| 8 | 23.3 |

### TTFT p50 (s)

| Streams | SGLang 0.5.22, DFlash2 |
|---|---|
| 1 | 0.188 |
| 2 | 0.187 |
| 3 | 0.193 |
| 4 | 0.199 |
| 5 | 0.212 |
| 6 | 0.241 |
| 7 | 0.222 |
| 8 | 0.236 |

### TTFT p95 (s)

| Streams | SGLang 0.5.22, DFlash2 |
|---|---|
| 1 | 0.189 |
| 2 | 0.231 |
| 3 | 0.198 |
| 4 | 0.203 |
| 5 | 0.251 |
| 6 | 0.265 |
| 7 | 0.233 |
| 8 | 0.280 |

### TPOT (ms)

| Streams | SGLang 0.5.22, DFlash2 |
|---|---|
| 1 | 27.5 |
| 2 | 30.4 |
| 3 | 35.1 |
| 4 | 33.7 |
| 5 | 49.6 |
| 6 | 50.2 |
| 7 | 41.3 |
| 8 | 43.0 |

### Peak memory (GiB)

| Streams | SGLang 0.5.22, DFlash2 |
|---|---|
| 1 | 62.9 |
| 2 | 62.9 |
| 3 | 62.9 |
| 4 | 62.8 |
| 5 | 62.8 |
| 6 | 64.1 |
| 7 | 63.7 |
| 8 | 64.3 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
