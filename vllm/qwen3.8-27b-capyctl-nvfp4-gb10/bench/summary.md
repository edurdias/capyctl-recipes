# Qwen3.8-27B CapyCTL NVFP4 on vLLM 0.32.0 with DFlash2

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with vLLM 0.32.0, DFlash2.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.32.0, DFlash2 | `q38-rcp-vl` | 2026-11-06 | capyctl=capyctl 0.1.10, variant=rcp-vl, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 80.4 |
| 1k | 41.4 |
| 2k | 41.3 |
| 4k | 37.7 |
| 8k | 42.1 |
| 16k | 34.0 |
| 32k | 37.1 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 1,252 |
| 1k | 2,059 |
| 2k | 2,583 |
| 4k | 2,837 |
| 8k | 2,900 |
| 16k | 2,724 |
| 32k | 2,381 |

### Time to first token (s, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 0.396 |
| 1k | 0.488 |
| 2k | 0.782 |
| 4k | 1.43 |
| 8k | 2.75 |
| 16k | 5.92 |
| 32k | 13.6 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 12.4 |
| 1k | 24.1 |
| 2k | 24.2 |
| 4k | 26.5 |
| 8k | 23.7 |
| 16k | 29.4 |
| 32k | 27.0 |

### Total time (s, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 1.98 |
| 1k | 3.55 |
| 2k | 3.86 |
| 4k | 4.80 |
| 8k | 5.76 |
| 16k | 9.53 |
| 32k | 17.0 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 51.3 |
| 1k | 51.3 |
| 2k | 51.3 |
| 4k | 51.3 |
| 8k | 51.3 |
| 16k | 51.3 |
| 32k | 51.3 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 418 |
| 1k | 192 |
| 2k | 202 |
| 4k | 181 |
| 8k | 174 |
| 16k | 168 |
| 32k | 171 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 33.5 |
| 2 | 61.7 |
| 3 | 82.8 |
| 4 | 101 |
| 5 | 93.4 |
| 6 | 110 |
| 7 | 144 |
| 8 | 177 |

### Per stream (tok/s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 33.9 |
| 2 | 31.9 |
| 3 | 30.5 |
| 4 | 29.9 |
| 5 | 20.7 |
| 6 | 20.0 |
| 7 | 23.4 |
| 8 | 26.0 |

### TTFT p50 (s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 0.209 |
| 2 | 0.291 |
| 3 | 0.369 |
| 4 | 0.424 |
| 5 | 0.489 |
| 6 | 0.509 |
| 7 | 0.498 |
| 8 | 0.528 |

### TTFT p95 (s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 0.210 |
| 2 | 0.341 |
| 3 | 0.399 |
| 4 | 0.429 |
| 5 | 0.509 |
| 6 | 0.513 |
| 7 | 0.506 |
| 8 | 0.534 |

### TPOT (ms)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 29.5 |
| 2 | 31.4 |
| 3 | 32.8 |
| 4 | 33.5 |
| 5 | 48.3 |
| 6 | 50.0 |
| 7 | 42.7 |
| 8 | 38.5 |

### Peak memory (GiB)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 51.3 |
| 2 | 51.3 |
| 3 | 51.3 |
| 4 | 51.3 |
| 5 | 51.3 |
| 6 | 51.3 |
| 7 | 51.4 |
| 8 | 51.4 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
