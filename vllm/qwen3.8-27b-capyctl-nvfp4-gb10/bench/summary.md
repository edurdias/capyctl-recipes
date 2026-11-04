# Qwen3.8-27B CapyCTL NVFP4 on vLLM 0.32.0 with DFlash2

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with vLLM 0.32.0, DFlash2.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.32.0, DFlash2 | `q38-rcp-vl` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=rcp-vl, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 39.5 |
| 1k | 47.5 |
| 2k | 43.0 |
| 4k | 38.0 |
| 8k | 40.0 |
| 16k | 41.9 |
| 32k | 33.6 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 1,250 |
| 1k | 2,059 |
| 2k | 2,591 |
| 4k | 2,857 |
| 8k | 2,916 |
| 16k | 2,729 |
| 32k | 2,389 |

### Time to first token (s, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 0.398 |
| 1k | 0.485 |
| 2k | 0.778 |
| 4k | 1.42 |
| 8k | 2.73 |
| 16k | 5.90 |
| 32k | 13.6 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 25.3 |
| 1k | 21.0 |
| 2k | 23.3 |
| 4k | 26.3 |
| 8k | 25.0 |
| 16k | 23.9 |
| 32k | 29.8 |

### Total time (s, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 3.62 |
| 1k | 3.16 |
| 2k | 3.73 |
| 4k | 4.76 |
| 8k | 5.90 |
| 16k | 8.83 |
| 32k | 17.3 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 51.7 |
| 1k | 51.7 |
| 2k | 51.7 |
| 4k | 51.7 |
| 8k | 51.7 |
| 16k | 51.7 |
| 32k | 51.7 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.32.0, DFlash2 |
|---|---|
| 0.5k | 191 |
| 1k | 218 |
| 2k | 206 |
| 4k | 179 |
| 8k | 189 |
| 16k | 199 |
| 32k | 157 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 34.9 |
| 2 | 58.5 |
| 3 | 85.2 |
| 4 | 102 |
| 5 | 96.5 |
| 6 | 111 |
| 7 | 162 |
| 8 | 187 |

### Per stream (tok/s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 35.3 |
| 2 | 31.6 |
| 3 | 32.3 |
| 4 | 29.4 |
| 5 | 20.9 |
| 6 | 20.1 |
| 7 | 25.1 |
| 8 | 25.3 |

### TTFT p50 (s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 0.207 |
| 2 | 0.290 |
| 3 | 0.368 |
| 4 | 0.426 |
| 5 | 0.486 |
| 6 | 0.505 |
| 7 | 0.494 |
| 8 | 0.524 |

### TTFT p95 (s)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 0.210 |
| 2 | 0.340 |
| 3 | 0.372 |
| 4 | 0.432 |
| 5 | 0.507 |
| 6 | 0.518 |
| 7 | 0.500 |
| 8 | 0.527 |

### TPOT (ms)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 28.3 |
| 2 | 31.6 |
| 3 | 31.0 |
| 4 | 34.0 |
| 5 | 47.9 |
| 6 | 49.9 |
| 7 | 39.9 |
| 8 | 39.5 |

### Peak memory (GiB)

| Streams | vLLM 0.32.0, DFlash2 |
|---|---|
| 1 | 51.7 |
| 2 | 51.7 |
| 3 | 51.7 |
| 4 | 51.7 |
| 5 | 51.8 |
| 6 | 51.8 |
| 7 | 51.8 |
| 8 | 51.8 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
