# Nemotron 3.5 Super VL 120B-A12B NVFP4 on vLLM 0.31.0, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running | `nemotron-super-vl` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 43.3 |
| 1k | 41.5 |
| 2k | 38.6 |
| 4k | 40.3 |
| 8k | 40.3 |
| 16k | 39.1 |
| 32k | 37.9 |
| 64k | 38.6 |
| 100k | 40.3 |

### Prompt processing (tok/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 670 |
| 1k | 1,137 |
| 2k | 1,772 |
| 4k | 2,325 |
| 8k | 2,875 |
| 16k | 3,001 |
| 32k | 2,949 |
| 64k | 2,710 |
| 100k | 2,460 |

### Time to first token (s, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 0.748 |
| 1k | 0.877 |
| 2k | 1.15 |
| 4k | 1.72 |
| 8k | 2.77 |
| 16k | 5.32 |
| 32k | 10.9 |
| 64k | 23.8 |
| 100k | 41.0 |

### Time per output token (ms, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 23.1 |
| 1k | 24.1 |
| 2k | 25.9 |
| 4k | 24.8 |
| 8k | 24.8 |
| 16k | 25.6 |
| 32k | 26.4 |
| 64k | 25.9 |
| 100k | 24.8 |

### Total time (s, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 3.68 |
| 1k | 3.95 |
| 2k | 4.44 |
| 4k | 4.89 |
| 8k | 6.07 |
| 16k | 8.58 |
| 32k | 14.4 |
| 64k | 27.1 |
| 100k | 44.1 |

### Peak memory (GiB, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 93.3 |
| 1k | 93.3 |
| 2k | 93.3 |
| 4k | 94.0 |
| 8k | 95.2 |
| 16k | 96.5 |
| 32k | 96.5 |
| 64k | 96.6 |
| 100k | 96.6 |

### Output bytes per second (B/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 0.5k | 205 |
| 1k | 205 |
| 2k | 193 |
| 4k | 194 |
| 8k | 191 |
| 16k | 175 |
| 32k | 168 |
| 64k | 174 |
| 100k | 195 |

## Concurrency

### Aggregate (tok/s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 1 | 41.6 |
| 2 | 52.5 |
| 3 | 63.1 |
| 4 | 70.7 |

### Per stream (tok/s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 1 | 42.7 |
| 2 | 28.8 |
| 3 | 23.1 |
| 4 | 20.1 |

### TTFT p50 (s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 1 | 0.347 |
| 2 | 0.513 |
| 3 | 0.704 |
| 4 | 0.746 |

### TTFT p95 (s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 1 | 0.353 |
| 2 | 0.643 |
| 3 | 0.726 |
| 4 | 0.778 |

### TPOT (ms)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 1 | 23.4 |
| 2 | 34.7 |
| 3 | 43.2 |
| 4 | 49.7 |

### Peak memory (GiB)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|
| 1 | 96.6 |
| 2 | 96.6 |
| 3 | 96.6 |
| 4 | 96.6 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
