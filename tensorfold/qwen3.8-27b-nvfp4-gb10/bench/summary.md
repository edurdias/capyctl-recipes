# Qwen3.8-27B NVFP4 on TensorFold 0.6.5, DFlash2 drafts

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.5.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.5 | `qwen38-27b` | 2026-10-04 | gpu=GB10, capyctl=1f2cfc7, engine=tensorfold, engine_version=0.6.5, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, parallel=8 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 135 |
| 1k | 136 |
| 2k | 45.9 |
| 4k | 47.5 |
| 8k | 43.0 |
| 16k | 39.4 |
| 32k | 44.6 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1,784 |
| 1k | 2,291 |
| 2k | 2,552 |
| 4k | 2,767 |
| 8k | 2,785 |
| 16k | 2,694 |
| 32k | 2,431 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.276 |
| 1k | 0.433 |
| 2k | 0.794 |
| 4k | 1.46 |
| 8k | 2.89 |
| 16k | 5.96 |
| 32k | 13.3 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 7.40 |
| 1k | 7.38 |
| 2k | 21.8 |
| 4k | 21.1 |
| 8k | 23.2 |
| 16k | 25.4 |
| 32k | 22.4 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1.23 |
| 1k | 1.39 |
| 2k | 3.61 |
| 4k | 4.15 |
| 8k | 5.85 |
| 16k | 9.21 |
| 32k | 16.2 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 29.0 |
| 1k | 28.7 |
| 2k | 28.9 |
| 4k | 29.8 |
| 8k | 31.1 |
| 16k | 32.2 |
| 32k | 35.2 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 729 |
| 1k | 716 |
| 2k | 219 |
| 4k | 206 |
| 8k | 206 |
| 16k | 193 |
| 32k | 200 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 42.3 |
| 2 | 69.6 |
| 3 | 95.4 |
| 4 | 121 |
| 5 | 137 |
| 6 | 157 |
| 7 | 176 |
| 8 | 195 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 42.7 |
| 2 | 39.4 |
| 3 | 36.6 |
| 4 | 34.5 |
| 5 | 32.0 |
| 6 | 30.7 |
| 7 | 29.4 |
| 8 | 28.4 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.114 |
| 2 | 0.143 |
| 3 | 0.184 |
| 4 | 0.220 |
| 5 | 0.202 |
| 6 | 0.235 |
| 7 | 0.255 |
| 8 | 0.432 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.116 |
| 2 | 0.143 |
| 3 | 0.329 |
| 4 | 0.346 |
| 5 | 0.380 |
| 6 | 0.420 |
| 7 | 0.430 |
| 8 | 0.451 |

### TPOT (ms)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 23.4 |
| 2 | 25.4 |
| 3 | 27.3 |
| 4 | 29.0 |
| 5 | 31.2 |
| 6 | 32.6 |
| 7 | 34.0 |
| 8 | 35.2 |

### Draft acceptance (%)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 20.6% |
| 2 | 20.3% |
| 3 | 21.6% |
| 4 | 21.0% |
| 5 | 22.4% |
| 6 | 23.6% |
| 7 | 25.0% |
| 8 | 26.5% |

### Peak memory (GiB)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 36.3 |
| 2 | 36.3 |
| 3 | 36.4 |
| 4 | 36.4 |
| 5 | 36.4 |
| 6 | 36.4 |
| 7 | 36.5 |
| 8 | 36.5 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
