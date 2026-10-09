# Qwen3.8-Flash-Next MLX 4-bit on TensorFold 0.6.5, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.5.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.5 | `qwen38-flash-next-tf` | 2026-10-09 | gpu=GB10, engine=tensorfold, engine_version=0.6.5, capyctl=eb214aad5e7a5e07e58d6c246f13f189cc844820, model=TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP@2b170fa6309d5d1ee380b35636075fac7945f286, drafter=built-in MTP, managed_limit=110GiB 11GiB |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 66.9 |
| 1k | 66.0 |
| 2k | 58.8 |
| 4k | 59.9 |
| 8k | 41.3 |
| 16k | 49.4 |
| 32k | 55.0 |
| 64k | 47.7 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 707 |
| 1k | 864 |
| 2k | 1,045 |
| 4k | 1,266 |
| 8k | 1,573 |
| 16k | 1,872 |
| 32k | 2,005 |
| 64k | 2,058 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.706 |
| 1k | 1.17 |
| 2k | 1.99 |
| 4k | 3.21 |
| 8k | 5.07 |
| 16k | 8.55 |
| 32k | 16.1 |
| 64k | 31.4 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 15.0 |
| 1k | 15.2 |
| 2k | 17.0 |
| 4k | 16.7 |
| 8k | 24.2 |
| 16k | 20.3 |
| 32k | 18.2 |
| 64k | 21.0 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 2.61 |
| 1k | 3.10 |
| 2k | 4.18 |
| 4k | 5.21 |
| 8k | 8.16 |
| 16k | 11.1 |
| 32k | 18.4 |
| 64k | 34.1 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 89.9 |
| 1k | 90.0 |
| 2k | 90.5 |
| 4k | 90.5 |
| 8k | 91.1 |
| 16k | 91.5 |
| 32k | 93.2 |
| 64k | 95.7 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 312 |
| 1k | 286 |
| 2k | 270 |
| 4k | 272 |
| 8k | 206 |
| 16k | 252 |
| 32k | 249 |
| 64k | 213 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 52.3 |
| 2 | 69.3 |
| 3 | 82.5 |
| 4 | 96.8 |
| 5 | 102 |
| 6 | 110 |
| 7 | 116 |
| 8 | 122 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 53.7 |
| 2 | 36.2 |
| 3 | 29.3 |
| 4 | 24.8 |
| 5 | 21.4 |
| 6 | 19.4 |
| 7 | 17.6 |
| 8 | 16.0 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.249 |
| 2 | 0.182 |
| 3 | 0.099 |
| 4 | 0.300 |
| 5 | 0.142 |
| 6 | 0.136 |
| 7 | 0.160 |
| 8 | 0.165 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.261 |
| 2 | 0.368 |
| 3 | 0.394 |
| 4 | 0.437 |
| 5 | 0.415 |
| 6 | 0.424 |
| 7 | 0.870 |
| 8 | 0.408 |

### TPOT (ms)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 18.6 |
| 2 | 27.6 |
| 3 | 34.2 |
| 4 | 40.3 |
| 5 | 46.7 |
| 6 | 51.5 |
| 7 | 56.8 |
| 8 | 62.3 |

### Draft acceptance (%)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 69.2% |
| 2 | 69.2% |
| 3 | 68.9% |
| 4 | 69.2% |
| 5 | 69.2% |
| 6 | 69.1% |
| 7 | 69.2% |
| 8 | 69.1% |

### Peak memory (GiB)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 99.8 |
| 2 | 99.7 |
| 3 | 98.5 |
| 4 | 99.2 |
| 5 | 98.7 |
| 6 | 100.0 |
| 7 | 99.0 |
| 8 | 99.7 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
