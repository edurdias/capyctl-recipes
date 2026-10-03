# FrogNano-4B-2609 MLX 4-bit on TensorFold 0.6.3, one RTX 4090 Laptop GPU

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.3, MLX 4-bit.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.3, MLX 4-bit | `frognano-4b-tf` | 2026-10-03 | capyctl=fix/tensorfold-parallel@35f1fbe, engine=tensorfold-0.6.3, gpu=RTX 4090 Laptop GPU 16 GB, model=capyctl/FrogNano-4B-2609-MLX-4bit@f779b2f7 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 49.5 |
| 1k | 49.5 |
| 2k | 50.4 |
| 4k | 49.8 |
| 8k | 49.6 |
| 16k | 49.5 |
| 32k | 49.9 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 3,839 |
| 1k | 5,231 |
| 2k | 6,764 |
| 4k | 7,216 |
| 8k | 7,269 |
| 16k | 7,178 |
| 32k | 6,273 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 0.131 |
| 1k | 0.189 |
| 2k | 0.297 |
| 4k | 0.553 |
| 8k | 1.10 |
| 16k | 2.23 |
| 32k | 5.15 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 20.2 |
| 1k | 20.2 |
| 2k | 19.9 |
| 4k | 20.1 |
| 8k | 20.2 |
| 16k | 20.2 |
| 32k | 20.0 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 2.70 |
| 1k | 2.76 |
| 2k | 2.82 |
| 4k | 3.12 |
| 8k | 3.67 |
| 16k | 4.81 |
| 32k | 7.71 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 3.52 |
| 1k | 3.54 |
| 2k | 3.82 |
| 4k | 4.03 |
| 8k | 4.60 |
| 16k | 5.17 |
| 32k | 6.71 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 191 |
| 1k | 199 |
| 2k | 205 |
| 4k | 200 |
| 8k | 200 |
| 16k | 196 |
| 32k | 197 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 50.4 |
| 2 | 93.4 |
| 3 | 135 |
| 4 | 176 |
| 5 | 212 |
| 6 | 251 |
| 7 | 284 |
| 8 | 319 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 50.6 |
| 2 | 46.9 |
| 3 | 45.1 |
| 4 | 44.3 |
| 5 | 42.6 |
| 6 | 42.2 |
| 7 | 40.9 |
| 8 | 40.3 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 0.053 |
| 2 | 0.061 |
| 3 | 0.063 |
| 4 | 0.066 |
| 5 | 0.078 |
| 6 | 0.095 |
| 7 | 0.099 |
| 8 | 0.153 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 0.053 |
| 2 | 0.091 |
| 3 | 0.072 |
| 4 | 0.134 |
| 5 | 0.141 |
| 6 | 0.136 |
| 7 | 0.146 |
| 8 | 0.159 |

### TPOT (ms)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 19.8 |
| 2 | 21.3 |
| 3 | 22.2 |
| 4 | 22.6 |
| 5 | 23.5 |
| 6 | 23.7 |
| 7 | 24.5 |
| 8 | 24.8 |

### Peak memory (GiB)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 7.24 |
| 2 | 7.24 |
| 3 | 7.25 |
| 4 | 7.25 |
| 5 | 7.25 |
| 6 | 7.25 |
| 7 | 7.25 |
| 8 | 7.26 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
