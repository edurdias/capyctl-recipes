# FrogNano-4B-2609 MLX 4-bit on TensorFold 0.6.3, one RTX 4090 Laptop GPU

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.3, MLX 4-bit.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.3, MLX 4-bit | `frognano-4b-tf` | 2026-10-03 | capyctl=main@c5ebc1a, engine=tensorfold-0.6.3, gpu=RTX 4090 Laptop GPU 16 GB, model=capyctl/FrogNano-4B-2609-MLX-4bit@f779b2f7 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 48.6 |
| 1k | 49.5 |
| 2k | 49.5 |
| 4k | 49.2 |
| 8k | 48.3 |
| 16k | 48.4 |
| 32k | 49.1 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 3,597 |
| 1k | 4,766 |
| 2k | 6,129 |
| 4k | 6,415 |
| 8k | 6,337 |
| 16k | 6,371 |
| 32k | 5,532 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 0.144 |
| 1k | 0.207 |
| 2k | 0.330 |
| 4k | 0.629 |
| 8k | 1.26 |
| 16k | 2.51 |
| 32k | 5.84 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 20.6 |
| 1k | 20.2 |
| 2k | 20.2 |
| 4k | 20.3 |
| 8k | 20.7 |
| 16k | 20.7 |
| 32k | 20.4 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 2.76 |
| 1k | 2.77 |
| 2k | 2.92 |
| 4k | 3.21 |
| 8k | 3.89 |
| 16k | 5.16 |
| 32k | 8.44 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 3.75 |
| 1k | 3.75 |
| 2k | 4.15 |
| 4k | 4.64 |
| 8k | 5.52 |
| 16k | 6.99 |
| 32k | 9.96 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 0.5k | 186 |
| 1k | 197 |
| 2k | 204 |
| 4k | 199 |
| 8k | 194 |
| 16k | 192 |
| 32k | 193 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 49.7 |
| 2 | 49.7 |
| 3 | 50.0 |
| 4 | 49.9 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 49.9 |
| 2 | 49.7 |
| 3 | 50.2 |
| 4 | 50.2 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 0.054 |
| 2 | 5.17 |
| 3 | 10.3 |
| 4 | 15.5 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 0.059 |
| 2 | 10.4 |
| 3 | 20.6 |
| 4 | 30.9 |

### TPOT (ms)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 20.0 |
| 2 | 20.1 |
| 3 | 19.9 |
| 4 | 19.9 |

### Peak memory (GiB)

| Streams | TensorFold 0.6.3, MLX 4-bit |
|---|---|
| 1 | 11.0 |
| 2 | 11.0 |
| 3 | 11.0 |
| 4 | 11.0 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
