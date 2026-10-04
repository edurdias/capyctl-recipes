# Nemotron 3.5 Lightning 30B-A3B 4-bit on TensorFold 0.6.5

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.5.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.5 | `nemotron-30b` | 2026-10-04 | gpu=GB10, capyctl=1f2cfc7, engine=tensorfold, engine_version=0.6.5, model=Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit@d9d758fb83953437f7263256b0d96157e2a348b8, drafter=none (built-in MTP) |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 143 |
| 1k | 146 |
| 2k | 173 |
| 4k | 151 |
| 8k | 149 |
| 16k | 154 |
| 32k | 143 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 4,031 |
| 1k | 5,708 |
| 2k | 7,071 |
| 4k | 7,155 |
| 8k | 7,119 |
| 16k | 6,815 |
| 32k | 6,246 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.125 |
| 1k | 0.175 |
| 2k | 0.286 |
| 4k | 0.563 |
| 8k | 1.12 |
| 16k | 2.34 |
| 32k | 5.16 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 7.01 |
| 1k | 6.85 |
| 2k | 5.77 |
| 4k | 6.63 |
| 8k | 6.71 |
| 16k | 6.50 |
| 32k | 6.97 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1.03 |
| 1k | 1.06 |
| 2k | 1.03 |
| 4k | 1.42 |
| 8k | 2.00 |
| 16k | 3.20 |
| 32k | 6.06 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 25.5 |
| 1k | 25.4 |
| 2k | 25.3 |
| 4k | 25.2 |
| 8k | 25.1 |
| 16k | 24.9 |
| 32k | 24.7 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 581 |
| 1k | 615 |
| 2k | 724 |
| 4k | 619 |
| 8k | 584 |
| 16k | 629 |
| 32k | 586 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 153 |
| 2 | 145 |
| 3 | 148 |
| 4 | 153 |
| 5 | 148 |
| 6 | 150 |
| 7 | 149 |
| 8 | 148 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 156 |
| 2 | 156 |
| 3 | 154 |
| 4 | 155 |
| 5 | 155 |
| 6 | 155 |
| 7 | 155 |
| 8 | 153 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.071 |
| 2 | 1.65 |
| 3 | 3.45 |
| 4 | 5.28 |
| 5 | 7.07 |
| 6 | 8.79 |
| 7 | 10.4 |
| 8 | 12.2 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.072 |
| 2 | 3.50 |
| 3 | 7.50 |
| 4 | 10.7 |
| 5 | 14.2 |
| 6 | 17.3 |
| 7 | 20.7 |
| 8 | 24.4 |

### TPOT (ms)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 6.39 |
| 2 | 6.41 |
| 3 | 6.47 |
| 4 | 6.47 |
| 5 | 6.45 |
| 6 | 6.46 |
| 7 | 6.45 |
| 8 | 6.56 |

### Draft acceptance (%)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 65.0% |
| 2 | 64.3% |
| 3 | 63.0% |
| 4 | 63.3% |
| 5 | 63.1% |
| 6 | 63.2% |
| 7 | 63.3% |
| 8 | 62.9% |

### Peak memory (GiB)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 24.7 |
| 2 | 24.6 |
| 3 | 24.6 |
| 4 | 24.6 |
| 5 | 24.6 |
| 6 | 24.6 |
| 7 | 24.6 |
| 8 | 24.5 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
