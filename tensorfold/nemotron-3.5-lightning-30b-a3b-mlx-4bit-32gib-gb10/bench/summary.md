# Nemotron 3.5 Lightning 30B-A3B MLX 4-bit on TensorFold 0.6.5, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.5.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.5 | `nemotron35-30b-tf` | 2026-10-07 | gpu=GB10, engine=tensorfold, engine_version=0.6.5, capyctl=7e50aa9, model=TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit@d9d758fb83953437f7263256b0d96157e2a348b8, drafter=MTP (in the checkpoint), managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 145 |
| 1k | 148 |
| 2k | 175 |
| 4k | 153 |
| 8k | 151 |
| 16k | 154 |
| 32k | 144 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 4,118 |
| 1k | 5,686 |
| 2k | 7,086 |
| 4k | 7,220 |
| 8k | 7,139 |
| 16k | 6,847 |
| 32k | 6,293 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.122 |
| 1k | 0.177 |
| 2k | 0.287 |
| 4k | 0.558 |
| 8k | 1.12 |
| 16k | 2.34 |
| 32k | 5.13 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 6.91 |
| 1k | 6.75 |
| 2k | 5.72 |
| 4k | 6.55 |
| 8k | 6.62 |
| 16k | 6.51 |
| 32k | 6.92 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1.01 |
| 1k | 1.05 |
| 2k | 1.02 |
| 4k | 1.40 |
| 8k | 1.98 |
| 16k | 3.19 |
| 32k | 6.02 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 24.0 |
| 1k | 23.9 |
| 2k | 23.9 |
| 4k | 23.8 |
| 8k | 23.8 |
| 16k | 23.7 |
| 32k | 23.6 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 589 |
| 1k | 623 |
| 2k | 731 |
| 4k | 628 |
| 8k | 592 |
| 16k | 634 |
| 32k | 590 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 154 |
| 2 | 147 |
| 3 | 151 |
| 4 | 154 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 158 |
| 2 | 157 |
| 3 | 156 |
| 4 | 156 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.073 |
| 2 | 1.64 |
| 3 | 3.62 |
| 4 | 5.05 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.076 |
| 2 | 3.76 |
| 3 | 7.14 |
| 4 | 10.6 |

### TPOT (ms)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 6.33 |
| 2 | 6.38 |
| 3 | 6.41 |
| 4 | 6.40 |

### Draft acceptance (%)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 65.0% |
| 2 | 64.0% |
| 3 | 62.9% |
| 4 | 63.2% |

### Peak memory (GiB)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 23.7 |
| 2 | 23.7 |
| 3 | 23.7 |
| 4 | 23.7 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
