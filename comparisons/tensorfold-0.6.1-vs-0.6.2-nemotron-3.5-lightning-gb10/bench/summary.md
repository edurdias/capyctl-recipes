# Nemotron 3.5 Lightning 30B-A3B 4-bit: TensorFold 0.6.1 vs 0.6.2

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.1.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.1 | `nemotron-tf061` | 2026-10-02 | gpu=GB10, engine=tensorfold, capyctl=91bfca9, model=Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit@d9d758fb83953437f7263256b0d96157e2a348b8, drafter=none (built-in MTP), parallel=8, source=converted from the 2026-10-02 Nemotron comparison's raw data, engine_version=0.6.1 |
| TensorFold 0.6.2 | `nemotron-tf062` | 2026-10-02 | gpu=GB10, engine=tensorfold, capyctl=91bfca9, model=Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit@d9d758fb83953437f7263256b0d96157e2a348b8, drafter=none (built-in MTP), parallel=8, source=converted from the 2026-10-02 Nemotron comparison's raw data, engine_version=0.6.2 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 158 | 157 (-0.8%) |
| 1k | 157 | 155 (-0.8%) |
| 2k | 161 | 160 (-0.9%) |
| 4k | 165 | 164 (-0.3%) |
| 8k | 154 | 153 (-0.6%) |
| 16k | 138 | 141 (+2.4%) |
| 32k | 178 | 179 (+0.5%) |
| 64k | 140 | 140 (-0.2%) |
| 128k | 109 | 112 (+2.8%) |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 3,756 | 3,715 (-1.1%) |
| 1k | 5,471 | 5,449 (-0.4%) |
| 2k | 6,998 | 6,919 (-1.1%) |
| 4k | 7,141 | 7,063 (-1.1%) |
| 8k | 7,070 | 7,033 (-0.5%) |
| 16k | 6,825 | 6,742 (-1.2%) |
| 32k | 6,214 | 6,176 (-0.6%) |
| 64k | 5,277 | 5,259 (-0.3%) |
| 128k | 4,063 | 4,062 (-0.0%) |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 0.134 | 0.136 (+1.1%) |
| 1k | 0.186 | 0.187 (+0.4%) |
| 2k | 0.292 | 0.295 (+1.1%) |
| 4k | 0.573 | 0.579 (+1.1%) |
| 8k | 1.16 | 1.16 (+0.5%) |
| 16k | 2.40 | 2.43 (+1.2%) |
| 32k | 5.27 | 5.30 (+0.6%) |
| 64k | 12.4 | 12.5 (+0.3%) |
| 128k | 32.3 | 32.3 (+0.0%) |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 6.31 | 6.36 (+0.8%) |
| 1k | 6.38 | 6.43 (+0.8%) |
| 2k | 6.19 | 6.25 (+0.9%) |
| 4k | 6.08 | 6.09 (+0.3%) |
| 8k | 6.50 | 6.54 (+0.6%) |
| 16k | 7.24 | 7.07 (-2.4%) |
| 32k | 5.61 | 5.59 (-0.5%) |
| 64k | 7.15 | 7.17 (+0.2%) |
| 128k | 9.19 | 8.94 (-2.8%) |

### Total time (s, lower is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 0.951 | 0.958 (+0.7%) |
| 1k | 1.01 | 1.02 (+0.8%) |
| 2k | 1.09 | 1.10 (+0.9%) |
| 4k | 1.36 | 1.37 (+0.7%) |
| 8k | 1.99 | 2.01 (+0.9%) |
| 16k | 3.34 | 3.34 (-0.0%) |
| 32k | 6.00 | 6.03 (+0.5%) |
| 64k | 13.3 | 13.4 (+0.2%) |
| 128k | 33.4 | 33.4 (-0.1%) |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 28.8 | 28.8 (-0.0%) |
| 1k | 28.8 | 28.9 (+0.0%) |
| 2k | 28.8 | 28.9 (+0.0%) |
| 4k | 28.8 | 28.8 (+0.0%) |
| 8k | 28.8 | 28.8 (+0.2%) |
| 16k | 28.7 | 28.8 (+0.2%) |
| 32k | 28.7 | 28.7 (-0.1%) |
| 64k | 28.8 | 28.8 (-0.1%) |
| 128k | 28.8 | 28.8 (+0.0%) |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 0.5k | 579 | 574 (-0.8%) |
| 1k | 607 | 602 (-0.8%) |
| 2k | 610 | 606 (-0.7%) |
| 4k | 600 | 598 (-0.3%) |
| 8k | 599 | 597 (-0.4%) |
| 16k | 618 | 615 (-0.5%) |
| 32k | 770 | 772 (+0.3%) |
| 64k | 589 | 588 (-0.2%) |
| 128k | 444 | 437 (-1.6%) |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 1 | 143 | 142 (-0.8%) |

### Per stream (tok/s)

| Streams | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 1 | 145 | 144 (-0.7%) |

### TTFT p50 (s)

| Streams | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 1 | 0.025 | 0.031 (+24.5%) |

### TTFT p95 (s)

| Streams | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 1 | 0.040 | 0.043 (+7.0%) |

### TPOT (ms)

| Streams | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 1 | 6.91 | 6.96 (+0.7%) |

### Draft acceptance (%)

| Streams | TensorFold 0.6.1 | TensorFold 0.6.2 |
|---|---|---|
| 1 | 57.6% | 57.6% (+0.0%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
