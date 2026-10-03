# FrogNano-4B-2609 BF16 on SGLang 0.5.21, one RTX 4090 Laptop GPU

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21, BF16.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21, BF16 | `frognano-4b-sglang` | 2026-10-03 | capyctl=main@c5ebc1a, engine=sglang-0.5.21, gpu=RTX 4090 Laptop GPU 16 GB, model=microsoft/FrogNano-4B-2609@b90468c1 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 61.0 |
| 1k | 60.6 |
| 2k | 59.9 |
| 4k | 60.0 |
| 8k | 59.0 |
| 16k | 57.5 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 5,885 |
| 1k | 7,341 |
| 2k | 7,730 |
| 4k | 7,979 |
| 8k | 7,790 |
| 16k | 7,427 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 0.088 |
| 1k | 0.136 |
| 2k | 0.260 |
| 4k | 0.506 |
| 8k | 1.03 |
| 16k | 2.16 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 16.4 |
| 1k | 16.5 |
| 2k | 16.7 |
| 4k | 16.7 |
| 8k | 16.9 |
| 16k | 17.4 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 2.17 |
| 1k | 2.23 |
| 2k | 2.38 |
| 4k | 2.63 |
| 8k | 3.20 |
| 16k | 4.37 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 13.4 |
| 1k | 13.4 |
| 2k | 13.4 |
| 4k | 13.4 |
| 8k | 13.4 |
| 16k | 13.4 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 245 |
| 1k | 241 |
| 2k | 244 |
| 4k | 242 |
| 8k | 232 |
| 16k | 228 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 60.8 |
| 2 | 114 |
| 3 | 169 |
| 4 | 223 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 61.1 |
| 2 | 57.6 |
| 3 | 56.6 |
| 4 | 56.1 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 0.045 |
| 2 | 0.075 |
| 3 | 0.077 |
| 4 | 0.080 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 0.045 |
| 2 | 0.336 |
| 3 | 0.242 |
| 4 | 0.236 |

### TPOT (ms)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 16.4 |
| 2 | 17.4 |
| 3 | 17.7 |
| 4 | 17.8 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 13.4 |
| 2 | 13.4 |
| 3 | 13.4 |
| 4 | 13.4 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
