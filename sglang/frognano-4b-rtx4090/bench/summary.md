# FrogNano-4B-2609 BF16 on SGLang 0.5.21, one RTX 4090 Laptop GPU

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21, BF16.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21, BF16 | `frognano-4b-sglang` | 2026-10-05 | capyctl=0.1.2, engine=sglang-0.5.21, gpu=RTX 4090 Laptop GPU 16 GB, model=microsoft/FrogNano-4B-2609@b90468c1 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 62.3 |
| 1k | 61.9 |
| 2k | 61.4 |
| 4k | 60.7 |
| 8k | 60.2 |
| 16k | 58.7 |
| 32k | 55.6 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 5,838 |
| 1k | 7,440 |
| 2k | 8,260 |
| 4k | 8,362 |
| 8k | 8,077 |
| 16k | 7,686 |
| 32k | 6,889 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 0.089 |
| 1k | 0.133 |
| 2k | 0.243 |
| 4k | 0.483 |
| 8k | 0.990 |
| 16k | 2.08 |
| 32k | 4.69 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 16.1 |
| 1k | 16.1 |
| 2k | 16.3 |
| 4k | 16.5 |
| 8k | 16.6 |
| 16k | 17.0 |
| 32k | 18.0 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 2.13 |
| 1k | 2.19 |
| 2k | 2.31 |
| 4k | 2.58 |
| 8k | 3.10 |
| 16k | 4.25 |
| 32k | 6.98 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 13.4 |
| 1k | 13.4 |
| 2k | 13.4 |
| 4k | 13.4 |
| 8k | 13.4 |
| 16k | 13.4 |
| 32k | 13.4 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21, BF16 |
|---|---|
| 0.5k | 250 |
| 1k | 248 |
| 2k | 254 |
| 4k | 247 |
| 8k | 237 |
| 16k | 235 |
| 32k | 223 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 62.2 |
| 2 | 117 |
| 3 | 172 |
| 4 | 227 |
| 5 | 279 |
| 6 | 330 |
| 7 | 381 |
| 8 | 232 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 62.5 |
| 2 | 58.8 |
| 3 | 57.8 |
| 4 | 57.2 |
| 5 | 56.1 |
| 6 | 55.5 |
| 7 | 54.9 |
| 8 | 54.8 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 0.045 |
| 2 | 0.045 |
| 3 | 0.074 |
| 4 | 0.076 |
| 5 | 0.080 |
| 6 | 0.093 |
| 7 | 0.094 |
| 8 | 0.095 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 0.046 |
| 2 | 0.073 |
| 3 | 0.077 |
| 4 | 0.077 |
| 5 | 0.083 |
| 6 | 0.095 |
| 7 | 0.096 |
| 8 | 9.46 |

### TPOT (ms)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 16.0 |
| 2 | 17.0 |
| 3 | 17.3 |
| 4 | 17.5 |
| 5 | 17.8 |
| 6 | 18.0 |
| 7 | 18.2 |
| 8 | 18.2 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21, BF16 |
|---|---|
| 1 | 13.4 |
| 2 | 13.4 |
| 3 | 13.4 |
| 4 | 13.4 |
| 5 | 13.4 |
| 6 | 13.4 |
| 7 | 13.4 |
| 8 | 13.4 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
