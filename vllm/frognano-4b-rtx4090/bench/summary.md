# FrogNano-4B-2609 BF16 on vLLM 0.30.0, one RTX 4090 Laptop GPU

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0, BF16.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0, BF16 | `frognano-4b-vllm` | 2026-10-03 | capyctl=main@c5ebc1a, engine=vllm-0.30.0, gpu=RTX 4090 Laptop GPU 16 GB, model=microsoft/FrogNano-4B-2609@b90468c1 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 60.3 |
| 1k | 59.8 |
| 2k | 58.7 |
| 4k | 56.4 |
| 8k | 58.5 |
| 16k | 56.9 |
| 32k | 53.9 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 5,769 |
| 1k | 6,162 |
| 2k | 7,463 |
| 4k | 7,425 |
| 8k | 7,490 |
| 16k | 7,223 |
| 32k | 6,458 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 0.090 |
| 1k | 0.160 |
| 2k | 0.272 |
| 4k | 0.548 |
| 8k | 1.07 |
| 16k | 2.22 |
| 32k | 5.00 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 16.6 |
| 1k | 16.7 |
| 2k | 17.0 |
| 4k | 17.7 |
| 8k | 17.1 |
| 16k | 17.6 |
| 32k | 18.6 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 2.20 |
| 1k | 2.29 |
| 2k | 2.44 |
| 4k | 2.81 |
| 8k | 3.24 |
| 16k | 4.44 |
| 32k | 7.35 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 13.2 |
| 1k | 13.2 |
| 2k | 13.2 |
| 4k | 13.2 |
| 8k | 13.2 |
| 16k | 13.2 |
| 32k | 13.2 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0, BF16 |
|---|---|
| 0.5k | 241 |
| 1k | 236 |
| 2k | 238 |
| 4k | 233 |
| 8k | 232 |
| 16k | 227 |
| 32k | 217 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0, BF16 |
|---|---|
| 1 | 59.9 |
| 2 | 111 |
| 3 | 163 |
| 4 | 215 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0, BF16 |
|---|---|
| 1 | 60.2 |
| 2 | 55.8 |
| 3 | 54.9 |
| 4 | 54.3 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0, BF16 |
|---|---|
| 1 | 0.056 |
| 2 | 0.098 |
| 3 | 0.113 |
| 4 | 0.114 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0, BF16 |
|---|---|
| 1 | 0.067 |
| 2 | 0.102 |
| 3 | 0.118 |
| 4 | 0.126 |

### TPOT (ms)

| Streams | vLLM 0.30.0, BF16 |
|---|---|
| 1 | 16.6 |
| 2 | 17.9 |
| 3 | 18.2 |
| 4 | 18.4 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0, BF16 |
|---|---|
| 1 | 13.2 |
| 2 | 13.2 |
| 3 | 13.2 |
| 4 | 13.2 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
