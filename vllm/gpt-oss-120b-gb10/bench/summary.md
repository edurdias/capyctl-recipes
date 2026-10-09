# gpt-oss-120b on vLLM 0.30.0, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `gpt-oss-120b-vllm` | 2026-10-09 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=31dbf26, model=openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a, drafter=none, managed_limit=96GiB |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 35.3 |
| 1k | 35.1 |
| 2k | 34.9 |
| 4k | 34.5 |
| 8k | 33.8 |
| 16k | 32.5 |
| 32k | 30.1 |
| 64k | 26.1 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 1,505 |
| 1k | 2,554 |
| 2k | 3,609 |
| 4k | 3,942 |
| 8k | 4,061 |
| 16k | 3,942 |
| 32k | 3,573 |
| 64k | 2,935 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.331 |
| 1k | 0.390 |
| 2k | 0.567 |
| 4k | 1.02 |
| 8k | 1.97 |
| 16k | 4.06 |
| 32k | 9.06 |
| 64k | 22.0 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 28.3 |
| 1k | 28.5 |
| 2k | 28.7 |
| 4k | 29.0 |
| 8k | 29.6 |
| 16k | 30.8 |
| 32k | 33.2 |
| 64k | 38.3 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 3.94 |
| 1k | 4.02 |
| 2k | 4.22 |
| 4k | 4.71 |
| 8k | 5.74 |
| 16k | 7.98 |
| 32k | 13.3 |
| 64k | 26.9 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 84.1 |
| 1k | 84.1 |
| 2k | 84.1 |
| 4k | 84.1 |
| 8k | 84.1 |
| 16k | 84.1 |
| 32k | 84.1 |
| 64k | 84.1 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 177 |
| 1k | 173 |
| 2k | 172 |
| 4k | 172 |
| 8k | 167 |
| 16k | 162 |
| 32k | 146 |
| 64k | 131 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 34.1 |
| 2 | 58.4 |
| 3 | 71.8 |
| 4 | 81.1 |
| 5 | 89.6 |
| 6 | 96.6 |
| 7 | 102 |
| 8 | 107 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 34.6 |
| 2 | 29.6 |
| 3 | 24.2 |
| 4 | 20.5 |
| 5 | 18.1 |
| 6 | 16.3 |
| 7 | 14.6 |
| 8 | 13.5 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.237 |
| 2 | 0.251 |
| 3 | 0.258 |
| 4 | 0.290 |
| 5 | 0.324 |
| 6 | 0.354 |
| 7 | 0.385 |
| 8 | 0.373 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.241 |
| 2 | 0.372 |
| 3 | 0.319 |
| 4 | 0.377 |
| 5 | 0.374 |
| 6 | 0.386 |
| 7 | 0.446 |
| 8 | 0.444 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 28.9 |
| 2 | 33.7 |
| 3 | 41.4 |
| 4 | 48.8 |
| 5 | 55.3 |
| 6 | 61.5 |
| 7 | 68.3 |
| 8 | 74.0 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 84.1 |
| 2 | 84.1 |
| 3 | 84.1 |
| 4 | 84.1 |
| 5 | 84.1 |
| 6 | 84.1 |
| 7 | 84.1 |
| 8 | 84.1 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
