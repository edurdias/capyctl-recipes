# Qwen3-4B on vLLM 0.30.0

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `qwen3-4b-vllm` | 2026-10-04 | gpu=GB10, capyctl=1f2cfc7, engine=vllm, engine_version=0.30.0, model=Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c, thinking=off |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 22.0 |
| 1k | 21.8 |
| 2k | 21.4 |
| 4k | 20.8 |
| 8k | 19.7 |
| 16k | 17.9 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 5,869 |
| 1k | 7,522 |
| 2k | 7,683 |
| 4k | 7,497 |
| 8k | 6,797 |
| 16k | 5,568 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.088 |
| 1k | 0.131 |
| 2k | 0.264 |
| 4k | 0.538 |
| 8k | 1.17 |
| 16k | 2.87 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 45.5 |
| 1k | 46.0 |
| 2k | 46.7 |
| 4k | 48.0 |
| 8k | 50.7 |
| 16k | 55.9 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 5.89 |
| 1k | 5.99 |
| 2k | 6.21 |
| 4k | 6.65 |
| 8k | 7.63 |
| 16k | 9.98 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 19.5 |
| 1k | 19.3 |
| 2k | 19.1 |
| 4k | 18.9 |
| 8k | 18.8 |
| 16k | 18.8 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 115 |
| 1k | 106 |
| 2k | 115 |
| 4k | 112 |
| 8k | 102 |
| 16k | 91.0 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 22.0 |
| 2 | 53.7 |
| 3 | 80.1 |
| 4 | 106 |
| 5 | 131 |
| 6 | 156 |
| 7 | 180 |
| 8 | 205 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 22.0 |
| 2 | 26.9 |
| 3 | 26.8 |
| 4 | 26.6 |
| 5 | 26.3 |
| 6 | 26.1 |
| 7 | 25.9 |
| 8 | 25.7 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.055 |
| 2 | 0.077 |
| 3 | 0.100 |
| 4 | 0.099 |
| 5 | 0.094 |
| 6 | 0.133 |
| 7 | 0.140 |
| 8 | 0.140 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.063 |
| 2 | 0.098 |
| 3 | 0.145 |
| 4 | 0.130 |
| 5 | 0.132 |
| 6 | 0.141 |
| 7 | 0.144 |
| 8 | 0.144 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 45.5 |
| 2 | 37.1 |
| 3 | 37.3 |
| 4 | 37.6 |
| 5 | 38.0 |
| 6 | 38.2 |
| 7 | 38.6 |
| 8 | 38.9 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 18.8 |
| 2 | 18.8 |
| 3 | 18.8 |
| 4 | 18.8 |
| 5 | 18.8 |
| 6 | 18.8 |
| 7 | 18.8 |
| 8 | 18.8 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
