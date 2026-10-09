# Qwen3.6-35B-A3B NVFP4 on vLLM 0.30.0, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `qwen36-35b-vllm` | 2026-10-07 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=0c4ccb4, model=nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 77.3 |
| 1k | 77.3 |
| 2k | 77.0 |
| 4k | 76.3 |
| 8k | 75.1 |
| 16k | 72.6 |
| 32k | 68.9 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 4,152 |
| 1k | 6,030 |
| 2k | 6,905 |
| 4k | 6,892 |
| 8k | 6,609 |
| 16k | 6,111 |
| 32k | 5,289 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.125 |
| 1k | 0.165 |
| 2k | 0.294 |
| 4k | 0.582 |
| 8k | 1.21 |
| 16k | 2.62 |
| 32k | 6.11 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 12.9 |
| 1k | 12.9 |
| 2k | 13.0 |
| 4k | 13.1 |
| 8k | 13.3 |
| 16k | 13.8 |
| 32k | 14.5 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 1.78 |
| 1k | 1.83 |
| 2k | 1.96 |
| 4k | 2.26 |
| 8k | 2.91 |
| 16k | 4.38 |
| 32k | 7.98 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 30.8 |
| 1k | 30.8 |
| 2k | 30.8 |
| 4k | 30.8 |
| 8k | 30.8 |
| 16k | 30.8 |
| 32k | 30.9 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 291 |
| 1k | 283 |
| 2k | 304 |
| 4k | 301 |
| 8k | 299 |
| 16k | 295 |
| 32k | 244 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 76.2 |
| 2 | 115 |
| 3 | 151 |
| 4 | 180 |
| 5 | 142 |
| 6 | 152 |
| 7 | 167 |
| 8 | 180 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 77.1 |
| 2 | 58.4 |
| 3 | 51.0 |
| 4 | 45.5 |
| 5 | 45.4 |
| 6 | 45.6 |
| 7 | 45.5 |
| 8 | 45.4 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.069 |
| 2 | 0.118 |
| 3 | 0.139 |
| 4 | 0.145 |
| 5 | 0.152 |
| 6 | 0.158 |
| 7 | 0.195 |
| 8 | 5.80 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.094 |
| 2 | 0.156 |
| 3 | 0.145 |
| 4 | 0.158 |
| 5 | 11.4 |
| 6 | 11.5 |
| 7 | 11.6 |
| 8 | 11.6 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 13.0 |
| 2 | 17.1 |
| 3 | 19.6 |
| 4 | 22.0 |
| 5 | 22.0 |
| 6 | 21.9 |
| 7 | 22.0 |
| 8 | 22.0 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 30.8 |
| 2 | 30.8 |
| 3 | 30.8 |
| 4 | 30.8 |
| 5 | 30.8 |
| 6 | 30.8 |
| 7 | 30.8 |
| 8 | 30.8 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
