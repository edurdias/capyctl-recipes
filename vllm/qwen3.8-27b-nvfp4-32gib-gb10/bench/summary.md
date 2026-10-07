# Qwen3.8-27B NVFP4 on vLLM 0.30.0, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `qwen38-27b-vllm` | 2026-10-06 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=0c4ccb4, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 12.5 |
| 1k | 12.5 |
| 2k | 12.4 |
| 4k | 12.3 |
| 8k | 12.2 |
| 16k | 12.0 |
| 32k | 11.7 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 2,324 |
| 1k | 2,776 |
| 2k | 2,705 |
| 4k | 2,771 |
| 8k | 2,620 |
| 16k | 2,460 |
| 32k | 2,156 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.214 |
| 1k | 0.358 |
| 2k | 0.753 |
| 4k | 1.46 |
| 8k | 3.05 |
| 16k | 6.51 |
| 32k | 15.0 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 80.3 |
| 1k | 80.2 |
| 2k | 80.8 |
| 4k | 81.1 |
| 8k | 81.8 |
| 16k | 83.2 |
| 32k | 85.3 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 10.4 |
| 1k | 10.6 |
| 2k | 11.0 |
| 4k | 11.8 |
| 8k | 13.5 |
| 16k | 17.1 |
| 32k | 25.8 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 30.7 |
| 1k | 30.7 |
| 2k | 31.1 |
| 4k | 32.1 |
| 8k | 32.9 |
| 16k | 33.8 |
| 32k | 35.4 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 65.4 |
| 1k | 63.3 |
| 2k | 60.4 |
| 4k | 54.8 |
| 8k | 58.6 |
| 16k | 49.5 |
| 32k | 52.2 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 12.4 |
| 2 | 24.0 |
| 3 | 35.3 |
| 4 | 46.2 |
| 5 | 29.9 |
| 6 | 35.4 |
| 7 | 40.9 |
| 8 | 46.2 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 12.4 |
| 2 | 12.0 |
| 3 | 11.8 |
| 4 | 11.6 |
| 5 | 11.6 |
| 6 | 11.6 |
| 7 | 11.6 |
| 8 | 11.6 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.129 |
| 2 | 0.241 |
| 3 | 0.265 |
| 4 | 0.316 |
| 5 | 0.318 |
| 6 | 0.326 |
| 7 | 0.340 |
| 8 | 22.4 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.156 |
| 2 | 0.266 |
| 3 | 0.295 |
| 4 | 0.343 |
| 5 | 44.5 |
| 6 | 44.6 |
| 7 | 44.5 |
| 8 | 44.7 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 80.7 |
| 2 | 83.0 |
| 3 | 84.5 |
| 4 | 86.1 |
| 5 | 86.2 |
| 6 | 86.0 |
| 7 | 86.0 |
| 8 | 86.1 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 38.0 |
| 2 | 34.0 |
| 3 | 32.8 |
| 4 | 36.2 |
| 5 | 35.4 |
| 6 | 34.6 |
| 7 | 33.2 |
| 8 | 33.6 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
