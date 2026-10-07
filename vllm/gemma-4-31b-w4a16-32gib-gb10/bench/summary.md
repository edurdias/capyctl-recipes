# Gemma 4 31B QAT W4A16 on vLLM 0.30.0, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `gemma4-31b-vllm` | 2026-10-07 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=7e50aa9, model=google/gemma-4-31B-it-qat-w4a16-ct@52f3f65bc7a02d555763bc923bd1d9094898219d, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 10.8 |
| 1k | 10.7 |
| 2k | 10.6 |
| 4k | 10.6 |
| 8k | 10.4 |
| 16k | 10.2 |
| 32k | 9.77 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 758 |
| 1k | 808 |
| 2k | 814 |
| 4k | 790 |
| 8k | 741 |
| 16k | 637 |
| 32k | 462 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.654 |
| 1k | 1.23 |
| 2k | 2.49 |
| 4k | 5.12 |
| 8k | 10.8 |
| 16k | 25.2 |
| 32k | 70.0 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 92.9 |
| 1k | 93.8 |
| 2k | 94.2 |
| 4k | 94.6 |
| 8k | 95.9 |
| 16k | 98.1 |
| 32k | 102 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 12.5 |
| 1k | 13.2 |
| 2k | 14.5 |
| 4k | 17.1 |
| 8k | 22.9 |
| 16k | 37.6 |
| 32k | 83.0 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 31.0 |
| 1k | 31.0 |
| 2k | 31.0 |
| 4k | 31.0 |
| 8k | 31.0 |
| 16k | 31.0 |
| 32k | 31.0 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 61.1 |
| 1k | 56.9 |
| 2k | 54.7 |
| 4k | 57.3 |
| 8k | 58.0 |
| 16k | 55.2 |
| 32k | 49.4 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 10.8 |
| 2 | 22.3 |
| 3 | 16.5 |
| 4 | 22.3 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 10.8 |
| 2 | 11.2 |
| 3 | 11.2 |
| 4 | 11.2 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.125 |
| 2 | 0.203 |
| 3 | 0.214 |
| 4 | 23.1 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.125 |
| 2 | 0.262 |
| 3 | 45.9 |
| 4 | 46.0 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 92.5 |
| 2 | 89.3 |
| 3 | 89.3 |
| 4 | 89.3 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 31.0 |
| 2 | 31.0 |
| 3 | 31.0 |
| 4 | 31.0 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
