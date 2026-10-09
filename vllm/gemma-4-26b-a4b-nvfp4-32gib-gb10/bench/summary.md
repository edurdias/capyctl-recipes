# Gemma 4 26B-A4B NVFP4 on vLLM 0.30.0, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `gemma4-26b-a4b-vllm` | 2026-10-07 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=7e50aa9, model=nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 30.3 |
| 1k | 30.0 |
| 2k | 29.9 |
| 4k | 29.8 |
| 8k | 29.3 |
| 16k | 28.8 |
| 32k | 28.0 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 4,349 |
| 1k | 5,970 |
| 2k | 6,978 |
| 4k | 6,821 |
| 8k | 6,064 |
| 16k | 4,606 |
| 32k | 2,699 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.116 |
| 1k | 0.167 |
| 2k | 0.289 |
| 4k | 0.596 |
| 8k | 1.32 |
| 16k | 3.50 |
| 32k | 12.0 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 33.0 |
| 1k | 33.3 |
| 2k | 33.4 |
| 4k | 33.6 |
| 8k | 34.1 |
| 16k | 34.7 |
| 32k | 35.8 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 4.33 |
| 1k | 4.41 |
| 2k | 4.55 |
| 4k | 4.87 |
| 8k | 5.66 |
| 16k | 7.92 |
| 32k | 16.5 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 28.6 |
| 1k | 28.6 |
| 2k | 28.6 |
| 4k | 28.6 |
| 8k | 28.6 |
| 16k | 28.6 |
| 32k | 28.6 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 151 |
| 1k | 154 |
| 2k | 162 |
| 4k | 153 |
| 8k | 157 |
| 16k | 143 |
| 32k | 137 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 30.1 |
| 2 | 63.3 |
| 3 | 79.0 |
| 4 | 105 |
| 5 | 70.3 |
| 6 | 86.4 |
| 7 | 92.3 |
| 8 | 106 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 30.2 |
| 2 | 31.8 |
| 3 | 26.5 |
| 4 | 26.4 |
| 5 | 26.6 |
| 6 | 26.5 |
| 7 | 26.5 |
| 8 | 26.6 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.078 |
| 2 | 0.083 |
| 3 | 0.119 |
| 4 | 0.139 |
| 5 | 0.140 |
| 6 | 0.137 |
| 7 | 0.166 |
| 8 | 9.60 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.079 |
| 2 | 0.131 |
| 3 | 0.139 |
| 4 | 0.146 |
| 5 | 19.6 |
| 6 | 19.5 |
| 7 | 19.6 |
| 8 | 19.5 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 33.1 |
| 2 | 31.4 |
| 3 | 37.8 |
| 4 | 37.9 |
| 5 | 37.6 |
| 6 | 37.7 |
| 7 | 37.7 |
| 8 | 37.6 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 28.6 |
| 2 | 28.6 |
| 3 | 28.6 |
| 4 | 28.6 |
| 5 | 28.6 |
| 6 | 28.6 |
| 7 | 28.6 |
| 8 | 28.6 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
