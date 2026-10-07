# Qwen3.6-35B-A3B NVFP4 on SGLang 0.5.21, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `qwen36-35b-sglang` | 2026-10-07 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=0c4ccb4, model=nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 83.9 |
| 1k | 83.7 |
| 2k | 83.2 |
| 4k | 82.3 |
| 8k | 80.7 |
| 16k | 77.6 |
| 32k | 73.9 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 3,722 |
| 1k | 5,081 |
| 2k | 5,852 |
| 4k | 6,063 |
| 8k | 5,917 |
| 16k | 5,611 |
| 32k | 4,910 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.138 |
| 1k | 0.195 |
| 2k | 0.343 |
| 4k | 0.666 |
| 8k | 1.35 |
| 16k | 2.85 |
| 32k | 6.59 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 11.9 |
| 1k | 11.9 |
| 2k | 12.0 |
| 4k | 12.2 |
| 8k | 12.4 |
| 16k | 12.9 |
| 32k | 13.5 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 1.67 |
| 1k | 1.73 |
| 2k | 1.88 |
| 4k | 2.23 |
| 8k | 2.94 |
| 16k | 4.50 |
| 32k | 8.32 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 35.3 |
| 1k | 35.3 |
| 2k | 35.3 |
| 4k | 35.3 |
| 8k | 35.3 |
| 16k | 35.3 |
| 32k | 35.3 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 334 |
| 1k | 332 |
| 2k | 346 |
| 4k | 348 |
| 8k | 318 |
| 16k | 287 |
| 32k | 265 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 82.6 |
| 2 | 124 |
| 3 | 142 |
| 4 | 186 |
| 5 | 149 |
| 6 | 160 |
| 7 | 164 |
| 8 | 187 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 83.5 |
| 2 | 63.1 |
| 3 | 47.9 |
| 4 | 47.0 |
| 5 | 47.2 |
| 6 | 47.3 |
| 7 | 47.5 |
| 8 | 47.2 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.067 |
| 2 | 0.104 |
| 3 | 0.149 |
| 4 | 0.172 |
| 5 | 0.174 |
| 6 | 0.172 |
| 7 | 0.201 |
| 8 | 5.62 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.069 |
| 2 | 0.128 |
| 3 | 0.153 |
| 4 | 0.175 |
| 5 | 11.1 |
| 6 | 11.1 |
| 7 | 11.2 |
| 8 | 11.3 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 12.0 |
| 2 | 15.8 |
| 3 | 20.9 |
| 4 | 21.3 |
| 5 | 21.2 |
| 6 | 21.1 |
| 7 | 21.0 |
| 8 | 21.2 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 35.3 |
| 2 | 35.3 |
| 3 | 35.3 |
| 4 | 35.3 |
| 5 | 35.3 |
| 6 | 35.3 |
| 7 | 35.3 |
| 8 | 35.3 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
