# Gemma 4 26B-A4B NVFP4 on SGLang 0.5.21, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `gemma4-26b-a4b-sglang` | 2026-10-07 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=7e50aa9, model=nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 29.7 |
| 1k | 29.5 |
| 2k | 29.5 |
| 4k | 29.3 |
| 8k | 29.0 |
| 16k | 28.4 |
| 32k | 27.5 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 3,668 |
| 1k | 4,378 |
| 2k | 4,289 |
| 4k | 3,373 |
| 8k | 2,332 |
| 16k | 2,346 |
| 32k | 2,112 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.134 |
| 1k | 0.226 |
| 2k | 0.486 |
| 4k | 1.20 |
| 8k | 3.43 |
| 16k | 6.83 |
| 32k | 15.3 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 33.6 |
| 1k | 33.9 |
| 2k | 33.9 |
| 4k | 34.2 |
| 8k | 34.4 |
| 16k | 35.2 |
| 32k | 36.4 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 4.42 |
| 1k | 4.55 |
| 2k | 4.81 |
| 4k | 5.54 |
| 8k | 7.82 |
| 16k | 11.3 |
| 32k | 19.9 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 32.0 |
| 1k | 32.0 |
| 2k | 32.0 |
| 4k | 32.7 |
| 8k | 33.5 |
| 16k | 34.2 |
| 32k | 34.2 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 163 |
| 1k | 147 |
| 2k | 159 |
| 4k | 154 |
| 8k | 152 |
| 16k | 150 |
| 32k | 139 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 29.8 |
| 2 | 62.0 |
| 3 | 76.8 |
| 4 | 103 |
| 5 | 68.9 |
| 6 | 84.1 |
| 7 | 90.0 |
| 8 | 103 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 29.9 |
| 2 | 31.2 |
| 3 | 25.7 |
| 4 | 25.7 |
| 5 | 26.1 |
| 6 | 25.7 |
| 7 | 25.8 |
| 8 | 25.8 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.084 |
| 2 | 0.097 |
| 3 | 0.090 |
| 4 | 0.092 |
| 5 | 0.094 |
| 6 | 0.098 |
| 7 | 0.099 |
| 8 | 10.0 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.085 |
| 2 | 1.46 |
| 3 | 0.093 |
| 4 | 0.093 |
| 5 | 20.1 |
| 6 | 20.1 |
| 7 | 20.1 |
| 8 | 20.3 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 33.4 |
| 2 | 32.0 |
| 3 | 39.0 |
| 4 | 38.9 |
| 5 | 38.4 |
| 6 | 38.9 |
| 7 | 38.8 |
| 8 | 38.8 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 34.2 |
| 2 | 34.2 |
| 3 | 34.2 |
| 4 | 34.2 |
| 5 | 34.2 |
| 6 | 34.2 |
| 7 | 34.2 |
| 8 | 34.2 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
