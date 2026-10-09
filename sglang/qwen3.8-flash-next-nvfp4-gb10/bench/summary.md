# Qwen3.8-Flash-Next NVFP4 on SGLang 0.5.21, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `qwen38-flash-next-sglang` | 2026-10-09 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=eb214aad5e7a5e07e58d6c246f13f189cc844820, model=RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594, drafter=built-in MTP (NEXTN), managed_limit=110GiB 8GiB |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 27.8 |
| 1k | 32.8 |
| 2k | 28.3 |
| 4k | 23.5 |
| 8k | 27.3 |
| 16k | 24.0 |
| 32k | 23.1 |
| 64k | 22.9 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 972 |
| 1k | 1,319 |
| 2k | 1,644 |
| 4k | 1,505 |
| 8k | 1,542 |
| 16k | 1,146 |
| 32k | 516 |
| 64k | 901 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.520 |
| 1k | 0.753 |
| 2k | 1.24 |
| 4k | 2.70 |
| 8k | 5.28 |
| 16k | 14.0 |
| 32k | 62.8 |
| 64k | 71.8 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 36.0 |
| 1k | 30.5 |
| 2k | 35.4 |
| 4k | 42.6 |
| 8k | 36.6 |
| 16k | 41.7 |
| 32k | 43.3 |
| 64k | 43.7 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 5.07 |
| 1k | 4.64 |
| 2k | 5.77 |
| 4k | 8.12 |
| 8k | 10.5 |
| 16k | 19.2 |
| 32k | 68.3 |
| 64k | 77.4 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 102 |
| 1k | 102 |
| 2k | 103 |
| 4k | 103 |
| 8k | 103 |
| 16k | 103 |
| 32k | 104 |
| 64k | 104 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 131 |
| 1k | 157 |
| 2k | 135 |
| 4k | 123 |
| 8k | 135 |
| 16k | 103 |
| 32k | 110 |
| 64k | 107 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 21.5 |
| 2 | 35.6 |
| 3 | 45.4 |
| 4 | 54.3 |
| 5 | 62.0 |
| 6 | 72.0 |
| 7 | 70.0 |
| 8 | 77.8 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 21.9 |
| 2 | 18.7 |
| 3 | 15.4 |
| 4 | 14.7 |
| 5 | 13.1 |
| 6 | 12.4 |
| 7 | 10.9 |
| 8 | 10.4 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.483 |
| 2 | 0.541 |
| 3 | 0.316 |
| 4 | 0.327 |
| 5 | 0.352 |
| 6 | 0.382 |
| 7 | 0.437 |
| 8 | 0.419 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.545 |
| 2 | 0.637 |
| 3 | 0.348 |
| 4 | 0.345 |
| 5 | 0.403 |
| 6 | 0.533 |
| 7 | 0.441 |
| 8 | 0.979 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 45.6 |
| 2 | 53.5 |
| 3 | 65.1 |
| 4 | 68.2 |
| 5 | 76.4 |
| 6 | 80.6 |
| 7 | 91.9 |
| 8 | 96.1 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 103 |
| 2 | 103 |
| 3 | 104 |
| 4 | 104 |
| 5 | 105 |
| 6 | 105 |
| 7 | 106 |
| 8 | 107 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
