# Nemotron 3.5 Lightning 30B-A3B NVFP4 on SGLang 0.5.21, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `nemotron35-30b-sglang` | 2026-10-07 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=7e50aa9, model=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4@bee7596271d1495f6992ae224aefde4410e816b8, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 70.1 |
| 1k | 69.9 |
| 2k | 69.7 |
| 4k | 69.5 |
| 8k | 69.1 |
| 16k | 68.6 |
| 32k | 67.3 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 3,004 |
| 1k | 4,691 |
| 2k | 5,425 |
| 4k | 5,347 |
| 8k | 5,568 |
| 16k | 5,308 |
| 32k | 5,105 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.166 |
| 1k | 0.213 |
| 2k | 0.377 |
| 4k | 0.756 |
| 8k | 1.43 |
| 16k | 3.01 |
| 32k | 6.32 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 14.3 |
| 1k | 14.3 |
| 2k | 14.3 |
| 4k | 14.4 |
| 8k | 14.5 |
| 16k | 14.6 |
| 32k | 14.9 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 1.99 |
| 1k | 2.04 |
| 2k | 2.21 |
| 4k | 2.60 |
| 8k | 3.28 |
| 16k | 4.88 |
| 32k | 8.22 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 31.0 |
| 1k | 31.0 |
| 2k | 31.0 |
| 4k | 31.6 |
| 8k | 32.8 |
| 16k | 33.0 |
| 32k | 33.0 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 294 |
| 1k | 299 |
| 2k | 293 |
| 4k | 298 |
| 8k | 276 |
| 16k | 258 |
| 32k | 271 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 68.9 |
| 2 | 109 |
| 3 | 123 |
| 4 | 163 |
| 5 | 128 |
| 6 | 140 |
| 7 | 144 |
| 8 | 164 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 69.7 |
| 2 | 55.3 |
| 3 | 41.6 |
| 4 | 41.3 |
| 5 | 41.5 |
| 6 | 41.4 |
| 7 | 41.5 |
| 8 | 41.4 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.086 |
| 2 | 0.143 |
| 3 | 0.176 |
| 4 | 0.185 |
| 5 | 0.192 |
| 6 | 0.196 |
| 7 | 0.206 |
| 8 | 6.41 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.090 |
| 2 | 0.168 |
| 3 | 0.182 |
| 4 | 0.198 |
| 5 | 12.6 |
| 6 | 12.7 |
| 7 | 12.7 |
| 8 | 12.7 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 14.3 |
| 2 | 18.1 |
| 3 | 24.1 |
| 4 | 24.2 |
| 5 | 24.1 |
| 6 | 24.1 |
| 7 | 24.1 |
| 8 | 24.2 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 33.0 |
| 2 | 33.0 |
| 3 | 32.9 |
| 4 | 33.0 |
| 5 | 33.0 |
| 6 | 33.0 |
| 7 | 33.0 |
| 8 | 33.0 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
