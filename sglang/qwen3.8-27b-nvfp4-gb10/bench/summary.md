# Qwen3.8-27B NVFP4 on SGLang 0.5.21, DFlash2 drafts

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `qwen38-27b-sglang` | 2026-10-04 | gpu=GB10, capyctl=1f2cfc7, engine=sglang, engine_version=0.5.21, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 61.9 |
| 1k | 31.9 |
| 2k | 30.9 |
| 4k | 34.7 |
| 8k | 29.6 |
| 16k | 33.1 |
| 32k | 24.9 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 1,677 |
| 1k | 2,135 |
| 2k | 2,302 |
| 4k | 2,216 |
| 8k | 1,784 |
| 16k | 1,728 |
| 32k | 1,585 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.295 |
| 1k | 0.468 |
| 2k | 0.872 |
| 4k | 1.80 |
| 8k | 4.48 |
| 16k | 9.31 |
| 32k | 20.4 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 16.2 |
| 1k | 31.4 |
| 2k | 32.3 |
| 4k | 28.8 |
| 8k | 33.7 |
| 16k | 30.2 |
| 32k | 40.2 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 2.36 |
| 1k | 4.46 |
| 2k | 5.05 |
| 4k | 5.48 |
| 8k | 8.87 |
| 16k | 13.1 |
| 32k | 25.5 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 71.8 |
| 1k | 71.8 |
| 2k | 71.8 |
| 4k | 72.2 |
| 8k | 73.5 |
| 16k | 73.5 |
| 32k | 73.4 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 328 |
| 1k | 148 |
| 2k | 146 |
| 4k | 160 |
| 8k | 142 |
| 16k | 151 |
| 32k | 110 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 32.0 |
| 2 | 47.4 |
| 3 | 65.5 |
| 4 | 81.6 |
| 5 | 80.7 |
| 6 | 91.8 |
| 7 | 109 |
| 8 | 120 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 32.4 |
| 2 | 26.3 |
| 3 | 23.4 |
| 4 | 22.5 |
| 5 | 17.1 |
| 6 | 16.2 |
| 7 | 17.4 |
| 8 | 17.2 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.210 |
| 2 | 0.211 |
| 3 | 0.220 |
| 4 | 0.229 |
| 5 | 0.236 |
| 6 | 0.248 |
| 7 | 0.272 |
| 8 | 0.266 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.213 |
| 2 | 0.231 |
| 3 | 0.221 |
| 4 | 0.233 |
| 5 | 0.251 |
| 6 | 0.274 |
| 7 | 0.274 |
| 8 | 0.322 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 30.9 |
| 2 | 38.1 |
| 3 | 42.7 |
| 4 | 44.5 |
| 5 | 58.6 |
| 6 | 61.6 |
| 7 | 57.4 |
| 8 | 58.1 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 73.2 |
| 2 | 73.2 |
| 3 | 74.4 |
| 4 | 74.2 |
| 5 | 75.6 |
| 6 | 75.8 |
| 7 | 76.7 |
| 8 | 77.9 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
