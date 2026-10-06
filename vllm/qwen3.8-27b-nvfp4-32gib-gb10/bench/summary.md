# Qwen3.8-27B NVFP4 on vLLM 0.30.0, within 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `qwen38-27b-vllm` | 2026-10-06 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=a6b2560, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80 resharded to 2 GiB files, drafter=none, managed_limit=32GiB |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 12.7 |
| 1k | 12.7 |
| 2k | 12.7 |
| 4k | 12.6 |
| 8k | 12.5 |
| 16k | 12.3 |
| 32k | 12.0 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 2,303 |
| 1k | 2,815 |
| 2k | 1,867 |
| 4k | 1,903 |
| 8k | 1,694 |
| 16k | 1,633 |
| 32k | 1,498 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.218 |
| 1k | 0.354 |
| 2k | 1.09 |
| 4k | 2.14 |
| 8k | 4.71 |
| 16k | 9.80 |
| 32k | 21.6 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 79.0 |
| 1k | 79.0 |
| 2k | 79.0 |
| 4k | 79.5 |
| 8k | 80.2 |
| 16k | 81.3 |
| 32k | 83.6 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 10.3 |
| 1k | 10.4 |
| 2k | 11.1 |
| 4k | 12.2 |
| 8k | 14.9 |
| 16k | 20.1 |
| 32k | 32.2 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 29.7 |
| 1k | 29.7 |
| 2k | 29.7 |
| 4k | 29.7 |
| 8k | 29.7 |
| 16k | 29.7 |
| 32k | 29.8 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 65.7 |
| 1k | 66.8 |
| 2k | 63.3 |
| 4k | 59.1 |
| 8k | 59.5 |
| 16k | 57.3 |
| 32k | 56.0 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 12.6 |
| 2 | 24.1 |
| 3 | 35.6 |
| 4 | 46.7 |
| 5 | 30.4 |
| 6 | 35.7 |
| 7 | 41.2 |
| 8 | 46.7 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 12.7 |
| 2 | 12.1 |
| 3 | 11.9 |
| 4 | 11.7 |
| 5 | 11.7 |
| 6 | 11.7 |
| 7 | 11.7 |
| 8 | 11.7 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.127 |
| 2 | 0.240 |
| 3 | 0.266 |
| 4 | 0.310 |
| 5 | 0.314 |
| 6 | 0.318 |
| 7 | 0.318 |
| 8 | 22.2 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.155 |
| 2 | 0.263 |
| 3 | 0.293 |
| 4 | 0.316 |
| 5 | 44.0 |
| 6 | 44.0 |
| 7 | 44.1 |
| 8 | 44.2 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 79.0 |
| 2 | 82.5 |
| 3 | 83.9 |
| 4 | 85.1 |
| 5 | 85.3 |
| 6 | 85.2 |
| 7 | 85.2 |
| 8 | 85.3 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 29.9 |
| 2 | 30.0 |
| 3 | 29.8 |
| 4 | 29.8 |
| 5 | 29.8 |
| 6 | 29.8 |
| 7 | 29.8 |
| 8 | 29.8 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
