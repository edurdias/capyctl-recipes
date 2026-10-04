# Qwen3.8-27B NVFP4 on vLLM 0.30.0, DFlash2 drafts

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `qwen38-27b-vllm` | 2026-10-04 | gpu=GB10, capyctl=1f2cfc7, engine=vllm, engine_version=0.30.0, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 57.0 |
| 1k | 54.3 |
| 2k | 33.5 |
| 4k | 32.2 |
| 8k | 34.2 |
| 16k | 33.8 |
| 32k | 34.2 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 1,190 |
| 1k | 1,342 |
| 2k | 2,173 |
| 4k | 2,415 |
| 8k | 2,467 |
| 16k | 2,349 |
| 32k | 2,094 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.417 |
| 1k | 0.741 |
| 2k | 0.939 |
| 4k | 1.67 |
| 8k | 3.23 |
| 16k | 6.83 |
| 32k | 15.4 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 17.5 |
| 1k | 18.4 |
| 2k | 29.9 |
| 4k | 31.0 |
| 8k | 29.3 |
| 16k | 29.6 |
| 32k | 29.3 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 2.66 |
| 1k | 3.10 |
| 2k | 4.75 |
| 4k | 5.64 |
| 8k | 6.99 |
| 16k | 10.5 |
| 32k | 19.2 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 52.4 |
| 1k | 52.4 |
| 2k | 52.4 |
| 4k | 52.4 |
| 8k | 52.4 |
| 16k | 52.4 |
| 32k | 52.4 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 303 |
| 1k | 281 |
| 2k | 164 |
| 4k | 158 |
| 8k | 163 |
| 16k | 153 |
| 32k | 141 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 30.5 |
| 2 | 50.2 |
| 3 | 71.7 |
| 4 | 89.5 |
| 5 | 82.0 |
| 6 | 89.6 |
| 7 | 120 |
| 8 | 141 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 30.9 |
| 2 | 26.2 |
| 3 | 25.1 |
| 4 | 24.1 |
| 5 | 18.5 |
| 6 | 17.2 |
| 7 | 20.4 |
| 8 | 19.3 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.239 |
| 2 | 0.332 |
| 3 | 0.423 |
| 4 | 0.455 |
| 5 | 0.527 |
| 6 | 0.567 |
| 7 | 0.584 |
| 8 | 0.623 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.240 |
| 2 | 0.395 |
| 3 | 0.424 |
| 4 | 0.462 |
| 5 | 0.537 |
| 6 | 0.569 |
| 7 | 0.592 |
| 8 | 0.631 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 32.3 |
| 2 | 38.2 |
| 3 | 39.8 |
| 4 | 41.5 |
| 5 | 54.0 |
| 6 | 58.2 |
| 7 | 49.1 |
| 8 | 51.8 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 52.4 |
| 2 | 52.4 |
| 3 | 52.4 |
| 4 | 52.5 |
| 5 | 52.5 |
| 6 | 52.5 |
| 7 | 52.5 |
| 8 | 52.5 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
