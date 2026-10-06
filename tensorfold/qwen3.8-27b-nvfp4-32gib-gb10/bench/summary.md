# Qwen3.8-27B NVFP4 on TensorFold 0.6.5, DFlash2 drafts, within 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.5.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.5 | `qwen38-27b-tf` | 2026-10-06 | gpu=GB10, engine=tensorfold, engine_version=0.6.5, capyctl=a6b2560, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, managed_limit=32GiB |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 144 |
| 1k | 144 |
| 2k | 48.6 |
| 4k | 50.2 |
| 8k | 45.5 |
| 16k | 41.8 |
| 32k | 46.9 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1,805 |
| 1k | 2,335 |
| 2k | 2,536 |
| 4k | 2,714 |
| 8k | 2,752 |
| 16k | 2,634 |
| 32k | 2,387 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.272 |
| 1k | 0.427 |
| 2k | 0.799 |
| 4k | 1.49 |
| 8k | 2.94 |
| 16k | 6.08 |
| 32k | 13.5 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 6.96 |
| 1k | 6.96 |
| 2k | 20.6 |
| 4k | 19.9 |
| 8k | 22.0 |
| 16k | 23.9 |
| 32k | 21.3 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1.16 |
| 1k | 1.32 |
| 2k | 3.45 |
| 4k | 4.02 |
| 8k | 5.74 |
| 16k | 9.12 |
| 32k | 16.2 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 27.8 |
| 1k | 27.9 |
| 2k | 28.1 |
| 4k | 29.0 |
| 8k | 30.3 |
| 16k | 31.4 |
| 32k | 34.5 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 774 |
| 1k | 759 |
| 2k | 232 |
| 4k | 218 |
| 8k | 217 |
| 16k | 203 |
| 32k | 211 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 44.7 |
| 2 | 73.5 |
| 3 | 65.0 |
| 4 | 77.6 |
| 5 | 70.0 |
| 6 | 76.1 |
| 7 | 72.5 |
| 8 | 77.2 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 45.0 |
| 2 | 41.5 |
| 3 | 41.7 |
| 4 | 41.1 |
| 5 | 41.5 |
| 6 | 40.9 |
| 7 | 41.6 |
| 8 | 41.3 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.110 |
| 2 | 0.135 |
| 3 | 0.140 |
| 4 | 5.72 |
| 5 | 12.1 |
| 6 | 12.6 |
| 7 | 14.4 |
| 8 | 18.9 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.112 |
| 2 | 0.141 |
| 3 | 12.5 |
| 4 | 14.5 |
| 5 | 25.2 |
| 6 | 27.3 |
| 7 | 37.5 |
| 8 | 40.0 |

### TPOT (ms)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 22.2 |
| 2 | 24.1 |
| 3 | 24.0 |
| 4 | 24.3 |
| 5 | 24.1 |
| 6 | 24.5 |
| 7 | 24.1 |
| 8 | 24.2 |

### Draft acceptance (%)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 20.6% |
| 2 | 20.4% |
| 3 | 20.4% |
| 4 | 20.6% |
| 5 | 20.4% |
| 6 | 20.4% |
| 7 | 20.4% |
| 8 | 20.5% |

### Peak memory (GiB)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 35.5 |
| 2 | 35.6 |
| 3 | 35.5 |
| 4 | 35.4 |
| 5 | 35.3 |
| 6 | 35.3 |
| 7 | 35.3 |
| 8 | 35.3 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
