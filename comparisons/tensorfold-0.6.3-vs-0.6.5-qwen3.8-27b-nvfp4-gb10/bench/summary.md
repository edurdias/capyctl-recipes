# Qwen3.8-27B: TensorFold 0.6.3 vs 0.6.5

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.3.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.3 | `qwen38-tf063` | 2026-10-04 | gpu=GB10, engine=tensorfold, engine_version=0.6.3, capyctl=9b90a56, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, parallel=8, passes=2, interleaved 0.6.3, 0.6.5, 0.6.3, 0.6.5; each a fresh warm start, context_sweep_deployment=context_length 262144, otherwise identical |
| TensorFold 0.6.5 | `qwen38-tf065` | 2026-10-04 | gpu=GB10, engine=tensorfold, engine_version=0.6.5, capyctl=9b90a56, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, parallel=8, passes=2, interleaved 0.6.3, 0.6.5, 0.6.3, 0.6.5; each a fresh warm start, context_sweep_deployment=context_length 262144, otherwise identical |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 45.8 | 45.9 (+0.0%) |
| 32k | 40.8 | 44.5 (+9.2%) |
| 128k | 30.0 | 29.6 (-1.3%) |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 2,544 | 2,550 (+0.3%) |
| 32k | 2,427 | 2,436 (+0.4%) |
| 128k | 1,476 | 1,481 (+0.3%) |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 0.793 | 0.791 (-0.3%) |
| 32k | 13.3 | 13.3 (-0.4%) |
| 128k | 87.2 | 86.9 (-0.4%) |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 21.8 | 21.8 (-0.0%) |
| 32k | 24.5 | 22.5 (-8.4%) |
| 128k | 33.3 | 33.8 (+1.3%) |

### Total time (s, lower is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 3.61 | 3.60 (-0.2%) |
| 32k | 16.4 | 16.1 (-2.0%) |
| 128k | 91.5 | 91.2 (-0.4%) |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 28.4 | 29.4 (+3.8%) |
| 32k | 32.8 | 33.9 (+3.3%) |
| 128k | 48.6 | 49.7 (+2.2%) |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 2k | 218 | 219 (+0.1%) |
| 32k | 191 | 200 (+4.6%) |
| 128k | 138 | 135 (-1.7%) |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 42.2 | 41.5 (-1.7%) |
| 4 | 121 | 121 (+0.1%) |
| 8 | 194 | 194 (-0.0%) |

### Per stream (tok/s)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 42.6 | 41.8 (-1.7%) |
| 4 | 34.7 | 34.9 (+0.8%) |
| 8 | 28.3 | 28.3 (-0.2%) |

### TTFT p50 (s)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 0.115 | 0.116 (+0.6%) |
| 4 | 0.210 | 0.214 (+1.9%) |
| 8 | 0.349 | 0.446 (+27.7%) |

### TTFT p95 (s)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 0.117 | 0.117 (+0.1%) |
| 4 | 0.388 | 0.386 (-0.7%) |
| 8 | 0.464 | 0.464 (+0.1%) |

### TPOT (ms)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 23.5 | 23.9 (+1.8%) |
| 4 | 28.9 | 28.6 (-0.8%) |
| 8 | 35.3 | 35.4 (+0.2%) |

### Draft acceptance (%)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 20.6% | 20.7% (+0.1%) |
| 4 | 21.6% | 22.2% (+2.9%) |
| 8 | 26.3% | 26.5% (+0.8%) |

### Peak memory (GiB)

| Streams | TensorFold 0.6.3 | TensorFold 0.6.5 |
|---|---|---|
| 1 | 27.0 | 27.2 (+0.9%) |
| 4 | 28.1 | 28.1 (+0.1%) |
| 8 | 29.5 | 29.5 (-0.0%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
