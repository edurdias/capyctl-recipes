# Qwen3.8-27B: TensorFold 0.6.2 vs 0.6.3

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.2.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.2 | `qwen38-tf062` | 2026-10-02 | gpu=GB10, engine=tensorfold, engine_version=0.6.2, capyctl=74a9b9b, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, parallel=8, passes=2, interleaved 0.6.2, 0.6.3, 0.6.2, 0.6.3; each a fresh warm start, context_sweep_deployment=context_length 262144, otherwise identical |
| TensorFold 0.6.3 | `qwen38-tf063` | 2026-10-02 | gpu=GB10, engine=tensorfold, engine_version=0.6.3, capyctl=74a9b9b, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, parallel=8, passes=2, interleaved 0.6.2, 0.6.3, 0.6.2, 0.6.3; each a fresh warm start, context_sweep_deployment=context_length 262144, otherwise identical |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 46.4 | 46.5 (+0.2%) |
| 32k | 41.2 | 41.3 (+0.0%) |
| 128k | 30.3 | 30.1 (-0.7%) |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 2,534 | 2,550 (+0.6%) |
| 32k | 2,422 | 2,425 (+0.1%) |
| 128k | 1,479 | 1,478 (-0.0%) |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 0.797 | 0.797 (+0.1%) |
| 32k | 13.3 | 13.3 (-0.1%) |
| 128k | 87.1 | 87.0 (-0.0%) |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 21.5 | 21.5 (-0.2%) |
| 32k | 24.2 | 24.2 (-0.0%) |
| 128k | 33.0 | 33.2 (+0.7%) |

### Total time (s, lower is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 3.58 | 3.57 (-0.2%) |
| 32k | 16.4 | 16.4 (-0.0%) |
| 128k | 91.3 | 91.3 (+0.0%) |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 28.9 | 28.9 (-0.1%) |
| 32k | 33.3 | 33.3 (+0.0%) |
| 128k | 49.1 | 49.1 (-0.0%) |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 2k | 221 | 222 (+0.6%) |
| 32k | 193 | 193 (+0.0%) |
| 128k | 139 | 139 (-0.5%) |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 42.7 | 42.7 (-0.1%) |
| 4 | 123 | 123 (+0.1%) |
| 8 | 195 | 195 (-0.4%) |

### Per stream (tok/s)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 43.1 | 43.0 (-0.1%) |
| 4 | 34.9 | 35.3 (+1.2%) |
| 8 | 28.4 | 28.3 (-0.3%) |

### TTFT p50 (s)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 0.114 | 0.115 (+1.0%) |
| 4 | 0.166 | 0.160 (-4.1%) |
| 8 | 0.438 | 0.434 (-1.0%) |

### TTFT p95 (s)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 0.115 | 0.116 (+1.0%) |
| 4 | 0.385 | 0.385 (-0.0%) |
| 8 | 0.448 | 0.492 (+9.7%) |

### TPOT (ms)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 23.2 | 23.2 (+0.1%) |
| 4 | 28.7 | 28.3 (-1.2%) |
| 8 | 35.2 | 35.3 (+0.3%) |

### Draft acceptance (%)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 20.6% | 20.6% (+0.0%) |
| 4 | 22.0% | 22.2% (+0.9%) |
| 8 | 26.3% | 25.9% (-1.5%) |

### Peak memory (GiB)

| Streams | TensorFold 0.6.2 | TensorFold 0.6.3 |
|---|---|---|
| 1 | 27.1 | 27.1 (-0.0%) |
| 4 | 28.2 | 28.5 (+0.7%) |
| 8 | 29.6 | 29.6 (+0.0%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
