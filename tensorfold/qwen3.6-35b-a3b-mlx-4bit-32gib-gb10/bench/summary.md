# Qwen3.6-35B-A3B MLX 4-bit with MTP drafts on TensorFold 0.6.5, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with TensorFold 0.6.5.

| Series | Model | Measured | Meta |
|---|---|---|---|
| TensorFold 0.6.5 | `qwen36-35b-tf` | 2026-10-07 | gpu=GB10, engine=tensorfold, engine_version=0.6.5, capyctl=0c4ccb4, model=TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP@f84b054c5677b6e59bdb969ae61915b0983d68bb, drafter=MTP (in the checkpoint), managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 233 |
| 1k | 159 |
| 2k | 170 |
| 4k | 167 |
| 8k | 155 |
| 16k | 137 |
| 32k | 140 |

### Prompt processing (tok/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 3,520 |
| 1k | 4,971 |
| 2k | 6,292 |
| 4k | 7,042 |
| 8k | 7,028 |
| 16k | 6,732 |
| 32k | 5,963 |

### Time to first token (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.143 |
| 1k | 0.200 |
| 2k | 0.326 |
| 4k | 0.570 |
| 8k | 1.14 |
| 16k | 2.39 |
| 32k | 5.42 |

### Time per output token (ms, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 4.29 |
| 1k | 6.31 |
| 2k | 5.89 |
| 4k | 6.01 |
| 8k | 6.43 |
| 16k | 7.30 |
| 32k | 7.13 |

### Total time (s, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 0.701 |
| 1k | 1.01 |
| 2k | 1.09 |
| 4k | 1.34 |
| 8k | 1.96 |
| 16k | 3.34 |
| 32k | 6.38 |

### Peak memory (GiB, lower is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 26.6 |
| 1k | 26.6 |
| 2k | 27.1 |
| 4k | 27.4 |
| 8k | 27.8 |
| 16k | 28.5 |
| 32k | 29.8 |

### Output bytes per second (B/s, higher is better)

| Context | TensorFold 0.6.5 |
|---|---|
| 0.5k | 1,040 |
| 1k | 632 |
| 2k | 687 |
| 4k | 648 |
| 8k | 570 |
| 16k | 474 |
| 32k | 470 |

## Concurrency

### Aggregate (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 159 |
| 2 | 214 |
| 3 | 249 |
| 4 | 279 |
| 5 | 304 |
| 6 | 325 |
| 7 | 347 |
| 8 | 368 |

### Per stream (tok/s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 162 |
| 2 | 111 |
| 3 | 89.9 |
| 4 | 73.5 |
| 5 | 64.1 |
| 6 | 57.7 |
| 7 | 53.4 |
| 8 | 50.0 |

### TTFT p50 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.065 |
| 2 | 0.073 |
| 3 | 0.120 |
| 4 | 0.156 |
| 5 | 0.193 |
| 6 | 0.236 |
| 7 | 0.293 |
| 8 | 0.302 |

### TTFT p95 (s)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 0.068 |
| 2 | 0.148 |
| 3 | 0.232 |
| 4 | 0.321 |
| 5 | 0.374 |
| 6 | 0.435 |
| 7 | 0.541 |
| 8 | 0.631 |

### TPOT (ms)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 6.18 |
| 2 | 9.02 |
| 3 | 11.1 |
| 4 | 13.6 |
| 5 | 15.6 |
| 6 | 17.3 |
| 7 | 18.7 |
| 8 | 20.0 |

### Draft acceptance (%)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 61.4% |
| 2 | 62.3% |
| 3 | 61.9% |
| 4 | 61.8% |
| 5 | 62.0% |
| 6 | 61.9% |
| 7 | 62.0% |
| 8 | 62.1% |

### Peak memory (GiB)

| Streams | TensorFold 0.6.5 |
|---|---|
| 1 | 28.7 |
| 2 | 27.3 |
| 3 | 27.4 |
| 4 | 27.5 |
| 5 | 26.9 |
| 6 | 27.0 |
| 7 | 27.0 |
| 8 | 27.1 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
