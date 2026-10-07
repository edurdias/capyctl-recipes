# Nemotron 3.5 Lightning 30B-A3B NVFP4 on vLLM 0.30.0, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `nemotron35-30b-vllm` | 2026-10-07 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=7e50aa9, model=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4@bee7596271d1495f6992ae224aefde4410e816b8, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 70.2 |
| 1k | 70.1 |
| 2k | 70.0 |
| 4k | 69.8 |
| 8k | 69.6 |
| 16k | 69.1 |
| 32k | 68.4 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 3,768 |
| 1k | 5,284 |
| 2k | 6,383 |
| 4k | 6,622 |
| 8k | 6,393 |
| 16k | 6,113 |
| 32k | 5,721 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.134 |
| 1k | 0.189 |
| 2k | 0.315 |
| 4k | 0.610 |
| 8k | 1.25 |
| 16k | 2.62 |
| 32k | 5.66 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 14.2 |
| 1k | 14.3 |
| 2k | 14.3 |
| 4k | 14.3 |
| 8k | 14.4 |
| 16k | 14.5 |
| 32k | 14.6 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 1.96 |
| 1k | 2.01 |
| 2k | 2.14 |
| 4k | 2.44 |
| 8k | 3.09 |
| 16k | 4.47 |
| 32k | 7.52 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 29.1 |
| 1k | 29.0 |
| 2k | 28.9 |
| 4k | 28.9 |
| 8k | 28.9 |
| 16k | 28.9 |
| 32k | 29.0 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 299 |
| 1k | 313 |
| 2k | 306 |
| 4k | 283 |
| 8k | 286 |
| 16k | 285 |
| 32k | 274 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 69.2 |
| 2 | 112 |
| 3 | 124 |
| 4 | 164 |
| 5 | 129 |
| 6 | 142 |
| 7 | 144 |
| 8 | 164 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 70.0 |
| 2 | 56.8 |
| 3 | 42.0 |
| 4 | 41.5 |
| 5 | 41.4 |
| 6 | 41.4 |
| 7 | 41.6 |
| 8 | 41.4 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.082 |
| 2 | 0.151 |
| 3 | 0.172 |
| 4 | 0.180 |
| 5 | 0.189 |
| 6 | 0.181 |
| 7 | 0.213 |
| 8 | 6.40 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.103 |
| 2 | 0.172 |
| 3 | 0.200 |
| 4 | 0.206 |
| 5 | 12.6 |
| 6 | 12.7 |
| 7 | 12.7 |
| 8 | 12.7 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 14.3 |
| 2 | 17.6 |
| 3 | 23.8 |
| 4 | 24.1 |
| 5 | 24.1 |
| 6 | 24.2 |
| 7 | 24.0 |
| 8 | 24.1 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 28.9 |
| 2 | 28.9 |
| 3 | 28.9 |
| 4 | 29.0 |
| 5 | 28.9 |
| 6 | 28.9 |
| 7 | 28.9 |
| 8 | 28.9 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
