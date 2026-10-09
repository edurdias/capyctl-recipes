# gpt-oss-20b on vLLM 0.30.0, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with vLLM 0.30.0.

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM 0.30.0 | `gpt-oss-20b-vllm` | 2026-10-07 | gpu=GB10, engine=vllm, engine_version=0.30.0, capyctl=7e50aa9, model=openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 47.5 |
| 1k | 47.2 |
| 2k | 47.0 |
| 4k | 46.4 |
| 8k | 45.4 |
| 16k | 43.8 |
| 32k | 40.8 |

### Prompt processing (tok/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 3,104 |
| 1k | 4,720 |
| 2k | 6,299 |
| 4k | 7,035 |
| 8k | 7,361 |
| 16k | 7,113 |
| 32k | 6,278 |

### Time to first token (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 0.164 |
| 1k | 0.211 |
| 2k | 0.323 |
| 4k | 0.571 |
| 8k | 1.09 |
| 16k | 2.25 |
| 32k | 5.14 |

### Time per output token (ms, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 21.1 |
| 1k | 21.2 |
| 2k | 21.3 |
| 4k | 21.6 |
| 8k | 22.0 |
| 16k | 22.8 |
| 32k | 24.5 |

### Total time (s, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 2.85 |
| 1k | 2.92 |
| 2k | 3.04 |
| 4k | 3.32 |
| 8k | 3.90 |
| 16k | 5.17 |
| 32k | 8.27 |

### Peak memory (GiB, lower is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 26.9 |
| 1k | 26.9 |
| 2k | 26.9 |
| 4k | 26.9 |
| 8k | 26.9 |
| 16k | 26.9 |
| 32k | 26.9 |

### Output bytes per second (B/s, higher is better)

| Context | vLLM 0.30.0 |
|---|---|
| 0.5k | 229 |
| 1k | 232 |
| 2k | 231 |
| 4k | 231 |
| 8k | 221 |
| 16k | 215 |
| 32k | 200 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 45.9 |
| 2 | 84.5 |
| 3 | 107 |
| 4 | 126 |
| 5 | 145 |
| 6 | 163 |
| 7 | 179 |
| 8 | 195 |

### Per stream (tok/s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 46.4 |
| 2 | 42.7 |
| 3 | 36.1 |
| 4 | 31.8 |
| 5 | 29.3 |
| 6 | 27.3 |
| 7 | 25.7 |
| 8 | 24.6 |

### TTFT p50 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.134 |
| 2 | 0.136 |
| 3 | 0.165 |
| 4 | 0.179 |
| 5 | 0.190 |
| 6 | 0.188 |
| 7 | 0.205 |
| 8 | 0.199 |

### TTFT p95 (s)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 0.137 |
| 2 | 0.184 |
| 3 | 0.171 |
| 4 | 0.210 |
| 5 | 0.216 |
| 6 | 0.193 |
| 7 | 0.233 |
| 8 | 0.224 |

### TPOT (ms)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 21.5 |
| 2 | 23.4 |
| 3 | 27.7 |
| 4 | 31.4 |
| 5 | 34.1 |
| 6 | 36.6 |
| 7 | 38.9 |
| 8 | 40.6 |

### Peak memory (GiB)

| Streams | vLLM 0.30.0 |
|---|---|
| 1 | 26.9 |
| 2 | 26.9 |
| 3 | 26.9 |
| 4 | 26.9 |
| 5 | 26.9 |
| 6 | 26.9 |
| 7 | 26.9 |
| 8 | 26.9 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
