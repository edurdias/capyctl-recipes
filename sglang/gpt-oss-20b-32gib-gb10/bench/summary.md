# gpt-oss-20b on SGLang 0.5.21, sized for 32 GiB

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with SGLang 0.5.21.

| Series | Model | Measured | Meta |
|---|---|---|---|
| SGLang 0.5.21 | `gpt-oss-20b-sglang` | 2026-10-07 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=7e50aa9, model=openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee, drafter=none, managed_limit=auto |

## Context sweep

### Generation (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 46.8 |
| 1k | 46.6 |
| 2k | 46.4 |
| 4k | 45.8 |
| 8k | 45.0 |
| 16k | 43.4 |
| 32k | 39.7 |

### Prompt processing (tok/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 3,066 |
| 1k | 4,952 |
| 2k | 7,012 |
| 4k | 8,178 |
| 8k | 8,410 |
| 16k | 7,716 |
| 32k | 6,177 |

### Time to first token (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 0.162 |
| 1k | 0.201 |
| 2k | 0.292 |
| 4k | 0.491 |
| 8k | 0.951 |
| 16k | 2.09 |
| 32k | 5.23 |

### Time per output token (ms, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 21.4 |
| 1k | 21.4 |
| 2k | 21.6 |
| 4k | 21.8 |
| 8k | 22.2 |
| 16k | 23.1 |
| 32k | 25.2 |

### Total time (s, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 2.90 |
| 1k | 2.94 |
| 2k | 3.05 |
| 4k | 3.27 |
| 8k | 3.79 |
| 16k | 5.03 |
| 32k | 8.47 |

### Peak memory (GiB, lower is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 31.0 |
| 1k | 31.0 |
| 2k | 31.0 |
| 4k | 31.0 |
| 8k | 31.0 |
| 16k | 31.0 |
| 32k | 31.2 |

### Output bytes per second (B/s, higher is better)

| Context | SGLang 0.5.21 |
|---|---|
| 0.5k | 229 |
| 1k | 227 |
| 2k | 229 |
| 4k | 224 |
| 8k | 218 |
| 16k | 210 |
| 32k | 192 |

## Concurrency

### Aggregate (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 45.3 |
| 2 | 83.2 |
| 3 | 91.0 |
| 4 | 125 |
| 5 | 126 |
| 6 | 146 |
| 7 | 161 |
| 8 | 191 |

### Per stream (tok/s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 45.8 |
| 2 | 42.0 |
| 3 | 30.5 |
| 4 | 31.6 |
| 5 | 25.5 |
| 6 | 24.4 |
| 7 | 23.1 |
| 8 | 24.0 |

### TTFT p50 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.137 |
| 2 | 0.123 |
| 3 | 0.137 |
| 4 | 0.128 |
| 5 | 0.140 |
| 6 | 0.146 |
| 7 | 0.163 |
| 8 | 0.164 |

### TTFT p95 (s)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 0.140 |
| 2 | 0.215 |
| 3 | 0.141 |
| 4 | 0.130 |
| 5 | 0.175 |
| 6 | 0.147 |
| 7 | 0.172 |
| 8 | 0.185 |

### TPOT (ms)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 21.8 |
| 2 | 23.8 |
| 3 | 32.7 |
| 4 | 31.7 |
| 5 | 39.2 |
| 6 | 41.0 |
| 7 | 43.2 |
| 8 | 41.7 |

### Peak memory (GiB)

| Streams | SGLang 0.5.21 |
|---|---|
| 1 | 31.1 |
| 2 | 31.0 |
| 3 | 31.0 |
| 4 | 31.1 |
| 5 | 31.1 |
| 6 | 31.0 |
| 7 | 31.1 |
| 8 | 31.1 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
