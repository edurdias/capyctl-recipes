# Bolt 24GB for several users on llama.cpp v0.7.0, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Bolt 24GB, 8 slots x 32k, no drafter.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Bolt 24GB, 8 slots x 32k, no drafter | `bolt-8x32k-nodraft` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, 4 slots x 64k, no drafter | `nemotron-bolt-4x64k-nodraft` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, 4 slots x 64k, LiLiCorr 7 | `nemotron-bolt-4x64k` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 37.7 | 37.8 (+0.3%) | 63.3 (+67.7%) |
| 8k | – | – | 66.1 |

### Prompt processing (tok/s, higher is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 1,098 | 1,070 (-2.6%) | 923 (-15.9%) |
| 8k | – | – | 1,141 |

### Time to first token (s, lower is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 0.908 | 0.932 (+2.6%) | 1.08 (+19.0%) |
| 8k | – | – | 7.07 |

### Time per output token (ms, lower is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 26.5 | 26.4 (-0.3%) | 15.8 (-40.3%) |
| 8k | – | – | 15.2 |

### Total time (s, lower is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 4.29 | 4.30 (+0.4%) | 3.10 (-27.6%) |
| 8k | – | – | 9.01 |

### Peak memory (GiB, lower is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 35.7 | 33.1 (-7.3%) | 40.8 (+14.2%) |
| 8k | – | – | 41.4 |

### Output bytes per second (B/s, higher is better)

| Context | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1k | 192 | 198 (+3.4%) | 311 (+62.4%) |
| 8k | – | – | 335 |

## Concurrency

### Aggregate (tok/s)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | 37.1 | 37.6 (+1.4%) | 71.5 (+92.9%) |
| 2 | 56.0 | 57.5 (+2.6%) | 64.3 (+14.8%) |
| 3 | 67.7 | 68.2 (+0.8%) | 68.2 (+0.7%) |
| 4 | 75.8 | 75.5 (-0.3%) | 72.3 (-4.6%) |
| 5 | 82.2 | – | – |
| 6 | 87.8 | – | – |
| 7 | 92.0 | – | – |
| 8 | 97.2 | – | – |

### Per stream (tok/s)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | 37.8 | 38.3 (+1.4%) | 74.9 (+98.2%) |
| 2 | 29.0 | 29.6 (+2.1%) | 34.3 (+18.4%) |
| 3 | 23.5 | 23.5 (+0.2%) | 25.8 (+10.0%) |
| 4 | 19.8 | 19.7 (-0.1%) | 21.0 (+6.2%) |
| 5 | 17.1 | – | – |
| 6 | 15.1 | – | – |
| 7 | 13.5 | – | – |
| 8 | 12.4 | – | – |

### TTFT p50 (s)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | 0.304 | 0.280 (-7.8%) | 0.332 (+9.4%) |
| 2 | 0.654 | 0.531 (-18.8%) | 0.582 (-11.0%) |
| 3 | 0.842 | 0.804 (-4.5%) | 0.952 (+13.1%) |
| 4 | 1.09 | 1.14 (+4.8%) | 1.27 (+16.9%) |
| 5 | 0.980 | – | – |
| 6 | 1.28 | – | – |
| 7 | 1.08 | – | – |
| 8 | 0.889 | – | – |

### TTFT p95 (s)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | 0.309 | 0.294 (-4.6%) | 0.337 (+9.1%) |
| 2 | 1.52 | 0.615 (-59.4%) | 0.737 (-51.4%) |
| 3 | 0.992 | 0.938 (-5.5%) | 1.05 (+6.0%) |
| 4 | 1.27 | 1.22 (-3.9%) | 1.40 (+10.3%) |
| 5 | 1.32 | – | – |
| 6 | 1.57 | – | – |
| 7 | 1.65 | – | – |
| 8 | 0.894 | – | – |

### TPOT (ms)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | 26.5 | 26.1 (-1.4%) | 13.3 (-49.6%) |
| 2 | 34.5 | 33.8 (-2.0%) | 29.2 (-15.5%) |
| 3 | 42.6 | 42.5 (-0.2%) | 38.8 (-9.1%) |
| 4 | 50.6 | 50.7 (+0.1%) | 47.7 (-5.9%) |
| 5 | 58.5 | – | – |
| 6 | 66.0 | – | – |
| 7 | 74.0 | – | – |
| 8 | 80.8 | – | – |

### Draft acceptance (%)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | – | – | 48.6% |
| 2 | – | – | 43.5% |
| 3 | – | – | 42.7% |
| 4 | – | – | 43.7% |
| 5 | – | – | – |
| 6 | – | – | – |
| 7 | – | – | – |
| 8 | – | – | – |

### Peak memory (GiB)

| Streams | Bolt 24GB, 8 slots x 32k, no drafter | Bolt 24GB, 4 slots x 64k, no drafter | Bolt 24GB, 4 slots x 64k, LiLiCorr 7 |
|---|---|---|---|
| 1 | 35.3 | 33.1 (-6.2%) | 41.6 (+18.0%) |
| 2 | 36.9 | 33.1 (-10.3%) | 41.8 (+13.2%) |
| 3 | 36.7 | 33.0 (-10.1%) | 41.7 (+13.5%) |
| 4 | 36.0 | 33.0 (-8.4%) | 41.5 (+15.3%) |
| 5 | 36.0 | – | – |
| 6 | 35.8 | – | – |
| 7 | 35.9 | – | – |
| 8 | 36.4 | – | – |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
