# Nemotron 3.5 Bolt 74B-A9B 24GB on llama.cpp v0.7.0, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Bolt 24GB, LiLiCorr 5, draft ubatch 512.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Bolt 24GB, LiLiCorr 5, draft ubatch 512 | `bolt-tuned` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 7 | `nemotron-bolt` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, no drafter | `nemotron-bolt-nodraft` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 67.4 | 66.7 (-1.1%) | 38.6 (-42.7%) |
| 1k | 67.7 | 63.9 (-5.7%) | 38.5 (-43.1%) |
| 2k | 64.8 | 63.8 (-1.5%) | 38.4 (-40.8%) |
| 4k | 67.3 | 63.5 (-5.7%) | 38.3 (-43.1%) |
| 8k | 67.5 | 62.5 (-7.4%) | 38.2 (-43.4%) |
| 16k | 71.6 | 63.1 (-11.9%) | 37.7 (-47.3%) |
| 32k | 70.9 | 62.8 (-11.5%) | 36.5 (-48.5%) |
| 64k | 74.5 | 65.7 (-11.8%) | – |
| 128k | 66.5 | 61.1 (-8.1%) | – |
| 200k | 54.3 | 52.2 (-3.8%) | – |

### Prompt processing (tok/s, higher is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 733 | 731 (-0.2%) | 847 (+15.6%) |
| 1k | 925 | 937 (+1.3%) | 1,084 (+17.2%) |
| 2k | 1,059 | 1,073 (+1.3%) | 1,232 (+16.4%) |
| 4k | 1,111 | 1,115 (+0.3%) | 1,296 (+16.6%) |
| 8k | 1,153 | 1,156 (+0.3%) | 1,333 (+15.6%) |
| 16k | 1,173 | 1,177 (+0.4%) | 1,349 (+15.1%) |
| 32k | 1,166 | 1,171 (+0.4%) | 1,338 (+14.7%) |
| 64k | 1,137 | 1,136 (-0.1%) | – |
| 128k | 1,068 | 1,068 (-0.0%) | – |
| 200k | 988 | 990 (+0.2%) | – |

### Time to first token (s, lower is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 0.689 | 0.691 (+0.2%) | 0.577 (-16.3%) |
| 1k | 1.08 | 1.07 (-1.3%) | 0.930 (-14.0%) |
| 2k | 1.90 | 1.90 (+0.0%) | 1.65 (-12.9%) |
| 4k | 3.62 | 3.61 (-0.2%) | 3.12 (-13.9%) |
| 8k | 6.92 | 6.90 (-0.3%) | 5.97 (-13.8%) |
| 16k | 13.6 | 13.6 (-0.2%) | 11.8 (-13.2%) |
| 32k | 27.7 | 27.5 (-0.4%) | 24.1 (-12.8%) |
| 64k | 56.6 | 56.7 (+0.2%) | – |
| 128k | 120 | 120 (-0.0%) | – |
| 200k | 204 | 204 (-0.1%) | – |

### Time per output token (ms, lower is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 14.8 | 15.0 (+1.1%) | 25.9 (+74.7%) |
| 1k | 14.8 | 15.6 (+6.0%) | 26.0 (+75.9%) |
| 2k | 15.4 | 15.7 (+1.6%) | 26.0 (+68.8%) |
| 4k | 14.9 | 15.8 (+6.0%) | 26.1 (+75.8%) |
| 8k | 14.8 | 16.0 (+8.0%) | 26.2 (+76.7%) |
| 16k | 14.0 | 15.8 (+13.5%) | 26.5 (+89.7%) |
| 32k | 14.1 | 15.9 (+13.0%) | 27.4 (+94.3%) |
| 64k | 13.4 | 15.2 (+13.4%) | – |
| 128k | 15.0 | 16.4 (+8.8%) | – |
| 200k | 18.4 | 19.1 (+3.9%) | – |

### Total time (s, lower is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 2.59 | 2.61 (+0.9%) | 3.89 (+50.2%) |
| 1k | 2.97 | 3.03 (+2.0%) | 4.24 (+43.0%) |
| 2k | 3.87 | 3.87 (+0.1%) | 4.98 (+28.6%) |
| 4k | 5.49 | 5.54 (+1.0%) | 6.44 (+17.4%) |
| 8k | 8.95 | 9.08 (+1.4%) | 9.31 (+4.0%) |
| 16k | 15.5 | 15.6 (+0.7%) | 15.2 (-1.7%) |
| 32k | 29.5 | 29.6 (+0.4%) | 27.6 (-6.3%) |
| 64k | 58.4 | 58.5 (+0.2%) | – |
| 128k | 122 | 123 (+0.2%) | – |
| 200k | 206 | 206 (+0.1%) | – |

### Peak memory (GiB, lower is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 35.8 | 37.6 (+4.9%) | 33.1 (-7.6%) |
| 1k | 35.9 | 37.7 (+5.2%) | 33.1 (-7.7%) |
| 2k | 35.9 | 38.5 (+7.3%) | 33.1 (-7.9%) |
| 4k | 36.3 | 38.5 (+6.2%) | 33.3 (-8.3%) |
| 8k | 36.7 | 38.7 (+5.3%) | 33.3 (-9.5%) |
| 16k | 37.3 | 38.7 (+3.8%) | 33.3 (-10.7%) |
| 32k | 38.0 | 38.7 (+1.8%) | 33.3 (-12.4%) |
| 64k | 38.4 | 38.7 (+0.9%) | – |
| 128k | 37.1 | 38.8 (+4.7%) | – |
| 200k | 37.3 | 38.8 (+3.9%) | – |

### Output bytes per second (B/s, higher is better)

| Context | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 0.5k | 342 | 339 (-0.8%) | 200 (-41.4%) |
| 1k | 324 | 300 (-7.4%) | 197 (-39.3%) |
| 2k | 342 | 337 (-1.5%) | 195 (-43.1%) |
| 4k | 344 | 332 (-3.5%) | 195 (-43.2%) |
| 8k | 349 | 323 (-7.4%) | 191 (-45.4%) |
| 16k | 341 | 313 (-8.2%) | 138 (-59.6%) |
| 32k | 352 | 322 (-8.4%) | 180 (-48.9%) |
| 64k | 293 | 249 (-15.0%) | – |
| 128k | 344 | 316 (-8.1%) | – |
| 200k | 278 | 267 (-3.8%) | – |

## Concurrency

### Aggregate (tok/s)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 75.0 | 69.3 (-7.6%) | 38.0 (-49.4%) |

### Per stream (tok/s)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 78.8 | 72.4 (-8.1%) | 38.7 (-50.9%) |

### TTFT p50 (s)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 0.337 | 0.329 (-2.2%) | 0.285 (-15.2%) |

### TTFT p95 (s)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 0.346 | 0.333 (-3.8%) | 0.289 (-16.6%) |

### TPOT (ms)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 12.7 | 13.8 (+8.8%) | 25.8 (+103.6%) |

### Draft acceptance (%)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 59.7% | 48.6% (-18.5%) | – |

### Peak memory (GiB)

| Streams | Bolt 24GB, LiLiCorr 5, draft ubatch 512 | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, no drafter |
|---|---|---|---|
| 1 | 38.0 | 38.5 (+1.3%) | 33.1 (-12.8%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
