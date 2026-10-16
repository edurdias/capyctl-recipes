# Bolt 24GB tuning on llama.cpp v0.7.0, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Bolt 24GB, LiLiCorr 7.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Bolt 24GB, LiLiCorr 7 | `nemotron-bolt` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 6 | `bolt-nmax6` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 5 | `bolt-nmax5` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 4 | `bolt-nmax4` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 7, ubatch 512 | `bolt-ub512` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 7, ubatch 1024 | `bolt-ub1024` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 7, ubatch 4096 | `bolt-ub4096` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, LiLiCorr 7, draft ubatch 512 | `bolt-dub512` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Bolt 24GB, 2 slots x 128k, LiLiCorr 7 | `bolt-2x128k` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 66.7 | – | – | – | – | – | – | – | – |
| 1k | 63.9 | 67.0 (+4.8%) | 67.4 (+5.5%) | 64.6 (+1.1%) | 63.8 (-0.1%) | 63.9 (+0.0%) | 65.7 (+2.8%) | 63.0 (-1.4%) | 65.9 (+3.1%) |
| 2k | 63.8 | – | – | – | – | – | – | – | – |
| 4k | 63.5 | – | – | – | – | – | – | – | – |
| 8k | 62.5 | 64.2 (+2.8%) | 68.1 (+9.0%) | 64.6 (+3.4%) | 63.3 (+1.3%) | 59.3 (-5.2%) | 68.5 (+9.7%) | 61.0 (-2.3%) | – |
| 16k | 63.1 | – | – | – | – | – | – | – | – |
| 32k | 62.8 | – | – | – | 64.0 (+2.0%) | 64.4 (+2.5%) | 74.6 (+18.8%) | 60.8 (-3.1%) | – |
| 64k | 65.7 | – | – | – | 55.6 (-15.4%) | 71.9 (+9.4%) | 61.2 (-6.8%) | 65.8 (+0.1%) | – |
| 128k | 61.1 | – | – | – | – | – | – | – | – |
| 200k | 52.2 | – | – | – | – | – | – | – | – |

### Prompt processing (tok/s, higher is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 731 | – | – | – | – | – | – | – | – |
| 1k | 937 | 928 (-0.9%) | 931 (-0.6%) | 933 (-0.4%) | 804 (-14.1%) | 928 (-0.9%) | 916 (-2.2%) | 915 (-2.3%) | 919 (-1.9%) |
| 2k | 1,073 | – | – | – | – | – | – | – | – |
| 4k | 1,115 | – | – | – | – | – | – | – | – |
| 8k | 1,156 | 1,157 (+0.0%) | 1,162 (+0.5%) | 1,170 (+1.2%) | 1,051 (-9.1%) | 1,127 (-2.6%) | 1,150 (-0.5%) | 1,156 (-0.0%) | – |
| 16k | 1,177 | – | – | – | – | – | – | – | – |
| 32k | 1,171 | – | – | – | 1,079 (-7.9%) | 1,145 (-2.2%) | 1,165 (-0.5%) | 1,166 (-0.5%) | – |
| 64k | 1,136 | – | – | – | 1,054 (-7.2%) | 1,116 (-1.7%) | 1,137 (+0.1%) | 1,135 (-0.0%) | – |
| 128k | 1,068 | – | – | – | – | – | – | – | – |
| 200k | 990 | – | – | – | – | – | – | – | – |

### Time to first token (s, lower is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 0.691 | – | – | – | – | – | – | – | – |
| 1k | 1.07 | 1.08 (+1.1%) | 1.08 (+1.2%) | 1.07 (+0.7%) | 1.24 (+16.2%) | 1.07 (+0.8%) | 1.09 (+2.2%) | 1.09 (+2.2%) | 1.08 (+1.7%) |
| 2k | 1.90 | – | – | – | – | – | – | – | – |
| 4k | 3.61 | – | – | – | – | – | – | – | – |
| 8k | 6.90 | 6.90 (-0.0%) | 6.91 (+0.1%) | 6.82 (-1.2%) | 7.59 (+10.1%) | 7.08 (+2.6%) | 6.94 (+0.6%) | 6.90 (+0.0%) | – |
| 16k | 13.6 | – | – | – | – | – | – | – | – |
| 32k | 27.5 | – | – | – | 29.9 (+8.5%) | 28.2 (+2.2%) | 27.8 (+0.9%) | 27.7 (+0.4%) | – |
| 64k | 56.7 | – | – | – | 61.3 (+8.0%) | 57.8 (+2.0%) | 56.8 (+0.2%) | 56.8 (+0.2%) | – |
| 128k | 120 | – | – | – | – | – | – | – | – |
| 200k | 204 | – | – | – | – | – | – | – | – |

### Time per output token (ms, lower is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 15.0 | – | – | – | – | – | – | – | – |
| 1k | 15.6 | 14.9 (-4.6%) | 14.8 (-5.2%) | 15.5 (-1.1%) | 15.7 (+0.1%) | 15.6 (-0.0%) | 15.2 (-2.7%) | 15.9 (+1.4%) | 15.2 (-3.0%) |
| 2k | 15.7 | – | – | – | – | – | – | – | – |
| 4k | 15.8 | – | – | – | – | – | – | – | – |
| 8k | 16.0 | 15.6 (-2.7%) | 14.7 (-8.3%) | 15.5 (-3.3%) | 15.8 (-1.3%) | 16.9 (+5.5%) | 14.6 (-8.8%) | 16.4 (+2.4%) | – |
| 16k | 15.8 | – | – | – | – | – | – | – | – |
| 32k | 15.9 | – | – | – | 15.6 (-2.0%) | 15.5 (-2.5%) | 13.4 (-15.8%) | 16.4 (+3.3%) | – |
| 64k | 15.2 | – | – | – | 18.0 (+18.1%) | 13.9 (-8.6%) | 16.3 (+7.3%) | 15.2 (-0.1%) | – |
| 128k | 16.4 | – | – | – | – | – | – | – | – |
| 200k | 19.1 | – | – | – | – | – | – | – | – |

### Total time (s, lower is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 2.61 | – | – | – | – | – | – | – | – |
| 1k | 3.03 | 2.99 (-1.4%) | 2.98 (-1.6%) | 3.05 (+0.9%) | 3.23 (+6.9%) | 3.07 (+1.5%) | 3.04 (+0.4%) | 3.13 (+3.3%) | 3.03 (+0.1%) |
| 2k | 3.87 | – | – | – | – | – | – | – | – |
| 4k | 5.54 | – | – | – | – | – | – | – | – |
| 8k | 9.08 | 9.02 (-0.6%) | 8.86 (-2.4%) | 8.92 (-1.8%) | 9.72 (+7.0%) | 9.35 (+3.0%) | 9.37 (+3.2%) | 9.14 (+0.6%) | – |
| 16k | 15.6 | – | – | – | – | – | – | – | – |
| 32k | 29.6 | – | – | – | 32.0 (+8.3%) | 30.1 (+1.9%) | 29.8 (+0.8%) | 29.8 (+0.7%) | – |
| 64k | 58.5 | – | – | – | 63.5 (+8.5%) | 59.6 (+1.8%) | 58.9 (+0.7%) | 58.6 (+0.1%) | – |
| 128k | 123 | – | – | – | – | – | – | – | – |
| 200k | 206 | – | – | – | – | – | – | – | – |

### Peak memory (GiB, lower is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 37.6 | – | – | – | – | – | – | – | – |
| 1k | 37.7 | 38.0 (+0.6%) | 37.8 (+0.2%) | 37.6 (-0.3%) | 34.5 (-8.4%) | 35.7 (-5.5%) | 41.4 (+9.8%) | 35.9 (-4.7%) | 37.9 (+0.5%) |
| 2k | 38.5 | – | – | – | – | – | – | – | – |
| 4k | 38.5 | – | – | – | – | – | – | – | – |
| 8k | 38.7 | 38.2 (-1.2%) | 38.0 (-1.6%) | 37.9 (-2.0%) | 34.6 (-10.6%) | 35.8 (-7.3%) | 42.7 (+10.4%) | 36.0 (-7.0%) | – |
| 16k | 38.7 | – | – | – | – | – | – | – | – |
| 32k | 38.7 | – | – | – | 34.6 (-10.6%) | 35.9 (-7.3%) | 42.7 (+10.3%) | 35.9 (-7.2%) | – |
| 64k | 38.7 | – | – | – | 34.6 (-10.6%) | 35.9 (-7.3%) | 42.6 (+10.0%) | 36.0 (-7.2%) | – |
| 128k | 38.8 | – | – | – | – | – | – | – | – |
| 200k | 38.8 | – | – | – | – | – | – | – | – |

### Output bytes per second (B/s, higher is better)

| Context | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 339 | – | – | – | – | – | – | – | – |
| 1k | 300 | 307 (+2.3%) | 322 (+7.3%) | 310 (+3.5%) | 332 (+10.7%) | 298 (-0.7%) | 310 (+3.3%) | 291 (-2.9%) | 310 (+3.3%) |
| 2k | 337 | – | – | – | – | – | – | – | – |
| 4k | 332 | – | – | – | – | – | – | – | – |
| 8k | 323 | 332 (+2.8%) | 352 (+9.0%) | 334 (+3.4%) | 285 (-12.0%) | 303 (-6.4%) | 267 (-17.4%) | 316 (-2.3%) | – |
| 16k | 313 | – | – | – | – | – | – | – | – |
| 32k | 322 | – | – | – | 298 (-7.5%) | 304 (-5.6%) | 277 (-14.0%) | 309 (-4.1%) | – |
| 64k | 249 | – | – | – | 269 (+7.7%) | 273 (+9.6%) | 330 (+32.1%) | 259 (+3.9%) | – |
| 128k | 316 | – | – | – | – | – | – | – | – |
| 200k | 267 | – | – | – | – | – | – | – | – |

## Concurrency

### Aggregate (tok/s)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 69.3 | 71.1 (+2.6%) | 73.5 (+6.0%) | 74.0 (+6.7%) | 71.6 (+3.3%) | 70.4 (+1.5%) | 69.5 (+0.3%) | 70.7 (+2.1%) | 72.1 (+4.1%) |
| 2 | – | – | – | – | – | – | – | – | 68.1 |

### Per stream (tok/s)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 72.4 | 74.4 (+2.8%) | 77.0 (+6.4%) | 76.1 (+5.1%) | 75.0 (+3.7%) | 73.7 (+1.8%) | 72.8 (+0.5%) | 74.2 (+2.5%) | 75.5 (+4.3%) |
| 2 | – | – | – | – | – | – | – | – | 35.6 |

### TTFT p50 (s)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.329 | 0.320 (-2.9%) | 0.319 (-3.0%) | 0.325 (-1.2%) | 0.332 (+0.9%) | 0.330 (+0.3%) | 0.332 (+0.9%) | 0.335 (+1.8%) | 0.323 (-1.7%) |
| 2 | – | – | – | – | – | – | – | – | 0.630 |

### TTFT p95 (s)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.333 | 0.327 (-1.6%) | 0.328 (-1.4%) | 0.329 (-1.2%) | 0.342 (+2.6%) | 0.337 (+1.3%) | 0.343 (+3.1%) | 0.343 (+3.0%) | 0.336 (+0.9%) |
| 2 | – | – | – | – | – | – | – | – | 0.734 |

### TPOT (ms)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.8 | 13.4 (-2.7%) | 13.0 (-6.0%) | 13.1 (-4.9%) | 13.3 (-3.5%) | 13.6 (-1.8%) | 13.7 (-0.5%) | 13.5 (-2.4%) | 13.2 (-4.1%) |
| 2 | – | – | – | – | – | – | – | – | 28.1 |

### Draft acceptance (%)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 48.6% | 53.2% (+9.3%) | 59.7% (+22.7%) | 68.9% (+41.7%) | 48.6% (+0.0%) | 48.6% (+0.0%) | 48.6% (+0.0%) | 48.6% (+0.0%) | 48.6% (+0.0%) |
| 2 | – | – | – | – | – | – | – | – | 42.9% |

### Peak memory (GiB)

| Streams | Bolt 24GB, LiLiCorr 7 | Bolt 24GB, LiLiCorr 6 | Bolt 24GB, LiLiCorr 5 | Bolt 24GB, LiLiCorr 4 | Bolt 24GB, LiLiCorr 7, ubatch 512 | Bolt 24GB, LiLiCorr 7, ubatch 1024 | Bolt 24GB, LiLiCorr 7, ubatch 4096 | Bolt 24GB, LiLiCorr 7, draft ubatch 512 | Bolt 24GB, 2 slots x 128k, LiLiCorr 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 38.5 | 37.9 (-1.7%) | 37.7 (-2.1%) | 37.5 (-2.5%) | 34.4 (-10.6%) | 35.7 (-7.3%) | 42.3 (+9.9%) | 35.8 (-7.1%) | 37.8 (-1.7%) |
| 2 | – | – | – | – | – | – | – | – | 37.8 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
