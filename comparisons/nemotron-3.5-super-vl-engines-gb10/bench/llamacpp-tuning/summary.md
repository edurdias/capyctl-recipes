# Super VL NVFP4 on llama.cpp v0.7.0: drafters and prompt batch

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Super VL NVFP4, LiLiCorr 7.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Super VL NVFP4, LiLiCorr 7 | `nemotron-super-vl` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, LiLiCorr 5 | `supervl-lc5` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, LiLiCorr 4 | `supervl-lc4` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, MTP 5 | `supervl-mtp5` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, MTP 3 | `supervl-mtp3` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, MTP 2 | `supervl-mtp2` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, MTP 3, ubatch 1024 | `supervl-mtp3-ub1024` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, MTP 3, ubatch 2048 | `supervl-mtp3-ub2048` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, LiLiCorr 5, ubatch 2048 | `supervl-lc5-ub2048` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, no drafter | `supervl-nodraft` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 30.4 | – | – | – | – | – | – | – | 29.7 (-2.4%) | – |
| 1k | 26.7 | 28.3 (+5.9%) | 28.1 (+5.2%) | 26.4 (-1.1%) | 27.9 (+4.5%) | 25.9 (-2.9%) | 27.2 (+1.9%) | 26.1 (-2.2%) | 28.5 (+6.8%) | 13.8 (-48.1%) |
| 2k | 26.9 | – | – | – | – | – | – | – | 32.0 (+19.0%) | – |
| 4k | 26.1 | – | – | – | – | – | – | – | 29.1 (+11.5%) | – |
| 8k | 28.7 | 30.3 (+5.3%) | 29.8 (+3.6%) | 25.8 (-10.1%) | 28.5 (-1.0%) | 27.0 (-6.1%) | 28.3 (-1.4%) | 26.2 (-9.0%) | 28.0 (-2.7%) | 13.8 (-52.0%) |
| 16k | 24.3 | – | – | – | – | – | – | – | 27.8 (+14.1%) | – |
| 32k | 26.0 | – | – | 24.6 (-5.2%) | 26.6 (+2.4%) | – | 26.3 (+1.1%) | 26.0 (-0.0%) | 26.6 (+2.5%) | 13.6 (-47.7%) |
| 64k | 24.0 | – | – | – | – | – | – | – | 25.8 (+7.3%) | – |
| 128k | 24.2 | – | – | – | – | – | – | – | 26.1 (+8.1%) | – |

### Prompt processing (tok/s, higher is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 469 | – | – | – | – | – | – | – | 465 (-0.9%) | – |
| 1k | 518 | 507 (-2.1%) | 511 (-1.3%) | 531 (+2.4%) | 533 (+2.8%) | 527 (+1.8%) | 642 (+23.9%) | 628 (+21.2%) | 620 (+19.6%) | 573 (+10.6%) |
| 2k | 572 | – | – | – | – | – | – | – | 738 (+29.1%) | – |
| 4k | 601 | – | – | – | – | – | – | – | 763 (+27.0%) | – |
| 8k | 618 | 617 (-0.1%) | 616 (-0.3%) | 645 (+4.3%) | 651 (+5.4%) | 641 (+3.7%) | 762 (+23.4%) | 810 (+31.1%) | 779 (+26.2%) | 672 (+8.8%) |
| 16k | 630 | – | – | – | – | – | – | – | 798 (+26.6%) | – |
| 32k | 642 | – | – | 663 (+3.3%) | 672 (+4.7%) | – | 776 (+20.9%) | 833 (+29.7%) | 803 (+25.2%) | 688 (+7.2%) |
| 64k | 649 | – | – | – | – | – | – | – | 794 (+22.3%) | – |
| 128k | 639 | – | – | – | – | – | – | – | 759 (+18.8%) | – |

### Time to first token (s, lower is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 1.08 | – | – | – | – | – | – | – | 1.09 (+0.9%) | – |
| 1k | 1.94 | 1.98 (+2.4%) | 1.95 (+0.7%) | 1.89 (-2.5%) | 1.88 (-2.8%) | 1.90 (-2.1%) | 1.55 (-19.8%) | 1.60 (-17.6%) | 1.62 (-16.5%) | 1.74 (-10.0%) |
| 2k | 3.54 | – | – | – | – | – | – | – | 2.76 (-22.0%) | – |
| 4k | 6.70 | – | – | – | – | – | – | – | 5.25 (-21.5%) | – |
| 8k | 12.9 | 12.9 (-0.1%) | 12.9 (+0.0%) | 12.4 (-4.2%) | 12.2 (-5.4%) | 12.5 (-3.8%) | 10.5 (-18.9%) | 9.85 (-23.9%) | 10.2 (-20.9%) | 11.9 (-8.3%) |
| 16k | 25.3 | – | – | – | – | – | – | – | 20.0 (-21.0%) | – |
| 32k | 50.3 | – | – | 48.7 (-3.2%) | 48.0 (-4.5%) | – | 41.7 (-17.1%) | 38.9 (-22.6%) | 40.2 (-20.1%) | 47.1 (-6.3%) |
| 64k | 99.6 | – | – | – | – | – | – | – | 81.2 (-18.4%) | – |
| 128k | 201 | – | – | – | – | – | – | – | 169 (-15.8%) | – |

### Time per output token (ms, lower is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 32.8 | – | – | – | – | – | – | – | 33.7 (+2.5%) | – |
| 1k | 37.5 | 35.4 (-5.6%) | 35.6 (-5.0%) | 37.9 (+1.1%) | 35.9 (-4.3%) | 38.6 (+2.9%) | 36.8 (-1.9%) | 38.3 (+2.3%) | 35.1 (-6.4%) | 72.3 (+92.8%) |
| 2k | 37.2 | – | – | – | – | – | – | – | 31.3 (-15.9%) | – |
| 4k | 38.3 | – | – | – | – | – | – | – | 34.4 (-10.3%) | – |
| 8k | 34.8 | 33.0 (-5.0%) | 33.6 (-3.5%) | 38.7 (+11.3%) | 35.1 (+1.0%) | 37.0 (+6.4%) | 35.3 (+1.4%) | 38.2 (+9.9%) | 35.7 (+2.8%) | 72.5 (+108.5%) |
| 16k | 41.1 | – | – | – | – | – | – | – | 36.0 (-12.3%) | – |
| 32k | 38.5 | – | – | 40.6 (+5.5%) | 37.6 (-2.3%) | – | 38.1 (-1.1%) | 38.5 (+0.0%) | 37.5 (-2.5%) | 73.5 (+91.1%) |
| 64k | 41.6 | – | – | – | – | – | – | – | 38.8 (-6.8%) | – |
| 128k | 41.3 | – | – | – | – | – | – | – | 38.3 (-7.5%) | – |

### Total time (s, lower is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 5.27 | – | – | – | – | – | – | – | 5.39 (+2.1%) | – |
| 1k | 6.70 | 6.50 (-2.9%) | 6.48 (-3.2%) | 6.70 (+0.0%) | 6.44 (-3.8%) | 6.81 (+1.7%) | 6.24 (-6.8%) | 6.47 (-3.3%) | 6.08 (-9.3%) | 10.9 (+63.3%) |
| 2k | 8.28 | – | – | – | – | – | – | – | 7.03 (-15.1%) | – |
| 4k | 11.6 | – | – | – | – | – | – | – | 9.72 (-16.3%) | – |
| 8k | 17.4 | 17.1 (-1.4%) | 17.2 (-0.9%) | 17.5 (+0.5%) | 16.7 (-3.8%) | 17.1 (-1.4%) | 15.0 (-13.6%) | 14.9 (-14.3%) | 14.9 (-14.5%) | 21.1 (+21.5%) |
| 16k | 30.6 | – | – | – | – | – | – | – | 24.6 (-19.6%) | – |
| 32k | 55.4 | – | – | 53.9 (-2.6%) | 52.9 (-4.4%) | – | 46.5 (-16.0%) | 43.7 (-21.1%) | 45.0 (-18.8%) | 56.4 (+1.9%) |
| 64k | 105 | – | – | – | – | – | – | – | 86.1 (-17.9%) | – |
| 128k | 207 | – | – | – | – | – | – | – | 174 (-15.7%) | – |

### Peak memory (GiB, lower is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 86.0 | – | – | – | – | – | – | – | 92.8 (+8.0%) | – |
| 1k | 86.0 | 86.1 (+0.1%) | 89.2 (+3.8%) | 88.8 (+3.3%) | 89.0 (+3.5%) | 89.8 (+4.4%) | 89.8 (+4.5%) | 93.4 (+8.7%) | 93.0 (+8.2%) | 85.9 (-0.1%) |
| 2k | 86.0 | – | – | – | – | – | – | – | 93.3 (+8.5%) | – |
| 4k | 86.2 | – | – | – | – | – | – | – | 93.9 (+9.0%) | – |
| 8k | 86.3 | 86.8 (+0.5%) | 89.8 (+4.1%) | 89.1 (+3.2%) | 89.0 (+3.1%) | 90.5 (+4.9%) | 91.3 (+5.8%) | 94.2 (+9.1%) | 93.9 (+8.7%) | 85.9 (-0.5%) |
| 16k | 86.5 | – | – | – | – | – | – | – | 91.4 (+5.7%) | – |
| 32k | 86.5 | – | – | 91.5 (+5.7%) | 88.7 (+2.6%) | – | 92.5 (+7.0%) | 95.7 (+10.6%) | 93.1 (+7.6%) | 84.2 (-2.7%) |
| 64k | 86.6 | – | – | – | – | – | – | – | 93.9 (+8.5%) | – |
| 128k | 86.6 | – | – | – | – | – | – | – | 94.0 (+8.5%) | – |

### Output bytes per second (B/s, higher is better)

| Context | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5k | 144 | – | – | – | – | – | – | – | 141 (-2.4%) | – |
| 1k | 129 | 138 (+6.8%) | 128 (-1.0%) | 123 (-5.1%) | 127 (-1.7%) | 122 (-5.3%) | 111 (-13.9%) | 114 (-12.1%) | 125 (-3.1%) | 67.3 (-48.0%) |
| 2k | 122 | – | – | – | – | – | – | – | 145 (+19.1%) | – |
| 4k | 125 | – | – | – | – | – | – | – | 134 (+7.4%) | – |
| 8k | 127 | 131 (+3.1%) | 129 (+1.3%) | 117 (-8.1%) | 136 (+7.3%) | 120 (-5.7%) | 132 (+3.6%) | 124 (-2.9%) | 125 (-1.8%) | 63.3 (-50.2%) |
| 16k | 116 | – | – | – | – | – | – | – | 128 (+10.2%) | – |
| 32k | 119 | – | – | 111 (-7.2%) | 127 (+6.6%) | – | 120 (+1.1%) | 127 (+7.0%) | 123 (+2.9%) | 62.3 (-47.7%) |
| 64k | 118 | – | – | – | – | – | – | – | 126 (+6.6%) | – |
| 128k | 113 | – | – | – | – | – | – | – | 119 (+4.8%) | – |

## Concurrency

### Aggregate (tok/s)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 25.1 | 26.7 (+6.5%) | 27.1 (+8.0%) | 25.5 (+1.6%) | 26.7 (+6.1%) | 25.2 (+0.2%) | 26.8 (+6.6%) | 26.8 (+6.9%) | 26.6 (+5.9%) | 13.6 (-45.9%) |

### Per stream (tok/s)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 25.5 | 27.2 (+6.4%) | 27.9 (+9.1%) | 26.2 (+2.6%) | 27.4 (+7.4%) | 25.6 (+0.0%) | 27.5 (+7.8%) | 27.1 (+6.1%) | 27.2 (+6.3%) | 13.8 (-46.1%) |

### TTFT p50 (s)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.552 | 0.571 (+3.3%) | 0.577 (+4.5%) | 0.556 (+0.7%) | 0.563 (+1.9%) | 0.549 (-0.6%) | 0.552 (-0.0%) | 0.582 (+5.4%) | 0.581 (+5.1%) | 0.530 (-4.1%) |

### TTFT p95 (s)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.562 | 0.584 (+3.8%) | 0.580 (+3.2%) | 0.580 (+3.1%) | 0.567 (+0.8%) | 0.561 (-0.3%) | 0.561 (-0.2%) | 0.649 (+15.3%) | 0.589 (+4.7%) | 0.556 (-1.1%) |

### TPOT (ms)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 39.1 | 36.8 (-6.0%) | 35.9 (-8.3%) | 38.1 (-2.6%) | 36.4 (-6.9%) | 39.1 (-0.0%) | 36.3 (-7.2%) | 36.9 (-5.8%) | 36.8 (-5.9%) | 72.6 (+85.5%) |

### Draft acceptance (%)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 43.5% | 53.5% (+23.0%) | 59.8% (+37.3%) | 56.1% (+28.9%) | 71.9% (+65.2%) | 80.1% (+84.0%) | 71.9% (+65.2%) | 72.0% (+65.5%) | 53.5% (+23.0%) | – |

### Peak memory (GiB)

| Streams | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, LiLiCorr 5 | Super VL NVFP4, LiLiCorr 4 | Super VL NVFP4, MTP 5 | Super VL NVFP4, MTP 3 | Super VL NVFP4, MTP 2 | Super VL NVFP4, MTP 3, ubatch 1024 | Super VL NVFP4, MTP 3, ubatch 2048 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86.7 | 87.9 (+1.4%) | 86.4 (-0.4%) | 89.9 (+3.7%) | 88.6 (+2.2%) | 91.1 (+5.1%) | 93.4 (+7.7%) | 93.1 (+7.4%) | 91.8 (+5.9%) | 84.1 (-3.0%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
