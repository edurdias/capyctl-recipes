# Nemotron 3.5 Super VL 120B-A12B on one GB10: vLLM 0.31.0 against llama.cpp v0.7.0

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | `supervl-vllm` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, MTP 5 | `supervl-vllm-mtp5` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, LiLiCorr 5, ubatch 2048 | `supervl-lc5-ub2048` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL Q4_0, LiLiCorr 5, ubatch 2048 | `supervl-q4-lc5` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, LiLiCorr 7 | `nemotron-super-vl` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |
| Super VL NVFP4, no drafter | `supervl-nodraft` | 2026-10-16 | capyctl=0.1.3, engine=llama.cpp-0.7.0+620d222, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 37.8 | – | 29.7 (-21.5%) | 36.4 (-3.9%) | 30.4 (-19.6%) | – |
| 1k | 45.3 | 37.5 (-17.2%) | 28.5 (-37.0%) | 36.6 (-19.1%) | 26.7 (-41.1%) | 13.8 (-69.4%) |
| 2k | 43.6 | – | 32.0 (-26.7%) | 34.6 (-20.6%) | 26.9 (-38.4%) | – |
| 4k | 42.9 | – | 29.1 (-32.1%) | 33.8 (-21.1%) | 26.1 (-39.2%) | – |
| 8k | 43.9 | 32.3 (-26.4%) | 28.0 (-36.2%) | 33.9 (-22.7%) | 28.7 (-34.5%) | 13.8 (-68.6%) |
| 16k | 35.0 | – | 27.8 (-20.7%) | 32.4 (-7.5%) | 24.3 (-30.5%) | – |
| 32k | 33.9 | 32.7 (-3.6%) | 26.6 (-21.5%) | 28.3 (-16.7%) | 26.0 (-23.4%) | 13.6 (-59.9%) |
| 64k | 35.2 | – | 25.8 (-26.7%) | 28.9 (-17.9%) | 24.0 (-31.7%) | – |
| 100k | 41.5 | – | – | – | – | – |
| 128k | – | – | 26.1 | 26.9 | 24.2 | – |

### Prompt processing (tok/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 670 | – | 465 (-30.6%) | 451 (-32.7%) | 469 (-30.0%) | – |
| 1k | 1,132 | 1,100 (-2.9%) | 620 (-45.3%) | 590 (-47.9%) | 518 (-54.2%) | 573 (-49.4%) |
| 2k | 1,742 | – | 738 (-57.6%) | 714 (-59.0%) | 572 (-67.2%) | – |
| 4k | 2,319 | – | 763 (-67.1%) | 739 (-68.1%) | 601 (-74.1%) | – |
| 8k | 2,875 | 2,862 (-0.4%) | 779 (-72.9%) | 761 (-73.5%) | 618 (-78.5%) | 672 (-76.6%) |
| 16k | 3,020 | – | 798 (-73.6%) | 778 (-74.2%) | 630 (-79.1%) | – |
| 32k | 3,005 | 2,759 (-8.2%) | 803 (-73.3%) | 781 (-74.0%) | 642 (-78.6%) | 688 (-77.1%) |
| 64k | 2,768 | – | 794 (-71.3%) | 772 (-72.1%) | 649 (-76.6%) | – |
| 100k | 2,469 | – | – | – | – | – |
| 128k | – | – | 759 | 744 | 639 | – |

### Time to first token (s, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 0.752 | – | 1.09 (+44.5%) | 1.11 (+47.6%) | 1.08 (+43.2%) | – |
| 1k | 0.887 | 0.907 (+2.2%) | 1.62 (+82.3%) | 1.69 (+90.7%) | 1.94 (+118.4%) | 1.74 (+96.5%) |
| 2k | 1.18 | – | 2.76 (+134.2%) | 2.83 (+140.7%) | 3.54 (+200.2%) | – |
| 4k | 1.75 | – | 5.25 (+200.5%) | 5.45 (+211.8%) | 6.70 (+283.0%) | – |
| 8k | 2.78 | 2.80 (+0.8%) | 10.2 (+268.8%) | 10.5 (+277.5%) | 12.9 (+366.3%) | 11.9 (+327.6%) |
| 16k | 5.29 | – | 20.0 (+278.5%) | 20.5 (+288.2%) | 25.3 (+379.0%) | – |
| 32k | 10.7 | 11.7 (+8.9%) | 40.2 (+274.2%) | 41.3 (+284.9%) | 50.3 (+368.3%) | 47.1 (+338.6%) |
| 64k | 23.3 | – | 81.2 (+247.9%) | 83.6 (+258.2%) | 99.6 (+326.7%) | – |
| 100k | 40.8 | – | – | – | – | – |
| 128k | – | – | 169 | 173 | 201 | – |

### Time per output token (ms, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 26.4 | – | 33.7 (+27.4%) | 27.5 (+4.0%) | 32.8 (+24.3%) | – |
| 1k | 22.1 | 26.7 (+20.8%) | 35.1 (+58.9%) | 27.3 (+23.6%) | 37.5 (+69.6%) | 72.3 (+227.1%) |
| 2k | 22.9 | – | 31.3 (+36.5%) | 28.9 (+26.0%) | 37.2 (+62.4%) | – |
| 4k | 23.3 | – | 34.4 (+47.3%) | 29.6 (+26.7%) | 38.3 (+64.3%) | – |
| 8k | 22.8 | 31.0 (+35.9%) | 35.7 (+56.8%) | 29.5 (+29.4%) | 34.8 (+52.6%) | 72.5 (+218.2%) |
| 16k | 28.6 | – | 36.0 (+26.1%) | 30.9 (+8.1%) | 41.1 (+43.9%) | – |
| 32k | 29.5 | 30.6 (+3.8%) | 37.5 (+27.3%) | 35.4 (+20.0%) | 38.5 (+30.5%) | 73.5 (+149.4%) |
| 64k | 28.4 | – | 38.8 (+36.5%) | 34.6 (+21.7%) | 41.6 (+46.4%) | – |
| 100k | 24.1 | – | – | – | – | – |
| 128k | – | – | 38.3 | 37.2 | 41.3 | – |

### Total time (s, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 4.13 | – | 5.39 (+30.6%) | 4.62 (+11.9%) | 5.27 (+27.8%) | – |
| 1k | 3.71 | 4.32 (+16.4%) | 6.08 (+63.9%) | 5.18 (+39.7%) | 6.70 (+80.6%) | 10.9 (+195.0%) |
| 2k | 4.08 | – | 7.03 (+72.3%) | 6.52 (+59.8%) | 8.28 (+102.9%) | – |
| 4k | 4.72 | – | 9.72 (+105.6%) | 9.10 (+92.7%) | 11.6 (+145.8%) | – |
| 8k | 5.67 | 6.75 (+19.1%) | 14.9 (+162.3%) | 14.3 (+152.5%) | 17.4 (+206.6%) | 21.1 (+272.5%) |
| 16k | 8.93 | – | 24.6 (+175.3%) | 24.7 (+176.7%) | 30.6 (+242.2%) | – |
| 32k | 14.5 | 15.7 (+8.3%) | 45.0 (+211.1%) | 46.0 (+217.8%) | 55.4 (+283.0%) | 56.4 (+290.4%) |
| 64k | 27.0 | – | 86.1 (+218.7%) | 87.8 (+224.9%) | 105 (+288.0%) | – |
| 100k | 44.0 | – | – | – | – | – |
| 128k | – | – | 174 | 177 | 207 | – |

### Peak memory (GiB, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 96.1 | – | 92.8 (-3.4%) | 84.0 (-12.6%) | 86.0 (-10.5%) | – |
| 1k | 96.0 | 93.5 (-2.5%) | 93.0 (-3.1%) | 84.0 (-12.4%) | 86.0 (-10.4%) | 85.9 (-10.5%) |
| 2k | 95.9 | – | 93.3 (-2.7%) | 84.2 (-12.3%) | 86.0 (-10.3%) | – |
| 4k | 97.2 | – | 93.9 (-3.4%) | 84.4 (-13.2%) | 86.2 (-11.4%) | – |
| 8k | 98.5 | 95.0 (-3.6%) | 93.9 (-4.7%) | 84.5 (-14.2%) | 86.3 (-12.4%) | 85.9 (-12.9%) |
| 16k | 100 | – | 91.4 (-9.0%) | 84.7 (-15.7%) | 86.5 (-13.9%) | – |
| 32k | 99.8 | 95.0 (-4.8%) | 93.1 (-6.7%) | 84.5 (-15.3%) | 86.5 (-13.3%) | 84.2 (-15.7%) |
| 64k | 99.8 | – | 93.9 (-6.0%) | 84.6 (-15.3%) | 86.6 (-13.3%) | – |
| 100k | 99.9 | – | – | – | – | – |
| 128k | – | – | 94.0 | 84.7 | 86.6 | – |

### Output bytes per second (B/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 0.5k | 176 | – | 141 (-20.0%) | 175 (-0.5%) | 144 (-18.1%) | – |
| 1k | 181 | 174 (-3.6%) | 125 (-30.7%) | 156 (-13.5%) | 129 (-28.4%) | 67.3 (-62.8%) |
| 2k | 228 | – | 145 (-36.5%) | 160 (-29.8%) | 122 (-46.7%) | – |
| 4k | 191 | – | 134 (-29.4%) | 154 (-19.4%) | 125 (-34.3%) | – |
| 8k | 195 | 149 (-23.6%) | 125 (-35.9%) | 139 (-28.9%) | 127 (-34.7%) | 63.3 (-67.5%) |
| 16k | 164 | – | 128 (-21.8%) | 126 (-23.2%) | 116 (-29.0%) | – |
| 32k | 164 | 152 (-6.9%) | 123 (-25.1%) | 126 (-22.9%) | 119 (-27.2%) | 62.3 (-61.9%) |
| 64k | 163 | – | 126 (-22.9%) | 142 (-12.9%) | 118 (-27.7%) | – |
| 100k | 157 | – | – | – | – | – |
| 128k | – | – | 119 | 130 | 113 | – |

## Concurrency

### Aggregate (tok/s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | 37.2 | 34.0 (-8.7%) | 26.6 (-28.5%) | 30.9 (-16.8%) | 25.1 (-32.5%) | 13.6 (-63.5%) |

### Per stream (tok/s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | 38.1 | 34.6 (-9.4%) | 27.2 (-28.8%) | 31.9 (-16.3%) | 25.5 (-33.0%) | 13.8 (-63.9%) |

### TTFT p50 (s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | 0.352 | 0.365 (+3.6%) | 0.581 (+64.9%) | 0.529 (+50.2%) | 0.552 (+56.9%) | 0.530 (+50.4%) |

### TTFT p95 (s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | 0.357 | 0.389 (+9.0%) | 0.589 (+65.1%) | 0.540 (+51.5%) | 0.562 (+57.7%) | 0.556 (+56.0%) |

### TPOT (ms)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | 26.2 | 28.9 (+10.4%) | 36.8 (+40.5%) | 31.3 (+19.5%) | 39.1 (+49.3%) | 72.6 (+177.0%) |

### Draft acceptance (%)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | – | – | 53.5% | 53.3% | 43.5% | – |

### Peak memory (GiB)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, LiLiCorr 5, ubatch 2048 | Super VL Q4_0, LiLiCorr 5, ubatch 2048 | Super VL NVFP4, LiLiCorr 7 | Super VL NVFP4, no drafter |
|---|---|---|---|---|---|---|
| 1 | 99.9 | 95.0 (-4.9%) | 91.8 (-8.1%) | 84.1 (-15.8%) | 86.7 (-13.2%) | 84.1 (-15.8%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
