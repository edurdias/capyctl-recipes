# Super VL NVFP4 on vLLM 0.31.0: LiLiCorr and MTP, 1 and 4 running

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | `supervl-vllm` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | `supervl-vllm-ro7` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | `supervl-vllm-lc5` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | `supervl-vllm-ro5` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, MTP 5 | `supervl-vllm-mtp5` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, MTP 3 | `supervl-vllm-mtp3` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | `supervl-vllm-lc7-4seq` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |
| Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running | `supervl-vllm-lc5-4seq` | 2026-10-16 | capyctl=0.1.3, engine=vllm-0.31.0, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 37.8 | – | – | – | – | – | – | – |
| 1k | 45.3 | 43.2 (-4.6%) | 46.8 (+3.3%) | 45.7 (+0.9%) | 37.5 (-17.2%) | 38.2 (-15.7%) | 40.6 (-10.4%) | 40.1 (-11.4%) |
| 2k | 43.6 | – | – | – | – | – | – | – |
| 4k | 42.9 | – | – | – | – | – | – | – |
| 8k | 43.9 | 43.0 (-1.9%) | 40.2 (-8.3%) | 41.1 (-6.4%) | 32.3 (-26.4%) | 36.5 (-16.8%) | 39.7 (-9.4%) | – |
| 16k | 35.0 | – | – | – | – | – | – | – |
| 32k | 33.9 | 36.7 (+8.2%) | 36.0 (+6.1%) | 33.8 (-0.5%) | 32.7 (-3.6%) | 36.1 (+6.3%) | 35.3 (+4.1%) | – |
| 64k | 35.2 | – | – | – | – | – | – | – |
| 100k | 41.5 | – | – | – | – | – | – | – |

### Prompt processing (tok/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 670 | – | – | – | – | – | – | – |
| 1k | 1,132 | 1,127 (-0.5%) | 1,154 (+1.9%) | 1,110 (-2.0%) | 1,100 (-2.9%) | 1,137 (+0.4%) | 1,128 (-0.3%) | 1,140 (+0.7%) |
| 2k | 1,742 | – | – | – | – | – | – | – |
| 4k | 2,319 | – | – | – | – | – | – | – |
| 8k | 2,875 | 2,873 (-0.1%) | 2,896 (+0.7%) | 2,873 (-0.1%) | 2,862 (-0.4%) | 2,955 (+2.8%) | 2,874 (-0.0%) | – |
| 16k | 3,020 | – | – | – | – | – | – | – |
| 32k | 3,005 | 2,955 (-1.7%) | 3,035 (+1.0%) | 3,008 (+0.1%) | 2,759 (-8.2%) | 2,787 (-7.2%) | 3,009 (+0.1%) | – |
| 64k | 2,768 | – | – | – | – | – | – | – |
| 100k | 2,469 | – | – | – | – | – | – | – |

### Time to first token (s, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 0.752 | – | – | – | – | – | – | – |
| 1k | 0.887 | 0.890 (+0.3%) | 0.867 (-2.2%) | 0.899 (+1.4%) | 0.907 (+2.2%) | 0.886 (-0.0%) | 0.884 (-0.3%) | 0.877 (-1.1%) |
| 2k | 1.18 | – | – | – | – | – | – | – |
| 4k | 1.75 | – | – | – | – | – | – | – |
| 8k | 2.78 | 2.78 (+0.1%) | 2.75 (-0.7%) | 2.78 (+0.1%) | 2.80 (+0.8%) | 2.69 (-2.9%) | 2.78 (+0.0%) | – |
| 16k | 5.29 | – | – | – | – | – | – | – |
| 32k | 10.7 | 10.9 (+1.9%) | 10.7 (-0.8%) | 10.7 (-0.1%) | 11.7 (+8.9%) | 11.6 (+8.1%) | 10.7 (-0.1%) | – |
| 64k | 23.3 | – | – | – | – | – | – | – |
| 100k | 40.8 | – | – | – | – | – | – | – |

### Time per output token (ms, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 26.4 | – | – | – | – | – | – | – |
| 1k | 22.1 | 23.2 (+4.9%) | 21.4 (-3.2%) | 21.9 (-0.9%) | 26.7 (+20.8%) | 26.2 (+18.6%) | 24.7 (+11.6%) | 24.9 (+12.9%) |
| 2k | 22.9 | – | – | – | – | – | – | – |
| 4k | 23.3 | – | – | – | – | – | – | – |
| 8k | 22.8 | 23.2 (+2.0%) | 24.9 (+9.1%) | 24.4 (+6.9%) | 31.0 (+35.9%) | 27.4 (+20.2%) | 25.2 (+10.4%) | – |
| 16k | 28.6 | – | – | – | – | – | – | – |
| 32k | 29.5 | 27.2 (-7.6%) | 27.8 (-5.8%) | 29.6 (+0.5%) | 30.6 (+3.8%) | 27.7 (-5.9%) | 28.3 (-3.9%) | – |
| 64k | 28.4 | – | – | – | – | – | – | – |
| 100k | 24.1 | – | – | – | – | – | – | – |

### Total time (s, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 4.13 | – | – | – | – | – | – | – |
| 1k | 3.71 | 3.83 (+3.4%) | 3.59 (-3.0%) | 3.70 (-0.3%) | 4.32 (+16.4%) | 4.19 (+13.1%) | 4.03 (+8.8%) | 4.06 (+9.6%) |
| 2k | 4.08 | – | – | – | – | – | – | – |
| 4k | 4.72 | – | – | – | – | – | – | – |
| 8k | 5.67 | 5.87 (+3.7%) | 5.93 (+4.6%) | 5.87 (+3.7%) | 6.75 (+19.1%) | 6.19 (+9.2%) | 5.97 (+5.3%) | – |
| 16k | 8.93 | – | – | – | – | – | – | – |
| 32k | 14.5 | 14.4 (-0.5%) | 14.2 (-2.0%) | 14.5 (+0.1%) | 15.7 (+8.3%) | 15.1 (+4.5%) | 14.3 (-0.9%) | – |
| 64k | 27.0 | – | – | – | – | – | – | – |
| 100k | 44.0 | – | – | – | – | – | – | – |

### Peak memory (GiB, lower is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 96.1 | – | – | – | – | – | – | – |
| 1k | 96.0 | 93.1 (-3.0%) | 96.0 (+0.1%) | 93.9 (-2.1%) | 93.5 (-2.5%) | 93.4 (-2.6%) | 96.0 (+0.1%) | 96.0 (+0.0%) |
| 2k | 95.9 | – | – | – | – | – | – | – |
| 4k | 97.2 | – | – | – | – | – | – | – |
| 8k | 98.5 | 95.7 (-2.8%) | 98.6 (+0.1%) | 96.6 (-2.0%) | 95.0 (-3.6%) | 94.9 (-3.7%) | 98.7 (+0.2%) | – |
| 16k | 100 | – | – | – | – | – | – | – |
| 32k | 99.8 | 97.1 (-2.7%) | 99.9 (+0.1%) | 97.9 (-1.9%) | 95.0 (-4.8%) | 94.9 (-4.9%) | 100 (+0.2%) | – |
| 64k | 99.8 | – | – | – | – | – | – | – |
| 100k | 99.9 | – | – | – | – | – | – | – |

### Output bytes per second (B/s, higher is better)

| Context | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 176 | – | – | – | – | – | – | – |
| 1k | 181 | 188 (+4.2%) | 206 (+13.7%) | 188 (+4.0%) | 174 (-3.6%) | 170 (-5.7%) | 199 (+10.1%) | 178 (-1.5%) |
| 2k | 228 | – | – | – | – | – | – | – |
| 4k | 191 | – | – | – | – | – | – | – |
| 8k | 195 | 198 (+1.7%) | 197 (+0.9%) | 191 (-2.0%) | 149 (-23.6%) | 176 (-9.5%) | 185 (-4.9%) | – |
| 16k | 164 | – | – | – | – | – | – | – |
| 32k | 164 | 166 (+1.2%) | 176 (+7.8%) | 176 (+7.7%) | 152 (-6.9%) | 180 (+10.2%) | 167 (+2.3%) | – |
| 64k | 163 | – | – | – | – | – | – | – |
| 100k | 157 | – | – | – | – | – | – | – |

## Concurrency

### Aggregate (tok/s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 1 | 37.2 | 38.9 (+4.5%) | 39.4 (+6.0%) | 38.4 (+3.1%) | 34.0 (-8.7%) | 36.0 (-3.2%) | 40.0 (+7.4%) | 40.3 (+8.4%) |
| 2 | – | – | – | – | – | – | 51.9 | 53.8 |
| 3 | – | – | – | – | – | – | 60.5 | 64.0 |
| 4 | – | – | – | – | – | – | 71.7 | 73.7 |

### Per stream (tok/s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 1 | 38.1 | 39.9 (+4.7%) | 40.5 (+6.0%) | 39.4 (+3.2%) | 34.6 (-9.4%) | 36.7 (-3.9%) | 41.1 (+7.7%) | 41.4 (+8.6%) |
| 2 | – | – | – | – | – | – | 27.5 | 30.0 |
| 3 | – | – | – | – | – | – | 21.0 | 23.1 |
| 4 | – | – | – | – | – | – | 19.3 | 20.4 |

### TTFT p50 (s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.352 | 0.361 (+2.6%) | 0.341 (-3.1%) | 0.364 (+3.3%) | 0.365 (+3.6%) | 0.316 (-10.2%) | 0.361 (+2.5%) | 0.337 (-4.4%) |
| 2 | – | – | – | – | – | – | 0.516 | 0.567 |
| 3 | – | – | – | – | – | – | 0.729 | 0.704 |
| 4 | – | – | – | – | – | – | 0.790 | 0.746 |

### TTFT p95 (s)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.357 | 0.369 (+3.5%) | 0.345 (-3.4%) | 0.379 (+6.3%) | 0.389 (+9.0%) | 0.337 (-5.4%) | 0.363 (+1.7%) | 0.343 (-3.8%) |
| 2 | – | – | – | – | – | – | 0.663 | 0.642 |
| 3 | – | – | – | – | – | – | 0.750 | 0.724 |
| 4 | – | – | – | – | – | – | 0.798 | 0.765 |

### TPOT (ms)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 1 | 26.2 | 25.0 (-4.5%) | 24.7 (-5.7%) | 25.4 (-3.1%) | 28.9 (+10.4%) | 27.3 (+4.0%) | 24.3 (-7.2%) | 24.1 (-7.9%) |
| 2 | – | – | – | – | – | – | 36.4 | 33.3 |
| 3 | – | – | – | – | – | – | 47.6 | 43.3 |
| 4 | – | – | – | – | – | – | 51.9 | 49.0 |

### Peak memory (GiB)

| Streams | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7 (2) | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5 (2) | Super VL NVFP4, vLLM 0.31.0, MTP 5 | Super VL NVFP4, vLLM 0.31.0, MTP 3 | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 7, 4 running | Super VL NVFP4, vLLM 0.31.0, LiLiCorr 5, 4 running |
|---|---|---|---|---|---|---|---|---|
| 1 | 99.9 | 97.1 (-2.8%) | 99.9 (+0.0%) | 97.9 (-2.0%) | 95.0 (-4.9%) | 94.9 (-5.0%) | 100 (+0.2%) | 95.9 (-4.0%) |
| 2 | – | – | – | – | – | – | 100 | 95.9 |
| 3 | – | – | – | – | – | – | 100 | 95.9 |
| 4 | – | – | – | – | – | – | 100 | 95.9 |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
