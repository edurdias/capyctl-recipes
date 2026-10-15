# Qwen3.8-27B on SGLang 0.5.21 through CapyCTL: DFlash2 and MTP/EAGLE variants

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with dflash-8 (host b).

| Series | Model | Measured | Meta |
|---|---|---|---|
| dflash-8 (host b) | `qwen38-w-d8` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=52e264cba3b3c9dacb728713be3beed1399294ca, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4 |
| dflash-mix-12 (host b) | `qwen38-w-dm12` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=52e264cba3b3c9dacb728713be3beed1399294ca, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4 |
| dflash-mix-22 (host b) | `qwen38-w-dm22` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=52e264cba3b3c9dacb728713be3beed1399294ca, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4 |
| eagle-12 (host a) | `qwen38-w-e12` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=52e264cba3b3c9dacb728713be3beed1399294ca, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=none (the checkpoint's MTP layer) |
| eagle-22 (host a) | `qwen38-w-e22` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=52e264cba3b3c9dacb728713be3beed1399294ca, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=none (the checkpoint's MTP layer) |

## Context sweep

### Generation (tok/s, higher is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 47.5 | 48.0 (+0.9%) | 47.6 (+0.0%) | 30.5 (-35.9%) | 25.5 (-46.4%) |
| 32k | 31.4 | 31.6 (+0.8%) | 31.4 (-0.0%) | 25.7 (-18.0%) | 25.9 (-17.6%) |
| 128k | 24.5 | 25.2 (+2.9%) | 25.0 (+2.1%) | 23.0 (-5.8%) | 20.1 (-17.8%) |

### Prompt processing (tok/s, higher is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 2,353 | 2,350 (-0.1%) | 2,310 (-1.8%) | 2,306 (-2.0%) | 2,308 (-1.9%) |
| 32k | 1,728 | 1,558 (-9.9%) | 1,495 (-13.5%) | 1,663 (-3.8%) | 1,671 (-3.3%) |
| 128k | 1,053 | 991 (-5.9%) | 966 (-8.3%) | 1,004 (-4.6%) | 1,005 (-4.5%) |

### Time to first token (s, lower is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 0.860 | 0.861 (+0.1%) | 0.876 (+1.9%) | 0.878 (+2.0%) | 0.877 (+2.0%) |
| 32k | 18.7 | 20.7 (+10.9%) | 21.6 (+15.6%) | 19.4 (+3.9%) | 19.3 (+3.4%) |
| 128k | 122 | 130 (+6.3%) | 133 (+9.0%) | 128 (+4.8%) | 128 (+4.7%) |

### Time per output token (ms, lower is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 23.2 | 23.0 (-0.9%) | 23.2 (-0.0%) | 33.7 (+44.9%) | 39.3 (+69.0%) |
| 32k | 32.0 | 31.8 (-0.7%) | 32.1 (+0.0%) | 38.9 (+21.3%) | 38.7 (+20.7%) |
| 128k | 41.2 | 39.9 (-3.2%) | 40.2 (-2.4%) | 43.5 (+5.6%) | 49.8 (+20.9%) |

### Total time (s, lower is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 3.83 | 3.80 (-0.7%) | 3.84 (+0.4%) | 5.16 (+34.9%) | 5.88 (+53.6%) |
| 32k | 22.8 | 24.8 (+8.8%) | 25.7 (+12.8%) | 24.4 (+7.0%) | 24.3 (+6.5%) |
| 128k | 128 | 135 (+5.9%) | 138 (+8.6%) | 134 (+4.9%) | 134 (+5.4%) |

### Peak memory (GiB, lower is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 74.9 | 83.9 (+12.0%) | 107 (+42.6%) | 82.2 (+9.7%) | 95.6 (+27.6%) |
| 32k | 75.0 | 83.9 (+11.9%) | 107 (+42.4%) | 82.2 (+9.7%) | 95.6 (+27.5%) |
| 128k | 75.0 | 84.0 (+12.0%) | 107 (+42.6%) | 82.4 (+9.8%) | 95.7 (+27.6%) |

### Output bytes per second (B/s, higher is better)

| Context | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 2k | 246 | 248 (+0.9%) | 246 (+0.0%) | 157 (-35.9%) | 119 (-51.5%) |
| 32k | 134 | 135 (+0.8%) | 134 (-0.0%) | 111 (-17.7%) | 117 (-12.6%) |
| 128k | 112 | 115 (+2.9%) | 114 (+2.1%) | 105 (-6.3%) | 94.8 (-15.4%) |

## Concurrency

### Aggregate (tok/s)

| Streams | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 1 | 27.8 | 28.3 (+1.6%) | 28.0 (+0.7%) | 24.1 (-13.4%) | 24.2 (-13.0%) |
| 4 | 80.5 | 85.5 (+6.2%) | 80.5 (+0.0%) | 88.6 (+10.0%) | 88.0 (+9.2%) |
| 8 | 127 | 118 (-7.3%) | 125 (-1.5%) | 163 (+28.0%) | 150 (+18.0%) |
| 16 | 125 | 133 (+7.1%) | 173 (+39.2%) | 173 (+39.0%) | 273 (+119.0%) |
| 22 | 123 | 155 (+26.3%) | 191 (+55.3%) | 210 (+71.0%) | 344 (+180.0%) |

### Per stream (tok/s)

| Streams | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 1 | 28.3 | 28.6 (+0.9%) | 28.3 (-0.1%) | 24.7 (-12.7%) | 24.8 (-12.4%) |
| 4 | 22.5 | 23.1 (+2.7%) | 23.5 (+4.5%) | 23.4 (+3.8%) | 23.4 (+3.7%) |
| 8 | 18.2 | 16.9 (-7.3%) | 17.5 (-4.0%) | 21.9 (+20.4%) | 21.3 (+17.0%) |
| 16 | 17.9 | 13.9 (-22.7%) | 11.9 (-33.4%) | 21.7 (+21.0%) | 19.3 (+7.4%) |
| 22 | 17.6 | 15.5 (-11.9%) | 9.53 (-46.0%) | 20.8 (+18.2%) | 17.2 (-2.3%) |

### TTFT p50 (s)

| Streams | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 1 | 0.206 | 0.214 (+3.9%) | 0.207 (+0.3%) | 0.298 (+44.8%) | 0.281 (+36.3%) |
| 4 | 0.248 | 0.246 (-0.9%) | 0.292 (+17.5%) | 0.333 (+34.2%) | 0.247 (-0.4%) |
| 8 | 0.272 | 0.274 (+0.6%) | 0.276 (+1.5%) | 0.271 (-0.6%) | 0.267 (-1.9%) |
| 16 | 12.9 | 0.324 (-97.5%) | 0.371 (-97.1%) | 0.332 (-97.4%) | 0.388 (-97.0%) |
| 22 | 29.5 | 0.331 (-98.9%) | 0.463 (-98.4%) | 0.323 (-98.9%) | 0.463 (-98.4%) |

### TTFT p95 (s)

| Streams | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 1 | 0.338 | 0.223 (-34.0%) | 0.212 (-37.3%) | 0.552 (+63.0%) | 0.531 (+56.9%) |
| 4 | 0.287 | 0.290 (+0.8%) | 0.316 (+10.0%) | 0.427 (+48.8%) | 0.288 (+0.1%) |
| 8 | 0.276 | 0.282 (+2.2%) | 0.280 (+1.6%) | 0.276 (+0.0%) | 0.273 (-0.9%) |
| 16 | 35.4 | 38.3 (+8.2%) | 0.386 (-98.9%) | 23.6 (-33.4%) | 1.59 (-95.5%) |
| 22 | 63.0 | 41.2 (-34.7%) | 0.479 (-99.2%) | 27.1 (-57.1%) | 0.474 (-99.2%) |

### TPOT (ms)

| Streams | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 1 | 35.3 | 35.0 (-0.9%) | 35.3 (+0.1%) | 40.4 (+14.5%) | 40.3 (+14.1%) |
| 4 | 44.4 | 43.2 (-2.6%) | 42.5 (-4.3%) | 42.8 (-3.7%) | 42.8 (-3.5%) |
| 8 | 54.9 | 59.2 (+7.8%) | 57.2 (+4.1%) | 45.6 (-16.9%) | 46.9 (-14.6%) |
| 16 | 55.8 | 72.1 (+29.3%) | 83.8 (+50.1%) | 46.1 (-17.3%) | 51.9 (-6.9%) |
| 22 | 56.7 | 64.3 (+13.5%) | 105 (+85.2%) | 48.0 (-15.4%) | 58.0 (+2.3%) |

### Peak memory (GiB)

| Streams | dflash-8 (host b) | dflash-mix-12 (host b) | dflash-mix-22 (host b) | eagle-12 (host a) | eagle-22 (host a) |
|---|---|---|---|---|---|
| 1 | 72.3 | 80.1 (+10.7%) | 100.0 (+38.2%) | 78.5 (+8.6%) | 88.0 (+21.7%) |
| 4 | 73.1 | 81.0 (+10.9%) | 100 (+37.0%) | 79.3 (+8.4%) | 89.0 (+21.7%) |
| 8 | 75.0 | 82.2 (+9.6%) | 101 (+35.1%) | 80.4 (+7.3%) | 90.2 (+20.2%) |
| 16 | 75.0 | 83.9 (+11.8%) | 104 (+38.2%) | 82.1 (+9.5%) | 92.5 (+23.3%) |
| 22 | 75.0 | 83.9 (+11.8%) | 107 (+42.3%) | 82.2 (+9.5%) | 95.6 (+27.4%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
