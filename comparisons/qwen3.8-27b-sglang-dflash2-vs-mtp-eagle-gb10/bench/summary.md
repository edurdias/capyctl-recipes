# Qwen3.8-27B on SGLang 0.5.21: DFlash2 vs MTP/EAGLE, one GB10

Measured through CapyCTL with capyctl-bench 0.1.0. Each cell is the median of the runs at that point; percentages compare with DFlash2 (SGLang 0.5.21).

| Series | Model | Measured | Meta |
|---|---|---|---|
| DFlash2 (SGLang 0.5.21) | `qwen38-dflash2` | 2026-10-14 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=d4b86843fccbdfe4a08a65a19403ada6ccbf03e7, model=nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80, drafter=z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, passes=2, interleaved DFlash2, MTP/EAGLE; each a fresh warm start |
| MTP/EAGLE (SGLang 0.5.21) | `qwen38-mtp-eagle` | 2026-10-15 | gpu=GB10, engine=sglang, engine_version=0.5.21, capyctl=d4b86843fccbdfe4a08a65a19403ada6ccbf03e7, model=RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@009632fef96dd349150baa780c984e62e70e91fe, drafter=none (the checkpoint's MTP layer), passes=2, interleaved DFlash2, MTP/EAGLE; each a fresh warm start |

## Context sweep

### Generation (tok/s, higher is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 56.9 | 19.3 (-66.2%) |
| 32k | 28.8 | 18.5 (-35.7%) |
| 128k | 30.4 | 16.9 (-44.4%) |

### Prompt processing (tok/s, higher is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 2,349 | 2,134 (-9.1%) |
| 32k | 1,734 | 1,509 (-13.0%) |
| 128k | 1,051 | 946 (-10.0%) |

### Time to first token (s, lower is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 0.862 | 0.941 (+9.1%) |
| 32k | 18.7 | 21.4 (+14.7%) |
| 128k | 122 | 136 (+11.1%) |

### Time per output token (ms, lower is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 17.6 | 51.9 (+195.7%) |
| 32k | 34.7 | 54.1 (+55.6%) |
| 128k | 32.9 | 59.2 (+79.9%) |

### Total time (s, lower is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 3.11 | 7.55 (+143.1%) |
| 32k | 23.1 | 28.6 (+23.8%) |
| 128k | 127 | 144 (+13.6%) |

### Peak memory (GiB, lower is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 72.5 | 84.4 (+16.4%) |
| 32k | 73.7 | 86.0 (+16.8%) |
| 128k | 73.7 | 86.1 (+16.9%) |

### Output bytes per second (B/s, higher is better)

| Context | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 2k | 298 | 90.4 (-69.7%) |
| 32k | 131 | 76.9 (-41.4%) |
| 128k | 137 | 76.4 (-44.3%) |

## Concurrency

### Aggregate (tok/s)

| Streams | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 1 | 32.4 | 17.7 (-45.4%) |
| 4 | 81.4 | 66.9 (-17.9%) |
| 8 | 123 | 119 (-3.2%) |
| 16 | 124 | 215 (+73.0%) |
| 22 | 122 | 281 (+130.2%) |

### Per stream (tok/s)

| Streams | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 1 | 33.1 | 17.9 (-46.1%) |
| 4 | 22.4 | 17.9 (-20.0%) |
| 8 | 17.6 | 17.2 (-2.3%) |
| 16 | 17.7 | 15.3 (-14.0%) |
| 22 | 17.4 | 14.4 (-17.4%) |

### TTFT p50 (s)

| Streams | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 1 | 0.209 | 0.271 (+29.5%) |
| 4 | 0.226 | 0.272 (+20.2%) |
| 8 | 0.265 | 0.314 (+18.5%) |
| 16 | 9.76 | 0.416 (-95.7%) |
| 22 | 29.3 | 0.513 (-98.2%) |

### TTFT p95 (s)

| Streams | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 1 | 0.498 | 0.501 (+0.6%) |
| 4 | 0.269 | 0.318 (+18.2%) |
| 8 | 0.276 | 0.517 (+87.7%) |
| 16 | 35.5 | 0.425 (-98.8%) |
| 22 | 62.7 | 0.782 (-98.8%) |

### TPOT (ms)

| Streams | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 1 | 30.2 | 56.0 (+85.5%) |
| 4 | 44.6 | 55.8 (+25.0%) |
| 8 | 56.8 | 58.2 (+2.4%) |
| 16 | 56.4 | 65.5 (+16.3%) |
| 22 | 57.4 | 69.4 (+21.0%) |

### Peak memory (GiB)

| Streams | DFlash2 (SGLang 0.5.21) | MTP/EAGLE (SGLang 0.5.21) |
|---|---|---|
| 1 | 72.3 | 85.6 (+18.5%) |
| 4 | 72.7 | 85.9 (+18.1%) |
| 8 | 73.8 | 87.3 (+18.4%) |
| 16 | 73.8 | 89.0 (+20.7%) |
| 22 | 73.8 | 92.0 (+24.7%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
