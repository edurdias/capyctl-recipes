# Qwen3.8-27B with DFlash2 drafts

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with NVIDIA, vLLM.

| Series | Model | Measured | Meta |
|---|---|---|---|
| NVIDIA, vLLM | `q38-vl-nv-df-mbf16` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=vl-nv-df-mbf16, gpu=GB10 |
| CapyCTL, vLLM (recipe) | `q38-vl-full-df-mbf16` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=vl-full-df-mbf16, gpu=GB10 |
| NVIDIA, SGLang | `q38-sg-nv-df-tuned` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-nv-df-tuned, gpu=GB10 |
| CapyCTL, SGLang (recipe) | `q38-sg-full-df-tuned` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned, gpu=GB10 |
| CapyCTL, SGLang, 3 decode steps | `q38-sg-full-df-cds3` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-full-df-cds3, gpu=GB10 |
| CapyCTL, SGLang, + compile | `q38-sg-full-df-tuned-compile` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned-compile, gpu=GB10 |
| CapyCTL, SGLang, 6 drafts | `q38-sg-full-df-tuned-k6` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned-k6, gpu=GB10 |
| CapyCTL, SGLang, 12 drafts | `q38-sg-full-df-tuned-k12` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned-k12, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 47.7 | 50.6 (+6.2%) | 47.9 (+0.4%) | 45.9 (-3.8%) | 45.7 (-4.1%) | 46.3 (-2.8%) | 39.5 (-17.2%) | 56.4 (+18.3%) |

### Prompt processing (tok/s, higher is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 2,321 | 2,624 (+13.0%) | 2,463 (+6.1%) | 2,720 (+17.2%) | 2,731 (+17.6%) | 2,573 (+10.9%) | 2,836 (+22.2%) | 2,561 (+10.3%) |

### Time to first token (s, lower is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 0.872 | 0.771 (-11.5%) | 0.824 (-5.5%) | 0.746 (-14.4%) | 0.744 (-14.7%) | 0.787 (-9.8%) | 0.714 (-18.1%) | 0.790 (-9.4%) |

### Time per output token (ms, lower is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 21.8 | 19.8 (-9.0%) | 21.3 (-2.3%) | 21.8 (+0.1%) | 21.9 (+0.5%) | 22.2 (+1.9%) | 25.7 (+17.9%) | 19.4 (-10.9%) |

### Total time (s, lower is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 3.64 | 3.29 (-9.6%) | 3.53 (-3.0%) | 3.51 (-3.4%) | 3.52 (-3.2%) | 3.61 (-0.9%) | 3.97 (+9.2%) | 3.25 (-10.6%) |

### Peak memory (GiB, lower is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 50.8 | 47.6 (-6.3%) | 64.4 (+26.7%) | 62.3 (+22.5%) | 62.3 (+22.6%) | 63.5 (+24.9%) | 60.2 (+18.5%) | 65.1 (+28.1%) |

### Output bytes per second (B/s, higher is better)

| Context | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 238 | 250 (+4.9%) | 218 (-8.5%) | 216 (-9.2%) | 216 (-9.5%) | 222 (-6.6%) | 188 (-21.0%) | 287 (+20.4%) |

## Concurrency

### Aggregate (tok/s)

| Streams | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 29.0 | 35.7 (+23.1%) | 30.4 (+4.8%) | 35.2 (+21.4%) | 35.1 (+21.2%) | 33.2 (+14.5%) | 34.4 (+18.7%) | 35.0 (+20.7%) |
| 4 | 96.5 | 106 (+10.3%) | 93.3 (-3.3%) | 109 (+12.7%) | 105 (+9.0%) | 98.7 (+2.3%) | 103 (+6.4%) | 73.8 (-23.5%) |
| 8 | 160 | 181 (+13.2%) | 150 (-6.4%) | 166 (+3.5%) | 168 (+4.8%) | 169 (+5.6%) | 139 (-13.4%) | 155 (-3.5%) |

### Per stream (tok/s)

| Streams | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 29.3 | 36.1 (+23.2%) | 30.6 (+4.5%) | 35.6 (+21.4%) | 35.5 (+21.1%) | 33.4 (+13.8%) | 34.8 (+18.6%) | 35.3 (+20.6%) |
| 4 | 27.7 | 29.6 (+7.0%) | 26.0 (-5.9%) | 29.4 (+6.1%) | 29.4 (+6.2%) | 29.3 (+5.9%) | 29.0 (+4.9%) | 19.9 (-28.2%) |
| 8 | 22.6 | 25.9 (+14.5%) | 21.6 (-4.6%) | 23.0 (+1.4%) | 23.4 (+3.3%) | 23.3 (+2.9%) | 18.8 (-17.1%) | 21.4 (-5.3%) |

### TTFT p50 (s)

| Streams | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.237 | 0.204 (-14.0%) | 0.122 (-48.5%) | 0.185 (-21.7%) | 0.184 (-22.2%) | 0.140 (-40.7%) | 0.185 (-21.7%) | 0.187 (-21.0%) |
| 4 | 0.449 | 0.411 (-8.5%) | 0.227 (-49.5%) | 0.199 (-55.6%) | 0.198 (-55.9%) | 0.195 (-56.6%) | 0.196 (-56.3%) | 0.199 (-55.8%) |
| 8 | 0.566 | 0.522 (-7.9%) | 0.260 (-54.1%) | 0.227 (-59.9%) | 0.231 (-59.3%) | 0.252 (-55.5%) | 0.228 (-59.7%) | 0.233 (-58.9%) |

### TTFT p95 (s)

| Streams | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.239 | 0.204 (-14.4%) | 0.214 (-10.2%) | 0.188 (-21.4%) | 0.188 (-21.4%) | 0.184 (-22.7%) | 0.188 (-21.2%) | 0.190 (-20.3%) |
| 4 | 0.455 | 0.424 (-6.8%) | 0.249 (-45.2%) | 0.232 (-49.0%) | 0.239 (-47.4%) | 0.212 (-53.4%) | 0.235 (-48.2%) | 0.239 (-47.4%) |
| 8 | 0.572 | 0.527 (-7.8%) | 0.268 (-53.2%) | 0.309 (-46.1%) | 0.261 (-54.4%) | 0.287 (-49.9%) | 0.242 (-57.7%) | 0.298 (-47.9%) |

### TPOT (ms)

| Streams | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 34.1 | 27.7 (-18.8%) | 32.7 (-4.2%) | 28.1 (-17.6%) | 28.2 (-17.4%) | 30.0 (-12.1%) | 28.8 (-15.6%) | 28.4 (-16.9%) |
| 4 | 36.1 | 33.7 (-6.6%) | 38.4 (+6.3%) | 34.0 (-5.8%) | 34.0 (-5.8%) | 34.1 (-5.6%) | 34.4 (-4.7%) | 50.3 (+39.3%) |
| 8 | 44.2 | 38.6 (-12.7%) | 46.3 (+4.8%) | 43.5 (-1.4%) | 42.8 (-3.2%) | 42.9 (-2.8%) | 53.2 (+20.6%) | 46.6 (+5.6%) |

### Peak memory (GiB)

| Streams | NVIDIA, vLLM | CapyCTL, vLLM (recipe) | NVIDIA, SGLang | CapyCTL, SGLang (recipe) | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, + compile | CapyCTL, SGLang, 6 drafts | CapyCTL, SGLang, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 50.8 | 47.5 (-6.5%) | 64.4 (+26.7%) | 62.2 (+22.5%) | 62.3 (+22.6%) | 62.5 (+22.9%) | 60.1 (+18.3%) | 64.9 (+27.8%) |
| 4 | 50.8 | 47.5 (-6.5%) | 64.7 (+27.4%) | 62.5 (+23.0%) | 62.6 (+23.1%) | 63.5 (+24.9%) | 60.3 (+18.7%) | 65.1 (+28.2%) |
| 8 | 50.8 | 47.5 (-6.5%) | 65.2 (+28.4%) | 62.7 (+23.4%) | 63.1 (+24.2%) | 63.5 (+25.0%) | 60.8 (+19.6%) | 66.1 (+30.1%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
