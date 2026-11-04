# Qwen3.8-27B with DFlash2 drafts

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with CapyCTL, vLLM, bf16 state (recipe).

| Series | Model | Measured | Meta |
|---|---|---|---|
| CapyCTL, vLLM, bf16 state (recipe) | `q38-vl-full-df-mbf16` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=vl-full-df-mbf16, gpu=GB10 |
| NVIDIA, vLLM, bf16 state | `q38-vl-nv-df-mbf16` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=vl-nv-df-mbf16, gpu=GB10 |
| CapyCTL, SGLang, tuned | `q38-sg-full-df-tuned` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned, gpu=GB10 |
| NVIDIA, SGLang, tuned | `q38-sg-nv-df-tuned` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-nv-df-tuned, gpu=GB10 |
| CapyCTL, SGLang, 3 decode steps | `q38-sg-full-df-cds3` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-full-df-cds3, gpu=GB10 |
| CapyCTL, SGLang, tuned + compile | `q38-sg-full-df-tuned-compile` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned-compile, gpu=GB10 |
| CapyCTL, SGLang, tuned, 6 drafts | `q38-sg-full-df-tuned-k6` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned-k6, gpu=GB10 |
| CapyCTL, SGLang, tuned, 12 drafts | `q38-sg-full-df-tuned-k12` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=sg-full-df-tuned-k12, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 50.6 | 47.7 (-5.8%) | 45.9 (-9.4%) | 47.9 (-5.4%) | 45.7 (-9.7%) | 46.3 (-8.5%) | 39.5 (-22.0%) | 56.4 (+11.4%) |

### Prompt processing (tok/s, higher is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 2,624 | 2,321 (-11.5%) | 2,720 (+3.7%) | 2,463 (-6.1%) | 2,731 (+4.1%) | 2,573 (-1.9%) | 2,836 (+8.1%) | 2,561 (-2.4%) |

### Time to first token (s, lower is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 0.771 | 0.872 (+13.0%) | 0.746 (-3.3%) | 0.824 (+6.9%) | 0.744 (-3.5%) | 0.787 (+2.0%) | 0.714 (-7.5%) | 0.790 (+2.5%) |

### Time per output token (ms, lower is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 19.8 | 21.8 (+9.9%) | 21.8 (+10.0%) | 21.3 (+7.4%) | 21.9 (+10.4%) | 22.2 (+12.0%) | 25.7 (+29.6%) | 19.4 (-2.1%) |

### Total time (s, lower is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 3.29 | 3.64 (+10.7%) | 3.51 (+6.9%) | 3.53 (+7.3%) | 3.52 (+7.1%) | 3.61 (+9.7%) | 3.97 (+20.9%) | 3.25 (-1.0%) |

### Peak memory (GiB, lower is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 47.6 | 50.8 (+6.7%) | 62.3 (+30.8%) | 64.4 (+35.3%) | 62.3 (+30.9%) | 63.5 (+33.3%) | 60.2 (+26.5%) | 65.1 (+36.7%) |

### Output bytes per second (B/s, higher is better)

| Context | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 2k | 250 | 238 (-4.7%) | 216 (-13.4%) | 218 (-12.8%) | 216 (-13.7%) | 222 (-10.9%) | 188 (-24.7%) | 287 (+14.8%) |

## Concurrency

### Aggregate (tok/s)

| Streams | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 35.7 | 29.0 (-18.8%) | 35.2 (-1.3%) | 30.4 (-14.9%) | 35.1 (-1.6%) | 33.2 (-6.9%) | 34.4 (-3.6%) | 35.0 (-2.0%) |
| 4 | 106 | 96.5 (-9.3%) | 109 (+2.2%) | 93.3 (-12.3%) | 105 (-1.2%) | 98.7 (-7.3%) | 103 (-3.5%) | 73.8 (-30.6%) |
| 8 | 181 | 160 (-11.7%) | 166 (-8.6%) | 150 (-17.3%) | 168 (-7.4%) | 169 (-6.7%) | 139 (-23.5%) | 155 (-14.7%) |

### Per stream (tok/s)

| Streams | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 36.1 | 29.3 (-18.8%) | 35.6 (-1.5%) | 30.6 (-15.2%) | 35.5 (-1.7%) | 33.4 (-7.6%) | 34.8 (-3.7%) | 35.3 (-2.1%) |
| 4 | 29.6 | 27.7 (-6.6%) | 29.4 (-0.9%) | 26.0 (-12.1%) | 29.4 (-0.8%) | 29.3 (-1.1%) | 29.0 (-2.0%) | 19.9 (-32.9%) |
| 8 | 25.9 | 22.6 (-12.7%) | 23.0 (-11.4%) | 21.6 (-16.7%) | 23.4 (-9.8%) | 23.3 (-10.1%) | 18.8 (-27.6%) | 21.4 (-17.3%) |

### TTFT p50 (s)

| Streams | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.204 | 0.237 (+16.3%) | 0.185 (-8.9%) | 0.122 (-40.1%) | 0.184 (-9.5%) | 0.140 (-31.0%) | 0.185 (-8.9%) | 0.187 (-8.2%) |
| 4 | 0.411 | 0.449 (+9.3%) | 0.199 (-51.5%) | 0.227 (-44.8%) | 0.198 (-51.8%) | 0.195 (-52.6%) | 0.196 (-52.3%) | 0.199 (-51.7%) |
| 8 | 0.522 | 0.566 (+8.5%) | 0.227 (-56.4%) | 0.260 (-50.2%) | 0.231 (-55.8%) | 0.252 (-51.7%) | 0.228 (-56.3%) | 0.233 (-55.4%) |

### TTFT p95 (s)

| Streams | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.204 | 0.239 (+16.8%) | 0.188 (-8.2%) | 0.214 (+4.9%) | 0.188 (-8.2%) | 0.184 (-9.7%) | 0.188 (-8.0%) | 0.190 (-6.9%) |
| 4 | 0.424 | 0.455 (+7.3%) | 0.232 (-45.3%) | 0.249 (-41.2%) | 0.239 (-43.6%) | 0.212 (-50.1%) | 0.235 (-44.4%) | 0.239 (-43.6%) |
| 8 | 0.527 | 0.572 (+8.5%) | 0.309 (-41.5%) | 0.268 (-49.3%) | 0.261 (-50.6%) | 0.287 (-45.7%) | 0.242 (-54.1%) | 0.298 (-43.5%) |

### TPOT (ms)

| Streams | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 27.7 | 34.1 (+23.1%) | 28.1 (+1.5%) | 32.7 (+17.9%) | 28.2 (+1.7%) | 30.0 (+8.3%) | 28.8 (+4.0%) | 28.4 (+2.3%) |
| 4 | 33.7 | 36.1 (+7.0%) | 34.0 (+0.9%) | 38.4 (+13.8%) | 34.0 (+0.8%) | 34.1 (+1.1%) | 34.4 (+2.1%) | 50.3 (+49.1%) |
| 8 | 38.6 | 44.2 (+14.5%) | 43.5 (+12.9%) | 46.3 (+20.0%) | 42.8 (+10.9%) | 42.9 (+11.3%) | 53.2 (+38.0%) | 46.6 (+20.9%) |

### Peak memory (GiB)

| Streams | CapyCTL, vLLM, bf16 state (recipe) | NVIDIA, vLLM, bf16 state | CapyCTL, SGLang, tuned | NVIDIA, SGLang, tuned | CapyCTL, SGLang, 3 decode steps | CapyCTL, SGLang, tuned + compile | CapyCTL, SGLang, tuned, 6 drafts | CapyCTL, SGLang, tuned, 12 drafts |
|---|---|---|---|---|---|---|---|---|
| 1 | 47.5 | 50.8 (+6.9%) | 62.2 (+31.0%) | 64.4 (+35.5%) | 62.3 (+31.1%) | 62.5 (+31.5%) | 60.1 (+26.4%) | 64.9 (+36.7%) |
| 4 | 47.5 | 50.8 (+7.0%) | 62.5 (+31.6%) | 64.7 (+36.3%) | 62.6 (+31.7%) | 63.5 (+33.7%) | 60.3 (+26.9%) | 65.1 (+37.1%) |
| 8 | 47.5 | 50.8 (+7.0%) | 62.7 (+32.1%) | 65.2 (+37.4%) | 63.1 (+32.9%) | 63.5 (+33.8%) | 60.8 (+28.0%) | 66.1 (+39.2%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
