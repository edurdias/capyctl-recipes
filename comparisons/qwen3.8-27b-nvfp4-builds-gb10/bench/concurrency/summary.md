# Qwen3.8-27B recipes past 8 streams

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with vLLM, 16 GiB KV (recipe).

| Series | Model | Measured | Meta |
|---|---|---|---|
| vLLM, 16 GiB KV (recipe) | `q38-rcp-vl-hi` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=rcp-vl-hi, gpu=GB10 |
| vLLM, 32 GiB KV | `q38-rcp-vl-hi` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=rcp-vl-hi, gpu=GB10 |
| SGLang, 94 GiB | `q38-rcp-sg-hi` | 2026-11-04 | capyctl=capyctl 0.1.10, variant=rcp-sg-hi, gpu=GB10 |

## Concurrency

### Aggregate (tok/s)

| Streams | vLLM, 16 GiB KV (recipe) | vLLM, 32 GiB KV | SGLang, 94 GiB |
|---|---|---|---|
| 8 | 183 | – | 168 (-8.5%) |
| 12 | 240 | – | 207 (-13.5%) |
| 16 | 267 | 280 (+4.8%) | 250 (-6.5%) |
| 24 | 254 | 354 (+39.3%) | 303 (+19.5%) |
| 32 | 300 | 390 (+29.8%) | 329 (+9.6%) |
| 40 | – | 338 | – |
| 48 | 292 | 359 (+22.9%) | 307 (+5.1%) |

### Per stream (tok/s)

| Streams | vLLM, 16 GiB KV (recipe) | vLLM, 32 GiB KV | SGLang, 94 GiB |
|---|---|---|---|
| 8 | 26.5 | – | 23.1 (-12.7%) |
| 12 | 23.5 | – | 19.6 (-16.6%) |
| 16 | 20.3 | 20.5 (+0.6%) | 17.5 (-13.8%) |
| 24 | 17.3 | 17.2 (-0.9%) | 13.6 (-21.5%) |
| 32 | 18.6 | 13.9 (-25.2%) | 11.3 (-39.5%) |
| 40 | – | 14.0 | – |
| 48 | 17.4 | 14.1 (-18.9%) | 11.7 (-32.5%) |

### TTFT p50 (s)

| Streams | vLLM, 16 GiB KV (recipe) | vLLM, 32 GiB KV | SGLang, 94 GiB |
|---|---|---|---|
| 8 | 0.504 | – | 0.233 (-53.8%) |
| 12 | 0.750 | – | 0.299 (-60.1%) |
| 16 | 0.795 | 0.793 (-0.3%) | 0.316 (-60.2%) |
| 24 | 1.03 | 1.06 (+2.2%) | 0.415 (-59.9%) |
| 32 | 1.04 | 1.32 (+26.8%) | 0.516 (-50.3%) |
| 40 | – | 1.30 | – |
| 48 | 28.8 | 1.53 (-94.7%) | 0.520 (-98.2%) |

### TTFT p95 (s)

| Streams | vLLM, 16 GiB KV (recipe) | vLLM, 32 GiB KV | SGLang, 94 GiB |
|---|---|---|---|
| 8 | 0.506 | – | 0.271 (-46.4%) |
| 12 | 0.757 | – | 0.312 (-58.8%) |
| 16 | 0.807 | 0.798 (-1.1%) | 0.328 (-59.3%) |
| 24 | 28.6 | 1.26 (-95.6%) | 0.477 (-98.3%) |
| 32 | 30.6 | 1.55 (-94.9%) | 0.537 (-98.2%) |
| 40 | – | 36.5 | – |
| 48 | 57.3 | 40.1 (-30.1%) | 47.6 (-16.9%) |

### TPOT (ms)

| Streams | vLLM, 16 GiB KV (recipe) | vLLM, 32 GiB KV | SGLang, 94 GiB |
|---|---|---|---|
| 8 | 37.8 | – | 43.2 (+14.5%) |
| 12 | 42.5 | – | 51.0 (+19.9%) |
| 16 | 49.2 | 48.9 (-0.6%) | 57.1 (+16.0%) |
| 24 | 57.8 | 58.3 (+0.9%) | 73.6 (+27.4%) |
| 32 | 53.7 | 71.7 (+33.6%) | 88.7 (+65.2%) |
| 40 | – | 71.4 | – |
| 48 | 57.6 | 71.0 (+23.2%) | 85.4 (+48.2%) |

### Peak memory (GiB)

| Streams | vLLM, 16 GiB KV (recipe) | vLLM, 32 GiB KV | SGLang, 94 GiB |
|---|---|---|---|
| 8 | 52.7 | – | 100.0 (+89.7%) |
| 12 | 52.7 | – | 101 (+91.2%) |
| 16 | 52.7 | 66.7 (+26.5%) | 102 (+93.4%) |
| 24 | 52.7 | 66.8 (+26.7%) | 105 (+99.7%) |
| 32 | 52.7 | 66.8 (+26.7%) | 108 (+104.0%) |
| 40 | – | 66.8 | – |
| 48 | 52.7 | 66.8 (+26.7%) | 108 (+104.0%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
