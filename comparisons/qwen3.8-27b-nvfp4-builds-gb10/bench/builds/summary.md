# Qwen3.8-27B NVFP4 builds, no drafts

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with CapyCTL NVFP4.

| Series | Model | Measured | Meta |
|---|---|---|---|
| CapyCTL NVFP4 | `q38-sg-full` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-full, gpu=GB10 |
| NVIDIA NVFP4 | `q38-sg-nv` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-nv, gpu=GB10 |
| gittensor NVFP4 | `q38-sg-gt` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-gt, gpu=GB10 |
| QUASAR NVFP4 | `q38-sg-qs` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-qs, gpu=GB10 |
| RedHatAI INT4 | `q38-sg-rh` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-rh, gpu=GB10 |
| unsloth NVFP4 | `q38-sg-un` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-un, gpu=GB10 |
| Qwen BF16 | `q38-sg-bf16` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-bf16, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 14.9 | 12.5 (-16.2%) | 14.8 (-0.5%) | 12.2 (-17.7%) | 13.7 (-8.1%) | 11.1 (-25.2%) | 4.44 (-70.1%) |
| 32k | 14.0 | 11.8 (-15.3%) | 13.9 (-0.5%) | 11.6 (-17.0%) | 12.9 (-7.6%) | 10.6 (-24.0%) | 4.36 (-68.8%) |

### Prompt processing (tok/s, higher is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 2,918 | 2,634 (-9.7%) | 2,872 (-1.6%) | 2,666 (-8.6%) | 1,197 (-59.0%) | 2,367 (-18.9%) | 1,327 (-54.5%) |
| 32k | 1,788 | 1,738 (-2.8%) | 1,764 (-1.3%) | 1,469 (-17.9%) | 1,054 (-41.0%) | 1,158 (-35.2%) | 1,206 (-32.5%) |

### Time to first token (s, lower is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 0.694 | 0.768 (+10.8%) | 0.705 (+1.6%) | 0.759 (+9.4%) | 1.69 (+143.9%) | 0.855 (+23.3%) | 1.53 (+119.9%) |
| 32k | 18.1 | 18.6 (+2.9%) | 18.3 (+1.4%) | 22.0 (+21.7%) | 30.6 (+69.6%) | 27.9 (+54.4%) | 26.8 (+48.4%) |

### Time per output token (ms, lower is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 67.3 | 80.3 (+19.3%) | 67.7 (+0.5%) | 81.8 (+21.5%) | 73.2 (+8.8%) | 90.0 (+33.7%) | 225 (+234.5%) |
| 32k | 71.6 | 84.6 (+18.1%) | 72.0 (+0.5%) | 86.3 (+20.5%) | 77.5 (+8.3%) | 94.2 (+31.6%) | 229 (+220.3%) |

### Total time (s, lower is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 9.24 | 11.0 (+18.6%) | 9.30 (+0.6%) | 11.1 (+20.6%) | 11.0 (+19.0%) | 12.3 (+32.9%) | 30.1 (+225.9%) |
| 32k | 27.2 | 29.3 (+8.0%) | 27.5 (+1.1%) | 33.0 (+21.3%) | 40.5 (+49.1%) | 39.9 (+46.7%) | 55.9 (+105.9%) |

### Peak memory (GiB, lower is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 53.9 | 57.5 (+6.7%) | 52.9 (-1.8%) | 54.3 (+0.8%) | 52.9 (-1.8%) | 57.8 (+7.3%) | 85.9 (+59.3%) |
| 32k | 54.7 | 58.0 (+6.0%) | 53.7 (-1.8%) | 55.3 (+1.0%) | 53.9 (-1.5%) | 58.8 (+7.5%) | 86.9 (+58.7%) |

### Output bytes per second (B/s, higher is better)

| Context | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 71.9 | 67.8 (-5.7%) | 71.3 (-0.8%) | 60.3 (-16.1%) | 67.2 (-6.5%) | 55.7 (-22.5%) | 21.5 (-70.1%) |
| 32k | 63.0 | 53.5 (-15.1%) | 68.0 (+8.0%) | 50.2 (-20.2%) | 57.5 (-8.7%) | 47.9 (-23.9%) | 20.9 (-66.7%) |

## Concurrency

### Aggregate (tok/s)

| Streams | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 14.9 | 12.5 (-16.1%) | 14.8 (-0.5%) | 12.4 (-16.7%) | 13.7 (-8.2%) | 11.1 (-25.2%) | 4.45 (-70.1%) |
| 4 | 55.2 | 46.2 (-16.3%) | 54.8 (-0.6%) | 48.8 (-11.6%) | 52.9 (-4.2%) | 42.0 (-23.8%) | 17.2 (-68.8%) |
| 8 | 101 | 85.9 (-14.9%) | 100 (-0.6%) | 90.9 (-9.9%) | 99.3 (-1.7%) | 79.0 (-21.7%) | 33.4 (-66.9%) |

### Per stream (tok/s)

| Streams | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 14.9 | 12.5 (-16.1%) | 14.8 (-0.5%) | 12.4 (-16.7%) | 13.7 (-8.1%) | 11.1 (-25.2%) | 4.45 (-70.1%) |
| 4 | 13.8 | 11.6 (-16.3%) | 13.8 (-0.6%) | 12.2 (-11.5%) | 13.3 (-4.0%) | 10.5 (-23.8%) | 4.32 (-68.8%) |
| 8 | 12.7 | 10.8 (-15.0%) | 12.6 (-0.6%) | 11.4 (-9.9%) | 12.5 (-1.4%) | 9.91 (-21.7%) | 4.19 (-67.0%) |

### TTFT p50 (s)

| Streams | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 0.102 | 0.118 (+15.7%) | 0.103 (+0.7%) | 0.125 (+22.4%) | 0.169 (+64.9%) | 0.136 (+33.3%) | 0.301 (+194.5%) |
| 4 | 0.194 | 0.225 (+16.0%) | 0.199 (+2.9%) | 0.223 (+15.3%) | 0.266 (+37.5%) | 0.250 (+29.0%) | 0.558 (+188.4%) |
| 8 | 0.234 | 0.269 (+15.0%) | 0.236 (+1.1%) | 0.265 (+13.5%) | 0.354 (+51.3%) | 0.301 (+28.6%) | 0.622 (+166.2%) |

### TTFT p95 (s)

| Streams | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 0.104 | 0.122 (+16.7%) | 0.104 (-0.2%) | 0.128 (+22.5%) | 0.172 (+65.5%) | 0.138 (+32.7%) | 0.303 (+191.0%) |
| 4 | 0.215 | 0.341 (+58.3%) | 0.211 (-2.0%) | 0.243 (+12.7%) | 0.306 (+41.9%) | 0.276 (+28.0%) | 0.591 (+174.5%) |
| 8 | 0.240 | 0.275 (+14.7%) | 0.242 (+0.9%) | 0.338 (+40.6%) | 0.360 (+50.0%) | 0.308 (+28.3%) | 0.642 (+167.4%) |

### TPOT (ms)

| Streams | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 67.1 | 80.0 (+19.2%) | 67.5 (+0.5%) | 80.6 (+20.1%) | 73.0 (+8.8%) | 89.7 (+33.7%) | 225 (+234.7%) |
| 4 | 72.3 | 86.3 (+19.5%) | 72.7 (+0.6%) | 81.7 (+13.0%) | 75.3 (+4.2%) | 94.8 (+31.3%) | 231 (+220.2%) |
| 8 | 79.0 | 92.8 (+17.6%) | 79.4 (+0.6%) | 87.6 (+11.0%) | 80.0 (+1.4%) | 101 (+27.8%) | 239 (+202.6%) |

### Peak memory (GiB)

| Streams | CapyCTL NVFP4 | NVIDIA NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 54.7 | 57.8 (+5.7%) | 53.7 (-1.8%) | 55.2 (+1.0%) | 53.9 (-1.4%) | 58.8 (+7.6%) | 86.9 (+58.9%) |
| 4 | 55.1 | 58.3 (+5.7%) | 54.5 (-1.2%) | 56.0 (+1.5%) | 54.4 (-1.4%) | 59.3 (+7.5%) | 87.4 (+58.5%) |
| 8 | 56.2 | 59.3 (+5.6%) | 55.5 (-1.2%) | 56.9 (+1.4%) | 55.3 (-1.5%) | 60.3 (+7.4%) | 87.6 (+56.0%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
