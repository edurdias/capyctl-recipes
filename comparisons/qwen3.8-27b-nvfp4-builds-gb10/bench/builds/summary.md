# Qwen3.8-27B NVFP4 builds, no drafts

Measured through CapyCTL with capyctl-bench 0.3.1. Each cell is the median of the runs at that point; percentages compare with NVIDIA NVFP4.

| Series | Model | Measured | Meta |
|---|---|---|---|
| NVIDIA NVFP4 | `q38-sg-nv` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-nv, gpu=GB10 |
| CapyCTL NVFP4 | `q38-sg-full` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-full, gpu=GB10 |
| gittensor NVFP4 | `q38-sg-gt` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-gt, gpu=GB10 |
| QUASAR NVFP4 | `q38-sg-qs` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-qs, gpu=GB10 |
| RedHatAI INT4 | `q38-sg-rh` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-rh, gpu=GB10 |
| unsloth NVFP4 | `q38-sg-un` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-un, gpu=GB10 |
| Qwen BF16 | `q38-sg-bf16` | 2026-11-03 | capyctl=capyctl 0.1.10, variant=sg-bf16, gpu=GB10 |

## Context sweep

### Generation (tok/s, higher is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 12.5 | 14.9 (+19.3%) | 14.8 (+18.7%) | 12.2 (-1.8%) | 13.7 (+9.6%) | 11.1 (-10.8%) | 4.44 (-64.3%) |
| 32k | 11.8 | 14.0 (+18.1%) | 13.9 (+17.5%) | 11.6 (-2.0%) | 12.9 (+9.1%) | 10.6 (-10.2%) | 4.36 (-63.1%) |

### Prompt processing (tok/s, higher is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 2,634 | 2,918 (+10.8%) | 2,872 (+9.0%) | 2,666 (+1.2%) | 1,197 (-54.6%) | 2,367 (-10.1%) | 1,327 (-49.6%) |
| 32k | 1,738 | 1,788 (+2.9%) | 1,764 (+1.5%) | 1,469 (-15.5%) | 1,054 (-39.3%) | 1,158 (-33.4%) | 1,206 (-30.6%) |

### Time to first token (s, lower is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 0.768 | 0.694 (-9.7%) | 0.705 (-8.3%) | 0.759 (-1.2%) | 1.69 (+120.1%) | 0.855 (+11.3%) | 1.53 (+98.5%) |
| 32k | 18.6 | 18.1 (-2.8%) | 18.3 (-1.5%) | 22.0 (+18.3%) | 30.6 (+64.9%) | 27.9 (+50.1%) | 26.8 (+44.3%) |

### Time per output token (ms, lower is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 80.3 | 67.3 (-16.2%) | 67.7 (-15.7%) | 81.8 (+1.8%) | 73.2 (-8.8%) | 90.0 (+12.1%) | 225 (+180.4%) |
| 32k | 84.6 | 71.6 (-15.3%) | 72.0 (-14.9%) | 86.3 (+2.0%) | 77.5 (-8.3%) | 94.2 (+11.4%) | 229 (+171.2%) |

### Total time (s, lower is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 11.0 | 9.24 (-15.7%) | 9.30 (-15.2%) | 11.1 (+1.6%) | 11.0 (+0.3%) | 12.3 (+12.0%) | 30.1 (+174.7%) |
| 32k | 29.3 | 27.2 (-7.4%) | 27.5 (-6.4%) | 33.0 (+12.4%) | 40.5 (+38.1%) | 39.9 (+35.9%) | 55.9 (+90.7%) |

### Peak memory (GiB, lower is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 57.5 | 53.9 (-6.2%) | 52.9 (-7.9%) | 54.3 (-5.5%) | 52.9 (-8.0%) | 57.8 (+0.6%) | 85.9 (+49.4%) |
| 32k | 58.0 | 54.7 (-5.6%) | 53.7 (-7.3%) | 55.3 (-4.6%) | 53.9 (-7.0%) | 58.8 (+1.5%) | 86.9 (+49.8%) |

### Output bytes per second (B/s, higher is better)

| Context | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 2k | 67.8 | 71.9 (+6.1%) | 71.3 (+5.3%) | 60.3 (-11.0%) | 67.2 (-0.8%) | 55.7 (-17.7%) | 21.5 (-68.3%) |
| 32k | 53.5 | 63.0 (+17.8%) | 68.0 (+27.2%) | 50.2 (-6.0%) | 57.5 (+7.6%) | 47.9 (-10.4%) | 20.9 (-60.8%) |

## Concurrency

### Aggregate (tok/s)

| Streams | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 12.5 | 14.9 (+19.2%) | 14.8 (+18.6%) | 12.4 (-0.7%) | 13.7 (+9.5%) | 11.1 (-10.8%) | 4.45 (-64.4%) |
| 4 | 46.2 | 55.2 (+19.5%) | 54.8 (+18.7%) | 48.8 (+5.6%) | 52.9 (+14.4%) | 42.0 (-9.0%) | 17.2 (-62.7%) |
| 8 | 85.9 | 101 (+17.6%) | 100 (+16.8%) | 90.9 (+5.9%) | 99.3 (+15.6%) | 79.0 (-8.0%) | 33.4 (-61.1%) |

### Per stream (tok/s)

| Streams | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 12.5 | 14.9 (+19.2%) | 14.8 (+18.6%) | 12.4 (-0.7%) | 13.7 (+9.6%) | 11.1 (-10.8%) | 4.45 (-64.4%) |
| 4 | 11.6 | 13.8 (+19.5%) | 13.8 (+18.8%) | 12.2 (+5.7%) | 13.3 (+14.6%) | 10.5 (-9.0%) | 4.32 (-62.7%) |
| 8 | 10.8 | 12.7 (+17.6%) | 12.6 (+16.9%) | 11.4 (+5.9%) | 12.5 (+16.0%) | 9.91 (-8.0%) | 4.19 (-61.1%) |

### TTFT p50 (s)

| Streams | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 0.118 | 0.102 (-13.5%) | 0.103 (-12.9%) | 0.125 (+5.8%) | 0.169 (+42.6%) | 0.136 (+15.3%) | 0.301 (+154.6%) |
| 4 | 0.225 | 0.194 (-13.8%) | 0.199 (-11.3%) | 0.223 (-0.6%) | 0.266 (+18.5%) | 0.250 (+11.2%) | 0.558 (+148.6%) |
| 8 | 0.269 | 0.234 (-13.0%) | 0.236 (-12.1%) | 0.265 (-1.3%) | 0.354 (+31.6%) | 0.301 (+11.8%) | 0.622 (+131.5%) |

### TTFT p95 (s)

| Streams | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 0.122 | 0.104 (-14.3%) | 0.104 (-14.5%) | 0.128 (+5.0%) | 0.172 (+41.8%) | 0.138 (+13.7%) | 0.303 (+149.3%) |
| 4 | 0.341 | 0.215 (-36.8%) | 0.211 (-38.1%) | 0.243 (-28.8%) | 0.306 (-10.4%) | 0.276 (-19.1%) | 0.591 (+73.4%) |
| 8 | 0.275 | 0.240 (-12.8%) | 0.242 (-12.0%) | 0.338 (+22.6%) | 0.360 (+30.8%) | 0.308 (+11.9%) | 0.642 (+133.1%) |

### TPOT (ms)

| Streams | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 80.0 | 67.1 (-16.1%) | 67.5 (-15.7%) | 80.6 (+0.7%) | 73.0 (-8.8%) | 89.7 (+12.1%) | 225 (+180.7%) |
| 4 | 86.3 | 72.3 (-16.3%) | 72.7 (-15.8%) | 81.7 (-5.4%) | 75.3 (-12.8%) | 94.8 (+9.9%) | 231 (+168.0%) |
| 8 | 92.8 | 79.0 (-15.0%) | 79.4 (-14.4%) | 87.6 (-5.6%) | 80.0 (-13.8%) | 101 (+8.7%) | 239 (+157.4%) |

### Peak memory (GiB)

| Streams | NVIDIA NVFP4 | CapyCTL NVFP4 | gittensor NVFP4 | QUASAR NVFP4 | RedHatAI INT4 | unsloth NVFP4 | Qwen BF16 |
|---|---|---|---|---|---|---|---|
| 1 | 57.8 | 54.7 (-5.4%) | 53.7 (-7.1%) | 55.2 (-4.5%) | 53.9 (-6.8%) | 58.8 (+1.7%) | 86.9 (+50.3%) |
| 4 | 58.3 | 55.1 (-5.4%) | 54.5 (-6.5%) | 56.0 (-3.9%) | 54.4 (-6.7%) | 59.3 (+1.7%) | 87.4 (+50.0%) |
| 8 | 59.3 | 56.2 (-5.3%) | 55.5 (-6.4%) | 56.9 (-4.0%) | 55.3 (-6.7%) | 60.3 (+1.7%) | 87.6 (+47.7%) |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
