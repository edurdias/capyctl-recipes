Greedy agreement on 62,658 identical BF16 answer positions (200 prompts).

| Build | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (by construction) | 100.000 | 331.5 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 |
| CapyCTL r1 | 94.515 | 19 | 92.274 | 95.453 | 98.381 | 92.778 | 99.801 |
| CapyCTL r2 | 94.805 | 20.5 | 92.817 | 95.771 | 98.207 | 93.084 | 99.734 |
| CapyCTL r3 (FP8 dense/attention/head) | 97.177 | 28 | 96.122 | 97.742 | 99.180 | 96.052 | 99.934 |
| CapyCTL r3e (FP8 dense/attention; BF16 head) | 97.191 | 30 | 96.104 | 97.819 | 99.160 | 96.045 | 99.934 |
| NVIDIA | 97.132 | 30 | 96.075 | 97.701 | 99.201 | 95.952 | 100.000 |
| RedHat | 94.476 | 19 | 92.297 | 95.489 | 98.299 | 92.598 | 99.801 |
| Unsloth (FP8 attention) | 96.018 | 22.5 | 94.384 | 96.900 | 98.903 | 94.451 | 99.867 |

First disagreement is zero-based; a fully agreeing answer is censored at its answer length. BF16's value is therefore its median answer length, not a measured mismatch.

| Build | Load s | Evaluation s | Wall s | Peak CUDA allocated GiB | Peak CUDA reserved GiB | Peak RSS GiB | Minimum host available GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (by construction) | 72.57 | 3186.19 | 3258.75 | 50.068 | 50.295 | 5.687 | 26.761 |
| CapyCTL r1 | 23.23 | 64.83 | 88.06 | 49.987 | 50.438 | 3.951 | 29.403 |
| CapyCTL r2 | 47.12 | 66.10 | 113.23 | 49.987 | 50.438 | 3.902 | 29.599 |
| CapyCTL r3 (FP8 dense/attention/head) | 22.45 | 65.55 | 88.00 | 53.579 | 54.590 | 4.915 | 24.187 |
| CapyCTL r3e (FP8 dense/attention; BF16 head) | 22.59 | 64.75 | 87.34 | 49.964 | 50.312 | 3.917 | 29.556 |
| NVIDIA | 41.57 | 65.05 | 106.62 | 50.068 | 50.160 | 3.932 | 29.634 |
| RedHat | 45.49 | 64.85 | 110.34 | 49.987 | 50.438 | 3.916 | 29.756 |
| Unsloth (FP8 attention) | 48.87 | 65.08 | 113.95 | 50.000 | 50.328 | 3.946 | 29.642 |

Wall times are measured inside each Python command (load plus evaluation); /usr/bin/time whole-command timings and exit codes are in the matching logs. Host-wide memory includes another workload's engine and filesystem cache; it is not the model's RSS.
