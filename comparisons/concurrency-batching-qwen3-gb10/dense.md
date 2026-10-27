## Dense 4B experiment — completed measurements

This replaces the earlier incomplete checkpoint. Both the Qwen3-4B-Instruct-2507 and separately authorized MoE passes are complete. All 219 indexed dense attempts are retained below, including interrupted infrastructure runs, invalid/error cohorts and latency rejections. All deployments were deleted and both experiment-owned roles stopped with ownership checks; their PIDs were independently verified absent.

**Decision: retain production defaults.** Raising useful concurrency produced large gains, but the high-concurrency points use explicitly experimental admission builds. Native batching changes are workload-specific, and pooled chunk-gap percentiles can hide worse isolated stalls. The results justify a separate admission/error-handling and recipe-validation discussion, not an automatic default change.

### Best three-repeat-confirmed service points

All three repeats passed the unchanged TTFT, SSE-gap, correctness and memory gates. `C` is offered client concurrency, not engine slots or token-batch size. Rates are actual output-token usage over the full cohort, including ramp and drain.

| Engine | Workload | C | Config | Output tok/s median (range) | Per-stream median tok/s | Worst p95 TTFT / gap, s |
|---|---|---:|---|---:|---:|---:|
| sglang | long | 8 | `s32-kv16-mem40-b32768` | 33.69 (33.66–34.92) | 7.26 | 9.593 / 0.094 |
| sglang | mixed | 20 | `s32-kv16-mem40-b32768` | 138.42 (138.29–138.89) | 8.07 | 9.764 / 0.143 |
| sglang | short | 104 | `s256-kv16-mem40-bdefault` | 1476.83 (1475.78–1481.85) | 15.79 | 1.796 / 0.082 |
| vllm | long | 8 | `s32-kv16-mem40-b32768` | 33.81 (33.67–33.93) | 7.16 | 9.941 / 0.144 |
| vllm | mixed | 14 | `s32-kv16-mem40-b32768` | 114.38 (112.41–114.46) | 10.48 | 8.392 / 0.088 |
| vllm | short | 124 | `s256-kv16-mem40-bdefault` | 1653.15 (1651.62–1657.22) | 14.04 | 1.973 / 0.083 |

The earlier SGLang short C=112 and vLLM mixed C=16 candidates passed only 2/3 repeats. They are rejected as confirmed service points; the C=104 and C=14 fallbacks above passed 3/3. vLLM long C=8 remains close to the TTFT boundary: 9.941s in its worst repeat. These observations are not an open-loop production SLA.

### Individual-stream tradeoff

These are the fastest observed median per-stream rates among passing dense cells, not separately repeated claims of a unique optimum. They are single screening references. Compare their actual aggregate rates with the selected points above; do not multiply stream medians by concurrency.

| Engine | Workload | Reference config / C | Actual aggregate tok/s | Median stream tok/s | Evidence ID |
|---|---|---|---:|---:|---|
| sglang | long | `baseline` / 2 | 22.77 | 19.84 | `f40ccb8dd220` |
| sglang | mixed | `baseline` / 2 | 41.36 | 22.73 | `52ebc2cd9d2f` |
| sglang | short | `baseline` / 2 | 53.14 | 26.85 | `1f6caca4c0f0` |
| vllm | long | `c4-b8192` / 2 | 22.44 | 19.51 | `7d20972160c3` |
| vllm | mixed | `baseline` / 2 | 39.22 | 22.38 | `b45322b837b4` |
| vllm | short | `baseline` / 2 | 50.04 | 26.16 | `902a4f18b3ca` |

The per-stream estimate is `(completion_tokens − 1) / (DONE arrival − first content arrival)`, computed per request; the selected-point column is the median of the three run medians. SSE content chunks need not be single tokens. This is not an exact GPU decode-kernel rate.

### Saturation is not a service setting or a universal hardware ceiling

- SGLang C=512: three valid token cohorts, median **2,428.44 tok/s**, range **2,288.39–2,475.82**; all fail latency gates. At C=1,024, two cohorts produced 2,517.62 and 2,524.83 tok/s, but the third had four empty HTTP-200 streams and therefore no valid throughput. A software thermal-slowdown event was recorded in that failed repeat; its relationship to the empty streams is unproven.
- vLLM C=512, 8,192-token batch budget: three valid cohorts, median **2,510.59 tok/s**, range **1,839.74–2,525.10**; all fail latency gates. The paired default-budget cohorts ranged **1,608.56–2,236.30**. The larger budget was **12.91–14.37%** faster within each matched pair, with order reversed in repeat 2, but absolute run-to-run variation is larger than that gain. This is conditional paired evidence, not a stable absolute-throughput guarantee. No thermal clock-event flag was recorded in these six cohorts; the large absolute variation remains unattributed.
- vLLM C=1,024 with the larger batch budget completed a valid screening cohort at **2,537.03 tok/s**, p95 TTFT **17.544s**. The default-budget C=1,024 attempt had 46 request timeouts and no valid aggregate throughput. Higher concurrency did not establish a latency-qualified capacity increase.

### Controls, attribution and limits

- Hardware: one NVIDIA GB10, driver 580.173.02; SGLang 0.5.22 and vLLM 0.32.0 in unchanged approved environments. BF16, 16,384-token context, graphs and speculation off, reusable prefix/radix caching disabled. Fixed seed-211 corpus; temperature zero; EOS respected. Short ≈145 actual prompt tokens / at most 256 output tokens; long ≈8,785 / at most 64; mixed alternates admissions 50/50.
- Measured checkpoint: `sha256:82e272a19cc370f4d207d937ccdd7177e5dece4f180e10da97241a89271da12e`, 8,044,982,000 weight bytes, recorded/nonprovisional. Fixture fingerprint labels are not substituted for this measured identity.
- Config key: `sN-kvK-memM-bB` means engine sequence slots N, declared KV K GiB, total memory request M GiB, and engine-specific batch/prefill budget B. `baseline` is 16 slots, KV=4/total=24 GiB; `cN-bB` keeps that memory budget and sets slots N/budget B. `bdefault` resolves to 8,192 prefill tokens for SGLang and 2,048 batched tokens for vLLM on this host, as documented in the earlier native-default query comment. These are distinct engine controls, not observed internal batch sizes.
- The six selected points use server `681a98d8…` / host `9ec03007…`, with only the two recorded admission constants raised to 1,024, plus the process-local host soft FD limit of 8,192. They are not unchanged-release qualification. Official v0.1.8 rows and the intermediate 256-bound builds remain explicitly labelled in the raw table.
- Selected-point telemetry sampled zero router-waiting requests, while engine waiting counts were nonzero. Lease-grant p95 ranged up to 50.833 ms; pre-forward timing was unavailable. The higher service envelope is consistent with admitted engine concurrency and workload-specific scheduling/KV effects, not evidence of a faster controller. There is no approved matched direct-engine baseline or internal token-batch trace, so exact engine/controller attribution remains unknown.
- Across dense attempts, minimum observed MemAvailable was **37.34 GiB**, above the unchanged 13 GiB stop. Peak system nonavailable memory was **84.35 GiB** and maximum observed memory PSI full avg10 was **0.23%**. Device total/used/free from nvidia-smi was N/A; CPU PSS and system nonavailable memory are not exact peak accelerator allocation. No whole-system energy claim is made.
- Serial warmup is excluded, but does not establish a fully populated stationary engine batch. The closed-loop cohorts include ramp/drain and often only tens of long-prompt completions. Tail precision and generalization are limited. Correctness means HTTP/SSE/usage and numbered-list structural checks, not general model quality. Failed or incomplete token accounting yields null aggregate throughput.
- Keep the original fixed-budget comparison separate: its small gains do not support a broad default change. In particular, the vLLM long C=2 larger-batch candidate reduced pooled p95 gaps while increasing an observed maximum stall from 396 ms to 1,183 ms. Supplemental p99/max and per-request maximum-gap data accompany the final publication.

Reproduction: [driver/observer source](https://github.com/edurdias/capyctl/issues/211#issuecomment-6435986028), [orchestration/reporting source](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443042524), and [setup/configurations/commands](https://github.com/edurdias/capyctl/issues/211#issuecomment-6436349184). Supplemental tables join by evidence ID: [occupancy/memory](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443586946), [latency tails/per-stream rates](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443587549), and [request/token accounting](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443588115).

### Every indexed dense attempt
| Evidence ID | Model | Engine | Build (server/host) | Phase/repeat | Config | Workload | Offered C | Output tok/s | p95 TTFT s | p95 chunk gap s | Successful/admitted | Verdict |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 46474450e500 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | short | 1 | 21.88 | 0.144 | 0.050 | 40/40 | pass |
| 1f6caca4c0f0 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | short | 2 | 53.14 | 0.147 | 0.041 | 40/40 | pass |
| 73dc8b7a1409 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | short | 4 | 95.37 | 0.185 | 0.046 | 41/41 | pass |
| aae96c3cea3c | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | short | 8 | 202.21 | 0.233 | 0.044 | 48/48 | pass |
| c292cee7f038 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | long | 1 | 14.13 | 1.365 | 0.056 | 40/40 | pass |
| f40ccb8dd220 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | long | 2 | 22.77 | 2.558 | 0.052 | 40/40 | pass |
| 712b35e3a9e2 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | long | 4 | 26.29 | 8.367 | 0.059 | 43/43 | pass |
| 9bef07bd9c56 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | long | 8 | 26.51 | 17.834 | 0.061 | 47/47 | ttft_threshold, sustained_ttft |
| a18924e3c7cf | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | mixed | 1 | 19.45 | 1.272 | 0.053 | 40/40 | pass |
| 52ebc2cd9d2f | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | mixed | 2 | 41.36 | 1.338 | 0.047 | 40/40 | pass |
| a0e418b2fe53 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | mixed | 4 | unknown | 1.960 | 0.051 | 29/33 | request_errors, incomplete_usage_or_cancelled_work, minimum_completions_not_reached, external_STOP |
| 3ab32818179c | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | mixed | 4 | 66.25 | 1.349 | 0.049 | 43/43 | pass |
| 3239c6970eb9 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | baseline | mixed | 8 | 92.41 | 3.313 | 0.056 | 46/46 | pass |
| 55732467a587 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c4-b2048 | short | 8 | 104.93 | 9.930 | 0.042 | 44/44 | ttft_threshold, sustained_ttft |
| 16dd83b7e31c | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c4-b2048 | long | 4 | 25.32 | 8.981 | 0.061 | 43/43 | pass |
| 064dc0719bf7 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c4-b2048 | mixed | 8 | 64.76 | 15.060 | 0.054 | 47/47 | ttft_threshold, sustained_ttft |
| 800b14811a15 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c4-b8192 | short | 8 | 96.10 | 10.021 | 0.044 | 45/45 | ttft_threshold, sustained_ttft |
| 3cb84049e87e | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c4-b8192 | long | 4 | 26.42 | 8.338 | 0.060 | 43/43 | pass |
| f3d4114b9790 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c4-b8192 | mixed | 8 | 65.92 | 15.384 | 0.052 | 47/47 | ttft_threshold, sustained_ttft |
| 73ec512580c8 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b2048 | short | 8 | 202.57 | 0.250 | 0.043 | 48/48 | pass |
| 68702ecd2ea1 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b2048 | long | 4 | 25.67 | 8.813 | 0.060 | 43/43 | pass |
| 305ebf3cd036 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b2048 | mixed | 8 | 90.50 | 6.634 | 0.109 | 47/47 | pass |
| 9ec409a49059 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b8192 | short | 8 | 202.66 | 0.249 | 0.042 | 48/48 | pass |
| d565ce814573 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b8192 | long | 4 | 26.45 | 8.328 | 0.060 | 43/43 | pass |
| 5948d9e8d549 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b8192 | mixed | 8 | unknown | 4.471 | 0.058 | 36/44 | request_errors, incomplete_usage_or_cancelled_work, minimum_completions_not_reached, external_STOP |
| d706e88e66bd | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | screen/0 | c8-b8192 | mixed | 8 | 96.03 | 4.908 | 0.061 | 47/47 | pass |
| 6c3bfe0f9d48 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/1 | baseline | short | 8 | 202.16 | 0.234 | 0.046 | 48/48 | pass |
| 984c7811d8fc | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/1 | c8-b8192 | short | 8 | 201.25 | 0.242 | 0.044 | 48/48 | pass |
| 99afd012082e | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/1 | baseline | long | 4 | 26.54 | 8.322 | 0.061 | 43/43 | pass |
| 0ae8fd79a0f9 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/1 | c8-b8192 | long | 4 | 26.58 | 8.292 | 0.060 | 43/43 | pass |
| 4d99a4f93282 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/1 | baseline | mixed | 8 | 92.71 | 3.293 | 0.056 | 46/46 | pass |
| 01419621fed7 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/1 | c8-b8192 | mixed | 8 | 96.18 | 5.023 | 0.064 | 47/47 | pass |
| 03f5ca3e6824 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/2 | c8-b8192 | short | 8 | 202.15 | 0.236 | 0.044 | 48/48 | pass |
| bb21ad87967f | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/2 | baseline | short | 8 | 201.73 | 0.360 | 0.044 | 48/48 | pass |
| 84eb6af6377d | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/2 | c8-b8192 | long | 4 | 26.54 | 8.302 | 0.059 | 43/43 | pass |
| a2bb45a6cb03 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/2 | baseline | long | 4 | 26.17 | 8.483 | 0.063 | 43/43 | pass |
| 74125ef5a7d9 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/2 | c8-b8192 | mixed | 8 | 95.50 | 4.370 | 0.056 | 47/47 | pass |
| 39ec8b395f3e | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/2 | baseline | mixed | 8 | 96.29 | 3.256 | 0.056 | 47/47 | pass |
| ebba2d0d74df | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/3 | baseline | short | 8 | 201.25 | 0.258 | 0.043 | 48/48 | pass |
| 0fbd0f5f2a36 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/3 | c8-b8192 | short | 8 | 201.72 | 0.243 | 0.043 | 48/48 | pass |
| 2cd09b94a257 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/3 | baseline | long | 4 | 26.39 | 8.359 | 0.058 | 43/43 | pass |
| 66344ec4e50d | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/3 | c8-b8192 | long | 4 | 26.51 | 8.313 | 0.059 | 43/43 | pass |
| 54c54fac4945 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/3 | baseline | mixed | 8 | 95.76 | 3.283 | 0.094 | 47/47 | pass |
| 8f5edab9d61e | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | repeat/3 | c8-b8192 | mixed | 8 | 96.11 | 5.016 | 0.055 | 47/47 | pass |
| 84b14a3b6712 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | short | 1 | 21.42 | 0.081 | 0.049 | 40/40 | pass |
| 902a4f18b3ca | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | short | 2 | 50.04 | 0.135 | 0.043 | 41/41 | pass |
| d866aa5a98a2 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | short | 4 | 97.79 | 0.159 | 0.042 | 42/42 | pass |
| b82cf24e26c4 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | short | 8 | 197.67 | 0.226 | 0.045 | 48/48 | pass |
| 1208aab9dfae | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | long | 1 | 13.78 | 1.362 | 0.055 | 40/40 | pass |
| 9b289525423c | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | long | 2 | 22.24 | 2.366 | 0.135 | 41/41 | pass |
| a2d760ea7ace | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | long | 4 | 26.53 | 6.494 | 0.314 | 43/43 | chunk_gap_threshold, sustained_chunk_gaps |
| 53590c3fdb5e | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | long | 8 | 26.85 | 14.836 | 0.314 | 47/47 | ttft_threshold, chunk_gap_threshold, sustained_ttft |
| 6da4d6e7395d | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | mixed | 1 | 19.27 | 1.365 | 0.054 | 40/40 | pass |
| b45322b837b4 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | mixed | 2 | 39.22 | 1.468 | 0.049 | 41/41 | pass |
| 3a1b37ca7d32 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | mixed | 4 | 65.50 | 1.471 | 0.127 | 43/43 | pass |
| b767566ee1d1 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | baseline | mixed | 8 | 96.76 | 2.825 | 0.284 | 47/47 | chunk_gap_threshold, sustained_chunk_gaps |
| 33ad3afa880c | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c4-b2048 | short | 8 | 98.67 | 10.130 | 0.042 | 46/46 | ttft_threshold, sustained_ttft |
| a775e9cf28d6 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c4-b2048 | long | 2 | 22.25 | 2.360 | 0.073 | 41/41 | pass |
| 922d0ee46d08 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c4-b2048 | mixed | 4 | 65.74 | 1.559 | 0.125 | 43/43 | pass |
| 56f7383d5c1b | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c4-b8192 | short | 8 | 98.68 | 10.082 | 0.042 | 46/46 | ttft_threshold, sustained_ttft |
| 7d20972160c3 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c4-b8192 | long | 2 | 22.44 | 2.517 | 0.054 | 41/41 | pass |
| 6243083fe035 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c4-b8192 | mixed | 4 | 65.59 | 1.383 | 0.050 | 43/43 | pass |
| 0d0e329497cc | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c8-b2048 | short | 8 | 198.55 | 0.319 | 0.044 | 48/48 | pass |
| 4d4aa7ea6b06 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c8-b2048 | long | 2 | 22.22 | 2.366 | 0.118 | 41/41 | pass |
| 825710fd410a | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c8-b2048 | mixed | 4 | 65.93 | 1.446 | 0.123 | 43/43 | pass |
| ccab3535b87d | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c8-b8192 | short | 8 | 199.05 | 0.230 | 0.043 | 48/48 | pass |
| 7578354adf1d | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c8-b8192 | long | 2 | 22.44 | 2.513 | 0.053 | 41/41 | pass |
| 14b8ea943c50 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | screen/0 | c8-b8192 | mixed | 4 | 65.55 | 1.404 | 0.052 | 43/43 | pass |
| febdae16ffeb | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/1 | baseline | short | 8 | 198.68 | 0.224 | 0.043 | 48/48 | pass |
| 81393bdd7961 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/1 | c8-b8192 | short | 8 | 199.02 | 0.224 | 0.043 | 48/48 | pass |
| c438033c1ba0 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/1 | baseline | long | 2 | 22.22 | 2.409 | 0.124 | 41/41 | pass |
| 74fdb320d580 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/1 | c8-b8192 | long | 2 | 22.46 | 2.520 | 0.053 | 41/41 | pass |
| 7b23f2bdea62 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/1 | baseline | mixed | 4 | 66.21 | 1.520 | 0.121 | 43/43 | pass |
| 4326f361e533 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/1 | c8-b2048 | mixed | 4 | 65.94 | 1.458 | 0.125 | 43/43 | pass |
| 1d87386bde27 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/2 | c8-b8192 | short | 8 | 198.86 | 0.238 | 0.043 | 48/48 | pass |
| d6fa96a9008f | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/2 | baseline | short | 8 | 197.43 | 0.309 | 0.046 | 48/48 | pass |
| 3bd06562aaab | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/2 | c8-b8192 | long | 2 | 22.46 | 2.510 | 0.053 | 41/41 | pass |
| 34e38e2512d7 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/2 | baseline | long | 2 | 22.23 | 2.374 | 0.070 | 41/41 | pass |
| b1ffd4c45899 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/2 | c8-b2048 | mixed | 4 | 65.97 | 1.503 | 0.122 | 43/43 | pass |
| 45251f3b9145 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/2 | baseline | mixed | 4 | 65.95 | 1.426 | 0.064 | 43/43 | pass |
| 415bea199cc7 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/3 | baseline | short | 8 | 199.54 | 0.213 | 0.043 | 48/48 | pass |
| 61a5b45cf4a5 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/3 | c8-b8192 | short | 8 | 198.38 | 0.214 | 0.043 | 48/48 | pass |
| c9bea079e1ad | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/3 | baseline | long | 2 | 22.29 | 2.356 | 0.120 | 41/41 | pass |
| 59501dc0886c | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/3 | c8-b8192 | long | 2 | 22.41 | 2.515 | 0.055 | 41/41 | pass |
| def53c1b269e | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/3 | baseline | mixed | 4 | 66.09 | 1.429 | 0.062 | 43/43 | pass |
| e7b182808986 | Qwen3-4B-Instruct-2507 | vllm | release-32/release-64 | repeat/3 | c8-b2048 | mixed | 4 | 66.05 | 1.440 | 0.122 | 43/43 | pass |
| de7430934848 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | ceiling/0 | s32-kv4-mem24-bdefault | short | 16 | 377.99 | 0.423 | 0.046 | 96/96 | pass |
| f53a7e6f5c61 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | ceiling/0 | s32-kv4-mem24-bdefault | short | 32 | 671.73 | 0.753 | 0.050 | 160/160 | pass |
| 1332f3b72255 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | ceiling/0 | s32-kv4-mem24-bdefault | long | 16 | 26.21 | 38.312 | 0.059 | 55/55 | ttft_threshold |
| 20e446417e95 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | ceiling/0 | s32-kv4-mem24-bdefault | long | 32 | 26.53 | 75.591 | 0.059 | 95/95 | ttft_threshold |
| 8c51fcd5762a | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | ceiling/0 | s32-kv4-mem24-bdefault | mixed | 16 | 109.52 | 15.721 | 0.060 | 55/55 | ttft_threshold |
| b55170256a95 | Qwen3-4B-Instruct-2507 | sglang | release-32/release-64 | ceiling/0 | s32-kv4-mem24-bdefault | mixed | 32 | 111.02 | 42.303 | 0.060 | 95/95 | ttft_threshold |
| a60cc5c24511 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/release-64 | ceiling/0 | s128-kv16-mem40-bdefault | short | 32 | 683.09 | 0.663 | 0.057 | 192/192 | pass |
| 37f4e7519cdd | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/release-64 | ceiling/0 | s128-kv16-mem40-bdefault | short | 64 | 1116.23 | 1.182 | 0.061 | 320/320 | pass |
| 89d8067ff806 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/release-64 | ceiling/0 | s128-kv16-mem40-bdefault | short | 128 | unknown | 1.151 | 0.061 | 64/128 | request_errors, incomplete_usage_or_cancelled_work, minimum_completions_not_reached, minimum_steady_duration_not_reached, sustained_errors |
| 87d80fc23232 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv16-mem40-bdefault | short | 64 | 1125.91 | 1.194 | 0.071 | 320/320 | pass |
| 493bb6cacdce | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv16-mem40-bdefault | short | 128 | 1427.97 | 2.144 | 0.081 | 384/384 | ttft_threshold |
| 1df870fda774 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s64-kv16-mem40-bdefault | long | 8 | 33.68 | 10.102 | 0.087 | 40/40 | ttft_threshold |
| bfde1123b838 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s64-kv16-mem40-bdefault | long | 16 | 34.88 | 27.407 | 0.117 | 55/55 | ttft_threshold |
| dda66736aaae | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s64-kv16-mem40-bdefault | long | 32 | 35.12 | 54.265 | 0.117 | 94/94 | ttft_threshold |
| f8153a96a127 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 8 | 94.87 | 3.351 | 0.061 | 47/47 | pass |
| 3418e543a791 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 16 | 122.85 | 7.439 | 0.084 | 54/54 | pass |
| d4c84739e3b9 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 32 | 147.21 | 15.329 | 0.131 | 95/95 | ttft_threshold |
| b5a6bd654663 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s256-kv16-mem40-bdefault | short | 96 | 1419.13 | 1.682 | 0.071 | 384/384 | pass |
| d44e129dd5d1 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s256-kv16-mem40-bdefault | short | 128 | 1620.51 | 2.156 | 0.130 | 384/384 | ttft_threshold |
| d5aca5432a88 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s256-kv16-mem40-bdefault | short | 256 | 1965.49 | 4.277 | 0.122 | 766/766 | ttft_threshold |
| d369c00dee40 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv48-mem72-bdefault | long | 16 | 36.19 | 20.256 | 0.132 | 48/48 | ttft_threshold |
| 7e49335d10ea | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv48-mem72-bdefault | long | 32 | 36.61 | 40.201 | 0.224 | 65/65 | ttft_threshold, chunk_gap_threshold |
| a60f24c2d4ab | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv48-mem72-bdefault | long | 64 | 37.66 | 95.691 | 0.269 | 181/181 | ttft_threshold, chunk_gap_threshold |
| d01c65d9ed90 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv48-mem72-bdefault | mixed | 16 | 122.21 | 7.547 | 0.084 | 54/54 | pass |
| eab288bdd155 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv48-mem72-bdefault | mixed | 32 | 149.82 | 15.803 | 0.166 | 95/95 | ttft_threshold |
| bcb052747f17 | Qwen3-4B-Instruct-2507 | sglang | experimental-9aa9e3f1/experimental-7b8f3eaf | ceiling/0 | s128-kv48-mem72-bdefault | mixed | 64 | 164.45 | 30.318 | 0.228 | 190/190 | ttft_threshold, chunk_gap_threshold |
| 6f4e183cf607 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 256 | 2171.31 | 4.308 | 0.123 | 512/512 | ttft_threshold |
| facb412709b3 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 512 | unknown | 8.259 | 0.204 | 495/547 | request_errors, incomplete_usage_or_cancelled_work, minimum_completions_not_reached, minimum_steady_duration_not_reached, ttft_threshold, chunk_gap_threshold, sustained_errors |
| 4923431c5820 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s1024-kv64-mem88-bdefault | short | 256 | 2167.64 | 4.332 | 0.171 | 512/512 | ttft_threshold |
| e035025e840b | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s1024-kv64-mem88-bdefault | short | 512 | 2475.82 | 8.499 | 0.212 | 1024/1024 | ttft_threshold, chunk_gap_threshold |
| a27cc72a7e3e | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s1024-kv64-mem88-bdefault | short | 1024 | 2517.62 | 16.688 | 0.421 | 2049/2049 | ttft_threshold, chunk_gap_threshold |
| 8ed6fd1ed88b | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | long | 8 | 33.50 | 9.669 | 0.086 | 41/41 | pass |
| d96714c849b6 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | long | 16 | 36.39 | 18.818 | 0.179 | 49/49 | ttft_threshold |
| b0d3d1f24bed | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | long | 32 | 39.23 | 37.268 | 0.224 | 64/64 | ttft_threshold, chunk_gap_threshold |
| a252bcabb464 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | mixed | 8 | 96.37 | 2.536 | 0.060 | 47/47 | pass |
| 36d22ed3c793 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | mixed | 16 | 127.17 | 7.370 | 0.108 | 57/57 | pass |
| 81b7cbeb30ee | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | mixed | 32 | 155.40 | 14.695 | 0.131 | 95/95 | ttft_threshold |
| 8eef9b50f42b | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-bdefault | short | 112 | 1534.13 | 1.926 | 0.077 | 448/448 | pass |
| ce6e9bc36e37 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-bdefault | short | 120 | 1573.52 | 2.051 | 0.087 | 480/480 | ttft_threshold |
| 0e08666f5b21 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | long | 6 | 32.16 | 7.156 | 0.076 | 43/43 | pass |
| b15894d119a6 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | long | 8 | 34.92 | 9.558 | 0.088 | 40/40 | pass |
| 091d74f2add7 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | long | 10 | 35.92 | 11.944 | 0.101 | 49/49 | ttft_threshold |
| e16e2c376636 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | mixed | 20 | 138.44 | 9.713 | 0.096 | 59/59 | pass |
| e991330c4408 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | mixed | 24 | 145.60 | 10.975 | 0.136 | 71/71 | ttft_threshold |
| 298925676b6e | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s256-kv16-mem40-bdefault | short | 112 | 1517.84 | 1.929 | 0.078 | 448/448 | pass |
| 7e81768428c0 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv16-mem40-b32768 | long | 8 | 34.92 | 9.554 | 0.092 | 40/40 | pass |
| 2e59b4c39db4 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv16-mem40-b32768 | mixed | 20 | 138.42 | 9.692 | 0.101 | 59/59 | pass |
| fd509693751b | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv16-mem40-b32768 | mixed | 20 | 138.29 | 9.764 | 0.143 | 59/59 | pass |
| a9b0b8f021a7 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv16-mem40-b32768 | long | 8 | 33.69 | 9.569 | 0.089 | 41/41 | pass |
| 45c8937fedff | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s256-kv16-mem40-bdefault | short | 112 | 1537.48 | 1.950 | 0.096 | 448/448 | pass |
| b3e78b943d83 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s256-kv16-mem40-bdefault | short | 112 | 1531.16 | 2.050 | 0.078 | 448/448 | ttft_threshold |
| f63f3e993a90 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv16-mem40-b32768 | long | 8 | 33.66 | 9.593 | 0.094 | 41/41 | pass |
| 13372f6f61b9 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv16-mem40-b32768 | mixed | 20 | 138.89 | 9.688 | 0.095 | 59/59 | pass |
| ef115330053a | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s1024-kv64-mem88-bdefault | short | 1024 | 2524.83 | 16.796 | 0.416 | 2050/2050 | ttft_threshold, chunk_gap_threshold |
| 8da1890df32b | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s1024-kv64-mem88-bdefault | short | 512 | 2428.44 | 8.631 | 0.220 | 1024/1024 | ttft_threshold, chunk_gap_threshold |
| 41dc926c15cb | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s1024-kv64-mem88-bdefault | short | 512 | 2288.39 | 8.539 | 0.212 | 1025/1025 | ttft_threshold, chunk_gap_threshold |
| c045abeed683 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s1024-kv64-mem88-bdefault | short | 1024 | unknown | 16.697 | 0.425 | 3066/3070 | request_errors, incomplete_usage_or_cancelled_work, ttft_threshold, chunk_gap_threshold |
| e594aa034a85 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-b32768 | short | 128 | 1398.06 | 2.234 | 0.086 | 385/385 | ttft_threshold |
| 56cd69e7ceff | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-b32768 | short | 160 | 1813.59 | 2.691 | 0.091 | 480/480 | ttft_threshold |
| 10a8039fa66d | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s256-kv16-mem40-bdefault | short | 104 | 1476.83 | 1.785 | 0.074 | 416/416 | pass |
| 2224f44f9eee | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s256-kv16-mem40-bdefault | short | 104 | 1481.85 | 1.796 | 0.082 | 416/416 | pass |
| 0f58d1fe8608 | Qwen3-4B-Instruct-2507 | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s256-kv16-mem40-bdefault | short | 104 | 1475.78 | 1.751 | 0.074 | 416/416 | pass |
| 1b31fbe4466f | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 16 | 376.26 | 0.337 | 0.046 | 96/96 | pass |
| cd8dbd3dcaae | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 32 | 592.99 | 0.580 | 0.054 | 162/162 | pass |
| e50b39dc0f05 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 64 | 1126.28 | 0.958 | 0.084 | 320/320 | pass |
| 9cd3b8d8bafe | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 96 | 1438.21 | 1.412 | 0.073 | 384/384 | pass |
| 5c70cf8a5cbf | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 112 | 1571.27 | 1.776 | 0.088 | 448/448 | pass |
| 5ca27e67fb94 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 128 | 1654.67 | 2.006 | 0.127 | 501/501 | ttft_threshold |
| 414c66f74ebb | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 256 | 2154.97 | 4.147 | 0.268 | 758/758 | ttft_threshold, chunk_gap_threshold |
| badcb3feacf7 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 512 | 2228.72 | 10.544 | 0.571 | 1529/1529 | ttft_threshold, chunk_gap_threshold |
| 2c954a89d5f9 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 1024 | unknown | 39.325 | 0.942 | 2016/2062 | request_errors, incomplete_usage_or_cancelled_work, ttft_threshold, chunk_gap_threshold, sustained_errors |
| 719b9ec3f627 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | long | 4 | 29.46 | 2.815 | 0.328 | 43/43 | chunk_gap_threshold |
| b9478cb25c34 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | long | 8 | 33.37 | 7.797 | 0.377 | 47/47 | chunk_gap_threshold |
| 1cb546de70b5 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | long | 16 | 28.88 | 22.087 | 0.524 | 55/55 | ttft_threshold, chunk_gap_threshold |
| c0c078ce0972 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | long | 32 | 28.53 | 55.047 | 0.523 | 95/95 | ttft_threshold, chunk_gap_threshold |
| a5e3e159906c | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 4 | unknown | 1.548 | 0.126 | 24/28 | request_errors, incomplete_usage_or_cancelled_work, minimum_completions_not_reached, minimum_steady_duration_not_reached, external_STOP |
| ba04c31232c8 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 4 | 66.00 | 1.484 | 0.121 | 43/43 | pass |
| d127cd98a11a | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 8 | 97.11 | 2.788 | 0.282 | 47/47 | chunk_gap_threshold |
| e0ebfea7b2af | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 16 | 123.81 | 7.468 | 0.345 | 55/55 | chunk_gap_threshold |
| e02c4927a83f | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-bdefault | mixed | 32 | 137.86 | 18.223 | 0.474 | 95/95 | ttft_threshold, chunk_gap_threshold |
| c3e1c7e6d253 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-bdefault | long | 16 | 27.29 | 21.630 | 0.583 | 55/55 | ttft_threshold, chunk_gap_threshold |
| 1553c0920969 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-bdefault | long | 32 | 26.62 | 56.317 | 0.595 | 95/95 | ttft_threshold, chunk_gap_threshold |
| eda3e9daae44 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-bdefault | long | 64 | unknown | 125.897 | 0.600 | 189/191 | request_errors, incomplete_usage_or_cancelled_work, ttft_threshold, chunk_gap_threshold |
| 068286bd9588 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-bdefault | mixed | 16 | 124.73 | 7.406 | 0.343 | 55/55 | chunk_gap_threshold |
| 66e3685e8771 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-bdefault | mixed | 32 | 136.25 | 18.239 | 0.512 | 95/95 | ttft_threshold, chunk_gap_threshold |
| a01ef68e48ce | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-bdefault | mixed | 64 | unknown | 46.598 | 0.605 | 190/191 | request_errors, incomplete_usage_or_cancelled_work, ttft_threshold, chunk_gap_threshold |
| 3e85221d3f17 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-b1024 | long | 4 | 31.28 | 2.663 | 0.183 | 43/43 | pass |
| 4c503067d17c | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-b1024 | long | 8 | 33.23 | 8.075 | 0.240 | 47/47 | chunk_gap_threshold |
| bb0684fb2374 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-b1024 | long | 16 | 32.61 | 22.612 | 0.244 | 55/55 | ttft_threshold, chunk_gap_threshold |
| b99d812083ff | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-b1024 | mixed | 4 | 66.84 | 1.445 | 0.135 | 43/43 | pass |
| de20ee6c298d | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-b1024 | mixed | 8 | 100.70 | 2.841 | 0.166 | 47/47 | pass |
| c21ad0f8fbf6 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s64-kv16-mem40-b1024 | mixed | 16 | 131.84 | 7.413 | 0.196 | 59/59 | pass |
| b903ff1e1c4e | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | long | 8 | 33.98 | 9.914 | 0.088 | 47/47 | pass |
| 8aac511dc097 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | long | 16 | 36.42 | 19.685 | 0.919 | 55/55 | ttft_threshold, chunk_gap_threshold |
| 5aad932aa48c | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | long | 32 | 37.56 | 36.056 | 4.729 | 93/93 | ttft_threshold, chunk_gap_threshold |
| 4a96ad6230dd | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | mixed | 8 | 94.92 | 4.640 | 0.082 | 47/47 | pass |
| 90417cd140a0 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | mixed | 16 | 121.52 | 7.934 | 0.086 | 55/55 | pass |
| ebe24f529211 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv48-mem72-b32768 | mixed | 32 | 152.31 | 17.658 | 0.138 | 95/95 | ttft_threshold |
| b7e6ce9ea028 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-b8192 | short | 128 | 1414.82 | 2.163 | 0.089 | 388/388 | ttft_threshold |
| 0de2c8cfa6f0 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-b8192 | short | 160 | 1840.63 | 2.590 | 0.090 | 480/480 | ttft_threshold |
| b3807988fd0e | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-b8192 | short | 512 | 2552.67 | 7.660 | 0.210 | 1521/1521 | ttft_threshold, chunk_gap_threshold |
| a752194a9ebf | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 120 | 1627.18 | 1.845 | 0.135 | 480/480 | pass |
| af1087e8d5af | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-bdefault | short | 124 | 1645.05 | 2.152 | 0.092 | 496/496 | ttft_threshold |
| 586890464b1f | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-bdefault | short | 112 | 1568.95 | 1.733 | 0.078 | 448/448 | pass |
| f7c63813ec6a | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-bdefault | short | 120 | 1616.31 | 1.841 | 0.083 | 480/480 | pass |
| 4e96b4789957 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-bdefault | short | 124 | 1660.68 | 1.838 | 0.081 | 496/496 | pass |
| 080939958bbe | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s256-kv16-mem40-bdefault | short | 128 | 1655.59 | 2.221 | 0.084 | 512/512 | ttft_threshold |
| bab9065d1194 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | long | 6 | 31.37 | 7.444 | 0.078 | 43/43 | pass |
| 14a28a952e48 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | long | 8 | 33.97 | 9.899 | 0.091 | 47/47 | pass |
| 3cea4ee72e4a | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | long | 10 | 35.08 | 11.163 | 0.187 | 48/48 | ttft_threshold |
| 07cc5eb95347 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | mixed | 16 | 123.59 | 8.149 | 0.086 | 54/54 | pass |
| f744a534da7d | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | mixed | 20 | 134.97 | 10.505 | 0.116 | 59/59 | ttft_threshold |
| feb05d9236c3 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | mixed | 24 | 142.58 | 11.748 | 0.110 | 71/71 | ttft_threshold |
| 0856f19e9711 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv16-mem40-b32768 | mixed | 28 | 143.99 | 14.764 | 0.112 | 83/83 | ttft_threshold |
| 797e7eaf7359 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s256-kv16-mem40-bdefault | short | 124 | 1657.22 | 1.973 | 0.081 | 496/496 | pass |
| 6b2c1264a953 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv16-mem40-b32768 | long | 8 | 33.81 | 9.573 | 0.102 | 47/47 | pass |
| af663f10b93a | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv16-mem40-b32768 | mixed | 16 | 123.40 | 9.627 | 0.085 | 54/54 | pass |
| 614fa3ba6104 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv16-mem40-b32768 | mixed | 16 | 121.32 | 8.380 | 0.085 | 55/55 | pass |
| 83bdbcb57232 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv16-mem40-b32768 | long | 8 | 33.67 | 9.681 | 0.144 | 45/45 | pass |
| be0d7da3c9f5 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s256-kv16-mem40-bdefault | short | 124 | 1653.15 | 1.832 | 0.083 | 496/496 | pass |
| aed9f8fd711c | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s256-kv16-mem40-bdefault | short | 124 | 1651.62 | 1.807 | 0.082 | 496/496 | pass |
| 376f15579277 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv16-mem40-b32768 | long | 8 | 33.93 | 9.941 | 0.088 | 47/47 | pass |
| 4c54db782ede | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv16-mem40-b32768 | mixed | 16 | 121.55 | 10.074 | 0.087 | 55/55 | ttft_threshold |
| cf7441b9252b | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s1024-kv64-mem88-bdefault | short | 512 | 1608.56 | 10.504 | 0.730 | 1526/1526 | ttft_threshold, chunk_gap_threshold |
| 23fd4b4380c4 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s1024-kv64-mem88-b8192 | short | 512 | 1839.74 | 7.569 | 0.384 | 1512/1512 | ttft_threshold, chunk_gap_threshold |
| c0ca72df114d | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s1024-kv64-mem88-b8192 | short | 512 | 2510.59 | 8.022 | 0.228 | 1483/1483 | ttft_threshold, chunk_gap_threshold |
| 703297f2b044 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s1024-kv64-mem88-bdefault | short | 512 | 2223.30 | 10.597 | 0.572 | 1529/1529 | ttft_threshold, chunk_gap_threshold |
| 0fdc16f1963b | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s1024-kv64-mem88-bdefault | short | 512 | 2236.30 | 10.504 | 0.566 | 1534/1534 | ttft_threshold, chunk_gap_threshold |
| 792b4cd551a1 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s1024-kv64-mem88-b8192 | short | 512 | 2525.10 | 8.577 | 0.215 | 1522/1522 | ttft_threshold, chunk_gap_threshold |
| 03177a80b686 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv64-mem88-b8192 | short | 1024 | 2537.03 | 17.544 | 1.188 | 3066/3066 | ttft_threshold, chunk_gap_threshold |
| 02c7edbf851f | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv16-mem40-b32768 | mixed | 14 | 114.38 | 6.952 | 0.088 | 51/51 | pass |
| 6ba9c5f0f65d | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv16-mem40-b32768 | mixed | 14 | 112.41 | 8.392 | 0.080 | 52/52 | pass |
| 5d9e642667e8 | Qwen3-4B-Instruct-2507 | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv16-mem40-b32768 | mixed | 14 | 114.46 | 7.380 | 0.080 | 51/51 | pass |
