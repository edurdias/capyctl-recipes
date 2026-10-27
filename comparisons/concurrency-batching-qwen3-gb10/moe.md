## Qwen3-30B-A3B — completed native experiment

The MoE pass is complete: **94 indexed measurement attempts**, including the two preserved observation interruptions. All six selected service points have three passing completed repeats. The interrupted vLLM short repeat was resumed as a separate attempt, not overwritten. All deployments were deleted; both experiment-owned roles were stopped with ownership checks and their PIDs verified absent. Final host MemAvailable was **117.71 GiB**, with memory PSI avg10=0. Host B was not used.

**Decision: retain production defaults.** The measured service envelope is model- and workload-specific, and several passing points leave little latency margin. All MoE rows use the recorded experimental admission binaries; they are not unchanged-release qualification. Promising recipe changes require separate release-compatible validation, not a new CapyCTL token scheduler.

Checkpoint: `sha256:ab8d4d059c4d30e0979935d5a1955cfb4e4f53e595c2b9dcf6d1175dce32f91b`; **61,066,575,648 weight bytes**, measured/recorded/nonprovisional. BF16, context 16,384, graphs/speculation off, prefix reuse disabled, thinking explicitly disabled; vLLM retains the existing Triton MoE backend. These curves are not an isolated dense-versus-MoE architecture benchmark.

### Best three-repeat-confirmed service points

| Engine | Workload | Config | C | Output tok/s median (range) | Median per-stream tok/s | Worst p95 TTFT / gap s |
|---|---|---|---:|---:|---:|---:|
| sglang | long | `s32-kv12-mem88-b32768` | 6 | 27.11 (25.99–27.27) | 8.70 | 9.990 / 0.128 |
| sglang | mixed | `s32-kv12-mem88-b32768` | 12 | 65.28 (65.00–67.86) | 6.21 | 6.935 / 0.199 |
| sglang | short | `s128-kv6-mem82-bdefault` | 24 | 180.69 (177.06–181.15) | 8.44 | 1.172 / 0.195 |
| vllm | long | `s32-kv12-mem88-b32768` | 5 | 26.12 (25.65–26.27) | 10.06 | 7.454 / 0.093 |
| vllm | mixed | `s32-kv12-mem88-b32768` | 10 | 67.33 (66.34–68.02) | 8.33 | 6.094 / 0.197 |
| vllm | short | `s128-kv6-mem82-bdefault` | 16 | 141.95 (141.43–143.55) | 9.98 | 0.723 / 0.193 |

The service gates remain p95 TTFT ≤2s short / ≤10s long and mixed; pooled p95 SSE content-chunk gap ≤200ms; zero request/contract errors; protected memory headroom. They are experimental selection criteria, not production SLAs. SGLang long reached 9.990s TTFT and mixed reached 199ms gaps in their worst repeats. vLLM short/mixed worst gaps were 193/197ms. Treat those margins accordingly.

Rejected confirmation candidates remain below: vLLM short C=24 and C=20 with budget 8,192, long C=6, and mixed C=12 each passed only 2/3. No failed latency repeat was discarded or replaced. Infrastructure interruptions are excluded from repeat summaries but remain explicit raw rows.

### Individual-stream tradeoff

Fastest observed median stream rates among passing MoE cohorts; these are single screening references, not separately repeated claims of a unique optimum. The selected-point stream column above is the median of three run medians. The estimate is `(completion_tokens − 1) / (DONE arrival − first content arrival)`, not aggregate throughput divided by concurrency.

| Engine | Workload | Reference config / C | Actual aggregate tok/s | Median stream tok/s | Evidence ID |
|---|---|---|---:|---:|---|
| sglang | short | `s16-kv6-mem82-bdefault` / 1 | 29.26 | 30.28 | `55d752d7fe0c` |
| sglang | long | `s16-kv6-mem82-bdefault` / 1 | 15.10 | 27.26 | `0663306a7fd9` |
| sglang | mixed | `s16-kv6-mem82-bdefault` / 1 | 24.07 | 28.61 | `4e582bec0151` |
| vllm | short | `s16-kv6-mem82-bdefault` / 1 | 28.96 | 29.91 | `a5d94f0f5801` |
| vllm | long | `s16-kv6-mem82-bdefault` / 1 | 14.51 | 26.78 | `f234f31bcdd8` |
| vllm | mixed | `s16-kv6-mem82-bdefault` / 1 | 23.45 | 28.54 | `46e8c2ed9b20` |

### High-concurrency diagnostic envelope—not service settings

These are single completed diagnostic cohorts, not three-repeat performance guarantees. Every row fails the unchanged service latency gates. The C=768/1,024 cells use KV=32 GiB and a separately registered 360-second client deadline; C=256/512 use KV=20 GiB and 180 seconds. Total requested memory remains 96 GiB. Changing cache/slot budgets means the whole sequence is not an isolated single-knob comparison.

| Engine | C | Output tok/s | p95 TTFT s | p95 gap s | Peak KV % | Min available GiB |
|---|---:|---:|---:|---:|---:|---:|
| sglang | 256 | 856.61 | 6.243 | 0.521 | 37.0 | 34.07 |
| sglang | 512 | 1277.67 | 11.163 | 0.647 | 79.0 | 33.43 |
| sglang | 768 | 1557.00 | 16.504 | 1.012 | 70.0 | 22.09 |
| sglang | 1024 | 1756.17 | 22.411 | 0.951 | 91.0 | 19.93 |
| vllm | 256 | 846.23 | 4.988 | 0.537 | 39.3 | 36.18 |
| vllm | 512 | 1031.03 | 10.272 | 0.871 | 77.1 | 34.94 |
| vllm | 768 | 1129.96 | 15.977 | 1.199 | 73.3 | 21.08 |
| vllm | 1024 | 1155.31 | 22.056 | 1.537 | 97.3 | 19.33 |

**No universal hardware ceiling was established.** vLLM rises only about 2.2% from C=768 to 1,024 while KV occupancy reaches 97.3%; this is an observed near-plateau with an almost-full KV cache, not proof of a compute maximum. Cache capacity is a plausible remaining constraint. SGLang still rises about 12.8%, with KV occupancy 91% and the compiled/request-slot bound at 1,024. Its maximum remains unresolved beyond this bounded, fixed-recipe experiment. Neither diagnostic endpoint is a usable service point.

### Observation, accounting and attribution limits

- Minimum observed MemAvailable across MoE attempts was **19.33 GiB**, above the unchanged 13-GiB stop. Memory pressure was not zero: the largest sampled full avg10 was **15.58%**. That maximum was the first measured sample of evidence `c8196c03feb9`, shortly after warmup; avg10 includes preceding history. Its full-stall counter increased another 0.562s across the sampled cohort. Do not turn available-memory headroom into a claim of zero pressure.
- The first SGLang C=768 attempt and one vLLM short C=16 repeat stopped when mandatory host observations became unavailable. Their partial/cancelled work has no aggregate throughput. Deletion was acknowledged before resumption. Server-side cancellation is not inferred from client cancellation counters.
- Sampled router waiting counts, engine running/waiting counts, actual client overlap and timing-header components are separate measurements. Selected MoE lease-grant p95 reached 23.425 ms; pre-forward timing was unavailable. More admitted engine work and different native scheduling settings change the envelope, but exact engine/controller overhead attribution is unknown without a matched direct-engine baseline and internal token-batch trace.
- Prefix reuse was disabled and each cell had excluded serial warmup. The reported aggregate rate includes ramp and drain; warmup does not prove a fully populated stationary batch. SSE gaps are not exact token latency, pooled quantiles can hide isolated stalls, and correctness is structural HTTP/SSE/usage/numbered-list validation rather than general model quality.
- Independent peak accelerator-memory measurements remain unknown on this GB10; CPU PSS and system nonavailable memory are not substitutes. No whole-system power/energy efficiency claim is made.

Config `sN-kvK-memM-bB` means engine slots, KV GiB, total requested GiB and engine-specific scheduling budget. `-deadline360` identifies the amended diagnostic timeout. Original defaults and approved engine environments were not modified.

Reproduction: [driver/observer](https://github.com/edurdias/capyctl/issues/211#issuecomment-6435986028), [orchestration/reporting](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443042524), [setup/configurations/commands](https://github.com/edurdias/capyctl/issues/211#issuecomment-6436349184). Evidence-ID supplements: [occupancy/memory](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443586946), [latency tails/stream rates](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443587549), [request/token accounting](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443588115).

### Every indexed MoE attempt

| Evidence ID | Model | Engine | Build (server/host) | Phase/repeat | Config | Workload | Offered C | Output tok/s | p95 TTFT s | p95 chunk gap s | Successful/admitted | Verdict |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 55d752d7fe0c | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 1 | 29.26 | 0.287 | 0.037 | 40/40 | pass |
| 91a82e1e542f | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 2 | 44.59 | 0.406 | 0.048 | 41/41 | pass |
| 7ff37b5df61b | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 4 | 64.07 | 0.641 | 0.065 | 43/43 | pass |
| 658d118fdab6 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 8 | 90.85 | 1.079 | 0.106 | 47/47 | pass |
| 0663306a7fd9 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 1 | 15.10 | 1.953 | 0.040 | 40/40 | pass |
| da9722d70d43 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 2 | 19.76 | 3.691 | 0.055 | 40/40 | pass |
| 3898eadc63c4 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 4 | 23.58 | 7.236 | 0.072 | 40/40 | pass |
| f146592d101c | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 8 | 25.60 | 19.073 | 0.095 | 43/43 | ttft_threshold |
| 4e582bec0151 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 1 | 24.07 | 1.910 | 0.039 | 40/40 | pass |
| 9ae2be691fcd | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 2 | 34.48 | 2.038 | 0.051 | 41/41 | pass |
| 69f629754de6 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 4 | 43.18 | 2.182 | 0.076 | 43/43 | pass |
| 7ac9fec723f7 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 8 | 57.18 | 4.783 | 0.099 | 47/47 | pass |
| 790a15f6c743 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 12 | 116.99 | 0.800 | 0.140 | 51/51 | pass |
| 7df1e7b214d0 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 16 | 138.25 | 1.099 | 0.110 | 55/55 | pass |
| c8293f187053 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 24 | 181.59 | 1.129 | 0.153 | 70/70 | pass |
| 5157e80c532d | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 32 | 215.17 | 1.295 | 0.348 | 91/91 | chunk_gap_threshold |
| 1078c5a0b4cb | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 64 | 360.45 | 1.942 | 0.375 | 180/180 | chunk_gap_threshold |
| 7ca5833915d7 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 128 | 573.07 | 2.912 | 0.426 | 362/362 | ttft_threshold, chunk_gap_threshold |
| dedccf8f0840 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-bdefault | long | 8 | 26.86 | 14.051 | 0.099 | 40/40 | ttft_threshold |
| 4def95519529 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | long | 4 | 24.87 | 6.499 | 0.073 | 41/41 | pass |
| 9098ca797b5e | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | long | 6 | 26.73 | 9.771 | 0.155 | 43/43 | pass |
| 6459d67dda6c | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | long | 8 | 28.47 | 13.122 | 0.098 | 40/40 | ttft_threshold |
| 669da929c92e | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 4 | 45.16 | 1.946 | 0.081 | 43/43 | pass |
| 28d72a2bdeee | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 8 | 58.35 | 3.746 | 0.096 | 47/47 | pass |
| 0d58b93d0484 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 12 | 67.06 | 6.905 | 0.175 | 51/51 | pass |
| 9ae930536195 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 16 | 77.69 | 10.081 | 0.128 | 55/55 | ttft_threshold |
| ce9528b29967 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s128-kv6-mem82-bdefault | short | 24 | 180.69 | 1.154 | 0.139 | 68/68 | pass |
| 26956b0e2050 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv12-mem88-b32768 | long | 6 | 27.27 | 9.698 | 0.090 | 42/42 | pass |
| 4ba7baceb18e | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv12-mem88-b32768 | mixed | 12 | 67.86 | 6.915 | 0.113 | 51/51 | pass |
| b8fdf6c3ccd6 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv12-mem88-b32768 | mixed | 12 | 65.00 | 6.820 | 0.191 | 51/51 | pass |
| 14967236e293 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv12-mem88-b32768 | long | 6 | 27.11 | 9.869 | 0.128 | 42/42 | pass |
| a71967d168c5 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s128-kv6-mem82-bdefault | short | 24 | 177.06 | 1.150 | 0.195 | 70/70 | pass |
| c30fe5f6d2d8 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s128-kv6-mem82-bdefault | short | 24 | 181.15 | 1.172 | 0.131 | 68/68 | pass |
| c8196c03feb9 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv12-mem88-b32768 | long | 6 | 25.99 | 9.990 | 0.089 | 43/43 | pass |
| c09a797135c2 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv12-mem88-b32768 | mixed | 12 | 65.28 | 6.935 | 0.199 | 51/51 | pass |
| b4f9a9841663 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s512-kv20-mem96-bdefault | short | 256 | 856.61 | 6.243 | 0.521 | 720/720 | ttft_threshold, chunk_gap_threshold |
| 176779cbbefa | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s512-kv20-mem96-bdefault | short | 512 | 1277.67 | 11.163 | 0.647 | 1438/1438 | ttft_threshold, chunk_gap_threshold |
| 6aa29156a24c | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv32-mem96-bdefault-deadline360 | short | 768 | unknown | 17.110 | 0.829 | 773/1541 | request_errors, incomplete_usage_or_cancelled_work, minimum_completions_not_reached, ttft_threshold, chunk_gap_threshold, external_STOP |
| eb3495e84db8 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv32-mem96-bdefault-deadline360 | short | 768 | 1557.00 | 16.504 | 1.012 | 2193/2193 | ttft_threshold, chunk_gap_threshold |
| 6e5292e2ffe2 | Qwen3-30B-A3B | sglang | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv32-mem96-bdefault-deadline360 | short | 1024 | 1756.17 | 22.411 | 0.951 | 2916/2916 | ttft_threshold, chunk_gap_threshold |
| a5d94f0f5801 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 1 | 28.96 | 0.291 | 0.037 | 40/40 | pass |
| 75f02b70f37d | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 2 | 42.33 | 0.304 | 0.049 | 41/41 | pass |
| 2e80e369d902 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 4 | 52.25 | 0.490 | 0.081 | 43/43 | pass |
| a23534fb82f8 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | short | 8 | 96.55 | 1.081 | 0.112 | 47/47 | pass |
| f234f31bcdd8 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 1 | 14.51 | 2.066 | 0.040 | 40/40 | pass |
| c48cd9a73c58 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 2 | 17.85 | 3.854 | 0.089 | 41/41 | pass |
| 794f3e95a280 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 4 | 20.09 | 4.157 | 0.470 | 43/43 | chunk_gap_threshold |
| 0a1f978c909f | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | long | 8 | 23.54 | 11.470 | 0.542 | 47/47 | ttft_threshold, chunk_gap_threshold |
| 46e8c2ed9b20 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 1 | 23.45 | 2.047 | 0.039 | 40/40 | pass |
| 52d63f38a1d9 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 2 | 32.36 | 2.115 | 0.054 | 41/41 | pass |
| db182e93c2fe | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 4 | 39.22 | 2.453 | 0.215 | 43/43 | chunk_gap_threshold |
| 1765b8329fb3 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s16-kv6-mem82-bdefault | mixed | 8 | 59.52 | 3.952 | 0.426 | 47/47 | chunk_gap_threshold |
| 0436b83a11cc | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 12 | 120.62 | 0.718 | 0.099 | 51/51 | pass |
| 565d5f2aa629 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 16 | 142.51 | 0.648 | 0.109 | 54/54 | pass |
| f918c7eca330 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 24 | 193.06 | 1.174 | 0.222 | 71/71 | chunk_gap_threshold |
| 8c18fc11f0cf | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 32 | 231.53 | 1.293 | 0.236 | 93/93 | chunk_gap_threshold |
| f72cf561e135 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 64 | 383.92 | 2.134 | 0.258 | 190/190 | ttft_threshold, chunk_gap_threshold |
| b35c105e9b97 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-bdefault | short | 128 | 601.95 | 3.792 | 0.305 | 381/381 | ttft_threshold, chunk_gap_threshold |
| 1ef59a81cee2 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-bdefault | long | 8 | 23.61 | 11.442 | 0.571 | 47/47 | ttft_threshold, chunk_gap_threshold |
| 444e1bf8f97b | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | long | 4 | 22.51 | 6.061 | 0.105 | 41/41 | pass |
| 37b24c372720 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | long | 6 | 27.28 | 8.735 | 0.099 | 43/43 | pass |
| c388f85a756b | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | long | 8 | 29.18 | 11.579 | 0.108 | 46/46 | ttft_threshold |
| 946c31c1b095 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 4 | 39.25 | 3.112 | 0.088 | 43/43 | pass |
| e899206b4d1a | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 8 | 61.09 | 5.746 | 0.094 | 47/47 | pass |
| 91ee36b82f27 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 12 | 72.06 | 9.144 | 0.113 | 51/51 | pass |
| 674500db8827 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s32-kv12-mem88-b32768 | mixed | 16 | 79.15 | 11.136 | 0.241 | 55/55 | ttft_threshold, chunk_gap_threshold |
| 4db4d1f2abbb | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-b8192 | short | 16 | 142.86 | 0.914 | 0.177 | 55/55 | pass |
| db85ad33b276 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s128-kv6-mem82-b8192 | short | 24 | 198.13 | 1.105 | 0.183 | 72/72 | pass |
| 3fa157f73141 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s128-kv6-mem82-b8192 | short | 24 | 199.67 | 1.013 | 0.129 | 72/72 | pass |
| cb907fc62161 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv12-mem88-b32768 | long | 6 | 27.38 | 8.801 | 0.092 | 44/44 | pass |
| 2c2aa9a96ee0 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv12-mem88-b32768 | mixed | 12 | 70.57 | 7.380 | 0.186 | 51/51 | pass |
| b10301327f30 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv12-mem88-b32768 | mixed | 12 | 67.81 | 8.890 | 0.195 | 51/51 | pass |
| c326707282f5 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv12-mem88-b32768 | long | 6 | 24.89 | 10.195 | 0.093 | 43/43 | ttft_threshold |
| 89d63d7c58f6 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s128-kv6-mem82-b8192 | short | 24 | 187.35 | 1.041 | 0.223 | 69/69 | chunk_gap_threshold |
| abcabbcd11c4 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s128-kv6-mem82-b8192 | short | 24 | 200.50 | 1.027 | 0.125 | 72/72 | pass |
| b8cc3861eff4 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv12-mem88-b32768 | long | 6 | 27.48 | 8.715 | 0.094 | 43/43 | pass |
| dbaa61fc3896 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv12-mem88-b32768 | mixed | 12 | 70.74 | 7.412 | 0.226 | 51/51 | chunk_gap_threshold |
| 7d2811f84934 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s512-kv20-mem96-b8192 | short | 256 | 846.23 | 4.988 | 0.537 | 767/767 | ttft_threshold, chunk_gap_threshold |
| 5341e922bc01 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s512-kv20-mem96-b8192 | short | 512 | 1031.03 | 10.272 | 0.871 | 1529/1529 | ttft_threshold, chunk_gap_threshold |
| 6ef540333773 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s128-kv6-mem82-b8192 | short | 20 | 169.36 | 0.966 | 0.211 | 60/60 | chunk_gap_threshold |
| 500ef36ad275 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv12-mem88-b32768 | long | 5 | 26.12 | 6.819 | 0.093 | 44/44 | pass |
| 864a4f8520c8 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s32-kv12-mem88-b32768 | mixed | 10 | 67.33 | 5.658 | 0.173 | 49/49 | pass |
| 0308292248a0 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv12-mem88-b32768 | mixed | 10 | 66.34 | 6.094 | 0.197 | 49/49 | pass |
| 8dbcd18e6c41 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s32-kv12-mem88-b32768 | long | 5 | 25.65 | 7.454 | 0.089 | 44/44 | pass |
| 417783f740d0 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s128-kv6-mem82-b8192 | short | 20 | 172.58 | 0.988 | 0.116 | 61/61 | pass |
| 33f60b05aec0 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s128-kv6-mem82-b8192 | short | 20 | 176.60 | 0.741 | 0.120 | 60/60 | pass |
| 1b0a9fae5ef7 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv12-mem88-b32768 | long | 5 | 26.27 | 6.751 | 0.088 | 44/44 | pass |
| 4a584d68fccf | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s32-kv12-mem88-b32768 | mixed | 10 | 68.02 | 5.379 | 0.146 | 49/49 | pass |
| 20524d0a7bec | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/1 | s128-kv6-mem82-bdefault | short | 16 | 143.55 | 0.722 | 0.193 | 55/55 | pass |
| b397308b5b8f | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s128-kv6-mem82-bdefault | short | 16 | unknown | 0.889 | 0.109 | 50/55 | request_errors, incomplete_usage_or_cancelled_work, external_STOP |
| 0bfb0ebdbe70 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/2 | s128-kv6-mem82-bdefault | short | 16 | 141.43 | 0.723 | 0.161 | 55/55 | pass |
| def5757479d3 | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/3 | s128-kv6-mem82-bdefault | short | 16 | 141.95 | 0.675 | 0.111 | 55/55 | pass |
| 743cd1b86f5f | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv32-mem96-b8192-deadline360 | short | 768 | 1129.96 | 15.977 | 1.199 | 2302/2302 | ttft_threshold, chunk_gap_threshold |
| 82fa8a05402f | Qwen3-30B-A3B | vllm | experimental-681a98d8/experimental-9ec03007 | ceiling/0 | s1024-kv32-mem96-b8192-deadline360 | short | 1024 | 1155.31 | 22.056 | 1.537 | 3065/3065 | ttft_threshold, chunk_gap_threshold |
