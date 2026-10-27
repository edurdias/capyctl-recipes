# Native TP2: direct engine vs CapyCTL on two RTX A6000s

Qwen3-30B-A3B on the same two GPUs, with each engine launched directly and
through CapyCTL. This comparison uses the retained native multi-GPU MG9 run,
not the single-GPU recipes or capyctl-bench. It measures aggregate output
throughput at matched concurrency, not decode-only speed or time to first token.

## Throughput

Output tokens/s across all concurrent requests. Change is
`100 × (CapyCTL / direct − 1)`, calculated from the retained two-decimal
throughput values; positive means the CapyCTL sample was higher.

| Engine | Concurrent requests | Direct engine | Through CapyCTL | Change |
|---|---:|---:|---:|---:|
| vLLM 0.32.0 | 1 | 120.49 | 121.62 | +0.94% |
| vLLM 0.32.0 | 4 | 232.82 | 219.41 | -5.76% |
| vLLM 0.32.0 | 8 | 453.02 | 467.62 | +3.22% |
| SGLang 0.5.22 | 1 | 123.74 | 121.00 | -2.21% |
| SGLang 0.5.22 | 4 | 334.49 | 327.25 | -2.16% |
| SGLang 0.5.22 | 8 | 521.16 | 513.55 | -1.46% |

In these six matched points, the observed differences range from -5.76% to
+3.22%. They do not establish a generalized speedup or a precise estimate of
router overhead. All 156 timed requests succeeded and returned 256 output
tokens each: 3, 12 and 24 requests per path/engine at concurrency 1, 4 and 8.

[bench/results.json](bench/results.json) contains the twelve measured summaries,
source-file hashes, settings, computed percentages and separate parking samples.
No generated chart or capyctl-bench report is implied by this custom harness data.

## Setup

| Item | Measured configuration |
|---|---|
| GPUs | 2× NVIDIA RTX A6000, nominal 48 GB each; NVIDIA telemetry reports 49,140 MiB per card |
| Execution | Native, single-host tensor parallelism across both GPUs (TP2), not containers |
| CapyCTL | `capyctl 0.1.8`, measured source commit [`354199f8cb7647cecf055c3a39d6343c88375d41`](https://github.com/edurdias/capyctl/commit/354199f8cb7647cecf055c3a39d6343c88375d41) |
| vLLM | 0.32.0; PyTorch 2.13.0+cu130; PyTorch CUDA 13.0 |
| SGLang | 0.5.22; PyTorch 2.14.1+cu130; PyTorch CUDA 13.0 |
| Model | [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39), revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Context limit | 32,768 tokens in both engines |
| Measurement date | 2026-10-27 UTC; retained run folder is named `2026-10-26-rerun5` |

The standard CapyCTL executable used by MG9 has SHA-256
`d0b8598b263466b0c95a0f699eb57a3b0214b5d2b5ec3615e8b3dda93a259ff0`.
These are measurements of that build, not measurements of the later
qualification cutover at `48591155` or a future release.

The matched launch settings include:

- **vLLM:** tensor parallel 2, pipeline parallel 1, multiprocessing executor,
  `gpu_memory_utilization=0.75`, `kv_cache_memory_bytes=4294967296`,
  `max_model_len=32768`, `max_num_seqs=32`, `reasoning_parser=qwen3`,
  `tool_call_parser=hermes`.
- **SGLang:** TP2, PP1, DP1, `context_length=32768`,
  `max_total_tokens=43690`, `mem_fraction_static=0.7091`,
  `reasoning_parser=qwen3`, `tool_call_parser=qwen25`;
  memory saver and both weights-backup flags disabled for MG9.

These settings are matched **within each engine pair**, not a claim that the
two engines have identical memory/scheduling settings. The retained normalized
effective-argument objects are equal between direct and CapyCTL for each engine.
The publication includes only an allowlisted settings subset, not raw launch
arguments, filesystem locations, GPU identifiers or authentication material.

## Method

The benchmark implementation is
[`mg.py`, `cmd_bench`](https://github.com/edurdias/capyctl/blob/354199f8cb7647cecf055c3a39d6343c88375d41/scripts/live/multi-gpu/mg.py);
launch matching and run order are in
[`remote.sh`, `bench_pair`](https://github.com/edurdias/capyctl/blob/354199f8cb7647cecf055c3a39d6343c88375d41/scripts/live/multi-gpu/remote.sh).

1. Start the engine through CapyCTL and measure concurrency 1, then 4, then 8.
   Capture the normalized effective engine arguments and stop the deployment.
2. Launch the same installed engine directly, with the captured arguments and
   the same two GPUs. Match the loopback rendezvous environment
   (`NCCL_SOCKET_IFNAME=lo`, `GLOO_SOCKET_IFNAME=lo`,
   `VLLM_HOST_IP=127.0.0.1`), CUDA/toolchain paths and offline model access.
   Wait for its models endpoint. Require the captured effective-argument
   objects to match, then measure concurrency 1, 4 and 8 in that order.
   The direct path bypasses the CapyCTL router and uses no API key;
   the CapyCTL path uses its authenticated inference endpoint.
3. For every engine/path/concurrency point, send one uncounted warm-up request,
   then three sequential rounds. Each round starts `C` client threads and waits
   for all of them before starting the next round: this is a finite burst
   benchmark, not a continuously replenished closed-loop load test.
4. Each request is a **non-streaming** `POST /v1/chat/completions` with one user
   message, `temperature=0`, `max_tokens=256`, and `ignore_eos=true`.
   No explicit thinking toggle, top-p, top-k or seed is set. The prompt is:

   ```text
   Request {i}: Write a long story about a lighthouse keeper, at least two thousand words.
   ```

   The warm-up uses `i=-1`; measured requests use `i=0` through `3C-1`.
   The prompt token count is not retained. No `/no_think` suffix is added to
   this benchmark (the separate parking canary does use one).
5. Sum the `usage.completion_tokens` values and divide by wall time spanning
   all three timed rounds, including thread launch/join, request handling,
   prefill and full response reads. The warm-up and engine startup are excluded.
   Each output JSON retains counters, wall seconds and throughput rounded to
   two decimals. Throughput is calculated before rounding wall time, so
   dividing by the published rounded wall seconds need not recover it exactly.

## Provenance and limits

The retained attempt is `20261027T002747520588460-2674047`, under
`target/live/multi-gpu/2026-10-26-rerun5/evidence/attempts/` in the CapyCTL
checkout. Its `raw/` directory supplies
`mg9-{vllm,sglang}-{hand,capyctl}-c{1,4,8}.json`, matching normalized
`*args.json`, `versions.txt`, and the native lifecycle/GPU samples.
MG9 ran around 01:16–01:39 UTC; the attempt completed at 02:44 UTC.
The checked-in JSON is a sanitized extraction, not a copy of the full private
run directory. Original logs and unsanitized launch material are not published.

- One host, one model revision, one short prompt pattern and one measured
  aggregate per engine/path/concurrency. Three rounds within an aggregate are
  not three independent experiment repetitions.
- CapyCTL always ran before the direct launch; the order was not randomized.
  Warm-up, caches, clocks and host scheduling can affect the differences.
- No per-request timing distribution, TTFT, decode-only speed, output-equality
  test or confidence interval is retained by MG9. This is not a model-quality
  comparison or an engine ranking.
- The harness's throughput acceptance condition is `CapyCTL/direct >= 0.90`
  at each concurrency. Passing that gate is not evidence of zero overhead.
- Parking samples below are separate lifecycle runs, not memory or latency
  measured during the MG9 throughput sweep.

## Separate parking measurements

MG4/MG5 tested **deep** residency for five cycles each; MG6 tested
**host-backed** residency for three cycles per engine, with the same model,
TP2 and 32,768-token context. These rows used the **qualification** executable,
not MG9's standard executable: `capyctl-qualify 0.1.8`, SHA-256
`0ebc5aa42ed67a81aeac2326d7ecbce4cd529fd383191b3729f963a6540552a7`.
The qualification build enabled multi-GPU deep parking for vLLM and SGLang
at the measured source revision. The later support-table cutover is not a
second set of measurements.

### Owned GPU memory

These are NVIDIA per-process owned-memory samples summed for the deployment
on each card (`owned_mib`), not total card usage, reservations or peak memory.
The woken samples were taken after a successful canary; the parked samples
after the lifecycle reached `parked`. Values were identical across the
recorded cycles. Both cards had the same value in each sample.

| Row / residency | Engine | Cycles | Woken, each card | Parked, each card | Woken, both cards total | Parked, both cards total |
|---|---|---:|---:|---:|---:|---:|
| MG4 / deep | vLLM | 5 | 34,238 MiB | 740 MiB | 68,476 MiB | 1,480 MiB |
| MG5 / deep | SGLang | 5 | 34,320 MiB | 2,710 MiB | 68,640 MiB | 5,420 MiB |
| MG6 / host-backed | vLLM | 3 | 34,238 MiB | 740 MiB | 68,476 MiB | 1,480 MiB |
| MG6 / host-backed | SGLang | 3 | 34,320 MiB | 2,710 MiB | 68,640 MiB | 5,420 MiB |

Relative to the woken samples, owned GPU memory was 97.84% lower for vLLM
and 92.10% lower for SGLang when parked. These are not percentages of total
card capacity and do not mean all VRAM was released.

Deep and host-backed have the same observed parked GPU footprint here, but
different host-memory accounting. The parked **system-domain reservation**
was 7,516,192,768 bytes (7 GiB) for deep and 101,024,922,976 bytes
(about 94.09 GiB) for host-backed, **per deployment, not per GPU**.
Those ledger reservations are not measurements of actual host RSS.
The parked GPU reservation was 2,415,919,104 bytes per card for vLLM and
2,841,640,960 bytes per card for SGLang; it is distinct from the owned-memory
measurements above. The host-backed fixture used 80% managed and parked
system-memory limits with a 20% free reserve; the harness requires at least
120 GiB of host RAM for those rows.

### Lifecycle timings

Seconds, mean with observed minimum–maximum in parentheses. These are
differences between retained event timestamps: `park_accepted` to
`parked_committed`, and `restore_accepted` to `restored_committed`.
They are not CLI acknowledgment times, request latency, model cold-start
time or an SLA. Per-cycle values at millisecond resolution are in the JSON.

| Residency | Engine | Cycles | Park seconds | Restore seconds |
|---|---|---:|---:|---:|
| Deep | vLLM | 5 | 8.52 (8.49–8.56) | 89.88 (88.48–92.02) |
| Deep | SGLang | 5 | 0.81 (0.78–0.87) | 17.26 (16.33–18.40) |
| Host-backed | vLLM | 3 | 21.18 (12.92–37.69) | 13.96 (13.82–14.04) |
| Host-backed | SGLang | 3 | 20.98 (17.91–23.28) | 8.76 (8.59–9.00) |

vLLM host-backed parking took 37.687 seconds on the first cycle, then
12.945 and 12.917 seconds. The mean alone hides that difference. Deep and
host-backed were separate deployments, not interleaved paired timing trials.
After every wake the harness required HTTP 200 and the canary answer `42`.
The canary check is not a throughput benchmark or a full correctness test.

Memory and reservation sources are
`MG4-deep-{park,wake}{1..5}-gpus.json`,
`MG5-deep-{park,wake}{1..5}-gpus.json`,
`MG6-{vllm,sglang}-{park,wake}{1..3}-gpus.json`, and the matching
`*-park*-charge.json` files. Durations come from `events-s2.jsonl`,
matched to deployment IDs from the corresponding `*-park1-status.json`.
The published data drops those IDs and retains row/engine/residency/cycle,
numeric values and source hashes. These qualification observations do not
establish production safety, performance on other hosts, or a universal
parking/restore latency.
