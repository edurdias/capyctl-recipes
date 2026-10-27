## Reproduction boundary, configurations and commands

Use a fresh isolated run on an explicitly authorized host. These are experiment tools, not production changes. Existing approved engine installations are prerequisites; the scripts do not create or modify them. Private SSH addresses, enrollment material and credentials are intentionally omitted.

### Pinned inputs

- Harness checkout: `83f179100d089c66ab38947539f72f583e1ba583`. Follow its `scripts/live/matrix/README.md` for isolated initialization/enrollment and ownership receipts.
- Official baseline: CapyCTL v0.1.8. Executable SHA-256: x86_64 `48c93cd29862350568bd2d00f7e1cc773eb9f4467f74b148bfa775f038054a7c`; aarch64 `970d5d07ee8b262e1ddfafd267532fa96c8e5256bd8df35a916eaefc9b89476a`.
- Native engines: SGLang 0.5.22 and vLLM 0.32.0, unchanged approved environments; one GB10, driver 580.173.02, approximately 121.7 GiB usable system RAM. Managed-memory limit: 104,530,536,038 bytes; protected reserve: 13,066,317,005 bytes; additional observed-MemAvailable stop: 13 GiB.
- Runtime profiles `sglang` and `vllm`, revision 1, select the existing environments. Their `executable` fields are `/home/me/sglang-0.5.22-venv/bin/python3` and `/home/me/vllm-0.32.0-venv/bin/vllm`, with `build_fingerprint` values `0.5.22` and `0.32.0`; both use `cuda_home: /usr/local/cuda` and the managed embedded runtime. Adapt paths/host IDs, not engine versions or safety protections. Retain the harness security settings, including default-on deep-parking protection even though these deployments use restart-only residency. Engine listeners/control stay loopback-only and authenticated. No engine key is read or published.
- Checkpoint hashes/weight bytes are in the model-specific results. Manifest fingerprint strings below are fixture labels: verify the measured, nonprovisional canonical digest in `status-before.json`. `revision: r1` is not an upstream Git revision.
- BF16 runtime dtype, no quantization override, unchanged default KV dtype, context 16,384, CUDA graphs off, speculation off. Both baseline and variants disable reusable prefix/radix caching. MoE retains vLLM `--moe-backend triton`; only MoE requests explicitly disable thinking.

### Private setup and official baseline

Obtain the [driver/observer source](https://github.com/edurdias/capyctl/issues/211#issuecomment-6435986028) and [orchestration/reporting source](https://github.com/edurdias/capyctl/issues/211#issuecomment-6443042524). The commands run from the pinned CapyCTL harness checkout, not this recipes checkout. Save that published source under `target/live/issue211/`, verified official binaries under `x86_64/` and `aarch64/`, and the four manifests below under `fixtures/`. Replace `ENROLLED_HOST_ID` with the authorized host ID. Keep `hosts.env`, enrollment/identity files and API credentials private. Never enable shell tracing. Copy both host sampler scripts to the remote experiment root and adapt their ownership-receipt path consistently.

The environment used was:

```bash
# Source only for issue #211; host B must never be passed to a live command.
export CAPYCTL_MATRIX_LIVE="$PWD/target/live/issue211/matrix"
export CAPYCTL_LOCAL_BIN="$PWD/target/live/issue211/x86_64/capyctl"
export CAPYCTL_REMOTE_TREE=/home/me/capyctl-issue211
export CAPYCTL_REMOTE_BIN=/home/me/capyctl-issue211/capyctl
export CAPYCTL_MATRIX_HOSTS_ENV="$PWD/target/live/issue211/hosts.env"
export CAPYCTL_SERVER_MGMT_PORT=37443
export CAPYCTL_SERVER_INFER_PORT=38443
export CAPYCTL_TIMING_HEADER=1
export CAPYCTL_GROUPS=0
export CAPYCTL_LOCAL_RUN_ROOT="$HOME/capyctl-issue211-control"
export CAPYCTL_REMOTE_RUN_ROOT=/home/me/capyctl-issue211/run
export CAPYCTL_HOST_CUDA_HOME=/usr/local/cuda
```

Use the existing harness for isolated initialization/enrollment, with private bootstrap/control listeners and loopback management/inference listeners. Start only `roles.sh server-up` and `roles.sh host-up "$HOST_A"`. Do not invoke generic two-host up/down/preflight/wait-online helpers. Host B remains excluded.

```bash
source target/live/issue211/env.sh
source scripts/live/matrix/lib.sh
load_run
load_api_key
# Run serially, while both roles still use the official binaries.
python3 target/live/issue211/run_experiment.py --host "$HOST_A" --engine sglang
python3 target/live/issue211/run_experiment.py --host "$HOST_A" --engine vllm
```

The fixed-budget runner performs screening and three alternating baseline/candidate repeat pairs. It uses a 60-second request deadline, 600-second phase cap, at least 60 seconds of admission and at least 40 measured completions. Diagnostics use a 180-second request deadline by default, at least `max(40, 2*C)` completions, and phase caps of 600 seconds for 4B or 1,800 seconds for MoE. The separately registered MoE C=768/1,024 extension uses `--request-timeout 360`; its config labels end in `-deadline360`. This changes no service-eligibility threshold. The manifest's 900-second deadline is not the client measurement deadline.

### Experimental admission builds

After verified deployment deletion, export release commit `63fdfeaa86bc3381d0cc26e22070a5aef0fb1200` into an isolated source directory. Do not edit the production checkout or install these binaries as services.

1. First server extension: change only `crates/capyctl-domain/src/launch.rs`'s `MAX_REQUESTS_PER_DEPLOYMENT` from 32 to 256. Build locally with `cargo build --release --locked --bin capyctl -j 4`, Rust 1.99.0. Recorded SHA-256: `9aa9e3f166b92512c2ac3369fba6e17e43dd64b2a2492b5b8010251325c70912`.
2. Host extension: also change `crates/capyctl-agent/src/ingress.rs`'s ingress `Semaphore::new(64)` to 256. Build on the authorized ARM host with its existing Rust 1.98.1/protoc 27.3, while no model is deployed. Recorded SHA-256: `7b8f3eafe05f34c13b314be1fcf1c8b72bce75a01f292959c252cea963949a17`.
3. Final extension: set both constants to 1,024 and rebuild both roles. Recorded server SHA-256: `681a98d81876e95182bc3e23ec1f6e7f6c55171384b2bcc03f93f08317dc7795`; host SHA-256: `9ec030074303420f44316731bd36a212b25751168ae521e68fddb34ae14450ee`.

Before replacement, verify no deployments remain and stop only the owned role with the ownership helpers. Override `CAPYCTL_LOCAL_BIN` / `CAPYCTL_REMOTE_BIN` for startup: the environment above still names official binaries. The runner's provenance arguments do not replace a running process. Record actual binary digests rather than assuming rebuilds are byte-identical.

For high concurrency, `raise_fd_limit.py` verified the host PID receipt, boot ID, start ticks and executable before raising only that process's soft RLIMIT_NOFILE from 1,024 to 8,192, retaining hard limit 524,288. Do not blindly rerun against a different process or an already-adjusted limit. No system/service limit changed; the adjustment disappears when the owned role exits.

Save `ceiling-1024-provenance.json` with `source_commit`, `changes`, `control_binary_sha256` and `host_binary_sha256` describing your running build. The orchestration scripts consume it. Place the experimental control executable at `ceiling-build/release/capyctl`, or adapt the script paths consistently.

### Reproducing diagnostic rows

Each `sN-kvK-memM-bB` row supplies the generic runner arguments. Set `--model 4` or `--model 30`; omit `--batch` for `bdefault`; retain the numbered repeat and actual running build identity. A `-deadline360` suffix additionally requires `--request-timeout 360`. `--per-workload-concurrency` accepts one concurrency per listed workload, instead of the Cartesian sweep supplied by `--concurrencies`; this permits long/mixed confirmation cells to share an unchanged deployment while retaining separate warmup and accounting. For example:

```bash
python3 target/live/issue211/ceiling.py --host "$HOST_A" --engine sglang --model 4 --slots 32 --kv-gib 16 --request-gib 40 --batch 32768 --concurrencies 8 --workloads long --repeat 1 --control-binary target/live/issue211/ceiling-build/release/capyctl --host-binary-sha256 "$EXPERIMENTAL_HOST_BINARY_SHA256"
```

`EXPERIMENTAL_HOST_BINARY_SHA256` must be the verified running host executable's digest. Use a fresh output tree for independent reruns: scripts retain and skip already indexed cells, preserving interrupted attempts separately. Never overwrite a failed cohort to obtain a passing table.

The orchestration sources encode the staged sweeps and repeat ordering. MoE starts with `moe_baseline.py --host "$HOST_A" --engine sglang`, then vLLM. `moe_extension.py` has service and saturation stages; do not launch saturation without reviewing the service-stage occupancy and memory evidence. Result tables, not optional code branches, identify cells actually executed.

### Evidence and cleanup

Each cell retains protocol argv/source hash, corpus, raw SSE/request records, result, telemetry and before/after status. `report.py` joins results and repeat summaries; `details.py` generates occupancy/memory, latency-tail and accounting tables. Evidence IDs hash the original relative cell paths. Warmup is excluded; aggregate throughput spans first measured dispatch through last completion, including ramp/drain, not a stationary-only window.

HTTP 200 without valid content, DONE, finish reason and usage is an error. An ended client worker is not proof of successful transport or server-side cancellation. Incomplete token accounting yields null aggregate throughput. Routine deployment deletion runs in `finally`; pending cleanup prevents another deployment.

After the last cell, verify `list deployments` is empty, then stop only the owned host-A and local-server roles:

```bash
scripts/live/matrix/roles.sh host-down "$HOST_A"
scripts/live/matrix/roles.sh server-down
```

These helpers validate ownership and wait for role disappearance. Retain receipts and check final available memory. Do not delete unrelated files, environments, processes or another host's workload.

### Base manifests

<details><summary>fixtures/sa-4.yaml</summary>

```json
{
  "schema_version": 1,
  "kind": "deployment",
  "name": "sa-4",
  "host": "ENROLLED_HOST_ID",
  "routes": [
    "sa-4"
  ],
  "runtime_profile": "sglang",
  "runtime_profile_revision": 1,
  "recipe": "standalone",
  "residency": "restart_only",
  "recovery": "reconcile",
  "request_deadline": "900s",
  "model": {
    "source": {
      "type": "local",
      "path": "/home/me/models/qwen3-4b-instruct"
    },
    "content_fingerprint": "sha256:qwen3-4b-instruct",
    "revision": "r1"
  },
  "devices": [
    {
      "id": "gpu0",
      "sharing": "shared"
    }
  ],
  "engine_config": {
    "dtype": "bfloat16",
    "context_length": 16384,
    "max_concurrent_requests": 16,
    "cuda_graphs": false,
    "memory": {
      "request": "25769803776B",
      "kv_cache": "4294967296B"
    }
  }
}
```

</details>

<details><summary>fixtures/va-4.yaml</summary>

```json
{
  "schema_version": 1,
  "kind": "deployment",
  "name": "va-4",
  "host": "ENROLLED_HOST_ID",
  "routes": [
    "va-4"
  ],
  "runtime_profile": "vllm",
  "runtime_profile_revision": 1,
  "recipe": "standalone",
  "residency": "restart_only",
  "recovery": "reconcile",
  "request_deadline": "900s",
  "model": {
    "source": {
      "type": "local",
      "path": "/home/me/models/qwen3-4b-instruct"
    },
    "content_fingerprint": "sha256:qwen3-4b-instruct",
    "revision": "r1"
  },
  "devices": [
    {
      "id": "gpu0",
      "sharing": "shared"
    }
  ],
  "engine_config": {
    "dtype": "bfloat16",
    "context_length": 16384,
    "max_concurrent_requests": 16,
    "cuda_graphs": false,
    "memory": {
      "request": "25769803776B",
      "kv_cache": "4294967296B"
    }
  }
}
```

</details>

<details><summary>fixtures/sa-30.yaml</summary>

```json
{
  "schema_version": 1,
  "kind": "deployment",
  "name": "sa-30",
  "host": "ENROLLED_HOST_ID",
  "routes": [
    "sa-30"
  ],
  "runtime_profile": "sglang",
  "runtime_profile_revision": 1,
  "recipe": "standalone",
  "residency": "restart_only",
  "recovery": "reconcile",
  "request_deadline": "900s",
  "model": {
    "source": {
      "type": "local",
      "path": "/home/me/models/qwen3-30b-a3b"
    },
    "content_fingerprint": "sha256:qwen3-30b-a3b",
    "revision": "r1"
  },
  "devices": [
    {
      "id": "gpu0",
      "sharing": "shared"
    }
  ],
  "engine_config": {
    "dtype": "bfloat16",
    "context_length": 16384,
    "max_concurrent_requests": 16,
    "cuda_graphs": false,
    "memory": {
      "request": "88046829568B",
      "kv_cache": "6442450944B"
    }
  }
}
```

</details>

<details><summary>fixtures/va-30.yaml</summary>

```json
{
  "schema_version": 1,
  "kind": "deployment",
  "name": "va-30",
  "host": "ENROLLED_HOST_ID",
  "routes": [
    "va-30"
  ],
  "runtime_profile": "vllm",
  "runtime_profile_revision": 1,
  "recipe": "standalone",
  "residency": "restart_only",
  "recovery": "reconcile",
  "request_deadline": "900s",
  "model": {
    "source": {
      "type": "local",
      "path": "/home/me/models/qwen3-30b-a3b"
    },
    "content_fingerprint": "sha256:qwen3-30b-a3b",
    "revision": "r1"
  },
  "devices": [
    {
      "id": "gpu0",
      "sharing": "shared"
    }
  ],
  "engine_config": {
    "dtype": "bfloat16",
    "context_length": 16384,
    "max_concurrent_requests": 16,
    "cuda_graphs": false,
    "memory": {
      "request": "88046829568B",
      "kv_cache": "6442450944B"
    },
    "accept_extra_args": true,
    "extra_args": [
      "--moe-backend",
      "triton"
    ]
  }
}
```

</details>

