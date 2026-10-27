# Raw evidence

The full per-request evidence behind every row of [`bench/results.json`](bench/results.json) is attached to the data-only release [`data/concurrency-batching-qwen3-gb10`](https://github.com/edurdias/capyctl-recipes/releases/tag/data/concurrency-batching-qwen3-gb10). It is kept out of Git history because it is about 434 MB compressed (7.5 GB uncompressed).

| Archive | Size | Contents |
|---|---:|---|
| `ceiling-sglang.tar.zst` | 136.3 MB | Qwen3-4B, SGLang: diagnostic, confirmation and saturation cells |
| `ceiling-vllm.tar.zst` | 177.3 MB | Qwen3-4B, vLLM: diagnostic, confirmation and saturation cells |
| `ceiling-sglang-30b-moe.tar.zst` | 59.2 MB | Qwen3-30B-A3B, SGLang: all MoE cells |
| `ceiling-vllm-30b-moe.tar.zst` | 62.4 MB | Qwen3-30B-A3B, vLLM: all MoE cells |
| `fixed-budget-runs.tar.zst` | 18.6 MB | Qwen3-4B fixed-budget screening and repeated baseline/candidate pairs (official v0.1.8) |
| `setup-and-sources.tar.zst` | 0.5 MB | Driver, observer and orchestration source, base manifests, protocol, provenance, setup smokes, sanitized host config |

[`bench/raw-index.json`](bench/raw-index.json) maps each `evidence_id` to its archive and cell directory. [`bench/SHA256SUMS`](bench/SHA256SUMS) lists the archive checksums.

## Download and verify

```bash
gh release download data/concurrency-batching-qwen3-gb10 -R edurdias/capyctl-recipes -D raw
cd raw && sha256sum -c ../bench/SHA256SUMS
zstd -dc --long=27 ceiling-vllm.tar.zst | tar -xf -
```

`--long=27` is required: the archives use a 128 MiB window.

## What each cell directory holds

| File | Contents |
|---|---|
| `requests.jsonl` | One record per request: send/end times, HTTP status and headers, CapyCTL timing, usage, assembled content, chunk offsets and every SSE chunk as received (`raw_chunks`, base64) |
| `result.json` | The cohort summary that `bench/results.json` was built from |
| `telemetry.jsonl` | Management load samples (router, engine running/waiting, KV use) and host memory, pressure and process samples |
| `status-before.json`, `status-after.json` | Deployment status around the cohort, including the measured checkpoint identity |
| `protocol.json`, `corpus.json`, `driver-output.txt` | The driver's exact arguments and source hash, the prompt corpus, and its console output |

Directories ending in `-attemptN` are retried cells; interrupted attempts are retained, as in the tables.

## Sanitization

The archives are the original files with only these substitutions:

- the operator's home directory became `/home/me`;
- the two machine names became `host-a` and `control-host`;
- every non-loopback IPv4 address became `192.0.2.1`;
- the GPU's unique ID became `GPU-00000000-0000-0000-0000-000000000000`.

The private harness files (host enrollment, SSH and address configuration) are not included. No engine key or other credential was ever recorded. CapyCTL's random host, deployment and request IDs, checkpoint digests and engine environment names are unchanged. Raw SSE chunks were checked for the same identifiers and needed no change, so their bytes are as received.
