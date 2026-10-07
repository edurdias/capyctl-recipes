# Qwen3.8-27B NVFP4 on SGLang 0.5.21, sized for 32 GiB, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | SGLang 0.5.21 in a venv (torch 2.13.0, FlashInfer 0.6.18, sglang-kernel 0.4.7) |
| Model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at `482ca0f3832238542f8f5295dde86b5f22711d80`, NVFP4 with FP8 layers, 21.9 GB |
| Drafter | none |
| CapyCTL | `main` at `0c4ccb4` (prints `capyctl 0.1.2`), release build; `capyctl start standalone` with its default limits |
| Measured | 2026-10-06 |

Qwen3.8-27B for a GPU with 32 GB: the deployment is sized so the model, once
ready, fits in 32 GiB, one request at a time, as the SGLang cookbook's RTX
5090 (32 GB) cell does. It was validated on a GB10, not on a 32 GB card. The
[128 GB recipe](../qwen3.8-27b-nvfp4-gb10/) uses the whole machine and a
drafter. The [TensorFold](../../tensorfold/qwen3.8-27b-nvfp4-32gib-gb10/) and
[vLLM](../../vllm/qwen3.8-27b-nvfp4-32gib-gb10/) recipes sized for the same
32 GiB are next to this one.

### The checkpoint

Of the two NVFP4 exports the SGLang cookbook lists, this uses NVIDIA's, which
packs the `lm_head` to NVFP4 too: its weights are 21.9 GB against 23.8 GB for
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, and the cookbook puts the BF16 head
at about 3.2 GB more at runtime, room that 32 GiB does not have.

### Memory

The deployment runs one request at a time (`max_concurrent_requests: 1`) with
a 1 GiB FP8 KV pool (`memory.kv_cache: 1GiB`), 2048-token prefill chunks, one
CUDA graph (`--cuda-graph-max-bs-decode 1`) and SGLang loading the weights one
file at a time (`--model-loader-extra-config '{"enable_multithread_load": false}'`).

| | |
|---|---|
| Ready footprint | 28.5 GiB after the first (cold) start, 29.3 GiB after a warm start: machine memory in use once ready, less what was in use before the start. SGLang's own GPU allocations are 24.3 GiB |
| Loading peak | 32.3 GiB during the cold start and 33.1 GiB during a warm one (`MemAvailable` drop); CapyCTL measured 32.3 and 33.0 GiB |
| Fits | 32 GiB: yes. 24 GiB: no |

On the GB10's unified memory the loading peak and the engine's CPU-side
memory come out of the same pool as the GPU's. On a discrete card most of
that lands in system RAM, so the ready footprint is the number to compare
with a card's VRAM.

CapyCTL reserves a model's measured loading peak for its later starts, so on
the GB10 the standalone's managed limit (half the machine's memory by
default) has to leave about 33 GiB for it.

### No parking

SGLang 0.5.21 cannot reload modelopt (NVFP4) weights from disk, and a deep
park's wake does that, so the file sets `residency: restart_only`: CapyCTL
stops the model and starts it again when it needs the memory, and
`capyctl park deployment` is refused. Deep park is unavailable for SGLang
modelopt checkpoints on CapyCTL up to `a6b2560`: without `restart_only`, that
build parked the model and the wake failed (`AttributeError: 'Parameter'
object has no attribute 'weight_loader'` in SGLang, and the request was
answered `activation_uncertain`). CapyCTL `main` from `0c4ccb4` refuses deep
park for SGLang modelopt checkpoints (`capability_missing:deep_park`); use
`restart_only`.

## Run it

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

Register the engine and allow the loader option:

```bash
capyctl engine add ~/sglang-0.5.21-venv --approve-option=--model-loader-extra-config
```

```text
Registered sglang (sglang 0.5.21)

  Executable     /home/me/sglang-0.5.21-venv/bin/python3
  Deep park      enabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 2)
  Published      yes
```

The revision is 2 because the vLLM profile was added before it.

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M49KVFCW05HCXA8FVKJKF9TX (reuse --request-id 01M49KVFCW05HCXA8FVKJKF9TX to recover this command)
Deployment qwen38-27b-sglang created (revision 1)

  Deployment ID       01M49KVFDVG6HMEPRZFWDWC1Y0
  Operation           01M49KVFDV56T84M2WX67NXHQV
  Checkpoint digest   being measured
the checkpoint digest of qwen38-27b-sglang is being measured; `capyctl start deployment qwen38-27b-sglang --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen38-27b-sglang --wait
```

```text
Request identity: 01M49KVFJ83TYWFXQZQESXV6VP (reuse --request-id 01M49KVFJ83TYWFXQZQESXV6VP to recover this command)
Waiting for the checkpoint digest of qwen38-27b-sglang to be measured (at most 900s)
Started qwen38-27b-sglang: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

```bash
capyctl status deployment qwen38-27b-sglang
```

```text
NAME                STATE   READY   REVISION   STARTUP    INITIALIZE   LAST OPERATION
qwen38-27b-sglang   ready   1/1     1          55.2 GiB   820s         initialize succeeded

INSTANCE   HOST         STATE   LIFECYCLE   DEVICES   LAST ERROR
0          host-a   ready   active      gpu0      -

Engine  sglang 0.5.21 (/home/me/sglang-0.5.21-venv/bin/python3)
Parsers tool calls: qwen3_coder, reasoning: qwen3 (model family qwen3_5)
note: deployment qwen38-27b-sglang serves /metrics without authentication on its loopback listener (read_only; a known and accepted limitation)
```

Qwen3.8 thinks before it answers. CapyCTL picks the `qwen3` reasoning parser
for this model family (and `qwen3_coder` for tool calls), so SGLang returns
the thinking in `reasoning_content` and the answer in `content`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen38-27b-sglang", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text


Jupiter is the largest planet in our solar system.
```

The park attempt below is from the earlier run of the same file on `a6b2560`:

```bash
capyctl park deployment qwen38-27b-sglang
```

```text
Request identity: 01M49JDTDQ5XBHA8KFWMR9C4H8 (reuse --request-id 01M49JDTDQ5XBHA8KFWMR9C4H8 to recover this command)
error [unsupported]: unsupported_capability: Requested capability is unavailable
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 178 s (the first start of the deployment; weights already in the model store; includes measuring the checkpoint digest) |
| Ready, warm | 174 s (`start` after `stop` finished; SGLang repeats its full startup, CUDA graph capture included) |
| Time to first token | 0.171 s median (0.134 to 0.174) |
| Decode, one stream | 13.0 tokens/s median (13.0 to 13.0) |
| Peak memory | 32.3 GiB measured by CapyCTL on the cold start, 33.0 GiB on the warm one; ready footprint in [Memory](#memory) |
| Park | refused (`residency: restart_only`, above) |
| Concurrency | one request at a time (`max_concurrent_requests: 1`); others wait in SGLang's queue |
| Context | 32,768 tokens, declared |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the model's default; the
first token counted is the first `reasoning_content` token). The three prompts
are the first three of capyctl-bench's prompt set (`explain-tcp`,
`python-lru`, `history-printing`); every request generated all 512 tokens. The
same three requests after the warm start gave 0.176 s and 13.0 tokens/s.

## Benchmark

[`bench/`](bench/) has the capyctl-bench results and report
([`report.html`](bench/report.html), [`summary.md`](bench/summary.md),
[`data.csv`](bench/data.csv)). Context sweep 0.5k to 32k tokens, three runs per
point, `max_tokens: 128`; concurrency 1 and 2 streams (the deployment runs one
request at a time, so more streams only queue), five rounds each,
`max_tokens: 512`; `temperature: 0`, thinking on.

| Context | Time to first token | Decode, one stream |
|---|---|---|
| 0.5k | 0.26 s | 12.9 tokens/s |
| 1k | 0.43 s | 12.9 tokens/s |
| 2k | 0.83 s | 12.9 tokens/s |
| 4k | 1.60 s | 12.8 tokens/s |
| 8k | 3.25 s | 12.7 tokens/s |
| 16k | 6.82 s | 12.6 tokens/s |
| 32k | 15.75 s | 12.2 tokens/s |

| Streams | Together | Each | Time to first token |
|---|---|---|---|
| 1 | 12.9 tokens/s | 12.9 tokens/s | 0.17 s |
| 2 | 12.9 tokens/s | 12.9 tokens/s | 20.05 s |

With two streams the second request waits for the first: the total stays at
12.9 tokens/s and the time to first token is the wait.

The report's memory column (machine memory in use, sampled every 0.5 s) is
left out here: model downloads for other recipes ran on the machine during
this benchmark, and their buffers are in it. SGLang's own GPU allocations
peaked at 24.7 GiB during the benchmark, against 24.3 GiB at rest.

![Summary](bench/summary/summary-wide.png)

The model needs about 21.9 GB in the model store; with the download, the first
start takes as long as the download plus the time above.
