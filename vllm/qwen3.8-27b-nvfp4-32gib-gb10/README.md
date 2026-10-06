# Qwen3.8-27B NVFP4 on vLLM 0.30.0 within 32 GiB, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64, held to 32 GiB (below) |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | vLLM 0.30.0 in a venv (torch 2.13.0+cu130, FlashInfer 0.6.18.post1) |
| Model | [`capyctl/Qwen3.8-27B-NVFP4-2GiB-shards`](https://huggingface.co/capyctl/Qwen3.8-27B-NVFP4-2GiB-shards) at `3619d612d4ba53292c8c0bfb6be6a15f7deca8bc`: [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at `482ca0f3832238542f8f5295dde86b5f22711d80` in 2 GiB files, tensors unchanged; NVFP4 with FP8 layers, 21.9 GB |
| Drafter | none |
| CapyCTL | `main` at `a6b2560` (prints `capyctl 0.1.2`), release build; `capyctl start standalone --set host.resource_policy.memory.system.managed_limit=32GiB` |
| Measured | 2026-10-06 |

Qwen3.8-27B for a machine with 32 GiB for the model, such as a 32 GB GPU: this
recipe holds CapyCTL to 32 GiB of the GB10's memory and runs the model inside
it, with up to 4 requests together and deep parking. The
[128 GB recipe](../qwen3.8-27b-nvfp4-gb10/) uses the whole machine and a
drafter. The [TensorFold](../../tensorfold/qwen3.8-27b-nvfp4-32gib-gb10/) and
[SGLang](../../sglang/qwen3.8-27b-nvfp4-32gib-gb10/) recipes for the same 32 GiB
are next to this one.

### The checkpoint

The deployment downloads
[`capyctl/Qwen3.8-27B-NVFP4-2GiB-shards`](https://huggingface.co/capyctl/Qwen3.8-27B-NVFP4-2GiB-shards) at
`3619d612d4ba53292c8c0bfb6be6a15f7deca8bc`: NVIDIA's
[`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
at `482ca0f3832238542f8f5295dde86b5f22711d80` with every tensor byte-identical,
rewritten from three files of up to 10 GB into eleven of at most 2 GiB (the
embedding table, 2.5 GB, has a file of its own). Its model card lists each
file's sha256. [`convert/reshard.py`](convert/reshard.py) rebuilds it from the
upstream revision:

```bash
hf download nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 --local-dir qwen3.8-27b-nvfp4
python3 convert/reshard.py qwen3.8-27b-nvfp4 qwen3.8-27b-nvfp4-2gib 2
```

The served model is the same; what changes is the load. On the GB10,
vLLM loading the upstream 10 GB files peaked at 50.6 GiB
(measured by CapyCTL), above the 32 GiB limit, so CapyCTL, which reserves a
model's measured peak for its next start, refuses that start; with the
2 GiB files the peak is 29.5 GiB. To use the upstream revision instead,
set `model: {hf: nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80}`
and allow more than 32 GiB.

The live runs used a local copy byte-identical to the pinned revision (the
output of `convert/reshard.py`, deployed by its directory name in the model
store); an `hf:` fetch check follows when the repository is public.

Of the two NVFP4 exports the SGLang cookbook lists, this uses NVIDIA's, which
packs the `lm_head` to NVFP4 too: its weights are 21.9 GB against 23.8 GB for
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, and the cookbook puts the BF16 head
at about 3.2 GB more at runtime, room that 32 GiB does not have.

### Memory

The standalone's managed limit is 32 GiB:

```bash
capyctl start standalone --set host.resource_policy.memory.system.managed_limit=32GiB
```

The deployment asks for a 1.5 GiB FP8 KV cache (`memory.kv_cache: 1536MiB`)
and a 30 GiB startup reservation (`memory.startup`); CapyCTL reserved
31.2 GiB for the start and measured a 29.5 GiB peak. Up to 4 requests run
together (`max_concurrent_requests: 4`); the checkpoint is multimodal and the
deployment serves the text model only (`language_model_only: true`).

## Run it

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

```bash
capyctl engine add ~/vllm-0.30.0-venv
```

```text
Registered vllm (vllm 0.30.0)

  Executable     /home/me/vllm-0.30.0-venv/bin/vllm
  Deep park      enabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 3)
  Published      yes
```

The revision is 3 because the TensorFold and SGLang profiles were added
before it.

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M49CTCRXW5E4WQFCTDQXGE6D (reuse --request-id 01M49CTCRXW5E4WQFCTDQXGE6D to recover this command)
Deployment qwen38-27b-vllm created (revision 1)

  Deployment ID       01M49CTCSX2G437BDPZBVB90DW
  Operation           01M49CTCSXPMNDE74CKYZQM8K8
  Checkpoint digest   being measured
the checkpoint digest of qwen38-27b-vllm is being measured; `capyctl start deployment qwen38-27b-vllm --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen38-27b-vllm --wait
```

```text
Request identity: 01M49CTCX9NH5Y7BJ7E7ZHXAJE (reuse --request-id 01M49CTCX9NH5Y7BJ7E7ZHXAJE to recover this command)
Waiting for the checkpoint digest of qwen38-27b-vllm to be measured (at most 900s)
Started qwen38-27b-vllm: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

Qwen3.8 thinks before it answers. CapyCTL picks the `qwen3` reasoning parser
for this model family (and `qwen3_coder` for tool calls), so vLLM returns the
thinking in `reasoning` and the answer in `content`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen38-27b-vllm", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text


Jupiter is the largest planet in our solar system.
```

### Park and wake

```bash
capyctl park deployment qwen38-27b-vllm
```

```text
Request identity: 01M49GJHBSM4PDA0VWDMNMKM8M (reuse --request-id 01M49GJHBSM4PDA0VWDMNMKM8M to recover this command)
Park requested for qwen38-27b-vllm

  Operation   01M49GJHCNQ3048D4JXATJ5ZHX
```

```bash
capyctl status deployment qwen38-27b-vllm
```

```text
NAME              STATE    READY   REVISION   STARTUP    INITIALIZE   LAST OPERATION
qwen38-27b-vllm   parked   0/1     1          31.2 GiB   820s         park succeeded

INSTANCE   HOST         STATE    LIFECYCLE   DEVICES   LAST ERROR
0          host-a   parked   active      gpu0      -

Engine  vllm 0.30.0 (/home/me/vllm-0.30.0-venv/bin/vllm)
Parsers tool calls: qwen3_coder, reasoning: qwen3 (model family qwen3_5)
warning: deployment qwen38-27b-vllm launches with vLLM development mode on (deep_park enabled (host_policy), sleep mode); exposed controls: /sleep, /wake_up, /is_sleeping, /collective_rpc; mitigations in force: loopback_engine_listener, per_launch_engine_key, engine_key_guard_middleware, no_ingress_or_router_path; not production-safe; use it on isolated hosts only
```

Parked, the model holds 3.6 GiB (CapyCTL's measured parked charge); memory in
use on the machine fell from 30.5 GiB to 9.4 GiB. The next request for it
wakes it: the same question as above, sent to the parked model, answered in
23.7 s, and 21.1 s on a second park and wake; both times include generating
the answer, thinking included (12.7 tokens/s). The park itself returns at once.

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 116 s (the first start of the deployment; weights already in the model store; includes measuring the checkpoint digest and vLLM's compile and warmup) |
| Ready, warm | 59.9 s (`start` after `stop` finished; vLLM repeats its startup) |
| Time to first token | 0.130 s median (0.129 to 0.133) |
| Decode, one stream | 12.7 tokens/s median (12.7 to 12.7) |
| Peak memory | 29.5 GiB measured by CapyCTL, against a 31.2 GiB startup reservation; `MemAvailable` fell by 30.1 GiB at most |
| Parked | 3.6 GiB held (measured); a request wakes it and is answered |
| Concurrency | up to 4 requests together (`max_concurrent_requests: 4`) |
| Context | 32,768 tokens, declared |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the model's default; the
first token counted is the first `reasoning` token). The three prompts are the
first three of capyctl-bench's prompt set (`explain-tcp`, `python-lru`,
`history-printing`); every request generated all 512 tokens. The same three
requests after the warm start gave 0.131 s and 12.6 tokens/s.

## Benchmark

[`bench/`](bench/) has the capyctl-bench results and report
([`report.html`](bench/report.html), [`summary.md`](bench/summary.md),
[`data.csv`](bench/data.csv)). Context sweep 0.5k to 32k tokens, three runs per
point, `max_tokens: 128`; concurrency 1 to 8 streams, five rounds each,
`max_tokens: 512`; `temperature: 0`, thinking on; machine memory in use
(`MemTotal - MemAvailable`, about 3.5 GiB before the model starts) sampled
every 0.5 s.

| Context | Time to first token | Decode, one stream | Memory in use, peak |
|---|---|---|---|
| 0.5k | 0.22 s | 12.7 tokens/s | 29.7 GiB |
| 1k | 0.35 s | 12.7 tokens/s | 29.7 GiB |
| 2k | 1.09 s | 12.7 tokens/s | 29.7 GiB |
| 4k | 2.14 s | 12.6 tokens/s | 29.7 GiB |
| 8k | 4.71 s | 12.5 tokens/s | 29.7 GiB |
| 16k | 9.80 s | 12.3 tokens/s | 29.7 GiB |
| 32k | 21.61 s | 12.0 tokens/s | 29.8 GiB |

| Streams | Together | Each | Time to first token |
|---|---|---|---|
| 1 | 12.6 tokens/s | 12.7 tokens/s | 0.13 s |
| 2 | 24.1 tokens/s | 12.1 tokens/s | 0.24 s |
| 3 | 35.6 tokens/s | 11.9 tokens/s | 0.27 s |
| 4 | 46.7 tokens/s | 11.7 tokens/s | 0.31 s |
| 5 | 30.4 tokens/s | 11.7 tokens/s | 0.31 s |
| 6 | 35.7 tokens/s | 11.7 tokens/s | 0.32 s |
| 7 | 41.2 tokens/s | 11.7 tokens/s | 0.32 s |
| 8 | 46.7 tokens/s | 11.7 tokens/s | 22.19 s |

Four streams decode 3.7 times as many tokens as one. Past four, the extra
requests wait for a running one to finish: at 5 to 8 streams the total falls
back to 30 to 47 tokens/s.

![Summary](bench/summary/summary-wide.png)

The model needs about 21.9 GB in the model store; with the download, the first
start takes as long as the download plus the time above. The status warns that
vLLM development mode is on: that is how CapyCTL parks vLLM (deep parking,
CapyCTL's default for vLLM).
