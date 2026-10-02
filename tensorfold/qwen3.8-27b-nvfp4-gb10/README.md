# Qwen3.8-27B NVFP4 on TensorFold 0.6.1 with DFlash2 drafts, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` (TensorFold builds its kernels with it) |
| Engine | TensorFold 0.6.1 in a venv (torch 2.13.0+cu130) |
| Model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at `482ca0f3832238542f8f5295dde86b5f22711d80`, NVFP4 with FP8 layers, 21.9 GB |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, 3.8 GB |
| CapyCTL | `main` at `e4a6e74` (prints `capyctl 0.1.1`); `capyctl start standalone` |
| Measured | 2026-10-02 |

This recipe needs CapyCTL newer than the 0.1.1 release: `main` at `e4a6e74`
or later, until the next release. It uses two changes made after 0.1.1:
`capyctl engine add --approve-option/--approve-path`, which allows the
`--drafter` option, and the first-request kernel build, without which the
first start fails while TensorFold builds the kernels its first request needs.
The deployment file itself validates with the 0.1.1 release, which is what
this repository's check runs.

TensorFold 0.6.1 runs the checkpoint in its own precision on compute
capability 12.x; the engine log says so at every start:

```text
[tensorfold] precision: checkpoint (the checkpoint's own math: its NVFP4 layers FP4 x FP4, per-16 scales under its static input scales; its FP8 layers FP8 x FP8)
```

## Run it

CapyCTL does not download drafters. Download it into a directory of drafters:

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir ~/drafters/Qwen3.8-27B-DFlash2
```

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

Register the engine and allow the drafter option for paths inside that
directory:

```bash
capyctl engine add ~/tensorfold-0.6.1-venv \
  --approve-option=--drafter --approve-path /home/me/drafters
```

```text
Registered tensorfold (tensorfold 0.6.1)

  Executable     /home/me/tensorfold-0.6.1-venv/bin/tensorfold
  Deep park      disabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 1)
  Published      yes
```

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M3YAGSSMQ0F7S8P3TX1VRHSF (reuse --request-id 01M3YAGSSMQ0F7S8P3TX1VRHSF to recover this command)
Deployment qwen38-27b created (revision 1)

  Deployment ID       01M3YAGSTK2ASGTDXMS4GW3D9H
  Operation           01M3YAGSTK2FNN8BBSYEARFRNW
  Checkpoint digest   being measured
the checkpoint digest of qwen38-27b is being measured; `capyctl start deployment qwen38-27b --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen38-27b --wait
```

```text
Request identity: 01M3YAGSV1KGX3VY5QVXR3KV1A (reuse --request-id 01M3YAGSV1KGX3VY5QVXR3KV1A to recover this command)
Waiting for the model source of qwen38-27b to be downloaded and verified (at most 1800s)
Started qwen38-27b: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

Qwen3.8 thinks before it answers and returns the thinking in
`reasoning_content`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen38-27b", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text
Jupiter is the largest planet in our solar system.
```

The response's `tensorfold` record counts the drafts:

```json
{"accepted": 53, "cached": 0, "decode_s": 0.7821099940047134, "drafted": 120, "drafts": true, "min_rows": 16, "prefill_s": 0.11490244200103916, "rounds": 8, "token_sha": "18388734e4b6"}
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 119 s (fresh state directory, so no kernel cache: includes verifying the weights, measuring the checkpoint digest, loading in 32 s and building five CUDA extensions) |
| Ready, warm | 10 s (`start` after `stop` finished; the kernels are reused) |
| Time to first token | 0.11 s median (0.107 to 0.131) |
| Decode, one stream | 48.2 tokens/s median (33.8 to 60.6), DFlash2 drafts on |
| Peak memory | 30.0 GiB measured by CapyCTL, against the declared 36 GiB `cold` and 34 GiB `ready` reservations; `MemAvailable` fell by 30.1 GiB at most |
| Context | 32,768 tokens, declared |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the model's default; the
first token counted is the first `reasoning_content` token). Two requests
generated all 512 tokens; the third finished on its own at 377. The same three
requests after the warm start gave the same numbers (0.109 s, 48.1 tokens/s).

The cold start and the first two runs used 56 GiB reservations. The 36 GiB
and 34 GiB in `deployment.yaml` were then set from the measured peak and
checked with a third start and run: 28.4 GiB peak, 48.1 tokens/s.

### What the drafter does

The same deployment without the drafter, on the same machine, the same
three requests:

| | DFlash2 drafts | No drafts |
|---|---|---|
| Decode, one stream | 48.2 tokens/s (33.8 to 60.6) | 11.9 tokens/s (11.8 to 11.9) |
| Time to first token | 0.11 s | 0.11 s |
| Peak memory (CapyCTL) | 30.0 GiB | 20.9 GiB |

Drafts make decode about 4 times faster and cost about 9 GiB. The outputs
were identical token for token (the same `token_sha` in each pair). Accepted
of drafted, per request: 423 of 1,336, 354 of 2,355 and 294 of 1,230, which
is 5.8, 3.3 and 4.6 tokens per round; decode speed follows that.

Leaving the drafter out of the deployment does not give you the no-drafts
numbers: CapyCTL then starts TensorFold with `--drafter none`, and
TensorFold 0.6.1 refuses to start this model that way:

```text
tensorfold: Qwen3.8 dense's CUDA engine drafts with z-lab/Qwen3.8-27B-DFlash2, which is not here: without it every round would decode one token. Run `tensorfold pull z-lab/Qwen3.8-27B-DFlash2` once (on both machines for --tp 2), or pass --no-drafts for the serial reference
```

The no-drafts run used a second profile that allows `--no-drafts`
(`capyctl engine add ~/tensorfold-0.6.1-venv --name tensorfold-nodraft
--approve-option=--no-drafts`) and `engine: tensorfold-nodraft` with
`extra_args: [--no-drafts]` in place of the drafter.

A downloaded copy needs about 21.9 GB in the model store and 3.8 GB for the
drafter; with the download, the first start takes as long as the download plus
the time above.
