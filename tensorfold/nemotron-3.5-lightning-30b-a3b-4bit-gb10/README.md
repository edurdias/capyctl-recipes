# Nemotron 3.5 Lightning 30B-A3B 4-bit on TensorFold 0.6.1, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` (TensorFold builds its kernels with it) |
| Engine | TensorFold 0.6.1 in a venv (torch 2.13.0+cu130) |
| Model | [`Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`](https://huggingface.co/Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit) at `d9d758fb83953437f7263256b0d96157e2a348b8`, MLX affine 4-bit, 18.5 GB |
| Drafter | none external; the checkpoint's built-in MTP heads draft (`"drafts":true`) |
| CapyCTL | `main` at `addde15` (prints `capyctl 0.1.1`); `capyctl start standalone` |
| Measured | 2026-10-02 |

## Run it

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

```bash
capyctl engine add ~/tensorfold-0.6.1-venv
```

```text
Registered tensorfold (tensorfold 0.6.1)

  Executable     /home/me/tensorfold-0.6.1-venv/bin/tensorfold
  Deep park      disabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 5)
  Published      yes
```

The revision is 5 because this ran after the vLLM and SGLang recipes' profiles
were added and removed.

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M3XGSXG2W4Y49N51QXQPZT5F (reuse --request-id 01M3XGSXG2W4Y49N51QXQPZT5F to recover this command)
Deployment nemotron-30b created (revision 1)

  Deployment ID       01M3XGSXH1FTZPAN4FDBRXAPZV
  Operation           01M3XGSXH185X54Y3HA1M4JHCV
  Checkpoint digest   being measured
the checkpoint digest of nemotron-30b is being measured; `capyctl start deployment nemotron-30b --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment nemotron-30b --wait
```

```text
Request identity: 01M3XGSXHC22SAG77VB2H5ZJ3X (reuse --request-id 01M3XGSXHC22SAG77VB2H5ZJ3X to recover this command)
Waiting for the model source of nemotron-30b to be downloaded and verified (at most 1800s)
Started nemotron-30b: ready

  Ready       1/1
  Hosts       spark
  Operation   initialize succeeded
```

Nemotron thinks before it answers and returns the thinking in
`reasoning_content`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "nemotron-30b", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text
Jupiter is the largest planet in our solar system.
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 104 s (weights already in the model store; includes verifying them and measuring the checkpoint digest) |
| Ready, warm | 8 s (`start` after `stop` finished) |
| Time to first token | 0.063 s median (0.059 to 0.073) |
| Decode, one stream | 132 tokens/s median (124 to 145), MTP drafts on |
| Peak memory | 27.6 GiB measured by CapyCTL, against the declared 32 GiB `cold` and 30 GiB `ready` reservations; `MemAvailable` fell by 27.6 GiB at most |
| Context | 32,768 tokens, declared |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the model's default; the
first token counted is the first `reasoning_content` token). Every request
generated all 512 tokens. A second run after the warm start gave the same
numbers (0.059 s, 132 tokens/s). Decode varies with the prompt because the
number of accepted drafts does.

A downloaded copy needs about 18.5 GB in the model store; with the download,
the first start takes as long as the download plus the time above.
