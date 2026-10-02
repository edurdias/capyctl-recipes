# Qwen3-4B on SGLang 0.5.20, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | SGLang 0.5.20 in a venv (torch 2.13.0+cu130) |
| Model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) at `1cfa9a7208912126459214e8b04321603b3df60c`, bf16, 8.0 GB |
| Drafter | none |
| CapyCTL | `main` at `addde15` (prints `capyctl 0.1.1`); `capyctl start standalone` |
| Measured | 2026-10-02 |

## Run it

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

```bash
capyctl engine add ~/sglang-0.5.20-venv
```

```text
Registered sglang (sglang 0.5.20)

  Executable     /home/me/sglang-0.5.20-venv/bin/python3
  Deep park      enabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 3)
  Published      yes
```

The revision is 3 because this ran after the vLLM recipe's profile was added
and removed.

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M3XG86SD0Z42ZVEHYN2EGYJQ (reuse --request-id 01M3XG86SD0Z42ZVEHYN2EGYJQ to recover this command)
Deployment qwen3-4b-sglang created (revision 1)

  Deployment ID       01M3XG86TACQC7D6Y2P0DH2Z0R
  Operation           01M3XG86TA7CYHPJZZZNJPGVWK
  Checkpoint digest   being measured
the checkpoint digest of qwen3-4b-sglang is being measured; `capyctl start deployment qwen3-4b-sglang --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen3-4b-sglang --wait
```

```text
Request identity: 01M3XG86TQEW2SMAXVC9ZFNTP3 (reuse --request-id 01M3XG86TQEW2SMAXVC9ZFNTP3 to recover this command)
Waiting for the checkpoint digest of qwen3-4b-sglang to be measured (at most 900s)
Started qwen3-4b-sglang: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-4b-sglang", "messages": [{"role": "user", "content": "Name the largest planet in one sentence. /no_think"}]}' \
  | jq -r '.choices[0].message.content'
```

```text
<think>

</think>

The largest planet in our solar system is Jupiter.
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 61 s (weights already in the model store; includes measuring the checkpoint digest) |
| Ready, warm | 66 s (`start` after `stop` finished; SGLang repeats its full startup) |
| Time to first token | 0.073 s median (0.071 to 0.122) |
| Decode, one stream | 22.4 tokens/s median (22.4 to 22.5) |
| Peak memory | 15.9 GiB measured by CapyCTL, against a 21.2 GiB reservation (CapyCTL's default startup size); `MemAvailable` fell by 16.0 GiB at most |
| Context | 29,120 tokens, fitted by CapyCTL |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the chat template's
default). The deployment sets no reasoning parser, so the thinking arrives in
`content`. Every request generated all 512 tokens. A second run after the warm
start gave the same numbers.

`capyctl status deployment` notes that SGLang serves `/metrics` without
authentication on its loopback listener.
