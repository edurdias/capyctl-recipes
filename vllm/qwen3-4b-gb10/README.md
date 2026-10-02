# Qwen3-4B on vLLM 0.30.0, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | vLLM 0.30.0 in a venv (torch 2.13.0+cu130) |
| Model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) at `1cfa9a7208912126459214e8b04321603b3df60c`, bf16, 8.0 GB |
| Drafter | none |
| CapyCTL | `main` at `addde15` (prints `capyctl 0.1.1`), the first commit that verifies vLLM 0.30.0; `capyctl start standalone` |
| Measured | 2026-10-02 |

## Run it

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

```bash
capyctl engine add ~/vllm-0.30-venv
```

```text
Registered vllm (vllm 0.30.0)

  Executable     /home/me/vllm-0.30-venv/bin/vllm
  Deep park      enabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 1)
  Published      yes
```

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M3XG05ZNYANYS6XM479DSAS0 (reuse --request-id 01M3XG05ZNYANYS6XM479DSAS0 to recover this command)
Deployment qwen3-4b-vllm created (revision 1)

  Deployment ID       01M3XG060JPKS9675CB4TKZDS8
  Operation           01M3XG060JF6Y1N43MYHZQZDBF
  Checkpoint digest   being measured
the checkpoint digest of qwen3-4b-vllm is being measured; `capyctl start deployment qwen3-4b-vllm --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen3-4b-vllm --wait
```

```text
Request identity: 01M3XG060XW019MJCDJA5DSRAT (reuse --request-id 01M3XG060XW019MJCDJA5DSRAT to recover this command)
Waiting for the model source of qwen3-4b-vllm to be downloaded and verified (at most 900s)
Started qwen3-4b-vllm: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-4b-vllm", "messages": [{"role": "user", "content": "Name the largest planet in one sentence. /no_think"}]}' \
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
| Ready, cold | 46 s (weights already in the model store; includes measuring the checkpoint digest) |
| Ready, warm | 34 s (`start` after `stop` finished) |
| Time to first token | 0.052 s median (0.051 to 0.066) |
| Decode, one stream | 22.6 tokens/s median (22.6 to 22.7) |
| Peak memory | 20.7 GiB measured by CapyCTL, against a 21.2 GiB reservation (CapyCTL's default startup size); `MemAvailable` fell by 21.5 GiB at most |
| Context | 29,104 tokens, fitted by CapyCTL |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the chat template's
default). The deployment sets no reasoning parser, so the thinking arrives in
`content`. Every request generated all 512 tokens. A second run after the warm
start gave the same numbers (0.054 s, 22.5 tokens/s).

The model runs with deep parking on (CapyCTL's default for vLLM), so
`capyctl status deployment` warns that vLLM development mode is on.
