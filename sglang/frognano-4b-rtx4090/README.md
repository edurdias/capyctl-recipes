# FrogNano-4B-2609 on SGLang 0.5.21, one RTX 4090 Laptop GPU

| | |
|---|---|
| Hardware | 1x NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB (16376 MiB), compute capability 8.9; Intel Core i9-14900HX, 64 GB RAM, x86_64 |
| System | Ubuntu 26.04, NVIDIA driver 615.71.09, CUDA 13.4 toolkit in `/usr/local/cuda` |
| Engine | SGLang 0.5.21 in a venv |
| Model | [`microsoft/FrogNano-4B-2609`](https://huggingface.co/microsoft/FrogNano-4B-2609) at `b90468c11a913c1916b4542b4b7a530ec42024a4`, BF16, 9.3 GB |
| Drafter | none |
| CapyCTL | 0.1.2; `capyctl start standalone` |
| Measured | 2026-10-04 |

This recipe needs CapyCTL 0.1.2 or later. CapyCTL picks the tool-call and
reasoning parsers from the model family (`qwen3_coder` and `qwen3` for
`qwen3_5`), sizes SGLang to fit a 16 GB card and turns SGLang's CUDA graphs on (the
0.1.2 default), so the deployment file sets nothing beyond the engine and the model. With deep
parking (CapyCTL's default), the parked model keeps 1.5 GiB of the card.

The same checkpoint on vLLM: [vllm/frognano-4b-rtx4090](../../vllm/frognano-4b-rtx4090/).
In 4-bit on TensorFold: [tensorfold/frognano-4b-mlx-4bit-rtx4090](../../tensorfold/frognano-4b-mlx-4bit-rtx4090/).

## Run it

The API key comes from the credentials file the start banner names:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
```

```bash
capyctl engine add ~/sglang-0.5.21-venv
```

```text
Registered sglang (sglang 0.5.21)

  Executable     /home/me/sglang-0.5.21-venv/bin/python3
  Deep park      enabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 1)
  Published      yes
```

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M44NAQ3A7THJV0DQ69BFJD52 (reuse --request-id 01M44NAQ3A7THJV0DQ69BFJD52 to recover this command)
Deployment frognano-4b-sglang created (revision 1)

  Deployment ID       01M44NAQ3K8DHN33MH3VXT5253
  Operation           01M44NAQ3KADJ6DTBYS5NJ0WQF
  Checkpoint digest   being measured
the checkpoint digest of frognano-4b-sglang is being measured; `capyctl start deployment frognano-4b-sglang --wait` waits for it and starts the deployment
```

The weights were already in the model store from an earlier deploy of this
file, so this start only measured the checkpoint digest; on a fresh machine it
downloads the 9.3 GB checkpoint first (7 minutes here).

```bash
capyctl start deployment frognano-4b-sglang --wait
```

```text
Request identity: 01M44NAQ4RKF23WQ39PM3KVHNF (reuse --request-id 01M44NAQ4RKF23WQ39PM3KVHNF to recover this command)
Waiting for the checkpoint digest of frognano-4b-sglang to be measured (at most 900s)
Started frognano-4b-sglang: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

```bash
capyctl status deployment frognano-4b-sglang
```

```text
NAME                 STATE   READY   REVISION   STARTUP                   INITIALIZE   LAST OPERATION
frognano-4b-sglang   ready   1/1     1          14.5 GiB (+4.0 GiB RAM)   700s         initialize succeeded

INSTANCE   HOST     STATE   LIFECYCLE   DEVICES   LAST ERROR
0          host-a   ready   active      gpu0      -

Engine  sglang 0.5.21 (/home/me/sglang-0.5.21-venv/bin/python3)
Parsers tool calls: qwen3_coder, reasoning: qwen3 (model family qwen3_5)
Running limited to 7 requests by the state cache
note: deployment frognano-4b-sglang serves /metrics without authentication on its loopback listener (read_only; a known and accepted limitation)
```

FrogNano thinks before it answers; SGLang returns the thinking in
`reasoning_content`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "frognano-4b-sglang", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text


Jupiter is the largest planet in our solar system.
```

Tool calls come back structured, with `finish_reason: tool_calls`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "frognano-4b-sglang", "messages": [{"role": "user", "content": "What is the weather in Lisbon, in celsius?"}],
       "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]}' \
  | jq -c '.choices[0] | {finish_reason, tool_calls: .message.tool_calls}'
```

```json
{"finish_reason":"tool_calls","tool_calls":[{"function":{"arguments":"{\"city\": \"Lisbon\", \"unit\": \"celsius\"}","name":"get_weather"},"id":"call_3bb92942ef104fc0958954a3","type":"function"}]}
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 40 s (`start` after `deploy model`, weights already in the model store; includes measuring the checkpoint digest) |
| Ready, warm | 42 s (`start` after `stop` finished) |
| Time to first token | 0.045 s median (0.045 to 0.080) |
| Decode, one stream | 62.5 tokens/s median (62.4 to 62.5) |
| Peak memory | 13.3 GiB of GPU memory, whole-card `nvidia-smi`, against the 14.5 GiB GPU and 4 GiB RAM reservation CapyCTL sized for the card (its default startup size) |
| Concurrency | up to 7 requests: `capyctl status` prints "Running limited to 7 requests by the state cache" |
| Context | 63,920 tokens, fitted by CapyCTL; SGLang accepts at most 42,735 input tokens (see below) |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0`, thinking on (the model's default; the
first token counted is the first `reasoning_content` token). The three prompts
are the first three of capyctl-bench's prompt set (`explain-tcp`, `python-lru`,
`history-printing`); every request generated all 512 tokens. The 0.080 s is the
first request after the start; a second run gave 0.045 s median (0.045 to
0.052) and the same 62.5 tokens/s.

On a discrete GPU, SGLang accepts fewer input tokens than the context
`capyctl status` reports: a 60,010-token prompt came back with `Input length
(60010 tokens) exceeds the maximum allowed length (42735 tokens)`, and a
40,010-token prompt was answered. This is a known gap in CapyCTL 0.1.2.

## Benchmark

[`bench/`](bench/) has the capyctl-bench results and report
([`report.html`](bench/report.html), [`summary.md`](bench/summary.md),
[`data.csv`](bench/data.csv)). Context sweep 0.5k to 32k tokens (the largest
point under the 42,735 input tokens SGLang accepts), three runs per point,
`max_tokens: 128`; concurrency 1 to 8 streams, five rounds each,
`max_tokens: 512`; `temperature: 0`, thinking on; GPU memory sampled with
`nvidia-smi` every 0.5 s.

| Context | Time to first token | Decode, one stream | GPU memory, peak |
|---|---|---|---|
| 0.5k | 0.09 s | 62.3 tokens/s | 13.4 GiB |
| 1k | 0.13 s | 61.9 tokens/s | 13.4 GiB |
| 2k | 0.24 s | 61.4 tokens/s | 13.4 GiB |
| 4k | 0.48 s | 60.7 tokens/s | 13.4 GiB |
| 8k | 0.99 s | 60.2 tokens/s | 13.4 GiB |
| 16k | 2.08 s | 58.7 tokens/s | 13.4 GiB |
| 32k | 4.69 s | 55.6 tokens/s | 13.4 GiB |

| Streams | Together | Each | Time to first token, median | Time to first token, p95 |
|---|---|---|---|---|
| 1 | 62.2 tokens/s | 62.5 tokens/s | 0.05 s | 0.05 s |
| 2 | 117.2 tokens/s | 58.8 tokens/s | 0.05 s | 0.07 s |
| 3 | 172.3 tokens/s | 57.8 tokens/s | 0.07 s | 0.08 s |
| 4 | 227.1 tokens/s | 57.2 tokens/s | 0.08 s | 0.08 s |
| 5 | 278.6 tokens/s | 56.1 tokens/s | 0.08 s | 0.08 s |
| 6 | 330.1 tokens/s | 55.5 tokens/s | 0.09 s | 0.10 s |
| 7 | 380.9 tokens/s | 54.9 tokens/s | 0.09 s | 0.10 s |
| 8 | 232.2 tokens/s | 54.8 tokens/s | 0.10 s | 9.46 s |

Every concurrency request generated all 512 tokens. Up to 7 streams, every
stream decodes at once; with 8, the eighth request waits for one of the first
seven to finish (the running limit of 7), so a round takes about twice as long and
its time to first token is about 9.5 s.

![Summary](bench/summary/summary-wide.png)
