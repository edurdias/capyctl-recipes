# Qwen3.8-27B NVFP4 on SGLang 0.5.20 with DFlash2 drafts, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | SGLang 0.5.20 in a venv (torch 2.13.0+cu130, FlashInfer 0.6.18) |
| Model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at `482ca0f3832238542f8f5295dde86b5f22711d80`, NVFP4 with FP8 layers, 21.9 GB |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, 3.8 GB |
| CapyCTL | `main` at `12ad397` (prints `capyctl 0.1.1`); `capyctl start standalone` |
| Measured | 2026-10-02 |

This recipe needs CapyCTL newer than the 0.1.1 release (`main` at `12ad397`
or later, until the next release) for `capyctl engine add
--approve-option/--approve-path`. The deployment file itself validates with
0.1.1, which is what this repository's check runs.

The checkpoint, the drafter and the measuring method are the same as in the
[TensorFold recipe](../../tensorfold/qwen3.8-27b-nvfp4-gb10/) and the
[vLLM recipe](../../vllm/qwen3.8-27b-nvfp4-gb10/) for this model. The
deployment names SGLang's `modelopt` quantization and an FP8 KV cache; the
drafter runs unquantized (`--speculative-draft-model-quantization unquant`).

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

Register the engine and allow `--speculative-draft-model-path` for drafters
inside that directory:

```bash
capyctl engine add ~/sglang-0.5.20-venv \
  --approve-option=--speculative-draft-model-path --approve-path /home/me/drafters
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
Request identity: 01M3YKRM6TFZPDM6BF2CRTB3X5 (reuse --request-id 01M3YKRM6TFZPDM6BF2CRTB3X5 to recover this command)
Deployment qwen38-27b-sglang created (revision 1)

  Deployment ID       01M3YKRM7T52D2R5YJXHPKAM5Q
  Operation           01M3YKRM7TZK4E0V0E2BNC4DEF
  Checkpoint digest   being measured
the checkpoint digest of qwen38-27b-sglang is being measured; `capyctl start deployment qwen38-27b-sglang --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen38-27b-sglang --wait
```

```text
Request identity: 01M3YKRM8DGV3VXBMWV7JVCDZW (reuse --request-id 01M3YKRM8DGV3VXBMWV7JVCDZW to recover this command)
Waiting for the checkpoint digest of qwen38-27b-sglang to be measured (at most 900s)
Started qwen38-27b-sglang: ready

  Ready       1/1
  Hosts       host-a
  Operation   initialize succeeded
```

Qwen3.8 thinks before it answers. The deployment sets no reasoning parser, so
the thinking arrives in `content`, ending at `</think>`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen38-27b-sglang", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text
We need answer user's simple request: "Name the largest planet in one sentence." Need final one sentence.
</think>

The largest planet is Jupiter.
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 155 s (weights already in the model store; includes measuring the checkpoint digest) |
| Ready, warm | 147 s (`start` after `stop` finished; SGLang repeats its full startup) |
| Time to first token | 0.214 s median (0.209 to 0.536) |
| Decode, one stream | 33.1 tokens/s median (24.5 to 42.0), DFlash2 drafts on |
| Peak memory | 44.9 GiB measured by CapyCTL, against a 49.2 GiB reservation (CapyCTL's default startup size for the 48 GiB request); `MemAvailable` fell by 45.3 GiB at most |
| Context | 32,768 tokens, declared |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the model's default; the
first token counted is the first thinking token, which arrives in `content`).
One request generated all 512 tokens; the other two finished on their own at
458 and 431. The same three requests after the warm start gave 0.213 s and
31.0 tokens/s (23.6 to 45.3): with sampling on, the outputs differ from run to
run, and so does how many drafted tokens are accepted.

The `timeouts: {initialize: 900s}` in `deployment.yaml` leaves room for a
first start that builds kernels; CapyCTL derives 340 s from the weights. The
starts measured here stayed under it.

### What the drafter does

The same deployment without `extra_args` (and so without the drafter), on the
same machine, the same three requests:

| | DFlash2 drafts | No drafts |
|---|---|---|
| Decode, one stream | 33.1 tokens/s (24.5 to 42.0) | 12.1 tokens/s (12.1 to 12.2) |
| Time to first token | 0.214 s | 0.182 s |
| Ready, first start of the deployment | 155 s | 134 s |
| Peak memory (CapyCTL) | 44.9 GiB | 44.0 GiB |

Drafts make decode about 2.7 times faster and cost about 1 GiB more at the
peak.

A downloaded copy needs about 21.9 GB in the model store and 3.8 GB for the
drafter; with the download, the first start takes as long as the download plus
the time above.
