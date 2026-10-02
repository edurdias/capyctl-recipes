# Qwen3.8-27B NVFP4 on vLLM 0.30.0 with DFlash2 drafts, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | vLLM 0.30.0 in a venv (torch 2.13.0+cu130, FlashInfer 0.6.18.post1) |
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
[SGLang recipe](../../sglang/qwen3.8-27b-nvfp4-gb10/) for this model.

vLLM accepts the checkpoint's own quantization config, so the deployment sets
no `quantization`. What vLLM picked, from its log:

```text
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN'].
```

vLLM 0.30.0 also has a B12X attention backend for compute capability 12.x. It
is not among the backends vLLM picks from on its own, it needs the `b12x`
package (not part of this venv) and it takes 64- or 128-token pages, while
vLLM runs this hybrid model with 1,648-token attention blocks. It was not
tried.

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

Register the engine and allow `--speculative-config` for drafters inside that
directory:

```bash
capyctl engine add ~/vllm-0.30-venv \
  --approve-option=--speculative-config --approve-path /home/me/drafters
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
Request identity: 01M3YJE12WXYJK3A1EP31ADYBR (reuse --request-id 01M3YJE12WXYJK3A1EP31ADYBR to recover this command)
Deployment qwen38-27b-vllm created (revision 1)

  Deployment ID       01M3YJE13E314BWEDZ4DGTVHAW
  Operation           01M3YJE13E5CEWY2H0TDVT37S4
  Checkpoint digest   being measured
the checkpoint digest of qwen38-27b-vllm is being measured; `capyctl start deployment qwen38-27b-vllm --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment qwen38-27b-vllm --wait
```

```text
Request identity: 01M3YJE13SARQH1NWAEGRDBMC7 (reuse --request-id 01M3YJE13SARQH1NWAEGRDBMC7 to recover this command)
Waiting for the checkpoint digest of qwen38-27b-vllm to be measured (at most 900s)
Started qwen38-27b-vllm: ready

  Ready       1/1
  Hosts       spark
  Operation   initialize succeeded
```

Qwen3.8 thinks before it answers. The deployment sets no reasoning parser, so
the thinking arrives in `content`, ending at `</think>`:

```bash
curl -s http://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen38-27b-vllm", "messages": [{"role": "user", "content": "Name the largest planet in one sentence."}]}' \
  | jq -r '.choices[0].message.content'
```

```text
We need answer user's request: "Name the largest planet in one sentence." Need produce final one sentence. Could say "Jupiter is the largest planet in our solar system." Need ensure one sentence. No extra.
</think>

Jupiter is the largest planet in our solar system.
```

## Measured through CapyCTL

| | |
|---|---|
| Ready, cold | 117 s (weights already in the model store; includes measuring the checkpoint digest; vLLM's compile cache from earlier starts on this machine was reused, see below) |
| Ready, warm | 121 s (`start` after `stop` finished; vLLM repeats its full startup) |
| Time to first token | 0.230 s median (0.229 to 0.288) |
| Decode, one stream | 34.5 tokens/s median (23.0 to 35.4), DFlash2 drafts on |
| Peak memory | 53.0 GiB measured by CapyCTL, above the 49.2 GiB reservation (CapyCTL's default startup size for the 48 GiB request); `MemAvailable` fell by 53.0 GiB at most |
| Context | 32,768 tokens, declared |

Three streaming chat completions through the CapyCTL endpoint, one at a time,
`max_tokens: 512`, `temperature: 0.6`, thinking on (the model's default; the
first token counted is the first thinking token, which arrives in `content`).
One request generated all 512 tokens; the other two finished on their own at
432 and 474. The same three requests after the warm start gave the same
numbers (0.233 s, 34.5 tokens/s).

The first start of this model on the machine took longer. vLLM's initial
profiling and warmup run took 236 s, and the start did not finish within the
340 s initialize timeout CapyCTL derives from the weights; the next start
reported 335 s for vLLM's engine initialization alone. The
`timeouts: {initialize: 900s}` in `deployment.yaml` leaves room for that first
start.

CapyCTL sizes the KV cache from the 48 GiB request and the checkpoint's
weights (vLLM reserved 19.6 GiB for it); it does not count the drafter, which
is why the peak is about 4 GiB above the no-drafts run and above the
reservation.

### What the drafter does

The same deployment without `extra_args` (and so without the drafter), on the
same machine, the same three requests:

| | DFlash2 drafts | No drafts |
|---|---|---|
| Decode, one stream | 34.5 tokens/s (23.0 to 35.4) | 11.8 tokens/s (11.8 to 11.8) |
| Time to first token | 0.230 s | 0.118 s |
| Ready, first start of the deployment | 117 s | 143 s |
| Peak memory (CapyCTL) | 53.0 GiB | 49.0 GiB |

Drafts make decode about 2.9 times faster. All three no-drafts requests
generated 512 tokens.

The no-drafts run keeps the 48 GiB request. With a 40 GiB request vLLM
reserves 11.6 GiB of KV cache and refuses to start:

```text
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (242). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. Please lower max_num_seqs to at most 242 or increase gpu_memory_utilization.
```

Setting `engine_config.max_concurrent_requests` (vLLM's `--max-num-seqs`)
below 242 is the other way out.

The model runs with deep parking on (CapyCTL's default for vLLM), so
`capyctl status deployment` warns that vLLM development mode is on.

A downloaded copy needs about 21.9 GB in the model store and 3.8 GB for the
drafter; with the download, the first start takes as long as the download plus
the time above.
