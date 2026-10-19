# Clef-Flash (BF16) on SGLang 0.5.22, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | SGLang 0.5.22 (`pip install sglang==0.5.22` in a virtual environment; it pulls `smg-grpc-servicer` 0.14.0) |
| Model | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) at `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` (safetensors, BF16) |
| CapyCTL | `main` at `8824b622` for the commands below; `main` at `075023d7` for the capyctl-bench run; `capyctl start standalone` |
| Measured | 2026-10-18 |

Clef-Flash is Cloudflare's 9B decision model (Qwen3.5-9B). This recipe serves its BF16 checkpoint (19.1 GB) on SGLang 0.5.22, the first SGLang release that serves decision models. It is a decision model: it does not generate text, it answers typed
questions about a text (`POST /v1/systemone`) with a probability for each answer.

It answers one question about a 200-token text in **155 ms**, the fastest Clef in the field, and an 8k-token text in 1.8 s (llama.cpp takes 3.8 s). It scores **93.1% on API-Bank**, the number Cloudflare reports for Clef-Flash on Decision Index 0.2.1, and 97.7% on Home appliances. The cost is memory and start time: it holds **51.9 GiB** and takes 3 minutes to become ready.

## Run it

Decision models need CapyCTL 0.1.5 or later (`api: decisions`).
Download the model into the models directory (the pinned command is in
`deployment.yaml`) and register the engine:

```bash
capyctl engine add ~/sglang-0.5.22-venv --name sglang
```

```text
Registered sglang (sglang 0.5.22)

  Executable     /home/me/sglang-0.5.22-venv/bin/python3
  Deep park      enabled
  CUDA           /usr/local/cuda
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 2)
  Published      yes
```

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M57B94QDKG8XHM86KKZ705ZZ (reuse --request-id 01M57B94QDKG8XHM86KKZ705ZZ to recover this command)
Deployment clef-flash created (revision 1)

  Deployment ID       01M57B94R0NAW60YZV9K3NHYJN
  Operation           01M57B94R0A0S1BBQKEMHX5GBG
  Checkpoint digest   being measured
the checkpoint digest of clef-flash is being measured; `capyctl start deployment clef-flash --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment clef-flash --wait
```

```text
Request identity: 01M57B94R9XEB7WA2M6W3YAS6D (reuse --request-id 01M57B94R9XEB7WA2M6W3YAS6D to recover this command)
Waiting for the checkpoint digest of clef-flash to be measured (at most 900s)
Started clef-flash: ready

  Ready       1/1
  Hosts       gpu-box
  Operation   initialize succeeded
```

Ask it something:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
curl -s http://127.0.0.1:8443/v1/systemone \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "clef-flash", "state": "The ticket says: my card was charged twice for one order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "charges, refunds, invoices", "shipping": "deliveries and tracking", "technical": "bugs and errors"}}, "urgent": {"type": "noul", "instructions": "Is the customer losing money right now?"}}}' \
  | python3 -m json.tool
```

```text
{
    "answers": {
        "team": {
            "choice": "billing",
            "confidence": 0.7995963971405613,
            "probabilities": {
                "billing": 0.8663975980937075,
                "shipping": 0.02345230826209376,
                "technical": 0.1101500936441987
            },
            "type": "choice"
        },
        "urgent": {
            "noul": 0.8955226768697631,
            "type": "noul"
        }
    },
    "model": "clef-flash",
    "usage": {
        "input_tokens": 247,
        "output_tokens": 0
    }
}
```

`choice` questions come back with a probability for each option, `noul` (yes or
no) questions with the probability of yes. Chat completions are refused:

```text
{"code":"api_mismatch","message":"model clef-flash serves the decisions API, which /v1/chat/completions does not answer; send its requests to POST /v1/systemone"}
HTTP 400
```

## Numbers

| Number | Value |
|---|---|
| Ready time, cold | 179.8 s (first deploy, including the checkpoint digest) |
| Ready time, warm | 127.4 s after `capyctl stop deployment clef-flash` |
| First decision | 140 ms after a cold start, 125 ms after a warm one (the request above) |
| Peak memory | 51.8 GiB measured by CapyCTL (`startup.measured`) against the 52 GiB it reserves |
| Decisions/s, 1 / 2 / 4 / 8 users | 1.62 / 1.70 / 1.88 / 1.91 (3 questions, a 2k-token text each) |
| Questions/s, 1 / 2 / 4 / 8 users | 4.86 / 5.11 / 5.63 / 5.74 |
| Repeated text | 626 ms p50 against 626 ms for new texts (3 questions, 2k tokens) |
| AG News (500 test rows) | 89.4% accuracy, Brier 0.165 |
| BoolQ (500 validation rows) | 89.6% accuracy, Brier 0.075 |
| Typed Decisions (400 cases, 2,000 decisions) | 70.2% accuracy, KL from gold 0.210, ECE 0.012 |
| Decision Index components | API-Bank 93.1%, GPQA Diamond 51.0%, Home appliances 97.7% |

Latency, one request at a time, request sent to the whole answer read, p50 / p95 in ms:

| Questions | 200-token text | 2k-token text | 8k-token text |
|---|---|---|---|
| 1 | 155 / 158 | 545 / 555 | 1,837 / 1,881 |
| 3 | 167 / 170 | 607 / 634 | 1,868 / 1,882 |
| 5 | 201 / 206 | 678 / 685 | 1,893 / 1,917 |

100 timed requests per point after 5 warm-up requests. Each request's text is new,
so no cache helps.

[Full report](bench/report.html) (download and open it locally),
[tables](bench/summary.md), [every point](bench/data.csv).

![Summary](bench/summary/summary-wide.png)

The capyctl-bench run was `capyctl_bench.py run --decisions --accuracy
--accuracy-sets all --concurrency 1,2,4,8`. Its README defines every metric,
and the report lists the pinned dataset revisions and row hashes. The same run
for seven more deployments is in the
[decision-model comparison](../../comparisons/decision-models-gb10/).

## Notes

- **Memory.** SGLang sizes its cache from the memory it is given, so the peak follows the reservation: the 48 GiB here is what was measured, not a minimum. The weights are 19 GB.
- **Start time.** SGLang compiles and captures CUDA graphs at every start: 180 s cold, 127 s warm. `capyctl start deployment --wait` waits for the first decision to answer (SGLang's `/health` answers 503 while decisions already work).
- **SGLang 0.5.22 needs CapyCTL 0.1.5 or later.** Earlier releases refused its launch (`plugin_closure_failed`): SGLang 0.5.22 installs plugins of its own.
- **No parking.** Decision models start and stop (`residency: restart_only`);
  when CapyCTL needs the memory it stops the model, and the next request starts
  it again in about 127 seconds.
