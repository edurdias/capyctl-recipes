# Kev-4B (Q8_0 GGUF) on llama.cpp v0.7.0, one GB10

| | |
|---|---|
| Hardware | 1x NVIDIA GB10 (DGX Spark class), compute capability 12.1, 128 GB unified memory, aarch64 |
| System | Ubuntu 24.04, NVIDIA driver 580.173.02, CUDA 13.0 toolkit in `/usr/local/cuda` |
| Engine | llama.cpp `llama-server` built from tag `v0.7.0` (commit `620d222`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real`) |
| Model | [`ggml-org/Kev-4B-GGUF`](https://huggingface.co/ggml-org/Kev-4B-GGUF) at `d924f2e2c3872da8b8aaf3eb4453b4126deceb79`: `Kev-4B-Q8_0.gguf` |
| CapyCTL | `main` at `8824b622` for the commands below; `main` at `075023d7` for the capyctl-bench run; `capyctl start standalone` |
| Measured | 2026-10-18 |

Kev-4B (`jaredpalmer/kev-4b`, built on Qwen3.5-4B-Base) is a 4B decision model, 4.5 GB as a Q8_0 GGUF. It is a decision model: it does not generate text, it answers typed
questions about a text (`POST /v1/systemone`) with a probability for each answer.

It answers one question about a 200-token text in **88 ms**, starts in 16 s, and gets **90.0% on AG News and 91.2% on BoolQ**, the best BoolQ in the field after Clef. On Typed Decisions it gets 66.7%, API-Bank 65.9%, and Home appliances 14.8%. AG News and BoolQ are among the datasets Kev's model card lists, so those two show the task works, not how it generalises.

## Run it

Decision models need CapyCTL 0.1.5 or later (`api: decisions`).
Download the model into the models directory (the pinned command is in
`deployment.yaml`) and register the engine:

```bash
capyctl engine add ~/llamacpp-v0.7.0/build/bin/llama-server --name llamacpp
```

```text
Registered llamacpp (llamacpp 0.7.0)

  Executable     /home/me/llamacpp-v0.7.0/build/bin/llama-server
  Deep park      disabled
  Engines file   /home/me/.config/capyctl/engines.yaml (revision 1)
  Published      yes
```

```bash
capyctl deploy model --file deployment.yaml
```

```text
Request identity: 01M57B4TQ7HBX2NNMD29XT7C85 (reuse --request-id 01M57B4TQ7HBX2NNMD29XT7C85 to recover this command)
Deployment kev-4b created (revision 1)

  Deployment ID       01M57B4TQSYY4VFM40VER8HVGM
  Operation           01M57B4TQSREVAMXAFZENFT7JJ
  Checkpoint digest   being measured
the checkpoint digest of kev-4b is being measured; `capyctl start deployment kev-4b --wait` waits for it and starts the deployment
```

```bash
capyctl start deployment kev-4b --wait
```

```text
Request identity: 01M57B4TR2SWGMSV3YTEH1CFMN (reuse --request-id 01M57B4TR2SWGMSV3YTEH1CFMN to recover this command)
Waiting for the checkpoint digest of kev-4b to be measured (at most 900s)
Started kev-4b: ready

  Ready       1/1
  Hosts       gpu-box
  Operation   initialize succeeded
```

Ask it something:

```bash
KEY=$(sed -n 's/^api_key: //p' ~/.local/state/capyctl/identity/credentials)
curl -s http://127.0.0.1:8443/v1/systemone \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "kev-4b", "state": "The ticket says: my card was charged twice for one order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "charges, refunds, invoices", "shipping": "deliveries and tracking", "technical": "bugs and errors"}}, "urgent": {"type": "noul", "instructions": "Is the customer losing money right now?"}}}' \
  | python3 -m json.tool
```

```text
{
    "answers": {
        "team": {
            "choice": "billing",
            "confidence": 0.8210366618417622,
            "probabilities": {
                "billing": 0.8806911078945082,
                "shipping": 0.012449899582278006,
                "technical": 0.10685899252321364
            },
            "type": "choice"
        },
        "urgent": {
            "noul": 0.2861099001229467,
            "type": "noul"
        }
    },
    "model": "kev-4b",
    "usage": {
        "input_tokens": 76,
        "output_tokens": 0
    }
}
```

`choice` questions come back with a probability for each option, `noul` (yes or
no) questions with the probability of yes. Chat completions are refused:

```text
{"code":"api_mismatch","message":"model kev-4b serves the decisions API, which /v1/chat/completions does not answer; send its requests to POST /v1/systemone"}
HTTP 400
```

## Numbers

| Number | Value |
|---|---|
| Ready time, cold | 15.7 s (first deploy, including the checkpoint digest) |
| Ready time, warm | 3.6 s after `capyctl stop deployment kev-4b` |
| First decision | 150 ms after a cold start, 138 ms after a warm one (the request above) |
| Peak memory | 25.9 GiB measured by CapyCTL (`startup.measured`) against the 28 GiB it reserves |
| Decisions/s, 1 / 2 / 4 / 8 users | 1.51 / 1.60 / 1.61 / 1.60 (3 questions, a 2k-token text each) |
| Questions/s, 1 / 2 / 4 / 8 users | 4.54 / 4.81 / 4.82 / 4.79 |
| Repeated text | 649 ms p50 against 658 ms for new texts (3 questions, 2k tokens) |
| AG News (500 test rows) | 90.0% accuracy, Brier 0.167 |
| BoolQ (500 validation rows) | 91.2% accuracy, Brier 0.074 |
| Typed Decisions (400 cases, 2,000 decisions) | 66.7% accuracy, KL from gold 0.223, ECE 0.092 |
| Decision Index components | API-Bank 65.9%, GPQA Diamond 39.3%, Home appliances 14.8% |

Latency, one request at a time, request sent to the whole answer read, p50 / p95 in ms:

| Questions | 200-token text | 2k-token text | 8k-token text |
|---|---|---|---|
| 1 | 88.1 / 90.7 | 551 / 560 | 2,604 / 2,679 |
| 3 | 205 / 210 | 661 / 681 | 2,739 / 2,787 |
| 5 | 316 / 322 | 1,223 / 1,248 | 5,322 / 5,432 |

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

- **Memory.** The 4.5 GB checkpoint peaks at 25.9 GiB with a 16,384-token context and batch. Shorter contexts were not measured; 16,384 tokens covers API-Bank's longest requests.
- **No parking.** Decision models start and stop (`residency: restart_only`);
  when CapyCTL needs the memory it stops the model, and the next request starts
  it again in about 4 seconds.
- **Unauthenticated loopback surfaces.** As for every llama.cpp deployment,
  `capyctl status deployment` notes that llama-server's own routes answer
  without a key on its loopback listener; CapyCTL's endpoint still requires one.
