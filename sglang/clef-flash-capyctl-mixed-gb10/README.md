# Clef-Flash mixed NVFP4/FP8 on SGLang 0.5.22

I publish my mixed NVFP4/FP8 r1 conversion of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), a Qwen3.5-9B decision model. It returns typed answer probabilities, not generated text. The source is pinned at `17f0b0ad64efb65d273590632833508766b2aae6`; my checkpoint is [capyctl/clef-flash-NVFP4-FP8](https://huggingface.co/capyctl/clef-flash-NVFP4-FP8), with `1e72872e4c70b563038336637391e05704f2ccb4` deliberately awaiting the upload commit.

**The 99% held-out BF16-agreement target was missed.** I publish r1 because its measured top-answer agreement exceeds the published quantized checkpoint it replaces, not because it passed the proposed gate. Agreement is not gold-label task accuracy or statistical equivalence.

The measured target is one NVIDIA GB10 with 128 GB unified memory. This does not qualify a 16 GB GPU.

## Run

Use CapyCTL 0.1.5 or later and an existing SGLang 0.5.22 virtual environment. Replace `1e72872e4c70b563038336637391e05704f2ccb4` in [deployment.yaml](deployment.yaml) with the published checkpoint commit before provisioning. Hub provisioning and this deployment manifest have not been exercised; retained measurements used local checkpoint paths.

```bash
capyctl engine add /home/me/sglang-0.5.22-venv --name sglang
capyctl deploy model --file deployment.yaml
capyctl start deployment clef-flash-capyctl-mixed --wait
```

This recipe uses `api: decisions` and `residency: restart_only`, matching the [BF16 decision recipe](../clef-flash-gb10/). Send authenticated requests to `POST /v1/systemone` with `model: clef-flash-capyctl-mixed`, `state`, and a `questions` mapping. `choice` returns option probabilities; `noul` returns the probability of yes. `/v1/chat/completions` is refused with `api_mismatch`. Decision models start and stop; they do not park. The API examples in the BF16 recipe describe the request schema; benchmark request text is not included here.

## Measured GB10 results

| Build (GB10, SGLang 0.5.22) | Same top / 5,294 | Agreement (95% Wilson) | Mean KL | Mean TV |
|---|---:|---|---:|---:|
| Published FP8 | 5,208 | 98.376% (97.998–98.683%) | 0.00164187 | 0.0116312 |
| CapyCTL FP8 r1 — not published | 5,218 | 98.564% (98.207–98.851%) | 0.00172332 | 0.0119396 |
| Published NVFP4 | 5,172 | 97.696% (97.255–98.066%) | 0.00344258 | 0.0179021 |
| CapyCTL mixed r1 | 5,200 | 98.224% (97.832–98.547%) | 0.00242694 | 0.0150904 |

The BF16 checkpoint is `Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6`. All rows use the same 2,192 held-out requests; Wilson intervals are question-level descriptive intervals, not independent-request confidence guarantees. KL uses natural logarithms and floor 1e-9. The [model card](convert/MODEL_CARD.md) pins the published alternatives, explains quantization and records the trade-offs.

Only the mixed checkpoint is offered. FP8 r1 remains experimental: its same-laptop agreement falls to 98.357% versus published FP8's 98.451%, and sampled device-memory headroom is only 23.1875 MiB. Its recipe, card and upload inventory are not published.

One-question whole-response latency, p50 / p95 ms:

| Nominal 200 | Nominal 2k | Nominal 8k | Near 16k |
|---|---|---|---|
| 49.35 / 50.75 | 277.21 / 286.74 | 1,794.37 / 1,827.83 | 3,934.74 / 3,985.80 |

Three rounds × 100 timed requests, five warmups per point; actual median input tokens are 204 / 2,043 / 8,186 / 16,188. Inputs are approximate, not token-identical. Direct SGLang 0.5.22 serving uses 16,384 context/token pool, one running request, FP8 E4M3 KV, BF16 Mamba state and 0.50 static memory fraction. No generated-token throughput applies. FP8 r1 long-input latency regresses versus BF16. GPU peak is unmeasured; package bytes (10,649,663,746) do not establish residency.

[Comparison](../../comparisons/clef-flash-builds-gb10/) · [Build report](../../comparisons/clef-flash-builds-gb10/results/capyctl-build-report.txt) · [Export audit](../../comparisons/clef-flash-builds-gb10/results/clef-flash-capyctl-mixed-r1-audit.json)

## Conversion and license

31 transformer layer families use NVFP4 and 33 use FP8. Every family observed to be NVFP4-sensitive in the diagnostic simulation is restored to FP8. NVFP4 fused projection pairs share their activation and global weight scales. I use ModelOpt 0.47.0 calibration/rounding and a compressed-tensors exporter. Calibration and diagnostic selection exclude the held-out set. Both the separate joint decision head and lexical head are byte-identical BF16; see the [model card](convert/MODEL_CARD.md) for hashes and evidence. The source checkpoint is Apache-2.0; its `LICENSE` is retained.
