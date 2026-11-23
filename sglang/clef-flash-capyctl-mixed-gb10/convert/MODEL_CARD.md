---
license: apache-2.0
base_model: Cloudflare/clef-flash
base_model_relation: quantized
library_name: sglang
tags:
- clef
- quantized
- compressed-tensors
---

# Clef-Flash mixed NVFP4/FP8 — CapyCTL r1

I publish my mixed NVFP4/FP8 r1 conversion of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), a Qwen3.5-9B decision model. It returns typed answer probabilities, not generated text. The source is pinned at `17f0b0ad64efb65d273590632833508766b2aae6`; my checkpoint is [capyctl/clef-flash-NVFP4-FP8](https://huggingface.co/capyctl/clef-flash-NVFP4-FP8), with `1e72872e4c70b563038336637391e05704f2ccb4` deliberately awaiting the upload commit.

**The 99% held-out BF16-agreement target was missed.** I publish r1 because its measured top-answer agreement exceeds the published quantized checkpoint it replaces, not because it passed the proposed gate. Agreement is not gold-label task accuracy or statistical equivalence.

## Quantization and export

31 transformer layer families use NVFP4 and 33 use FP8. Every family observed to be NVFP4-sensitive in the diagnostic simulation is restored to FP8. NVFP4 fused projection pairs share their activation and global weight scales. ModelOpt 0.47.0 supplies calibration and rounding; I export compressed-tensors safetensors with my own exporter because the approved ModelOpt FP8 loader accepts per-tensor, not per-channel, weight scales. The earlier published FP8 layout round-tripped all 200 quantized linears with identical FP8 values and zero scale difference before this export.

Calibration uses 18 independently generated synthetic decision requests. Scale tuning and layout selection use the separate 41-request / 104-question diagnostic set. The 2,192-request / 5,294-question held-out set is excluded from calibration and selection. Sensitivity and layout-selection experiments use BF16-arithmetic QDQ simulation; final agreement and latency use SGLang's unmodified stock loader, not those hooks.

The separate `joint_head.safetensors` and lexical `lm_head.weight` remain **byte-identical BF16**. Joint-head SHA-256 is `19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0`. Non-targeted tensors retain source precision. Exact measured package size is 10,649,663,746 bytes; this is not runtime memory.

## BF16 agreement and published alternatives

| Build (GB10, SGLang 0.5.22) | Same top / 5,294 | Agreement (95% Wilson) | Mean KL | Mean TV |
|---|---:|---|---:|---:|
| Published FP8 | 5,208 | 98.376% (97.998–98.683%) | 0.00164187 | 0.0116312 |
| CapyCTL FP8 r1 — not published | 5,218 | 98.564% (98.207–98.851%) | 0.00172332 | 0.0119396 |
| Published NVFP4 | 5,172 | 97.696% (97.255–98.066%) | 0.00344258 | 0.0179021 |
| CapyCTL mixed r1 | 5,200 | 98.224% (97.832–98.547%) | 0.00242694 | 0.0150904 |

Published FP8 is `kurcontko/clef-flash-FP8-Dynamic@7b43c2d799e7b5e9fef96b6137840088d75af0d6`; published NVFP4 is `alpha-x-ai/clef-flash-NVFP4@e676d8e6e14f5ee74830d33fa1fad643d8f53acf`. The BF16 checkpoint uses the source revision above, identical ordered requests and question keys. All rows here are GB10 captures. Question-level Wilson intervals are descriptive: questions in a request can be correlated. KL is BF16-to-candidate with natural logarithms and candidate probability floor 1e-9; TV is total variation. FP8 r1 improves top agreement but has worse KL/TV than published FP8. Mixed improves published NVFP4 agreement and drift at 3.46% more package bytes, but does not reach published-FP8 drift or the 99% target.

I publish only this GB10 mixed checkpoint. FP8 r1 is measured but not published: despite 98.564% GB10 agreement, its same-laptop agreement is 98.357% versus published FP8's 98.451%, and its sampled device-memory headroom is only 23.1875 MiB. Its recipe, model card and upload inventory are removed; the comparison retains its measurements.

## Latency and limits

Whole-request p50 / p95 milliseconds, one question:

| Nominal 200 | Nominal 2k | Nominal 8k | Near 16k |
|---|---|---|---|
| 49.35 / 50.75 | 277.21 / 286.74 | 1,794.37 / 1,827.83 | 3,934.74 / 3,985.80 |

Three rounds, five warmups and 100 timed requests per point per round; actual token medians (ranges) are 204 (200–210), 2,043 (1,938–2,142), 8,186 (7,841–8,496), and 16,188 (15,569–16,377). Inputs are approximate, not token-identical. SGLang 0.5.22 uses 16,384 context/token pool, one running request, FP8 E4M3 KV, BF16 Mamba state and static memory fraction 0.50. These are direct-engine GB10 results, not a CapyCTL deployment benchmark. FP8 r1 long-input latency regresses versus BF16. GPU-memory peak is unmeasured because NVML does not support the query on this machine.

The measured target is one NVIDIA GB10 with 128 GB unified memory. This does not qualify a 16 GB GPU. Text-only finite cohorts do not qualify images, video, other hardware, concurrency or broader capabilities.

## Evidence and use

The repository [recipe](../README.md) describes the decisions API. Retained [build report](../../../comparisons/clef-flash-builds-gb10/results/capyctl-build-report.txt), [export audit](../../../comparisons/clef-flash-builds-gb10/results/clef-flash-capyctl-mixed-r1-audit.json), [held-out comparison](../../../comparisons/clef-flash-builds-gb10/results/clef-flash-capyctl-mixed-r1-vs-bf16-heldout.json), calibration manifest, round-trip and sensitivity records live in the [comparison](../../../comparisons/clef-flash-builds-gb10/). Request texts are not redistributed.

## License

The source [model card](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/README.md) declares Apache-2.0, following Qwen/Qwen3.5-9B. The checkpoint retains the source `LICENSE`; quantization does not replace its terms.
