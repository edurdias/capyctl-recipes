# Nemotron 3.5 Super VL: why I stopped the in_proj NVFP4 experiment

**2026-11-25 — stopped after activation checks; no checkpoint converted or published.** I captured inputs from the unchanged NVIDIA model and simulated selective Mamba input-projection requantization on CPU. The added activation zeros were enough to stop this route, not evidence of a measured task-quality regression.

## What I tried and why

I considered converting only the forty always-active Mamba `in_proj` matrices from NVIDIA's scalar-scaled FP8 to block-16 NVFP4. Experts, out_proj, head, MTP and vision would stay unchanged. The source is [NVIDIA's NVFP4 release at `9838c915`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tree/9838c91537e6be21dec2f5276b784211a06fbeea), including its processor correction. I used the released FP8-reconstructed values to retain its quantization-aware-distilled parameters, rather than splice in different BF16 weights.

The [physical tensor inventory](source-inventory.json) gives 3,040,870,720 bytes before and 1,710,489,920 bytes after the proposed layout: **1,330,380,800 bytes saved**. **[INFERENCE] About 1.33 GB fewer weight bytes per token** if one sequence reads each selected weight and scale once per ordinary target decoding step. This is not measured memory traffic or a throughput prediction; batching, caching and speculative acceptance change the relationship.

## Why not weights-only W4A16?

These are source-inspection findings for SGLang 0.5.22, commit `baa9befa381511cd4d56f83980951e5a45fffc6d`, not a candidate load test:

- [`mamba.py:292–305`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/attention/mamba/mamba.py#L292-L305) constructs a five-part `MergedColumnParallelLinear`.
- [`linear.py:140–162`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/linear.py#L140-L162) allows loader v2 for `ModelOptFp4LinearMethod`, but not the dedicated W4A16 method. [`modelopt_quant.py:2216–2243`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/quantization/modelopt_quant.py#L2216-L2243) creates five global-scale slots and marks `weight_scale_2` for scalar-to-array loading.
- Loader v1 selects only slot zero ([`linear.py:748–759`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/linear.py#L748-L759)); its helper accepts only a scalar or `[1]` ([`190–209`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/linear.py#L190-L209)). A scalar leaves four slots uninitialized; `[5]` fails. The model's loader maps Q/K/V shard IDs, not five separate in_proj parts ([`nemotron_h.py:1078–1122`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/models/nemotron_h.py#L1078-L1122)). Changing input_scale alone cannot fix this.

W4A16 therefore needs an engine patch, not a checkpoint-layout workaround. I did not patch the engine.

## The W4A4 route

The unpatched alternative is W4A4: [`modelopt_quant.py:1047–1061`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/quantization/modelopt_quant.py#L1047-L1061) dispatches `NVFP4` to that method, and loader v2 fills all logical scalar slots ([`linear.py:1000–1024`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/linear.py#L1000-L1024)). Actual candidate execution was not tested.

I reused NVIDIA's calibration bound: FP8 input_scale is `amax/448`, so NVFP4's global activation scale `amax/(6*448)` is **old input_scale / 6**, with dynamic block-16 activation scales. This reinterprets an existing bound; it is not fresh calibration or evidence that FP4 activations preserve quality. I checked this before converting anything.

## Capture and CPU results

The unchanged model ran without MTP or CUDA graphs: context 4096, one request, 8192 token slots, ten Mamba states, prefill chunk/cap 512/1024. An initial four-state attempt failed before readiness; ten states were explicitly authorized. The capture used a 95 GiB/no-swap scope and a continuous 20 GiB available-memory guard. These limits and the planning arithmetic are in [capture-safety.json](results/capture-safety.json), not a measured peak-memory claim.

All fifteen chat/code/tool captures across layers 0/18/38/60/86 contain 128 finite, correctly attributed tokens. The CPU check used a 6 GiB/no-swap scope and two CPU threads, comparing 32 evenly spaced projection tokens per capture. Percentages below use all 128 tokens for activation statistics. Projection relative L2 errors compare against simulated FP8 activations with the released FP8-reconstructed weights; A/W/A+W mean activation-only, weight-only and combined changes.

| Layer | Case | Below FP4 threshold % | FP8 zeros % | FP4 zeros % | Projection A % | Projection W % | Projection A+W % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | chat | 13.120 | 0.017 | 13.187 | 3.780 | 3.521 | 5.201 |
| 0 | code | 13.435 | 0.011 | 13.499 | 3.758 | 3.546 | 5.186 |
| 0 | tool | 10.463 | 0.007 | 10.524 | 3.927 | 3.668 | 5.406 |
| 18 | chat | 8.495 | 0.059 | 8.499 | 4.791 | 4.425 | 6.518 |
| 18 | code | 8.239 | 0.064 | 8.244 | 4.861 | 4.487 | 6.623 |
| 18 | tool | 7.946 | 0.063 | 7.951 | 5.042 | 4.602 | 6.818 |
| 38 | chat | 7.715 | 0.063 | 7.715 | 4.739 | 4.343 | 6.442 |
| 38 | code | 7.590 | 0.066 | 7.590 | 4.847 | 4.441 | 6.598 |
| 38 | tool | 7.593 | 0.067 | 7.593 | 5.114 | 4.611 | 6.884 |
| 60 | chat | 7.719 | 0.044 | 7.719 | 5.008 | 4.540 | 6.747 |
| 60 | code | 7.668 | 0.055 | 7.668 | 5.088 | 4.671 | 6.902 |
| 60 | tool | 7.674 | 0.067 | 7.674 | 5.785 | 5.172 | 7.807 |
| 86 | chat | 10.598 | 0.056 | 10.598 | 4.862 | 4.239 | 6.463 |
| 86 | code | 10.467 | 0.051 | 10.467 | 4.849 | 4.186 | 6.426 |
| 86 | tool | 10.024 | 0.064 | 10.024 | 4.432 | 3.876 | 5.887 |

The rounding threshold is `0.25 × rounded dynamic block scale × new input_scale`; midpoint ties round to even zero. FP8 activation relative L2 error is 2.595–2.662%, versus 9.017–9.440% for FP4. Clipping is zero except one value out of 524288 in layer 60/chat. The concern is consistent additional zeroing, roughly 7.5–13.5 percentage points, not anomalous clipping. Five layers and three short cases cannot qualify all forty layers, stopping behavior, or MTP acceptance.

## Decision and retained method

**I stopped after this activation check.** The qualitative gate rejected substantial extra zeros; I did not invent a numerical task-quality cutoff. No conversion, candidate load, task-quality qualification, performance measurement or checkpoint upload followed. There is no speedup claim.

[activation-check.json](results/activation-check.json) retains the full metrics and hashes; [captures-10/](results/captures-10/) retains tensors and request/response evidence. The CPU checker and its helper modules remain here to make the finding reproducible. With CPU-only Torch 2.9.1, NumPy 2.3.5, safetensors 0.7.0 and packaging 25.0, rerun against the pinned source using:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 -B activation_check.py --source /home/me/models/Nemotron-Super-NVFP4 \
  --captures results/captures-10 --report /home/me/activation-check.json
```

Use a 6 GiB/no-swap CPU scope as in the recorded check. This command does not launch an engine or convert a checkpoint. The unused converter, audit, tests and capture launcher were archived rather than advertised as a usable recipe. See [publication-audit.json](publication-audit.json). The [drafter comparison](../nemotron-3.5-super-vl-drafters-gb10/) remains a separate result.
