# GPT-OSS-120B: MXFP4 experts and FP8 attention

I convert OpenAI's pinned GPT-OSS-120B checkpoint for vLLM 0.32.0 on one GB10. [Weights](https://huggingface.co/capyctl/gpt-oss-120b-MXFP4-FP8) · [comparison and raw evidence](../../comparisons/gpt-oss-120b-builds-gb10/) · [model card](convert/MODEL_CARD.md).

## What changed

I preserve OpenAI's MXFP4 experts, E8M0 scales and expert biases in a reversible per-expert Quark layout. Only the 144 attention Q/K/V/O matrices change to E4M3 FP8 with one FP32 multiplicative scale per output channel. Scale is `max(abs(weight))/448`, clamped to positive FP32 tiny; weights are divided by scale, clamped to [-448,448], then cast to E4M3. The BF16 output head, embeddings, routers, norms, sinks and other biases remain unchanged. There is no calibration or fine-tuning.

The exhaustive CPU audit covers 687 source tensors and 28,263 output tensors. All 216 expert packed-weight/scale/bias tensors reconstruct byte-identically, all retained tensors are unchanged, and attention bytes/scales match the conversion rule. Tensor payload is 64,294,452,864 bytes.

[INFERENCE] Shape-derived serialized selected-weight reads fall from **5.003 to 4.049 GB/token** (5,002,907,904 to 4,048,545,024 bytes/token), a 19.076% reduction. This counts one read of selected experts, dense weights and head plus one embedding row; it excludes KV/activation traffic, padding, reuse and caching. It is not measured bandwidth or a speed prediction.
## Measured results

All speeds are tokens/s. Short c1 is post-first-token decode; c2/c4/c8 are total completion tokens divided by round wall time. Long columns are one-stream post-first-token decode, not total request throughput. Each timing is the median of three measured rounds after warm-up, fixed chat/code prompts, 512 completion-token cap and normal EOS.

| Setup | Short c1 decode | c2 aggregate | c4 aggregate | c8 aggregate | Decode 8k / 32k / 64k / 110k |
|---|---:|---:|---:|---:|---|
| CapyCTL · vLLM 0.32 | 45.345 | 66.911 | 88.191 | 112.054 | 43.848 / 40.259 / 35.882 / 31.735 |
| OpenAI · vLLM 0.32 | 34.796 | 57.685 | 79.736 | 104.849 | 33.874 / 31.721 / 29.242 / 26.123 |
| Published OpenAI · vLLM 0.30 | 34.591 | 58.534 | 80.504 | 105.733 | 33.153 / 29.393 / 25.493 / 21.472 |

The matched vLLM 0.32 rows use expandable segments, lazy loading, 131072 context, eight sequences, an 8 GiB FP8 E4M3 KV pool, Marlin experts, no drafter, no prefix caching and LMCache off. Final timings come from the shipped-file runs; full quality comes from their completed full rows. Candidate gains are 30.3% short c1 decode, 6.9% c8 aggregate and 21.5% 110k decode. The 28-cell length/concurrency sweep is retained in [final matrix](../../comparisons/gpt-oss-120b-builds-gb10/bench/final-matrix.json); every paired candidate aggregate cell is higher, by 5.3–29.6%. These are finite observations, not uncertainty bounds or a universal speed claim.

Both matched checkpoints score GSM8K 239/250, tools 30/30 and HumanEval 157/164, with zero HTTP errors and truncations. Greedy generation uses low reasoning effort, eight concurrent requests, a 32768 total-output-token cap and pinned task records. Length-limited outputs count as wrong. The stated same-engine Wilson/truncation/tool gates pass; this is not statistical equivalence or broad capability qualification.

**One retrieval regression remains:** CapyCTL scores 47/48 observations and 143/144 codes versus OpenAI 48/48 and 144/144. The miss is one of six 111k one-line requests (5/6, 17/18 codes); both pass all six original-format 111k documents. Both checkpoints pass 100/100 strict copy cases. No retrieval/copy request has an HTTP error or output-cap hit.

The published vLLM 0.30 recipe is a serving-setup comparison, not an engine-version-only experiment: it has prefix caching enabled and auto KV, not confirmed FP8 KV. Its quality is only a 50/30/50 screen: GSM8K 48/50, tools 30/30, HumanEval 49/50. Repeated long-document TTFT includes cache hits and must not be compared with uncached prefill on 0.32.

## Loading and allocator finding

Default lazy loading and eager-per-file loading both failed during candidate shard reads before KV allocation or graph capture, hitting the 3 GiB available-memory guard. The retained minima are 2,867,150,848 and 1,277,296,640 bytes. Eager loading did not fix this. Explicit lazy loading plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reached readiness without changing checkpoint bytes or installed packages; available memory stayed roughly 49–51 GB during the 37 shard reads, with a 41,606,955,008-byte minimum through readiness.

[INFERENCE] This intervention strongly supports CUDA caching-allocator fragmentation across the 28,263 output tensors, rather than a necessary extra live model copy. Process-local allocated/reserved counters were unavailable; fragmentation was not directly measured. The recipe therefore keeps expandable segments **and lazy loading**. Register the engine with `--approve-env PYTORCH_CUDA_ALLOC_CONF`; do not remove the variable merely because the checkpoint is small enough on disk.

Final shipped-file readiness was 480.64 seconds for CapyCTL versus 529.67 for matched OpenAI, with whole-run minimum MemAvailable 39,686,987,776 versus 25,940,623,360 bytes. These are sampled unified-memory system observations, not model-only GPU allocations. Local-path execution passed; Hub provisioning at `628e2ca15177aaea61dfe8cb4fbe1be9737129dc` has not been exercised.

## Use and reproduce

`deployment.yaml` pins `capyctl/gpt-oss-120b-MXFP4-FP8@628e2ca15177aaea61dfe8cb4fbe1be9737129dc`. Replace only that revision after upload; it is intentionally not an existing revision claim. Register vLLM 0.32.0 with the verified tokenizer vocabulary and `--approve-env PYTORCH_CUDA_ALLOC_CONF`, then use this deployment. Keep other large workloads off the device during loading and monitor unified MemAvailable. The 8 GiB KV grant does not imply the model fits in 8 GiB.

The conversion uses torch and safetensors, CPU only, buffers at most one layer, and never reconstructs whole-model BF16. Run from this directory with an existing compatible Python environment:

```sh
python3 -B convert/download_source.py --help
CUDA_VISIBLE_DEVICES='' python3 -B convert/mixed_attention.py /home/me/models/gpt-oss-120b /home/me/models/gpt-oss-120b-MXFP4-FP8 --scale channel
CUDA_VISIBLE_DEVICES='' python3 -B convert/audit_mixed.py /home/me/models/gpt-oss-120b /home/me/models/gpt-oss-120b-MXFP4-FP8 --vllm-root "$VLLM_PACKAGE" --out checkpoint-audit.json
```

Retain the source LICENSE, NOTICE where supplied and usage policy with the derivative. The pinned download verification and exhaustive checkpoint audit are under `convert/`. CPU/offline checks alone are not engine qualification. No claim is made for other runtimes, hardware, sampling or broad capability equivalence.

## License

Apache-2.0 derivative of [OpenAI GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b). This is my independent conversion, not an OpenAI release or endorsement. Preserve the upstream license and usage policy.
