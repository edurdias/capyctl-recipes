# Qwen3.6-35B-A3B CapyCTL NVFP4 on GB10

**Use the DeltaNet-only checkpoint with DFlash2 block 8 on vLLM 0.32.0.** It beats the published GB10 recipes tested in the [comparison](../../comparisons/qwen3.6-35b-a3b-builds-gb10/), not every tuning alternative. NVIDIA on our DFlash2 settings remains faster at eight short-prompt streams.

| Item | Pinned setup |
|---|---|
| Hardware | One NVIDIA GB10, unified memory |
| Engine / controller | vLLM 0.32.0 / CapyCTL 0.1.12 |
| Weights | [capyctl/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/capyctl/Qwen3.6-35B-A3B-NVFP4) at `cb64dfd952ed32865b3a7420d02d730522f498d8` |
| Drafter | `incoai/Qwen3.6-35B-A3B-DFlash2` at `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`, block 8 (seven speculative tokens plus bonus) |
| Context / cache | 262144 tokens, 24 GiB FP8 KV, eight running requests; LMCache off |
| Residency | Restart-only; parking/waking not qualified |

NVIDIA's experts and head stay NVFP4; DeltaNet `in_proj_qkv`, `in_proj_z` and `out_proj` become searched W4A16 NVFP4. Full attention stays NVIDIA's FP8. MTP, routers, small projections, norms and vision are retained. This changes projection weights, not DeltaNet recurrent arithmetic. See the [model card](convert/MODEL_CARD.md) for conversion, logical bytes/token, license, plain-vLLM command and limitations.

## Start it

Install [CapyCTL](https://github.com/edurdias/capyctl) 0.1.12 and vLLM 0.32.0 in an environment suitable for GB10. Register that existing environment and start the controller:

```bash
capyctl engine add /home/me/vllm-0.32.0-venv
capyctl start standalone
```

Configure host memory admission for available unified memory before starting the deployment. The 24 GiB KV request is **not** the full footprint or a minimum-memory recommendation. In another terminal, from this directory:

```bash
capyctl deploy model --file deployment.yaml
capyctl start deployment qwen3.6-35b-a3b-capyctl-vllm --wait
curl -s "$CAPYCTL_INFERENCE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-capyctl-vllm","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":512}'
```

Set `CAPYCTL_INFERENCE_URL` from your controller's inference endpoint and use its configured authentication. The examples show commands, not invented terminal output. Thinking remains at the model default; do not disable it when reproducing these results.

The deployment pins both repositories and explicitly lists runtime files, excluding conversion provenance. The repository is private until the PR merges; links use its final public URL. Hub materialization is untested. The shipped YAML was validated offline and run with only weights/drafter sources replaced by audited local copies; schema validation is not host resolution, downloading or engine qualification.

## Measured through CapyCTL

Fixed chat/code corpus, temperature zero, normal EOS, 512 total completion tokens including reasoning, one warm-up and three measured rounds. Aggregate means native completion tokens / round wall span, including one stream.

| Measurement | Elected checkpoint (`cd`) |
|---|---:|
| Aggregate tokens/s, 1 / 2 / 4 / 8 streams | 129.7 / 192.1 / 252.0 / 328.4 |
| Short TTFT, 1 / 2 / 4 / 8 streams, s | 0.087 / 0.141 / 0.202 / 0.269 |
| Ready, local weights | 241.18 s |
| Ready system-used-memory delta | 52.49 GiB |
| Peak system-used-memory delta | 54.02 GiB |
| GSM8K / tools / HumanEval | 234/250 / 30/30 / 159/164 |
| Task truncations, same order | 10 / 0 / 1 |

OS/file/JIT caches were not cleared; a separately measured warm restart is not available. Memory deltas are not absolute system usage or isolated model allocations.

| Exact input tokens | Decode tokens/s | TTFT s | Accepted length |
|---:|---:|---:|---:|
| 8,000 | 133.1 | 1.171 | 3.513 |
| 32,000 | 137.6 | 5.523 | 3.770 |
| 64,000 | 129.4 | 13.970 | 3.784 |
| 110,000 | 130.6 | 31.579 | 4.231 |

Three requests per exact native input, 512 output-token cap. Decode excludes TTFT. Accepted length includes the bonus token; it is not a probability. The separate shipped-file reproduction gives 128.6/336.3 aggregate at c1/c8, 131.2 c1 decode and a single 110k decode observation of 140.3; it does not replace these ranking medians.

Full quality uses a common 32768-token output cap. Truncations count wrong, tools require 30/30, HTTP errors must be zero and outputs are reviewed. Accuracy/truncation rates must meet the worse-direction endpoints of the baseline's ordinary two-sided 95% Wilson interval. This checkpoint passes. Small score differences are not proof of accuracy equivalence or superiority.

Long-context retrieval passes 60/60 observations (180/180 codes), including ordered exact extraction, at original input lengths 32k/64k/96k/111k/220k. Strict copy passes 100/100 on 2000-token documents. These are finite corpora, not universal 262k accuracy. See [full evidence and all competing rows](../../comparisons/qwen3.6-35b-a3b-builds-gb10/README.md).

## Limits and operator notes

Only GB10 was measured. CPU format/offline agreement checks and short compatibility probes on other engines are not engine qualification. Vision/multimodal quality and other hardware are unqualified. SGLang's cold FlashInfer JIT needed `MAX_JOBS=2` after an OOM; that is not a steady-state memory measurement.

The shipped-file run left an owned engine behind after CapyCTL shutdown. Confirmed owned parent/child processes were separately terminated; [cleanup evidence](../../comparisons/qwen3.6-35b-a3b-builds-gb10/bench/runs/cd-final/cleanup.json) records the clean boundary. Do not assume controller shutdown alone freed memory. No parking/waking qualification is claimed.

Apache-2.0 derivative of Qwen's model and NVIDIA's build. Retain upstream notices; this independent conversion implies no upstream endorsement.
