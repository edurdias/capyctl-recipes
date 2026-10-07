# CapyCTL recipes

Deployment files for [CapyCTL](https://github.com/edurdias/capyctl), one per
engine, model and GPU, each measured through CapyCTL on the hardware it names.

A recipe is a directory with two files:

- `deployment.yaml`, which `capyctl deploy model --file` accepts as it is;
- `README.md`, which states the hardware, the engine and its exact version, the
  model and its pinned revision, the drafter if there is one, the commands from
  `capyctl engine add` to a first answer, and the numbers measured through
  CapyCTL, with the date and the CapyCTL version they were measured on.

Recipes hold configuration only: no container images or engine patches, and
no scripts except one that rebuilds a converted checkpoint a recipe serves (in
the recipe's `convert/`). CapyCTL runs the engine installation you already
have; see its [engine guide](https://github.com/edurdias/capyctl/blob/main/docs/guide/engines.md).

## Use one

1. Install CapyCTL and start it on the machine
   ([Run on one machine](https://github.com/edurdias/capyctl/blob/main/docs/guide/one-machine.md)).
2. Install the engine version the recipe names in a virtual environment, then
   register it with `capyctl engine add <venv>`.
3. `capyctl deploy model --file <recipe>/deployment.yaml`, then
   `capyctl start deployment <name> --wait`.

The recipe's README lists the exact commands and what they printed.

## Recipes

| Recipe | Engine | Model | Hardware | Decode, one stream | Measured |
|---|---|---|---|---|---|
| [tensorfold/nemotron-3.5-lightning-30b-a3b-mlx-4bit-32gib-gb10](tensorfold/nemotron-3.5-lightning-30b-a3b-mlx-4bit-32gib-gb10/) | TensorFold 0.6.5 | Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, MTP drafts | 1x GB10, sized for 32 GiB | 137 tokens/s (one request at a time) | 2026-10-07 |
| [vllm/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10](vllm/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Nemotron 3.5 Lightning 30B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 69.9 tokens/s (164 tokens/s at 4 streams) | 2026-10-07 |
| [sglang/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10](sglang/nemotron-3.5-lightning-30b-a3b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Nemotron 3.5 Lightning 30B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 69.9 tokens/s (163 tokens/s at 4 streams) | 2026-10-07 |
| [tensorfold/qwen3.6-35b-a3b-mlx-4bit-32gib-gb10](tensorfold/qwen3.6-35b-a3b-mlx-4bit-32gib-gb10/) | TensorFold 0.6.5 | Qwen3.6-35B-A3B MLX 4-bit, MTP drafts | 1x GB10, sized for 32 GiB | 164 tokens/s (368 tokens/s at 8 streams) | 2026-10-07 |
| [vllm/qwen3.6-35b-a3b-nvfp4-32gib-gb10](vllm/qwen3.6-35b-a3b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Qwen3.6-35B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 76.7 tokens/s (180 tokens/s at 4 streams) | 2026-10-07 |
| [sglang/qwen3.6-35b-a3b-nvfp4-32gib-gb10](sglang/qwen3.6-35b-a3b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Qwen3.6-35B-A3B NVFP4 | 1x GB10, sized for 32 GiB | 83.5 tokens/s (186 tokens/s at 4 streams) | 2026-10-07 |
| [vllm/gpt-oss-20b-32gib-gb10](vllm/gpt-oss-20b-32gib-gb10/) | vLLM 0.30.0 | gpt-oss-20b, MXFP4 experts | 1x GB10, sized for 32 GiB | 46.7 tokens/s (195 tokens/s at 8 streams) | 2026-10-07 |
| [sglang/gpt-oss-20b-32gib-gb10](sglang/gpt-oss-20b-32gib-gb10/) | SGLang 0.5.21 | gpt-oss-20b, MXFP4 experts | 1x GB10, sized for 32 GiB | 45.8 tokens/s (191 tokens/s at 8 streams) | 2026-10-07 |
| [vllm/gemma-4-26b-a4b-nvfp4-32gib-gb10](vllm/gemma-4-26b-a4b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Gemma 4 26B-A4B NVFP4 | 1x GB10, sized for 32 GiB | 30.3 tokens/s (105 tokens/s at 4 streams) | 2026-10-07 |
| [sglang/gemma-4-26b-a4b-nvfp4-32gib-gb10](sglang/gemma-4-26b-a4b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Gemma 4 26B-A4B NVFP4 | 1x GB10, sized for 32 GiB | 30.0 tokens/s (103 tokens/s at 4 streams) | 2026-10-07 |
| [tensorfold/qwen3.8-27b-nvfp4-32gib-gb10](tensorfold/qwen3.8-27b-nvfp4-32gib-gb10/) | TensorFold 0.6.5 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10, sized for 32 GiB | 41.5 tokens/s (73.5 tokens/s at 2 streams) | 2026-10-06 |
| [vllm/qwen3.8-27b-nvfp4-32gib-gb10](vllm/qwen3.8-27b-nvfp4-32gib-gb10/) | vLLM 0.30.0 | Qwen3.8-27B NVFP4 | 1x GB10, sized for 32 GiB | 12.5 tokens/s (46.2 tokens/s at 4 streams) | 2026-10-06 |
| [sglang/qwen3.8-27b-nvfp4-32gib-gb10](sglang/qwen3.8-27b-nvfp4-32gib-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4 | 1x GB10, sized for 32 GiB | 13.0 tokens/s (one request at a time) | 2026-10-06 |
| [sglang/frognano-4b-rtx4090](sglang/frognano-4b-rtx4090/) | SGLang 0.5.21 | FrogNano-4B-2609, bf16 | 1x RTX 4090 Laptop GPU 16 GB | 62.5 tokens/s (381 tokens/s at 7 streams) | 2026-10-04 |
| [tensorfold/nemotron-3.5-lightning-30b-a3b-4bit-gb10](tensorfold/nemotron-3.5-lightning-30b-a3b-4bit-gb10/) | TensorFold 0.6.5 | Nemotron 3.5 Lightning 30B-A3B 4-bit, MTP drafts | 1x GB10 | 135 tokens/s (one request at a time on CUDA) | 2026-10-04 |
| [tensorfold/qwen3.8-27b-nvfp4-gb10](tensorfold/qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.5 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 38.9 tokens/s (195 tokens/s at 8 streams) | 2026-10-04 |
| [vllm/qwen3-4b-gb10](vllm/qwen3-4b-gb10/) | vLLM 0.30.0 | Qwen3-4B, bf16 | 1x GB10 | 22.1 tokens/s (205 tokens/s at 8 streams) | 2026-10-04 |
| [sglang/qwen3-4b-gb10](sglang/qwen3-4b-gb10/) | SGLang 0.5.20 | Qwen3-4B, bf16 | 1x GB10 | 22.4 tokens/s | 2026-10-02 |
| [vllm/qwen3.8-27b-nvfp4-gb10](vllm/qwen3.8-27b-nvfp4-gb10/) | vLLM 0.30.0 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 23.7 tokens/s (141 tokens/s at 8 streams) | 2026-10-04 |
| [sglang/qwen3.8-27b-nvfp4-gb10](sglang/qwen3.8-27b-nvfp4-gb10/) | SGLang 0.5.21 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 28.2 tokens/s (120 tokens/s at 8 streams) | 2026-10-04 |
| [tensorfold/frognano-4b-mlx-4bit-rtx4090](tensorfold/frognano-4b-mlx-4bit-rtx4090/) | TensorFold 0.6.3 | FrogNano-4B-2609, MLX 4-bit | 1x RTX 4090 Laptop GPU 16 GB | 52.4 tokens/s (319 tokens/s at 8 streams) | 2026-10-03 |
| [vllm/frognano-4b-rtx4090](vllm/frognano-4b-rtx4090/) | vLLM 0.30.0 | FrogNano-4B-2609, bf16 | 1x RTX 4090 Laptop GPU 16 GB | 59.9 tokens/s | 2026-10-03 |

### Qwen3.8-27B NVFP4 on three engines

The same checkpoint (`nvidia/Qwen3.8-27B-NVFP4` at `482ca0f3`), the same
DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c`), the same GB10, the
same CapyCTL commit (`1f2cfc7`) and the same three requests:

| Engine | Decode, DFlash2 drafts | Decode, no drafts | Time to first token, drafts | Peak memory, drafts | Ready, warm |
|---|---|---|---|---|---|
| TensorFold 0.6.5 | 38.9 tokens/s | 11.7 tokens/s | 0.12 s | 30.3 GiB | 14 s |
| vLLM 0.30.0 | 23.7 tokens/s | 11.9 tokens/s | 0.24 s | 51.5 GiB | 102 s |
| SGLang 0.5.21 | 28.2 tokens/s | 12.3 tokens/s | 0.21 s | 68.9 GiB | 253 s |

Without drafts the three decode within 6% of each other. With drafts and
sampling on, decode varies from request to request with the drafts accepted;
each recipe's README has the ranges, the settings, what each engine needed and
a capyctl-bench report with context and concurrency sweeps.

## Comparisons

The same model and machine on two engine versions, or on several engines,
measured through CapyCTL.
Each directory has the deployment files, the method and a capyctl-bench report.

| Comparison | Engine versions | Model | Hardware | Result | Measured |
|---|---|---|---|---|---|
| [comparisons/tensorfold-0.6.1-vs-0.6.2-nemotron-3.5-lightning-gb10](comparisons/tensorfold-0.6.1-vs-0.6.2-nemotron-3.5-lightning-gb10/) | TensorFold 0.6.1, 0.6.2 | Nemotron 3.5 Lightning 30B-A3B 4-bit, MTP drafts | 1x GB10 | Identical outputs; one-stream speed within 1% (one stream only: neither version batches this model on CUDA) | 2026-10-02 |
| [comparisons/tensorfold-0.6.1-vs-0.6.2-qwen3.8-27b-nvfp4-gb10](comparisons/tensorfold-0.6.1-vs-0.6.2-qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.1, 0.6.2 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | Identical outputs; 0.6.2 0.5% to 2.0% higher aggregate at 1 to 8 streams; context sweep to 128k within about 1% | 2026-10-02 |
| [comparisons/tensorfold-0.6.2-vs-0.6.3-qwen3.8-27b-nvfp4-gb10](comparisons/tensorfold-0.6.2-vs-0.6.3-qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.2, 0.6.3 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | Quick check (1, 4, 8 streams; 2k, 32k, 128k): identical outputs; throughput and decode within 1% at every point | 2026-10-02 |
| [comparisons/tensorfold-0.6.3-vs-0.6.5-qwen3.8-27b-nvfp4-gb10](comparisons/tensorfold-0.6.3-vs-0.6.5-qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.3, 0.6.5 | Qwen3.8-27B NVFP4, DFlash2 drafts; Nemotron 3.5 Lightning one stream | 1x GB10 | Quick check (1, 4, 8 streams; 2k, 32k, 128k): throughput and prefill within 0.4% except one slower one-stream round; identical outputs except the 32k and 128k replies (0.6.4 attention sums); Nemotron one stream 3.4% faster | 2026-10-04 |
| [comparisons/qwen3.8-27b-nvfp4-three-engines-gb10](comparisons/qwen3.8-27b-nvfp4-three-engines-gb10/) | TensorFold 0.6.3, vLLM 0.30.0, SGLang 0.5.21 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 8 running requests on each: TensorFold highest aggregate at 1 to 8 streams (194 against 136 and 122 tokens/s at 8) and fastest prefill; SGLang lowest time to first token at 5, 6 and 8 streams; vLLM fastest one-stream decode at 2k and 256k and flat ~51 GiB memory | 2026-10-03 |
| [comparisons/frognano-4b-nvfp4-three-engines-gb10](comparisons/frognano-4b-nvfp4-three-engines-gb10/) | TensorFold 0.6.3, vLLM 0.30.0, SGLang 0.5.21 | FrogNano-4B-2609 NVFP4, no drafts | 1x GB10 | 8 running requests on each: vLLM and SGLang 10% to 18% higher decode at 1 to 8 streams (414 and 416 against 352 tokens/s at 8); TensorFold fastest prefill, lowest time to first token at 1 to 5 streams, least memory (10 GiB at 8 streams) and a 5 s warm start | 2026-10-04 |
| [comparisons/qwen3.6-35b-a3b-three-engines-gb10](comparisons/qwen3.6-35b-a3b-three-engines-gb10/) | TensorFold 0.6.5, vLLM 0.30.0, SGLang 0.5.21 | Qwen3.6-35B-A3B: TensorFold MLX 4-bit with MTP drafts, vLLM and SGLang NVFP4 without drafts | 1x GB10 | The three 32 GiB recipes as served (vLLM and SGLang run 4 requests, TensorFold 8): TensorFold about twice the one-stream decode (162 against 77 and 84 tokens/s) and 368 against 180 and 187 tokens/s at 8 streams; vLLM fastest prefill and time to first token up to 2k; TensorFold least memory | 2026-10-07 |

## Contribute one

Open a pull request that adds `<engine>/<model>-<gpu>/` with a
`deployment.yaml`, a `README.md` and a capyctl-bench report, and a row in the
Recipes table. A recipe is accepted only if:

- it was run through CapyCTL, not against the engine directly;
- it includes a [capyctl-bench](tools/capyctl-bench/) report in `bench/`: the
  results JSON from `capyctl_bench.py run` and the `report.html`, `summary.md`
  and `data.csv` that `capyctl_bench.py report` renders from it (the PNG charts
  are optional);
- its README has the numbers below, measured on the hardware it names;
- it names the exact CapyCTL version (or commit), engine version, model
  revision and drafter revision it was measured with, and the date.

The numbers each recipe reports:

| Number | How |
|---|---|
| Ready time, cold | `capyctl start deployment <name> --wait` after `deploy model`, weights already in the model store |
| Ready time, warm | the same command again after `capyctl stop deployment <name>` has finished |
| Time to first token | streaming chat completion through the CapyCTL inference endpoint, request sent to first content or reasoning token |
| Decode tokens/s, one stream | the same request, (completion tokens - 1) / (last token time - first token time) |
| Peak memory | the peak CapyCTL measured (`capyctl status deployment <name> --json`, `startup.measured`), next to the reservation the deployment holds |

Use three different prompts, `max_tokens: 512`, one request at a time, and
report the median. The capyctl-bench report adds the context sweep and the
concurrency curves; its README explains each metric and the rules that keep
two reports comparable. State the thinking setting and sampling parameters. Commands
and outputs in the README are copied from the run; shorten paths to
`/home/me/...` and replace the machine's name.

Every pull request runs `capyctl validate config --file` on each
`deployment.yaml` with the latest CapyCTL release. That check is offline: it
does not prove the recipe runs.

## License

Apache-2.0, as CapyCTL. See [LICENSE](LICENSE).
