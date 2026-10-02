# CapyCTL recipes

Deployment files for [CapyCTL](https://github.com/edurdias/capyctl), one per
engine, model and GPU, each measured through CapyCTL on the hardware it names.

A recipe is a directory with two files:

- `deployment.yaml`, which `capyctl deploy model --file` accepts as it is;
- `README.md`, which states the hardware, the engine and its exact version, the
  model and its pinned revision, the drafter if there is one, the commands from
  `capyctl engine add` to a first answer, and the numbers measured through
  CapyCTL, with the date and the CapyCTL version they were measured on.

Recipes hold configuration only: no scripts, container images or engine
patches. CapyCTL runs the engine installation you already have; see its
[engine guide](https://github.com/edurdias/capyctl/blob/main/docs/guide/engines.md).

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
| [tensorfold/nemotron-3.5-lightning-30b-a3b-4bit-gb10](tensorfold/nemotron-3.5-lightning-30b-a3b-4bit-gb10/) | TensorFold 0.6.1 | Nemotron 3.5 Lightning 30B-A3B 4-bit, MTP drafts | 1x GB10 | 132 tokens/s | 2026-10-02 |
| [tensorfold/qwen3.8-27b-nvfp4-gb10](tensorfold/qwen3.8-27b-nvfp4-gb10/) | TensorFold 0.6.1 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 48.2 tokens/s | 2026-10-02 |
| [vllm/qwen3-4b-gb10](vllm/qwen3-4b-gb10/) | vLLM 0.30.0 | Qwen3-4B, bf16 | 1x GB10 | 22.6 tokens/s | 2026-10-02 |
| [sglang/qwen3-4b-gb10](sglang/qwen3-4b-gb10/) | SGLang 0.5.20 | Qwen3-4B, bf16 | 1x GB10 | 22.4 tokens/s | 2026-10-02 |
| [vllm/qwen3.8-27b-nvfp4-gb10](vllm/qwen3.8-27b-nvfp4-gb10/) | vLLM 0.30.0 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 34.5 tokens/s | 2026-10-02 |
| [sglang/qwen3.8-27b-nvfp4-gb10](sglang/qwen3.8-27b-nvfp4-gb10/) | SGLang 0.5.20 | Qwen3.8-27B NVFP4, DFlash2 drafts | 1x GB10 | 33.1 tokens/s | 2026-10-02 |

### Qwen3.8-27B NVFP4 on three engines

The same checkpoint (`nvidia/Qwen3.8-27B-NVFP4` at `482ca0f3`), the same
DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c`), the same GB10 and
the same three requests:

| Engine | Decode, DFlash2 drafts | Decode, no drafts | Time to first token, drafts | Peak memory, drafts | Ready, warm |
|---|---|---|---|---|---|
| TensorFold 0.6.1 | 48.2 tokens/s | 11.9 tokens/s | 0.11 s | 30.0 GiB | 10 s |
| vLLM 0.30.0 | 34.5 tokens/s | 11.8 tokens/s | 0.23 s | 53.0 GiB | 121 s |
| SGLang 0.5.20 | 33.1 tokens/s | 12.1 tokens/s | 0.21 s | 44.9 GiB | 147 s |

Without drafts the three decode within 3% of each other. Each recipe's README has
the ranges, the settings and what each engine needed.

## Contribute one

Open a pull request that adds `<engine>/<model>-<gpu>/` with a
`deployment.yaml`, a `README.md` and a capyctl-bench report, and a row in the
table above. A recipe is accepted only if:

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
