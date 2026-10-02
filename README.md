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

## Contribute one

Open a pull request that adds `<engine>/<model>-<gpu>/` with a
`deployment.yaml` and a `README.md`, and a row in the table above. A recipe is
accepted only if:

- it was run through CapyCTL, not against the engine directly;
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
report the median. State the thinking setting and sampling parameters. Commands
and outputs in the README are copied from the run; shorten paths to
`/home/me/...` and replace the machine's name.

Every pull request runs `capyctl validate config --file` on each
`deployment.yaml` with the latest CapyCTL release. That check is offline: it
does not prove the recipe runs.

## License

Apache-2.0, as CapyCTL. See [LICENSE](LICENSE).
