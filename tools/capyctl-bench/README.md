# capyctl-bench

The benchmark every recipe and comparison in this repository uses. `run`
measures a model through the CapyCTL inference endpoint and writes one results
file; `report` puts any number of results files (versions, engines, settings)
on the same charts and tables. Chat models are measured with streaming chat
completions; decision models (`api: decisions`, `POST /v1/systemone`) with
`run --decisions`, see [Measure a decision model](#measure-a-decision-model).

`capyctl_bench.py` is one file (plus `di_home_appliance.py`, the Decision
Index's Home appliances generator, vendored from its MIT kit) and needs Python
3.10 or later, standard library only. The PNG charts of `report` need matplotlib; run `report` through
[uv](https://docs.astral.sh/uv/) to get it for that one command:
`uv run --with matplotlib python3 capyctl_bench.py report ...`. Without
matplotlib, `report` still writes the HTML report, the Markdown tables and the
CSV, and says that it skipped the PNGs.

## Run a recipe's sweep

Against CapyCTL standalone on the same machine, with a deployment called
`qwen3-4b-vllm` that is ready:

```bash
python3 tools/capyctl-bench/capyctl_bench.py run \
  --api-key-file ~/.local/state/capyctl/identity/credentials \
  --model qwen3-4b-vllm --label "vLLM 0.30.0" \
  --context-sweep 0.5k,1k,2k,4k,8k,16k,32k,64k,128k \
  --concurrency 1-8 \
  --memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'" \
  --meta capyctl=0.2.0 --meta engine=vllm-0.30.0 --meta gpu=GB10 \
  --out results/vllm-0.30.0.json
```

`--endpoint` defaults to `http://127.0.0.1:8443/v1`. The key comes from
`--api-key-file` (the standalone `credentials` file or the server's
`server-credentials.json`) or from `CAPYCTL_API_KEY`.

## Measure a decision model

A decision model answers typed questions about a state in one forward pass and
generates no tokens, so it has no time to first token and no tokens per second.
`run --decisions` measures what a decision deployment is used for: how long a
request takes, what a cached state saves, how many requests and questions per
second it serves at a fixed number of users, how often it is right and how
well its probabilities are calibrated.

```bash
python3 tools/capyctl-bench/capyctl_bench.py run --decisions \
  --api-key-file ~/.local/state/capyctl/identity/credentials \
  --model clef-flash-llamacpp --label "Clef-Flash, llama.cpp v0.7.0" \
  --accuracy \
  --memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'" \
  --meta capyctl=0.1.5 --meta engine=llamacpp-v0.7.0 --meta gpu=GB10 \
  --meta model_revision=<sha> --meta gguf_sha256=<sha> \
  --out results/clef-flash-llamacpp.json
```

Add `--plan` to print what that run sends and an estimate of how long it
takes, without sending anything:

```text
$ capyctl_bench.py run --decisions --model m --label x --accuracy --plan
Plan of one decisions run (no request sent). Time assumes 60 ms per request plus its input tokens at 2500 tok/s (--plan-overhead-ms, --plan-prefill-tps).
  calibration           2 requests  ~   0.0 min  2 requests
  latency grid        945 requests  ~  23.4 min  9 points x (100 runs + 5 warm-up)
  cache test          206 requests  ~   3.2 min  100 cold + 1 + 100 repeated
  throughput          823 requests  ~  12.7 min  4 levels x 200 (+ warm-up), serial upper bound
  AG News             500 requests  ~   1.1 min  ~175 input tokens each
  BoolQ               500 requests  ~   1.2 min  ~225 input tokens each
  Typed Decisions     400 requests  ~   2.0 min  ~585 input tokens each
  API-Bank            508 requests  ~  23.8 min  ~6,875 input tokens each
  GPQA Diamond        196 requests  ~   0.4 min  ~190 input tokens each
  Home appliances      88 requests  ~   1.3 min  ~2,050 input tokens each
  total             4,168 requests  ~  69.1 min
```

The request counts are exact; the time is an assumption (a fixed cost per
request plus the input tokens at a prefill rate, both flags), not a
measurement, and the throughput levels are counted as if serial. The token
estimates come from the measured mean request size of each set at 4
characters per token. API-Bank sends the whole 53-tool catalog with every
request (about 27,500 characters): the deployment needs a context of about
10k tokens, and the 8k-token latency points need about 9k.

Every request is one `POST {endpoint}/systemone` with the deployment name as
`model`, a `state` and a map of `questions`, sent with the same API key as chat.
It runs, in order:

1. **Calibration.** Two requests with filler states of 20 and 200 sentences fit
   the state tokens per sentence from `usage.input_tokens`. An engine that
   reports no usage gets a fixed estimate of 4 characters per token instead; the
   results say which (`calibration.method`).
2. **Latency grid.** One request at a time, for every question count
   (`--question-counts`, default 1, 3, 5) and state length (`--state-sweep`,
   default about 200, 2k and 8k tokens), `--runs` timed requests each (100),
   after `--warmup` unrecorded requests of the same shape (5). The questions
   are the first N of a fixed set of five mixed ones (choice, noul, score,
   choice, noul).
3. **Cache test**, unless `--cache-runs 0`: `--cache-runs` requests (100), each
   with a new state (cold prefix), then one untimed request with one more new
   state and `--cache-runs` repeats of exactly that request (repeated state).
   The questions and state length are `--concurrency-questions` (3) and
   `--concurrency-state` (2k).
4. **Throughput** at each of `--concurrency` streams (1, 2, 4, 8), unless
   `--throughput-requests 0`: a closed loop, each stream sending its next
   request as soon as its last one returns, `--throughput-requests` requests
   per level (200) after `max(streams, --warmup)` unrecorded ones, with the
   same question count and state length as the cache test.
5. **Quality**, with `--accuracy`: the built-in sets below, all of them unless
   `--accuracy-sets` names some.

Every filler state is deterministic text with a seed of its own, warm-ups
included (the seeds continue across sections), so no request reuses another's
prefix cache except the cache test's repeated block, and two runs send the same
requests. The results keep the request manifest (`decisions.manifest`: the five
questions, their types and option counts, the body fields and the JSON
serialization) so another tool can replay the same shape.

### Speed numbers

| Number | Definition |
|---|---|
| Latency p50, p95, ms | Client side: request written to the whole answer read. Linear-interpolation percentiles over the timed runs at a point (100 by default, so the p95 rests on about five requests; read it as a tail indication, not a precise quantile). |
| Per question p50, ms | Each request's latency over the questions it answered, p50. Amortized cost per answered question, not the latency of a one-question request (the 1-question points measure that). |
| Input tokens | `usage.input_tokens` per request: state, questions and the engine's template. The same text counts differently on different models; the tables show what each engine reported. |
| Cold prefix, repeated state | The cache test's two blocks; repeated p50 / cold p50 below 1 means the engine reuses work for a state it has seen (a prefix or result cache). |
| Requests per second | Throughput: answered requests / (first request sent to last request finished) at a fixed number of streams. |
| Questions per second | Throughput: answered questions over the same time. |
| Error rate, % | Throughput: failed requests / requests sent at that level. |
| Peak memory, GiB | The highest `--memory-cmd` sample while the request, level or set ran. A recipe quotes CapyCTL's own `startup.measured` peak instead. |

A response counts only when every question has an answer of its type (a
`choice` among the options, a number for `noul` and `score`); anything else, an
HTTP error included, is an error with the engine's message.

### Quality sets

| Set (`--accuracy-sets` name) | Source, pin | Requests | What a request is | Licence |
|---|---|---:|---|---|
| AG News (`ag_news`) | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) test at `eb185aade064a813bc0b7f42de02595523103ca4` | 500 of 7,600, sampled | choice: "Which section of a news site does this article belong to?" with World, Sports, Business and Sci/Tech, each described; the state is the article | see the dataset card |
| BoolQ (`boolq`) | [`google/boolq`](https://huggingface.co/datasets/google/boolq) validation at `35b264d03638db9f4ce671b711558bf7ff0f80d5` | 500 of 3,270, sampled | noul: the row's question, capitalized, with "?"; the state is the passage | CC BY-SA 3.0 |
| Typed Decisions (`typed_decisions`) | [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `all` test at `7189f46b02248bd2569df7325c9df8e3b24e90a7` | all 400 | the row's `state` and `questions`, parsed from their JSON strings: the case and its five questions in one request, as the dataset card asks | Apache-2.0 |
| API-Bank (`api_bank`) | Decision Index catalog 3, built from [`AlibabaResearch/DAMO-ConvAI`](https://github.com/AlibabaResearch/DAMO-ConvAI) `api-bank/` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` | all 508 | one per API turn of the lv1-lv2 sample dialogues: the dialogue before the turn and the whole 53-tool catalog as state, one choice of the tool among all 53, shuffled per row | see `api-bank/LICENSE` there |
| GPQA Diamond (`gpqa_diamond`) | Decision Index catalog 25, built from [`idavidrein/gpqa`](https://github.com/idavidrein/gpqa) `dataset.zip` (sha256 `461ae732…`; the password is public in its README, no account needed) | 196 of 198 | the question as instructions, an empty state, the four published options keyed A to D, shuffled per row | CC BY 4.0; the authors ask that the questions not be published in plain text |
| Home appliances (`home_appliances`) | Decision Index catalog 9, generated by the kit's own code (vendored as `di_home_appliance.py`, MIT) | 88 of 160 | a household request with the home's devices, groups and policy as state; 17 to 19 choice questions (handling, every device's membership in the target set, outcome, a state read, post-conditions) | MIT |

`--accuracy-sets` also takes suite names: `classic` (AG News, BoolQ),
`typed`, `decision-index` (the three components) and `all` (the default).

**AG News and BoolQ** are small public sets that map one to one onto the two
common question types; they continue the earlier measurements. The sample is
`--accuracy-samples` rows per set (500), the rows with the smallest sha256 of
`seed:set:row_idx` for `--accuracy-seed` (0), the same on every machine and
Python. They are named training sources of some decision models: an
application check, not evidence of generalization.

**Typed Decisions** is the full 400-case test split, 2,000 decisions in four
workflows. Its gold is the mean of three samples of a small teacher model, so
a score measures agreement with that teacher, not correctness (the card's
teacher self-agreement ceiling is 0.735).

**The Decision Index components** are rebuilt from pinned upstream sources
exactly as the [decision-index kit](https://github.com/apolinario/decision-index)
(`apolinario/decision-index` at `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`,
edition 0.3) builds them, then checked byte for byte against the sha256 the
kit's `hub/manifest.json` records for its normalized file of the benchmark
(`33d2cdbe…` API-Bank, `96e00f12…` GPQA Diamond, `13862aa6…` Home appliances);
a source that changed stops the run. The kit's cuts follow: GPQA Diamond drops
the two rows its `hub/excluded-questions.json` excludes (byte-identical
options), Home appliances keeps the 88 rows of its 0.2.1 subset (duplicate test
states and states identical to a dev row dropped; the kept run ids are checked
against the kit's `run_ids_sha256`). The kit publishes no request files
(`multimodalart/decision-index-suite-*` is not public) and asks that the suite
not be redistributed, so the tool builds it on your machine. Each row is sent
unchanged and scored as the kit scores it.

Datasets-server sets come from the Hugging Face
[datasets-server](https://huggingface.co/docs/dataset-viewer) JSON rows API,
so no parquet reader is needed. That API serves a repository's current
revision only, so each set is pinned by the revision it was read at and the
sha256 of its rows (canonical JSON, one line per row, in row order): if the
split changes, the run stops instead of measuring other rows. The first
`--accuracy` run downloads and builds every set (a few minutes) into
`--dataset-cache` (`~/.cache/capyctl-bench/datasets`); later runs read the
cache. `--accuracy-limit N` scores only the first N rows of Typed Decisions and
of each component, for a smoke test or a pilot; the results mark those sets
`partial`.

**Do not redistribute the sets.** Publish results files, reports and the
request manifests, never the rows: the results keep row ids, payload hashes
(for the Decision Index the kit's own `payload_sha256`), answers and scores,
never the rows' text. Several sources restrict republication (GPQA asks that
its questions stay off the open web; the combined corpus carries each source's
own terms).

### Quality numbers

| Number | Sets | Definition |
|---|---|---|
| Accuracy, % | AG News, BoolQ | Share of the answered rows right; a noul counts as yes at 0.5 or more. |
| Macro-F1, % | AG News, BoolQ | Mean over labels (seen as truth or prediction) of 2TP / (2TP + FP + FN). |
| Brier, vector | AG News, BoolQ | Mean over rows of Σₖ (pₖ − yₖ)², the sum over every class. 0 is perfect. |
| Brier, binary | BoolQ | Mean of (p(yes) − y)²: half the two-class vector value. Always 0.5 scores 0.25. |
| NLL | AG News, BoolQ | Mean −ln p(true class), p clipped at 1e-9. |
| ECE | AG News, BoolQ, Typed Decisions | Top label: the most probable option (ties to the first) and its probability, 10 equal-width bins over [0, 1], Σ (nᵦ / n) · \|accuracyᵦ − mean confidenceᵦ\|. The results keep every bin's count, mean confidence and accuracy; the report draws them as reliability diagrams. |
| Accuracy, % | Typed Decisions | The answer's most probable option (ties to the first in the gold's order) equals the gold's hard `label`. Per question type too. |
| Accuracy, gold ties, % | Typed Decisions | Right when the prediction is tied for the gold distribution's top (35 of the 2,000 golds have a tie), the convention of the public third-party scorer below. |
| KL from gold | Typed Decisions | KL(gold ‖ prediction) = Σ gold · ln(gold / max(p, 1e-9)) per decision, mean over decisions. |
| Brier, soft gold | Typed Decisions | Σ over options (p − gold)² per decision, mean over decisions (the card's per-decision Brier). |
| Brier, hard label | Typed Decisions | Σ over options (p − one-hot label)² per decision, mean over decisions. |
| Score MAE | Typed Decisions | Score questions: \|the answer's `score` − the gold's expected score\|, in rubric levels. |
| Coverage, % | Typed Decisions, components | Answered requests / requests. A failed case loses all five of its decisions. |
| Score, % | components | As the kit scores it, over the answered rows, on the answer's `choice` (not the argmax of its probabilities): accuracy for API-Bank and GPQA Diamond, case exact accuracy (every one of a request's fields right) for Home appliances. |
| Raw, skill, % | components | The board's per-benchmark value: raw = score (rounded to 4 places) × coverage, unanswered counting as wrong; skill = clip((raw − chance) / (1 − chance), 0, 1) with the kit's chance (API-Bank 0.0189, GPQA Diamond 0.25, Home appliances 0). |

A Typed Decisions noul answer is its probability of true and its complement; a
choice or score answer without a `probabilities` map is scored one-hot on its
choice (or nearest level) and counted in `derived_probabilities`. An HTTP 400,
413 or 422 whose message names a capacity limit (the kit's list) is
`unsupported`, any other failure an `error`; both count against coverage.

### Parity with published numbers

- **Typed Decisions.** The card documents the request shape and names its
  metrics (accuracy, KL from gold, Brier, ECE) but not their formulas, and
  notes that ECE is computed differently by different submitters. KL and the
  soft-gold Brier as defined here reproduce the card's reference row for a
  uniform answer exactly (0.444 and 0.238 on the pinned test split). The card's
  uniform accuracy (0.308) and ECE (0.169), and its train-prior row, are not
  reproducible from the card: they depend on an unstated tie-break and binning.
  The public scorer of
  [SaylorTwift/typed-decisions-evals](https://huggingface.co/buckets/SaylorTwift/typed-decisions-evals)
  (`scripts/score_model.py`, used for the rows of discussion 37) counts gold
  ties as right and bins ECE on that; fed its published Clef-Flash predictions,
  this tool's scorer gives the same accuracy (0.7100, as "gold ties"), KL
  (0.2093), Brier (0.1105), ECE (0.0164, `ece_gold_ties`) and score MAE (0.2954),
  and 0.7035 against the hard label. Rows self-reported in the card's
  discussions used their own definitions.
- **Decision Index.** The three components are single benchmarks of the 37 in
  the public index; together they are not a Decision Index score, and the
  board's Full score adds private tests that cannot be run locally. The rows
  and the score definitions match the kit's; raw and skill reproduce the kit's
  board fixture (`tests/fixtures/board-0.3.json`, checked in the tests). Board
  numbers were measured on other hardware and engines (often BF16 on vLLM), so
  a quantized model on llama.cpp need not match them.
- **Speed.** The board's latency protocol (750 mixed requests on an RTX PRO
  6000) and vendors' numbers come from other hardware and request mixes;
  compare them with a CapyCTL run only as a separate reference, never as a
  speedup.

`report` draws latency against state length for each question count (p50
solid, p95 dashed), requests and questions per second and latency against
streams, and a reliability diagram per calibrated set; it tables the cache
test, Typed Decisions (every metric above, per type and per workflow), the
components (score, coverage, skill) and AG News and BoolQ (accuracy, macro-F1,
Brier, NLL, ECE, with every reliability bin's count in `summary.md`). With
several results files it compares them, one series each, ready to commit as a
recipes `comparisons/` entry. For up to three series of one model the summary
shows the quality bars (accuracy, then the components) first, with changes
against the first series, then latency at the most questions and questions per
second against streams; without quality sets it leads with requests per second
at the fewest and most streams (latency p50 without throughput). A field of
models gets the scoreboard layout below. Reports compare decisions results with
decisions results only.

### Compare decision models

Run every model with the same options on the same host, one deployment at a
time, then render them together. The output directory is a recipes
comparison entry's `bench/`:

```bash
for m in clef-flash-llamacpp kev-4b-llamacpp clef-flash-sglang; do
  python3 tools/capyctl-bench/capyctl_bench.py run --decisions --accuracy \
    --model "$m" --label "$m" --meta gpu=GB10 ... --out "results/$m.json"
done
uv run --with matplotlib --with pillow python3 tools/capyctl-bench/capyctl_bench.py report \
  results/*.json --out comparisons/decision-models-gb10/bench \
  --title "Open decision models on one GB10" --summary \
  --relabel "clef-flash-llamacpp=Clef-Flash · llama.cpp" --relabel "kev-4b-llamacpp=Kev-4B · llama.cpp" \
  --series-meta clef-flash-llamacpp:engine_peak_gib=15.2 --series-meta kev-4b-llamacpp:engine_peak_gib=25.0
```

`--relabel FILE_LABEL=NEW_LABEL` (repeatable) shows the series whose results
file says `label: FILE_LABEL` as `NEW_LABEL`, and `--series-meta
FILE_LABEL:key=value` (repeatable) adds `key: value` to that series' meta, so
the files stay as `run` wrote them. Both go by the label in the file
(`--series-meta` before `--relabel`); a label no file has stops the report and
lists the ones there are.

With more than three series, or series of different models, the results are a
field of models rather than variants of one, and a change against the first
series means nothing. So no table gives one (`summary.md` and the report's
tables show values only), and the summary replaces its panels with:

- a **scoreboard**: a row per series, a column each for Typed Decisions
  accuracy, the API-Bank, GPQA Diamond and Home appliances scores, AG News and
  BoolQ accuracy, Typed Decisions ECE, latency p50 at 1 question and a 2k-token
  state, answered questions per second at 1 stream (the throughput shape) and
  memory; each cell shaded by its rank in the column (best darkest,
  lower-is-better columns marked ↓, values that print the same tie). Each row
  starts with the series' marker, the same color and shape as in the chart, so
  there is no separate legend. `summary.md` repeats it as a table.
- **quality vs speed**: Typed Decisions accuracy against ms per answered
  question (p50, 3 questions, 2k-token state, log scale), y fitted to the data.
  Every label sits directly beside its own marker (right, left, above or below),
  clear of every other label and marker and nearer its own marker than any
  other; the placements are chosen together, so a crowded point gets the free
  side. Only a point with no free side gets a label a little further out, with a
  short leader line that crosses no label, marker or other leader.

**Memory in the scoreboard.** When every series carries meta `engine_peak_gib`
(CapyCTL's measured peak of the engine process, for example from `capyctl`'s
`startup.measured`, added with `--series-meta`), the column is that, headed
"Memory, engine peak, GiB". Otherwise it is the highest `--memory-cmd` sample of
the run, headed "host peak": on a host-wide command that includes the OS
(about 5 GiB on a GB10) and anything else running, not the model's footprint.
`summary.md` says which one it is.

Up to three series of one model keep the bars and change labels. `charts/`
holds the latency, throughput and reliability charts, and `summary/` the
shareable page. Keep general zero-shot models apart from models fine-tuned on
a set's train split (Typed Decisions lists them separately), and say which
models name AG News or BoolQ as training sources.

## Compare two results

```bash
uv run --with matplotlib python3 tools/capyctl-bench/capyctl_bench.py report \
  results/tensorfold-0.6.2.json results/vllm-0.30.0.json \
  --out report/ --title "Qwen3.8-27B NVFP4 on one GB10" --summary
```

The pages open in the light theme; the sun/moon button switches to dark and
the browser remembers the choice. Print and PDF are always light.

The summary shows 2 to 4 panels, picked from what the results hold. The bars
are aggregate tok/s on the prompt set at the fewest and the most streams every
series ran; without a concurrency sweep they are one-stream generation tok/s at
about 1k tokens and at the longest context. The sweep's filler prompts are easy
for a drafter to accept, so they stay in the line charts when real prompts were
measured. The lines are generation, prompt processing and time to first token
against context (log scale when it spans a wide range), then aggregate tok/s
against streams, as room allows. With one series the bars show the values; with
several, each also carries the change against the first series, sized to fit
its bar. The subtitle takes the
hardware from `--meta gpu=...` (or `hardware=...`).

The first file is the baseline: it is drawn in orange, and `summary.md` gives
every other series' change against it. The output directory holds:

| File | What |
|---|---|
| `report.html` | One self-contained page: charts as inline SVG, a full table per series, run metadata and method. Print it from the browser ("Save as PDF") for a PDF. |
| `charts/*.png` | Each chart at 1600x900 for READMEs (needs matplotlib). Light; `--theme dark` for dark. |
| `summary/` | With `--summary`: a one-page shareable summary. `summary.html`, plus `summary-wide.png` (1200x675 at 2x, for X and LinkedIn) and `summary-tall.png` (1080x1350 at 2x, for phones and feeds), each also as `-dark.png`. |
| `summary.md` | Markdown tables, one per metric, ready to paste into a recipe README. |
| `data.csv` | Every point and metric in long form: median, min, max and count. |
| `report.zip` | All of the above plus the input results files. |

## Options of `run`

| Option | Default | What |
|---|---|---|
| `--decisions` | off | Measure a decision model through `POST /v1/systemone` (above) instead of chat completions. |
| `--context-sweep` | | Chat: prompt sizes, `k` = 1000 tokens. One request at a time. |
| `--runs` | 3 (100 with `--decisions`) | Requests per context or latency point. |
| `--max-tokens` | 128 | `max_tokens` of each context sweep request. |
| `--concurrency` | (`1,2,4,8` with `--decisions`) | Stream counts, `1-8` or `1,2,4,8`. |
| `--rounds` | 5 | Chat: rounds per stream count. |
| `--concurrency-max-tokens` | 512 | `max_tokens` of each concurrency request. |
| `--prompts` | `prompts.json` | Prompt set for concurrency (8 prompts that ask for long answers). Every round, warm-up included, takes the next prompts of the set across all stream counts; the set repeats only when it runs out, and the tool says so (a repeated long prompt can hit the engine's prefix cache). |
| `--warmup` | 1 (5 with `--decisions`) | Unrecorded requests (sweep) or rounds (each stream count) first. With `--decisions`: requests before every latency point and the cache test, `max(streams, N)` before each throughput level. |
| `--temperature` | 0 | Sampling temperature. |
| `--extra-body` | | JSON object merged into every request body, for example `'{"ignore_eos": true}'` or `'{"chat_template_kwargs": {"enable_thinking": false}}'`. |
| `--memory-cmd` | | Shell command whose output's last number is the memory in use, in GiB. Sampled every `--memory-interval` seconds (0.5). |
| `--record-chunks` | off | Keep the arrival time of every token chunk, for replays. |
| `--meta KEY=VALUE` | | Run metadata (CapyCTL version, engine version, GPU, commit). Repeatable. |
| `--label` | | The series name in reports. |

With `--decisions` only (a chat run refuses them, and `--decisions` refuses
`--context-sweep`, `--record-chunks` and `--rounds`):

| Option | Default | What |
|---|---|---|
| `--question-counts` | `1,3,5` | Questions per latency request, 1 to 5. |
| `--state-sweep` | `0.2k,2k,8k` | State lengths of the latency grid, in tokens. |
| `--concurrency-questions` | 3 | Questions per cache-test and throughput request. |
| `--concurrency-state` | `2k` | State length of each cache-test and throughput request. |
| `--cache-runs` | 100 | Requests of each block of the cache test; 0 skips it. |
| `--throughput-requests` | 200 | Requests per throughput level; 0 skips throughput. |
| `--accuracy` | off | Also measure quality on the built-in sets. |
| `--accuracy-sets` | `all` | Set names or suites (`classic`, `typed`, `decision-index`, `all`), comma-separated. |
| `--accuracy-samples` | 500 | Rows sampled from AG News and BoolQ. |
| `--accuracy-seed` | 0 | Seed of that sample. |
| `--accuracy-limit` | 0 (all) | Score only the first N rows of Typed Decisions and each component (a smoke test); the results say `partial`. |
| `--dataset-cache` | `~/.cache/capyctl-bench/datasets` | Where downloaded and built sets are kept. |
| `--datasets-url` | `https://datasets-server.huggingface.co` | The datasets-server to read rows from (a mirror; the tests use the fake server). |
| `--plan` | off | Print the requests and an estimated time of the run, then exit without sending anything. |
| `--plan-overhead-ms`, `--plan-prefill-tps` | 60, 2500 | The `--plan` time model: fixed time per request, input tokens per second. |

Memory commands that work:

```bash
# Unified memory (GB10) or any host: RAM in use on this machine
--memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'"
# A discrete GPU on another machine
--memory-cmd "ssh gpu-box nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += \$1} END {print s / 1024}'"
```

## What each metric means

For chat runs (the decision numbers are under
[Measure a decision model](#the-numbers)): every request is a streaming chat completion with
`stream_options.include_usage`; token counts come from the final usage chunk.
A token chunk is a chunk whose delta carries `content` or reasoning text
(`reasoning_content`, or `reasoning`, counted once). A stream that closes before
`data: [DONE]` is an error even when tokens arrived: something cut it, for
example CapyCTL's stream idle bound during a long prefill.

| Metric | Definition |
|---|---|
| Time to first token (TTFT), s | Request sent to the first token chunk. |
| Generation, tok/s | (completion tokens - 1) / (last token chunk - first token chunk). The decode rate one stream sees, drafts included. |
| Time per output token (TPOT), ms | The inverse of generation: (last - first token chunk) / (completion tokens - 1). |
| Prompt processing, tok/s | Prompt tokens / TTFT. Includes queueing and the first decode step, so it understates raw prefill at short prompts. |
| Total time, s | Request sent to the end of the stream. |
| Output bytes or chars per second | UTF-8 bytes (or characters) of content and reasoning over the generation window. Compares engines whose tokenizers differ. |
| Peak memory, GiB | The highest `--memory-cmd` sample taken while the request (or round) ran. |
| Aggregate, tok/s | Concurrency: a round's completion tokens / (first request sent to last stream finished). Median over rounds. |
| Per stream, tok/s | Concurrency: median generation tok/s over every request at that stream count. |
| TTFT p50, p95 | Concurrency: over every request at that stream count. |
| Draft acceptance | Accepted / drafted tokens, summed, when the engine reports draft counts in the stream (see engine extras). |

Context sweep prompts are deterministic filler text with a different seed for
every request, so no request reuses another's prefix cache. Two calibration
requests (`max_tokens: 1`) fit the prompt size to the target; the tables show
the prompt tokens the engine reported, which land within a few percent of the
target. A model can stop before `max_tokens`; pass `--extra-body
'{"ignore_eos": true}'` where the engine accepts it if you need every request
to run to the limit.

## Honesty rules

A report compares fairly only when:

- every series ran on the same host, with the same prompts (or states,
  questions and accuracy sample) and settings (`report` names any setting that
  differs between series);
- the run-to-run spread is stated: tables show the median with the min-max
  range under it, and a recipe README quotes the range next to the median;
- the numbers were measured through CapyCTL, not against the engine directly,
  and the report says so;
- the CapyCTL version, engine version, model revision and drafter revision are
  in `--meta`, and nothing else ran on the machine during the run.

## Results schema (version 3)

One JSON object per run. Times are seconds unless the name says otherwise;
`*_at` fields are ISO 8601 UTC except `sent_at`, which is a Unix timestamp.
Version 2 added `mode` and the `decisions` section. Version 3 changes only
decisions runs: per-question latency, the request `manifest`, the `cache`
section, closed-loop throughput levels (`requests` instead of burst `rounds`)
and quality sets of three kinds. `report` reads versions 1 to 3: a version 1
file is a chat run from before `mode` existed, a version 2 decisions file shows
its burst rounds' median as requests per second and its AG News and BoolQ
accuracy and Brier.

```text
schema_version     3
mode               "chat" or "decisions" (absent in version 1: chat)
tool               {name, version, python}
label              series name, e.g. "TensorFold 0.6.2"
model              model name sent in each request
endpoint           scheme, port and path only: "http://<host>:8443/v1"
started_at, finished_at
meta               {key: value} from --meta; sample: true marks sample data (banner in reports)
settings           chat: {temperature, extra_body, memory_command (bool), memory_interval_s, record_chunks}
                   decisions: {extra_body, memory_command, memory_interval_s}
context_sweep      chat: null, or
  settings         {targets, runs, max_tokens, warmup, unit}
  calibration      {per_sentence_tokens, overhead_tokens, probes: [{sentences, prompt_tokens}]}
  points[]         {label "4k", target_prompt_tokens, sentences, requests[], summary}
concurrency        chat: null, or
  settings         {levels, rounds, warmup, max_tokens, prompt_set, prompts: [ids]}
  levels[]         {concurrency, rounds[], summary}
    rounds[]       {round, wall_s, completion_tokens, aggregate_tps, errors, memory_peak_gib, requests[]}
decisions          decisions only:
  manifest         {endpoint, body, serialization, questions: {id: question}, question_order,
                    question_types, options, n_questions, state}
  calibration      {method ("usage.input_tokens" or "4 characters per token"), per_sentence_tokens,
                    overhead_tokens, probes: [{sentences, input_tokens}]}
  latency
    settings       {question_counts, state_targets, runs, warmup, warmup_scope, unit, concurrency,
                    questions: {id: question}}
    points[]       {label "3q 2k", questions, target_state_tokens, state_label, sentences, requests[], summary}
  cache            null, or
    settings       {questions, state_tokens, sentences, runs, warmup}
    cold           {requests[], summary}
    repeated       {first (the untimed first sight), requests[], summary}
  concurrency      null, or
    settings       {levels, requests, warmup, mode, questions, state_tokens, sentences}
    levels[]       {concurrency, requests[], summary}
  accuracy         null, or
    settings       {sets, samples, seed, limit, prob_eps, ece_bins}
    sets[]         every set: {name, kind ("classic", "typed" or "di"), title, dataset, summary, items[]}
      classic      dataset {repo, revision, config, split, rows, sha256, source, license}, type,
                   question, state, sample {n, seed, method}
        items[]    {row_idx, label, prediction, correct, sent_at, latency_s, input_tokens, error,
                    probabilities (choice) or noul (noul)}
      typed        dataset (as classic), request {state, questions, payload_sha256}, rows {n, of, partial}
        items[]    {row_idx, id, workflow, payload_sha256, sent_at, latency_s, input_tokens, error,
                    decisions[]: {qid, type, label, prediction, confidence, correct, correct_gold_ties,
                    kl, brier, brier_hard, probabilities, derived, score and score_error (score)}}
      di           dataset {kit {repo, url, commit, edition}, catalog_id, track, source, sha256, rows,
                    kept, excluded, license}, request {state, questions, questions_per_request,
                    options_per_question, prediction, payload_sha256}, rows {n, of, partial}
        items[]    {id, run_id, payload_sha256, status ("ok", "unsupported", "error"), sent_at,
                    latency_s, input_tokens, error, fields, fields_correct, correct,
                    prediction and gold (one-question rows), role_hits (Home appliances)}
memory             null, or {unit "GiB", interval_s, errors, samples: [[unix_time, gib], ...]}
```

Each chat request:

```text
sent_at            Unix time the request was sent
error              null, or a message (HTTP status, stream error, connection failure,
                   a stream that ended without [DONE])
finish_reason      from the last choice that set one
prompt_tokens      usage.prompt_tokens
completion_tokens  usage.completion_tokens (token chunks if the engine sent no usage; usage_missing is then true)
token_chunks       number of token chunks
ttft_s, total_s, decode_s (last - first token chunk)
decode_tps, tpot_ms, prompt_tps
output_bytes, output_chars, output_bytes_per_s, output_chars_per_s
memory_peak_gib    null without --memory-cmd
extras             top-level keys of the stream chunks that are not OpenAI fields, merged,
                   e.g. {"tensorfold": {"drafted": 812, "accepted": 640, "token_sha": "..."}}
chunk_times_s      with --record-chunks: every token chunk's time since sent_at
run                (context sweep) run index; prompt_id and stream (concurrency)
```

Each decision request: `sent_at`, `error` (null, or the HTTP status and body, a
connection failure, or an answer missing or not of its question's type),
`latency_s`, `questions` (count), `input_tokens` and `output_tokens` (from
`usage`), `extras` (top-level response keys that are not System One fields),
`memory_peak_gib`, and `run` (latency) or `stream` and `seq` (throughput).

Each chat `summary` holds, for `ttft_s`, `total_s`, `decode_tps`, `tpot_ms`,
`prompt_tps`, `output_bytes_per_s`, `output_chars_per_s`, `completion_tokens`,
`prompt_tokens` and `memory_peak_gib`, an object `{median, min, max, n}` over
the requests without an error, plus `requests`, `errors`, `ttft_p50_s`,
`ttft_p95_s` and `draft_acceptance` (0 to 1, or null). A concurrency summary
adds `aggregate_tps` and `per_stream_tps` in the same `{median, min, max, n}`
form, and its `memory_peak_gib` is over rounds.

A decisions `summary` holds `latency_s`, `input_tokens` and `memory_peak_gib` as
`{median, min, max, n}`, `latency_p50_s`, `latency_p95_s`,
`per_question_p50_s`, `per_question_p95_s`, `requests` and `errors`; a
throughput level's adds `wall_s`, `requests_per_s`, `questions_per_s` and
`error_rate` (0 to 1). Shares in quality summaries are 0 to 1:

- classic: `items`, `answered`, `errors`, `accuracy`, `macro_f1`, `brier` (noul
  sets, else null), `brier_vector`, `nll`, `nll_eps`, `ece`, `ece_bins`,
  `calibrated` (rows with a distribution), latency and input tokens;
- typed: `cases`, `answered`, `errors`, `coverage`, `decisions`, `accuracy`,
  `accuracy_gold_ties`, `kl`, `kl_eps`, `brier`, `brier_hard`, `ece`,
  `ece_gold_ties`, `ece_bins`, `score_mae`, `score_within_1`,
  `derived_probabilities`, `by_type` and `by_workflow` (each with `decisions`,
  `accuracy`, `accuracy_gold_ties`, `kl`, `brier`, `brier_hard`), latency and
  input tokens;
- di: `requests`, `answered`, `unsupported`, `errors`, `metric`, `score`,
  `field_accuracy`, `case_exact_accuracy`, `coverage`, `raw`, `chance`,
  `skill`, `field_accuracy_by_role` (Home appliances), latency and input tokens.

`ece_bins` lists all 10 bins, `{lo, hi, n, confidence, accuracy}` (null
confidence and accuracy when n is 0).

`report` reads only `label`, `model`, `started_at`, `meta`, `settings`, `mode`,
the `settings` of each section, each point's `label`, `target_prompt_tokens`
and `summary`, and each level's `concurrency` and `summary` (for decisions:
each latency point's `label`, `questions`, `target_state_tokens`,
`state_label` and `summary`, the cache test's `settings` and summaries, each
level's `concurrency` and `summary`, and each quality set without its
`items`). To compare a run made with another tool, write a small adapter that
emits those fields with `schema_version: 3`; the requests and items arrays can
stay empty.

## Tests

```bash
uv run -q --with matplotlib --with pillow python -m unittest discover -s tools/capyctl-bench/tests -t tools/capyctl-bench
```

The tests run `run` and `report` against `tests/fake_server.py`, a local
OpenAI-style streaming server with deterministic pacing. It also answers
`POST /v1/systemone` (deterministic decisions, faster for a state it has seen,
HTTP 400 without `instructions`, as llama.cpp) and serves small fake copies of
every quality source: AG News, BoolQ and Typed Decisions in the shape of the
datasets-server rows API, an API-Bank checkout in the shape of the GitHub tree
API and raw files, and a GPQA Diamond zip. So `run --decisions --accuracy` and
its report are tested end to end; the tests pin the fake copies' hashes. Home
appliances needs no download: the tests build its real rows and check them
against the kit's pin. The scorers are also checked against hand-computed
values and the kit's board fixture. Start the fake server by hand to try the
tool without a GPU: `python3 tests/fake_server.py --port 18443`, then
`run --endpoint http://127.0.0.1:18443/v1 --model sample-model --label "Engine A (sample)" ...`
(with `--decisions` too; there `--accuracy` reads the real pinned sets, and
`--datasets-url` pointed at the fake stops at the pin check, as it should).
The fake server marks every answer as sample data, so `run` sets
`meta.sample: true` and every report and summary rendered from that file
carries a "SAMPLE DATA — not a measurement" banner. `--meta sample=true` sets
the same flag by hand, for example on hand-made fixtures.
