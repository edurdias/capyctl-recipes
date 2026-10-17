# capyctl-bench

The benchmark every recipe and comparison in this repository uses. `run`
measures a model through the CapyCTL inference endpoint and writes one results
file; `report` puts any number of results files (versions, engines, settings)
on the same charts and tables. Chat models are measured with streaming chat
completions; decision models (`api: decisions`, `POST /v1/systemone`) with
`run --decisions`, see [Measure a decision model](#measure-a-decision-model).

`capyctl_bench.py` is one file and needs Python 3.10 or later, standard library
only. The PNG charts of `report` need matplotlib; run `report` through
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
`run --decisions` measures what a decision deployment is used for: how long one
decision takes, how many decisions per second it serves as users increase, and
how often it is right.

```bash
python3 tools/capyctl-bench/capyctl_bench.py run --decisions \
  --api-key-file ~/.local/state/capyctl/identity/credentials \
  --model clef-flash-llamacpp --label "llama.cpp v0.7.0" \
  --concurrency 1,2,4,8 --accuracy \
  --memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'" \
  --meta capyctl=0.1.5 --meta engine=llamacpp-v0.7.0 --meta gpu=GB10 \
  --meta model_revision=<sha> \
  --out results/llamacpp-v0.7.0.json
```

Every request is one `POST {endpoint}/systemone` with the deployment name as
`model`, a `state` and a map of `questions`, sent with the same API key as chat.
It runs, in order:

1. **Calibration.** Two requests with filler states of 20 and 200 sentences fit
   the state tokens per sentence from `usage.input_tokens`. An engine that
   reports no usage gets a fixed estimate of 4 characters per token instead; the
   results say which (`calibration.method`).
2. **Latency sweep.** One request at a time, for every question count
   (`--question-counts`, default 1, 3, 5) and state length (`--state-sweep`,
   default about 200, 2k and 8k tokens), `--runs` requests each (default 10),
   after `--warmup` unrecorded requests. The questions are the first N of a
   fixed set of five mixed ones (choice, noul, score, choice, noul), kept in the
   results.
3. **Concurrency**, with `--concurrency`: N requests start together, each with
   `--concurrency-questions` questions (3) about a `--concurrency-state` state
   (2k), for `--rounds` rounds per stream count after `--warmup` rounds.
4. **Accuracy**, with `--accuracy`: a fixed sample of each built-in public set,
   one question per request (below).

Every state is deterministic filler text with a seed of its own, warm-ups and
every concurrency round included (the seeds continue across stream counts), so
no request reuses another's prefix cache, and two runs send the same states.

### The numbers

| Number | Definition |
|---|---|
| Latency p50, p95, ms | Client side: request written to the whole answer read. Linear-interpolation percentiles over the runs at a point, or over every request at a stream count. |
| Input tokens | `usage.input_tokens`: state, questions and the engine's template. |
| Decisions per second | Concurrency: a round's answered requests / (first request sent to last answer read). Median over rounds. One decision is one request, whatever its question count. |
| Accuracy, % | Share of the answered rows of a set the model got right. A noul counts as yes at a probability of 0.5 or more. |
| Brier score | Noul sets: mean of (probability of yes − truth)². 0 is perfect; always answering 0.5 scores 0.25. It rewards calibrated probabilities, not only the right side of 0.5. |
| Peak memory, GiB | The highest `--memory-cmd` sample while the request, round or set ran. A recipe quotes CapyCTL's own `startup.measured` peak instead. |

A response counts only when every question has an answer of its type (a
`choice` among the options, a number for `noul` and `score`); anything else, an
HTTP error included, is an error with the engine's message.

### Accuracy sets

| Set | Repository, split, revision | Question | Rows |
|---|---|---|---|
| AG News | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) test at `eb185aade064a813bc0b7f42de02595523103ca4` | choice: "Which section of a news site does this article belong to?" with World, Sports, Business and Sci/Tech, each described; the state is the article | 7,600 |
| BoolQ | [`google/boolq`](https://huggingface.co/datasets/google/boolq) validation at `35b264d03638db9f4ce671b711558bf7ff0f80d5` | noul: the row's question, capitalized, with "?"; the state is the passage | 3,270 |

They are small public sets that map one to one onto the two common question
types: a 4-way topic choice with balanced classes, and a yes/no question that
needs the state to answer. Both are the canonical copies on Hugging Face.

The rows come from the Hugging Face
[datasets-server](https://huggingface.co/docs/dataset-viewer) JSON rows API,
so no parquet reader is needed. That API serves a repository's current revision
only, so each set is pinned by the revision it was read at and the sha256 of
its rows (canonical JSON, one line per row, in row order): if the split changes,
the run stops instead of measuring other rows. The first `--accuracy` run
downloads both splits (a few minutes: the API rate-limits after a few dozen
pages and the tool waits it out) into `--dataset-cache`
(`~/.cache/capyctl-bench/datasets`); later runs read the cache.

The sample is `--accuracy-samples` rows per set (500), the rows with the
smallest sha256 of `seed:set:row_idx` for `--accuracy-seed` (0), the same on
every machine and Python. The results keep every sampled row's index, truth,
prediction, probabilities (or noul), latency and input tokens.

`report` draws latency against state length for each question count (p50
solid, p95 dashed), decisions per second and latency against streams, and an
accuracy table; the summary shows decisions per second at the fewest and most
streams (latency p50 without a concurrency sweep), latency at the most
questions, decisions per second against streams and accuracy, its changes in
percentage points. Reports compare decisions results with decisions results
only.

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
| `--runs` | 3 (10 with `--decisions`) | Requests per context or latency point. |
| `--max-tokens` | 128 | `max_tokens` of each context sweep request. |
| `--concurrency` | | Stream counts, `1-8` or `1,2,4,8`. |
| `--rounds` | 5 | Rounds per stream count. |
| `--concurrency-max-tokens` | 512 | `max_tokens` of each concurrency request. |
| `--prompts` | `prompts.json` | Prompt set for concurrency (8 prompts that ask for long answers). Every round, warm-up included, takes the next prompts of the set across all stream counts; the set repeats only when it runs out, and the tool says so (a repeated long prompt can hit the engine's prefix cache). |
| `--warmup` | 1 | Unrecorded requests (sweep) or rounds (each stream count) first. |
| `--temperature` | 0 | Sampling temperature. |
| `--extra-body` | | JSON object merged into every request body, for example `'{"ignore_eos": true}'` or `'{"chat_template_kwargs": {"enable_thinking": false}}'`. |
| `--memory-cmd` | | Shell command whose output's last number is the memory in use, in GiB. Sampled every `--memory-interval` seconds (0.5). |
| `--record-chunks` | off | Keep the arrival time of every token chunk, for replays. |
| `--meta KEY=VALUE` | | Run metadata (CapyCTL version, engine version, GPU, commit). Repeatable. |
| `--label` | | The series name in reports. |

With `--decisions` only (a chat run refuses them, and `--decisions` refuses
`--context-sweep` and `--record-chunks`):

| Option | Default | What |
|---|---|---|
| `--question-counts` | `1,3,5` | Questions per latency request, 1 to 5. |
| `--state-sweep` | `0.2k,2k,8k` | State lengths of the latency sweep, in tokens. |
| `--concurrency-questions` | 3 | Questions per concurrency request. |
| `--concurrency-state` | `2k` | State length of each concurrency request. |
| `--accuracy` | off | Also measure accuracy on the built-in sets. |
| `--accuracy-sets` | `ag_news,boolq` | Which of them. |
| `--accuracy-samples` | 500 | Rows sampled per set. |
| `--accuracy-seed` | 0 | Seed of the sample. |
| `--dataset-cache` | `~/.cache/capyctl-bench/datasets` | Where downloaded sets are kept. |
| `--datasets-url` | `https://datasets-server.huggingface.co` | The datasets-server to read rows from (a mirror; the tests use the fake server). |

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

## Results schema (version 2)

One JSON object per run. Times are seconds unless the name says otherwise;
`*_at` fields are ISO 8601 UTC except `sent_at`, which is a Unix timestamp.
Version 2 adds `mode` and the `decisions` section; a version 1 file is a chat
run from before `mode` existed, and `report` still reads it.

```text
schema_version     2
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
  calibration      {method ("usage.input_tokens" or "4 characters per token"), per_sentence_tokens,
                    overhead_tokens, probes: [{sentences, input_tokens}]}
  latency
    settings       {question_counts, state_targets, runs, warmup, unit, questions: {id: question}}
    points[]       {label "3q 2k", questions, target_state_tokens, state_label, sentences, requests[], summary}
  concurrency      null, or
    settings       {levels, rounds, warmup, questions, state_tokens, sentences}
    levels[]       {concurrency, rounds[], summary}
      rounds[]     {round, wall_s, decisions, decisions_per_s, errors, memory_peak_gib, requests[]}
  accuracy         null, or
    settings       {sets, samples, seed}
    sets[]         {name, title, type, dataset {repo, revision, config, split, rows, sha256, source},
                    question, state, sample {n, seed, method}, summary, items[]}
      items[]      {row_idx, label, prediction, correct, sent_at, latency_s, input_tokens, error,
                    probabilities (choice) or noul (noul)}
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
`memory_peak_gib`, and `run` (latency) or `stream` (concurrency).

Each chat `summary` holds, for `ttft_s`, `total_s`, `decode_tps`, `tpot_ms`,
`prompt_tps`, `output_bytes_per_s`, `output_chars_per_s`, `completion_tokens`,
`prompt_tokens` and `memory_peak_gib`, an object `{median, min, max, n}` over
the requests without an error, plus `requests`, `errors`, `ttft_p50_s`,
`ttft_p95_s` and `draft_acceptance` (0 to 1, or null). A concurrency summary
adds `aggregate_tps` and `per_stream_tps` in the same `{median, min, max, n}`
form, and its `memory_peak_gib` is over rounds.

A decisions `summary` holds `latency_s`, `input_tokens` and `memory_peak_gib` as
`{median, min, max, n}`, `latency_p50_s`, `latency_p95_s`, `requests` and
`errors`; a concurrency level's adds `decisions_per_s` (over rounds). An
accuracy set's holds `items`, `answered`, `errors`, `accuracy` (0 to 1),
`brier` (noul sets, else null), `latency_s`, `input_tokens` and
`memory_peak_gib`.

`report` reads only `label`, `model`, `started_at`, `meta`, `settings`, `mode`,
the `settings` of each section, each point's `label`, `target_prompt_tokens`
and `summary`, and each level's `concurrency` and `summary` (for decisions:
each latency point's `label`, `questions`, `target_state_tokens`,
`state_label` and `summary`, each level's `concurrency` and `summary`, and each
accuracy set without its `items`). To compare a run made with another tool,
write a small adapter that emits those fields with `schema_version: 2`; the
requests and items arrays can stay empty.

## Tests

```bash
cd tools/capyctl-bench && python3 -m unittest discover -s tests -v
```

The tests run `run` and `report` against `tests/fake_server.py`, a local
OpenAI-style streaming server with deterministic pacing. It also answers
`POST /v1/systemone` (deterministic decisions, HTTP 400 without
`instructions`, as llama.cpp) and serves small fake copies of the accuracy sets
in the shape of the datasets-server rows API, so `run --decisions --accuracy`
and its report are tested end to end; the tests pin the fake copies' hashes.
Start it by hand to try the tool without a GPU:
`python3 tests/fake_server.py --port 18443`, then
`run --endpoint http://127.0.0.1:18443/v1 --model sample-model --label "Engine A (sample)" ...`
(with `--decisions` too; there `--accuracy` reads the real pinned sets, and
`--datasets-url` pointed at the fake stops at the pin check, as it should).
The fake server marks every answer as sample data, so `run` sets
`meta.sample: true` and every report and summary rendered from that file
carries a "SAMPLE DATA — not a measurement" banner. `--meta sample=true` sets
the same flag by hand, for example on hand-made fixtures.
