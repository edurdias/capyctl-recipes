# capyctl-bench

The benchmark every recipe and comparison in this repository uses. `run`
measures a model through the CapyCTL inference endpoint and writes one results
file; `report` puts any number of results files (versions, engines, settings)
on the same charts and tables.

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

## Compare two results

```bash
uv run --with matplotlib python3 tools/capyctl-bench/capyctl_bench.py report \
  results/tensorfold-0.6.2.json results/vllm-0.30.0.json \
  --out report/ --title "Qwen3.8-27B NVFP4 on one GB10" --summary
```

The pages open in the light theme; the sun/moon button switches to dark and
the browser remembers the choice. Print and PDF are always light.

The summary shows 2 to 4 panels, picked from what the results hold: generation
tok/s as bars at about 1k tokens and at the longest context every series ran,
prompt processing and time to first token against context (log scale when it
spans a wide range), and aggregate tok/s against streams when there is a
concurrency sweep. With one series it shows the values; with several, the bars
also carry the change against the first series. The subtitle takes the
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
| `--context-sweep` | | Prompt sizes, `k` = 1000 tokens. One request at a time. |
| `--runs` | 3 | Requests per context point. |
| `--max-tokens` | 128 | `max_tokens` of each context sweep request. |
| `--concurrency` | | Stream counts, `1-8` or `1,2,4,8`. |
| `--rounds` | 5 | Rounds per stream count. |
| `--concurrency-max-tokens` | 512 | `max_tokens` of each concurrency request. |
| `--prompts` | `prompts.json` | Prompt set for concurrency (8 prompts that ask for long answers). |
| `--warmup` | 1 | Unrecorded requests (sweep) or rounds (each stream count) first. |
| `--temperature` | 0 | Sampling temperature. |
| `--extra-body` | | JSON object merged into every request body, for example `'{"ignore_eos": true}'` or `'{"chat_template_kwargs": {"enable_thinking": false}}'`. |
| `--memory-cmd` | | Shell command whose output's last number is the memory in use, in GiB. Sampled every `--memory-interval` seconds (0.5). |
| `--record-chunks` | off | Keep the arrival time of every token chunk, for replays. |
| `--meta KEY=VALUE` | | Run metadata (CapyCTL version, engine version, GPU, commit). Repeatable. |
| `--label` | | The series name in reports. |

Memory commands that work:

```bash
# Unified memory (GB10) or any host: RAM in use on this machine
--memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'"
# A discrete GPU on another machine
--memory-cmd "ssh gpu-box nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += \$1} END {print s / 1024}'"
```

## What each metric means

Every request is a streaming chat completion with
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

- every series ran on the same host, with the same prompts and settings
  (`report` names any setting that differs between series);
- the run-to-run spread is stated: tables show the median with the min-max
  range under it, and a recipe README quotes the range next to the median;
- the numbers were measured through CapyCTL, not against the engine directly,
  and the report says so;
- the CapyCTL version, engine version, model revision and drafter revision are
  in `--meta`, and nothing else ran on the machine during the run.

## Results schema (version 1)

One JSON object per run. Times are seconds unless the name says otherwise;
`*_at` fields are ISO 8601 UTC except `sent_at`, which is a Unix timestamp.

```text
schema_version     1
tool               {name, version, python}
label              series name, e.g. "TensorFold 0.6.2"
model              model name sent in each request
endpoint           scheme, port and path only: "http://<host>:8443/v1"
started_at, finished_at
meta               {key: value} from --meta; sample: true marks sample data (banner in reports)
settings           {temperature, extra_body, memory_command (bool), memory_interval_s, record_chunks}
context_sweep      null, or
  settings         {targets, runs, max_tokens, warmup, unit}
  calibration      {per_sentence_tokens, overhead_tokens, probes: [{sentences, prompt_tokens}]}
  points[]         {label "4k", target_prompt_tokens, sentences, requests[], summary}
concurrency        null, or
  settings         {levels, rounds, warmup, max_tokens, prompt_set, prompts: [ids]}
  levels[]         {concurrency, rounds[], summary}
    rounds[]       {round, wall_s, completion_tokens, aggregate_tps, errors, memory_peak_gib, requests[]}
memory             null, or {unit "GiB", interval_s, errors, samples: [[unix_time, gib], ...]}
```

Each request:

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

Each `summary` holds, for `ttft_s`, `total_s`, `decode_tps`, `tpot_ms`,
`prompt_tps`, `output_bytes_per_s`, `output_chars_per_s`, `completion_tokens`,
`prompt_tokens` and `memory_peak_gib`, an object `{median, min, max, n}` over
the requests without an error, plus `requests`, `errors`, `ttft_p50_s`,
`ttft_p95_s` and `draft_acceptance` (0 to 1, or null). A concurrency summary
adds `aggregate_tps` and `per_stream_tps` in the same `{median, min, max, n}`
form, and its `memory_peak_gib` is over rounds.

`report` reads only `label`, `model`, `started_at`, `meta`, `settings`, the
`settings` of each section, each point's `label`, `target_prompt_tokens` and
`summary`, and each level's `concurrency` and `summary`. To compare a run made
with another tool, write a small adapter that emits those fields with
`schema_version: 1`; the requests arrays can stay empty.

## Tests

```bash
cd tools/capyctl-bench && python3 -m unittest discover -s tests -v
```

The tests run `run` and `report` against `tests/fake_server.py`, a local
OpenAI-style streaming server with deterministic pacing. Start it by hand to
try the tool without a GPU: `python3 tests/fake_server.py --port 18443`, then
`run --endpoint http://127.0.0.1:18443/v1 --model sample-model --label "Engine A (sample)" ...`.
The fake server marks every answer as sample data, so `run` sets
`meta.sample: true` and every report and summary rendered from that file
carries a "SAMPLE DATA — not a measurement" banner. `--meta sample=true` sets
the same flag by hand, for example on hand-made fixtures.
