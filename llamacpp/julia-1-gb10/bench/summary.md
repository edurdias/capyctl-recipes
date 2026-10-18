# Julia-1 on llama.cpp v0.7.0

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with Julia-1 · llama.cpp v0.7.0, differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Julia-1 · llama.cpp v0.7.0 | `julia1-lc` | 2026-10-18 | capyctl=pr148-a33c4f0d, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=4 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 20.0 |
| 1 question, 2k-token state | 56.2 |
| 1 question, 8k-token state | 392 |
| 3 questions, 200-token state | 26.3 |
| 3 questions, 2k-token state | 229 |
| 3 questions, 8k-token state | 1,119 |
| 5 questions, 200-token state | 35.6 |
| 5 questions, 2k-token state | 337 |
| 5 questions, 8k-token state | 1,822 |

### Latency p95 (ms, lower is better)

| Point | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 23.8 |
| 1 question, 2k-token state | 59.8 |
| 1 question, 8k-token state | 424 |
| 3 questions, 200-token state | 30.0 |
| 3 questions, 2k-token state | 237 |
| 3 questions, 8k-token state | 1,220 |
| 5 questions, 200-token state | 38.8 |
| 5 questions, 2k-token state | 375 |
| 5 questions, 8k-token state | 1,996 |

### Per answered question, p50 (ms, lower is better)

| Point | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 20.0 |
| 1 question, 2k-token state | 56.2 |
| 1 question, 8k-token state | 392 |
| 3 questions, 200-token state | 8.78 |
| 3 questions, 2k-token state | 76.3 |
| 3 questions, 8k-token state | 373 |
| 5 questions, 200-token state | 7.11 |
| 5 questions, 2k-token state | 67.3 |
| 5 questions, 8k-token state | 364 |

### Input tokens (tok)

| Point | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 259 |
| 1 question, 2k-token state | 2,064 |
| 1 question, 8k-token state | 8,106 |
| 3 questions, 200-token state | 733 |
| 3 questions, 2k-token state | 6,178 |
| 3 questions, 8k-token state | 24,304 |
| 5 questions, 200-token state | 1,218 |
| 5 questions, 2k-token state | 10,284 |
| 5 questions, 8k-token state | 40,498 |

### Peak memory (GiB, lower is better)

| Point | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 5.11 |
| 1 question, 2k-token state | 5.11 |
| 1 question, 8k-token state | 5.13 |
| 3 questions, 200-token state | 5.13 |
| 3 questions, 2k-token state | 5.13 |
| 3 questions, 8k-token state | 5.13 |
| 5 questions, 200-token state | 5.11 |
| 5 questions, 2k-token state | 5.11 |
| 5 questions, 8k-token state | 5.12 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| Cold p50 | 221 |
| Cold p95 | 238 |
| Repeated p50 | 211 |
| Repeated p95 | 217 |
| Repeated p50 / cold p50 | 0.953 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 | 4.43 |
| 2 | 5.31 |
| 4 | 4.90 |
| 8 | 4.95 |

### Answered questions per second (questions/s, higher is better)

| Streams | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 | 13.3 |
| 2 | 15.9 |
| 4 | 14.7 |
| 8 | 14.9 |

### Latency p50 (ms, lower is better)

| Streams | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 | 228 |
| 2 | 377 |
| 4 | 768 |
| 8 | 1,456 |

### Latency p95 (ms, lower is better)

| Streams | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 | 234 |
| 2 | 388 |
| 4 | 1,249 |
| 8 | 2,324 |

### Error rate (%, lower is better)

| Streams | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 1 | 5.15 |
| 2 | 5.14 |
| 4 | 5.14 |
| 8 | 5.14 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 73.0% |
| Accuracy, gold ties (%, higher is better) | 73.8% |
| Accuracy, noul (%, higher is better) | 80.8% |
| Accuracy, choice (%, higher is better) | 72.5% |
| Accuracy, score (%, higher is better) | 67.6% |
| KL from gold (lower is better) | 2.96 |
| Brier, soft gold (lower is better) | 0.333 |
| Brier, hard label (lower is better) | 0.491 |
| ECE (lower is better) | 0.224 |
| ECE, gold ties (lower is better) | 0.217 |
| Score MAE (levels, lower is better) | 0.428 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| agent_trace_observability | 70.2% |
| customer_service | 70.6% |
| invoice_processing | 80.6% |
| security_incidents | 70.8% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 0.0% at 33.2% (n 1) |
| 0.4–0.5 | 16.7% at 46.5% (n 12) |
| 0.5–0.6 | 50.0% at 55.2% (n 60) |
| 0.6–0.7 | 54.3% at 64.5% (n 46) |
| 0.7–0.8 | 59.7% at 75.4% (n 67) |
| 0.8–0.9 | 56.6% at 85.3% (n 99) |
| 0.9–1.0 | 76.3% at 99.4% (n 1715) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 2.0% |
| GPQA Diamond | 25.5% |
| Home appliances | 0.0% |

### Coverage (%, higher is better)

| Component | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 0.1% |
| GPQA Diamond | 0.7% |
| Home appliances | 0.0% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 71.2% |
| Macro-F1 (%, higher is better) | 69.1% |
| Brier, vector (lower is better) | 0.465 |
| NLL (lower is better) | 1.19 |
| ECE (lower is better) | 0.160 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 0.0% at 38.4% (n 2) |
| 0.4–0.5 | 41.7% at 46.9% (n 12) |
| 0.5–0.6 | 56.8% at 53.8% (n 37) |
| 0.6–0.7 | 50.0% at 65.1% (n 36) |
| 0.7–0.8 | 46.8% at 75.4% (n 47) |
| 0.8–0.9 | 62.3% at 85.3% (n 61) |
| 0.9–1.0 | 82.6% at 97.3% (n 305) |

### BoolQ

| Metric | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 63.2% |
| Macro-F1 (%, higher is better) | 49.8% |
| Brier, vector (lower is better) | 0.637 |
| Brier, binary (lower is better) | 0.318 |
| NLL (lower is better) | 1.65 |
| ECE (lower is better) | 0.299 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Julia-1 · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 51.9% at 54.5% (n 27) |
| 0.6–0.7 | 33.3% at 66.9% (n 12) |
| 0.7–0.8 | 50.0% at 76.5% (n 30) |
| 0.8–0.9 | 57.1% at 85.8% (n 42) |
| 0.9–1.0 | 66.6% at 98.6% (n 389) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
