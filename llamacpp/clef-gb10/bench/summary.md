# Clef on llama.cpp v0.7.0

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with Clef · llama.cpp v0.7.0, differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Clef · llama.cpp v0.7.0 | `clef-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=91 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 560 |
| 1 question, 2k-token state | 2,202 |
| 1 question, 8k-token state | 12,176 |
| 3 questions, 200-token state | 693 |
| 3 questions, 2k-token state | 2,356 |
| 3 questions, 8k-token state | 12,414 |
| 5 questions, 200-token state | 751 |
| 5 questions, 2k-token state | 2,567 |
| 5 questions, 8k-token state | 12,672 |

### Latency p95 (ms, lower is better)

| Point | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 567 |
| 1 question, 2k-token state | 2,233 |
| 1 question, 8k-token state | 12,387 |
| 3 questions, 200-token state | 734 |
| 3 questions, 2k-token state | 2,387 |
| 3 questions, 8k-token state | 12,627 |
| 5 questions, 200-token state | 761 |
| 5 questions, 2k-token state | 2,657 |
| 5 questions, 8k-token state | 12,926 |

### Per answered question, p50 (ms, lower is better)

| Point | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 560 |
| 1 question, 2k-token state | 2,202 |
| 1 question, 8k-token state | 12,176 |
| 3 questions, 200-token state | 231 |
| 3 questions, 2k-token state | 785 |
| 3 questions, 8k-token state | 4,138 |
| 5 questions, 200-token state | 150 |
| 5 questions, 2k-token state | 513 |
| 5 questions, 8k-token state | 2,534 |

### Input tokens (tok)

| Point | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 389 |
| 1 question, 2k-token state | 2,201 |
| 1 question, 8k-token state | 8,250 |
| 3 questions, 200-token state | 560 |
| 3 questions, 2k-token state | 2,383 |
| 3 questions, 8k-token state | 8,428 |
| 5 questions, 200-token state | 732 |
| 5 questions, 2k-token state | 2,554 |
| 5 questions, 8k-token state | 8,604 |

### Peak memory (GiB, lower is better)

| Point | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 39.0 |
| 1 question, 2k-token state | 39.0 |
| 1 question, 8k-token state | 39.2 |
| 3 questions, 200-token state | 39.3 |
| 3 questions, 2k-token state | 39.3 |
| 3 questions, 8k-token state | 39.3 |
| 5 questions, 200-token state | 39.3 |
| 5 questions, 2k-token state | 39.3 |
| 5 questions, 8k-token state | 39.3 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Clef · llama.cpp v0.7.0 |
|---|---|
| Cold p50 | 2,353 |
| Cold p95 | 2,399 |
| Repeated p50 | 2,322 |
| Repeated p95 | 2,330 |
| Repeated p50 / cold p50 | 0.987 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 | 0.425 |
| 2 | 0.430 |
| 4 | 0.430 |
| 8 | 0.430 |

### Answered questions per second (questions/s, higher is better)

| Streams | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 | 1.28 |
| 2 | 1.29 |
| 4 | 1.29 |
| 8 | 1.29 |

### Latency p50 (ms, lower is better)

| Streams | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 | 2,348 |
| 2 | 4,640 |
| 4 | 4,649 |
| 8 | 11,618 |

### Latency p95 (ms, lower is better)

| Streams | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 | 2,392 |
| 2 | 4,786 |
| 4 | 4,769 |
| 8 | 11,788 |

### Error rate (%, lower is better)

| Streams | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Clef · llama.cpp v0.7.0 |
|---|---|
| 1 | 39.3 |
| 2 | 39.3 |
| 4 | 39.4 |
| 8 | 39.3 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Clef · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 72.1% |
| Accuracy, gold ties (%, higher is better) | 72.7% |
| Accuracy, noul (%, higher is better) | 84.5% |
| Accuracy, choice (%, higher is better) | 63.8% |
| Accuracy, score (%, higher is better) | 69.0% |
| KL from gold (lower is better) | 0.191 |
| Brier, soft gold (lower is better) | 0.103 |
| Brier, hard label (lower is better) | 0.370 |
| ECE (lower is better) | 0.020 |
| ECE, gold ties (lower is better) | 0.024 |
| Score MAE (levels, lower is better) | 0.269 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Clef · llama.cpp v0.7.0 |
|---|---|
| agent_trace_observability | 62.6% |
| customer_service | 74.2% |
| invoice_processing | 77.2% |
| security_incidents | 74.4% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | 0.0% at 28.8% (n 1) |
| 0.3–0.4 | 36.2% at 37.2% (n 47) |
| 0.4–0.5 | 47.7% at 45.9% (n 277) |
| 0.5–0.6 | 57.4% at 55.0% (n 373) |
| 0.6–0.7 | 69.0% at 65.1% (n 300) |
| 0.7–0.8 | 77.0% at 75.0% (n 305) |
| 0.8–0.9 | 84.7% at 85.0% (n 294) |
| 0.9–1.0 | 96.3% at 94.9% (n 403) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Clef · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 91.9% |
| GPQA Diamond | 46.4% |
| Home appliances | 81.8% |

### Coverage (%, higher is better)

| Component | Clef · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Clef · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 91.8% |
| GPQA Diamond | 28.6% |
| Home appliances | 81.8% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Clef · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 90.0% |
| Macro-F1 (%, higher is better) | 90.1% |
| Brier, vector (lower is better) | 0.168 |
| NLL (lower is better) | 0.340 |
| ECE (lower is better) | 0.041 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 0.0% at 32.8% (n 1) |
| 0.4–0.5 | 50.0% at 47.0% (n 4) |
| 0.5–0.6 | 73.3% at 55.2% (n 15) |
| 0.6–0.7 | 68.2% at 65.2% (n 22) |
| 0.7–0.8 | 89.5% at 76.1% (n 19) |
| 0.8–0.9 | 74.2% at 86.0% (n 62) |
| 0.9–1.0 | 95.2% at 97.0% (n 377) |

### BoolQ

| Metric | Clef · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 92.6% |
| Macro-F1 (%, higher is better) | 92.2% |
| Brier, vector (lower is better) | 0.122 |
| Brier, binary (lower is better) | 0.061 |
| NLL (lower is better) | 0.221 |
| ECE (lower is better) | 0.018 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 62.5% at 55.5% (n 8) |
| 0.6–0.7 | 70.6% at 66.5% (n 17) |
| 0.7–0.8 | 73.7% at 75.1% (n 19) |
| 0.8–0.9 | 76.7% at 85.6% (n 30) |
| 0.9–1.0 | 96.0% at 97.1% (n 426) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
