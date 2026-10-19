# Clef-Flash on llama.cpp v0.7.0

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with Clef-Flash · llama.cpp v0.7.0, differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Clef-Flash · llama.cpp v0.7.0 | `clefflash-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=23 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 203 |
| 1 question, 2k-token state | 718 |
| 1 question, 8k-token state | 3,813 |
| 3 questions, 200-token state | 239 |
| 3 questions, 2k-token state | 769 |
| 3 questions, 8k-token state | 3,903 |
| 5 questions, 200-token state | 256 |
| 5 questions, 2k-token state | 839 |
| 5 questions, 8k-token state | 3,934 |

### Latency p95 (ms, lower is better)

| Point | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 207 |
| 1 question, 2k-token state | 729 |
| 1 question, 8k-token state | 3,874 |
| 3 questions, 200-token state | 252 |
| 3 questions, 2k-token state | 780 |
| 3 questions, 8k-token state | 3,937 |
| 5 questions, 200-token state | 259 |
| 5 questions, 2k-token state | 860 |
| 5 questions, 8k-token state | 4,046 |

### Per answered question, p50 (ms, lower is better)

| Point | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 203 |
| 1 question, 2k-token state | 718 |
| 1 question, 8k-token state | 3,813 |
| 3 questions, 200-token state | 79.7 |
| 3 questions, 2k-token state | 256 |
| 3 questions, 8k-token state | 1,301 |
| 5 questions, 200-token state | 51.1 |
| 5 questions, 2k-token state | 168 |
| 5 questions, 8k-token state | 787 |

### Input tokens (tok)

| Point | Clef-Flash · llama.cpp v0.7.0 |
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

| Point | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 19.5 |
| 1 question, 2k-token state | 19.5 |
| 1 question, 8k-token state | 19.6 |
| 3 questions, 200-token state | 19.7 |
| 3 questions, 2k-token state | 19.7 |
| 3 questions, 8k-token state | 19.7 |
| 5 questions, 200-token state | 19.7 |
| 5 questions, 2k-token state | 19.7 |
| 5 questions, 8k-token state | 19.7 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| Cold p50 | 766 |
| Cold p95 | 787 |
| Repeated p50 | 757 |
| Repeated p95 | 762 |
| Repeated p50 / cold p50 | 0.988 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 | 1.30 |
| 2 | 1.36 |
| 4 | 1.36 |
| 8 | 1.35 |

### Answered questions per second (questions/s, higher is better)

| Streams | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 | 3.91 |
| 2 | 4.07 |
| 4 | 4.07 |
| 8 | 4.06 |

### Latency p50 (ms, lower is better)

| Streams | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 | 766 |
| 2 | 1,473 |
| 4 | 1,474 |
| 8 | 3,687 |

### Latency p95 (ms, lower is better)

| Streams | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 | 781 |
| 2 | 1,515 |
| 4 | 1,509 |
| 8 | 3,739 |

### Error rate (%, lower is better)

| Streams | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 1 | 19.8 |
| 2 | 19.8 |
| 4 | 19.7 |
| 8 | 19.7 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 70.7% |
| Accuracy, gold ties (%, higher is better) | 71.3% |
| Accuracy, noul (%, higher is better) | 82.0% |
| Accuracy, choice (%, higher is better) | 70.7% |
| Accuracy, score (%, higher is better) | 62.1% |
| KL from gold (lower is better) | 0.209 |
| Brier, soft gold (lower is better) | 0.110 |
| Brier, hard label (lower is better) | 0.391 |
| ECE (lower is better) | 0.013 |
| ECE, gold ties (lower is better) | 0.013 |
| Score MAE (levels, lower is better) | 0.296 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| agent_trace_observability | 61.4% |
| customer_service | 76.2% |
| invoice_processing | 75.4% |
| security_incidents | 69.6% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | 0.0% at 29.6% (n 1) |
| 0.3–0.4 | 37.1% at 36.9% (n 70) |
| 0.4–0.5 | 49.2% at 45.8% (n 258) |
| 0.5–0.6 | 53.0% at 54.8% (n 351) |
| 0.6–0.7 | 64.1% at 64.9% (n 315) |
| 0.7–0.8 | 76.7% at 75.1% (n 275) |
| 0.8–0.9 | 85.8% at 85.2% (n 360) |
| 0.9–1.0 | 95.1% at 94.8% (n 370) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 92.9% |
| GPQA Diamond | 51.5% |
| Home appliances | 96.6% |

### Coverage (%, higher is better)

| Component | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 92.8% |
| GPQA Diamond | 35.4% |
| Home appliances | 96.6% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 89.6% |
| Macro-F1 (%, higher is better) | 89.7% |
| Brier, vector (lower is better) | 0.165 |
| NLL (lower is better) | 0.344 |
| ECE (lower is better) | 0.038 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | 33.3% at 46.6% (n 6) |
| 0.5–0.6 | 25.0% at 53.3% (n 8) |
| 0.6–0.7 | 20.0% at 62.2% (n 5) |
| 0.7–0.8 | 78.6% at 75.1% (n 14) |
| 0.8–0.9 | 79.6% at 86.3% (n 49) |
| 0.9–1.0 | 94.0% at 96.4% (n 418) |

### BoolQ

| Metric | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 89.6% |
| Macro-F1 (%, higher is better) | 89.2% |
| Brier, vector (lower is better) | 0.151 |
| Brier, binary (lower is better) | 0.075 |
| NLL (lower is better) | 0.266 |
| ECE (lower is better) | 0.031 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef-Flash · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 22.2% at 55.8% (n 9) |
| 0.6–0.7 | 45.5% at 64.8% (n 11) |
| 0.7–0.8 | 45.5% at 75.1% (n 22) |
| 0.8–0.9 | 81.4% at 85.7% (n 43) |
| 0.9–1.0 | 95.4% at 96.0% (n 415) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
