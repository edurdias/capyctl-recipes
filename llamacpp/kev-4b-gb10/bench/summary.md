# Kev-4B on llama.cpp v0.7.0

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with Kev-4B · llama.cpp v0.7.0, differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Kev-4B · llama.cpp v0.7.0 | `kev4b-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=14 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 88.1 |
| 1 question, 2k-token state | 551 |
| 1 question, 8k-token state | 2,604 |
| 3 questions, 200-token state | 205 |
| 3 questions, 2k-token state | 661 |
| 3 questions, 8k-token state | 2,739 |
| 5 questions, 200-token state | 316 |
| 5 questions, 2k-token state | 1,223 |
| 5 questions, 8k-token state | 5,322 |

### Latency p95 (ms, lower is better)

| Point | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 90.7 |
| 1 question, 2k-token state | 560 |
| 1 question, 8k-token state | 2,679 |
| 3 questions, 200-token state | 210 |
| 3 questions, 2k-token state | 681 |
| 3 questions, 8k-token state | 2,787 |
| 5 questions, 200-token state | 322 |
| 5 questions, 2k-token state | 1,248 |
| 5 questions, 8k-token state | 5,432 |

### Per answered question, p50 (ms, lower is better)

| Point | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 88.1 |
| 1 question, 2k-token state | 551 |
| 1 question, 8k-token state | 2,604 |
| 3 questions, 200-token state | 68.2 |
| 3 questions, 2k-token state | 220 |
| 3 questions, 8k-token state | 913 |
| 5 questions, 200-token state | 63.1 |
| 5 questions, 2k-token state | 245 |
| 5 questions, 8k-token state | 1,064 |

### Input tokens (tok)

| Point | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 258 |
| 1 question, 2k-token state | 2,070 |
| 1 question, 8k-token state | 8,119 |
| 3 questions, 200-token state | 710 |
| 3 questions, 2k-token state | 6,180 |
| 3 questions, 8k-token state | 24,316 |
| 5 questions, 200-token state | 1,176 |
| 5 questions, 2k-token state | 10,280 |
| 5 questions, 8k-token state | 40,530 |

### Peak memory (GiB, lower is better)

| Point | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 question, 200-token state | 34.3 |
| 1 question, 2k-token state | 35.1 |
| 1 question, 8k-token state | 40.2 |
| 3 questions, 200-token state | 37.3 |
| 3 questions, 2k-token state | 37.3 |
| 3 questions, 8k-token state | 37.3 |
| 5 questions, 200-token state | 37.3 |
| 5 questions, 2k-token state | 37.3 |
| 5 questions, 8k-token state | 37.5 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| Cold p50 | 658 |
| Cold p95 | 676 |
| Repeated p50 | 649 |
| Repeated p95 | 657 |
| Repeated p50 / cold p50 | 0.987 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 | 1.51 |
| 2 | 1.60 |
| 4 | 1.61 |
| 8 | 1.60 |

### Answered questions per second (questions/s, higher is better)

| Streams | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 | 4.54 |
| 2 | 4.81 |
| 4 | 4.82 |
| 8 | 4.79 |

### Latency p50 (ms, lower is better)

| Streams | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 | 658 |
| 2 | 1,246 |
| 4 | 1,841 |
| 8 | 5,009 |

### Latency p95 (ms, lower is better)

| Streams | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 | 681 |
| 2 | 1,765 |
| 4 | 3,742 |
| 8 | 5,058 |

### Error rate (%, lower is better)

| Streams | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 1 | 37.5 |
| 2 | 37.5 |
| 4 | 37.5 |
| 8 | 37.5 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 66.7% |
| Accuracy, gold ties (%, higher is better) | 67.3% |
| Accuracy, noul (%, higher is better) | 75.8% |
| Accuracy, choice (%, higher is better) | 64.5% |
| Accuracy, score (%, higher is better) | 61.5% |
| KL from gold (lower is better) | 0.223 |
| Brier, soft gold (lower is better) | 0.125 |
| Brier, hard label (lower is better) | 0.465 |
| ECE (lower is better) | 0.092 |
| ECE, gold ties (lower is better) | 0.098 |
| Score MAE (levels, lower is better) | 0.397 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| agent_trace_observability | 56.8% |
| customer_service | 71.6% |
| invoice_processing | 68.2% |
| security_incidents | 70.2% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | 28.6% at 29.2% (n 7) |
| 0.3–0.4 | 41.8% at 36.2% (n 297) |
| 0.4–0.5 | 55.9% at 44.8% (n 417) |
| 0.5–0.6 | 64.8% at 54.7% (n 491) |
| 0.6–0.7 | 79.6% at 64.8% (n 377) |
| 0.7–0.8 | 80.7% at 74.2% (n 171) |
| 0.8–0.9 | 84.4% at 84.8% (n 122) |
| 0.9–1.0 | 98.3% at 95.4% (n 118) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 65.9% |
| GPQA Diamond | 39.3% |
| Home appliances | 14.8% |

### Coverage (%, higher is better)

| Component | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| API-Bank | 65.3% |
| GPQA Diamond | 19.1% |
| Home appliances | 14.8% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 90.0% |
| Macro-F1 (%, higher is better) | 90.1% |
| Brier, vector (lower is better) | 0.167 |
| NLL (lower is better) | 0.329 |
| ECE (lower is better) | 0.019 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 100.0% at 36.1% (n 1) |
| 0.4–0.5 | 45.5% at 46.4% (n 11) |
| 0.5–0.6 | 68.8% at 53.3% (n 16) |
| 0.6–0.7 | 74.2% at 65.0% (n 31) |
| 0.7–0.8 | 79.4% at 75.0% (n 34) |
| 0.8–0.9 | 85.1% at 86.2% (n 67) |
| 0.9–1.0 | 95.9% at 95.5% (n 340) |

### BoolQ

| Metric | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| Accuracy (%, higher is better) | 91.2% |
| Macro-F1 (%, higher is better) | 90.7% |
| Brier, vector (lower is better) | 0.149 |
| Brier, binary (lower is better) | 0.074 |
| NLL (lower is better) | 0.275 |
| ECE (lower is better) | 0.035 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Kev-4B · llama.cpp v0.7.0 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 69.2% at 54.7% (n 13) |
| 0.6–0.7 | 75.0% at 65.2% (n 28) |
| 0.7–0.8 | 65.4% at 75.2% (n 26) |
| 0.8–0.9 | 88.5% at 86.0% (n 52) |
| 0.9–1.0 | 95.3% at 97.6% (n 381) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
