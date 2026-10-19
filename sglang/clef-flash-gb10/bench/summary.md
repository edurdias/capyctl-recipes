# Clef-Flash on SGLang 0.5.22

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with Clef-Flash · SGLang 0.5.22, differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Clef-Flash · SGLang 0.5.22 | `clefflash-sgl` | 2026-10-18 | capyctl=main-075023d7, engine=sglang-0.5.22, gpu=GB10, ready_s=154 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 155 |
| 1 question, 2k-token state | 545 |
| 1 question, 8k-token state | 1,837 |
| 3 questions, 200-token state | 167 |
| 3 questions, 2k-token state | 607 |
| 3 questions, 8k-token state | 1,868 |
| 5 questions, 200-token state | 201 |
| 5 questions, 2k-token state | 678 |
| 5 questions, 8k-token state | 1,893 |

### Latency p95 (ms, lower is better)

| Point | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 158 |
| 1 question, 2k-token state | 555 |
| 1 question, 8k-token state | 1,881 |
| 3 questions, 200-token state | 170 |
| 3 questions, 2k-token state | 634 |
| 3 questions, 8k-token state | 1,882 |
| 5 questions, 200-token state | 206 |
| 5 questions, 2k-token state | 685 |
| 5 questions, 8k-token state | 1,917 |

### Per answered question, p50 (ms, lower is better)

| Point | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 155 |
| 1 question, 2k-token state | 545 |
| 1 question, 8k-token state | 1,837 |
| 3 questions, 200-token state | 55.6 |
| 3 questions, 2k-token state | 202 |
| 3 questions, 8k-token state | 623 |
| 5 questions, 200-token state | 40.3 |
| 5 questions, 2k-token state | 136 |
| 5 questions, 8k-token state | 379 |

### Input tokens (tok)

| Point | Clef-Flash · SGLang 0.5.22 |
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

| Point | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 56.0 |
| 1 question, 2k-token state | 56.0 |
| 1 question, 8k-token state | 57.7 |
| 3 questions, 200-token state | 57.7 |
| 3 questions, 2k-token state | 57.7 |
| 3 questions, 8k-token state | 58.9 |
| 5 questions, 200-token state | 59.4 |
| 5 questions, 2k-token state | 59.4 |
| 5 questions, 8k-token state | 61.0 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Clef-Flash · SGLang 0.5.22 |
|---|---|
| Cold p50 | 626 |
| Cold p95 | 637 |
| Repeated p50 | 626 |
| Repeated p95 | 631 |
| Repeated p50 / cold p50 | 1.00 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 | 1.62 |
| 2 | 1.70 |
| 4 | 1.88 |
| 8 | 1.91 |

### Answered questions per second (questions/s, higher is better)

| Streams | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 | 4.86 |
| 2 | 5.11 |
| 4 | 5.63 |
| 8 | 5.74 |

### Latency p50 (ms, lower is better)

| Streams | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 | 623 |
| 2 | 1,173 |
| 4 | 2,127 |
| 8 | 4,193 |

### Latency p95 (ms, lower is better)

| Streams | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 | 637 |
| 2 | 1,204 |
| 4 | 2,171 |
| 8 | 4,265 |

### Error rate (%, lower is better)

| Streams | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 1 | 61.0 |
| 2 | 60.4 |
| 4 | 60.5 |
| 8 | 63.6 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Clef-Flash · SGLang 0.5.22 |
|---|---|
| Accuracy (%, higher is better) | 70.2% |
| Accuracy, gold ties (%, higher is better) | 70.8% |
| Accuracy, noul (%, higher is better) | 81.3% |
| Accuracy, choice (%, higher is better) | 70.0% |
| Accuracy, score (%, higher is better) | 61.9% |
| KL from gold (lower is better) | 0.210 |
| Brier, soft gold (lower is better) | 0.111 |
| Brier, hard label (lower is better) | 0.392 |
| ECE (lower is better) | 0.012 |
| ECE, gold ties (lower is better) | 0.012 |
| Score MAE (levels, lower is better) | 0.296 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Clef-Flash · SGLang 0.5.22 |
|---|---|
| agent_trace_observability | 60.6% |
| customer_service | 75.4% |
| invoice_processing | 75.4% |
| security_incidents | 69.2% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 35.7% at 36.8% (n 70) |
| 0.4–0.5 | 47.1% at 45.8% (n 259) |
| 0.5–0.6 | 51.1% at 54.9% (n 350) |
| 0.6–0.7 | 65.6% at 64.9% (n 314) |
| 0.7–0.8 | 76.0% at 75.1% (n 275) |
| 0.8–0.9 | 85.8% at 85.2% (n 366) |
| 0.9–1.0 | 95.1% at 94.9% (n 366) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Clef-Flash · SGLang 0.5.22 |
|---|---|
| API-Bank | 93.1% |
| GPQA Diamond | 51.0% |
| Home appliances | 97.7% |

### Coverage (%, higher is better)

| Component | Clef-Flash · SGLang 0.5.22 |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Clef-Flash · SGLang 0.5.22 |
|---|---|
| API-Bank | 93.0% |
| GPQA Diamond | 34.7% |
| Home appliances | 97.7% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Clef-Flash · SGLang 0.5.22 |
|---|---|
| Accuracy (%, higher is better) | 89.4% |
| Macro-F1 (%, higher is better) | 89.5% |
| Brier, vector (lower is better) | 0.165 |
| NLL (lower is better) | 0.342 |
| ECE (lower is better) | 0.040 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | 0.0% at 46.6% (n 5) |
| 0.5–0.6 | 27.3% at 54.5% (n 11) |
| 0.6–0.7 | 33.3% at 61.9% (n 3) |
| 0.7–0.8 | 80.0% at 75.1% (n 15) |
| 0.8–0.9 | 79.6% at 86.4% (n 49) |
| 0.9–1.0 | 94.0% at 96.4% (n 417) |

### BoolQ

| Metric | Clef-Flash · SGLang 0.5.22 |
|---|---|
| Accuracy (%, higher is better) | 89.6% |
| Macro-F1 (%, higher is better) | 89.2% |
| Brier, vector (lower is better) | 0.150 |
| Brier, binary (lower is better) | 0.075 |
| NLL (lower is better) | 0.265 |
| ECE (lower is better) | 0.031 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef-Flash · SGLang 0.5.22 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 27.3% at 55.7% (n 11) |
| 0.6–0.7 | 46.2% at 65.5% (n 13) |
| 0.7–0.8 | 47.1% at 75.3% (n 17) |
| 0.8–0.9 | 80.9% at 86.1% (n 47) |
| 0.9–1.0 | 95.4% at 96.0% (n 412) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
