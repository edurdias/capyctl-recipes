# Clef on SGLang 0.5.22

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with Clef · SGLang 0.5.22, differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Clef · SGLang 0.5.22 | `clef-sgl` | 2026-10-18 | capyctl=main-075023d7, engine=sglang-0.5.22, gpu=GB10, ready_s=377 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Clef · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 395 |
| 1 question, 2k-token state | 1,581 |
| 1 question, 8k-token state | 5,731 |
| 3 questions, 200-token state | 473 |
| 3 questions, 2k-token state | 1,734 |
| 3 questions, 8k-token state | 5,918 |
| 5 questions, 200-token state | 594 |
| 5 questions, 2k-token state | 1,904 |
| 5 questions, 8k-token state | 5,975 |

### Latency p95 (ms, lower is better)

| Point | Clef · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 403 |
| 1 question, 2k-token state | 1,607 |
| 1 question, 8k-token state | 5,876 |
| 3 questions, 200-token state | 497 |
| 3 questions, 2k-token state | 1,744 |
| 3 questions, 8k-token state | 5,964 |
| 5 questions, 200-token state | 601 |
| 5 questions, 2k-token state | 1,920 |
| 5 questions, 8k-token state | 6,029 |

### Per answered question, p50 (ms, lower is better)

| Point | Clef · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 395 |
| 1 question, 2k-token state | 1,581 |
| 1 question, 8k-token state | 5,731 |
| 3 questions, 200-token state | 158 |
| 3 questions, 2k-token state | 578 |
| 3 questions, 8k-token state | 1,973 |
| 5 questions, 200-token state | 119 |
| 5 questions, 2k-token state | 381 |
| 5 questions, 8k-token state | 1,195 |

### Input tokens (tok)

| Point | Clef · SGLang 0.5.22 |
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

| Point | Clef · SGLang 0.5.22 |
|---|---|
| 1 question, 200-token state | 90.7 |
| 1 question, 2k-token state | 90.8 |
| 1 question, 8k-token state | 93.6 |
| 3 questions, 200-token state | 93.1 |
| 3 questions, 2k-token state | 93.1 |
| 3 questions, 8k-token state | 94.5 |
| 5 questions, 200-token state | 95.3 |
| 5 questions, 2k-token state | 95.3 |
| 5 questions, 8k-token state | 97.0 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Clef · SGLang 0.5.22 |
|---|---|
| Cold p50 | 1,733 |
| Cold p95 | 1,745 |
| Repeated p50 | 1,719 |
| Repeated p95 | 1,724 |
| Repeated p50 / cold p50 | 0.992 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Clef · SGLang 0.5.22 |
|---|---|
| 1 | 0.577 |
| 2 | 0.588 |
| 4 | 0.602 |
| 8 | 0.620 |

### Answered questions per second (questions/s, higher is better)

| Streams | Clef · SGLang 0.5.22 |
|---|---|
| 1 | 1.73 |
| 2 | 1.77 |
| 4 | 1.81 |
| 8 | 1.86 |

### Latency p50 (ms, lower is better)

| Streams | Clef · SGLang 0.5.22 |
|---|---|
| 1 | 1,734 |
| 2 | 3,400 |
| 4 | 6,690 |
| 8 | 11,363 |

### Latency p95 (ms, lower is better)

| Streams | Clef · SGLang 0.5.22 |
|---|---|
| 1 | 1,746 |
| 2 | 3,422 |
| 4 | 6,749 |
| 8 | 21,583 |

### Error rate (%, lower is better)

| Streams | Clef · SGLang 0.5.22 |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Clef · SGLang 0.5.22 |
|---|---|
| 1 | 97.1 |
| 2 | 97.1 |
| 4 | 97.1 |
| 8 | 98.8 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Clef · SGLang 0.5.22 |
|---|---|
| Accuracy (%, higher is better) | 72.4% |
| Accuracy, gold ties (%, higher is better) | 72.9% |
| Accuracy, noul (%, higher is better) | 84.3% |
| Accuracy, choice (%, higher is better) | 64.7% |
| Accuracy, score (%, higher is better) | 69.1% |
| KL from gold (lower is better) | 0.191 |
| Brier, soft gold (lower is better) | 0.103 |
| Brier, hard label (lower is better) | 0.370 |
| ECE (lower is better) | 0.023 |
| ECE, gold ties (lower is better) | 0.026 |
| Score MAE (levels, lower is better) | 0.270 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Clef · SGLang 0.5.22 |
|---|---|
| agent_trace_observability | 62.8% |
| customer_service | 74.2% |
| invoice_processing | 78.4% |
| security_incidents | 74.0% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · SGLang 0.5.22 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 33.3% at 36.9% (n 45) |
| 0.4–0.5 | 50.7% at 45.8% (n 270) |
| 0.5–0.6 | 55.9% at 54.9% (n 383) |
| 0.6–0.7 | 71.9% at 64.9% (n 292) |
| 0.7–0.8 | 74.5% at 74.9% (n 306) |
| 0.8–0.9 | 84.6% at 84.8% (n 299) |
| 0.9–1.0 | 96.3% at 94.9% (n 405) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Clef · SGLang 0.5.22 |
|---|---|
| API-Bank | 91.5% |
| GPQA Diamond | 48.0% |
| Home appliances | 79.5% |

### Coverage (%, higher is better)

| Component | Clef · SGLang 0.5.22 |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Clef · SGLang 0.5.22 |
|---|---|
| API-Bank | 91.4% |
| GPQA Diamond | 30.6% |
| Home appliances | 79.5% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Clef · SGLang 0.5.22 |
|---|---|
| Accuracy (%, higher is better) | 90.4% |
| Macro-F1 (%, higher is better) | 90.5% |
| Brier, vector (lower is better) | 0.168 |
| NLL (lower is better) | 0.340 |
| ECE (lower is better) | 0.040 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · SGLang 0.5.22 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 0.0% at 35.6% (n 1) |
| 0.4–0.5 | 66.7% at 46.1% (n 3) |
| 0.5–0.6 | 68.8% at 54.9% (n 16) |
| 0.6–0.7 | 72.7% at 65.2% (n 22) |
| 0.7–0.8 | 85.7% at 76.2% (n 21) |
| 0.8–0.9 | 77.4% at 86.2% (n 62) |
| 0.9–1.0 | 94.9% at 97.0% (n 375) |

### BoolQ

| Metric | Clef · SGLang 0.5.22 |
|---|---|
| Accuracy (%, higher is better) | 92.4% |
| Macro-F1 (%, higher is better) | 92.0% |
| Brier, vector (lower is better) | 0.123 |
| Brier, binary (lower is better) | 0.061 |
| NLL (lower is better) | 0.222 |
| ECE (lower is better) | 0.022 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · SGLang 0.5.22 |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 50.0% at 55.0% (n 8) |
| 0.6–0.7 | 76.5% at 66.5% (n 17) |
| 0.7–0.8 | 68.4% at 75.2% (n 19) |
| 0.8–0.9 | 77.4% at 85.9% (n 31) |
| 0.9–1.0 | 96.0% at 97.2% (n 425) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
