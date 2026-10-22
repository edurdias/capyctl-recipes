# basal-2.0-4.5B on basal-serve 2.2.0 (custom engine), 1x GB10

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point; percentages compare with basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine), differences of shares are percentage points.

| Series | Model | Measured | Meta |
|---|---|---|---|
| basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) | `basal-20` | 2026-10-21 | capyctl=0.1.6-dev, engine=custom:basal-serve-2.2.0, mode=fast, gpu=GB10, model_revision=e5ba9149760d99b73b1963d8e0ccf948b58afacb |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 question, 200-token state | 89.7 |
| 1 question, 2k-token state | 793 |
| 1 question, 8k-token state | 1,844 |
| 3 questions, 200-token state | 166 |
| 3 questions, 2k-token state | 894 |
| 3 questions, 8k-token state | 1,514 |
| 5 questions, 200-token state | 188 |
| 5 questions, 2k-token state | 926 |
| 5 questions, 8k-token state | 1,542 |

### Latency p95 (ms, lower is better)

| Point | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 question, 200-token state | 103 |
| 1 question, 2k-token state | 799 |
| 1 question, 8k-token state | 1,902 |
| 3 questions, 200-token state | 169 |
| 3 questions, 2k-token state | 902 |
| 3 questions, 8k-token state | 2,045 |
| 5 questions, 200-token state | 198 |
| 5 questions, 2k-token state | 936 |
| 5 questions, 8k-token state | 2,071 |

### Per answered question, p50 (ms, lower is better)

| Point | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 question, 200-token state | 89.7 |
| 1 question, 2k-token state | 793 |
| 1 question, 8k-token state | 1,844 |
| 3 questions, 200-token state | 55.2 |
| 3 questions, 2k-token state | 298 |
| 3 questions, 8k-token state | 505 |
| 5 questions, 200-token state | 37.7 |
| 5 questions, 2k-token state | 185 |
| 5 questions, 8k-token state | 308 |

### Input tokens (tok)

| Point | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 question, 200-token state | 368 |
| 1 question, 2k-token state | 2,171 |
| 1 question, 8k-token state | 8,242 |
| 3 questions, 200-token state | 1,011 |
| 3 questions, 2k-token state | 6,459 |
| 3 questions, 8k-token state | 24,666 |
| 5 questions, 200-token state | 1,634 |
| 5 questions, 2k-token state | 10,724 |
| 5 questions, 8k-token state | 41,006 |

### Peak memory (GiB, lower is better)

| Point | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 question, 200-token state | 32.2 |
| 1 question, 2k-token state | 32.2 |
| 1 question, 8k-token state | 33.1 |
| 3 questions, 200-token state | 33.1 |
| 3 questions, 2k-token state | 33.1 |
| 3 questions, 8k-token state | 33.1 |
| 5 questions, 200-token state | 33.1 |
| 5 questions, 2k-token state | 33.1 |
| 5 questions, 8k-token state | 33.1 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| Cold p50 | 895 |
| Cold p95 | 900 |
| Repeated p50 | 795 |
| Repeated p95 | 801 |
| Repeated p50 / cold p50 | 0.888 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 | 1.15 |
| 2 | 1.17 |
| 4 | 1.52 |
| 8 | 2.06 |

### Answered questions per second (questions/s, higher is better)

| Streams | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 | 3.45 |
| 2 | 3.50 |
| 4 | 4.57 |
| 8 | 6.17 |

### Latency p50 (ms, lower is better)

| Streams | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 | 896 |
| 2 | 1,691 |
| 4 | 2,623 |
| 8 | 3,860 |

### Latency p95 (ms, lower is better)

| Streams | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 | 905 |
| 2 | 1,795 |
| 4 | 2,703 |
| 8 | 3,868 |

### Error rate (%, lower is better)

| Streams | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 | 0.0% |
| 2 | 0.0% |
| 4 | 0.0% |
| 8 | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 1 | 33.1 |
| 2 | 33.2 |
| 4 | 33.2 |
| 8 | 33.2 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| Accuracy (%, higher is better) | 64.8% |
| Accuracy, gold ties (%, higher is better) | 65.5% |
| Accuracy, noul (%, higher is better) | 77.2% |
| Accuracy, choice (%, higher is better) | 61.8% |
| Accuracy, score (%, higher is better) | 57.8% |
| KL from gold (lower is better) | 0.255 |
| Brier, soft gold (lower is better) | 0.133 |
| Brier, hard label (lower is better) | 0.448 |
| ECE (lower is better) | 0.038 |
| ECE, gold ties (lower is better) | 0.032 |
| Score MAE (levels, lower is better) | 0.346 |
| Coverage (%, higher is better) | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| agent_trace_observability | 59.4% |
| customer_service | 69.4% |
| invoice_processing | 71.2% |
| security_incidents | 59.2% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | 33.3% at 29.5% (n 3) |
| 0.3–0.4 | 29.5% at 36.4% (n 78) |
| 0.4–0.5 | 43.8% at 45.7% (n 306) |
| 0.5–0.6 | 54.5% at 54.9% (n 356) |
| 0.6–0.7 | 56.1% at 64.7% (n 351) |
| 0.7–0.8 | 70.9% at 75.0% (n 268) |
| 0.8–0.9 | 79.9% at 85.5% (n 328) |
| 0.9–1.0 | 95.2% at 93.7% (n 310) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| API-Bank | 84.3% |
| GPQA Diamond | 33.2% |
| Home appliances | 15.9% |

### Coverage (%, higher is better)

| Component | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| API-Bank | 100.0% |
| GPQA Diamond | 100.0% |
| Home appliances | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| API-Bank | 83.9% |
| GPQA Diamond | 10.9% |
| Home appliances | 15.9% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| Accuracy (%, higher is better) | 88.2% |
| Macro-F1 (%, higher is better) | 88.3% |
| Brier, vector (lower is better) | 0.185 |
| NLL (lower is better) | 0.356 |
| ECE (lower is better) | 0.034 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | 100.0% at 30.3% (n 1) |
| 0.4–0.5 | 40.0% at 48.2% (n 10) |
| 0.5–0.6 | 56.2% at 56.2% (n 16) |
| 0.6–0.7 | 68.4% at 65.3% (n 19) |
| 0.7–0.8 | 60.9% at 75.5% (n 23) |
| 0.8–0.9 | 79.4% at 85.8% (n 63) |
| 0.9–1.0 | 95.1% at 97.1% (n 368) |

### BoolQ

| Metric | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| Accuracy (%, higher is better) | 89.4% |
| Macro-F1 (%, higher is better) | 88.9% |
| Brier, vector (lower is better) | 0.191 |
| Brier, binary (lower is better) | 0.096 |
| NLL (lower is better) | 0.335 |
| ECE (lower is better) | 0.109 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | basal-2.0-4.5B, basal-serve 2.2.0 --mode fast (custom engine) |
|---|---|
| 0.0–0.1 | – |
| 0.1–0.2 | – |
| 0.2–0.3 | – |
| 0.3–0.4 | – |
| 0.4–0.5 | – |
| 0.5–0.6 | 66.1% at 54.8% (n 56) |
| 0.6–0.7 | 72.5% at 66.1% (n 69) |
| 0.7–0.8 | 95.2% at 75.6% (n 104) |
| 0.8–0.9 | 95.8% at 85.5% (n 190) |
| 0.9–1.0 | 97.5% at 92.7% (n 81) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
