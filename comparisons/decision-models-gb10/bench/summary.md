# Decision models on one GB10

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point. The series are different models, so the tables give values only, no change against the first.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Clef · SGLang | `clef-sgl` | 2026-10-18 | capyctl=main-075023d7, engine=sglang-0.5.22, gpu=GB10, ready_s=377, engine_peak_gib=86.0 |
| Clef · llama.cpp | `clef-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=91, engine_peak_gib=35.3 |
| Clef-Flash · SGLang | `clefflash-sgl` | 2026-10-18 | capyctl=main-075023d7, engine=sglang-0.5.22, gpu=GB10, ready_s=154, engine_peak_gib=51.5 |
| Clef-Flash · llama.cpp | `clefflash-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=23, engine_peak_gib=15.2 |
| Kev-4B · llama.cpp | `kev4b-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=14, engine_peak_gib=25.0 |
| lev · llama.cpp | `lev-lc` | 2026-10-18 | capyctl=main-075023d7, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=32, engine_peak_gib=11.6 |
| Julia-1 · llama.cpp | `julia1-lc` | 2026-10-18 | capyctl=pr148-a33c4f0d, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=4, engine_peak_gib=0.58 |
| Laya · llama.cpp | `laya-lc` | 2026-10-18 | capyctl=pr148-a33c4f0d, engine=llama.cpp-0.7.0+620d222, gpu=GB10, ready_s=7, engine_peak_gib=1.90 |

## Scoreboard

The summary's scoreboard. Typed is Typed Decisions; ↓ marks columns where lower is better. Memory is CapyCTL's measured peak of the engine process (meta engine_peak_gib), GiB.

| Series | Typed accuracy % | API-Bank score % | GPQA score % | Home score % | AG News accuracy % | BoolQ accuracy % | Typed ECE ↓ | Latency 1q 2k ms ↓ | Questions/s 1 stream 3q 2k | Memory engine peak GiB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Clef · SGLang | 72.4 | 91.5 | 48.0 | 79.5 | 90.4 | 92.4 | 0.023 | 1,581 | 1.73 | 86.0 |
| Clef · llama.cpp | 72.1 | 91.9 | 46.4 | 81.8 | 90.0 | 92.6 | 0.020 | 2,202 | 1.28 | 35.3 |
| Clef-Flash · SGLang | 70.2 | 93.1 | 51.0 | 97.7 | 89.4 | 89.6 | 0.012 | 545 | 4.86 | 51.5 |
| Clef-Flash · llama.cpp | 70.7 | 92.9 | 51.5 | 96.6 | 89.6 | 89.6 | 0.013 | 718 | 3.91 | 15.2 |
| Kev-4B · llama.cpp | 66.7 | 65.9 | 39.3 | 14.8 | 90.0 | 91.2 | 0.092 | 551 | 4.54 | 25.0 |
| lev · llama.cpp | 63.2 | 78.0 | 40.3 | 14.8 | 90.4 | 86.0 | 0.045 | 320 | 7.48 | 11.6 |
| Julia-1 · llama.cpp | 73.0 | 2.0 | 25.5 | 0.0 | 71.2 | 63.2 | 0.224 | 56.2 | 13.3 | 0.580 |
| Laya · llama.cpp | 35.9 | 2.0 | 27.0 | 0.0 | 91.4 | 74.4 | 0.177 | 141 | 4.84 | 1.90 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 question, 200-token state | 395 | 560 | 155 | 203 | 88.1 | 158 | 20.0 | 27.7 |
| 1 question, 2k-token state | 1,581 | 2,202 | 545 | 718 | 551 | 320 | 56.2 | 141 |
| 1 question, 8k-token state | 5,731 | 12,176 | 1,837 | 3,813 | 2,604 | 995 | 392 | 1,004 |
| 3 questions, 200-token state | 473 | 693 | 167 | 239 | 205 | 220 | 26.3 | 46.0 |
| 3 questions, 2k-token state | 1,734 | 2,356 | 607 | 769 | 661 | 396 | 229 | 628 |
| 3 questions, 8k-token state | 5,918 | 12,414 | 1,868 | 3,903 | 2,739 | 1,092 | 1,119 | 2,975 |
| 5 questions, 200-token state | 594 | 751 | 201 | 256 | 316 | 423 | 35.6 | 74.9 |
| 5 questions, 2k-token state | 1,904 | 2,567 | 678 | 839 | 1,223 | 753 | 337 | 944 |
| 5 questions, 8k-token state | 5,975 | 12,672 | 1,893 | 3,934 | 5,322 | 2,092 | 1,822 | 4,946 |

### Latency p95 (ms, lower is better)

| Point | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 question, 200-token state | 403 | 567 | 158 | 207 | 90.7 | 161 | 23.8 | 29.6 |
| 1 question, 2k-token state | 1,607 | 2,233 | 555 | 729 | 560 | 328 | 59.8 | 145 |
| 1 question, 8k-token state | 5,876 | 12,387 | 1,881 | 3,874 | 2,679 | 1,008 | 424 | 1,015 |
| 3 questions, 200-token state | 497 | 734 | 170 | 252 | 210 | 224 | 30.0 | 52.4 |
| 3 questions, 2k-token state | 1,744 | 2,387 | 634 | 780 | 681 | 422 | 237 | 636 |
| 3 questions, 8k-token state | 5,964 | 12,627 | 1,882 | 3,937 | 2,787 | 1,123 | 1,220 | 3,001 |
| 5 questions, 200-token state | 601 | 761 | 206 | 259 | 322 | 430 | 38.8 | 80.0 |
| 5 questions, 2k-token state | 1,920 | 2,657 | 685 | 860 | 1,248 | 770 | 375 | 1,056 |
| 5 questions, 8k-token state | 6,029 | 12,926 | 1,917 | 4,046 | 5,432 | 2,130 | 1,996 | 5,862 |

### Per answered question, p50 (ms, lower is better)

| Point | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 question, 200-token state | 395 | 560 | 155 | 203 | 88.1 | 158 | 20.0 | 27.7 |
| 1 question, 2k-token state | 1,581 | 2,202 | 545 | 718 | 551 | 320 | 56.2 | 141 |
| 1 question, 8k-token state | 5,731 | 12,176 | 1,837 | 3,813 | 2,604 | 995 | 392 | 1,004 |
| 3 questions, 200-token state | 158 | 231 | 55.6 | 79.7 | 68.2 | 73.3 | 8.78 | 15.3 |
| 3 questions, 2k-token state | 578 | 785 | 202 | 256 | 220 | 132 | 76.3 | 209 |
| 3 questions, 8k-token state | 1,973 | 4,138 | 623 | 1,301 | 913 | 364 | 373 | 992 |
| 5 questions, 200-token state | 119 | 150 | 40.3 | 51.1 | 63.1 | 84.6 | 7.11 | 15.0 |
| 5 questions, 2k-token state | 381 | 513 | 136 | 168 | 245 | 151 | 67.3 | 189 |
| 5 questions, 8k-token state | 1,195 | 2,534 | 379 | 787 | 1,064 | 418 | 364 | 989 |

### Input tokens (tok)

| Point | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 question, 200-token state | 389 | 389 | 389 | 389 | 258 | 487 | 259 | 260 |
| 1 question, 2k-token state | 2,201 | 2,201 | 2,201 | 2,201 | 2,070 | 2,284 | 2,064 | 2,078 |
| 1 question, 8k-token state | 8,250 | 8,250 | 8,250 | 8,250 | 8,119 | 8,336 | 8,106 | 8,107 |
| 3 questions, 200-token state | 560 | 560 | 560 | 560 | 710 | 951 | 733 | 730 |
| 3 questions, 2k-token state | 2,383 | 2,383 | 2,383 | 2,383 | 6,180 | 4,567 | 6,178 | 6,218 |
| 3 questions, 8k-token state | 8,428 | 8,428 | 8,428 | 8,428 | 24,316 | 16,651 | 24,304 | 24,294 |
| 5 questions, 200-token state | 732 | 732 | 732 | 732 | 1,176 | 1,606 | 1,218 | 1,210 |
| 5 questions, 2k-token state | 2,554 | 2,554 | 2,554 | 2,554 | 10,280 | 7,913 | 10,284 | 10,337 |
| 5 questions, 8k-token state | 8,604 | 8,604 | 8,604 | 8,604 | 40,530 | 29,137 | 40,498 | 40,450 |

### Peak memory (GiB, lower is better)

| Point | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 question, 200-token state | 90.7 | 39.0 | 56.0 | 19.5 | 34.3 | 16.1 | 5.11 | 5.57 |
| 1 question, 2k-token state | 90.8 | 39.0 | 56.0 | 19.5 | 35.1 | 16.1 | 5.11 | 5.59 |
| 1 question, 8k-token state | 93.6 | 39.2 | 57.7 | 19.6 | 40.2 | 16.2 | 5.13 | 5.62 |
| 3 questions, 200-token state | 93.1 | 39.3 | 57.7 | 19.7 | 37.3 | 16.2 | 5.13 | 5.63 |
| 3 questions, 2k-token state | 93.1 | 39.3 | 57.7 | 19.7 | 37.3 | 16.2 | 5.13 | 5.63 |
| 3 questions, 8k-token state | 94.5 | 39.3 | 58.9 | 19.7 | 37.3 | 16.2 | 5.13 | 5.64 |
| 5 questions, 200-token state | 95.3 | 39.3 | 59.4 | 19.7 | 37.3 | 16.2 | 5.11 | 5.65 |
| 5 questions, 2k-token state | 95.3 | 39.3 | 59.4 | 19.7 | 37.3 | 16.2 | 5.11 | 5.65 |
| 5 questions, 8k-token state | 97.0 | 39.3 | 61.0 | 19.7 | 37.5 | 16.2 | 5.12 | 5.65 |

## Cache

100 requests with a new state each (cold prefix), then 100 repeats of one state after its first sight; 3 questions, 2k-token state.

| ms, lower is better | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| Cold p50 | 1,733 | 2,353 | 626 | 766 | 658 | 394 | 221 | 623 |
| Cold p95 | 1,745 | 2,399 | 637 | 787 | 676 | 420 | 238 | 639 |
| Repeated p50 | 1,719 | 2,322 | 626 | 757 | 649 | 385 | 211 | 593 |
| Repeated p95 | 1,724 | 2,330 | 631 | 762 | 657 | 388 | 217 | 599 |
| Repeated p50 / cold p50 | 0.992 | 0.987 | 1.00 | 0.988 | 0.987 | 0.979 | 0.953 | 0.952 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.577 | 0.425 | 1.62 | 1.30 | 1.51 | 2.49 | 4.43 | 1.61 |
| 2 | 0.588 | 0.430 | 1.70 | 1.36 | 1.60 | 2.73 | 5.31 | 1.72 |
| 4 | 0.602 | 0.430 | 1.88 | 1.36 | 1.61 | 2.73 | 4.90 | 1.64 |
| 8 | 0.620 | 0.430 | 1.91 | 1.35 | 1.60 | 2.74 | 4.95 | 1.64 |

### Answered questions per second (questions/s, higher is better)

| Streams | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.73 | 1.28 | 4.86 | 3.91 | 4.54 | 7.48 | 13.3 | 4.84 |
| 2 | 1.77 | 1.29 | 5.11 | 4.07 | 4.81 | 8.20 | 15.9 | 5.16 |
| 4 | 1.81 | 1.29 | 5.63 | 4.07 | 4.82 | 8.19 | 14.7 | 4.92 |
| 8 | 1.86 | 1.29 | 5.74 | 4.06 | 4.79 | 8.21 | 14.9 | 4.93 |

### Latency p50 (ms, lower is better)

| Streams | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,734 | 2,348 | 623 | 766 | 658 | 397 | 228 | 625 |
| 2 | 3,400 | 4,640 | 1,173 | 1,473 | 1,246 | 732 | 377 | 1,162 |
| 4 | 6,690 | 4,649 | 2,127 | 1,474 | 1,841 | 1,106 | 768 | 2,062 |
| 8 | 11,363 | 11,618 | 4,193 | 3,687 | 5,009 | 2,560 | 1,456 | 4,426 |

### Latency p95 (ms, lower is better)

| Streams | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,746 | 2,392 | 637 | 781 | 681 | 422 | 234 | 645 |
| 2 | 3,422 | 4,786 | 1,204 | 1,515 | 1,765 | 750 | 388 | 1,202 |
| 4 | 6,749 | 4,769 | 2,171 | 1,509 | 3,742 | 2,934 | 1,249 | 4,459 |
| 8 | 21,583 | 11,788 | 4,265 | 3,739 | 5,058 | 5,483 | 2,324 | 8,583 |

### Error rate (%, lower is better)

| Streams | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Peak memory (GiB, lower is better)

| Streams | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 1 | 97.1 | 39.3 | 61.0 | 19.8 | 37.5 | 16.2 | 5.15 | 5.66 |
| 2 | 97.1 | 39.3 | 60.4 | 19.8 | 37.5 | 16.2 | 5.14 | 5.66 |
| 4 | 97.1 | 39.4 | 60.5 | 19.7 | 37.5 | 16.2 | 5.14 | 5.66 |
| 8 | 98.8 | 39.3 | 63.6 | 19.7 | 37.5 | 16.2 | 5.14 | 5.66 |

## Typed Decisions

Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold expected score| in rubric levels.

| Metric | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| Accuracy (%, higher is better) | 72.4% | 72.1% | 70.2% | 70.7% | 66.7% | 63.2% | 73.0% | 35.9% |
| Accuracy, gold ties (%, higher is better) | 72.9% | 72.7% | 70.8% | 71.3% | 67.3% | 63.8% | 73.8% | 36.5% |
| Accuracy, noul (%, higher is better) | 84.3% | 84.5% | 81.3% | 82.0% | 75.8% | 75.8% | 80.8% | 48.3% |
| Accuracy, choice (%, higher is better) | 64.7% | 63.8% | 70.0% | 70.7% | 64.5% | 60.5% | 72.5% | 28.2% |
| Accuracy, score (%, higher is better) | 69.1% | 69.0% | 61.9% | 62.1% | 61.5% | 55.9% | 67.6% | 32.2% |
| KL from gold (lower is better) | 0.191 | 0.191 | 0.210 | 0.209 | 0.223 | 0.298 | 2.96 | 0.568 |
| Brier, soft gold (lower is better) | 0.103 | 0.103 | 0.111 | 0.110 | 0.125 | 0.166 | 0.333 | 0.316 |
| Brier, hard label (lower is better) | 0.370 | 0.370 | 0.392 | 0.391 | 0.465 | 0.492 | 0.491 | 0.750 |
| ECE (lower is better) | 0.023 | 0.020 | 0.012 | 0.013 | 0.092 | 0.045 | 0.224 | 0.177 |
| ECE, gold ties (lower is better) | 0.026 | 0.024 | 0.012 | 0.013 | 0.098 | 0.049 | 0.217 | 0.170 |
| Score MAE (levels, lower is better) | 0.270 | 0.269 | 0.296 | 0.296 | 0.397 | 0.408 | 0.428 | 0.696 |
| Coverage (%, higher is better) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### Typed Decisions accuracy by workflow (%, higher is better)

| Workflow | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| agent_trace_observability | 62.8% | 62.6% | 60.6% | 61.4% | 56.8% | 53.6% | 70.2% | 38.6% |
| customer_service | 74.2% | 74.2% | 75.4% | 76.2% | 71.6% | 68.6% | 70.6% | 38.0% |
| invoice_processing | 78.4% | 77.2% | 75.4% | 75.4% | 68.2% | 63.8% | 80.6% | 35.8% |
| security_incidents | 74.0% | 74.4% | 69.2% | 69.6% | 70.2% | 67.0% | 70.8% | 31.0% |

#### Reliability, Typed Decisions

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | – | – | – | – | – | – | – | – |
| 0.1–0.2 | – | – | – | – | – | – | – | – |
| 0.2–0.3 | – | 0.0% at 28.8% (n 1) | – | 0.0% at 29.6% (n 1) | 28.6% at 29.2% (n 7) | 38.1% at 26.0% (n 97) | – | 16.0% at 27.7% (n 75) |
| 0.3–0.4 | 33.3% at 36.9% (n 45) | 36.2% at 37.2% (n 47) | 35.7% at 36.8% (n 70) | 37.1% at 36.9% (n 70) | 41.8% at 36.2% (n 297) | 33.0% at 36.3% (n 106) | 0.0% at 33.2% (n 1) | 26.9% at 35.3% (n 349) |
| 0.4–0.5 | 50.7% at 45.8% (n 270) | 47.7% at 45.9% (n 277) | 47.1% at 45.8% (n 259) | 49.2% at 45.8% (n 258) | 55.9% at 44.8% (n 417) | 46.2% at 45.6% (n 314) | 16.7% at 46.5% (n 12) | 28.5% at 45.1% (n 382) |
| 0.5–0.6 | 55.9% at 54.9% (n 383) | 57.4% at 55.0% (n 373) | 51.1% at 54.9% (n 350) | 53.0% at 54.8% (n 351) | 64.8% at 54.7% (n 491) | 56.2% at 54.9% (n 390) | 50.0% at 55.2% (n 60) | 43.2% at 54.6% (n 607) |
| 0.6–0.7 | 71.9% at 64.9% (n 292) | 69.0% at 65.1% (n 300) | 65.6% at 64.9% (n 314) | 64.1% at 64.9% (n 315) | 79.6% at 64.8% (n 377) | 68.4% at 65.0% (n 332) | 54.3% at 64.5% (n 46) | 41.3% at 64.3% (n 281) |
| 0.7–0.8 | 74.5% at 74.9% (n 306) | 77.0% at 75.0% (n 305) | 76.0% at 75.1% (n 275) | 76.7% at 75.1% (n 275) | 80.7% at 74.2% (n 171) | 78.0% at 74.7% (n 241) | 59.7% at 75.4% (n 67) | 41.3% at 74.9% (n 189) |
| 0.8–0.9 | 84.6% at 84.8% (n 299) | 84.7% at 85.0% (n 294) | 85.8% at 85.2% (n 366) | 85.8% at 85.2% (n 360) | 84.4% at 84.8% (n 122) | 72.1% at 85.4% (n 294) | 56.6% at 85.3% (n 99) | 35.5% at 83.7% (n 93) |
| 0.9–1.0 | 96.3% at 94.9% (n 405) | 96.3% at 94.9% (n 403) | 95.1% at 94.9% (n 366) | 95.1% at 94.8% (n 370) | 98.3% at 95.4% (n 118) | 89.4% at 93.6% (n 226) | 76.3% at 99.4% (n 1715) | 54.2% at 94.4% (n 24) |

## Decision Index components

Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, not a Decision Index.

### Score (%, higher is better)

| Component | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| API-Bank | 91.5% | 91.9% | 93.1% | 92.9% | 65.9% | 78.0% | 2.0% | 2.0% |
| GPQA Diamond | 48.0% | 46.4% | 51.0% | 51.5% | 39.3% | 40.3% | 25.5% | 27.0% |
| Home appliances | 79.5% | 81.8% | 97.7% | 96.6% | 14.8% | 14.8% | 0.0% | 0.0% |

### Coverage (%, higher is better)

| Component | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| API-Bank | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| GPQA Diamond | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Home appliances | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### Skill, chance-corrected (%, higher is better)

| Component | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| API-Bank | 91.4% | 91.8% | 93.0% | 92.8% | 65.3% | 77.5% | 0.1% | 0.1% |
| GPQA Diamond | 30.6% | 28.6% | 34.7% | 35.4% | 19.1% | 20.4% | 0.7% | 2.7% |
| Home appliances | 79.5% | 81.8% | 97.7% | 96.6% | 14.8% | 14.8% | 0.0% | 0.0% |

## AG News and BoolQ

One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of (p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean −ln p(true class), p clipped at 1e-09. ECE: top label, 10 equal-width bins.

### AG News

| Metric | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| Accuracy (%, higher is better) | 90.4% | 90.0% | 89.4% | 89.6% | 90.0% | 90.4% | 71.2% | 91.4% |
| Macro-F1 (%, higher is better) | 90.5% | 90.1% | 89.5% | 89.7% | 90.1% | 90.5% | 69.1% | 91.6% |
| Brier, vector (lower is better) | 0.168 | 0.168 | 0.165 | 0.165 | 0.167 | 0.153 | 0.465 | 0.132 |
| NLL (lower is better) | 0.340 | 0.340 | 0.342 | 0.344 | 0.329 | 0.306 | 1.19 | 0.259 |
| ECE (lower is better) | 0.040 | 0.041 | 0.040 | 0.038 | 0.019 | 0.048 | 0.160 | 0.033 |

#### Reliability, AG News

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | – | – | – | – | – | – | – | – |
| 0.1–0.2 | – | – | – | – | – | – | – | – |
| 0.2–0.3 | – | – | – | – | – | 50.0% at 27.1% (n 2) | – | – |
| 0.3–0.4 | 0.0% at 35.6% (n 1) | 0.0% at 32.8% (n 1) | – | – | 100.0% at 36.1% (n 1) | – | 0.0% at 38.4% (n 2) | 100.0% at 39.0% (n 1) |
| 0.4–0.5 | 66.7% at 46.1% (n 3) | 50.0% at 47.0% (n 4) | 0.0% at 46.6% (n 5) | 33.3% at 46.6% (n 6) | 45.5% at 46.4% (n 11) | 45.5% at 47.2% (n 11) | 41.7% at 46.9% (n 12) | 44.4% at 46.4% (n 9) |
| 0.5–0.6 | 68.8% at 54.9% (n 16) | 73.3% at 55.2% (n 15) | 27.3% at 54.5% (n 11) | 25.0% at 53.3% (n 8) | 68.8% at 53.3% (n 16) | 58.8% at 54.1% (n 17) | 56.8% at 53.8% (n 37) | 63.2% at 54.9% (n 19) |
| 0.6–0.7 | 72.7% at 65.2% (n 22) | 68.2% at 65.2% (n 22) | 33.3% at 61.9% (n 3) | 20.0% at 62.2% (n 5) | 74.2% at 65.0% (n 31) | 76.9% at 64.8% (n 26) | 50.0% at 65.1% (n 36) | 60.0% at 65.9% (n 20) |
| 0.7–0.8 | 85.7% at 76.2% (n 21) | 89.5% at 76.1% (n 19) | 80.0% at 75.1% (n 15) | 78.6% at 75.1% (n 14) | 79.4% at 75.0% (n 34) | 70.2% at 75.4% (n 47) | 46.8% at 75.4% (n 47) | 81.4% at 75.5% (n 43) |
| 0.8–0.9 | 77.4% at 86.2% (n 62) | 74.2% at 86.0% (n 62) | 79.6% at 86.4% (n 49) | 79.6% at 86.3% (n 49) | 85.1% at 86.2% (n 67) | 93.9% at 85.7% (n 98) | 62.3% at 85.3% (n 61) | 89.1% at 85.9% (n 92) |
| 0.9–1.0 | 94.9% at 97.0% (n 375) | 95.2% at 97.0% (n 377) | 94.0% at 96.4% (n 417) | 94.0% at 96.4% (n 418) | 95.9% at 95.5% (n 340) | 97.3% at 94.4% (n 299) | 82.6% at 97.3% (n 305) | 98.4% at 96.1% (n 316) |

### BoolQ

| Metric | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| Accuracy (%, higher is better) | 92.4% | 92.6% | 89.6% | 89.6% | 91.2% | 86.0% | 63.2% | 74.4% |
| Macro-F1 (%, higher is better) | 92.0% | 92.2% | 89.2% | 89.2% | 90.7% | 85.3% | 49.8% | 71.3% |
| Brier, vector (lower is better) | 0.123 | 0.122 | 0.150 | 0.151 | 0.149 | 0.221 | 0.637 | 0.354 |
| Brier, binary (lower is better) | 0.061 | 0.061 | 0.075 | 0.075 | 0.074 | 0.110 | 0.318 | 0.177 |
| NLL (lower is better) | 0.222 | 0.221 | 0.265 | 0.266 | 0.275 | 0.385 | 1.65 | 0.538 |
| ECE (lower is better) | 0.022 | 0.018 | 0.031 | 0.031 | 0.035 | 0.082 | 0.299 | 0.084 |

#### Reliability, BoolQ

Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.

| Confidence | Clef · SGLang | Clef · llama.cpp | Clef-Flash · SGLang | Clef-Flash · llama.cpp | Kev-4B · llama.cpp | lev · llama.cpp | Julia-1 · llama.cpp | Laya · llama.cpp |
|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | – | – | – | – | – | – | – | – |
| 0.1–0.2 | – | – | – | – | – | – | – | – |
| 0.2–0.3 | – | – | – | – | – | – | – | – |
| 0.3–0.4 | – | – | – | – | – | – | – | – |
| 0.4–0.5 | – | – | – | – | – | – | – | – |
| 0.5–0.6 | 50.0% at 55.0% (n 8) | 62.5% at 55.5% (n 8) | 27.3% at 55.7% (n 11) | 22.2% at 55.8% (n 9) | 69.2% at 54.7% (n 13) | 50.0% at 53.9% (n 12) | 51.9% at 54.5% (n 27) | 58.3% at 54.8% (n 48) |
| 0.6–0.7 | 76.5% at 66.5% (n 17) | 70.6% at 66.5% (n 17) | 46.2% at 65.5% (n 13) | 45.5% at 64.8% (n 11) | 75.0% at 65.2% (n 28) | 44.4% at 66.2% (n 9) | 33.3% at 66.9% (n 12) | 50.0% at 65.8% (n 42) |
| 0.7–0.8 | 68.4% at 75.2% (n 19) | 73.7% at 75.1% (n 19) | 47.1% at 75.3% (n 17) | 45.5% at 75.1% (n 22) | 65.4% at 75.2% (n 26) | 30.8% at 77.4% (n 13) | 50.0% at 76.5% (n 30) | 62.0% at 75.7% (n 92) |
| 0.8–0.9 | 77.4% at 85.9% (n 31) | 76.7% at 85.6% (n 30) | 80.9% at 86.1% (n 47) | 81.4% at 85.7% (n 43) | 88.5% at 86.0% (n 52) | 56.2% at 85.5% (n 32) | 57.1% at 85.8% (n 42) | 75.7% at 84.7% (n 152) |
| 0.9–1.0 | 96.0% at 97.2% (n 425) | 96.0% at 97.1% (n 426) | 95.4% at 96.0% (n 412) | 95.4% at 96.0% (n 415) | 95.3% at 97.6% (n 381) | 91.7% at 97.0% (n 434) | 66.6% at 98.6% (n 389) | 91.0% at 95.1% (n 166) |

Pinned sets:

- AG News: `fancyzhx/ag_news` test at revision `eb185aade064a813bc0b7f42de02595523103ca4` (rows sha256 `edaf2c78a140e852…`), 500 of 7600 rows, seed 0; licence: see the dataset card
- BoolQ: `google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` (rows sha256 `c81b6520f1a51a8b…`), 500 of 3270 rows, seed 0; licence: CC BY-SA 3.0
- Typed Decisions: `LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7` (rows sha256 `ec99662b8129a775…`), 400 of 400 cases; licence: Apache-2.0
- API-Bank: Decision Index catalog 3 (API-Bank-tool-selection), built from `AlibabaResearch/DAMO-ConvAI` at `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `33d2cdbe003f271f…`), 508 of 508 scored rows; licence: see api-bank/LICENSE in DAMO-ConvAI
- GPQA Diamond: Decision Index catalog 25 (GPQA-Diamond), built from https://github.com/idavidrein/gpqa/raw/main/dataset.zip (sha256 `461ae7329f15a3e3…`) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `96e00f1248b6af1d…`), 196 of 196 scored rows; licence: CC BY 4.0; the authors ask that the questions not be published in plain text
- Home appliances: Decision Index catalog 9 (Home-Appliance), built from di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py) as the kit `apolinario/decision-index` at `f5a6e4fd85fe` builds it (rows sha256 `13862aa65d24c0ad…`), 88 of 88 scored rows; licence: MIT (generated by the kit's own code)

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
