# CapyCTL overhead on decisions: llama.cpp v0.7.0 directly and through CapyCTL

Measured through CapyCTL with capyctl-bench 0.3.0. Latency cells are the p50 or p95 of the runs at a point. The series are different models, so the tables give values only, no change against the first.

| Series | Model | Measured | Meta |
|---|---|---|---|
| Julia-1 direct | `julia1-oh` | 2026-10-18 | path=direct |
| Julia-1 via CapyCTL | `julia1-oh` | 2026-10-18 | path=capyctl |
| Clef-Flash direct | `clefflash-oh` | 2026-10-18 | path=direct |
| Clef-Flash via CapyCTL | `clefflash-oh` | 2026-10-18 | path=capyctl |

## Scoreboard

The summary's scoreboard. Typed is Typed Decisions; ↓ marks columns where lower is better. Memory is the highest --memory-cmd sample during the run, GiB: host-wide, so it includes the OS and anything else running, not the model's own footprint.

| Series | Latency 1q 2k ms ↓ | Questions/s 1 stream 3q 2k |
|---|---|---|
| Julia-1 direct | 45.1 | 14.5 |
| Julia-1 via CapyCTL | 57.9 | 13.3 |
| Clef-Flash direct | 704 | 3.96 |
| Clef-Flash via CapyCTL | 724 | 3.85 |

## Latency

One request at a time, client side: request sent to the whole answer read. Per question is the request's latency over the questions it answered (amortized, not the latency of one question).

### Latency p50 (ms, lower is better)

| Point | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 question, 200-token state | 7.52 | 19.1 | 184 | 206 |
| 1 question, 2k-token state | 45.1 | 57.9 | 704 | 724 |
| 3 questions, 200-token state | 15.2 | 26.9 | 220 | 242 |
| 3 questions, 2k-token state | 204 | 230 | 757 | 778 |

### Latency p95 (ms, lower is better)

| Point | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 question, 200-token state | 8.77 | 23.4 | 200 | 210 |
| 1 question, 2k-token state | 47.9 | 60.5 | 726 | 734 |
| 3 questions, 200-token state | 16.9 | 29.1 | 231 | 255 |
| 3 questions, 2k-token state | 213 | 236 | 770 | 787 |

### Per answered question, p50 (ms, lower is better)

| Point | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 question, 200-token state | 7.52 | 19.1 | 184 | 206 |
| 1 question, 2k-token state | 45.1 | 57.9 | 704 | 724 |
| 3 questions, 200-token state | 5.07 | 8.96 | 73.2 | 80.8 |
| 3 questions, 2k-token state | 68.2 | 76.6 | 252 | 259 |

### Input tokens (tok)

| Point | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 question, 200-token state | 259 | 259 | 389 | 389 |
| 1 question, 2k-token state | 2,064 | 2,064 | 2,201 | 2,201 |
| 3 questions, 200-token state | 736 | 736 | 559 | 559 |
| 3 questions, 2k-token state | 6,168 | 6,168 | 2,378 | 2,378 |

## Throughput

Closed loop: N streams, each sending its next request when its last one returns, 200 requests per level; 3 questions and a 2k-token state each, every request its own state.

### Requests per second (requests/s, higher is better)

| Streams | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 | 4.83 | 4.44 | 1.32 | 1.28 |
| 8 | 4.92 | 4.93 | 1.33 | 1.34 |

### Answered questions per second (questions/s, higher is better)

| Streams | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 | 14.5 | 13.3 | 3.96 | 3.85 |
| 8 | 14.8 | 14.8 | 4.00 | 4.01 |

### Latency p50 (ms, lower is better)

| Streams | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 | 208 | 226 | 755 | 777 |
| 8 | 1,465 | 1,463 | 3,745 | 3,729 |

### Latency p95 (ms, lower is better)

| Streams | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 | 215 | 235 | 772 | 792 |
| 8 | 2,527 | 2,549 | 3,800 | 3,788 |

### Error rate (%, lower is better)

| Streams | Julia-1 direct | Julia-1 via CapyCTL | Clef-Flash direct | Clef-Flash via CapyCTL |
|---|---|---|---|---|
| 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| 8 | 0.0% | 0.0% | 0.0% | 0.0% |

Produced by `capyctl-bench report`; settings and per-run spread are in report.html.
