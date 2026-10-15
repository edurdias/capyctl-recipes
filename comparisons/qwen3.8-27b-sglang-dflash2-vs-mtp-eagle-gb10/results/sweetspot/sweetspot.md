# Qwen3.8-27B on SGLang 0.5.21 through CapyCTL: the sweet spot between DFlash2 and MTP/EAGLE

Every variant runs the catalog checkpoint `nvidia/Qwen3.8-27B-NVFP4@482ca0f3` (NVFP4 head), FP8 KV,
a 262144-token context, typed parsers and CapyCTL's default residency (`deep`), one at a time,
each deleted before the next. Short prompts: C1 to C22, 3 rounds, `max_tokens` 512; context:
2k, 32k, 128k, 2 runs, `max_tokens` 128; one stall trial; one park and wake by request.

## Hosts

| Host | Checkpoint | SGLang | torch | FlashInfer | Driver | Kernel |
|---|---|---|---|---|---|---|
| a | {hf: nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80} | 0.5.21 | 2.13.0 | 0.6.18 | 580.173.02 | 6.17.0-1031-nvidia |
| b | {hf: nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80} | 0.5.21 | 2.13.0 | 0.6.18 | 580.173.02 | 6.17.0-1031-nvidia |

## Host comparability

eagle-22 ran on host a; host b ran the same deployment first as a calibration.

| Figure | Host b (calibration) | Host a (eagle-22) | Difference |
|---|---|---|---|
| C1 per-stream decode, tok/s | 25.6 | 24.8 | +3.0% |
| C8 aggregate, tok/s | 169.2 | 150.4 | +12.5% |

The hosts differ by up to 12.5%: comparisons between variants run on different hosts are approximate.

## Short prompts

| Variant | Host | Request | Running | Engine KV tokens | C1 aggregate | C4 aggregate | C8 aggregate | C16 aggregate | C22 aggregate | Per stream C1 / C22 | TTFT p50 C1 / C22 | Accept length |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dflash-8 | b | 65GiB | 8 | 399457 | 27.8 | 80.5 | 127.4 | 124.6 | 122.7 | 28.3 / 17.6 | 0.21 / 29.51 s | 3.29 |
| dflash-mix-22 | b | 94GiB (named 94 GiB) | 22 | 399457 | 28.0 | 80.5 | 125.5 | 173.4 | 190.7 | 28.3 / 9.5 | 0.21 / 0.46 s | 3.30 |
| dflash-mix-12 | b | 75GiB (named 75 GiB) | 12 | 399457 | 28.3 | 85.5 | 118.1 | 133.5 | 155.0 | 28.6 / 15.5 | 0.21 / 0.33 s | 3.34 |
| eagle-22 | a | 84GiB | 22 | 894373 | 24.2 | 88.0 | 150.4 | 272.8 | 343.7 | 24.8 / 17.2 | 0.28 / 0.46 s | 2.60 |
| eagle-12 | a | 75GiB | 12 | 894373 | 24.1 | 88.6 | 163.0 | 173.1 | 209.9 | 24.7 / 20.8 | 0.30 / 0.32 s | 2.67 |
| run 1: DFlash2 catalog, 8 running | a | n/a | n/a | n/a | 32.4 | 81.4 | 123.4 | 124.1 | 122.0 | 33.1 / 17.4 | 0.21 / 29.29 s | 3.29 |
| run 1: MTP/EAGLE, BF16-head checkpoint, 79 GiB | a | n/a | n/a | n/a | 17.7 | 66.9 | 119.4 | 214.7 | 280.7 | 17.9 / 14.4 | 0.27 / 0.51 s | 2.60 |
| run 2: raw SGLang, BF16-head checkpoint, no CapyCTL | a | n/a | 22 | 884688 | 17.5 | n/a | 122.5 | n/a | 278.6 | 17.8 / 14.0 | 0.36 / 0.52 s | 2.65 |
| run 2: MTP/EAGLE, NVFP4 head, restart_only, 84 GiB | a | n/a | 22 | 894373 | 24.7 | n/a | 156.5 | n/a | 336.6 | 24.9 / 17.2 | 0.21 / 0.57 s | 2.60 |

## Context

| Variant | Host | 2k decode, TTFT | 32k decode, TTFT | 128k decode, TTFT |
|---|---|---|---|---|
| dflash-8 | b | 47.5 tok/s, 0.9 s | 31.4 tok/s, 18.7 s | 24.5 tok/s, 122.3 s |
| dflash-mix-22 | b | 47.6 tok/s, 0.9 s | 31.4 tok/s, 21.6 s | 25.0 tok/s, 133.3 s |
| dflash-mix-12 | b | 48.0 tok/s, 0.9 s | 31.6 tok/s, 20.7 s | 25.2 tok/s, 130.0 s |
| eagle-22 | a | 25.5 tok/s, 0.9 s | 25.9 tok/s, 19.3 s | 20.1 tok/s, 128.1 s |
| eagle-12 | a | 30.5 tok/s, 0.9 s | 25.7 tok/s, 19.4 s | 23.0 tok/s, 128.2 s |
| run 1: DFlash2 catalog, 8 running | a | 56.9 tok/s, 0.9 s | 28.8 tok/s, 18.7 s | 30.4 tok/s, 122.4 s |
| run 1: MTP/EAGLE, BF16-head checkpoint, 79 GiB | a | 19.3 tok/s, 0.9 s | 18.5 tok/s, 21.4 s | 16.9 tok/s, 136.0 s |

## Memory, park and wake, stall

| Variant | Host | Ready footprint (MemAvailable drop) | CapyCTL Ready charge | Start | Park | Parked charge | MemAvailable while parked (drop) | Wake, first token | Wake answer | Stall: longest gap, cold prompt TTFT |
|---|---|---|---|---|---|---|---|---|---|---|
| dflash-8 | b | 69.1 GiB | 66.2 GiB | 479 s | 1.2 s | 34.4 GiB (measured) | 81.2 GiB (36.2) | 2.57 s | correct | 55.7 s, 60.2 s |
| dflash-mix-22 | b | 95.7 GiB | 95.2 GiB | 288 s | 1.2 s | 34.9 GiB (measured) | 81.4 GiB (36.0) | 3.58 s | correct | 8.8 s, 67.7 s |
| dflash-mix-12 | b | 76.5 GiB | 76.2 GiB | 288 s | 1.2 s | 34.7 GiB (measured) | 81.6 GiB (36.6) | 2.78 s | correct | 8.6 s, 65.3 s |
| eagle-22 | a | 84.0 GiB | 85.2 GiB | 628 s | 1.2 s | 35.9 GiB (measured) | 80.0 GiB (37.3) | 3.19 s | correct | 8.5 s, 63.5 s |
| eagle-12 | a | 75.1 GiB | 76.2 GiB | 610 s | 1.1 s | 34.2 GiB (measured) | 82.2 GiB (35.9) | 2.85 s | correct | 8.5 s, 63.5 s |
| run 1: DFlash2 catalog, 8 running | a | 70.0 GiB | n/a | n/a s | n/a s | n/a | n/a GiB (n/a) | n/a s | n/a | 57.1 s, 61.1 s |
| run 1: MTP/EAGLE, BF16-head checkpoint, 79 GiB | a | 78.7 GiB | n/a | n/a s | n/a s | n/a | n/a GiB (n/a) | n/a s | n/a | 9.0 s, 67.7 s |
| run 2: raw SGLang, BF16-head checkpoint, no CapyCTL | a | 83.9 GiB | n/a | n/a s | n/a s | n/a | n/a GiB (n/a) | n/a s | n/a | n/a s, n/a s |
| run 2: MTP/EAGLE, NVFP4 head, restart_only, 84 GiB | a | 78.9 GiB | n/a | n/a s | n/a s | n/a | n/a GiB (n/a) | n/a s | n/a | 8.5 s, 64.6 s |

## Draft depth on eagle-22

| Steps / top-k / draft tokens | C1 per stream | C8 aggregate | Accept length |
|---|---|---|---|
| 3 / 1 / 4 (catalog) | 24.8 | 150.4 | 2.60 |
| 2 / 1 / 3 | 24.1 | 154.8 | 2.31 |
| 5 / 1 / 6 | 27.6 | 135.0 | 2.96 |

## Which recipe for which workload

The best of the five variants on each figure, from the tables above.
The hosts are not within 3%: a pick that beats a variant from the other host by less than the host difference is not established.

- One user at a time (C1 per-stream decode): **dflash-mix-12** (28.6 tok/s, host b)
- A few users (C8 aggregate): **eagle-12** (163.0 tok/s, host a)
- Many users (C22 aggregate): **eagle-22** (343.7 tok/s, host a)
- Interactive under load (C22 TTFT p50): **eagle-12** (0.32 s, host a)
- Long context (128k decode): **dflash-mix-12** (25.2 tok/s, host b)
- Long prompts beside running streams (stall gap): **eagle-22** (8.5 s, host a)
- Least memory (CapyCTL Ready charge): **dflash-8** (66.2 GiB, host b)
- Fastest wake from park (first token): **dflash-8** (2.57 s, host b)

## Notes

- The first CapyCTL start on each host built the kernels (shared cache per host), so its start time
  includes the build; later starts are warm.
- DFlash2 runs at most its running count (8, 12 or 22); at more streams the rest wait in CapyCTL.
- Reference rows: run 1 (2026-10-14) ran 2 interleaved passes of 5 rounds at 32768 context for short
  prompts; run 2 (2026-10-15) ran C1, C8 and C22 only, 3 rounds, `restart_only`.
- One machine per variant, one pass; medians of 3 rounds (short prompts) and 2 runs (context).
