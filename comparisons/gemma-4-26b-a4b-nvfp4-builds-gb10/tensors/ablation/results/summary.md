# Family ablation: 62,658 shared BF16 answer positions, 200 prompts

All new rows use r2 experts; all unmentioned weights are BF16 source weights. F and full r2 reuse the original measurements.

| Variant | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: r2 experts; BF16 dense/attention | 97.397 | 31.5 | 96.347 | 98.009 | 99.252 | 96.319 | 100.000 |
| B: r2 experts+dense; BF16 attention | 96.455 | 27 | 94.940 | 97.275 | 98.924 | 95.152 | 99.867 |
| C: r2 experts+attention; BF16 dense | 95.512 | 22 | 93.793 | 96.259 | 98.401 | 94.158 | 99.934 |
| D: r2 experts+dense; FP8 tensor attention | 96.344 | 24 | 94.727 | 97.203 | 98.801 | 95.098 | 99.867 |
| D-channel: r2 experts+dense; FP8 channel attention | 96.393 | 24.5 | 94.804 | 97.311 | 98.852 | 95.045 | 99.867 |
| E: r2 experts; FP8 tensor dense/attention | 97.151 | 29 | 96.051 | 97.834 | 99.170 | 95.912 | 99.934 |
| G: E + per-channel FP8 head | 97.134 | 29 | 96.010 | 97.839 | 99.108 | 95.912 | 100.000 |
| H: E + searched NVFP4 head | 96.861 | 28 | 95.673 | 97.619 | 99.139 | 95.425 | 99.934 |
| I: A + per-channel FP8 head | 97.355 | 31 | 96.323 | 97.998 | 99.201 | 96.219 | 100.000 |
| F: NVIDIA experts; BF16 rest (reused) | 97.132 | 30 | 96.075 | 97.701 | 99.201 | 95.952 | 100.000 |
| Full r2 NVFP4 baseline (reused) | 94.805 | 20.5 | 92.817 | 95.771 | 98.207 | 93.084 | 99.734 |

[INFERENCE] Weight/weight-scale bytes read per decode token, arithmetic only: attention + dense MLP + eight routed experts per layer + router + output head. Input embeddings remain BF16; only their lookup row is needed, not a second full matrix read. Attention Q/K norms and router scales are included. NVFP4 counts packed weights plus block/global weight scales; FP8 counts one byte/weight plus its scales. The expert term is bank payload times 8/128.

| Variant | Attention B/token | Dense MLP B/token | Eight experts B/token | Router B/token | Family sum B/token | Output head B/token | Total incl head B/token | Total incl head GiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2,220,395,520 | 1,070,530,560 | 802,900,800 | 21,803,520 | 4,115,630,400 | 1,476,395,008 | 5,592,025,408 | 5.207980 |
| B | 2,220,395,520 | 301,087,080 | 802,900,800 | 21,803,520 | 3,346,186,920 | 1,476,395,008 | 4,822,581,928 | 4.491379 |
| C | 624,512,460 | 1,070,530,560 | 802,900,800 | 21,803,520 | 2,519,747,340 | 1,476,395,008 | 3,996,142,348 | 3.721698 |
| D | 1,110,216,140 | 301,087,080 | 802,900,800 | 21,803,520 | 2,236,007,540 | 1,476,395,008 | 3,712,402,548 | 3.457444 |
| D-channel | 1,110,886,400 | 301,087,080 | 802,900,800 | 21,803,520 | 2,236,677,800 | 1,476,395,008 | 3,713,072,808 | 3.458069 |
| E | 1,110,216,140 | 535,265,640 | 802,900,800 | 21,803,520 | 2,470,186,100 | 1,476,395,008 | 3,946,581,108 | 3.675540 |
| G | 1,110,216,140 | 535,265,640 | 802,900,800 | 21,803,520 | 2,470,186,100 | 738,721,792 | 3,208,907,892 | 2.988528 |
| H | 1,110,216,140 | 535,265,640 | 802,900,800 | 21,803,520 | 2,470,186,100 | 415,236,100 | 2,885,422,200 | 2.687259 |
| I | 2,220,395,520 | 1,070,530,560 | 802,900,800 | 21,803,520 | 4,115,630,400 | 738,721,792 | 4,854,352,192 | 4.520968 |
| F | 2,220,395,520 | 1,070,530,560 | 802,900,800 | 21,803,520 | 4,115,630,400 | 1,476,395,008 | 5,592,025,408 | 5.207980 |
| r2 | 624,512,460 | 301,087,080 | 802,900,800 | 21,803,520 | 1,750,303,860 | 1,476,395,008 | 3,226,698,868 | 3.005097 |

[INFERENCE] Compatible engines allocate the channel head scales in FP32 rather than this simulation's BF16. Add 524,288 bytes per token to G/I's raw payload totals; loaded-FP32-scale totals: G 3,209,432,180 B/token; I 4,854,876,480 B/token.

These are not measured memory transactions or tokens/second. The totals now include the output head (BF16 except G/H/I), but omit other layer state, the embedding lookup row, activations, KV-cache traffic, temporary buffers and batching/cache reuse. In the offline accuracy runs every variant is reconstructed to BF16, so actual torch weight traffic is not this packed-format arithmetic.

| New variant | Load s | Evaluation s | Wall s | Peak CUDA allocated GiB | Peak CUDA reserved GiB | Peak RSS GiB | Minimum host available GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 17.84 | 64.05 | 81.90 | 50.068 | 50.160 | 3.977 | 29.620 |
| B | 17.79 | 64.33 | 82.12 | 50.009 | 50.137 | 3.978 | 29.613 |
| C | 18.26 | 64.45 | 82.71 | 49.991 | 50.426 | 3.946 | 29.463 |
| D | 18.51 | 64.39 | 82.90 | 49.979 | 50.312 | 4.012 | 29.399 |
| D-channel | 18.60 | 64.52 | 83.12 | 49.979 | 50.312 | 4.012 | 29.406 |
| E | 18.86 | 64.39 | 83.25 | 50.001 | 50.426 | 4.012 | 29.356 |
| G | 19.93 | 64.07 | 84.00 | 57.056 | 58.160 | 4.006 | 22.006 |
| H | 43.38 | 65.23 | 108.61 | 52.929 | 53.420 | 4.003 | 26.617 |
| I | 25.13 | 64.42 | 89.55 | 57.123 | 57.879 | 3.996 | 21.755 |
