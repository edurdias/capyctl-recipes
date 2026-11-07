#!/usr/bin/env python3
"""Summarise read_tensors.py CSVs and next_token_agree.py scores into Markdown tables.

usage: summarize_tensors.py <dir with read-*.csv and score-*.json>
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict


def f(x):
    return float(x) if x not in ("", None) else None


def wavg(rows, key):
    """Mean of `key` weighted by tensor size, over rows that have it."""
    num = den = 0.0
    for r in rows:
        v = f(r[key])
        if v is not None:
            n = int(r["numel"])
            num += v * v * n  # errors combine as energy
            den += n
    return (num / den) ** 0.5 if den else None


def pct(x):
    return "–" if x is None else f"{x * 100:.2f}%"


def main() -> int:
    d = sys.argv[1]
    builds = {}
    for p in sorted(glob.glob(os.path.join(d, "read-*.csv"))):
        name = os.path.basename(p)[5:-4]
        builds[name] = list(csv.DictReader(open(p)))

    kinds = ["mlp.gate", "mlp.up", "mlp.down", "deltanet.qkv", "deltanet.z", "deltanet.out",
             "attn.q", "attn.k", "attn.v", "attn.o", "lm_head"]
    print("## Relative weight error by tensor type (size-weighted)\n")
    print("| Type | " + " | ".join(f"{b} format | {b} error | {b} after scale search" for b in builds) + " |")
    print("|---|" + "---|---|---|" * len(builds))
    for k in kinds:
        cells = []
        for b, rows in builds.items():
            rs = [r for r in rows if r["kind"] == k]
            fmt = "/".join(sorted({r["format"] for r in rs})) or "–"
            cells += [fmt, pct(wavg(rs, "rel_err")), pct(wavg(rs, "rel_err_scale_search"))]
        print(f"| {k} | " + " | ".join(cells) + " |")

    print("\n## Gain-weighted error (input channels weighted by their norm gain squared)\n")
    print("| Type | " + " | ".join(f"{b} | {b} gain-aware search" for b in builds) + " |")
    print("|---|" + "---|---|" * len(builds))
    for k in kinds:
        cells = []
        for b, rows in builds.items():
            rs = [r for r in rows if r["kind"] == k]
            cells += [pct(wavg(rs, "gain_rel_err")), pct(wavg(rs, "gain_rel_err_gain_search"))]
        if any(c != "–" for c in cells):
            print(f"| {k} | " + " | ".join(cells) + " |")

    print("\n## Error on each tensor's top 32 input directions, against uniform noise\n")
    print("| Type | " + " | ".join(f"{b} share / uniform" for b in builds) + " |")
    print("|---|" + "---|" * len(builds))
    for k in kinds:
        cells = []
        for b, rows in builds.items():
            rs = [r for r in rows if r["kind"] == k and f(r["top32_err_share"]) is not None]
            if not rs:
                cells.append("–")
                continue
            share = sum(f(r["top32_err_share"]) for r in rs) / len(rs)
            uni = sum(f(r["top32_uniform_share"]) for r in rs) / len(rs)
            cells.append(f"{share:.4f} / {uni:.4f} ({share / uni:.2f}x)")
        print(f"| {k} | " + " | ".join(cells) + " |")

    print("\n## Scales and saturation (NVFP4 tensors)\n")
    print("| Build | Blocks with a subnormal FP8 scale | Zero scales | Scales equal to max calibration |")
    print("|---|---|---|---|")
    for b, rows in builds.items():
        rs = [r for r in rows if r["format"] == "nvfp4"]
        n = sum(int(r["numel"]) for r in rs)
        sub = sum(f(r["subnormal_scale_frac"]) * int(r["numel"]) for r in rs) / n if n else None
        zero = sum(f(r["zero_scale_frac"]) * int(r["numel"]) for r in rs) / n if n else None
        mx = sum(f(r["scales_are_max"]) * int(r["numel"]) for r in rs) / n if n else None
        print(f"| {b} | {pct(sub)} | {pct(zero)} | {pct(mx)} |")

    for b, rows in builds.items():
        print(f"\n## {b}: the 15 tensors with the largest gain-weighted error (else plain)\n")
        key = lambda r: f(r["gain_rel_err"]) if f(r["gain_rel_err"]) is not None else f(r["rel_err"])
        print("| Tensor | Format | Error | Gain-weighted | Worst row | Top-32 share x uniform |")
        print("|---|---|---|---|---|---|")
        for r in sorted(rows, key=key, reverse=True)[:15]:
            ratio = f(r["top32_err_share"]) / f(r["top32_uniform_share"])
            print(f"| {r['tensor'].replace('model.language_model.', '')} | {r['format']} | {pct(f(r['rel_err']))} | "
                  f"{pct(f(r['gain_rel_err']))} | {pct(f(r['worst_row_rel_err']))} | {ratio:.2f}x |")

    print("\n## By layer: size-weighted error of the NVFP4 tensors\n")
    for b, rows in builds.items():
        by = defaultdict(list)
        for r in rows:
            if r["format"] == "nvfp4" and int(r["layer"]) >= 0:
                by[int(r["layer"])].append(r)
        line = ", ".join(f"{l}:{wavg(rs, 'rel_err') * 100:.2f}" for l, rs in sorted(by.items()))
        print(f"- {b}: {line}")

    scores = {os.path.basename(p)[6:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "score-*.json")))}
    if scores:
        print("\n## Greedy next-token agreement with BF16 (BF16's own answers as the prefix)\n")
        cats = sorted({c for s in scores.values() for c in s["summary"]["by_category"]})
        print("| Build | Positions | Agreement | Median first disagreement | " + " | ".join(cats) + " |")
        print("|---|---|---|---|" + "---|" * len(cats))
        for b, s in scores.items():
            m = s["summary"]
            print(f"| {b} | {m['positions']} | {m['agreement'] * 100:.2f}% | {m['median_first_disagreement']} | "
                  + " | ".join(f"{m['by_category'].get(c, 0) * 100:.2f}%" for c in cats) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
