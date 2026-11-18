#!/usr/bin/env python3
"""Print every measured recipe against the production baseline as Markdown tables.

python3 recipe-table.py

Reads bench/final-matrix.json and each row's bench/runs/<speed_run>/measure.json; never
runs a benchmark. Throughput is total completion tokens / round wall time (median of the
three recorded rounds) at 1, 2, 4 and 8 streams. Time to first token is the median over
every request of those rounds (short prompts of about 50 tokens), plus the separate
exactly-24,000-token runs at 1 and 4 streams. Changes are against the baseline row.
"""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = (1, 2, 4, 8)
BASELINE = ("NVIDIA", "SGLang 0.5.22", "none")

# (build, engine, drafter, label) in display order; the baseline first.
RECIPES = (
    (*BASELINE, "**Baseline:** NVIDIA, SGLang, no drafter (production configuration)"),
    ("NVIDIA", "vLLM 0.32.0", "none", "NVIDIA, vLLM, no drafter"),
    ("NVIDIA", "vLLM 0.32.0", "MTP 3", "NVIDIA, vLLM, MTP 3"),
    ("RedHat", "vLLM 0.32.0", "none", "RedHat NVFP4, vLLM, no drafter"),
    ("RedHat", "vLLM 0.32.0", "MTP 3", "RedHat NVFP4, vLLM, MTP 3"),
    ("Unsloth", "vLLM 0.32.0", "none", "Unsloth NVFP4, vLLM, no drafter"),
    ("Unsloth", "vLLM 0.32.0", "MTP 3", "Unsloth NVFP4, vLLM, MTP 3"),
    ("r2", "SGLang 0.5.22", "none", "CapyCTL r2, SGLang, no drafter"),
    ("r2", "vLLM 0.32.0", "MTP 3", "CapyCTL r2, vLLM, MTP 3"),
    ("r3", "SGLang 0.5.22", "none", "CapyCTL r3, SGLang, no drafter"),
    ("r3", "vLLM 0.32.0", "none", "CapyCTL r3, vLLM, no drafter"),
    ("r3", "SGLang 0.5.22", "MTP 2", "CapyCTL r3, SGLang, MTP 2"),
    ("r3", "SGLang 0.5.22", "MTP 3", "CapyCTL r3, SGLang, MTP 3"),
    ("r3", "SGLang 0.5.22", "DFlash 8", "**CapyCTL r3, SGLang, DFlash 8 (shipped)**"),
    ("r3", "vLLM 0.32.0", "MTP 3", "**CapyCTL r3, vLLM, MTP 3 (shipped)**"),
)


def measure(row):
    path = HERE / "bench" / "runs" / row["speed_run"] / "measure.json"
    return json.loads(path.read_text()) if path.exists() else None


def levels(data):
    """{streams: (aggregate tok/s, short-prompt TTFT s)} and {streams: 24k TTFT s}."""
    out = {}
    for level in data.get("concurrency", []):
        rounds = level.get("rounds") or []
        if not rounds:
            continue
        ttfts = [r["ttft_s"] for rnd in rounds for r in rnd.get("requests", [])
                 if r.get("ttft_s") is not None and not r.get("error")]
        out[level["streams"]] = (statistics.median(rnd["aggregate_tps"] for rnd in rounds),
                                 statistics.median(ttfts) if ttfts else None)
    long = {lv["streams"]: lv.get("ttft_s") for lv in (data.get("ttft24k") or {}).get("levels", [])
            if lv.get("valid", True)}
    return out, long


def change(value, base, lower_is_better=False):
    if value is None or base in (None, 0):
        return ""
    pct = 100 * (value / base - 1)
    good = pct < 0 if lower_is_better else pct > 0
    if abs(pct) < 0.5:
        return " (=)"
    return f" ({pct:+.0f}%{'' if good else ' ▼'})"


def main():
    matrix = json.loads((HERE / "bench" / "final-matrix.json").read_text())
    rows = {}
    for row in matrix["rows"]:
        key = (row["build"], row["engine"], row.get("drafter"))
        if row.get("speed_run") and key not in rows:
            data = measure(row)
            if data and data.get("concurrency"):
                rows[key] = levels(data)
    base, base_long = rows[BASELINE]

    print("| Recipe | " + " | ".join(f"{n} stream{'s' if n > 1 else ''}" for n in LEVELS) + " |")
    print("|---|" + "---:|" * len(LEVELS))
    for build, engine, drafter, label in RECIPES:
        if (build, engine, drafter) not in rows:
            continue
        tps, _ = rows[(build, engine, drafter)]
        cells = [f"{tps[n][0]:.1f}{'' if (build, engine, drafter) == BASELINE else change(tps[n][0], base[n][0])}"
                 if n in tps else "—" for n in LEVELS]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    print("| Recipe | " + " | ".join(f"{n} stream{'s' if n > 1 else ''}" for n in LEVELS)
          + " | 24k input, 1 stream | 24k input, 4 streams |")
    print("|---|" + "---:|" * (len(LEVELS) + 2))
    for build, engine, drafter, label in RECIPES:
        if (build, engine, drafter) not in rows:
            continue
        tps, long = rows[(build, engine, drafter)]
        is_base = (build, engine, drafter) == BASELINE
        cells = [f"{tps[n][1] * 1000:.0f} ms{'' if is_base else change(tps[n][1], base[n][1], True)}"
                 if n in tps and tps[n][1] is not None else "—" for n in LEVELS]
        cells += [f"{long[n]:.2f} s{'' if is_base else change(long[n], base_long.get(n), True)}"
                  if long.get(n) is not None else "—" for n in (1, 4)]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
