#!/usr/bin/env python3
"""Print recipe throughput and TTFT against production; never change evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = (1, 2, 4, 8)
LABELS = {
    "sg522-default": "Baseline: OpenAI, SGLang 0.5.22, production",
    "sg522-marlin": "OpenAI, SGLang 0.5.22, Marlin",
    "vl032-auto": "OpenAI, vLLM 0.32.0, Marlin",
    "vl033-auto": "OpenAI, vLLM 0.33.0, Marlin",
    "llama-b12242": "ggml-org, llama.cpp b12242, no drafter",
    "llama-b12242-eagle3": "ggml-org, llama.cpp EAGLE3 K3 (rejected)",
    "sg522-dflash8": "OpenAI, SGLang DFlash8",
    "sg522-dflash4": "OpenAI, SGLang DFlash4",
    "sg522-dflash6": "OpenAI, SGLang DFlash6",
    "sg522-dflash12": "OpenAI, SGLang DFlash12 (screen)",
    "sg522-dflash16": "OpenAI, SGLang DFlash16 (screen)",
}
EXTRA = (
    ("capyctl", "CapyCTL, vLLM 0.32.0, no drafter (recipe)", "full"),
    ("capyctl-triton", "CapyCTL, vLLM Triton attention, no drafter", "triton-baseline"),
    ("capyctl-dflash8", "CapyCTL, vLLM Triton + DFlash8", "tri-dflash8"),
    ("capyctl-nebius3", "CapyCTL, vLLM Triton + Nebius3 (rejected)", "tri-nebius3"),
    ("capyctl-nebius5", "CapyCTL, vLLM Triton + Nebius5 (rejected)", "tri-nebius5"),
)


def read_json(path):
    return json.loads(path.read_text())


def from_measure(key, label, data):
    return {"key": key, "label": label,
            "concurrency": {str(p["streams"]): {k: p[k] for k in ("aggregate_tps", "decode_tps", "ttft_s")}
                            for p in data.get("concurrency", [])},
            "ttft24k": {str(p["streams"]): p["ttft_s"]
                         for p in (data.get("ttft24k") or {}).get("levels", [])}}


def rows(matrix, root=HERE):
    out = [dict(r, label=LABELS.get(r["key"], r["key"]))
           for r in matrix["rows"] if r.get("concurrency")]
    for key, label, directory in EXTRA:
        out.append(from_measure(key, label, read_json(root / "bench/fp8-attention" / directory / "measure.json")))
    public = read_json(root / "bench/public-builds/pub-lms-fi-001/evidence.json")
    out.append(from_measure("lmsys", "LMSYS, vLLM 0.32.0, BF16 loaded MoE", public["measure.json"]))
    return out


def change(value, baseline, lower=False):
    pct = 100 * (value / baseline - 1)
    loss = pct > 0 if lower else pct < 0
    signed = f"{pct:+.1f}" if abs(pct) < 10 else f"{pct:+.0f}"
    return f"{signed}%{' ▼' if loss else ''}"


def markdown(matrix, root=HERE):
    measured = rows(matrix, root)
    baseline = next(r for r in measured if r["key"] == "sg522-default")
    lines = ["### Every measured setup against the production baseline", "",
             "Total completion tokens / round wall time, median of three rounds. The 1-stream column is aggregate, not post-first-token decode. ▼ marks a regression; — means not measured.", "",
             "| Setup | 1 stream | 2 streams | 4 streams | 8 streams |",
             "|---|---:|---:|---:|---:|"]
    for row in measured:
        cells = []
        for n in LEVELS:
            p = row["concurrency"].get(str(n))
            value = p["aggregate_tps"] if p else None
            base = baseline["concurrency"][str(n)]["aggregate_tps"]
            cells.append("—" if value is None else f"{value:.1f}" +
                         ("" if row is baseline else f" ({change(value, base)})"))
        lines.append("| " + " | ".join([row["label"]] + cells) + " |")
    lines += ["", "Time to first reasoning or content token; lower is better. Short prompts are the fixed chat/code corpus; 24k inputs are a separate exact-token workload.", "",
              "| Setup | 1 stream, ms | 2 streams, ms | 4 streams, ms | 8 streams, ms | 24k, 1 stream, s | 24k, 4 streams, s |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in measured:
        cells = []
        for n in LEVELS:
            p = row["concurrency"].get(str(n))
            value = p["ttft_s"] if p else None
            base = baseline["concurrency"][str(n)]["ttft_s"]
            cells.append("—" if value is None else f"{value * 1000:.0f}" +
                         ("" if row is baseline else f" ({change(value, base, True)})"))
        for n in (1, 4):
            value = row["ttft24k"].get(str(n))
            base = baseline["ttft24k"][str(n)]
            cells.append("—" if value is None else f"{value:.3f}" +
                         ("" if row is baseline else f" ({change(value, base, True)})"))
        lines.append("| " + " | ".join([row["label"]] + cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(read_json(HERE / "bench/final-matrix.json")))
