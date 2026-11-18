#!/usr/bin/env python3
"""Create the pooled agreement table and resource table from real run JSON."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ORDER = ["bf16", "r1", "r2", "r3", "r3e", "nvidia", "redhat", "unsloth"]
CATEGORIES = ["chat", "coding", "math", "multilingual", "tool-use"]
LABELS = {"bf16": "BF16 (by construction)", "r1": "CapyCTL r1", "r2": "CapyCTL r2", "r3": "CapyCTL r3 (FP8 dense/attention/head)", "r3e": "CapyCTL r3e (FP8 dense/attention; BF16 head)", "nvidia": "NVIDIA", "redhat": "RedHat", "unsloth": "Unsloth (FP8 attention)"}
reports = [json.loads((RESULTS / f"score-{b}.json").read_text()) for b in ORDER if (RESULTS / f"score-{b}.json").exists()]
if not reports:
    raise SystemExit("No completed reports")
assert len({r["reference_sha256"] for r in reports}) == 1, "Different reference positions"
assert len({r["summary"]["positions"] for r in reports}) == 1, "Different position totals"
assert all([(x["id"], x["n"]) for x in r["records"]] == [(x["id"], x["n"]) for x in reports[0]["records"]] for r in reports)
lines = [f"Greedy agreement on {reports[0]['summary']['positions']:,} identical BF16 answer positions ({len(reports[0]['records'])} prompts).", "",
         "| Build | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in reports:
    s = r["summary"]
    cells = [LABELS[r["build"]], f"{100*s['agreement']:.3f}", f"{s['median_first_disagreement']:g}"]
    cells += [f"{100*s['by_category'][c]:.3f}" for c in CATEGORIES]
    lines.append("| " + " | ".join(cells) + " |")
lines += ["", "First disagreement is zero-based; a fully agreeing answer is censored at its answer length. BF16's value is therefore its median answer length, not a measured mismatch.", "",
          "| Build | Load s | Evaluation s | Wall s | Peak CUDA allocated GiB | Peak CUDA reserved GiB | Peak RSS GiB | Minimum host available GiB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in reports:
    cells = [LABELS[r["build"]]] + [f"{r[k]:.2f}" for k in ["load_seconds", "evaluation_seconds", "wall_seconds"]]
    cells += [f"{r[k]/2**30:.3f}" for k in ["cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_rss_bytes"]]
    cells += [f"{r['memory_guard']['minimum_observed_available_bytes']/2**30:.3f}"]
    lines.append("| " + " | ".join(cells) + " |")
lines += ["", "Wall times are measured inside each Python command (load plus evaluation); /usr/bin/time whole-command timings and exit codes are in the matching logs. Host-wide memory includes another workload's engine and filesystem cache; it is not the model's RSS.", ""]
text = "\n".join(lines)
(RESULTS / "summary.md").write_text(text)
(RESULTS / "summary.json").write_text(json.dumps({r["build"]: {k: v for k, v in r.items() if k != "records"} for r in reports}, indent=2) + "\n")
print(text)
