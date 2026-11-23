#!/usr/bin/env python3
"""Summarize captured outputs without claiming a missing BF16 comparison."""
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "capyctl-bench"))
import capyctl_bench as bench


def main():
    root = HERE / "results"
    tag = os.environ.get("CLEF_TAG", "laptop-fp8-sglang0522")
    cohorts = {}
    for name in ("diagnostic", "heldout"):
        rows = [json.loads(line) for line in (root / f"{tag}-{name}.jsonl").read_text().splitlines()]
        questions = errors = 0
        for row in rows:
            errors += bool(row["error"])
            for qid, answer in row.get("answers", {}).items():
                p = {"true": answer["noul"], "false": 1 - answer["noul"]} if answer["type"] == "noul" else answer["probabilities"]
                assert p and all(0 <= x <= 1 for x in p.values()) and abs(sum(p.values()) - 1) < 1e-4, (row["id"], qid, p)
                questions += 1
        cohorts[name] = {"requests": len(rows), "answered_questions": questions, "errors": errors,
                         "latency_p50_ms": bench.percentile([r["latency_s"] * 1000 for r in rows if not r["error"]], 50),
                         "latency_p95_ms": bench.percentile([r["latency_s"] * 1000 for r in rows if not r["error"]], 95)}
    groups = {}
    for rnd in range(1, 4):
        for point in json.loads((root / f"{tag}-latency-round{rnd}.json").read_text()):
            key = (point["questions"], point["target_total_tokens"])
            groups.setdefault(key, []).append(point)
    latency = []
    for (qn, target), points in groups.items():
        reqs = [r for p in points for r in p["requests"]]
        tokens = [r["input_tokens"] for r in reqs]
        latency.append({"questions": qn, "target_total_tokens": target, "timed_requests": len(reqs),
                        "actual_total_tokens_min": min(tokens), "actual_total_tokens_median": statistics.median(tokens), "actual_total_tokens_max": max(tokens),
                        "pooled_summary": bench.summarize_decisions(reqs),
                        "rounds": [{"round": p["round"], "summary": p["summary"]} for p in points]})
    levels = {}
    for rnd in range(1, 4):
        for level in json.loads((root / f"{tag}-throughput-round{rnd}.json").read_text()):
            levels.setdefault(level["concurrency"], []).append(level["summary"])
    throughput = [{"concurrency": c, "median_requests_per_s": statistics.median(r["requests_per_s"] for r in rounds),
                   "median_questions_per_s": statistics.median(r["questions_per_s"] for r in rounds), "rounds": rounds} for c, rounds in levels.items()]
    report = {"reference_status": "BF16 agreement is recorded separately by ordered capture comparison",
              "cohorts": cohorts, "latency": latency, "throughput": throughput,
              "memory_and_startup": json.loads((root / f"{tag}-run.json").read_text())}
    (root / f"{tag}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
