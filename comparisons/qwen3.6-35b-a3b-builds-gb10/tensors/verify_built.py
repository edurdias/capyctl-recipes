#!/usr/bin/env python3
"""Compare serialized-checkpoint scoring with each corresponding virtual ablation."""
import hashlib
import json
from pathlib import Path


def main():
    results = Path(__file__).resolve().parent/"results"
    rows = []
    for layout in ("delta", "combined"):
        virtual_path = results/f"score-{layout}.json"
        built_path = results/f"score-built-{layout}.json"
        virtual, built = (json.loads(p.read_text()) for p in (virtual_path, built_path))
        assert virtual["reference_sha256"] == built["reference_sha256"]
        assert [(r["id"], r["picked_ids"]) for r in virtual["records"]] == [
            (r["id"], r["picked_ids"]) for r in built["records"]], layout
        assert virtual["summary"] == built["summary"], layout
        rows.append({"layout": layout, "all_picked_token_ids_identical": True,
                     "summary": built["summary"], "reference_sha256": built["reference_sha256"],
                     "virtual_score_sha256": hashlib.sha256(virtual_path.read_bytes()).hexdigest(),
                     "built_score_sha256": hashlib.sha256(built_path.read_bytes()).hexdigest()})
    report = {"passed": True, "method": "Actual saved checkpoint weights loaded and reconstructed; every teacher-forced picked ID compared against the virtual ablation", "variants": rows}
    (results/"built-equivalence.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
