#!/usr/bin/env python3
"""Report offline agreement and inferred logical batch-one target weight traffic."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import math
from weights import Reader, family


def traffic(reader, selected):
    groups = defaultdict(float)
    for name, meta in reader.meta.items():
        if name.startswith("mtp.") or ".mtp." in name or name.startswith("model.visual."):
            continue
        if name == "model.language_model.embed_tokens.weight":
            groups["embedding_row"] += meta["shape"][-1] * 2
            continue
        base = name.rsplit(".", 1)[0]
        kind = family(base + ".weight")
        if kind in selected:
            if name.endswith(".weight"):
                # U8 packed weights + one E4M3 scale per 16 weights + FP32 global.
                groups[kind] += math.prod(meta["shape"]) * 9/16 + 4
            continue
        size = meta["data_offsets"][1]-meta["data_offsets"][0]
        groups[kind] += size / 32 if kind == "routed_experts" else size
    return {"bytes_per_token": sum(groups.values()), "by_family": dict(sorted(groups.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=Path.home()/"models/qwen3.6-35b-a3b-nvidia-nvfp4")
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent/"results")
    a = ap.parse_args()
    reader = Reader(a.reference, "cpu")
    rows = []
    for label, selected in [("nvidia", set()), ("delta", {"deltanet"}), ("attention", {"attention"}),
                            ("combined", {"deltanet", "attention"})]:
        result = json.loads((a.results/f"score-{label}.json").read_text())
        rows.append({"variant": label, "agreement": result["summary"], "traffic": traffic(reader, selected),
                     "reference_sha256": result["reference_sha256"]})
    assert len({r["reference_sha256"] for r in rows}) == 1
    report = {"method": "Agreement against BF16-generated answers; all weights reconstructed as BF16; no activation quantization",
              "traffic_method": "[INFERENCE] Sum stored active tensor payloads, routed experts weighted 8/256; one embedding row; excludes MTP, vision, KV/state/activation traffic and cache reuse",
              "shared_experts": "Already NVFP4 in every layout; retained byte-identically",
              "quality_warning": "Diagnostic agreement is not GSM8K, tools, or HumanEval acceptance",
              "variants": rows}
    (a.results/"family-ablation.json").write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["variant"], f"{row['agreement']['agreement']:.6%}", f"{row['traffic']['bytes_per_token']:,.0f} bytes/token")

if __name__ == "__main__":
    main()
