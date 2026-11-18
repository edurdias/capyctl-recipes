#!/usr/bin/env python3
"""Summarize ablations, reusing measured NVIDIA and full-r2 baseline scores."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablate

RESULTS = HERE / "results"
LABELS = {
    "A": "A: r2 experts; BF16 dense/attention",
    "B": "B: r2 experts+dense; BF16 attention",
    "C": "C: r2 experts+attention; BF16 dense",
    "D": "D: r2 experts+dense; FP8 tensor attention",
    "D-channel": "D-channel: r2 experts+dense; FP8 channel attention",
    "E": "E: r2 experts; FP8 tensor dense/attention",
    "G": "G: E + per-channel FP8 head",
    "H": "H: E + searched NVFP4 head",
    "I": "I: A + per-channel FP8 head",
    "F": "F: NVIDIA experts; BF16 rest (reused)",
    "r2": "Full r2 NVFP4 baseline (reused)",
}
ORDER = ["A", "B", "C", "D", "D-channel", "E", "G", "H", "I", "F", "r2"]
models = Path.home() / "models"
source_meta = ablate.common.Reader(models / ablate.common.BUILDS["bf16"][0]).meta
for label, original in [("F", "nvidia"), ("r2", "r2")]:
    report = json.loads((HERE.parent / f"results/score-{original}.json").read_text())
    quant_meta = ablate.common.Reader(models / ablate.common.BUILDS[original][0]).meta
    policy = {"dense_mlp": "bf16", "attention": "bf16"} if label == "F" else {"dense_mlp": "nvfp4", "attention": "nvfp4"}
    report.update({"variant": label, "reused_from": f"../../results/score-{original}.json",
                   "policy": {"experts": f"{original}_nvfp4", **policy, "router": "bf16"},
                   "byte_arithmetic": ablate.byte_arithmetic(source_meta, quant_meta, policy)})
    ablate.common.save(RESULTS / f"score-{label}.json", report)
reports = [json.loads((RESULTS / f"score-{v}.json").read_text()) for v in ORDER if (RESULTS / f"score-{v}.json").exists()]
r2_meta = ablate.common.Reader(models / ablate.common.BUILDS["r2"][0]).meta
nvidia_meta = ablate.common.Reader(models / ablate.common.BUILDS["nvidia"][0]).meta
for report in reports:
    report["byte_arithmetic"] = ablate.byte_arithmetic(
        source_meta, nvidia_meta if report["variant"] == "F" else r2_meta, report["policy"])
assert len({r["reference_sha256"] for r in reports}) == 1
assert all(r["summary"]["positions"] == 62658 for r in reports)
assert all([(x["id"], x["n"]) for x in r["records"]] == [(x["id"], x["n"]) for x in reports[0]["records"]] for r in reports)
lines = ["# Family ablation: 62,658 shared BF16 answer positions, 200 prompts", "",
         "All new rows use r2 experts; all unmentioned weights are BF16 source weights. F and full r2 reuse the original measurements.", "",
         "| Variant | Agreement % | Median first disagreement | Chat % | Coding % | Math % | Multilingual % | Tool-use % |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in reports:
    s = r["summary"]
    cells = [LABELS[r["variant"]], f"{100*s['agreement']:.3f}", f"{s['median_first_disagreement']:g}"]
    cells += [f"{100*s['by_category'][c]:.3f}" for c in ["chat", "coding", "math", "multilingual", "tool-use"]]
    lines.append("| " + " | ".join(cells) + " |")
lines += ["", "[INFERENCE] Weight/weight-scale bytes read per decode token, arithmetic only: attention + dense MLP + eight routed experts per layer + router + output head. Input embeddings remain BF16; only their lookup row is needed, not a second full matrix read. Attention Q/K norms and router scales are included. NVFP4 counts packed weights plus block/global weight scales; FP8 counts one byte/weight plus its scales. The expert term is bank payload times 8/128.", "",
          "| Variant | Attention B/token | Dense MLP B/token | Eight experts B/token | Router B/token | Family sum B/token | Output head B/token | Total incl head B/token | Total incl head GiB/token |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in reports:
    b = r["byte_arithmetic"]
    cells = [r["variant"]] + [f"{b['bytes_per_token'][f]:,}" for f in ["attention", "dense_mlp", "routed_experts_top8", "router"]]
    cells += [f"{b['total_bytes_per_token']:,}", f"{b['output_head_bytes_per_token']:,}",
              f"{b['total_including_output_head_bytes_per_token']:,}",
              f"{b['total_including_output_head_bytes_per_token']/2**30:.6f}"]
    lines.append("| " + " | ".join(cells) + " |")
channel_heads = [r for r in reports if r["policy"].get("output_head") == "fp8_channel"]
if channel_heads:
    lines += ["", "[INFERENCE] Compatible engines allocate the channel head scales in FP32 rather than this simulation's BF16. Add 524,288 bytes per token to G/I's raw payload totals; loaded-FP32-scale totals: " +
              "; ".join(f"{r['variant']} {r['byte_arithmetic']['total_including_head_if_channel_scales_loaded_fp32']:,} B/token" for r in channel_heads) + "."]
lines += ["", "These are not measured memory transactions or tokens/second. The totals now include the output head (BF16 except G/H/I), but omit other layer state, the embedding lookup row, activations, KV-cache traffic, temporary buffers and batching/cache reuse. In the offline accuracy runs every variant is reconstructed to BF16, so actual torch weight traffic is not this packed-format arithmetic.", "",
          "| New variant | Load s | Evaluation s | Wall s | Peak CUDA allocated GiB | Peak CUDA reserved GiB | Peak RSS GiB | Minimum host available GiB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in reports:
    if r.get("reused_from"):
        continue
    cells = [r["variant"]] + [f"{r[k]:.2f}" for k in ["load_seconds", "evaluation_seconds", "wall_seconds"]]
    cells += [f"{r[k]/2**30:.3f}" for k in ["cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_rss_bytes"]]
    cells += [f"{r['memory_guard']['minimum_observed_available_bytes']/2**30:.3f}"]
    lines.append("| " + " | ".join(cells) + " |")
lines += [""]
text = "\n".join(lines)
(RESULTS / "summary.md").write_text(text)
ablate.common.save(RESULTS / "summary.json", {r["variant"]: {k: v for k, v in r.items() if k not in ("records", "fp8_simulation")} for r in reports})
print(text)
