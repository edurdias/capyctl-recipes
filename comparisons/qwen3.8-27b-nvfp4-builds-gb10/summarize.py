#!/usr/bin/env python3
"""Summarize run-host.sh results directories: accuracy against the BF16 source, startup.

usage: summarize.py <results dir> [<results dir> ...]   (prints Markdown tables)

A file is read from the first directory that has it, so run 1 (with the BF16 reference)
and run 2 combine. Agreement is b18.py's: exact-match rate of greedy replies against the
BF16 run, by prompt id, and where the first divergence falls. GSM8K and tool calls are as
b18.py scores them; "finished" leaves out answers cut at the token limit.
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "b18", HERE.parent / "qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10" / "b18.py")
b18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b18)

dirs = [Path(p) for p in sys.argv[1:]]


def load(name):
    for d in dirs:
        if (d / name).exists():
            return json.loads((d / name).read_text())
    return None


names = {"sg-bf16": "BF16 (Qwen/Qwen3.8-27B)", "sg-nv": "nvidia/Qwen3.8-27B-NVFP4",
         "sg-full": "capyctl/Qwen3.8-27B-NVFP4",
         "sg-gt": "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090", "sg-qs": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
         "sg-rh": "RedHatAI/Qwen3.8-27B-INT4", "sg-un": "unsloth/Qwen3.8-27B-NVFP4",
         "vl-gt": "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 (vLLM)",
         "vl-qs": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 (vLLM)",
         "vl-rh": "RedHatAI/Qwen3.8-27B-INT4 (vLLM)", "vl-un": "unsloth/Qwen3.8-27B-NVFP4 (vLLM)",
         "cu-mia-df": "Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw (exllamav3, DFlash2)"}

print("| Checkpoint (SGLang 0.5.22 unless noted, no drafts) | GSM8K | cut at the limit | GSM8K, finished answers "
      "| tool calls | agreement with BF16 (200 prompts, greedy) |")
print("|---|---|---|---|---|---|")
ref = load("agree-sg-bf16.json")
for k, label in names.items():
    g, t, a = load(f"gsm8k-{k}.json"), load(f"tools-{k}.json"), load(f"agree-{k}.json")
    if not (g or t or a):
        continue
    gs = g["summary"] if g else {}
    ts = t["summary"] if t else {}
    fin = "-"
    if g:
        done = [r for r in g["records"] if r["finish_reason"] != "length"]
        fin = f"{sum(r['correct'] for r in done)}/{len(done)} ({100 * sum(r['correct'] for r in done) / len(done):.1f}%)"
    agr = "reference" if k == "sg-bf16" else "-"
    if ref and a and k != "sg-bf16":
        s = b18.agreement(a, ref)
        tok = s.get("first_divergence_tokens") or {}
        agr = f"{s['exact']}/{s['n']} identical" + (f"; first difference at token {tok['median']} (median)" if tok else "")
    gsm = f"{gs['correct']}/{gs['n']} ({gs['accuracy'] * 100:.1f}%)" if gs else "-"
    print(f"| {label} | {gsm} | {gs.get('truncated', '-')} | {fin} | {ts.get('pass', '-')}/{ts.get('n', '-')} | {agr} |")

print()
print("| Variant | Ready (s) | MemAvailable drop (GiB) |")
print("|---|---|---|")
for d in dirs:
    tsv = d / "startup.tsv"
    for line in tsv.read_text().splitlines() if tsv.exists() else []:
        k, ready, drop = line.split("\t")
        print(f"| {k} | {ready} | {drop} |")
