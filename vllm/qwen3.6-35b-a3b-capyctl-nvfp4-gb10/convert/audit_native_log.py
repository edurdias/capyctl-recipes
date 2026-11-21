#!/usr/bin/env python3
"""Require actual packed NVFP4 execution for every changed fused dense module."""
import argparse
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--layout", required=True, choices=["delta", "attention", "combined"])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    text = a.log.read_text()
    assert "QWEN36-LOAD-DONE" in text, "Native process did not complete the short-prompt check"
    records = []
    for line in text.splitlines():
        marker = "QWEN36-NATIVE-DENSE "
        if marker in line:
            records.append(json.JSONDecoder().raw_decode(line.split(marker, 1)[1])[0])
    expected = set()
    for layer in range(40):
        if layer % 4 != 3 and a.layout in ("delta", "combined"):
            expected.update((layer, "linear_attn", p) for p in ("in_proj_qkvz", "out_proj"))
        if layer % 4 == 3 and a.layout in ("attention", "combined"):
            expected.update((layer, "self_attn", p) for p in ("qkv_proj", "o_proj"))
    found = {}
    for record in records:
        match = re.search(r"\.layers\.(\d+)\.(linear_attn|self_attn)\.(\w+)$", record["prefix"] or "")
        if not match:
            continue
        key = (int(match[1]), match[2], match[3])
        if key not in expected:
            continue
        assert record["successful_apply"], record
        assert record["is_w4a16_nvfp4"], record
        assert "marlin" in record["kernel"].lower(), record
        assert record["weight_dtype"] in ("torch.int32", "torch.uint8"), record
        found[key] = record
    assert set(found) == expected, {"missing": sorted(expected-set(found)), "found": sorted(found)}
    dtype = re.search(r"kv_cache_dtype=(torch\.[a-z0-9_]+)", text)
    if dtype is None:
        dtype = re.search(r"KV Cache[^\n]*dtype: (torch\.[a-z0-9_]+)", text)
    assert dtype is not None, "Missing actual KV dtype in native runtime logs"
    assert dtype[1] == "torch.float8_e4m3fn", dtype[1]
    report = {"passed": True, "layout": a.layout, "executed_changed_fused_dense_modules": len(found),
              "lmcache": "off", "requested_kv_dtype": "fp8_e4m3", "records": list(found.values()),
              "actual_kv_dtype": dtype[1], "short_prompt_completed": True}
    a.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}))

if __name__ == "__main__":
    main()
