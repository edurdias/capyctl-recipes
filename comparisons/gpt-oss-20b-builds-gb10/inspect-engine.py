#!/usr/bin/env python3
"""Read-only package fingerprints and exact installed speculative-loader evidence."""
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import re
import sys

for name in ("sglang", "vllm", "torch", "flashinfer-python", "transformers", "lmcache"):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    print(json.dumps({"package": name, "version": dist.version,
                      "direct_url": dist.read_text("direct_url.json")}))
roots = [pathlib.Path(p) for p in sys.path if p.endswith("site-packages")]
names = (
    "sglang/srt/server_args.py", "sglang/srt/speculative/spec_info.py",
    "sglang/srt/models/gpt_oss.py", "sglang/srt/layers/quantization/mxfp4.py",
    "sglang/srt/models/dspark.py", "sglang/srt/models/registry.py",
    "sglang/srt/arg_groups/speculative_hook.py", "sglang/srt/utils/hf_transformers_utils.py",
    "sglang/srt/speculative/dflash_worker_v2.py",
    "vllm/config/speculative.py", "vllm/model_executor/models/registry.py",
    "vllm/model_executor/models/gpt_oss.py", "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py",
    "vllm/v1/spec_decode/dflash.py",
)
pattern = re.compile(r"mxfp4|marlin|flashinfer|sm12|SM12|triton_kernel|bias|swiglu|gpt_oss|gpt-oss|dflash|eagle3|spec_accept", re.I)
for root in roots:
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        print(json.dumps({"file": name, "sha256": hashlib.sha256(data).hexdigest()}))
        lines = data.decode().splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if pattern.search(line):
                selected.update(range(max(0, i - 2), min(len(lines), i + 4)))
        for i in sorted(selected):
            print(f"{i + 1}:{lines[i]}")
