#!/usr/bin/env python3
"""Patch NVIDIA's mixed layout with searched ModelOpt 0.47 NVFP4 families.

No full-model calibration is invented: existing expert/head bytes and all auxiliary
weights are retained exactly. New W4A16 projections need no activation calibration.
Their inherited FP8 input_scale is removed. ModelOpt packs source BF16 weights with
21 searched, FP8-rounded block scales. MTP remains byte-identical BF16.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
TENSORS = HERE.parents[2]/"comparisons/qwen3.6-35b-a3b-builds-gb10/tensors"
if not TENSORS.exists():
    TENSORS = Path.home()/"qwen36-work/tensors"
sys.path.insert(0, str(TENSORS))
from weights import Reader, SOURCE_REVISION, REFERENCE_REVISION, family, fused_global_scale, headers, search

def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()

def digest_tensor(directory, meta):
    result = hashlib.sha256()
    with (directory/meta["file"]).open("rb") as f:
        start, end = meta["data_offsets"]
        f.seek(meta["header_bytes"]+start)
        remaining = end-start
        while remaining:
            block = f.read(min(8 << 20, remaining))
            if not block:
                raise EOFError(meta)
            result.update(block)
            remaining -= len(block)
    return result.hexdigest()

def audit(checkpoint, out, changed):
    before, after = headers(checkpoint), headers(out)
    allowed = set()
    for base in changed:
        allowed.update(base+suffix for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"))
    assert set(before)-set(after) == {b+".input_scale" for b in changed}
    assert set(after)-set(before) == {b+".weight_scale_2" for b in changed}
    preserved = Counter()
    for name, meta in before.items():
        if name in allowed:
            continue
        assert meta["dtype"] == after[name]["dtype"] and meta["shape"] == after[name]["shape"], name
        assert digest_tensor(checkpoint, meta) == digest_tensor(out, after[name]), name
        preserved[family(name)] += 1
    reader = Reader(out)
    for base in changed:
        assert after[base+".weight"]["dtype"] == "U8", base
        assert after[base+".weight_scale"]["dtype"] == "F8_E4M3", base
        assert after[base+".weight_scale_2"]["dtype"] == "F32", base
        assert after[base+".weight"]["shape"][-1]*2 == before[base+".weight"]["shape"][-1], base
        for suffix in (".weight_scale", ".weight_scale_2"):
            value = reader.get(base+suffix).float()
            assert torch.isfinite(value).all() and (value > 0).all(), base+suffix
    return {"passed": True, "changed_matrices": len(changed), "byte_identical_preserved_tensors": dict(preserved),
            "mtp_bf16": all(m["dtype"] == "BF16" for n,m in after.items() if family(n) == "mtp"),
            "payload_bytes": sum(m["data_offsets"][1]-m["data_offsets"][0] for m in after.values())}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout", choices=["delta", "attention", "combined"])
    ap.add_argument("--source", type=Path, default=Path.home()/"models/qwen3.6-35b-a3b")
    ap.add_argument("--base", type=Path, default=Path.home()/"models/qwen3.6-35b-a3b-nvidia-nvfp4")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    assert importlib.metadata.version("nvidia-modelopt") == "0.47.0"
    assert not a.out.exists(), "Refuse to overwrite an existing checkpoint"
    a.out.mkdir(parents=True)
    selected = {"delta": {"deltanet"}, "attention": {"attention"}, "combined": {"deltanet", "attention"}}[a.layout]
    started = time.monotonic()
    source = Reader(a.source)
    base_metadata = headers(a.base)
    changed = sorted(n[:-7] for n in base_metadata if n.endswith(".weight") and family(n) in selected)
    errors = []
    global_scales = {}
    for path in sorted(a.base.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            tensors = {n: f.get_tensor(n) for n in f.keys()}
        for base in changed:
            if base+".weight" not in tensors:
                continue
            w = source.get(base+".weight")
            with torch.inference_mode():
                packed, scale, global_scale, error = search(w, fused_global_scale(base+".weight", source, global_scales))
            tensors[base+".weight"] = packed.cpu()
            tensors[base+".weight_scale"] = scale.cpu()
            tensors[base+".weight_scale_2"] = global_scale.float().cpu().reshape(())
            del tensors[base+".input_scale"]
            errors.append({"tensor": base+".weight", "relative_error": error})
            del w, packed, scale, global_scale
            print(f"quantized {len(errors)}/{len(changed)} {base} rel={error:.6f}", flush=True)
        save_file(tensors, a.out/path.name, metadata=metadata)
        del tensors
    # Copy only known model/processor metadata, never logs, locks or runtime state.
    for path in a.base.iterdir():
        if path.is_file() and (path.suffix in (".json", ".jinja", ".txt", ".model") or path.name == "merges.txt"):
            if path.name not in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
                shutil.copyfile(path, a.out/path.name)
    config = json.loads((a.base/"config.json").read_text())
    hf = json.loads((a.base/"hf_quant_config.json").read_text())
    hf["producer"] = {"name": "modelopt", "version": "0.47.0"}
    for mapping in (hf["quantization"]["quantized_layers"], config["quantization_config"]["quantized_layers"]):
        for base in changed:
            mapping[base] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    qconfig = config["quantization_config"]
    for group in qconfig["config_groups"].values():
        group["targets"] = [n for n in group["targets"] if n not in changed]
    qconfig["config_groups"]["capyctl_w4a16"] = {
        "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
        "input_activations": None, "targets": changed}
    qconfig["producer"] = hf["producer"]
    for filename, data in (("config.json", config), ("hf_quant_config.json", hf)):
        (a.out/filename).write_text(json.dumps(data, indent=2)+"\n")
    meta = headers(a.out)
    index = {"metadata": {"total_size": sum(m["data_offsets"][1]-m["data_offsets"][0] for m in meta.values())},
             "weight_map": {n:m["file"] for n,m in sorted(meta.items())}}
    (a.out/"model.safetensors.index.json").write_text(json.dumps(index, indent=2)+"\n")
    evidence = audit(a.base, a.out, changed)
    manifest = {"source": "Qwen/Qwen3.6-35B-A3B", "source_revision": SOURCE_REVISION,
                "base": "nvidia/Qwen3.6-35B-A3B-NVFP4", "base_revision": REFERENCE_REVISION,
                "layout": a.layout, "method": "ModelOpt NVFP4QTensor 0.47.0 packing; block MSE search; W4A16",
                "block_size": 16, "scale_multipliers": [0.60+0.025*i for i in range(21)],
                "calibration": "weights only; no new activation quantizers; inherited experts/head unchanged",
                "preserved": "all untargeted tensors byte-identical, including MTP, vision, experts/head, routers, a/b, conv and norms",
                "audit": evidence, "tensor_errors": errors, "wall_seconds": time.monotonic()-started,
                "software": {"torch": torch.__version__, "nvidia-modelopt": importlib.metadata.version("nvidia-modelopt")},
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(a.out.iterdir()) if p.is_file()}}
    (a.out/"conversion-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(evidence), flush=True)

if __name__ == "__main__":
    main()
