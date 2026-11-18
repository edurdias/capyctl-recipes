#!/usr/bin/env python3
"""Compare every shared NVFP4 projection against pinned BF16, including fused experts.

Supports ModelOpt's weight/weight_scale_2 and compressed-tensors' RedHat
weight_packed/weight_global_scale (the latter stores the reciprocal global scale).
Relative family error is sqrt(sum squared error / sum squared BF16 weight).
No inference engine is imported or started.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import time

import torch
from safetensors import safe_open

from convert_gemma4_26b_a4b_nvfp4 import headers
from memory_guard import MemoryGuard

E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
E2M1_MID = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
MULTS = [0.60 + .025 * i for i in range(21)]
EXPERT = re.compile(r"^(.*\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")


class Reader:
    def __init__(self, directory, device="cuda"):
        self.directory, self.device = Path(directory), device
        self.meta = headers(self.directory)
        index_path = self.directory / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            assert set(self.meta) == set(index["weight_map"]), f"Incomplete checkpoint: {self.directory}"
            payload = sum(m["data_offsets"][1] - m["data_offsets"][0] for m in self.meta.values())
            assert payload == index["metadata"]["total_size"], f"Checkpoint payload mismatch: {self.directory}"
        self.opened = {}
        self.source_files = {}

    def handle(self, key):
        shard = self.meta[key]["shard"]
        if shard not in self.opened:
            self.opened[shard] = safe_open(self.directory / shard, framework="pt", device=self.device, backend="pread")
        return self.opened[shard]

    def get(self, key):
        return self.handle(key).get_tensor(key).to(self.device)

    def source_weight(self, key):
        if key in self.meta:
            return self.get(key).float()
        match = EXPERT.fullmatch(key)
        if not match:
            raise KeyError(key)
        prefix, expert, proj = match.groups()
        fused = prefix + (".down_proj" if proj == "down" else ".gate_up_proj")
        meta = self.meta[fused]
        assert meta["dtype"] == "BF16", fused
        _, rows, cols = meta["shape"]
        first_row = rows // 2 if proj == "up" else 0
        selected_rows = rows if proj == "down" else rows // 2
        nbytes = selected_rows * cols * 2
        offset = meta["header_bytes"] + meta["data_offsets"][0] + (int(expert) * rows + first_row) * cols * 2
        # safetensors 0.8 pread get_slice reads the ENTIRE fused bank first.
        # These matrices are contiguous; read only the selected expert/half.
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=str(self.device).startswith("cuda"))
        if meta["shard"] not in self.source_files:
            self.source_files[meta["shard"]] = (self.directory / meta["shard"]).open("rb", buffering=0)
        file = self.source_files[meta["shard"]]
        file.seek(offset)
        assert file.readinto(memoryview(host.numpy())) == nbytes, f"Short source read: {key}"
        return host.view(torch.bfloat16).reshape(selected_rows, cols).to(self.device).float()

    def quantized_modules(self):
        return {k[:-len(".weight_scale")] for k in self.meta
                if k.endswith(".weight_scale") and
                (self.meta.get(k[:-len(".weight_scale")] + ".weight", {}).get("dtype") == "U8"
                 or k[:-len(".weight_scale")] + ".weight_packed" in self.meta)}

    def dequantize(self, module):
        if self.meta.get(module + ".weight", {}).get("dtype") in ("F8_E4M3", "F8_E4M3FN"):
            return self.get(module + ".weight").float() * self.get(module + ".weight_scale").float()
        compressed = module + ".weight_packed" in self.meta
        packed = self.get(module + (".weight_packed" if compressed else ".weight"))
        scale = self.get(module + ".weight_scale").float()
        global_scale = self.get(module + (".weight_global_scale" if compressed else ".weight_scale_2")).float()
        if compressed:
            global_scale = global_scale.reciprocal()
        return unpack_e2m1(packed) * scale.repeat_interleave(16, dim=-1) * global_scale


def unpack_e2m1(packed):
    lut = E2M1.to(packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1)
    values = lut[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values).flatten(-2)


def round_e2m1(x):
    return torch.sign(x) * E2M1.to(x.device)[torch.bucketize(x.abs().clamp(max=6.), E2M1_MID.to(x.device))]


def fp8(x):
    return x.clamp(min=0., max=448.).to(torch.float8_e4m3fn).float()


def family(name):
    if ".experts." in name:
        return "experts"
    if ".mlp." in name:
        return "shared_mlp"
    if ".self_attn." in name:
        return "attention"
    raise ValueError(name)


def energy(w, wq):
    # FP64 accumulation, bounded to one projection (roughly two million weights
    # per routed expert), rather than materializing an entire layer's fused bank.
    return (wq.double() - w.double()).square().sum().item(), w.double().square().sum().item()


def main(memory):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--ckpt", action="append", required=True, help="build_name=/path/to/checkpoint; repeat")
    p.add_argument("--out", required=True, type=Path, help="Per-tensor CSV; summary is adjacent .summary.json")
    a = p.parse_args()
    start = time.monotonic()
    assert torch.cuda.is_available(), "This measurement requires host A's GPU"
    torch.cuda.reset_peak_memory_stats()
    src = Reader(a.source)
    builds = {name: Reader(path) for name, path in (s.split("=", 1) for s in a.ckpt)}
    modules = {name: r.quantized_modules() for name, r in builds.items()}
    common = set.intersection(*modules.values())
    sums = defaultdict(lambda: {"squared_error": 0., "squared_weight": 0., "weights": 0, "tensors": 0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["build", "tensor", "family", "weights", "squared_error", "squared_weight", "relative_error", "common_all_builds"])
        writer.writeheader()
        union = sorted(set.union(*modules.values()))
        for i, module in enumerate(union):
            w = src.source_weight(module + ".weight")
            fam = family(module)
            for name, reader in builds.items():
                if module not in modules[name]:
                    continue
                wq = reader.dequantize(module)
                assert wq.shape == w.shape and torch.isfinite(wq).all(), module
                err, norm = energy(w, wq)
                writer.writerow({"build": name, "tensor": module + ".weight", "family": fam, "weights": w.numel(), "squared_error": err, "squared_weight": norm, "relative_error": (err / norm) ** .5, "common_all_builds": module in common})
                for subset in ("all_quantized", "common_all_builds") if module in common else ("all_quantized",):
                    s = sums[(name, fam, subset)]
                    s["squared_error"] += err
                    s["squared_weight"] += norm
                    s["weights"] += w.numel()
                    s["tensors"] += 1
                del wq
            del w
            if i % 128 == 0:
                print(f"[weights] {i + 1}/{len(union)} {module}", flush=True)
                f.flush()
    summary = [{"build": name, "family": fam, "subset": subset, **s, "relative_error": (s["squared_error"] / s["squared_weight"]) ** .5}
               for (name, fam, subset), s in sorted(sums.items())]
    report = {"definition": "sqrt(sum((Wq-Wbf16)^2)/sum(Wbf16^2)); no activation or task accuracy claim", "elapsed_seconds": time.monotonic() - start, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "common_quantized_modules": len(common), "families": summary}
    report["memory_guard"] = memory.metrics()
    a.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with MemoryGuard() as memory:
        main(memory)
