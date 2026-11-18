#!/usr/bin/env python3
"""Revision 2: re-round every NVFP4 projection using BF16 block-scale MSE search.

Preserve global weight/input scales, all BF16 tensors, configs and assets. Search
21 multipliers (0.60..1.10 in 0.025 steps) of each block's max scale, plus its
original revision-1 scale; FP8-round each candidate before scoring. This is a
finite candidate search, not a claim of a globally optimal continuous scale.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import read_tensors as rt
from convert_gemma4_26b_a4b_nvfp4 import audit, headers, save_manifest, sha256, tensor_digest
from memory_guard import MemoryGuard


def requant(w, global_scale, original_scales, chunk=1 << 18):
    rows, cols = w.shape
    blocks = w.reshape(-1, 16)
    codes = torch.empty_like(blocks, dtype=torch.uint8)
    scales = torch.empty(blocks.shape[0], device=w.device)
    mid = rt.E2M1_MID.to(w.device)
    for first in range(0, blocks.shape[0], chunk):
        b = blocks[first:first + chunk]
        s0 = b.abs().amax(dim=1) / 6. / global_scale
        old = original_scales.reshape(-1)[first:first + chunk].float()
        best = torch.full_like(s0, float("inf"))
        chosen = torch.zeros_like(s0)
        for mult in [None, *rt.MULTS]:
            s = (old if mult is None else rt.fp8(s0 * mult)).clamp_min(1e-30)
            dequant = rt.round_e2m1(b / (s * global_scale)[:, None]) * (s * global_scale)[:, None]
            err = (dequant - b).square().sum(dim=1)
            take = err < best
            best = torch.where(take, err, best)
            chosen = torch.where(take, s, chosen)
        x = b / (chosen * global_scale)[:, None]
        mag = torch.bucketize(x.abs().clamp(max=6.), mid).to(torch.uint8)
        sign = ((x < 0) & (mag > 0)).to(torch.uint8)
        codes[first:first + chunk] = mag | (sign << 3)
        scales[first:first + chunk] = chosen
    nibbles = codes.reshape(rows, cols)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return packed.contiguous(), scales.reshape(rows, cols // 16).to(torch.float8_e4m3fn)


def main(memory):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--in-place", action="store_true", help="Search only stored U8 matrices in an audited mixed export")
    a = p.parse_args()
    assert torch.cuda.is_available(), "Run in the existing host A ModelOpt environment"
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    a.out.mkdir(parents=True, exist_ok=True)
    assert a.in_place or not list(a.out.glob("*.safetensors")), "Output checkpoint already exists"
    assert not a.in_place or a.ckpt.resolve() == a.out.resolve(), "--in-place requires identical input/output paths"
    src = rt.Reader(a.source)
    before_meta = headers(a.ckpt)
    checkpoint = rt.Reader(a.ckpt, device="cpu")
    pending_scales = defaultdict(dict)
    boundary_scale_updates = 0
    original_manifest = json.loads((a.ckpt / "conversion-manifest.json").read_text())
    mixed = original_manifest["revision"] == 3
    assert original_manifest["revision"] in (1, 3) and original_manifest["layout_audit"]["passed"]
    assert not a.in_place or mixed, "In-place search is for mixed revision-3 exports only"
    parent_hash = sha256(a.ckpt / "conversion-manifest.json")
    preserved = {k: tensor_digest(a.ckpt, v) for k, v in before_meta.items()
                 if mixed and ".experts." not in k}
    expected_count = 11520 if mixed else 11725
    for f in a.ckpt.iterdir():
        if not a.in_place and f.is_file() and f.suffix != ".safetensors" and f.name not in ("conversion-manifest.json", "layout-audit.json", "memory-guard.json", "offline-sanity-load.json"):
            shutil.copyfile(f, a.out / f.name)
    sums = defaultdict(lambda: {"squared_weight": 0., "r1_squared_error": 0., "r2_squared_error": 0., "tensors": 0, "weights": 0})
    log = []
    for shard in sorted(a.ckpt.glob("*.safetensors")):
        with shard.open("rb") as f:
            import struct
            n = struct.unpack("<Q", f.read(8))[0]
            metadata = json.loads(f.read(n)).get("__metadata__")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensors = {k: handle.get_tensor(k) for k in handle.keys()}
            tensors.update(pending_scales.pop(shard.name, {}))
            for key in sorted(k for k in tensors if k.endswith(".weight") and tensors[k].dtype == torch.uint8):
                module = key[:-7]
                assert not mixed or ".experts." in module, key
                global_key = module + ".weight_scale_2"
                scale_key = module + ".weight_scale"
                g = (tensors[global_key] if global_key in tensors else checkpoint.get(global_key)).float().item()
                w = src.source_weight(key)
                old_scale = (tensors[scale_key] if scale_key in tensors else checkpoint.get(scale_key)).cuda()
                old = rt.unpack_e2m1(tensors[key].cuda()) * old_scale.float().repeat_interleave(16, dim=-1) * g
                old_err, norm = rt.energy(w, old)
                del old
                packed, scales = requant(w, g, old_scale)
                decoded = rt.unpack_e2m1(packed) * scales.float().repeat_interleave(16, dim=-1) * g
                new_err, _ = rt.energy(w, decoded)
                assert torch.isfinite(decoded).all() and new_err <= old_err * (1. + 1e-5) + 1e-12, key
                tensors[key] = packed.cpu()
                scale_shard = before_meta[scale_key]["shard"]
                if scale_shard == shard.name:
                    tensors[scale_key] = scales.cpu()
                else:
                    pending_scales[scale_shard][scale_key] = scales.cpu()
                    boundary_scale_updates += 1
                log.append({"tensor": key, "family": rt.family(key), "weights": w.numel(), "squared_weight": norm, "r1_squared_error": old_err, "r2_squared_error": new_err})
                s = sums[rt.family(key)]
                s["squared_weight"] += norm
                s["r1_squared_error"] += old_err
                s["r2_squared_error"] += new_err
                s["weights"] += w.numel()
                s["tensors"] += 1
                if len(log) % 128 == 0:
                    print(f"[requant] {len(log)}/{expected_count} {key}", flush=True)
                del w, old_scale, packed, scales, decoded
            target = a.out / (shard.name + ".tmp" if a.in_place else shard.name)
            save_file(tensors, target, metadata=metadata)
            if a.in_place:
                os.replace(target, a.out / shard.name)
            del tensors
    # A weight and its block scale may straddle a 5-GB export boundary.
    # Only boundary scales need deferred writes; all other tensors stay bounded
    # to one shard and every tensor keeps its original index/shard assignment.
    for shard_name, updates in pending_scales.items():
        shard = a.out / shard_name
        with shard.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            metadata = json.loads(f.read(n)).get("__metadata__")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensors = {k: handle.get_tensor(k) for k in handle.keys()}
            tensors.update(updates)
            target = a.out / (shard_name + ".tmp")
            save_file(tensors, target, metadata=metadata)
            os.replace(target, shard)
            del tensors
    assert len(log) == expected_count, len(log)
    after_meta = headers(a.out)
    assert {k: (v["shape"], v["dtype"]) for k, v in before_meta.items()} == {k: (v["shape"], v["dtype"]) for k, v in after_meta.items()}
    assert all(tensor_digest(a.out, after_meta[k]) == v for k, v in preserved.items()), "Non-expert/FP8 tensor changed during scale search"
    layout = audit(a.source, a.out, original_manifest.get("layout", "all-nvfp4"))
    for s in sums.values():
        s["r1_relative_error"] = (s["r1_squared_error"] / s["squared_weight"]) ** .5
        s["r2_relative_error"] = (s["r2_squared_error"] / s["squared_weight"]) ** .5
    (a.out / "requant-scales.json").write_text(json.dumps(log, indent=1) + "\n")
    report = {"families": dict(sums), "elapsed_seconds": time.monotonic() - start, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    report["cross_shard_block_scale_updates"] = boundary_scale_updates
    report["memory_guard"] = memory.metrics()
    (a.out / "memory-guard.json").write_text(json.dumps(report["memory_guard"], indent=2) + "\n")
    (a.out / "requant-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {**original_manifest, "revision": 3 if mixed else 2, "parent_manifest_sha256": parent_hash,
                "parent_conversion": {k: original_manifest[k] for k in ("elapsed_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_rss_bytes", "memory_guard")},
                **{k: report[k] for k in ("elapsed_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_rss_bytes")},
                "requantization": {"script_sha256": sha256(Path(__file__)), "multipliers": rt.MULTS, "includes_original_block_scale": True, **report},
                "layout_audit": layout,
                "memory_guard": report["memory_guard"],
                "shards": {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in sorted(a.out.glob("*.safetensors"))}}
    manifest.pop("offline_sanity_load", None)
    save_manifest(a.out, manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with MemoryGuard() as memory:
        main(memory)
