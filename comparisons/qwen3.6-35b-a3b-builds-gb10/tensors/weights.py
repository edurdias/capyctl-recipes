#!/usr/bin/env python3
"""ModelOpt mixed FP8/NVFP4 reconstruction and bounded-memory block-scale search."""
from __future__ import annotations
import json
from pathlib import Path
import re
import struct
import torch
from safetensors import safe_open

SOURCE_REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
REFERENCE_REVISION = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
MULTS = [0.60 + 0.025 * i for i in range(21)]
LEVELS = [0., .5, 1., 1.5, 2., 3., 4., 6.]

def family(name):
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if ".linear_attn." in name and re.search(r"\.(in_proj_qkv|in_proj_z|out_proj)(\.weight)?$", name):
        return "deltanet"
    if ".self_attn." in name and re.search(r"\.[qkvo]_proj(\.weight)?$", name):
        return "attention"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.shared_expert." in name:
        return "shared_experts"
    if name.startswith("lm_head."):
        return "head"
    return "preserved"

def headers(directory):
    result = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        for name, h in meta.items():
            if name != "__metadata__":
                result[name] = {**h, "file": path.name, "header_bytes": n + 8}
    return result

class Reader:
    def __init__(self, directory, device="cuda"):
        self.directory = Path(directory)
        self.meta = headers(directory)
        self.device = device
        self.open = {}

    def get(self, name):
        path = self.directory / self.meta[name]["file"]
        if path not in self.open:
            self.open[path] = safe_open(path, framework="pt", device="cpu")
        return self.open[path].get_tensor(name).to(self.device, copy=True)

    def dequantize(self, base):
        w = self.get(base + ".weight")
        if w.dtype == torch.uint8:
            lut = torch.tensor(LEVELS + [-x for x in LEVELS], device=w.device)
            value = torch.stack((lut[(w & 15).long()], lut[(w >> 4).long()]), -1).flatten(-2)
            scale = self.get(base + ".weight_scale").float().repeat_interleave(16, -1)
            return value * scale * self.get(base + ".weight_scale_2").float()
        if w.dtype == torch.float8_e4m3fn:
            return w.float() * self.get(base + ".weight_scale").float()
        return w


def fused_global_scale(name, source, cache):
    """Use one global scale for linears the approved native engines fuse."""
    prefix, projection, _ = name.rsplit(".", 2)
    if projection in ("q_proj", "k_proj", "v_proj"):
        projections = ("q_proj", "k_proj", "v_proj")
    elif projection in ("in_proj_qkv", "in_proj_z"):
        projections = ("in_proj_qkv", "in_proj_z")
    else:
        projections = (projection,)
    group = prefix + "." + projections[0]
    if group not in cache:
        maxima = []
        for proj in projections:
            weight = source.get(prefix + "." + proj + ".weight")
            maxima.append(weight.abs().amax().float())
            del weight
        cache[group] = torch.stack(maxima).amax() / (6 * 448)
    return cache[group]

def search(w, global_scale=None, chunk=65536):
    """Search 21 FP8-rounded block scales; ModelOpt 0.47 packs the final weights."""
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    w = w.float()
    if global_scale is None:
        global_scale = NVFP4QTensor.get_weights_scaling_factor_2(w)
    blocks = w.reshape(-1, 16)
    scales = torch.empty(blocks.shape[0], device=w.device, dtype=torch.float8_e4m3fn)
    levels = torch.tensor(LEVELS, device=w.device)
    mids = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=w.device)
    error = 0.
    for start in range(0, len(blocks), chunk):
        b = blocks[start:start+chunk]
        initial = b.abs().amax(-1) / (6 * global_scale)
        best = torch.full_like(initial, float("inf"))
        chosen = torch.zeros_like(initial)
        for multiplier in MULTS:
            scale = (initial * multiplier).clamp(min=2**-9, max=448).to(torch.float8_e4m3fn).float()
            denom = (scale * global_scale).clamp_min(1e-30)[:, None]
            x = b / denom
            rounded = torch.sign(x) * levels[torch.bucketize(x.abs().clamp(max=6), mids)]
            err = ((rounded * denom - b) ** 2).sum(-1)
            take = err < best
            best = torch.where(take, err, best)
            chosen = torch.where(take, scale, chosen)
        scales[start:start+len(b)] = chosen.to(torch.float8_e4m3fn)
        error += best.double().sum().item()
    scales = scales.reshape(*w.shape[:-1], w.shape[-1] // 16)
    # Zero blocks use a nonzero scale to avoid 0/0 inside ModelOpt's packer.
    scales = scales.float().clamp_min(2**-9).to(torch.float8_e4m3fn)
    quant, scales, global_scale = NVFP4QTensor.quantize(w, 16, weights_scaling_factor=scales,
                                                     weights_scaling_factor_2=global_scale)
    packed = quant._quantized_data
    return packed, scales, global_scale, (error / w.double().square().sum().item()) ** .5

def reconstructed_search(w, global_scale=None):
    packed, scale, global_scale, error = search(w, global_scale)
    lut = torch.tensor(LEVELS + [-x for x in LEVELS], device=w.device)
    value = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1).flatten(-2)
    return value * scale.float().repeat_interleave(16, -1) * global_scale, error
