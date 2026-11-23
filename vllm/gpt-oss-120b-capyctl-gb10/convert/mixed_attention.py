#!/usr/bin/env python3
"""CPU-only GPT-OSS export: unchanged MXFP4 experts, FP8 E4M3 attention.

Use the existing ModelOpt 0.47.0 Python environment; no ModelOpt/Quark export
package is needed. See source-feasibility.md for engine-specific load contracts.
This script does not launch an engine, access CUDA, or quantize the output head.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
ATTENTION = re.compile(r"^(model\.layers\.\d+\.self_attn\.)([qkvo]_proj)\.weight$")
EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)_(blocks|scales|bias)$")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def quantization_config(scale):
    fp8 = {
        "weight": {"dtype": "fp8_e4m3", "qscheme": "per_" + scale,
                   "is_dynamic": False, "symmetric": True},
        "input_tensors": {"dtype": "fp8_e4m3", "qscheme": "per_channel",
                          "is_dynamic": True, "symmetric": True},
        "output_tensors": None, "bias": None,
    }
    return {
        "quant_method": "quark",
        "export": {"pack_method": "reorder", "kv_cache_group": []},
        "global_quant_config": {
            "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32,
                       "is_dynamic": False, "symmetric": True, "scale_format": "e8m0"},
            "input_tensors": None, "output_tensors": None, "bias": None,
        },
        "layer_type_quant_config": {},
        "layer_quant_config": {"*.self_attn." + p + "_proj": fp8 for p in "qkvo"},
        "exclude": ["lm_head", "*embed_tokens*", "*.mlp.router", "*.mlp.gate"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--scale", choices=["tensor", "channel"], default="channel")
    a = ap.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "gpt_oss" or config.get("quantization_config", {}).get("quant_method") != "mxfp4":
        raise ValueError("Expected the native OpenAI GPT-OSS MXFP4 checkpoint")
    if any(config[key] <= 0 for key in ("hidden_size", "num_hidden_layers", "num_local_experts")):
        raise ValueError("Expected positive GPT-OSS geometry")
    verification = json.loads((source / "download-verification.json").read_text())
    if verification["revision"] != REVISION or verification["model"] != "openai/gpt-oss-120b":
        raise ValueError("Pinned source verification identity mismatch")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    source_shards = defaultdict(list)
    for name, shard in weight_map.items():
        source_shards[shard].append(name)
    # Bound output buffers to one layer, regardless of source shard boundaries.
    groups = defaultdict(list)
    for name in weight_map:
        match = re.match(r"model\.layers\.(\d+)\.", name)
        groups[int(match[1]) if match else -1].append(name)
    # Refuse to overwrite or leave a partially overwritten checkpoint.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    new_map, experts, shared_qkv_scales = {}, {}, {}
    total_size = 0
    with ExitStack() as stack, torch.inference_mode():
        readers = {shard: stack.enter_context(safe_open(str(source / shard), framework="pt", device="cpu"))
                   for shard in source_shards}

        def get(name):
            return readers[weight_map[name]].get_tensor(name)

        for shard_number, (_, names) in enumerate(sorted(groups.items()), 1):
            tensors = {}
            for name in names:
                value = get(name)
                attn, expert = ATTENTION.fullmatch(name), EXPERT.fullmatch(name)
                if attn:
                    prefix, projection = attn.groups()
                    dense = value.float()
                    if a.scale == "channel":
                        scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
                    elif projection != "o_proj":
                        # Fused QKV is rescaled to its largest shard scale by
                        # vLLM. Export a common scale to avoid a second rounding.
                        if prefix not in shared_qkv_scales:
                            shared_qkv_scales[prefix] = max(
                                get(prefix + p + "_proj.weight").float().abs().max().item()
                                for p in "qkv") / 448.0
                        scale = torch.tensor(shared_qkv_scales[prefix], dtype=torch.float32)
                    else:
                        scale = dense.abs().max() / 448.0
                    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
                    tensors[name] = (dense / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                    scale_name = name.removesuffix(".weight") + ".weight_scale"
                    tensors[scale_name] = scale.reshape(-1).contiguous()
                    del dense
                elif expert:
                    prefix, projection, kind = expert.groups()
                    if value.dtype not in (torch.uint8, torch.bfloat16):
                        raise ValueError(f"Unexpected expert dtype: {name}: {value.dtype}")
                    # Canonical source-byte hashes allow the later agreement gate
                    # to prove that no expert weight, E8M0 scale, or bias changed.
                    experts[name] = {"sha256": hashlib.sha256(memoryview(value.contiguous().view(torch.uint8).numpy())).hexdigest(),
                                     "shape": list(value.shape), "dtype": str(value.dtype)}
                    if kind == "blocks":
                        value = value.reshape(value.shape[0], value.shape[1], -1)
                    suffix = {"blocks": "weight", "scales": "weight_scale", "bias": "bias"}[kind]
                    # Keep [gate0, up0, gate1, up1, ...] row order. The
                    # dedicated GPT-OSS backend converter does the permutation.
                    for e in range(config["num_local_experts"]):
                        tensors[f"{prefix}.{e}.{projection}.{suffix}"] = value[e].contiguous()
                else:
                    tensors[name] = value
            filename = f"model-{shard_number:05d}-of-{len(groups):05d}.safetensors"
            save_file(tensors, str(output / filename), metadata={"format": "pt"})
            for name, value in tensors.items():
                new_map[name] = filename
                total_size += value.numel() * value.element_size()
            del tensors
    # Preserve tokenizer, template, generation defaults and license. Never copy
    # old quantization descriptors, old indexes, or redundant representations.
    for path in source.iterdir():
        if path.is_file() and path.name not in {
            "config.json", "model.safetensors.index.json", "hf_quant_config.json", "quantize_config.json"
        } and path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}:
            shutil.copy2(path, output / path.name)
    config["quantization_config"] = quantization_config(a.scale)
    dump(output / "config.json", config)
    dump(output / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": new_map})
    dump(output / "conversion.json", {
        "source_model": "openai/gpt-oss-120b", "source_revision": REVISION,
        "source_verification": verification,
        "geometry": {key: config[key] for key in ("hidden_size", "num_hidden_layers", "num_local_experts", "num_experts_per_tok")},
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "source_index_sha256": hashlib.sha256((source / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "layout": "vllm-quark", "attention_scale": a.scale,
        "changed_tensors": "attention q/k/v/o weights only; added FP32 multiplicative scales",
        "expert_storage": "unchanged values and packed bits, engine-specific reversible repack",
        "lm_head": "unchanged BF16", "experts": experts,
    })
    print(json.dumps({"output": str(output), "tensor_bytes": total_size, "expert_tensors": len(experts)}))


if __name__ == "__main__":
    main()
