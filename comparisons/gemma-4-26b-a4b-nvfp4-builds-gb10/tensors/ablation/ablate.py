#!/usr/bin/env python3
"""Family ablation using the shared agreement loader and original BF16 reference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import next_token_agree as common

# All new variants use r2 experts and the original BF16 source for unchanged
# parameters. FP8 is simulated from BF16, not borrowed from another checkpoint.
POLICIES = {
    "A": {"dense_mlp": "bf16", "attention": "bf16"},
    "B": {"dense_mlp": "nvfp4", "attention": "bf16"},
    "C": {"dense_mlp": "bf16", "attention": "nvfp4"},
    "D": {"dense_mlp": "nvfp4", "attention": "fp8_tensor"},
    "D-channel": {"dense_mlp": "nvfp4", "attention": "fp8_channel"},
    "E": {"dense_mlp": "fp8_tensor", "attention": "fp8_tensor"},
    "G": {"dense_mlp": "fp8_tensor", "attention": "fp8_tensor", "output_head": "fp8_channel"},
    "H": {"dense_mlp": "fp8_tensor", "attention": "fp8_tensor", "output_head": "nvfp4_searched"},
    "I": {"dense_mlp": "bf16", "attention": "bf16", "output_head": "fp8_channel"},
}


def family(name):
    if name in ("lm_head", "lm_head.weight"):
        return "output_head"
    # Vision/audio modules also have attention/MLP names but are inactive in
    # this text-only experiment and must not enter either mixing or byte sums.
    if not name.startswith("model.language_model.layers."):
        return None
    if ".experts." in name:
        return "experts"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "dense_mlp"
    if ".router." in name:
        return "router"
    return None


def projection(name):
    return name.endswith("_proj.weight") or name == "lm_head.weight"


def searched_output_head(value, rows_per_chunk=1024):
    """Simulate a new NVFP4 head with the converter's exact r2 search."""
    import read_tensors as rt
    from requant_scales import requant

    global_scale = value.abs().amax().float() / (6. * 448.)
    assert global_scale > 0
    rebuilt = torch.empty_like(value, dtype=torch.bfloat16)
    squared_error = squared_weight = 0.
    minimum_scale, maximum_scale = float("inf"), 0.
    for first in range(0, value.shape[0], rows_per_chunk):
        weight = value[first:first + rows_per_chunk].float()
        blocks = weight.reshape(-1, 16)
        original_scales = rt.fp8(blocks.abs().amax(dim=1) / (6. * global_scale))
        packed, scales = requant(weight, global_scale, original_scales)
        decoded = rt.unpack_e2m1(packed) * scales.float().repeat_interleave(16, dim=-1) * global_scale
        rebuilt[first:first + weight.shape[0]] = decoded
        error = rebuilt[first:first + weight.shape[0]].float() - weight
        squared_error += error.square().sum(dtype=torch.float64).item()
        squared_weight += weight.square().sum(dtype=torch.float64).item()
        minimum_scale = min(minimum_scale, scales.float().min().item())
        maximum_scale = max(maximum_scale, scales.float().max().item())
        del weight, blocks, original_scales, packed, scales, decoded, error
    report = {"name": "lm_head.weight", "format": "NVFP4", "group_size": 16,
              "global_scale_dtype": "float32", "global_scale": global_scale.item(),
              "global_scale_rule": "source BF16 head amax / (6 * 448)",
              "search": "existing converter requant_scales.requant; max-derived FP8 baseline plus 21 FP8-rounded multipliers",
              "multipliers": rt.MULTS, "block_scale_dtype": "float8_e4m3fn",
              "block_scale_shape": [value.shape[0], value.shape[1] // 16],
              "block_scale_min": minimum_scale, "block_scale_max": maximum_scale,
              "relative_weight_error_after_bf16_reconstruction": (squared_error / squared_weight) ** .5,
              "weight_bytes": value.numel() // 2, "block_scale_bytes": value.numel() // 16,
              "global_scale_bytes": 4}
    return rebuilt, report


class MixedReader:
    def __init__(self, source, r2, policy):
        self.source = common.Reader(source)
        self.r2 = common.Reader(r2)
        self.policy = policy
        # Omitting the fused BF16 banks makes the shared loader reconstruct
        # the individual r2 expert projections, just as for a normal r2 load.
        self.meta = {k: v for k, v in self.source.meta.items() if family(k) != "experts"}
        if "output_head" in policy:
            self.meta["lm_head.weight"] = self.source.meta["model.language_model.embed_tokens.weight"]
        self.nvfp4_head_record = None
        self.fp8_records = []
        self.selected_nvfp4 = {m for m in self.r2.quantized_modules()
                               if family(m) == "experts" or policy.get(family(m)) == "nvfp4"}

    def quantized_modules(self):
        return self.selected_nvfp4

    def dequantize(self, module):
        assert module in self.selected_nvfp4, module
        return self.r2.dequantize(module)

    def get(self, name):
        source_name = "model.language_model.embed_tokens.weight" if name == "lm_head.weight" else name
        value = self.source.get(source_name)
        mode = self.policy.get(family(name), "bf16")
        if mode == "nvfp4_searched":
            assert name == "lm_head.weight"
            rebuilt, self.nvfp4_head_record = searched_output_head(value)
            return rebuilt
        if not projection(name) or not mode.startswith("fp8_"):
            return value
        value = value.float()
        if mode == "fp8_tensor":
            amax = value.abs().amax()
            scale = torch.where(amax == 0, 1., amax / 448.)
            scale_dtype = "float32"
            scale_bytes = 4
        else:
            amax = value.abs().amax(dim=1, keepdim=True)
            # Unsloth stores its channel scales as BF16 [out_features, 1].
            scale = torch.where(amax == 0, 1., amax / 448.).to(torch.bfloat16).float()
            scale_dtype = "bfloat16"
            scale_bytes = 2 * value.shape[0]
        quantized = (value / scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
        rebuilt = (quantized.float() * scale).to(torch.bfloat16)
        self.fp8_records.append({"name": name, "family": family(name), "format": "float8_e4m3fn",
                                 "strategy": mode, "scale_dtype": scale_dtype,
                                 "scale_shape": list(scale.shape), "scale_min": scale.min().item(),
                                 "scale_max": scale.max().item(), "amax": amax.max().item(),
                                 "weight_bytes": value.numel(), "scale_bytes": scale_bytes})
        return rebuilt


def payload(meta):
    return meta["data_offsets"][1] - meta["data_offsets"][0]


def byte_arithmetic(source_meta, quant_meta, policy, top_k=8, num_experts=128):
    """[INFERENCE] One weight/weight-scale read, no cache or engine modelling."""
    totals = Counter()
    tensors = Counter()
    for name, meta in source_meta.items():
        f = family(name)
        if f not in ("attention", "dense_mlp", "router"):
            continue
        mode = policy.get(f, "bf16") if projection(name) else "bf16"
        if mode == "nvfp4":
            module = name[:-7]
            nbytes = sum(payload(quant_meta[module + suffix])
                         for suffix in (".weight", ".weight_scale", ".weight_scale_2"))
        elif mode == "fp8_tensor":
            nbytes = math.prod(meta["shape"]) + 4
        elif mode == "fp8_channel":
            nbytes = math.prod(meta["shape"]) + 2 * meta["shape"][0]
        else:
            nbytes = payload(meta)
        totals[f] += nbytes
        tensors[f] += 1
    bank_bytes = sum(payload(meta) for name, meta in quant_meta.items()
                     if family(name) == "experts" and
                     name.endswith((".weight", ".weight_scale", ".weight_scale_2")))
    assert bank_bytes * top_k % num_experts == 0
    totals["routed_experts_top8"] = bank_bytes * top_k // num_experts
    head_shape = source_meta["model.language_model.embed_tokens.weight"]["shape"]
    head_weights = math.prod(head_shape)
    head_mode = policy.get("output_head", "bf16")
    if head_mode == "fp8_channel":
        head_bytes = head_weights + 2 * head_shape[0]
    elif head_mode == "nvfp4_searched":
        head_bytes = head_weights // 2 + head_weights // 16 + 4
    else:
        head_bytes = 2 * head_weights
    engine_head_bytes = head_bytes + (2 * head_shape[0] if head_mode == "fp8_channel" else 0)
    return {"status": "[INFERENCE] arithmetic, not measured GPU traffic or speed",
            "assumption": "One read of each selected packed weight and its weight scales per decode token; 8 of 128 same-shaped experts at each of 30 layers",
            "bytes_per_token": dict(totals), "total_bytes_per_token": sum(totals.values()),
            "output_head_format": head_mode, "output_head_bytes_per_token": head_bytes,
            "total_including_output_head_bytes_per_token": sum(totals.values()) + head_bytes,
            "output_head_bytes_if_channel_scales_loaded_fp32": engine_head_bytes,
            "total_including_head_if_channel_scales_loaded_fp32": sum(totals.values()) + engine_head_bytes,
            "full_expert_bank_weight_and_scale_bytes": bank_bytes,
            "source_family_tensor_counts": dict(tensors),
            "exclusions": ["lm_head/tied embedding", "non-attention layer norms and other state outside named families",
                           "activations and their quantization scales", "KV-cache reads/writes", "temporary buffers", "cache reuse and request batching"],
            "common_bf16_lm_head_weight_bytes": payload(source_meta["model.language_model.embed_tokens.weight"])}


def main(memory):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", choices=POLICIES)
    ap.add_argument("--models", type=Path, default=Path.home() / "models")
    a = ap.parse_args()
    start = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.compiler.set_stance("force_eager")
    torch.cuda.reset_peak_memory_stats()
    results = HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    reference_path = HERE.parent / "results/bf16-answers.json"
    reference = json.loads(reference_path.read_text())
    assert common.sha(reference_path) == "ac75c312aa1342efc482bbc1db2e772d3a7eb6f344a049692850c1522a96ee3c"
    assert len(reference["records"]) == 200
    assert sum(len(r["answer_ids"]) for r in reference["records"]) == 62658
    source = a.models / common.BUILDS["bf16"][0]
    r2 = a.models / common.BUILDS["r2"][0]
    reader = MixedReader(source, r2, POLICIES[a.variant])
    with torch.inference_mode():
        untied = "output_head" in POLICIES[a.variant]
        model, coverage = common.load_model(r2, source, "grouped_mm", reader=reader, untie_output_head=untied)
        assert (model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr()) == untied
        assert model.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
        torch.cuda.synchronize()
        loaded = time.monotonic()
        scored = []
        for i, r in enumerate(reference["records"]):
            ids = torch.tensor([r["prompt_ids"] + r["answer_ids"][:-1]], device="cuda")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(r["answer_ids"])).logits
            assert torch.isfinite(logits).all(), r["id"]
            scored.append(common.score_record(r, logits[0].argmax(-1).tolist()))
            del ids, logits
            if (i + 1) % 10 == 0:
                print(f"{a.variant}: scored {i + 1}/200", flush=True)
        torch.cuda.synchronize()
    fp8_counts = Counter(r["strategy"] for r in reader.fp8_records)
    coverage["unquantized_state_tensors"] -= len(reader.fp8_records)
    coverage.update({f"simulated_{k}_matrices": v for k, v in fp8_counts.items()})
    if reader.nvfp4_head_record is not None:
        coverage["unquantized_state_tensors"] -= 1
        coverage["simulated_searched_nvfp4_head_matrices"] = 1
    report = {"variant": a.variant, "policy": {"experts": "r2_nvfp4", **POLICIES[a.variant], "router": "bf16"},
              "method": "Family-mixed reconstructed BF16 weights; simulated FP8 E4M3 from original BF16; no engine or activation quantization",
              "reference_sha256": common.sha(reference_path), "experts_implementation": "grouped_mm",
              "output_head_untied": untied, "input_embedding_dtype": str(model.model.language_model.embed_tokens.weight.dtype),
              "final_logit_softcapping": model.config.text_config.final_logit_softcapping,
              "nvfp4_head_simulation": reader.nvfp4_head_record,
              "load_seconds": loaded - start, "evaluation_seconds": time.monotonic() - loaded,
              "wall_seconds": time.monotonic() - start, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "memory_guard": memory.metrics(), "weight_coverage": coverage, "fp8_simulation": reader.fp8_records,
              "byte_arithmetic": byte_arithmetic(reader.source.meta, reader.r2.meta, POLICIES[a.variant]),
              "software": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
              "summary": common.summary(scored), "records": scored}
    common.save(results / f"score-{a.variant}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "fp8_simulation")}), flush=True)


if __name__ == "__main__":
    with common.MemoryGuard() as guard:
        main(guard)
