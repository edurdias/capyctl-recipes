#!/usr/bin/env python3
"""Load the exported checkpoint into a BF16 Transformers model, without an engine.

Packed NVFP4 tensors are explicitly decoded, and per-expert weights are stacked
back into Transformers' fused expert parameters. This checks serialization,
complete parameter coverage and finite/coherent execution; it does NOT qualify
an inference engine's native NVFP4 kernels or packed-weight loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoTokenizer, Gemma4Config, Gemma4ForConditionalGeneration

from convert_gemma4_26b_a4b_nvfp4 import save_manifest
from read_tensors import Reader
from memory_guard import MemoryGuard


def main(memory):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    a = p.parse_args()
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    torch.compiler.set_stance("force_eager")
    data = json.loads((a.ckpt / "config.json").read_text())
    data.pop("quantization_config", None)
    config = Gemma4Config.from_dict(data)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.text_config._experts_implementation = "eager"
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(config)
    reader = Reader(a.ckpt)
    loaded = 0
    with torch.inference_mode():
        state = list(model.named_parameters()) + [(n, b) for n, b in model.named_buffers() if n in reader.meta]
        for name, parameter in state:
            if name == "lm_head.weight" and config.text_config.tie_word_embeddings:
                continue
            if name.endswith(".weight") and name[:-7] + ".weight_scale" in reader.meta:
                value = reader.dequantize(name[:-7]).to(torch.bfloat16)
            elif name in reader.meta:
                value = reader.get(name).to(torch.bfloat16)
            elif name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
                prefix, proj = name.rsplit(".", 1)
                value = torch.empty(tuple(parameter.shape), device="cuda", dtype=torch.bfloat16)
                for e in range(value.shape[0]):
                    if proj == "gate_up_proj":
                        half = value.shape[1] // 2
                        value[e, :half] = reader.dequantize(f"{prefix}.{e}.gate_proj")
                        value[e, half:] = reader.dequantize(f"{prefix}.{e}.up_proj")
                    else:
                        value[e] = reader.dequantize(f"{prefix}.{e}.down_proj")
            else:
                raise KeyError(f"Checkpoint cannot populate parameter: {name}")
            assert tuple(value.shape) == tuple(parameter.shape), name
            assert torch.isfinite(value).all(), name
            set_module_tensor_to_device(model, name, "cuda", value=value, dtype=value.dtype)
            loaded += 1
            del value
        model.tie_weights()
        tied = model.lm_head.weight is model.model.language_model.embed_tokens.weight
        assert tied == config.text_config.tie_word_embeddings
        assert model.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
        if not tied:
            assert model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr()
        assert not any(p.is_meta for p in model.parameters())
        model.to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is 2 + 2? Answer briefly."}], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")["input_ids"]
        logits = model(input_ids=prompt, use_cache=False, logits_to_keep=1).logits
        assert torch.isfinite(logits).all()
        tokens = model.generate(input_ids=prompt, do_sample=False, max_new_tokens=16, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(tokens[0, prompt.shape[-1]:], skip_special_tokens=True)
        assert text.strip().lstrip("*").startswith("4"), f"Incorrect arithmetic sanity output: {text!r}"
    report = {"passed": True, "method": "Explicit stored NVFP4/FP8 decode into BF16 Transformers; not native engine qualification", "loaded_state_tensors": loaded,
              "tied_embedding_head": tied, "finite_logits": True, "prompt": "What is 2 + 2? Answer briefly.", "output": text,
              "elapsed_seconds": time.monotonic() - start, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    report["memory_guard"] = memory.metrics()
    (a.ckpt / "offline-sanity-load.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = json.loads((a.ckpt / "conversion-manifest.json").read_text())
    manifest["offline_sanity_load"] = report
    save_manifest(a.ckpt, manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with MemoryGuard() as memory:
        main(memory)
