#!/usr/bin/env python3
"""Torch-only BF16 greedy reference and reconstructed-weight teacher forcing.

Uses the existing Gemma reader and streaming-loader conventions. No activation
quantizers, inference engines, or native quantized kernels are involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoTokenizer, GenerationConfig, Gemma4Config, Gemma4ForConditionalGeneration

HERE = Path(__file__).resolve().parent
LOCAL_CONVERT = HERE.parents[2] / "vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/convert"
CONVERT = LOCAL_CONVERT if LOCAL_CONVERT.exists() else Path.home() / "gemma4-work/convert"
sys.path.insert(0, str(CONVERT))
from memory_guard import MemoryGuard
from read_tensors import Reader

BUILDS = {
    "bf16": ("gemma-4-26b-a4b-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
    "r1": ("Gemma-4-26B-A4B-NVFP4-capyctl-r1", None),
    "r2": ("Gemma-4-26B-A4B-NVFP4-capyctl-r2", None),
    "r3": ("Gemma-4-26B-A4B-NVFP4-capyctl-r3", None),
    "r3e": ("Gemma-4-26B-A4B-NVFP4-capyctl-r3e", None),
    "nvidia": ("gemma-4-26b-a4b-nvidia-nvfp4", "a19cfe00be84568a6867111c9a68c9c44fdcffe6"),
    "redhat": ("gemma-4-26b-a4b-redhat-nvfp4", "5557756b8dce33ac72f2bd702b11729fdba3b839"),
    "unsloth": ("unsloth--gemma-4-26B-A4B-it-NVFP4@20df0542b1a86ce19f495ac2eca2c7c12bce82f9", "20df0542b1a86ce19f495ac2eca2c7c12bce82f9"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_model(directory, source, experts, reader=None, untie_output_head=False):
    # Always use the source architecture; quantization wrappers are never built.
    config = Gemma4Config.from_dict(json.loads((source / "config.json").read_text()))
    if untie_output_head:
        config.tie_word_embeddings = False
        config.text_config.tie_word_embeddings = False
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.text_config._experts_implementation = experts
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(config)
    reader = Reader(directory) if reader is None else reader
    quantized = reader.quantized_modules()
    counts = Counter()
    state = list(model.named_parameters()) + [(n, b) for n, b in model.named_buffers() if n in reader.meta]
    for name, parameter in state:
        if name == "lm_head.weight" and not untie_output_head:
            continue
        if name.endswith(".weight") and name[:-7] in quantized:
            value = reader.dequantize(name[:-7]).to(torch.bfloat16)
            counts["nvfp4_matrices"] += 1
        elif name in reader.meta:
            if reader.meta[name]["dtype"] in ("F8_E4M3", "F8_E4M3FN"):
                # Both formats use multiplicative scales: ModelOpt stores one
                # FP32 scale; compressed-tensors may store [out_features, 1].
                scale_name = name[:-7] + ".weight_scale"
                scale_shape = tuple(reader.meta[scale_name]["shape"])
                per_tensor = scale_shape in ((), (1,), (1, 1))
                assert per_tensor or scale_shape == (parameter.shape[0], 1), (name, scale_shape)
                value = reader.dequantize(name[:-7]).to(torch.bfloat16)
                counts["fp8_tensor_matrices" if per_tensor else "fp8_channel_matrices"] += 1
            else:
                value = reader.get(name).to(torch.bfloat16)
                counts["unquantized_state_tensors"] += 1
        elif name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
            prefix, proj = name.rsplit(".", 1)
            value = torch.empty(tuple(parameter.shape), device="cuda", dtype=torch.bfloat16)
            for e in range(value.shape[0]):
                if proj == "gate_up_proj":
                    half = value.shape[1] // 2
                    value[e, :half] = reader.dequantize(f"{prefix}.{e}.gate_proj")
                    value[e, half:] = reader.dequantize(f"{prefix}.{e}.up_proj")
                    counts["nvfp4_matrices"] += 2
                else:
                    value[e] = reader.dequantize(f"{prefix}.{e}.down_proj")
                    counts["nvfp4_matrices"] += 1
        else:
            raise KeyError(f"Checkpoint cannot populate parameter: {name}")
        assert tuple(value.shape) == tuple(parameter.shape), name
        assert torch.isfinite(value).all(), name
        set_module_tensor_to_device(model, name, "cuda", value=value, dtype=torch.bfloat16)
        del value
    model.tie_weights()
    tied = model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert tied != untie_output_head
    assert not any(p.is_meta for p in model.parameters())
    model.eval()
    print("loaded", directory.name, dict(counts), flush=True)
    return model, dict(counts)


def summary(records):
    positions = sum(r["n"] for r in records)
    agree = sum(r["agree"] for r in records)
    return {"positions": positions, "agree": agree, "agreement": agree / positions,
            "median_first_disagreement": statistics.median(r["first_disagreement"] for r in records),
            "by_category": {c: sum(r["agree"] for r in records if r["category"] == c) /
                            sum(r["n"] for r in records if r["category"] == c)
                            for c in sorted({r["category"] for r in records})}}


def score_record(record, picked):
    agree = [x == y for x, y in zip(record["answer_ids"], picked, strict=True)]
    return {"id": record["id"], "category": record["category"], "n": len(agree),
            "agree": sum(agree), "first_disagreement": next((i for i, ok in enumerate(agree) if not ok), len(agree)),
            "picked_ids": picked, "agreement_mask": agree}


def generate(model, tokenizer, source, prompts, max_tokens, batch_size, results):
    generation = GenerationConfig.from_pretrained(source, local_files_only=True)
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    eos = generation.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    records = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        # Keep the exact 200 Qwen messages and tools, rendered by Gemma's pinned
        # canonical template. Disable thinking identically for all categories.
        texts = [tokenizer.apply_chat_template(p["messages"], tools=p.get("tools"),
                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in batch]
        inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
        tokens = model.generate(**inputs, generation_config=generation, max_new_tokens=max_tokens, use_cache=True)
        for p, mask, ids, output in zip(batch, inputs["attention_mask"], inputs["input_ids"], tokens):
            answer = output[inputs["input_ids"].shape[1]:].tolist()
            end = next((i + 1 for i, token in enumerate(answer) if token in eos), len(answer))
            records.append({"id": p["id"], "category": p["category"],
                            "prompt_ids": ids[mask.bool()].tolist(), "answer_ids": answer[:end]})
        save(results / "bf16-answers.partial.json", {"records": records})
        print(f"generated {len(records)}/{len(prompts)}; {sum(len(r['answer_ids']) for r in records)} positions", flush=True)
        del inputs, tokens
    return records


def main(memory):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build", choices=BUILDS)
    ap.add_argument("--models", type=Path, default=Path.home() / "models")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--prompts", type=Path, default=HERE / "agreement-prompts.json")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--experts", choices=["eager", "grouped_mm"], default="grouped_mm")
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.compiler.set_stance("force_eager")
    torch.cuda.reset_peak_memory_stats()
    source = a.models / BUILDS["bf16"][0]
    directory = a.models / BUILDS[a.build][0]
    checkpoint_config = json.loads((directory / "config.json").read_text())
    checkpoint_text_config = checkpoint_config.get("text_config", checkpoint_config)
    untie_output_head = not checkpoint_text_config.get("tie_word_embeddings", True)
    with torch.inference_mode():
        model, coverage = load_model(directory, source, a.experts, untie_output_head=untie_output_head)
        torch.cuda.synchronize()
        loaded = time.monotonic()
        answers_path = a.results / "bf16-answers.json"
        if a.build == "bf16":
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            tokenizer.padding_side = "left"
            prompts = json.loads(a.prompts.read_text())["prompts"]
            records = generate(model, tokenizer, source, prompts, a.max_tokens, a.batch_size, a.results)
            save(answers_path, {"build": "bf16", "revision": BUILDS["bf16"][1], "prompt_sha256": sha(a.prompts),
                               "chat_template_sha256": sha(source / "chat_template.jinja"),
                               "max_tokens": a.max_tokens, "generation_batch_size": a.batch_size,
                               "enable_thinking": False, "records": records})
            scored = [score_record(r, r["answer_ids"]) for r in records]
            (a.results / "bf16-answers.partial.json").unlink()
        else:
            reference = json.loads(answers_path.read_text())
            records = reference["records"]
            scored = []
            for i, r in enumerate(records):
                # Row n_prompt-1 predicts answer[0]. The final answer token need
                # not be an input. Retain only the n_answer prediction logits.
                ids = torch.tensor([r["prompt_ids"] + r["answer_ids"][:-1]], device="cuda")
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(r["answer_ids"])).logits
                assert torch.isfinite(logits).all(), r["id"]
                picked = logits[0].argmax(-1).tolist()
                scored.append(score_record(r, picked))
                del ids, logits
                if (i + 1) % 10 == 0:
                    print(f"scored {i + 1}/{len(records)}", flush=True)
        torch.cuda.synchronize()
    report = {"build": a.build, "model_directory": directory.name, "revision": BUILDS[a.build][1],
              "method": "weights-only reconstructed BF16; torch Transformers SDPA; no inference engine",
              "reference_sha256": sha(answers_path), "experts_implementation": a.experts,
              "output_head_untied": untie_output_head,
              "input_embedding_dtype": str(model.model.language_model.embed_tokens.weight.dtype),
              "final_logit_softcapping": model.config.text_config.final_logit_softcapping,
              "bf16_baseline": "100% by construction from generation, not a separate teacher-forcing rerun" if a.build == "bf16" else None,
              "load_seconds": loaded - start, "evaluation_seconds": time.monotonic() - loaded,
              "wall_seconds": time.monotonic() - start, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "memory_guard": memory.metrics(), "weight_coverage": coverage,
              "software": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
              "summary": summary(scored), "records": scored}
    save(a.results / f"score-{a.build}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    with MemoryGuard() as guard:
        main(guard)
