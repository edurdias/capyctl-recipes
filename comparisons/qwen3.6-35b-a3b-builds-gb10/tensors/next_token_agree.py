#!/usr/bin/env python3
"""BF16 greedy answers and weights-only teacher forcing; no inference engine."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
from weights import Reader, SOURCE_REVISION, REFERENCE_REVISION, family, fused_global_scale, reconstructed_search

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def source_name(name):
    return name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name

def load_model(source, directory, variant, experts):
    config = Qwen3_5MoeTextConfig(**json.loads((source / "config.json").read_text())["text_config"])
    config._attn_implementation = "sdpa"
    config._experts_implementation = experts
    with init_empty_weights():
        model = Qwen3_5MoeForCausalLM(config)
    reader, original = Reader(directory), Reader(source)
    selected = {"delta": {"deltanet"}, "attention": {"attention"},
                "combined": {"deltanet", "attention"}}.get(variant, set())
    coverage, errors = Counter(), []
    global_scales = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        key = source_name(name)
        if key.endswith((".experts.gate_up_proj", ".experts.down_proj")) and key not in reader.meta:
            prefix, proj = key.rsplit(".", 1)
            value = torch.empty(tuple(parameter.shape), dtype=torch.bfloat16, device="cuda")
            for expert in range(value.shape[0]):
                if proj == "gate_up_proj":
                    half = value.shape[1] // 2
                    value[expert, :half] = reader.dequantize(f"{prefix}.{expert}.gate_proj")
                    value[expert, half:] = reader.dequantize(f"{prefix}.{expert}.up_proj")
                    coverage["nvfp4_matrices"] += 2
                else:
                    value[expert] = reader.dequantize(f"{prefix}.{expert}.down_proj")
                    coverage["nvfp4_matrices"] += 1
        elif family(key) in selected:
            value, error = reconstructed_search(original.get(key), fused_global_scale(key, original, global_scales))
            errors.append({"tensor": key, "relative_error": error})
            coverage["searched_nvfp4_matrices"] += 1
        elif key.endswith(".weight") and reader.meta[key]["dtype"] in ("U8", "F8_E4M3", "F8_E4M3FN"):
            value = reader.dequantize(key[:-7])
            coverage[reader.meta[key]["dtype"] + "_matrices"] += 1
        else:
            value = reader.get(key)
            coverage["unquantized_state_tensors"] += 1
        value = value.to(torch.bfloat16)
        assert tuple(value.shape) == tuple(parameter.shape), (key, value.shape, parameter.shape)
        assert torch.isfinite(value).all(), key
        set_module_tensor_to_device(model, name, "cuda", value=value, dtype=torch.bfloat16)
        del value
        if index % 80 == 0:
            print(f"loaded {index} parameters; CUDA allocated {torch.cuda.memory_allocated()}", flush=True)
    assert not any(p.is_meta for p in model.parameters())
    model.eval()
    return model, dict(coverage), errors

def score(record, picked):
    mask = [a == b for a, b in zip(record["answer_ids"], picked, strict=True)]
    return {"id": record["id"], "category": record["category"], "n": len(mask), "agree": sum(mask),
            "first_disagreement": next((i for i, ok in enumerate(mask) if not ok), len(mask)),
            "picked_ids": picked, "agreement_mask": mask}

def summarize(records):
    n, k = sum(r["n"] for r in records), sum(r["agree"] for r in records)
    return {"positions": n, "agree": k, "agreement": k / n,
            "median_first_disagreement": statistics.median(r["first_disagreement"] for r in records),
            "by_category": {c: {"positions": sum(r["n"] for r in records if r["category"] == c),
                                    "agree": sum(r["agree"] for r in records if r["category"] == c)}
                            for c in sorted({r["category"] for r in records})}}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", choices=["bf16", "nvidia", "delta", "attention", "combined", "built"])
    ap.add_argument("--source", type=Path, default=Path.home()/"models/qwen3.6-35b-a3b")
    ap.add_argument("--reference", type=Path, default=Path.home()/"models/qwen3.6-35b-a3b-nvidia-nvfp4")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--label")
    ap.add_argument("--results", type=Path, default=HERE/"results")
    ap.add_argument("--prompts", type=Path, default=HERE/"agreement-prompts.json")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--experts", choices=["grouped_mm", "eager"], default="grouped_mm")
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.compiler.set_stance("force_eager")
    started = time.monotonic()
    directory = a.source if a.variant == "bf16" else (a.checkpoint or a.reference)
    answers = a.results/"bf16-answers.json"
    with torch.inference_mode():
        model, coverage, errors = load_model(a.source, directory, a.variant, a.experts)
        loaded = time.monotonic()
        print("MODEL-LOADED", dict(coverage), flush=True)
        if a.variant == "bf16":
            tokenizer = AutoTokenizer.from_pretrained(a.source, local_files_only=True, padding_side="left")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            generation = GenerationConfig.from_pretrained(a.source, local_files_only=True)
            generation.do_sample = False
            generation.temperature = generation.top_p = generation.top_k = None
            generation.pad_token_id = tokenizer.pad_token_id
            eos = generation.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            prompts = json.loads(a.prompts.read_text())["prompts"]
            records = []
            for start in range(0, len(prompts), a.batch_size):
                batch = prompts[start:start+a.batch_size]
                texts = [tokenizer.apply_chat_template(p["messages"], tools=p.get("tools"), tokenize=False,
                         add_generation_prompt=True, enable_thinking=False) for p in batch]
                inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
                tokens = model.generate(**inputs, generation_config=generation, max_new_tokens=a.max_tokens, use_cache=True)
                for p, mask, ids, out in zip(batch, inputs["attention_mask"], inputs["input_ids"], tokens):
                    answer = out[inputs["input_ids"].shape[1]:].tolist()
                    end = next((i+1 for i, token in enumerate(answer) if token in eos), len(answer))
                    records.append({"id": p["id"], "category": p["category"], "prompt_ids": ids[mask.bool()].tolist(),
                                    "answer_ids": answer[:end], "hit_cap": end == a.max_tokens and answer[end-1] not in eos})
                save(a.results/"bf16-answers.partial.json", {"records": records})
                print(f"generated {len(records)}/{len(prompts)}", flush=True)
                del inputs, tokens
            save(answers, {"revision": SOURCE_REVISION, "prompt_sha256": sha(a.prompts),
                           "chat_template_sha256": sha(a.source/"chat_template.jinja"), "enable_thinking": False,
                           "max_tokens": a.max_tokens, "generation_batch_size": a.batch_size,
                           "note": "Fixed-length diagnostic continuations, not a task-quality or truncation gate",
                           "records": records})
            (a.results/"bf16-answers.partial.json").unlink()
            scored = [score(r, r["answer_ids"]) for r in records]
        else:
            reference = json.loads(answers.read_text())
            assert reference["prompt_sha256"] == sha(a.prompts)
            scored = []
            for i, r in enumerate(reference["records"]):
                ids = torch.tensor([r["prompt_ids"] + r["answer_ids"][:-1]], device="cuda")
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(r["answer_ids"])).logits
                assert torch.isfinite(logits).all(), r["id"]
                scored.append(score(r, logits[0].argmax(-1).tolist()))
                del ids, logits
                if (i+1) % 10 == 0:
                    print(f"scored {i+1}/{len(reference['records'])}", flush=True)
                    save(a.results/f"score-{a.label or a.variant}.partial.json", {"records": scored})
        torch.cuda.synchronize()
    report = {"variant": a.label or a.variant, "checkpoint": directory.name, "source_revision": SOURCE_REVISION,
              "reference_revision": REFERENCE_REVISION if a.variant != "bf16" else None,
              "method": "weights reconstructed into BF16; Transformers SDPA; grouped experts; no activation quantization or engine",
              "target_only": "Vision and MTP inactive and not loaded for text next-token agreement",
              "bf16_baseline": "100% by construction, not a separate teacher-forcing rerun" if a.variant == "bf16" else None,
              "reference_sha256": sha(answers), "weight_coverage": coverage, "searched_weights": errors,
              "load_seconds": loaded-started, "evaluation_seconds": time.monotonic()-loaded,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "software": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
              "summary": summarize(scored), "records": scored}
    save(a.results/f"score-{a.label or a.variant}.json", report)
    partial = a.results/f"score-{a.label or a.variant}.partial.json"
    partial.unlink(missing_ok=True)
    print(json.dumps(report["summary"]), flush=True)

if __name__ == "__main__":
    from memory_guard import MemoryGuard
    with MemoryGuard():
        main()
