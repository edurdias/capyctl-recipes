#!/usr/bin/env python3
"""Offline Gemma 4 NVFP4 conversion using the existing ModelOpt 0.47.0 environment.

Use --only audit to check an existing output without loading an inference engine.
See NOTES.md for the pinned inputs and commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "gemma4-nvfp4_all_linear-kv_none.yaml"
EXPORT_SHARD_SIZE = "5GB"
SOURCE_REPO = "google/gemma-4-26B-A4B-it"
SOURCE_REV = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
CALIB_SHA = "a3e005c39c3c71918ffff53504091ed32bd02baf3d37241072e5a616b0dc31eb"
PARTS = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32", "input_scale": "F32"}
LINEAR = re.compile(r"^model\.language_model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.[qkvo]_proj)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def headers(directory):
    result = {}
    for shard in sorted(Path(directory).glob("*.safetensors")):
        with shard.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        payload_end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
        assert shard.stat().st_size == 8 + n + payload_end, f"Truncated shard: {shard}"
        for name, meta in h.items():
            if name != "__metadata__":
                assert name not in result, f"Duplicate tensor: {name}"
                result[name] = {**meta, "shard": shard.name, "header_bytes": 8 + n}
    return result


def verify_source(source):
    idx = json.loads((source / "model.safetensors.index.json").read_text())
    got = headers(source)
    assert set(got) == set(idx["weight_map"]), "Source tensor inventory differs from HF index"
    payload = sum(m["data_offsets"][1] - m["data_offsets"][0] for m in got.values())
    assert payload == idx["metadata"]["total_size"], "Source tensor payload differs from HF index"
    url = f"https://huggingface.co/api/models/{SOURCE_REPO}/revision/{SOURCE_REV}?blobs=true"
    with urllib.request.urlopen(url) as f:
        remote = json.load(f)
    siblings = {s["rfilename"]: s for s in remote["siblings"]}
    shards = {}
    for name in sorted(set(idx["weight_map"].values())):
        assert (source / name).stat().st_size == siblings[name]["size"], f"HF file size mismatch: {name}"
        digest = sha256(source / name)
        assert digest == siblings[name]["lfs"]["sha256"], f"HF shard SHA-256 mismatch: {name}"
        shards[name] = {"bytes": (source / name).stat().st_size, "sha256": digest}
    return {"tensor_payload_bytes": payload, "shards": shards}


def expected_shapes(base):
    quant, retained = {}, {}
    for name, meta in base.items():
        if name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
            experts, rows, cols = meta["shape"]
            prefix, part = name.rsplit(".", 1)
            for e in range(experts):
                projs = ("gate", "up") if part == "gate_up_proj" else ("down",)
                for proj in projs:
                    quant[f"{prefix}.{e}.{proj}_proj"] = [rows // 2 if part == "gate_up_proj" else rows, cols]
        elif name.endswith(".weight") and LINEAR.fullmatch(name[:-7]):
            quant[name[:-7]] = meta["shape"]
        else:
            retained[name] = meta
    assert len(quant) == 11725, f"Unexpected quantized module count: {len(quant)}"
    return quant, retained


def tensor_digest(directory, meta):
    h = hashlib.sha256()
    start, end = meta["data_offsets"]
    with (Path(directory) / meta["shard"]).open("rb") as f:
        f.seek(meta["header_bytes"] + start)
        left = end - start
        while left:
            b = f.read(min(left, 8 << 20))
            assert b, "Truncated tensor"
            h.update(b)
            left -= len(b)
    return h.digest()


def audit(source, out, expected_layout=None):
    import torch
    from safetensors import safe_open
    got, base = headers(out), headers(source)
    quant, retained = expected_shapes(base)
    qc = json.loads((out / "hf_quant_config.json").read_text())
    mixed = qc["quantization"]["quant_algo"] == "MIXED_PRECISION"
    cfg = json.loads((out / "config.json").read_text())
    untied = not cfg["text_config"]["tie_word_embeddings"]
    if expected_layout is not None:
        assert mixed == (expected_layout != "all-nvfp4"), "Requested quantization layout differs"
        assert untied == (expected_layout == "G"), "Requested output-head layout differs"
    fp8 = {}
    if mixed:
        fp8 = {n: quant.pop(n) for n in list(quant) if ".experts." not in n}
        if untied:
            fp8["lm_head"] = base["model.language_model.embed_tokens.weight"]["shape"]
        assert len(quant) == 11520 and len(fp8) == 205 + int(untied)
    wanted = set(retained) | {f"{n}.{p}" for n in quant for p in PARTS} | {f"{n}.{p}" for n in fp8 for p in ("weight", "weight_scale", "input_scale")}
    assert set(got) == wanted, {"missing": sorted(wanted - set(got))[:30], "unexpected": sorted(set(got) - wanted)[:30]}
    opened = {}
    counts = Counter()
    for module, (rows, cols) in quant.items():
        for part, dtype in PARTS.items():
            name = f"{module}.{part}"
            m = got[name]
            shape = [rows, cols // 2] if part == "weight" else [rows, cols // 16] if part == "weight_scale" else None
            assert m["dtype"] == dtype, (name, m["dtype"], dtype)
            assert m["shape"] == shape if shape else m["shape"] in ([], [1]), (name, m["shape"], shape)
            if part != "weight":
                if m["shard"] not in opened:
                    opened[m["shard"]] = safe_open(out / m["shard"], framework="pt", device="cpu")
                t = opened[m["shard"]].get_tensor(name).float()
                assert torch.isfinite(t).all() and (t >= 0).all(), name
                if part != "weight_scale":
                    assert (t > 0).all(), f"Uncalibrated scale: {name}"
            counts[part] += 1
    for module, shape in fp8.items():
        for part in ("weight", "weight_scale", "input_scale"):
            name = f"{module}.{part}"
            m = got[name]
            assert m["dtype"] == ("F8_E4M3" if part == "weight" else "F32"), (name, m)
            assert m["shape"] == shape if part == "weight" else m["shape"] in ([], [1]), (name, m)
            if m["shard"] not in opened:
                opened[m["shard"]] = safe_open(out / m["shard"], framework="pt", device="cpu")
            value = opened[m["shard"]].get_tensor(name).float()
            assert torch.isfinite(value).all(), name
            if part != "weight":
                assert (value > 0).all(), f"Uncalibrated FP8 scale: {name}"
            counts[f"fp8_{part}"] += 1
    for name, m in retained.items():
        assert (got[name]["shape"], got[name]["dtype"]) == (m["shape"], m["dtype"]), name
        assert tensor_digest(source, m) == tensor_digest(out, got[name]), f"Changed preserved tensor: {name}"
        counts[f"preserved_{m['dtype']}"] += 1
    assert cfg["text_config"]["tie_word_embeddings"] == (not untied)
    assert ("lm_head.weight" in got) == untied
    if mixed:
        assert cfg.get("tie_word_embeddings", not untied) == (not untied), "Outer/text embedding tie differs"
        expected_layers = {n.split(".experts.", 1)[0] + ".experts": {"quant_algo": "NVFP4", "group_size": 16} for n in quant}
        expected_layers.update({n[:-len(".experts")] + ".moe.experts": spec
                                for n, spec in list(expected_layers.items())})
        expected_layers.update({n: {"quant_algo": "FP8"} for n in fp8})
        assert qc["quantization"]["quantized_layers"] == expected_layers, "Mixed per-layer format differs"
    else:
        assert not untied and qc["quantization"]["quant_algo"] == "NVFP4" and qc["quantization"]["group_size"] == 16
    assert qc["quantization"].get("kv_cache_quant_algo") == ("FP8" if mixed else None)
    assert cfg["quantization_config"]["quant_method"].startswith("modelopt")
    from fnmatch import fnmatchcase
    from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
    assert cfg["quantization_config"] == convert_hf_quant_config_format(qc), "Quantization configs disagree"
    excludes = qc["quantization"].get("exclude_modules", [])
    assert not [n for n in (*quant, *fp8) if any(fnmatchcase(n, p) for p in excludes)], "Quantized modules excluded in engine config"
    for f in source.iterdir():
        if is_asset(f):
            assert sha256(f) == sha256(out / f.name), f"Asset changed: {f.name}"
    result = {"passed": True, "tensors": len(got), "layout": dict(counts), "tensor_payload_bytes": sum(m["data_offsets"][1] - m["data_offsets"][0] for m in got.values())}
    (out / "layout-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[audit]", json.dumps(result), flush=True)
    return result


def is_asset(f):
    return f.is_file() and (f.name.startswith(("tokenizer", "chat_template", "preprocessor", "processor", "video_preprocessor", "generation_config", "special_tokens", "added_tokens", "vocab", "merges")) or f.name in ("LICENSE", "NOTICE"))


def finish(source, out):
    qc_path = out / "hf_quant_config.json"
    qc = json.loads(qc_path.read_text())
    if qc["quantization"]["quant_algo"] == "MIXED_PRECISION":
        from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
        qc["quantization"]["kv_cache_quant_algo"] = "FP8"
        layers = qc["quantization"]["quantized_layers"]
        layers.update({name[:-len(".experts")] + ".moe.experts": spec
                       for name, spec in list(layers.items())
                       if name.endswith(".experts") and not name.endswith(".moe.experts")})
        qc_path.write_text(json.dumps(qc, indent=2) + "\n")
        cfg_path = out / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["quantization_config"] = convert_hf_quant_config_format(qc)
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    for f in source.iterdir():
        if is_asset(f):
            shutil.copyfile(f, out / f.name)


def load_bf16_streaming(source, gpu_gib, layout="all-nvfp4"):
    """Materialize one BF16 state tensor at a time, directly on GPU 0.

    HF automatic loading and CPU safetensors materialization showed excessive
    host residency beside the 33-GiB resident engine. Safetensors 0.8's CUDA
    pread path uses one bounded pinned staging tensor, avoiding that copy.
    """
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma4ForConditionalGeneration
    source = Path(source)
    meta = headers(source)
    payload = sum(m["data_offsets"][1] - m["data_offsets"][0] for m in meta.values())
    assert payload <= gpu_gib * 2**30, "BF16 parameters exceed the fixed GPU placement budget"
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration._from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    opened = {}
    loaded = set()
    state = list(model.named_parameters()) + [(n, b) for n, b in model.named_buffers() if n in meta]
    for name, parameter in state:
        if name == "lm_head.weight":
            continue
        m = meta[name]
        if m["shard"] not in opened:
            opened[m["shard"]] = safe_open(source / m["shard"], framework="pt", device="cuda:0", backend="pread")
        value = opened[m["shard"]].get_tensor(name)
        assert tuple(value.shape) == tuple(parameter.shape) and value.dtype == torch.bfloat16, name
        set_module_tensor_to_device(model, name, "cuda:0", value=value, dtype=value.dtype)
        loaded.add(name)
        del value
        if len(loaded) % 64 == 0:
            print(f"[placement] {len(loaded)}/{len(meta)} BF16 state tensors; allocated={torch.cuda.memory_allocated()} reserved={torch.cuda.memory_reserved()} bytes", flush=True)
    assert loaded == set(meta), {"unloaded_source_tensors": sorted(set(meta) - loaded)}
    model.tie_weights()
    model.to("cuda:0").eval()
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    if layout == "G":
        model.config.tie_word_embeddings = False
        model.config.text_config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
        assert model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr()
    assert all(p.device.type == "cuda" for p in model.parameters())
    print(f"[placement] {len(loaded)} tensors; GPU allocated {torch.cuda.memory_allocated()} bytes", flush=True)
    return model


def install_calibration_guard(report_path, layout="all-nvfp4"):
    """Used inside hf_ptq's process; no environment or package modifications.

    ModelOpt has two SHARED input quantizers per fused experts module. Observe
    their inputs separately for every routed expert, rather than falsely claiming
    ModelOpt maintains 128 independent activation quantizers.
    """
    import torch
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts, force_eager_experts_impl_on_the_fly
    original = mtq.quantize

    def guarded(model, config, forward_loop=None, **kwargs):
        if layout != "all-nvfp4":
            config = copy.deepcopy(config)
            patterns = ["*language_model.layers.*.mlp.gate_proj*", "*language_model.layers.*.mlp.up_proj*",
                        "*language_model.layers.*.mlp.down_proj*", "*language_model.layers.*.self_attn*proj*"]
            if layout == "G":
                patterns.append("*lm_head*")
            for pattern in patterns:
                for kind in ("weight_quantizer", "input_quantizer"):
                    config["quant_cfg"].append({"quantizer_name": pattern + kind, "enable": True,
                                                "cfg": {"num_bits": (4, 3), "axis": None}})
        force_eager_experts_impl_on_the_fly(model)
        evidence, tracked = {}, []

        def calibrate(converted):
            if layout != "all-nvfp4":
                fp8_linears = []
                for name, module in converted.named_modules():
                    if not isinstance(module, torch.nn.Linear):
                        continue
                    expected = bool(LINEAR.fullmatch(name)) or (name == "lm_head" and layout == "G")
                    weight = getattr(module, "weight_quantizer", None)
                    inputs = getattr(module, "input_quantizer", None)
                    assert bool(weight is not None and weight.is_enabled) == expected, f"Unexpected linear quantizer: {name}"
                    if expected:
                        assert inputs is not None and inputs.is_enabled, f"Disabled FP8 input quantizer: {name}"
                        fp8_linears.append(name)
                assert len(fp8_linears) == 205 + int(layout == "G"), len(fp8_linears)
            handles = []
            for name, module in converted.named_modules():
                if not isinstance(module, _QuantFusedExperts):
                    continue
                assert module.num_experts == 128, name
                for proj in ("gate_up_proj", "down_proj"):
                    q = getattr(module, f"{proj}_input_quantizer")
                    assert q.is_enabled, f"Disabled expert input quantizer: {name}.{proj}"
                    maxima = torch.zeros(module.num_experts, device=module.gate_up_proj.device)
                    hits = torch.zeros_like(maxima, dtype=torch.int64)
                    tracked.append((name, module, proj, q, maxima, hits))

                    def record(q, inputs, output, module=module, maxima=maxima, hits=hits):
                        idx = module._current_expert_idx
                        maxima[idx] = torch.maximum(maxima[idx], inputs[0].detach().abs().amax().float())
                        hits[idx] += 1
                    handles.append(q.register_forward_hook(record))
            try:
                return forward_loop(converted)
            finally:
                for h in handles:
                    h.remove()

        result = original(model, config, forward_loop=calibrate if forward_loop else None, **kwargs)
        if tracked:
            errors, coverage, weight_count = [], {}, 0
            if len(tracked) != 60:
                errors.append(f"Expected 60 shared expert input quantizers, got {len(tracked)}")
            for name, module, proj, q, maxima, hits in tracked:
                key = f"{name}.{proj}"
                amax = getattr(q, "_amax", None)
                valid = amax is not None and bool(torch.isfinite(amax).all() and (amax > 0).all())
                if not valid:
                    errors.append(f"Missing input amax: {key}")
                selected = hits > 0
                if not bool(selected.any()):
                    errors.append(f"Eager expert input quantizer never called: {key}")
                if proj == "gate_up_proj":
                    coverage[name] = {"selected_count": int(selected.sum().item()),
                                      "never_routed_experts": (~selected).nonzero().flatten().cpu().tolist()}
                weights = []
                for e, wq in enumerate(getattr(module, f"{proj}_weight_quantizers")):
                    wa = getattr(wq, "_amax", None)
                    weight_count += 1
                    weights.append({"expert": e, "enabled": bool(wq.is_enabled),
                                    "amax": wa.detach().float().cpu().tolist() if wa is not None else None})
                    if not wq.is_enabled:
                        errors.append(f"Disabled expert weight quantizer: {key}.{e}")
                    if wa is None or not bool(torch.isfinite(wa).all() and (wa > 0).all()):
                        errors.append(f"Missing expert weight amax: {key}.{e}")
                evidence[key] = {"shared_quantizer_amax": amax.detach().float().cpu().tolist() if amax is not None else None,
                                 "per_expert_observed_amax": maxima.cpu().tolist(), "per_expert_calls": hits.cpu().tolist(),
                                 "weight_quantizers": weights}
            if weight_count != 7680:
                errors.append(f"Expected 7680 per-expert fused weight quantizers, got {weight_count}")
            proof = {"passed": not errors, "eager": True, "expert_count": 3840,
                     "gate": {"shared_input_amax": "all 60 finite and positive",
                              "expert_weight_amax": "all 7680 present, finite and positive",
                              "exported_scales_requirement": "all input_scale and weight_scale_2 finite and positive",
                              "routing_coverage_required": False},
                     "shared_input_quantizers": len(tracked), "expert_weight_quantizers": weight_count,
                     "errors": errors, "routing_coverage": coverage, "quantizers": evidence}
            Path(report_path).write_text(json.dumps(proof, indent=2) + "\n")
            assert not errors, errors
        return result
    mtq.quantize = guarded


RUNNER = r'''
import atexit, runpy, sys, time, json
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
from convert_gemma4_26b_a4b_nvfp4 import EXPORT_SHARD_SIZE, install_calibration_guard, load_bf16_streaming
from memory_guard import MemoryGuard
report = sys.argv.pop(1)
gib = float(sys.argv.pop(1))
layout = sys.argv.pop(1)
import example_utils
def get_fixed_model(ckpt_path, *args, **kwargs):
    return load_bf16_streaming(ckpt_path, gib, layout)
example_utils.get_model = get_fixed_model
import functools
import modelopt.torch.export as modelopt_export
modelopt_export.export_hf_checkpoint = functools.partial(modelopt_export.export_hf_checkpoint, max_shard_size=EXPORT_SHARD_SIZE)
install_calibration_guard(report, layout)
sys.argv = ["hf_ptq.py", *sys.argv[1:]]
started = time.monotonic()
memory = MemoryGuard()
def record_memory():
    import torch
    p = Path(report)
    import resource
    usage = {**memory.metrics(), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
             "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
             "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    (p.parent / "memory-guard.json").write_text(json.dumps(usage, indent=2) + "\n")
    if p.exists():
        data = json.loads(p.read_text())
        data["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        data["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        data["calibration_export_seconds"] = time.monotonic() - started
        p.write_text(json.dumps(data, indent=2) + "\n")
atexit.register(record_memory)
with memory:
    runpy.run_path("hf_ptq.py", run_name="__main__")
'''


def save_manifest(out, manifest):
    path = out / "conversion-manifest.json"
    for _ in range(4):
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        sizes = {"files_bytes": sum(f.stat().st_size for f in out.iterdir() if f.is_file()),
                 "allocated_bytes": sum(f.stat().st_blocks * 512 for f in out.iterdir() if f.is_file())}
        if all(manifest.get(k) == v for k, v in sizes.items()):
            return
        manifest.update(sizes)
    raise RuntimeError("Directory sizes changed while writing conversion manifest")


def write_manifest(args, layout, elapsed, source_info=None):
    import torch
    out = args.out
    report = json.loads((out / "expert-calibration-audit.json").read_text())
    assert report["passed"] and report["expert_count"] == 3840
    memory = json.loads((out / "memory-guard.json").read_text())
    manifest = {"source": {"repository": SOURCE_REPO, "revision": SOURCE_REV, **(source_info or {})}, "revision": 1 if args.layout == "all-nvfp4" else 3, "layout": args.layout,
                "recipe": {"file": args.recipe.name, "sha256": sha256(args.recipe), "max_shard_size": EXPORT_SHARD_SIZE},
                "kv_cache": {"quant_algo": "FP8" if args.layout != "all-nvfp4" else None,
                             "scales_present": False, "calibrated": False},
                "calibration": {"sha256": sha256(args.calib), "samples": 512, "sequence_length": 1024, "batch_size": args.batch_size, "chat_template_sha256": sha256(args.source / "chat_template.jinja")},
                "expert_calibration": {"passed": report["passed"], "gate": report["gate"],
                                       "shared_input_quantizers": report["shared_input_quantizers"],
                                       "expert_weight_quantizers": report["expert_weight_quantizers"],
                                       "routing_coverage": report["routing_coverage"],
                                       "audit_sha256": sha256(out / "expert-calibration-audit.json")},
                "tools": {d: md.version(d) for d in ("nvidia-modelopt", "torch", "transformers", "accelerate", "datasets", "safetensors")},
                "model_optimizer_commit": subprocess.check_output(["git", "-C", str(args.hf_ptq_dir), "rev-parse", "HEAD"], text=True).strip(),
                "converter_sha256": sha256(Path(__file__)), "elapsed_seconds": elapsed,
                "cuda_peak_allocated_bytes": report.get("cuda_peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": report.get("cuda_peak_reserved_bytes"),
                "calibration_export_seconds": report.get("calibration_export_seconds"),
                "memory_guard": memory,
                "process_peak_rss_bytes": max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                              resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
                                              memory["process_peak_rss_bytes"]),
                "hardware": {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "machine": os.uname().machine},
                "layout_audit": layout,
                "shards": {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in sorted(out.glob("*.safetensors"))}}
    save_manifest(out, manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--hf-ptq-dir", type=Path, required=True)
    p.add_argument("--recipe", type=Path, default=RECIPE)
    p.add_argument("--gpu-gib", type=float, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--layout", choices=("all-nvfp4", "G", "E"), default="all-nvfp4",
                   help="G: FP8 body and untied FP8 head; E: FP8 body and tied BF16 head; both search expert block scales")
    p.add_argument("--only", choices=("audit",))
    args = p.parse_args()
    start = time.monotonic()
    assert sha256(args.calib) == CALIB_SHA, "Calibration set differs from the Qwen 512-conversation precedent"
    source_info = verify_source(args.source)
    if not args.only:
        args.out.mkdir(parents=True, exist_ok=True)
        assert not list(args.out.glob("*.safetensors")), "Output checkpoint already exists"
        cmd = [sys.executable, "-c", RUNNER, str(HERE), str(args.out / "expert-calibration-audit.json"), str(args.gpu_gib), args.layout,
               "--pyt_ckpt_path", str(args.source), "--recipe", str(args.recipe), "--dataset", str(args.calib),
               "--calib_size", "512", "--calib_seq", "1024", "--batch_size", str(args.batch_size), "--export_path", str(args.out), "--skip_generate"]
        subprocess.run(cmd, cwd=args.hf_ptq_dir, check=True, env={**os.environ, "PYTHONPATH": str(args.hf_ptq_dir), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        finish(args.source, args.out)
    layout = audit(args.source, args.out, args.layout)
    if not args.only:
        write_manifest(args, layout, time.monotonic() - start, source_info)
        if args.layout != "all-nvfp4":
            subprocess.run([sys.executable, str(HERE / "requant_scales.py"), "--source", str(args.source),
                            "--ckpt", str(args.out), "--out", str(args.out), "--in-place"], check=True)
            subprocess.run([sys.executable, str(HERE / "sanity_load.py"), "--ckpt", str(args.out)], check=True)


if __name__ == "__main__":
    main()
