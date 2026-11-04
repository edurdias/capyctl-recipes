#!/usr/bin/env python3
"""Make capyctl/Qwen3.8-27B-NVFP4 from Qwen/Qwen3.8-27B (BF16) with NVIDIA ModelOpt.

Layout (quant_method "modelopt"):

  - NVFP4 W4A4, group size 16, static input_scale: MLP gate/up/down, lm_head, self_attn
    q/k/v/o, linear_attn in_proj_qkv / in_proj_z / out_proj
  - BF16, bit-identical to the source: embed_tokens, linear_attn in_proj_a / in_proj_b,
    conv1d, A_log, dt_bias, norms, vision tower (model.visual.*), MTP head (mtp.*)
  - KV cache: not quantized (no k_scale / v_scale), as nvidia/Qwen3.8-27B-NVFP4

Steps:
  1. quant    - ModelOpt examples/hf_ptq/hf_ptq.py with the recipe next to this script.
  2. finish   - the source's tokenizer files and chat template, generation_config.json as the
                source ships it.
  3. audit    - reads every tensor header of the output and fails unless each tensor has the
                format above; writes conversion-manifest.json (source, tools, calibration set,
                layout counts) into the output.

Run with the ModelOpt venv's python. Calibration needs a CUDA GPU and about 70 GB of memory
(the BF16 model is 55.6 GB); FP4 hardware is not needed.

Example:
  python convert_qwen3_8_27b_nvfp4.py \
      --source ~/models/qwen3.8-27b --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
      --out ~/models/Qwen3.8-27B-NVFP4 \
      --hf-ptq-dir ~/Model-Optimizer-0.47.0/examples/hf_ptq \
      --calib calib-chat-code-512.jsonl --gpu-gib 100 --skip-generate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "qwen3_5-nvfp4_all_linear-kv_none.yaml"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "preprocessor_config.json",
                   "video_preprocessor_config.json", "generation_config.json")

# Projections quantized to NVFP4: weight (U8, two e2m1 codes a byte), weight_scale (F8_E4M3,
# one per 16 inputs), weight_scale_2 and input_scale (F32 scalars).
NVFP4 = re.compile(r"(\.mlp\.(gate|up|down)_proj|\.self_attn\.[qkvo]_proj"
                   r"|\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|^lm_head)$")
NVFP4_PARTS = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32", "input_scale": "F32"}


# hf_ptq.py places the model with accelerate, which sizes the GPU from cudaMemGetInfo. On a
# unified-memory machine (GB10) that is the memory free right now, page cache excluded, so a
# warm page cache pushed part of the 55.6 GB model to disk offload (found 2026-11-03). With
# --gpu-gib the budget is fixed instead: the whole model on the GPU.
RUNNER = r"""
import runpy, sys
import example_utils
gib = float(sys.argv.pop(1))
example_utils.get_max_memory = lambda: {0: int(gib * 2**30), "cpu": 0}
sys.argv = ["hf_ptq.py", *sys.argv[1:]]
runpy.run_path("hf_ptq.py", run_name="__main__")
"""


def quantize(args) -> None:
    ptq = ["--pyt_ckpt_path", str(args.source),
           "--recipe", str(args.recipe),
           "--dataset", str(args.calib),
           "--calib_size", str(args.calib_size),
           "--calib_seq", str(args.calib_seq),
           "--batch_size", str(args.batch_size),
           "--export_path", str(args.out)]
    if args.skip_generate:
        ptq.append("--skip_generate")
    if args.gpu_gib:
        cmd = [sys.executable, "-c", RUNNER, str(args.gpu_gib), *ptq]
    else:
        cmd = [sys.executable, "hf_ptq.py", *ptq]
    print("[quant] hf_ptq.py", " ".join(ptq), f"(GPU budget {args.gpu_gib or 'accelerate'} GiB)", flush=True)
    env = {**os.environ, "PYTHONPATH": str(args.hf_ptq_dir),
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    subprocess.run(cmd, cwd=args.hf_ptq_dir, check=True, env=env)


def finish(src: Path, out: Path) -> None:
    for f in out.iterdir():
        if f.is_file():
            f.chmod(0o644)
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            (out / name).unlink(missing_ok=True)
            shutil.copyfile(src / name, out / name)  # the source's own bytes, not re-serialised


def headers(ckpt: Path) -> dict[str, dict]:
    import struct
    tensors = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        tensors.update(h)
    return tensors


def audit(src: Path, out: Path) -> dict:
    """Fail unless every output tensor has the layout in the docstring."""

    got, base = headers(out), headers(src)
    errors, layout = [], Counter()
    for name, meta in got.items():
        module, _, part = name.rpartition(".")
        if NVFP4.search(module) and not module.startswith("mtp.") and "visual" not in module:
            want = NVFP4_PARTS.get(part)
            if want != meta["dtype"]:
                errors.append(f"{name}: {meta['dtype']}, want {want}")
            layout[f"NVFP4 {part}"] += 1
            continue
        if part in ("k_scale", "v_scale"):
            errors.append(f"{name}: KV scales are not part of this layout")
            continue
        src_meta = base.get(name)
        if src_meta is None:
            errors.append(f"{name}: not in the source")
        elif (src_meta["dtype"], src_meta["shape"]) != (meta["dtype"], meta["shape"]):
            errors.append(f"{name}: {meta['dtype']}{meta['shape']}, source {src_meta['dtype']}{src_meta['shape']}")
        layout[f"unchanged {meta['dtype']}"] += 1
    missing = [n for n in base if n not in got
               and not (NVFP4.search(n.rpartition(".")[0]) and not n.startswith("mtp.") and "visual" not in n)]
    errors += [f"{n}: in the source, missing from the output" for n in missing]
    if not any(n.startswith("mtp.") for n in got):
        errors.append("no MTP head")
    for k, v in sorted(layout.items()):
        print(f"[audit] {v:5d} {k}")
    if errors:
        for e in errors[:40]:
            print("[audit] ERROR", e)
        raise SystemExit(f"[audit] {len(errors)} tensors do not match the layout")
    print(f"[audit] pass: {len(got)} tensors")
    return dict(layout)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(args, layout: dict) -> None:
    import importlib.metadata as md
    ptq = Path(args.hf_ptq_dir)
    commit = subprocess.run(["git", "-C", str(ptq), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    versions = {d: md.version(d) for d in ("nvidia-modelopt", "torch", "transformers", "accelerate", "datasets", "safetensors")}
    out = {
        "source": {"repository": "Qwen/Qwen3.8-27B", "revision": args.source_revision},
        "recipe": {"file": RECIPE.name, "sha256": sha256(Path(args.recipe))},
        "calibration": {"file": Path(args.calib).name, "sha256": sha256(Path(args.calib)),
                        "algorithm": "max", "samples": args.calib_size, "sequence_length": args.calib_seq,
                        "batch_size": args.batch_size},
        "tools": {"model_optimizer_commit": commit, **versions, "python": sys.version.split()[0]},
        "converter": {"file": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "layout": layout,
    }
    (Path(args.out) / "conversion-manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[manifest] {Path(args.out) / 'conversion-manifest.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="Qwen/Qwen3.8-27B checkpoint directory")
    p.add_argument("--source-revision", required=True, help="the source's Hugging Face commit")
    p.add_argument("--out", type=Path, required=True, help="output checkpoint directory")
    p.add_argument("--hf-ptq-dir", type=Path, required=True, help="ModelOpt examples/hf_ptq directory")
    p.add_argument("--calib", type=Path, required=True, help="calibration JSONL with a messages column")
    p.add_argument("--recipe", type=Path, default=RECIPE)
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--calib-seq", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-generate", action="store_true", help="skip hf_ptq's before/after preview")
    p.add_argument("--gpu-gib", type=float, help="fixed GPU memory budget for placing the model "
                   "(unified memory: set it, e.g. 100 on a 128 GB GB10); default: accelerate's reading")
    p.add_argument("--only", choices=("quant", "finish", "audit"), help="run one step")
    args = p.parse_args()
    if args.only in (None, "quant"):
        quantize(args)
    if args.only in (None, "finish"):
        finish(args.source, args.out)
    if args.only in (None, "audit"):
        manifest(args, audit(args.source, args.out))


if __name__ == "__main__":
    main()
