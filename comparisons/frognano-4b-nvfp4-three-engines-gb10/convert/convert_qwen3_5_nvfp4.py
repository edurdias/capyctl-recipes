#!/usr/bin/env python3
"""Make a ModelOpt NVFP4 / FP8 mixed-precision checkpoint of a dense qwen3_5 model.

Target layout is the one nvidia/Qwen3.8-27B-NVFP4 ships (quant_method "modelopt",
quant_algo "MIXED_PRECISION"):

  - MLP gate/up/down and lm_head: NVFP4 W4A4, group size 16 (static input_scale)
  - self_attn q/k/v/o and linear_attn in_proj_qkv / in_proj_z / out_proj: FP8 per-tensor W8A8
  - BF16: embed_tokens, linear_attn in_proj_a / in_proj_b, conv1d, A_log, dt_bias, norms,
    vision tower (model.visual.*), MTP head (mtp.*)
  - KV cache: not quantized

Steps:
  1. quant   - ModelOpt examples/hf_ptq/hf_ptq.py with the recipe next to this script. A tied
               head is untied after loading (lm_head.weight = copy of embed_tokens,
               tie_word_embeddings false): TensorFold's CUDA loader needs a separate lm_head.
  2. finish  - original tokenizer files and chat template, generation_config.json with the
               given eos ids, config checks.

Run with the ModelOpt venv's python. Needs a CUDA GPU for calibration (no FP4 hardware needed).

Example:
  python convert_qwen3_5_nvfp4.py \
      --source /path/to/FrogNano-4B-2609 \
      --out /path/to/FrogNano-4B-2609-NVFP4 \
      --hf-ptq-dir /path/to/Model-Optimizer-0.47.0/examples/hf_ptq \
      --calib /path/to/calib.jsonl \
      --eos 248046 248044
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "qwen3_5-nvfp4_mlp-fp8_attn-kv_none.yaml"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
                   "preprocessor_config.json", "video_preprocessor_config.json")


RUNNER = r"""
import runpy, sys
import torch
import example_utils

_get_model = example_utils.get_model


def untie(model):
    # TensorFold's CUDA loader reads a separate lm_head; give the head its own copy of the embedding.
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if head is not None and head.weight.data_ptr() == emb.weight.data_ptr():
        head.weight = torch.nn.Parameter(emb.weight.detach().clone(), requires_grad=False)
        print(f"[untie] lm_head.weight = copy of embed_tokens {tuple(head.weight.shape)}", flush=True)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.tie_word_embeddings = False
    return model


def get_model(*a, **k):
    return untie(_get_model(*a, **k))


example_utils.get_model = get_model
sys.argv = ["hf_ptq.py", *sys.argv[1:]]
runpy.run_path("hf_ptq.py", run_name="__main__")
"""


def quantize(args, out: Path) -> None:
    """hf_ptq.py with the recipe; the loaded model is untied before calibration."""

    cmd = [sys.executable, "-c", RUNNER,
           "--pyt_ckpt_path", str(args.source),
           "--recipe", str(args.recipe),
           "--dataset", str(args.calib),
           "--calib_size", str(args.calib_size),
           "--calib_seq", str(args.calib_seq),
           "--batch_size", str(args.batch_size),
           "--export_path", str(out)]
    if args.skip_generate:
        cmd.append("--skip_generate")
    print("[quant] hf_ptq.py", " ".join(cmd[3:]), flush=True)
    # expandable segments: the lm_head export (248320 rows) OOMs a 16 GB card on fragmentation otherwise
    env = {**os.environ, "PYTHONPATH": str(args.hf_ptq_dir),
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    subprocess.run(cmd, cwd=args.hf_ptq_dir, check=True, env=env)


def finish(src: Path, out: Path, eos: list[int]) -> None:
    for f in out.iterdir():
        if f.is_file():
            f.chmod(0o644)
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            (out / name).unlink(missing_ok=True)            # sidecars copied from a cache can be read-only
            shutil.copyfile(src / name, out / name)          # the source's own files, not re-serialised ones
    gen = {"eos_token_id": eos}
    old = out / "generation_config.json"
    if old.exists():
        gen = {**json.loads(old.read_text()), **gen}
    old.write_text(json.dumps(gen, indent=2) + "\n")
    cfg = json.loads((out / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    check(out)


def check(out: Path) -> None:
    """Print the layout summary and fail on things the three engines would refuse."""

    import re
    from collections import Counter

    cfg = json.loads((out / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    hq = json.loads((out / "hf_quant_config.json").read_text())["quantization"]
    assert qc.get("quant_method") == "modelopt", qc.get("quant_method")
    assert hq.get("kv_cache_quant_algo") in (None, "null"), hq.get("kv_cache_quant_algo")
    pat = Counter((re.sub(r"\.\d+\.", ".N.", k), json.dumps(v, sort_keys=True))
                  for k, v in (hq.get("quantized_layers") or {}).items())
    print(f"[check] quant_algo={hq.get('quant_algo')} exclude={hq.get('exclude_modules')}")
    for (k, v), n in sorted(pat.items()):
        print(f"[check] {n:4d} {k} {v}")
    index = out / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())["weight_map"]
    else:                                                   # one shard: names from its header
        from safetensors import safe_open
        with safe_open(str(out / "model.safetensors"), framework="pt") as f:
            idx = dict.fromkeys(f.keys(), "model.safetensors")
    assert "lm_head.weight" in idx, "no separate lm_head"
    assert not any(k.endswith(("k_scale", "v_scale")) for k in idx), "KV scales present"
    print(f"[check] tensors={len(idx)} tie_word_embeddings={cfg.get('tie_word_embeddings')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="BF16 Hugging Face checkpoint directory")
    p.add_argument("--out", type=Path, required=True, help="output checkpoint directory")
    p.add_argument("--hf-ptq-dir", type=Path, required=True, help="ModelOpt examples/hf_ptq directory")
    p.add_argument("--calib", type=Path, required=True, help="calibration JSONL with a messages column")
    p.add_argument("--recipe", type=Path, default=RECIPE)
    p.add_argument("--calib-size", type=int, default=256)
    p.add_argument("--calib-seq", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eos", type=int, nargs="+", required=True, help="generation_config eos_token_id list")
    p.add_argument("--skip-generate", action="store_true", help="skip hf_ptq's before/after preview")
    p.add_argument("--only", choices=("quant", "finish"), help="run one step")
    args = p.parse_args()
    if args.only in (None, "quant"):
        quantize(args, args.out)
    if args.only in (None, "finish"):
        finish(args.source, args.out, args.eos)


if __name__ == "__main__":
    main()
