#!/usr/bin/env python3
"""Construct MXFP4-expert GGUF K-quant variants using an existing quantizer.

Requires a preexisting approved llama-quantize executable; none was found in
build/bin of the three approved installs. Never builds or installs anything.
GPT-OSS's 2880-wide rows force Q6_K/Q5_K/Q4_K to fall back to Q8_0/Q5_1/Q5_0;
only attention output projections (4096 contracting width) use the K type.
See source-feasibility.md before interpreting the filenames.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output_directory", type=Path)
    ap.add_argument("--quantizer", type=Path, required=True)
    ap.add_argument("--llamacpp", type=Path, default=Path.home() / "llamacpp-b12242")
    a = ap.parse_args()
    if not a.quantizer.is_file():
        raise FileNotFoundError("An existing approved llama-quantize executable is required")
    a.output_directory.mkdir(parents=True, exist_ok=True)
    intermediate = a.output_directory / "gpt-oss-20b-MXFP4-BF16-dense.gguf"
    outputs = {q: a.output_directory / f"gpt-oss-20b-MXFP4-dense-{q}-fallback.gguf"
               for q in ("q6_k", "q5_k", "q4_k")}
    if intermediate.exists() or any(p.exists() for p in outputs.values()):
        raise FileExistsError("Use a new output directory; this script does not overwrite GGUFs")
    # sys.executable must be the approved ModelOpt Python, with installed
    # torch/numpy/transformers/safetensors/tokenizer dependencies.
    subprocess.run([sys.executable, str(a.llamacpp / "convert_hf_to_gguf.py"),
                    str(a.source), "--outfile", str(intermediate), "--outtype", "bf16"], check=True)
    for quant, output in outputs.items():
        # Do not pass --output-tensor-type or --token-embedding-type: those
        # bypass b12242's shape fallback. Regex overrides retain the checks.
        # Native MXFP4 experts select the same type and are byte-copied; no
        # --allow-requantize is needed or wanted.
        subprocess.run([
            str(a.quantizer),
            "--tensor-type", r"(ffn_gate|ffn_up|ffn_down)_exps\.weight=mxfp4",
            "--tensor-type", r"^(blk\.[0-9]+\.attn_(q|k|v|output)|output|token_embd)\.weight$=" + quant,
            str(intermediate), str(output), "MXFP4_MOE",
        ], check=True)


if __name__ == "__main__":
    main()
