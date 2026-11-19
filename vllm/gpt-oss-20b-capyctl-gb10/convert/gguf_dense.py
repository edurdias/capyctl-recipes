#!/usr/bin/env python3
"""Use b12242's converter/quantizers without changing its installed sources.

Repack native OpenAI MXFP4 experts losslessly; quantize only attention, output
head and embeddings to Q8_0, Q5_0 or Q4_0. All other tensors follow b12242's
BF16 conversion policy (biases, routers, sinks and norms are forced to F32).
No CUDA, engine launch, compiled llama-quantize binary or installation is needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--llamacpp", type=Path, default=Path.home() / "llamacpp-b12242")
    ap.add_argument("--dense-type", choices=["q8_0", "q5_0", "q4_0"], required=True)
    a = ap.parse_args()
    config = json.loads((a.source / "config.json").read_text())
    if config.get("model_type") != "gpt_oss" or config.get("quantization_config", {}).get("quant_method") != "mxfp4":
        raise ValueError("Expected OpenAI's native GPT-OSS MXFP4 checkpoint")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # The module search paths select b12242's own gguf Python implementation.
    root = a.llamacpp.resolve()
    os.environ.pop("NO_LOCAL_GGUF", None)
    sys.path[:0] = [str(root), str(root / "gguf-py")]
    from conversion.gpt_oss import GptOssModel
    from conversion.base import gguf
    dense = re.compile(r"^(?:blk\.\d+\.attn_(?:q|k|v|output)|output|token_embd)\.weight$")
    dtype = getattr(gguf.GGMLQuantizationType, a.dense_type.upper())
    original = GptOssModel.tensor_force_quant

    def dense_quant(self, name, new_name, bid, n_dims):
        if n_dims == 2 and dense.fullmatch(new_name):
            return dtype
        return original(self, name, new_name, bid, n_dims)

    # This changes only the in-process conversion policy, never engine source.
    # Native expert repack still runs GptOssModel.generate_extra_tensors.
    GptOssModel.tensor_force_quant = dense_quant
    spec = importlib.util.spec_from_file_location("gptoss_b12242_converter", root / "convert_hf_to_gguf.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    sys.argv = [str(root / "convert_hf_to_gguf.py"), str(a.source.resolve()),
                "--outfile", str(a.output.resolve()), "--outtype", "bf16"]
    converter.main()


if __name__ == "__main__":
    main()
