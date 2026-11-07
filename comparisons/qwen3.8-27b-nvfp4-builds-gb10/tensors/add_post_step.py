"""Record the block scale search in the candidate's conversion-manifest.json."""
import json
import sys

path = sys.argv[1]
m = json.load(open(path))
m["post_steps"] = [{
    "name": "block scale search",
    "base": ("capyctl/Qwen3.8-27B-NVFP4 revision 4a47ea5eaf0bbc80ede6b392aa94e0dcc1980ab6 "
             "(weights as 8b03c818950c8b92ebf4e66e06432cdd6618bacf)"),
    "method": ("every NVFP4 weight re-rounded from the BF16 source: each 16-weight block takes, of 21 scales "
               "from 0.60x to 1.10x of its max-calibrated scale (FP8 E4M3-rounded), the one that minimises the "
               "block's squared error; weight_scale_2, input_scale, every BF16 tensor, the config and the "
               "tokenizer unchanged"),
    "script": {"path": "vllm/qwen3.8-27b-capyctl-nvfp4-gb10/convert/requant_scales.py",
               "sha256": "44e477dd1c58ca72766da28cbc088300f748832a740e3699931c03c72ffaffac"},
    "imports": {"path": "vllm/qwen3.8-27b-capyctl-nvfp4-gb10/convert/read_tensors.py",
                "sha256": "0b21767fbd6456d6a9808d8ce48c921d1412e7a0a0fad874587756ff5d49f363"},
    "tools": {"torch": "2.13.0+cu130", "safetensors": "0.8.0"},
    "tensors": 401,
    "mean_relative_weight_error": {"before": 0.094828, "after": 0.085910},
    "per_tensor": "requant-scales.json",
}]
json.dump(m, open(path, "w"), indent=2)
print("post_steps added:", m["post_steps"][0]["mean_relative_weight_error"])
