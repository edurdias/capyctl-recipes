#!/usr/bin/env python3
"""Count ModelOpt 0.47.0's sufficient lossless MXFP4->NVFP4 scale window.

CPU only; reads E8M0 scales, never packed weights or CUDA. Mirrors installed
modelopt/torch/quantization/utils/numeric_utils.py:61-101. Exponent byte zero
is treated as an all-zero block, as in that implementation. This is a cast
representability diagnostic, not proof of an exported checkpoint's accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    a = ap.parse_args()
    started = time.monotonic()
    index_path = a.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    rows = []
    for name, shard in sorted(index["weight_map"].items()):
        if not name.endswith("_scales") or ".mlp.experts." not in name:
            continue
        with safe_open(str(a.source / shard), framework="np", device="cpu") as f:
            scales = f.get_tensor(name)
        histogram = np.bincount(scales.reshape(-1), minlength=256)
        if histogram[255]:
            raise ValueError(f"Invalid E8M0 NaN exponent in {name}")
        positive = np.flatnonzero(histogram[1:]) + 1
        emax = int(positive[-1]) if len(positive) else 127
        emin = int(positive[0]) if len(positive) else 127
        n = int(scales.size)
        zeros = int(histogram[0])
        lossless = zeros + int(histogram[max(1, emax - 17):255].sum())
        rows.append({"name": name, "shape": list(scales.shape), "k_min": emin - 127,
                     "k_max": emax - 127, "m": emax - 127 - 8,
                     "n_total_blocks": n, "n_zero_blocks": zeros,
                     "n_lossless_blocks": lossless, "pct_lossless": 100 * lossless / n,
                     "exponent_histogram": histogram.tolist()})
    total = sum(r["n_total_blocks"] for r in rows)
    if not total:
        raise ValueError("No native MXFP4 expert scales found")
    lossless = sum(r["n_lossless_blocks"] for r in rows)
    result = {
        "method": "ModelOpt closed-form scale window: E8M0 zero or k >= k_max - 17, per source tensor",
        "device": "CPU; scales only", "scale_tensors": len(rows), "scale_bytes_read": total,
        "n_total_blocks": total, "n_lossless_blocks": lossless,
        "pct_lossless": 100 * lossless / total,
        "fully_lossless_tensors": sum(r["n_total_blocks"] == r["n_lossless_blocks"] for r in rows),
        "source_config_sha256": hashlib.sha256((a.source / "config.json").read_bytes()).hexdigest(),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic() - started, "tensors": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
