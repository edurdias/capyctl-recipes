#!/usr/bin/env python3
"""Rewrite a safetensors checkpoint into smaller files, every tensor unchanged.

usage: reshard.py SRC DST [MAX_GIB]

SRC is a checkpoint directory with model.safetensors.index.json. DST gets the
same tensors (same names, dtypes, shapes and bytes) in files of at most
MAX_GIB GiB each (default 2; a tensor larger than that gets a file of its
own), a new index, and a copy of every other file. Needs torch and safetensors.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

ELEMENT_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cap = int(float(sys.argv[3] if len(sys.argv) > 3 else 2) * 2**30)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if os.path.isfile(path) and not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            shutil.copy2(path, os.path.join(dst, name))

    # Plan the files first, so each file name carries the total count.
    groups, current, used, meta = [], [], 0, None
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(os.path.join(src, shard), "pt") as f:
            meta = meta or f.metadata()
            for name in sorted(f.keys()):
                t = f.get_slice(name)
                nbytes = ELEMENT_BYTES[t.get_dtype()]
                for d in t.get_shape():
                    nbytes *= d
                if current and used + nbytes > cap:
                    groups.append(current)
                    current, used = [], 0
                current.append((shard, name))
                used += nbytes
    groups.append(current)

    weight_map = {}
    for i, group in enumerate(groups, 1):
        out = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        for shard, name in group:
            with safe_open(os.path.join(src, shard), "pt") as f:
                tensors[name] = f.get_tensor(name)
            weight_map[name] = out
        save_file(tensors, os.path.join(dst, out), metadata=meta)
        del tensors
        print(out, flush=True)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": weight_map}, f, indent=2)
    print(f"{len(weight_map)} tensors in {len(groups)} files")


if __name__ == "__main__":
    main()
