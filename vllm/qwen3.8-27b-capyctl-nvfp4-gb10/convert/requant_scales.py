#!/usr/bin/env python3
"""A candidate checkpoint: every NVFP4 weight re-rounded with each 16-weight block's scale chosen to
minimise that block's error (weights only), instead of the block's largest value.

usage: requant_scales.py --source <bf16 dir> --ckpt <nvfp4 dir> --out <new dir>

Kept as they are: weight_scale_2 (per tensor), input_scale (activations), every BF16 tensor, the
config and the tokenizer. Only weight (U8, two E2M1 codes a byte, the even element in the low
nibble) and weight_scale (F8_E4M3, a scale per 16 weights) change. Same layout, same size, same speed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import read_tensors as rt


def requant(w: torch.Tensor, g: float, chunk: int = 1 << 21) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(packed U8 [out, in/2], F8_E4M3 scales [out, in/16], relative error) with searched block scales."""
    out_dim, in_dim = w.shape
    b_all = w.reshape(-1, 16)
    codes = torch.empty_like(b_all, dtype=torch.uint8)
    scales = torch.empty(b_all.shape[0], device=w.device)
    err = 0.0
    mid = rt.E2M1_MID.to(w.device)
    for c0 in range(0, b_all.shape[0], chunk):
        b = b_all[c0:c0 + chunk]
        s0 = b.abs().amax(dim=1) / 6.0 / g
        best = torch.full((b.shape[0],), float("inf"), device=w.device)
        best_s = torch.zeros_like(best)
        for m in rt.MULTS:
            s = rt.fp8(s0 * m).clamp_min(1e-30)
            e = ((rt.round_e2m1(b / (s * g)[:, None]) * (s * g)[:, None] - b) ** 2).sum(dim=1)
            take = e < best
            best, best_s = torch.where(take, e, best), torch.where(take, s, best_s)
        x = b / (best_s * g)[:, None]
        mag = torch.bucketize(x.abs().clamp(max=6.0), mid)  # 0..7, index into E2M1
        sign = ((x < 0) & (mag > 0)).to(torch.uint8)
        codes[c0:c0 + chunk] = mag.to(torch.uint8) | (sign << 3)
        scales[c0:c0 + chunk] = best_s
        err += best.double().sum().item()
    nib = codes.reshape(out_dim, in_dim)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    rel = (err / b_all.double().pow(2).sum().item()) ** 0.5
    return packed.contiguous(), scales.reshape(out_dim, in_dim // 16).to(torch.float8_e4m3fn), rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = rt.Reader(os.path.expanduser(a.source), "cuda")
    ck, out = os.path.expanduser(a.ckpt), os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(ck):  # config, tokenizer, index, manifest: as they are
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(ck, f)):
            shutil.copy2(os.path.join(ck, f), out)
    log = []
    for shard in sorted(glob.glob(os.path.join(ck, "*.safetensors"))):
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(n)).get("__metadata__")
        tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as so:
            for k in so.keys():
                tensors[k] = so.get_tensor(k)
        for k in [k for k in tensors if k.endswith(".weight") and tensors[k].dtype == torch.uint8]:
            base = k[: -len(".weight")]
            g = tensors[base + ".weight_scale_2"].float().item()
            w = src.get(k).float()
            old = rt.unpack_e2m1(tensors[k].cuda()) * tensors[base + ".weight_scale"].cuda().float().repeat_interleave(16, dim=1) * g
            old_rel = rt.stats(w, old)[0]
            packed, scales, rel = requant(w, g)
            # check the written bytes decode to what the search chose
            back = rt.unpack_e2m1(packed) * scales.float().repeat_interleave(16, dim=1) * g
            assert abs(rt.stats(w, back)[0] - rel) < 1e-4, k
            tensors[k], tensors[base + ".weight_scale"] = packed.cpu(), scales.cpu()
            log.append({"tensor": k, "rel_err_before": old_rel, "rel_err_after": rel})
            del w, old, back
            torch.cuda.empty_cache()
        save_file(tensors, os.path.join(out, os.path.basename(shard)), metadata=meta)
        print(f"wrote {os.path.basename(shard)}: {sum(1 for x in log)} tensors so far", flush=True)
        del tensors
    json.dump(log, open(os.path.join(out, "requant-scales.json"), "w"), indent=1)
    b = sum(x["rel_err_before"] for x in log) / len(log)
    c = sum(x["rel_err_after"] for x in log) / len(log)
    print(f"{len(log)} NVFP4 tensors re-rounded: mean relative error {b:.4%} -> {c:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
