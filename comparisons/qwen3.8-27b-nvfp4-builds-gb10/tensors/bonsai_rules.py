#!/usr/bin/env python3
"""Which rounding rule Bonsai's trits follow, on a sample of tensors: agreement with (a) round(clip(w_f / d))
using Bonsai's own scale d, (b) BitNet's absmean rule (s = mean |w_f| per 128, t = round(clip(w_f / s))),
(c) the error-optimal ternary. w_f is the BF16 weight in Bonsai's folded basis, W D H."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_bonsai import GLOBAL, MAPPING, Q2_0, optimal_ternary, q2_0, read_gguf, rotate_rows, hadamard, vperm  # noqa
from read_tensors import Reader  # noqa: E402

src_dir, gguf = os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2])
dev = "cuda"
kv, infos, mm = read_gguf(gguf)
signs, o = {}, 0
for w in kv["prism.hadamard.sign_widths"]:
    signs[w] = torch.tensor(kv["prism.hadamard.sign_values"][o:o + w], dtype=torch.float32, device=dev)
    o += w
nv, nk = int(kv["qwen35.ssm.time_step_rank"]), int(kv["qwen35.ssm.group_count"])
hd, hk = int(kv["qwen35.ssm.inner_size"]) // nv, int(kv["qwen35.ssm.state_size"])
H = hadamard(dev)
src = Reader(src_dir, dev)
sample = ["blk.0.attn_qkv.weight", "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.20.ffn_down.weight",
          "blk.40.ffn_up.weight", "blk.55.attn_k.weight", "blk.63.ssm_out.weight", "output.weight"]
print("tensor                  own-scale-round  absmean  optimal  | zeros: bonsai absmean | d/absmean")
for name in sample:
    dims, _, off = infos[name]
    if name in GLOBAL:
        hf, stem = GLOBAL[name], name
    else:
        _, layer, stem = name.split(".", 2)
        hf = f"model.language_model.layers.{layer}." + MAPPING[stem]
    w = src.get(hf).float()
    t, d, _ = q2_0(mm, dims, off, dev)
    if nv != nk and stem in ("attn_qkv.weight", "attn_gate.weight"):
        qk = 2 * nk * hk if stem == "attn_qkv.weight" else 0
        p = torch.cat([torch.arange(qk, device=dev), qk + vperm(nv, nk, hd, dev)])
        t, d = t[p], d[p]
    n = a1 = a2 = a3 = zb = za = 0.0
    ratio = []
    for r0 in range(0, w.shape[0], 4096):
        sl = slice(r0, r0 + 4096)
        wf = rotate_rows(w[sl] * signs[w.shape[1]], H)
        tb = t[sl].float()
        db = d[sl].float().repeat_interleave(64, dim=1)
        own = torch.clamp(torch.round(wf / db), -1, 1)
        g = wf.reshape(-1, 128)
        s = g.abs().mean(1, keepdim=True)
        am = torch.clamp(torch.round(g / s), -1, 1).reshape(wf.shape)
        to, _ = optimal_ternary(wf)
        n += tb.numel()
        a1 += (own == tb).sum().item()
        a2 += (am == tb).sum().item()
        a3 += (to == tb).sum().item()
        zb += (tb == 0).sum().item()
        za += (am == 0).sum().item()
        ratio.append((d[sl, 0::2].float().reshape(-1) / s.reshape(-1)).cpu())
    print(f"{name:24} {a1 / n * 100:14.1f}% {a2 / n * 100:7.1f}% {a3 / n * 100:7.1f}%  | "
          f"{zb / n * 100:5.1f}% {za / n * 100:5.1f}%        | {torch.cat(ratio).median().item():.3f}", flush=True)
    del w, t, d
    torch.cuda.empty_cache()
