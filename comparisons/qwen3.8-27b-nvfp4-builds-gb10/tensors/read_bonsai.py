#!/usr/bin/env python3
"""Weights only: how far PrismML's Ternary Bonsai 2 27B moved Qwen3.8-27B's weights, and how that compares
with the best plain ternary rounding of the same weights.

usage: read_bonsai.py --source <bf16 dir> --gguf <Ternary-Bonsai-2-27B-Q2_0.gguf> --out <csv>

The GGUF (Q2_0 packing: 64 weights a block, an FP16 scale then 16 bytes of 2-bit codes, value (q - 1) * d)
stores each listed matrix folded: W_q = W D H, D the explicit +-1 sign vector for the input width, H the
normalized Sylvester Walsh-Hadamard in 1024-wide blocks along the input. The runtime computes
y = W_q H (D x) (llama.cpp #29600, PrismML's MLX runtime `fwht`), so the original-basis weight is W_q H D:
each row Hadamard-transformed, then multiplied by the signs. token_embd (the inverse-after-lookup table)
reverts the same way. Row order follows PrismML's MLX runtime `reorder`: the GDN value heads of attn_qkv and
attn_gate are tiled in the GGUF and restored here to the Hugging Face grouped order; ssm_out is kept grouped
(`prism.hadamard.gdn_v_grouped`).

Per tensor, besides Bonsai's own error: Bonsai's trits with the least-squares scale per 128 (is the gap in
the scales?), the optimal ternary rounding per 128 in the same rotated basis (what rounding alone gets), the
same without the rotation, and how often Bonsai's trits equal the optimal rounding's. Nothing runs through
the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_tensors import Reader, gain_of, gain_rel_err, kind_of, layer_of, stats, top_share  # noqa: E402

Q2_0, BF16, F32 = 42, 30, 0
BLOCK = 1024
ROWS = 4096

MAPPING = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
GLOBAL = {"output.weight": "lm_head.weight", "token_embd.weight": "model.language_model.embed_tokens.weight"}


def read_gguf(path: str):
    """KV metadata, tensor infos (name -> (dims, type, absolute offset)) and a byte memmap."""
    f = open(path, "rb")

    def u(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    def s():
        return f.read(u("Q")).decode()

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = u("I"), u("Q")
            if et in scalar:
                fmt = scalar[et]
                return list(struct.unpack(f"<{n}{fmt}", f.read(n * struct.calcsize(fmt))))
            return [value(et) for _ in range(n)]
        return u(scalar[t])

    assert f.read(4) == b"GGUF"
    version, n_t, n_kv = u("I"), u("Q"), u("Q")
    assert version == 3, version
    kv = {}
    for _ in range(n_kv):
        k = s()
        kv[k] = value(u("I"))
    infos = {}
    for _ in range(n_t):
        name = s()
        dims = [u("Q") for _ in range(u("I"))]
        infos[name] = (dims, u("I"), u("Q"))
    align = kv.get("general.alignment", 32)
    start = (f.tell() + align - 1) // align * align
    f.close()
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return kv, {k: (d, t, start + o) for k, (d, t, o) in infos.items()}, mm


def q2_0(mm, dims, off, dev):
    """Trits [out, in] (int8) and the FP16 scale of each 64-weight block [out, in / 64]; the count of code 3."""
    n_in, n_out = dims
    blocks = n_in * n_out // 64
    raw = torch.from_numpy(np.array(mm[off:off + blocks * 18])).to(dev).view(blocks, 18)
    d = raw[:, :2].contiguous().view(torch.float16).reshape(n_out, n_in // 64)
    qs = raw[:, 2:].to(torch.int16)
    codes = torch.stack([(qs >> sh) & 3 for sh in (0, 2, 4, 6)], dim=2).reshape(n_out, n_in)
    return (codes - 1).to(torch.int8), d, int((codes == 3).sum().item())


def hadamard(dev) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < BLOCK:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(BLOCK)).float().to(dev)


def rotate_rows(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Blockwise H along the last dimension (H is symmetric and its own inverse)."""
    r, c = x.shape
    return (x.reshape(-1, BLOCK) @ H).reshape(r, c)


def vperm(nv: int, nk: int, unit: int, dev) -> torch.Tensor:
    return torch.arange(nv * unit, device=dev).reshape(nv // nk, nk, unit).transpose(0, 1).reshape(-1)


def optimal_ternary(x: torch.Tensor, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Per group, the trits t and scale s minimising ||x - s t||: for the k largest |x| set to sign(x),
    s = mean of those |x|, the error is ||x||^2 - (sum of top-k |x|)^2 / k; take the best k."""
    r, c = x.shape
    g = x.reshape(-1, group)
    a, _ = g.abs().sort(dim=1, descending=True)
    cs = a.cumsum(dim=1)
    k = torch.arange(1, group + 1, device=x.device, dtype=x.dtype)
    best = (cs * cs / k).argmax(dim=1)
    s = cs.gather(1, best[:, None]) / (best[:, None] + 1).to(x.dtype)
    thr = a.gather(1, best[:, None])
    t = torch.where(g.abs() >= thr, torch.sign(g), torch.zeros_like(g))
    return t.reshape(r, c), s.reshape(r, c // group)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N tensors only (smoke run)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kv, infos, mm = read_gguf(os.path.expanduser(a.gguf))
    assert kv["prism.hadamard.block_size"] == BLOCK and kv["prism.hadamard.sign_mode"] == "explicit"
    assert kv["prism.hadamard.transform"] == "normalized-sylvester-walsh-hadamard"
    assert kv.get("prism.hadamard.gdn_v_grouped") is True
    signs, o = {}, 0
    for w in kv["prism.hadamard.sign_widths"]:
        signs[w] = torch.tensor(kv["prism.hadamard.sign_values"][o:o + w], dtype=torch.float32, device=dev)
        o += w
    folded = set(kv["prism.hadamard.weight_names"]) | set(kv["prism.hadamard.inverse_weight_names"])
    nv, nk = int(kv["qwen35.ssm.time_step_rank"]), int(kv["qwen35.ssm.group_count"])
    hd = int(kv["qwen35.ssm.inner_size"]) // nv
    hk = int(kv["qwen35.ssm.state_size"])
    H = hadamard(dev)
    src = Reader(os.path.expanduser(a.source), dev)
    names = sorted(n for n, (_, t, _) in infos.items() if t == Q2_0)
    if a.limit:
        names = names[: a.limit]
    print(f"{len(names)} Q2_0 tensors, {len(folded)} folded; nv {nv} nk {nk} hd {hd} hk {hk}", flush=True)
    rows = []
    for i, name in enumerate(names):
        dims, _, off = infos[name]
        if name in GLOBAL:
            hf, stem = GLOBAL[name], name
        else:
            _, layer, stem = name.split(".", 2)
            hf = f"model.language_model.layers.{layer}." + MAPPING[stem]
        assert name in folded, name
        w = src.get(hf).float()
        t, d, code3 = q2_0(mm, dims, off, dev)
        if nv != nk and stem in ("attn_qkv.weight", "attn_gate.weight"):
            qk = 2 * nk * hk if stem == "attn_qkv.weight" else 0
            p = torch.cat([torch.arange(qk, device=dev), qk + vperm(nv, nk, hd, dev)])
            t, d = t[p], d[p]
        assert tuple(t.shape) == tuple(w.shape), (name, t.shape, w.shape)
        sg = signs[w.shape[1]]
        n_out, n_in = w.shape
        # Bonsai's scales per 64-weight block: do neighbours share one scale (group 128)?
        pairs_equal = (d[:, 0::2] == d[:, 1::2]).float().mean().item()
        rec = torch.empty_like(w)  # Bonsai in the original basis
        refit = torch.empty_like(w)  # Bonsai's trits, least-squares scale per 128
        opt = torch.empty_like(w)  # optimal ternary rounding, rotated basis, back in the original
        plain = torch.empty_like(w)  # optimal ternary rounding without the rotation
        agree = zeros_b = zeros_o = 0.0
        ratios = []
        for r0 in range(0, n_out, ROWS):
            sl = slice(r0, r0 + ROWS)
            tb = t[sl].float()
            wq = tb * d[sl].float().repeat_interleave(64, dim=1)
            rec[sl] = rotate_rows(wq, H) * sg
            wf = rotate_rows(w[sl] * sg, H)  # W D H: the folded basis Bonsai's trits live in
            g = wf.reshape(-1, 128)
            tg = tb.reshape(-1, 128)
            s_ls = (g * tg).sum(1, keepdim=True) / (tg * tg).sum(1, keepdim=True).clamp_min(1)
            refit[sl] = rotate_rows((tg * s_ls).reshape(wf.shape), H) * sg
            to, so = optimal_ternary(wf)
            opt[sl] = rotate_rows(to * so.repeat_interleave(128, dim=1), H) * sg
            tp, sp = optimal_ternary(w[sl])
            plain[sl] = tp * sp.repeat_interleave(128, dim=1)
            agree += (to == tb).float().sum().item()
            zeros_b += (tb == 0).float().sum().item()
            zeros_o += (to == 0).float().sum().item()
            sb = d[sl, 0::2].float().reshape(-1)
            ratios.append((sb / so.reshape(-1).clamp_min(1e-30)).cpu())
        numel = w.numel()
        row = {"tensor": hf, "gguf": name, "layer": layer_of(hf), "kind": kind_of(hf) if hf != GLOBAL["token_embd.weight"] else "embed",
               "numel": numel, "in_dim": n_in}
        row["rel_err"], row["cos"], row["worst_row_rel_err"] = stats(w, rec)
        row["rel_err_refit_scale"] = stats(w, refit)[0]
        row["rel_err_opt_rotated"] = stats(w, opt)[0]
        row["rel_err_opt_plain"] = stats(w, plain)[0]
        row["trit_agree_opt"] = agree / numel
        row["zero_frac"], row["zero_frac_opt"] = zeros_b / numel, zeros_o / numel
        row["scale_ratio_median"] = torch.cat(ratios).median().item()
        row["scale_pairs_equal"] = pairs_equal
        row["code3"] = code3
        c = gain_of(src, hf, n_in)
        c2 = c.pow(2) if c is not None else None
        row["gain_rel_err"] = gain_rel_err(w, rec, c2) if c2 is not None else ""
        row["gain_rel_err_opt_rotated"] = gain_rel_err(w, opt, c2) if c2 is not None else ""
        if hf.endswith("embed_tokens.weight") or hf == "lm_head.weight":
            row["top32_err_share"] = row["top32_err_share_opt"] = row["top32_uniform_share"] = ""
        else:
            row["top32_err_share"], row["top32_uniform_share"] = top_share(w, rec)
            row["top32_err_share_opt"] = top_share(w, opt)[0]
        rows.append(row)
        print(f"{i + 1}/{len(names)} {hf} err {row['rel_err']:.4f} refit {row['rel_err_refit_scale']:.4f} "
              f"opt {row['rel_err_opt_rotated']:.4f} plain {row['rel_err_opt_plain']:.4f} "
              f"agree {row['trit_agree_opt']:.3f} pairs {pairs_equal:.3f} code3 {code3}", flush=True)
        del w, t, d, rec, refit, opt, plain
        if dev == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
