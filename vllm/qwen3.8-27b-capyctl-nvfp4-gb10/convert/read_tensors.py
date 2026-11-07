#!/usr/bin/env python3
"""Weights only: dequantize every quantized tensor of a checkpoint and compare it with the BF16 source.

usage: read_tensors.py --source <bf16 dir> --ckpt <quantized dir> --name <label> --out <csv>
       [--search]   also the weights-only per-block scale search (NVFP4 tensors)

Formats (ModelOpt exports):
  NVFP4: weight U8 [out, in/2] (two E2M1 values a byte, the even element in the low nibble),
         weight_scale F8_E4M3 [out, in/16] (a scale per 16 weights), weight_scale_2 F32 (per tensor);
         W ~= e2m1 * weight_scale * weight_scale_2.
  FP8:   weight F8_E4M3 [out, in], weight_scale F32 (per tensor); W ~= fp8 * weight_scale.
Nothing runs through the model: no activations, only the safetensors files.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import struct

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # rounding thresholds
FP8_MIN_NORMAL = 2.0 ** -6
MULTS = [0.60 + 0.025 * i for i in range(21)]  # 0.60 .. 1.10


def index(d: str) -> dict[str, str]:
    """tensor name -> shard file."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__":
                out[k] = f
    return out


class Reader:
    def __init__(self, d: str, dev: str):
        self.idx = index(d)
        self.dev = dev
        self.open: dict[str, object] = {}

    def get(self, k: str) -> torch.Tensor:
        f = self.idx[k]
        if f not in self.open:
            self.open[f] = safe_open(f, framework="pt", device="cpu")
        return self.open[f].get_tensor(k).to(self.dev)

    def has(self, k: str) -> bool:
        return k in self.idx


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """U8 [out, in/2] -> float [out, in] E2M1 values."""
    lut = E2M1.to(packed.device)
    lo, hi = packed & 0x0F, packed >> 4

    def val(n):
        mag = lut[(n & 0x07).long()]
        return torch.where((n & 0x08) > 0, -mag, mag)

    return torch.stack([val(lo), val(hi)], dim=-1).reshape(packed.shape[0], -1)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 value (saturating at 6)."""
    lut, mid = E2M1.to(x.device), E2M1_MID.to(x.device)
    return torch.sign(x) * lut[torch.bucketize(x.abs().clamp(max=6.0), mid)]


def fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(max=448.0).to(torch.float8_e4m3fn).float()


def kind_of(name: str) -> str:
    for pat, k in (("mlp.gate_proj", "mlp.gate"), ("mlp.up_proj", "mlp.up"), ("mlp.down_proj", "mlp.down"),
                   ("self_attn.q_proj", "attn.q"), ("self_attn.k_proj", "attn.k"), ("self_attn.v_proj", "attn.v"),
                   ("self_attn.o_proj", "attn.o"), ("in_proj_qkv", "deltanet.qkv"), ("in_proj_z", "deltanet.z"),
                   ("linear_attn.out_proj", "deltanet.out"), ("lm_head", "lm_head")):
        if pat in name:
            return k
    return "other"


def layer_of(name: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def stats(w: torch.Tensor, wq: torch.Tensor, rows: int = 4096) -> tuple[float, float, float]:
    """Relative error, cosine and the worst output row's relative error, `rows` rows at a time:
    lm_head is 1.27 billion weights, tens of GB as one float64 copy."""
    err2 = ww = qq = wq_dot = 0.0
    worst = 0.0
    for r0 in range(0, w.shape[0], rows):
        a, b = w[r0:r0 + rows].double(), wq[r0:r0 + rows].double()
        d = b - a
        err2 += (d * d).sum().item()
        ww += (a * a).sum().item()
        qq += (b * b).sum().item()
        wq_dot += (a * b).sum().item()
        # the worst output row: an output whose weights the format holds worst
        worst = max(worst, (d.norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)).max().item())
    return (err2 / ww) ** 0.5, wq_dot / (ww * qq) ** 0.5, worst


def nvfp4_search(w: torch.Tensor, g: float, c2: torch.Tensor | None = None, chunk: int = 1 << 21) -> tuple:
    """Each 16-weight block's scale chosen, from MULTS x the max-calibrated scale (FP8-rounded), to
    minimise that block's squared error, weighted per input channel by `c2` (the squared norm gain)
    when given. Returns (relative error, gain-weighted relative error) of the result. Weights only;
    blocks are taken `chunk` at a time so the temporaries stay small (GPU memory is the machine's)."""
    out_dim, in_dim = w.shape
    b_all = w.reshape(-1, 16)
    cw = (c2 if c2 is not None else torch.ones(in_dim, device=w.device)).reshape(-1, 16)  # [in/16, 16]
    per_row = in_dim // 16
    e_plain = e_gain = 0.0
    for c0 in range(0, b_all.shape[0], chunk):
        b = b_all[c0:c0 + chunk]
        cb = cw[(torch.arange(c0, c0 + b.shape[0], device=w.device) % per_row)]
        s0 = b.abs().amax(dim=1) / 6.0 / g
        best = torch.full((b.shape[0],), float("inf"), device=w.device)
        best_plain = torch.zeros_like(best)
        for m in MULTS:
            s = fp8(s0 * m).clamp_min(1e-30)
            d2 = (round_e2m1(b / (s * g)[:, None]) * (s * g)[:, None] - b) ** 2
            obj = (d2 * cb).sum(dim=1)
            take = obj < best
            best = torch.where(take, obj, best)
            best_plain = torch.where(take, d2.sum(dim=1), best_plain)
        e_plain += best_plain.double().sum().item()
        e_gain += best.double().sum().item()
    base_plain = b_all.double().pow(2).sum().item()
    base_gain = (b_all.double().pow(2) * cw.double().repeat(out_dim, 1)).sum().item()
    return (e_plain / base_plain) ** 0.5, (e_gain / base_gain) ** 0.5


def gain_of(src: "Reader", name: str, in_dim: int) -> torch.Tensor | None:
    """The norm gain on this tensor's input channels, when one norm sets them (weights only).
    Qwen3.5's layer norms are GemmaRMSNorm (gain 1 + w); DeltaNet's output norm is RMSNormGated (gain w),
    one 128-wide gain repeated for each value head. o_proj and down_proj inputs pass no norm."""
    m = re.match(r"(.*\.layers\.\d+)\.", name)
    pre = m.group(1) if m else ""
    if re.search(r"self_attn\.[qkv]_proj|linear_attn\.in_proj_(qkv|z)", name):
        return 1.0 + src.get(pre + ".input_layernorm.weight").float()
    if re.search(r"mlp\.(gate|up)_proj", name):
        return 1.0 + src.get(pre + ".post_attention_layernorm.weight").float()
    if name.startswith("lm_head"):
        return 1.0 + src.get("model.language_model.norm.weight").float()
    if "linear_attn.out_proj" in name:
        gw = src.get(pre + ".linear_attn.norm.weight").float()
        return gw.repeat(in_dim // gw.numel())
    return None


def gain_rel_err(w: torch.Tensor, wq: torch.Tensor, c2: torch.Tensor, rows: int = 4096) -> float:
    e = b = 0.0
    for r0 in range(0, w.shape[0], rows):
        a, q = w[r0:r0 + rows].double(), wq[r0:r0 + rows].double()
        e += ((q - a).pow(2) * c2.double()).sum().item()
        b += (a.pow(2) * c2.double()).sum().item()
    return (e / b) ** 0.5


def top_share(w: torch.Tensor, wq: torch.Tensor, k: int = 32) -> tuple[float, float]:
    """Share of the error's energy on W's top-k input directions (right singular vectors), and the
    share uniform noise would put there (k / in). A ratio above 1 means the error lines up with the
    directions that carry most of what the tensor does."""
    _, _, v = torch.svd_lowrank(w, q=k + 8, niter=3)
    v = v[:, :k]
    num = den = 0.0
    for r0 in range(0, w.shape[0], 4096):
        d = wq[r0:r0 + 4096] - w[r0:r0 + 4096]
        num += (d @ v).pow(2).sum().item()
        den += d.pow(2).sum().item()
    return num / den, k / w.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src, q = Reader(os.path.expanduser(a.source), dev), Reader(os.path.expanduser(a.ckpt), dev)
    names = sorted(k[: -len(".weight_scale")] for k in q.idx if k.endswith(".weight_scale"))
    rows = []
    for i, base in enumerate(names):
        wk = base + ".weight"
        if not src.has(wk):
            print(f"skip {wk}: not in the source")
            continue
        w = src.get(wk).float()
        qw, sc = q.get(wk), q.get(base + ".weight_scale")
        row = {"build": a.name, "tensor": wk, "layer": layer_of(wk), "kind": kind_of(wk), "numel": w.numel()}
        c = gain_of(src, wk, w.shape[1])
        c2 = c.pow(2) if c is not None else None
        if qw.dtype == torch.uint8:
            g = q.get(base + ".weight_scale_2").float().item()
            s = sc.float()
            wq = unpack_e2m1(qw) * s.repeat_interleave(16, dim=1) * g
            row["format"] = "nvfp4"
            row["rel_err"], row["cos"], row["worst_row_rel_err"] = stats(w, wq)
            # blocks whose FP8 scale is below the smallest normal FP8 value lose scale precision
            row["subnormal_scale_frac"] = (s < FP8_MIN_NORMAL).float().mean().item()
            row["zero_scale_frac"] = (s == 0).float().mean().item()
            s_max = fp8(w.reshape(-1, 16).abs().amax(dim=1) / 6.0 / g).reshape(s.shape)
            row["scales_are_max"] = (s == s_max).float().mean().item()
            if a.search:
                row["rel_err_scale_search"] = nvfp4_search(w, g)[0]
                row["gain_rel_err_gain_search"] = nvfp4_search(w, g, c2)[1] if c2 is not None else ""
            else:
                row["rel_err_scale_search"] = row["gain_rel_err_gain_search"] = ""
        else:
            wq = qw.float() * sc.float()
            row["format"] = "fp8"
            row["rel_err"], row["cos"], row["worst_row_rel_err"] = stats(w, wq)
            for k in ("subnormal_scale_frac", "zero_scale_frac", "scales_are_max", "rel_err_scale_search",
                      "gain_rel_err_gain_search"):
                row[k] = ""
        row["has_gain"] = c is not None
        row["gain_rel_err"] = gain_rel_err(w, wq, c2) if c2 is not None else ""
        row["top32_err_share"], row["top32_uniform_share"] = top_share(w, wq)
        rows.append(row)
        if i % 25 == 0:
            print(f"{i + 1}/{len(names)} {wk} {row['format']} rel_err {row['rel_err']:.4f}", flush=True)
        del w, qw, sc, wq
        if dev == "cuda":
            torch.cuda.empty_cache()  # freed blocks stay cached otherwise: 88 GB by tensor 100
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
