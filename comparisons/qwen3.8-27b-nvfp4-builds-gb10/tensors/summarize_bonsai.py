#!/usr/bin/env python3
"""Summarise read-bonsai.csv beside the NVFP4 builds' read-*.csv: relative error pooled over tensors by
numel (each tensor's squared relative error weighted by its weight count), per family and overall."""
import csv
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "out"


def load(name):
    return list(csv.DictReader(open(f"{d}/read-{name}.csv")))


def pooled(rows, col, key=lambda r: "all"):
    num, den = defaultdict(float), defaultdict(float)
    for r in rows:
        if r.get(col, "") == "":
            continue
        n = float(r["numel"])
        for k in {key(r), "all"}:
            num[k] += n * float(r[col]) ** 2
            den[k] += n
    return {k: (num[k] / den[k]) ** 0.5 for k in den}


b = load("bonsai")
fam = lambda r: r["kind"].split(".")[0]
cols = ["rel_err", "rel_err_refit_scale", "rel_err_opt_rotated", "rel_err_opt_plain", "gain_rel_err",
        "gain_rel_err_opt_rotated"]
res = {c: pooled(b, c, fam) for c in cols}
fams = sorted(res["rel_err"])
print("Bonsai, pooled relative error by family")
print("family".ljust(10), " ".join(c.replace("rel_err", "e").ljust(14) for c in cols))
for f in fams:
    print(f.ljust(10), " ".join(f"{res[c].get(f, float('nan')) * 100:13.2f}%" for c in cols))
mean = lambda col: sum(float(r[col]) * float(r["numel"]) for r in b) / sum(float(r["numel"]) for r in b)
print(f"trit agreement with the optimal rounding {mean('trit_agree_opt') * 100:.1f}%; zeros: Bonsai "
      f"{mean('zero_frac') * 100:.1f}%, optimal {mean('zero_frac_opt') * 100:.1f}%; median scale ratio "
      f"{sorted(float(r['scale_ratio_median']) for r in b)[len(b) // 2]:.3f}; code 3: "
      f"{sum(int(r['code3']) for r in b)}; scale pairs equal {min(float(r['scale_pairs_equal']) for r in b):.3f}")
ts = [r for r in b if r["top32_err_share"] != ""]
ratio = lambda col: sum(float(r[col]) / float(r["top32_uniform_share"]) for r in ts) / len(ts)
print(f"error share on the top-32 input directions vs uniform: Bonsai {ratio('top32_err_share'):.2f}x, "
      f"optimal rounding {ratio('top32_err_share_opt'):.2f}x ({len(ts)} tensors)")
print()
print("Against the NVFP4 builds (rel_err pooled, all quantized tensors each build has):")
for name in ["capyctl", "nvidia", "radixark", "gittensor"]:
    try:
        r = load(name)
    except FileNotFoundError:
        continue
    print(f"  {name:10} {pooled(r, 'rel_err')['all'] * 100:.2f}%")
print(f"  {'bonsai':10} {res['rel_err']['all'] * 100:.2f}%  (optimal ternary in its basis "
      f"{res['rel_err_opt_rotated']['all'] * 100:.2f}%, without rotation {res['rel_err_opt_plain']['all'] * 100:.2f}%)")
