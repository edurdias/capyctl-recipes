#!/usr/bin/env python3
"""Speed table from a run-host.sh results directory (capyctl-bench results files).

usage: speed.py <results dir> [<results dir> ...]   (prints Markdown)

Per variant: median aggregate tokens/s at each concurrency level (completion tokens of a
round over its wall time), median per-stream decode at one stream, and the context sweep's
median decode and time to first token per prompt size.
"""

import json
import statistics as st
import sys
from pathlib import Path

rows = []
for res in map(Path, sys.argv[1:]):
    for f in sorted(res.glob("bench-*.json")):
        d = json.loads(f.read_text())
        key = f.stem[len("bench-"):]
        conc, one = {}, None
        for lv in d["concurrency"]["levels"]:
            aggs, per = [], []
            for r in lv["rounds"]:
                reqs = [q for q in r["requests"] if not q.get("error")]
                if not reqs:
                    continue
                span = max(q["sent_at"] + q["total_s"] for q in reqs) - min(q["sent_at"] for q in reqs)
                aggs.append(sum(q["completion_tokens"] for q in reqs) / span)
                per += [q["decode_tps"] for q in reqs if q.get("decode_tps")]
            if aggs:
                conc[lv["concurrency"]] = st.median(aggs)
            if lv["concurrency"] == 1 and per:
                one = st.median(per)
        ctx = {}
        for p in (d.get("context_sweep") or {}).get("points", []):
            ok = [x for x in p.get("requests", []) if not x.get("error")]
            if ok:
                ctx[p["label"]] = (st.median(x["decode_tps"] for x in ok), st.median(x["ttft_s"] for x in ok))
        rows.append((key, one, conc, ctx))

levels = sorted({c for _, _, conc, _ in rows for c in conc})
labels = sorted({c for *_, ctx in rows for c in ctx}, key=str)
head = ["Variant", "Decode, 1 stream"] + [f"Aggregate, {c} streams" for c in levels] + \
       [f"Decode / TTFT after {c}" for c in labels]
print("| " + " | ".join(head) + " |")
print("|" + "---|" * len(head))
for key, one, conc, ctx in rows:
    cells = [key, f"{one:.1f}" if one else "-"] + [f"{conc[c]:.1f}" if c in conc else "-" for c in levels] + \
            [f"{ctx[c][0]:.1f} / {ctx[c][1]:.2f} s" if c in ctx else "-" for c in labels]
    print("| " + " | ".join(cells) + " |")
