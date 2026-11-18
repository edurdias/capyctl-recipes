#!/usr/bin/env python3
"""Print only measured rows; unknown launch-log fields remain explicitly unknown."""
import json
from pathlib import Path
import sys

print('| Config | Decode c1 | Aggregate c1/c2/c4/c8 tok/s | 24k TTFT c1/c4 s | Ready s | Peak delta GiB | Greedy/tools | Acceptance |')
print('|---|---:|---|---|---:|---:|---|---|')
for directory in map(Path, sys.argv[1:]):
    run = json.loads((directory/'run.json').read_text())
    if run.get('mode') == 'accuracy-only':
        continue
    if run['status'] != 'measured':
        print(f"| {run['key']} | not measured: {run.get('error', 'unknown failure')} | — | — | — | — | — | — |")
        continue
    result = json.loads((directory/'measure.json').read_text())
    levels = {x['streams']: x for x in result.get('concurrency', [])}
    ttft = {x['streams']: x['ttft_s'] for x in result.get('ttft24k', {}).get('levels', [])}
    guard = result['guard']
    decode = f"{levels[1]['decode_tps']:.2f}" if 1 in levels else '—'
    speed = '/'.join(f"{levels[c]['aggregate_tps']:.2f}" if c in levels else '—' for c in (1,2,4,8))
    latency = '/'.join(f'{ttft[c]:.3f}' if c in ttft else '—' for c in (1,4))
    greedy = {True: 'match', False: 'MISMATCH', None: 'reference'}[guard['matches_baseline']]
    tools = 'pass' if guard['tools_passed'] else 'FAIL'
    print(f"| {run['key']} | {decode} | {speed} | {latency} | {run['ready_s']:.2f} | {run['peak_delta_bytes']/2**30:.2f} | {greedy}/{tools} | UNKNOWN; inspect engine-evidence.txt |")
