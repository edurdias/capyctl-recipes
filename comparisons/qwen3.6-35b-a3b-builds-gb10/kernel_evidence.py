#!/usr/bin/env python3
"""Retain startup kernel/dtype evidence, including rotated native logs, without identities."""
import argparse
from pathlib import Path
import re

PATTERN = re.compile(r'Using.*(?:backend|Marlin|FlashInfer|attention)|KV Cache is allocated|KV cache.*dtype|kv.cache.*dtype|apply_fp4_marlin|NVFP4.*(?:kernel|backend)', re.I)
PRIVATE = re.compile(r'https?://|/opt/|Bearer|api.key|ServerArgs|Namespace|\b(?:\d{1,3}\.){3}\d{1,3}\b', re.I)
FIELDS = re.compile(r"\b(moe_runner_backend|attention_backend|speculative_draft_moe_runner_backend|mamba_ssm_dtype)(?:=|['\"]?\s*:)\s*['\"]?([A-Za-z0-9_]+)")

def extract(state):
    evidence = []
    paths = set((state/'logs').rglob('*.log*')) | set(state.glob('*.log*'))
    for path in sorted(paths):
        for line in path.read_text(errors='replace').splitlines():
            # Extract only closed-word values from the full native argument
            # record; never retain that record's listeners or credentials.
            for name, value in FIELDS.findall(line):
                if value != 'None':
                    evidence.append(f'Configured native server field: {name}={value}')
            if PATTERN.search(line) and not PRIVATE.search(line):
                evidence.append(line)
    return '\n'.join(dict.fromkeys(evidence))+'\n'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(extract(a.state))
    print(a.out.read_text())
