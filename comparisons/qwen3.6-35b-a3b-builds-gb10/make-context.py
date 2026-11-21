#!/usr/bin/env python3
"""Prepare fresh exact-token prompts using the pinned Qwen chat template."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    cb = load('capyctl_bench', ROOT/'tools/capyctl-bench/capyctl_bench.py')
    depth = load('canonical_long_depth', HERE.parent/'gemma-4-26b-a4b-nvfp4-builds-gb10/long-depth.py')
    result = {'tokenizer': 'nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5',
              'thinking': 'chat-template default', 'levels': {}, 'long': {}}
    for target, levels in [(24000, (1, 4)), (8000, (1,)), (32000, (1,)), (64000, (1,)), (110000, (1,))]:
        for c in levels:
            rounds = []
            for r in range(a.rounds):
                rows = []
                for i in range(c):
                    seed = uuid.uuid4().int
                    messages, count = depth.prepare(tokenizer, cb, target, seed)
                    rows.append({'id': f'{target}-{seed}', 'messages': messages, 'token_count': count,
                                 'seed': str(seed), 'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()})
                rounds.append(rows)
            if target == 24000:
                result['levels'][str(c)] = rounds
            else:
                result['long'][str(target)] = rounds
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False)+'\n')
    manifest = {k: v for k, v in result.items() if k not in ('levels', 'long')}
    for section in ('levels', 'long'):
        manifest[section] = {key: [[{k:v for k,v in row.items() if k != 'messages'} for row in rows]
                                   for rows in rounds] for key, rounds in result[section].items()}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'prepared_targets': [24000, 8000, 32000, 64000, 110000], 'rounds': a.rounds}), flush=True)
